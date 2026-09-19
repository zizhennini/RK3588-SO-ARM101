"""板端闭环主程序。

结构照搬 D-Robotics/rdk_LeRobot_tools 的 `bpu_control_robot.py`：
**单线程、队列空了才推理、推理完直接出队执行**。ACT 一次推理约 121 ms，
而一个动作块（100 步 @30fps）覆盖 3.3 秒，inline 推理完全够用 ——
没必要上多线程那套复杂度。

机械臂与相机都交给 **LeRobot 自己的驱动**（`SOFollower` + `OpenCVCamera`），
不再自己实现 Feetech 协议。这些模块不依赖 torch，板端用
`pip install --no-deps lerobot` + 少量运行时依赖即可（见 requirements.txt）。

三种模式，必须按顺序放行：

    --once      只跑一次推理，打印延迟/形状/数值范围，不碰舵机
    --dry-run   持续闭环，但不写舵机（验证延迟、队列、饥饿率）
    （默认）     真机闭环（5 秒倒计时）

未实测：本文件尚未在真实 RK3588 + SO-ARM101 上运行过。
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from act_rknn import ActConfig, ActRKNN
from action_queue import ActionQueue

log = logging.getLogger("rkrobot")


def busy_wait(seconds: float) -> None:
    """等到 seconds 秒之后（尾段自旋，避免 time.sleep 精度不足）。

    自己实现是为了不依赖 LeRobot 内部工具函数的版本位置
    （rdk 的脚本用的是旧版路径 `lerobot.common.robot_devices.control_utils`，
     在 0.4.4 上已经不存在）。
    """
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    if seconds > 0.01:
        time.sleep(seconds - 0.005)
    while time.perf_counter() < deadline:
        pass


class _Stop:
    def __init__(self) -> None:
        self._flag = False

    def __call__(self, *_a) -> None:
        if not self._flag:
            log.warning("收到停止信号，正在停机……")
            self._flag = True

    @property
    def stopped(self) -> bool:
        return self._flag


def build_robot(cfg: dict):
    """用 LeRobot 自己的 SOFollower 驱动搭建机械臂（含相机）。"""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    rc = cfg["robot"]
    cc = cfg["camera"]

    # ⚠️ LeRobot 的相机默认 color_mode=RGB —— 正是 act_rknn._prep_image 期望的输入
    cameras = {
        slot: OpenCVCameraConfig(
            index_or_path=cc["index_or_path"],
            fps=int(cc["fps"]),
            width=int(cc["width"]),
            height=int(cc["height"]),
            fourcc=cc.get("fourcc"),
        )
        for slot in cfg["policy"]["image_slots"]
    }

    robot_cfg = SOFollowerRobotConfig(
        port=rc["port"],
        id=rc.get("id"),
        cameras=cameras,
        # LeRobot 自带的安全限速：单次目标相对当前位置的最大跳变（度）
        max_relative_target=rc.get("max_relative_target"),
        # 与数据集/策略的动作空间保持一致（LeRobot 默认 True）
        use_degrees=bool(rc.get("use_degrees", True)),
    )
    return SOFollower(robot_cfg)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_act(cfg: dict) -> ActRKNN:
    p = cfg["policy"]
    return ActRKNN(
        ActConfig(
            rknn_model=p["rknn_model"],
            manifest=p["manifest"],
            norm_stats_dir=p["norm_stats_dir"],
            action_dim=int(p["action_dim"]),
            state_dim=int(p["state_dim"]),
            chunk_size=int(p["chunk_size"]),
            image_pad_anchor=p.get("image_pad_anchor", "top_left"),
            zero_ratio_error=float(cfg["runtime"].get("zero_ratio_error", 0.98)),
        )
    )


def obs_to_state(obs: dict, joints: list[str]) -> np.ndarray:
    """LeRobot 的 observation 是 {'<motor>.pos': float, '<camera>': ndarray}。"""
    return np.asarray([float(obs[f"{j}.pos"]) for j in joints], dtype=np.float32)


def state_to_action(target: np.ndarray, joints: list[str]) -> dict:
    return {f"{j}.pos": float(v) for j, v in zip(joints, target, strict=True)}


def clamp_action(cfg: dict, target: np.ndarray, limits: np.ndarray | None) -> np.ndarray:
    if limits is None:
        return target
    return np.clip(target, limits[:, 0], limits[:, 1]).astype(np.float32)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rt = cfg["runtime"]
    pol = cfg["policy"]
    joints = list(cfg["robot"]["joints"])
    stop = _Stop()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    log_file = Path(cfg["logging"]["jsonl"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl = log_file.open("a", encoding="utf-8")

    act = build_act(cfg)
    image_slots = act.image_slots
    fps = float(cfg["camera"]["fps"])

    limits = None
    if cfg["robot"].get("angle_limits"):
        limits = np.asarray(
            [cfg["robot"]["angle_limits"][j] for j in joints], dtype=np.float32
        )

    robot = None
    if not args.dry_run and not args.once:
        robot = build_robot(cfg)

    # ---------------- --once：单次推理，不碰舵机 ----------------
    if args.once:
        robot = build_robot(cfg)
        robot.connect()
        try:
            obs = robot.get_observation()
            state = obs_to_state(obs, joints)
            frames = {s: np.asarray(obs[s]) for s in image_slots}
            log.info("观测: state=%s | 相机=%s", np.round(state, 2), {k: v.shape for k, v in frames.items()})
            act.warmup(3)
            action = act.infer(frames, state)
            log.info("输出 shape=%s  延迟=%s", action.shape, act.latency.summary())
            log.info("前 3 步:\n%s", np.round(action[:3], 3))
            log.info("每关节范围: %s", {
                j: (round(float(action[:, i].min()), 2), round(float(action[:, i].max()), 2))
                for i, j in enumerate(joints)
            })
        finally:
            robot.disconnect()
            act.close()
            jsonl.close()
        return 0

    # ---------------- 闭环 ----------------
    q = ActionQueue(
        action_dim=int(pol["action_dim"]),
        dataset_fps=float(pol["dataset_fps"]),
        control_freq_hz=float(rt["control_freq_hz"]),
        max_chunk_age_s=float(rt["max_chunk_age_s"]),
        interpolate=bool(rt["interpolate"]),
        blend_steps=int(rt.get("blend_steps", 0)),
        on_empty=str(rt["on_queue_empty"]),
    )

    if robot is not None:
        robot.connect()
        log.info("机械臂已连接: %s", cfg["robot"]["port"])
    else:
        log.info("dry-run：只推理不写舵机")

    act.warmup(3)
    n = 0
    t_start = time.monotonic()
    last_log = t_start

    try:
        while not stop.stopped:
            t0 = time.perf_counter()

            obs = robot.get_observation() if robot is not None else None
            if obs is not None:
                state = obs_to_state(obs, joints)
                frames = {s: np.asarray(obs[s]) for s in image_slots}
            else:
                state = act.stats.state_mean.copy()
                frames = {s: np.zeros((int(cfg["camera"]["height"]), int(cfg["camera"]["width"]), 3), np.uint8)
                          for s in image_slots}

            # 照 rdk 的做法：队列空了才推理（inline）
            if q.steps_available <= 0:
                q.push(act.infer(frames, state), pushed_at=time.monotonic())

            target = clamp_action(cfg, q.pop(), limits)
            if robot is not None:
                robot.send_action(state_to_action(target, joints))

            n += 1
            now = time.monotonic()
            if now - last_log >= 1.0:
                jsonl.write(json.dumps({
                    "t": round(now - t_start, 2),
                    "target": [round(float(v), 2) for v in target],
                    "queue": q.steps_available,
                    **q.stats.as_dict(),
                    **{f"lat_{k}": v for k, v in act.latency.summary().items()},
                }, ensure_ascii=False) + "\n")
                jsonl.flush()
                last_log = now

            busy_wait(1.0 / fps - (time.perf_counter() - t0))

    finally:
        if robot is not None:
            try:
                robot.disconnect()
                log.info("机械臂已断开（扭矩已按 LeRobot 配置释放）")
            except Exception as e:  # noqa: BLE001
                log.error("断开失败（请手动断电）: %s", e)
        act.close()
        jsonl.close()

    log.info("控制步数=%d  用时=%.1fs", n, time.monotonic() - t_start)
    log.info("队列统计: %s", q.stats.as_dict())
    log.info("推理延迟: %s", act.latency.summary())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="RK3588 + SO-ARM101 ACT 闭环（LeRobot 驱动）")
    ap.add_argument("--config", default="configs/robot.yaml")
    ap.add_argument("--once", action="store_true", help="只跑一次推理，不写舵机")
    ap.add_argument("--dry-run", action="store_true", help="闭环但不写舵机")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    if not args.once and not args.dry_run:
        log.warning("=" * 68)
        log.warning("即将驱动真机。请确认：工作区已清空 / 急停可触达 / 已跑过 --once 与 --dry-run")
        log.warning("=" * 68)
        for i in range(5, 0, -1):
            log.warning("%d ...", i)
            time.sleep(1)

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
