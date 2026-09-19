"""闭环主程序：相机 → ACT(RKNN) → 动作块队列 → 舵机。

三种运行模式，**必须按顺序放行**：

    --once      只跑一次推理，打印延迟/形状/数值统计后退出（不碰舵机）
    --dry-run   持续闭环，但**不写舵机**（验证延迟、队列、饥饿率）
    （默认）     真机闭环

线程模型
--------
    control 线程 @ control_freq_hz
        └─ ActionQueue.pop() → 限位钳制 → 写舵机
           队列剩余 < refill_threshold 时置事件，通知推理线程补块

    inference 线程
        └─ 取最新帧 + 当前关节状态 → ActRKNN.infer() → 反归一化 → push 动作块

推理异步是关键：T_infer ≈ 121 ms 远小于一个动作块的时长（100 步 / 30 fps ≈ 3.3 s），
所以可以提前算好下一块，控制线程永远不会被推理阻塞。

未实测：本文件尚未在真实 RK3588 + SO-ARM101 上运行过。
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from action_queue import ActionQueue
from act_rknn import ActConfig, ActRKNN

log = logging.getLogger("rkrobot")


class _Stopper:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def stop(self, *_a) -> None:
        if not self._ev.is_set():
            log.warning("收到停止信号，正在停机……")
            self._ev.set()

    @property
    def stopped(self) -> bool:
        return self._ev.is_set()


def load_configs(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_act(cfg: dict) -> ActRKNN:
    pol = cfg["policy"]
    cam = cfg["camera"]
    return ActRKNN(
        ActConfig(
            rknn_model=pol["rknn_model"],
            manifest=pol["manifest"],
            denorm_json=pol["denorm_json"],
            action_dim=int(pol["action_dim"]),
            state_dim=int(pol["state_dim"]),
            chunk_size=int(pol["chunk_size"]),
            image_norm=pol.get("image_norm", "zero_one"),
            image_resize=pol.get("image_resize", "squash"),
            image_pad_anchor=pol.get("image_pad_anchor", "top_left"),
            zero_ratio_error=float(cfg["runtime"].get("zero_ratio_error", 0.98)),
        )
    )


def clamp_raw(targets: np.ndarray, cfg: dict) -> np.ndarray:
    """按 configs/robot.yaml 的 raw_limits 做最后一道限位。"""
    joints = cfg["robot"]["joints"]
    limits = cfg["robot"]["raw_limits"]
    out = np.asarray(targets, dtype=np.float32).copy()
    for i, j in enumerate(joints):
        lo, hi = limits[j]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


class Camera:
    """最简单的取流封装：只保留最新一帧。"""

    def __init__(self, cfg: dict):
        import cv2

        cam = cfg["camera"]
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(cam["index_or_path"])
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开相机: {cam['index_or_path']}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cam["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._t.start()

    def _loop(self) -> None:
        misses = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                misses += 1
                if misses % 50 == 0:
                    log.warning("连续 %d 次取帧失败", misses)
                time.sleep(0.01)
                continue
            misses = 0
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)
        self.cap.release()


def run(args: argparse.Namespace) -> int:
    cfg = load_configs(args.config)
    rt = cfg["runtime"]
    stop = _Stopper()
    signal.signal(signal.SIGINT, stop.stop)
    signal.signal(signal.SIGTERM, stop.stop)

    log_file = Path(cfg["logging"]["jsonl"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl = log_file.open("a", encoding="utf-8")

    act = build_act(cfg)
    joints = cfg["robot"]["joints"]
    ids = [cfg["robot"]["motor_ids"][j] for j in joints]

    image_slots = list(act.manifest.get("image_slots", ["front"]))
    if len(image_slots) > 1:
        log.warning(
            "模型需要 %d 个图像槽位 %s，但当前只有 1 路相机 —— 所有槽位将收到同一帧。"
            "多相机需要扩展 Camera 以支持多路取流；单相机训练时就应只用一个槽位。",
            len(image_slots), image_slots,
        )

    def to_frames(frame: np.ndarray) -> dict[str, np.ndarray]:
        return {s: frame for s in image_slots}

    # ---------- 模式 1：单次推理 ----------
    if args.once:
        cam = Camera(cfg)
        cam.start()
        for _ in range(50):
            if cam.latest() is not None:
                break
            time.sleep(0.05)
        frame = cam.latest()
        cam.close()
        if frame is None:
            log.error("取不到图像")
            return 2
        act.warmup(3)
        state = np.full(len(joints), 2047.0, dtype=np.float32)
        raw = act.infer(to_frames(frame), state)
        log.info("输出 shape=%s  延迟=%s", raw.shape, act.latency.summary())
        log.info("前 3 步原始计数:\n%s", np.round(raw[:3], 1))
        log.info("每关节范围: %s", {j: (round(float(raw[:, i].min()), 1),
                                       round(float(raw[:, i].max()), 1))
                                    for i, j in enumerate(joints)})
        act.close()
        jsonl.close()
        return 0

    # ---------- 模式 2/3：闭环 ----------
    bus = None
    if not args.dry_run:
        from feetech_bus import FeetechBus, load_bus_config

        bus_cfg = load_bus_config(args.config, args.feetech_config)
        bus = FeetechBus(bus_cfg)
        bus.open()
        bus.enable_torque(ids, False)  # 先松扭矩，确认能通信
        pos = bus.read_present_positions(ids)
        log.info("初始位置(原始计数): %s", pos)

    cam = Camera(cfg)
    cam.start()

    q = ActionQueue(
        dataset_fps=float(cfg["policy"]["dataset_fps"]),
        control_freq_hz=float(rt["control_freq_hz"]),
        action_dim=int(cfg["policy"]["action_dim"]),
        queue_max_steps=int(rt["queue_max_steps"]),
        max_chunk_age_s=float(rt["max_chunk_age_s"]),
        interpolate=bool(rt["interpolate"]),
        on_queue_empty=rt["on_queue_empty"],
        blend_steps=int(rt.get("blend_steps", 5)),
    )
    refill = threading.Event()
    refill.set()
    last_error: list[str] = []

    def inference_worker() -> None:
        def state_now() -> np.ndarray:
            if bus is None:
                return np.full(len(joints), 2047.0, dtype=np.float32)
            p = bus.read_present_positions(ids)
            return np.asarray([p[i] for i in ids], dtype=np.float32)

        while not stop.stopped:
            if not refill.wait(timeout=0.2):
                continue
            frame = cam.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                raw = act.infer(to_frames(frame), state_now())
            except Exception as e:  # noqa: BLE001
                log.error("推理失败: %s", e)
                last_error.append(str(e))
                stop.stop()
                return
            q.push(raw, pushed_at=time.monotonic())
            refill.clear()

    inf_t = threading.Thread(target=inference_worker, daemon=True)
    inf_t.start()

    act.warmup(3)
    if bus is not None:
        bus.enable_torque(ids, True)
        log.info("扭矩已使能，开始闭环控制")

    period = 1.0 / float(rt["control_freq_hz"])
    prev: np.ndarray | None = None
    max_delta = float(rt.get("max_delta_per_step", 0) or 0)
    n = 0
    t_start = time.monotonic()
    next_t = t_start

    try:
        while not stop.stopped:
            next_t += period
            target = clamp_raw(q.pop(), cfg)

            if max_delta > 0 and prev is not None:
                delta = np.clip(target - prev, -max_delta, max_delta)
                target = prev + delta
            prev = target

            if bus is not None:
                bus.write_goal_positions({i: int(round(v)) for i, v in zip(ids, target)})

            n += 1
            if q.steps_available < int(rt["refill_threshold"]):
                refill.set()

            if n % int(cfg["camera"]["fps"]) == 0:
                jsonl.write(json.dumps({
                    "t": round(time.monotonic() - t_start, 3),
                    "target": [round(float(v), 1) for v in target],
                    "queue": q.steps_available,
                    **q.stats.as_dict(),
                    **{f"lat_{k}": v for k, v in act.latency.summary().items()},
                }, ensure_ascii=False) + "\n")
                jsonl.flush()

            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # 落后了就重新对齐，避免累积漂移

    finally:
        cam.close()
        if bus is not None:
            try:
                bus.enable_torque(ids, False)
                log.info("扭矩已释放")
            except Exception as e:  # noqa: BLE001
                log.error("释放扭矩失败（请手动断电）: %s", e)
            bus.close()
        act.close()
        jsonl.close()

    log.info("控制步数=%d  用时=%.1fs", n, time.monotonic() - t_start)
    log.info("队列统计: %s", q.stats.as_dict())
    log.info("推理延迟: %s", act.latency.summary())
    if last_error:
        log.error("运行中出现错误: %s", last_error[-1])
        return 3
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="RK3588 + SO-ARM101 ACT 闭环")
    ap.add_argument("--config", default="configs/robot.yaml")
    ap.add_argument("--feetech-config", default="configs/feetech_sts3215.yaml")
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
