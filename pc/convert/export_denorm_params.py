"""生成"模型输出 → 舵机原始计数"的仿射参数（denorm.json）。

问题
----
ACT 的输出是**归一化动作**，要经过两步才能写进舵机：

    模型输出(归一化)  →×std + mean →  LeRobot 动作空间  →标定变换→  舵机原始计数(0..4095)

两步都是**逐关节仿射变换**，可以合并成：

    raw = scale[j] * out[j] + offset[j]

**但不要手写这两步的公式。** LeRobot 的内部实现（归一化模式、homing_offset 的应用方式、
range 裁剪、drive_mode 方向）随版本变化，抄错一处动作幅度就整体偏，而且**不会报错**。

做法：让 LeRobot 自己算，我们只拟合
------------------------------------
1. 采样若干组随机的归一化动作（覆盖真实取值范围）
2. 喂给 **LeRobot 自己的** postprocessor + 机器人标定转换，得到对应的原始计数
3. 用最小二乘拟合出 `scale` / `offset`
4. **断言残差 ≈ 0** —— 残差不为 0 说明该关节不是仿射的，或者两端配置不一致

这样得到的参数与 LeRobot 版本无关，并且自带验证。

⚠️ 本脚本需要在**采集/训练用的那个 LeRobot 环境**里跑（不是 .venv-rknn）。
⚠️ 下面标了 `# ⚠️ 适配点` 的地方需要按你的 LeRobot 版本核对 API 名称。

用法
----
    python pc/convert/export_denorm_params.py \
        --policy outputs/act_pen_place/checkpoints/last/pretrained_model \
        --dataset 你的HF用户名/so101-pen-place \
        --robot-id my_follower \
        --out configs/denorm.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("denorm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _load_dataset_stats(repo_id: str, roots: list[str] | None):
    """取数据集里 action 的 mean/std（MEAN_STD 归一化用）。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # ⚠️ 适配点

    ds = LeRobotDataset(repo_id, root=roots[0] if roots else None)
    stats = ds.meta.stats["action"]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    log.info("数据集 action 统计: mean=%s", np.round(mean, 2))
    log.info("                    std =%s", np.round(std, 2))
    return mean, std


def _make_unnormalizer(policy_path: str, dataset_repo_id: str, roots: list[str] | None):
    """返回一个函数: 归一化动作 (N,D) -> LeRobot 动作空间 (N,D)。"""
    # ⚠️ 适配点：不同 LeRobot 版本函数名不同。常见路径：
    #   from lerobot.policies.factory import make_pre_post_processors
    #   pre, post = make_pre_post_processors(policy_cfg, dataset_stats=ds.meta.stats)
    #   action = post(action_tensor)
    # 若你的版本没有现成 postprocessor，直接用上面取到的 mean/std：
    #   action = norm * std + mean     （NormalizationMode.MEAN_STD）
    try:
        from lerobot.policies.factory import make_pre_post_processors  # noqa: F401
    except Exception as e:  # noqa: BLE001
        log.error(
            "找不到 make_pre_post_processors（%s）。\n"
            "请按你的 LeRobot 版本改用等价的 postprocessor，"
            "或直接用 dataset 的 mean/std 做 `norm * std + mean`。\n"
            "改完后重跑；不要手写标定部分的公式。",
            e,
        )
        raise

    raise NotImplementedError(
        "请在这里接上你的 LeRobot 版本的 postprocessor。\n"
        "见函数内注释。这是唯一需要手工适配的地方。"
    )


def _make_raw_converter(robot_id: str, port: str | None):
    """返回一个函数: LeRobot 动作空间 (N,D) -> 舵机原始计数 (N,D)。"""
    # ⚠️ 适配点：这里要用 LeRobot 的 SO follower 标定把动作转成原始计数。
    # 思路（按你的版本挑一个）：
    #   a) 实例化 SOFollower，调用其 bus 的 _unnormalize / convert 方法
    #   b) 直接读标定 JSON，用 leRobot 相同的公式换算
    # 关键：**必须复用 LeRobot 的函数**，不要自己推公式。
    raise NotImplementedError(
        "请在这里接上 LeRobot 的标定转换（LeRobotRobot action -> raw counts）。\n"
        "标定文件通常在 ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json"
    )


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐关节最小二乘拟合 y = scale*x + offset，返回 (scale, offset, 残差)。"""
    d = x.shape[1]
    scale = np.zeros(d)
    offset = np.zeros(d)
    resid = np.zeros(d)
    for j in range(d):
        A = np.stack([x[:, j], np.ones_like(x[:, j])], axis=1)
        sol, *_ = np.linalg.lstsq(A, y[:, j], rcond=None)
        scale[j], offset[j] = sol
        pred = A @ sol
        resid[j] = float(np.abs(pred - y[:, j]).max())
    return scale, offset, resid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", required=True, help="LeRobot dataset repo_id")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--robot-port", default=None)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/denorm.json")
    ap.add_argument("--resid-tol", type=float, default=1e-3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    mean, std = _load_dataset_stats(args.dataset, [args.dataset_root] if args.dataset_root else None)
    unnormalize = _make_unnormalizer(args.policy, args.dataset, [args.dataset_root] if args.dataset_root else None)
    to_raw = _make_raw_converter(args.robot_id, args.robot_port)

    # ---------- 采样 ----------
    # 归一化动作近似 N(0,1)，覆盖 ±4σ 足够；仿射映射下外推是精确的
    rng = np.random.default_rng(args.seed)
    norm = np.clip(rng.normal(0.0, 1.0, size=(args.samples, len(mean))), -4.0, 4.0)

    act_space = unnormalize(norm)
    raw = to_raw(act_space)

    if not (np.isfinite(act_space).all() and np.isfinite(raw).all()):
        log.error("采样过程中出现 NaN/Inf，检查 postprocessor 与标定")
        return 2

    # ---------- 拟合与断言 ----------
    scale, offset, resid = fit_affine(norm, raw)
    log.info("拟合结果:")
    for j, name in enumerate(JOINTS[: len(scale)]):
        log.info("  %-14s scale=%10.4f  offset=%10.2f  残差=%.3e", name, scale[j], offset[j], resid[j])

    worst = float(resid.max())
    if worst > args.resid_tol:
        log.error("=" * 70)
        log.error("残差过大 (%.3e > %.3e) —— 该变换不是逐关节仿射，或两端配置不一致。", worst, args.resid_tol)
        log.error("可能原因：")
        log.error("  1) 标定 JSON 与数据集不是同一条机械臂 / 不是同一次校准")
        log.error("  2) 归一化模式不是 MEAN_STD（检查 policy config 的 normalization_mapping）")
        log.error("  3) 存在 range 裁剪（MIN_MAX 模式会截断，导致非线性）")
        log.error("**不生成 denorm.json** —— 参数错了会让机械臂动作幅度整体偏，且不会报错。")
        log.error("=" * 70)
        return 3

    # ---------- 写文件 ----------
    out = {
        "format": "rkrobot.denorm.v1",
        "created_from": {"policy": args.policy, "dataset": args.dataset, "robot_id": args.robot_id},
        "joints": JOINTS[: len(scale)],
        "scale": [round(float(v), 6) for v in scale],
        "offset": [round(float(v), 4) for v in offset],
        "fit_residual_max": worst,
        "samples": args.samples,
        "note": "由最小二乘拟合 + 残差断言生成，禁止手改",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("已写出 %s", p)
    log.info("下一步：python pc/convert/convert_to_rknn.py --onnx <你的 onnx> --out models/act.rknn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
