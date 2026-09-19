"""从训练数据集导出归一化统计量（.npy），供板端 numpy 直接使用。

为什么需要这个
--------------
LeRobot 的 ACT 默认 `normalization_mapping` 是：

    VISUAL = MEAN_STD      <-- 图像也要做 z-score，不是只 /255！
    STATE  = MEAN_STD
    ACTION = MEAN_STD

所以板端必须用**训练数据集的 per-camera mean/std** 对图像做归一化，
再用 action 的 mean/std 把输出反归一化。只做 `/255` 是错的 —— 模型会收到
分布完全不同的输入，动作会整体偏。

做法（对齐 D-Robotics/rdk_LeRobot_tools）
-----------------------------------------
它把统计量存成一组 `.npy` 放在模型目录旁边，板端用 numpy 加载：

    front_mean.npy / front_std.npy     # 每路相机，形状 (3,1,1) 便于对 NCHW 广播
    state_mean.npy / state_std.npy
    action_mean.npy / action_std.npy

本脚本输出同样的布局，这样板端不需要 torch，也不需要 LeRobot。

用法（在 lerobot 环境里跑）
    python pc/convert/export_norm_stats.py \
        --dataset 你的HF用户名/so101-pen-place \
        --out models/norm_stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EPS = 1e-8


def log(msg: str) -> None:
    print(f"[export_norm_stats] {msg}")


def load_stats(repo_id: str, root: str | None) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    log(f"打开数据集 {repo_id}" + (f" (root={root})" if root else ""))
    ds = LeRobotDataset(repo_id, root=root)
    stats = ds.meta.stats
    log(f"数据集里有统计量的特征: {sorted(stats)}")
    return stats


def to_image_shape(arr: np.ndarray) -> np.ndarray:
    """把 (3,) 的 mean/std 变成 (3,1,1)，便于对 (1,3,H,W) 广播。"""
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    return a.reshape(3, 1, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ACT 归一化统计量")
    ap.add_argument("--dataset", required=True, help="LeRobot dataset repo_id")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out", required=True, help="输出目录（.npy 与 manifest.json）")
    ap.add_argument("--camera", action="append", default=None,
                    help="相机名（可重复）。默认自动从数据集里发现")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        stats = load_stats(args.dataset, args.dataset_root)
    except Exception as e:  # noqa: BLE001
        log(f"❌ 读取数据集失败: {type(e).__name__}: {e}")
        return 2

    written: dict[str, dict] = {}

    # ---------- 每路相机 ----------
    if args.camera:
        cameras = list(args.camera)
    else:
        cameras = sorted(
            k[len(IMAGE_PREFIX):] for k in stats if k.startswith(IMAGE_PREFIX)
        )
    if not cameras:
        log("❌ 数据集里没有 observation.images.* 统计量")
        return 3

    for cam in cameras:
        key = f"{IMAGE_PREFIX}{cam}"
        if key not in stats:
            log(f"❌ 数据集里缺少 {key}")
            return 3
        s = stats[key]
        if "mean" not in s or "std" not in s:
            log(f"❌ {key} 缺少 mean/std（LeRobot 版本差异？）")
            return 3
        mean = to_image_shape(s["mean"])
        std = to_image_shape(s["std"]) + EPS
        np.save(out / f"{cam}_mean.npy", mean)
        np.save(out / f"{cam}_std.npy", std)
        written[key] = {"mean_file": f"{cam}_mean.npy", "std_file": f"{cam}_std.npy",
                        "shape": list(mean.shape)}
        log(f"  {key}")
        log(f"    mean={np.round(mean.reshape(-1), 4)}  std={np.round(std.reshape(-1), 4)}")

    # ---------- state / action ----------
    for key, name, shape in (
        (STATE_KEY, "state", None),
        (ACTION_KEY, "action", None),
    ):
        if key not in stats:
            log(f"❌ 数据集里缺少 {key}")
            return 3
        s = stats[key]
        mean = np.asarray(s["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(s["std"], dtype=np.float32).reshape(-1) + EPS
        np.save(out / f"{name}_mean.npy", mean)
        np.save(out / f"{name}_std.npy", std)
        written[key] = {"mean_file": f"{name}_mean.npy", "std_file": f"{name}_std.npy",
                        "shape": list(mean.shape)}
        log(f"  {key}")
        log(f"    mean={np.round(mean, 3)}")
        log(f"    std ={np.round(std, 3)}")

    manifest = {
        "format": "rkrobot.norm_stats.v1",
        "source_dataset": args.dataset,
        "normalization": "MEAN_STD",
        "cameras": cameras,
        "stats": written,
        # 板端计算方式：
        #   图像: x_norm = (x/255 - camera_mean) / camera_std     （NCHW，mean/std 形状 (3,1,1)）
        #   状态: s_norm = (s - state_mean) / state_std
        #   动作: a      = a_norm * action_std + action_mean
        "formulas": {
            "image": "x_norm = (x/255 - camera_mean) / camera_std",
            "state": "state_norm = (state - state_mean) / state_std",
            "action": "action = action_norm * action_std + action_mean",
        },
        "note": "由 export_norm_stats.py 生成，禁止手改；必须与训练数据集一致",
    }
    (out / "norm_stats.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"✅ 已写出 {out}")
    log("   把整个目录一起拷到板端，并在 configs/robot.yaml 里指向它")
    return 0


if __name__ == "__main__":
    sys.exit(main())
