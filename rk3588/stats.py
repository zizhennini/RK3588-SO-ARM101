"""载入并应用归一化统计量（纯 numpy，不需要 torch / LeRobot）。

对应 pc/convert/export_norm_stats.py 的输出目录。

ACT 的归一化是 MEAN_STD，三路都要做：

    图像  x_norm = (x/255 - camera_mean) / camera_std
    状态  s_norm = (s - state_mean) / state_std
    动作  a      = a_norm * action_std + action_mean

以前只对图像做 /255 是错的 —— 那样模型收到的输入分布与训练时不同，
动作会整体偏，而且不会报任何错。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
EPS = 1e-8


@dataclass
class NormStats:
    root: Path
    cameras: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None

    @classmethod
    def load(cls, root: str | Path) -> "NormStats":
        root = Path(root)
        meta_path = root / "norm_stats.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"缺少 {meta_path}。\n"
                "板端必须用训练数据集的归一化统计量；没有它动作会整体偏且不报错。\n"
                "请在 PC 上用 pc/convert/export_norm_stats.py 生成后一起拷过来。"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("normalization") != "MEAN_STD":
            raise ValueError(f"只支持 MEAN_STD，实际 {meta.get('normalization')}")

        self = cls(root=root)

        for cam in meta["cameras"]:
            m = np.load(root / f"{cam}_mean.npy").astype(np.float32)
            s = np.load(root / f"{cam}_std.npy").astype(np.float32) + EPS
            self.cameras[cam] = (m, s)

        self.state_mean = np.load(root / "state_mean.npy").astype(np.float32)
        self.state_std = np.load(root / "state_std.npy").astype(np.float32) + EPS
        self.action_mean = np.load(root / "action_mean.npy").astype(np.float32)
        self.action_std = np.load(root / "action_std.npy").astype(np.float32) + EPS

        log.info("归一化统计量已载入: 相机=%s | state=%d 维 | action=%d 维",
                 list(self.cameras), self.state_mean.size, self.action_mean.size)
        return self

    # ---------- 前向 ----------

    def normalize_image(self, chw: np.ndarray, camera: str) -> np.ndarray:
        """输入 (1,3,H,W) float32，值域 [0,1]；返回归一化后的同形状数组。"""
        if camera not in self.cameras:
            raise KeyError(f"没有相机 {camera!r} 的统计量；可用: {list(self.cameras)}")
        mean, std = self.cameras[camera]
        return (chw - mean) / std

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.state_mean is None or s.size != self.state_mean.size:
            raise ValueError(f"state 维度不符: 传入 {s.size}，统计量 {None if self.state_mean is None else self.state_mean.size}")
        return (s - self.state_mean) / self.state_std

    # ---------- 后向 ----------

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """(N, D) 归一化动作 -> 机器人动作空间。"""
        a = np.asarray(action_norm, dtype=np.float32)
        if self.action_mean is None or a.shape[-1] != self.action_mean.size:
            raise ValueError(
                f"动作维度不符: 传入 {a.shape[-1]}，统计量 {None if self.action_mean is None else self.action_mean.size}"
            )
        return (a * self.action_std[None, :] + self.action_mean[None, :]).astype(np.float32)
