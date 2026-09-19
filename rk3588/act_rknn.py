"""ACT 的 RKNN 推理封装。

防住两类**静默失败**：

1. **RKNN 会重排输入顺序。** 实测：ONNX 声明 `[cam_high, cam_left, state]`，
   转换后的 RKNN 可能期望 `[state, cam_high, cam_left]`。若两路图像形状相同，
   顺序互换不会报错，只会让动作全错。
   → **没有 manifest 就拒绝运行**（manifest 由 convert_to_rknn.py 暴力枚举确认后生成）

2. **输出可能静默全零 / 非有限。**
   → 每次推理都检查 shape / finite / 零值比例

另外两个来自真实踩坑的要点：

* `rknn.inference()` 的 `data_format` **默认是 nhwc**，而我们的 ONNX 是 NCHW，
  必须显式传入。data_format 从 manifest 的 `image_layout` 读取（对齐 IB_Robot 的
  RKNNSession 做法）。
* 归一化必须是 MEAN_STD（图像也要 z-score），见 stats.py。

未实测：本文件尚未在真实 RK3588 上运行过。
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from stats import NormStats

log = logging.getLogger(__name__)


@dataclass
class ActConfig:
    rknn_model: str
    manifest: str
    norm_stats_dir: str
    action_dim: int = 6
    state_dim: int = 6
    chunk_size: int = 100
    # pad 模式下的补零锚点：top_left = 内容靠右下（与 IB_Robot 实测一致）
    image_pad_anchor: str = "top_left"
    zero_ratio_error: float = 0.98
    latency_window: int = 100


@dataclass
class _Latency:
    window: int = 100
    _s: deque = field(default_factory=deque, init=False)

    def add(self, ms: float) -> None:
        self._s.append(ms)
        while len(self._s) > self.window:
            self._s.popleft()

    def summary(self) -> dict:
        if not self._s:
            return {}
        a = np.asarray(self._s, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean_ms": round(float(a.mean()), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "max_ms": round(float(a.max()), 2),
        }


class ActRKNN:
    def __init__(self, cfg: ActConfig):
        self.cfg = cfg
        self.latency = _Latency(cfg.latency_window)

        self.manifest = self._load_manifest(cfg.manifest)
        self.stats = NormStats.load(cfg.norm_stats_dir)

        self.input_order: list[str] = list(self.manifest["input_order"])
        self.image_slots: list[str] = list(self.manifest.get("image_slots", []))
        self.image_size: tuple[int, int] = tuple(self.manifest["image_size"])  # (H, W)
        self.data_format: str = str(self.manifest.get("image_layout", "nchw")).lower()
        if self.data_format not in ("nchw", "nhwc"):
            raise ValueError(f"manifest.image_layout 非法: {self.data_format}")
        if self.data_format != "nchw":
            raise ValueError(
                f"manifest 声明 image_layout={self.data_format}，但本类的预处理产出 NCHW。"
                "要么改 manifest，要么改 _prep_image —— 不要在这里悄悄转置。"
            )

        # 相机槽位名（front / wrist ...）用于查归一化统计量
        for slot in self.image_slots:
            if slot not in self.stats.cameras:
                raise ValueError(
                    f"manifest 的图像槽位 {slot!r} 在归一化统计量里不存在；"
                    f"可用: {list(self.stats.cameras)}"
                )

        from rknnlite.api import RKNNLite  # 延迟导入，便于 PC 上做单元测试

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(cfg.rknn_model) != 0:
            raise RuntimeError(f"load_rknn 失败: {cfg.rknn_model}")
        if self.rknn.init_runtime(target=None) != 0:
            raise RuntimeError("init_runtime 失败（检查 librknnrt 版本是否与 toolkit 匹配）")

        log.info("ACT RKNN 就绪 | 输入顺序=%s | 图像 %s | data_format=%s",
                 self.input_order, self.image_size, self.data_format)

    # ---------- 加载与校验 ----------

    @staticmethod
    def _load_manifest(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"缺少 {path}。\n"
                "输入顺序未经确认时禁止运行 —— 顺序错了不会报错，只会让机械臂乱动。\n"
                "请先执行 pc/convert/convert_to_rknn.py 生成 manifest。"
            )
        m = json.loads(p.read_text(encoding="utf-8"))
        for key in ("input_order", "image_size"):
            if key not in m:
                raise ValueError(f"manifest 缺少字段 {key}")
        return m

    # ---------- 预处理 ----------

    def _prep_image(self, frame: np.ndarray, slot: str) -> np.ndarray:
        """frame 必须是 **RGB** HWC uint8。

        注意：LeRobot 的 OpenCVCamera 默认 `color_mode=RGB`，
        所以从 `robot.get_observation()` 拿到的就已经是 RGB，
        **不要再做 BGR->RGB 转换**（会把红蓝通道反转，静默失效）。
        如果换用其它取流方式（cv2.VideoCapture 给的是 BGR），
        请在调用前自己转成 RGB。
        """
        h, w = self.image_size
        if frame.shape[0] != h or frame.shape[1] != w:
            scale = min(h / frame.shape[0], w / frame.shape[1])
            nh, nw = int(round(frame.shape[0] * scale)), int(round(frame.shape[1] * scale))
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if (nh, nw) == (h, w):
                frame = resized
            else:
                canvas = np.zeros((h, w, 3), dtype=frame.dtype)
                if self.cfg.image_pad_anchor == "bottom_right":
                    canvas[:nh, :nw] = resized
                else:  # top_left（默认，与 IB_Robot 实测一致：内容靠右下）
                    canvas[h - nh :, w - nw :] = resized
                frame = canvas

        x = frame.astype(np.float32) / 255.0
        chw = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])  # (1,3,H,W)
        # ⚠️ 关键：ACT 的 VISUAL 归一化是 MEAN_STD，必须用数据集的 per-camera 统计量
        return np.ascontiguousarray(self.stats.normalize_image(chw, slot))

    # ---------- 推理 ----------

    def infer(self, frames: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """frames: {槽位名: BGR 帧}；state: 原始关节状态。

        返回 (chunk_size, action_dim)，已反归一化到机器人动作空间。
        """
        if not self.image_slots:
            raise ValueError("manifest 没有声明 image_slots")

        slots: dict[str, np.ndarray] = {
            "observation.state": self.stats.normalize_state(state).reshape(1, -1).astype(np.float32),
            "state": self.stats.normalize_state(state).reshape(1, -1).astype(np.float32),
        }
        for slot in self.image_slots:
            if slot not in frames:
                raise ValueError(f"缺少相机帧 {slot!r}；已提供 {sorted(frames)}")
            slots[slot] = self._prep_image(frames[slot], slot)

        missing = [n for n in self.input_order if n not in slots]
        if missing:
            raise ValueError(
                f"manifest.input_order 需要的输入 {missing} 未提供；已提供 {sorted(slots)}"
            )
        inputs = [slots[n] for n in self.input_order]

        t0 = time.perf_counter()
        # ⚠️ 必须显式 data_format（来自 manifest）。rknn.inference 默认 nhwc，
        #    会把 (1,3,H,W) 判成形状错误：
        #      The input(ndarray) shape (1,3,480,640) is wrong, expect 'nhwc' like (1,480,640,3)
        outputs = self.rknn.inference(inputs=inputs, data_format=self.data_format)
        self.latency.add((time.perf_counter() - t0) * 1000.0)

        if not outputs:
            raise RuntimeError("RKNN 未返回任何输出")
        out = np.asarray(outputs[0], dtype=np.float32)
        self._validate_output(out)
        return self.stats.denormalize_action(out.reshape(-1, self.cfg.action_dim))

    def _validate_output(self, out: np.ndarray) -> None:
        if out.ndim != 3 or out.shape[0] != 1:
            raise RuntimeError(f"输出形状异常: {out.shape}，期望 (1, chunk, dim)")
        if out.shape[2] != self.cfg.action_dim:
            raise RuntimeError(f"动作维度异常: {out.shape[2]}，期望 {self.cfg.action_dim}")
        if not np.isfinite(out).all():
            raise RuntimeError(f"输出含 {int((~np.isfinite(out)).sum())} 个 NaN/Inf —— 立即停机")
        zero_ratio = float((np.abs(out) < 1e-9).mean())
        if zero_ratio > self.cfg.zero_ratio_error:
            raise RuntimeError(
                f"输出 {zero_ratio:.1%} 为零值（阈值 {self.cfg.zero_ratio_error:.0%}）—— "
                "疑似 runtime fallback 失败或输入顺序错误，立即停机"
            )

    def warmup(self, n: int = 3) -> None:
        dummy_img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        dummy_state = self.stats.state_mean.copy()
        frames = {s: dummy_img for s in self.image_slots}
        for _ in range(n):
            self.infer(frames, dummy_state)
        self.latency._s.clear()
        log.info("warmup 完成（已清空延迟统计）")

    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:  # noqa: BLE001
            pass
