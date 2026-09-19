"""ACT 的 RKNN 推理封装。

本文件存在的唯一理由是防住两类**静默失败**：

1. **RKNN 会重排输入顺序。** 实测：ONNX 输入顺序是 `[cam_high, cam_left, state]`，
   转换后的 RKNN 期望 `[state, cam_high, cam_left]`。如果两个图像输入形状相同，
   顺序互换**不会报错**，只会让动作全错。

   对策：**没有 manifest 就拒绝运行**。manifest 由 pc/convert/convert_to_rknn.py
   暴力枚举排列、与 ONNX Runtime 逐值比对后生成。

2. **输出可能静默全零 / 非有限。** 同类项目里出现过（INT64 ReduceMin 导致 CPU fallback 失败）。
   对策：每次推理都检查 shape / finite / 零值比例。

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

log = logging.getLogger(__name__)

# ImageNet 归一化（仅当 graph 内未包含归一化时才需要）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ActConfig:
    rknn_model: str
    manifest: str
    denorm_json: str
    action_dim: int = 6
    state_dim: int = 6
    chunk_size: int = 100
    # 图像预处理（默认需用 PC 侧 dump 的真实张量逐元素核对，不要猜）
    image_norm: str = "zero_one"  # none | zero_one | imagenet
    image_resize: str = "squash"  # squash | pad
    # pad 模式的补零锚点。
    #   top_left    = 补零在顶部/左侧，图像内容靠右下
    #   bottom_right= 补零在底部/右侧，图像内容靠左上
    # 默认 top_left：与 IB_Robot 板端实测记录一致
    # （"480x640 -> 512x512 后顶部 128 行为零"）。
    # ⚠️ 仍必须用 PC 侧 dump 的真实张量逐元素核对 —— 方向错了不会报错，只会让策略失效。
    image_pad_anchor: str = "top_left"
    zero_ratio_error: float = 0.98
    latency_window: int = 100


@dataclass
class _LatencyTracker:
    window: int = 100
    _samples: deque = field(default_factory=deque, init=False)

    def add(self, ms: float) -> None:
        self._samples.append(ms)
        while len(self._samples) > self.window:
            self._samples.popleft()

    def summary(self) -> dict:
        if not self._samples:
            return {}
        a = np.asarray(self._samples, dtype=np.float64)
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
        self.latency = _LatencyTracker(cfg.latency_window)

        self.manifest = self._load_manifest(cfg.manifest)
        self.scale, self.offset, self.joint_names = self._load_denorm(cfg.denorm_json)

        if len(self.scale) != cfg.action_dim:
            raise ValueError(
                f"denorm.json 有 {len(self.scale)} 个关节，配置 action_dim={cfg.action_dim}，不一致"
            )

        from rknnlite.api import RKNNLite  # 延迟导入，便于在 PC 上做单元测试

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(cfg.rknn_model) != 0:
            raise RuntimeError(f"load_rknn 失败: {cfg.rknn_model}")
        if self.rknn.init_runtime(target=None) != 0:  # None = 本机 NPU
            raise RuntimeError("init_runtime 失败（检查 librknnrt 版本是否与 toolkit 匹配）")

        self.input_order: list[str] = self.manifest["input_order"]
        self.image_size: tuple[int, int] = tuple(self.manifest["image_size"])  # (H, W)
        self.image_layout: str = str(self.manifest.get("image_layout", "nchw")).lower()
        if self.image_layout not in ("nchw", "nhwc"):
            raise ValueError(f"manifest.image_layout 非法: {self.image_layout}")
        log.info("ACT RKNN 就绪 | 输入顺序=%s | 图像尺寸=%s | layout=%s",
                 self.input_order, self.image_size, self.image_layout)

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

    @staticmethod
    def _load_denorm(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        scale = np.asarray(d["scale"], dtype=np.float32)
        offset = np.asarray(d["offset"], dtype=np.float32)
        joints = list(d.get("joints", [f"j{i}" for i in range(len(scale))]))
        residual = d.get("fit_residual_max")
        if residual is not None and residual > 1e-3:
            log.warning("denorm 拟合残差偏大 (%.4g)，反归一化可能不准，请回到 PC 侧核对", residual)
        return scale, offset, joints

    # ---------- 预处理 ----------

    def _prep_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
            if self.cfg.image_resize == "pad":
                # 等比例缩放 + 补零（几何必须与 LeRobot resize_with_pad 一致）
                scale = min(h / frame_bgr.shape[0], w / frame_bgr.shape[1])
                nh = int(round(frame_bgr.shape[0] * scale))
                nw = int(round(frame_bgr.shape[1] * scale))
                resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros((h, w, 3), dtype=frame_bgr.dtype)
                if self.cfg.image_pad_anchor == "bottom_right":
                    canvas[:nh, :nw] = resized      # 内容靠左上，补零在右下
                else:  # top_left（默认）
                    canvas[h - nh :, w - nw :] = resized  # 内容靠右下，补零在左上
                frame_bgr = canvas
            else:
                frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.cfg.image_norm == "zero_one":
            rgb /= 255.0
        elif self.cfg.image_norm == "imagenet":
            rgb /= 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        # none: 原样传（仅当 graph 内已含归一化时使用）
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])  # (1,3,H,W)

    # ---------- 推理 ----------

    def infer(self, frames: dict[str, np.ndarray], state_raw: np.ndarray) -> np.ndarray:
        """返回 (chunk_size, action_dim) 的**原始计数**目标位置。

        frames: {槽位名: BGR 图像}，槽位名取自 manifest.image_slots。
                单相机配置下，所有槽位传同一帧即可（会打警告）。
        """
        slots: dict[str, np.ndarray] = {
            "state": np.asarray(state_raw, dtype=np.float32).reshape(1, -1)
        }
        for name, frame in frames.items():
            img = self._prep_image(frame)
            # 多相机若形状不一致，说明训练与部署的相机配置不同 —— 必须拦下
            if name in self.manifest.get("image_shapes", {}):
                want = tuple(self.manifest["image_shapes"][name])
                if tuple(img.shape) != want:
                    raise ValueError(f"槽位 {name} 形状 {img.shape} 与训练时 {want} 不一致")
            slots[name] = img

        missing = [n for n in self.input_order if n not in slots]
        if missing:
            raise ValueError(
                f"manifest.input_order 需要的槽位 {missing} 未提供；"
                f"已提供 {sorted(slots)}。检查 configs/robot.yaml 的 camera 配置。"
            )
        # 按 manifest 声明的语义顺序组装（state 是 2 维，图像是 4 维）
        inputs = [slots[name] for name in self.input_order]

        # 本类产出的图像张量是 NCHW（_prep_image 里 transpose 过）
        if self.image_layout != "nchw":
            raise ValueError(
                f"manifest 声明 image_layout={self.image_layout}，但本类的预处理产出 NCHW。"
                "要么改 manifest，要么改 _prep_image —— 不要在这里悄悄转置。"
            )

        t0 = time.perf_counter()
        # ⚠️ 必须显式声明 data_format='nchw'。
        #    rknn.inference() 的默认值是 'nhwc'，会把 NCHW 的 (1,3,H,W) 当成
        #    (1,H,W,3) 去解释，直接报
        #      "The input(ndarray) shape (1,3,480,640) is wrong, expect 'nhwc' like (1,480,640,3)"
        #    这类错误在板端表现为推理失败或数值全错，必须在代码里钉死。
        outputs = self.rknn.inference(inputs=inputs, data_format="nchw")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.add(dt_ms)

        if not outputs:
            raise RuntimeError("RKNN 未返回任何输出")
        out = np.asarray(outputs[0], dtype=np.float32)

        self._validate_output(out)

        flat = out.reshape(-1, self.cfg.action_dim)
        raw = flat * self.scale[None, :] + self.offset[None, :]
        return raw.astype(np.float32)

    def _validate_output(self, out: np.ndarray) -> None:
        if out.ndim != 3 or out.shape[0] != 1:
            raise RuntimeError(f"输出形状异常: {out.shape}，期望 (1, chunk, dim)")
        if out.shape[2] != self.cfg.action_dim:
            raise RuntimeError(f"动作维度异常: {out.shape[2]}，期望 {self.cfg.action_dim}")
        if not np.isfinite(out).all():
            n_bad = int((~np.isfinite(out)).sum())
            raise RuntimeError(f"输出含 {n_bad} 个 NaN/Inf —— 立即停机")
        zero_ratio = float((np.abs(out) < 1e-9).mean())
        if zero_ratio > self.cfg.zero_ratio_error:
            raise RuntimeError(
                f"输出 {zero_ratio:.1%} 为零值（阈值 {self.cfg.zero_ratio_error:.0%}）—— "
                "疑似 runtime fallback 失败或输入顺序错误，立即停机"
            )

    def warmup(self, n: int = 3) -> None:
        dummy_img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        dummy_state = np.full((self.cfg.state_dim,), 2047.0, dtype=np.float32)
        frames = {s: dummy_img for s in self.manifest.get("image_slots", ["front"])}
        for _ in range(n):
            self.infer(frames, dummy_state)
        self.latency._samples.clear()
        log.info("warmup 完成（已清空延迟统计）")

    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:  # noqa: BLE001
            pass
