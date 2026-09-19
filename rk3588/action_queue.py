"""动作块队列：插值、跨块平滑、陈旧丢弃、饥饿兜底。

为什么必须有这一层
------------------
ACT 一次产出 `chunk_size`（默认 100）步动作。**绝不能"算一步、走一步"**：

  * 控制线程以固定频率出队，一步一个目标位置
  * 推理线程异步补块，`T_infer`（≈121 ms）远小于一个块的时长
    （100 步 / 30 fps ≈ 3.3 s），所以可以提前算好下一块
  * 队列空时必须 hold，并计数 —— **饥饿率是核心健康指标**

跨块平滑
--------
两个动作块在交界处可能跳变。RTC（Real-Time Chunking）是学术界的标准解法，
但它依赖 autograd，**在静态 ONNX/RKNN 图上无法表达**。
这里用一个便宜的替代：新块生效时，从上一块的末值线性过渡若干步。
有实测支持的同类做法（A2C2 / Soare 的 inpainting 近似）表明，
即使是简单混合也能显著减小交界抖动。

本文件是纯逻辑，不依赖 rknnlite / pyserial，可以直接单元测试。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class QueueStats:
    pushed_chunks: int = 0
    pushed_steps: int = 0
    popped_steps: int = 0
    starve_count: int = 0
    dropped_stale: int = 0
    dropped_overflow: int = 0
    blend_events: int = 0

    def as_dict(self) -> dict:
        total = self.popped_steps or 1
        return {
            "pushed_chunks": self.pushed_chunks,
            "pushed_steps": self.pushed_steps,
            "popped_steps": self.popped_steps,
            "starve_count": self.starve_count,
            "starve_ratio": round(self.starve_count / total, 5),
            "dropped_stale": self.dropped_stale,
            "dropped_overflow": self.dropped_overflow,
            "blend_events": self.blend_events,
        }


@dataclass
class _Chunk:
    actions: np.ndarray  # (N, D) 原始计数（已反归一化）
    pushed_at: float


@dataclass
class ActionQueue:
    dataset_fps: float
    control_freq_hz: float
    action_dim: int
    queue_max_steps: int = 200
    max_chunk_age_s: float = 3.0
    interpolate: bool = True
    blend_steps: int = 5
    on_queue_empty: str = "hold"  # hold | error

    _chunks: deque[_Chunk] = field(default_factory=deque, init=False)
    _cursor: float = field(default=0.0, init=False)  # 当前块内的采样位置（单位：数据集帧）
    _last_action: np.ndarray | None = field(default=None, init=False)
    _blend_from: np.ndarray | None = field(default=None, init=False)
    _blend_left: int = field(default=0, init=False)
    _pending_error: str | None = field(default=None, init=False)
    stats: QueueStats = field(default_factory=QueueStats, init=False)

    # ---------- 生产端 ----------

    def push(self, actions: np.ndarray, pushed_at: float | None = None) -> None:
        """推理线程调用。actions: (N, D) 原始计数。"""
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != self.action_dim:
            raise ValueError(f"动作块形状应为 (N,{self.action_dim})，实际 {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("动作块含 NaN/Inf，拒绝入队")

        self._chunks.append(_Chunk(a, pushed_at if pushed_at is not None else time.monotonic()))
        self.stats.pushed_chunks += 1
        self.stats.pushed_steps += a.shape[0]
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        """缓存步数超过上限时丢最旧的块（推理比消费快太多时才会发生）。"""
        while self._total_steps() > self.queue_max_steps and len(self._chunks) > 1:
            self._chunks.popleft()
            self.stats.dropped_overflow += 1

    def _total_steps(self) -> int:
        return sum(c.actions.shape[0] for c in self._chunks)

    # ---------- 消费端 ----------

    @property
    def steps_available(self) -> int:
        """当前块剩余 + 后续块总步数（用于 refill 判断）。"""
        if not self._chunks:
            return 0
        cur = self._chunks[0]
        remaining = max(0.0, cur.actions.shape[0] - self._cursor)
        return int(remaining) + sum(c.actions.shape[0] for c in list(self._chunks)[1:])

    def pop(self) -> np.ndarray:
        """控制线程每个节拍调用一次，返回 (D,) 目标原始计数。"""
        self.stats.popped_steps += 1
        self._drop_stale()

        if not self._chunks:
            return self._handle_empty()

        step = max(self.dataset_fps / self.control_freq_hz, 1e-6)
        chunk = self._chunks[0]
        n = chunk.actions.shape[0]

        if self._cursor >= n:
            # 当前块用完 -> 切到下一块，触发跨块平滑
            self._chunks.popleft()
            if not self._chunks:
                return self._handle_empty()
            self._cursor = 0.0
            if self._last_action is not None and self.blend_steps > 0:
                self._blend_from = self._last_action.copy()
                self._blend_left = self.blend_steps
                self.stats.blend_events += 1
            chunk = self._chunks[0]
            n = chunk.actions.shape[0]

        action = self._sample(chunk.actions, self._cursor)

        # 跨块线性过渡
        if self._blend_left > 0 and self._blend_from is not None:
            alpha = 1.0 - (self._blend_left / max(self.blend_steps, 1))
            action = (1.0 - alpha) * self._blend_from + alpha * action
            self._blend_left -= 1
            if self._blend_left == 0:
                self._blend_from = None

        self._cursor += step
        self._last_action = action
        return action

    def _sample(self, actions: np.ndarray, pos: float) -> np.ndarray:
        n = actions.shape[0]
        if not self.interpolate:
            return actions[min(int(pos), n - 1)].copy()
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, n - 1)
        frac = float(pos - i0)
        if i0 >= n - 1:
            return actions[n - 1].copy()
        return ((1.0 - frac) * actions[i0] + frac * actions[i1]).astype(np.float32)

    def _drop_stale(self) -> None:
        """丢掉时间戳过旧的块，避免执行过期动作。"""
        now = time.monotonic()
        # 只检查"还没开始消费"的块；正在消费的块如果已陈旧，也一并处理
        while self._chunks and (now - self._chunks[0].pushed_at) > self.max_chunk_age_s:
            self._chunks.popleft()
            self.stats.dropped_stale += 1
            self._cursor = 0.0
        if self.stats.dropped_stale:
            log.warning("丢弃陈旧动作块 %d 个", self.stats.dropped_stale)

    def _handle_empty(self) -> np.ndarray:
        self.stats.starve_count += 1
        if self._last_action is not None and self.on_queue_empty == "hold":
            return self._last_action.copy()
        if self.on_queue_empty == "error":
            self._pending_error = "动作队列饥饿 (on_queue_empty=error)"
            raise RuntimeError(self._pending_error)
        # 从未收到过任何动作：返回中位，让机械臂保持不动
        return np.full(self.action_dim, 2047.0, dtype=np.float32)

    def consume_error(self) -> str | None:
        err, self._pending_error = self._pending_error, None
        return err
