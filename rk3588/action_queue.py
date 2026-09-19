"""动作块队列 —— 刻意保持简单。

参考 D-Robotics/rdk_LeRobot_tools 的做法：那边就是一个
`deque([], maxlen=n_action_steps)`，队列空了才推理，然后 `popleft()`。
没有必要为此写一个复杂的状态机。

这里只补三件它没有、但真机上必须有的东西：

1. **饥饿计数**：队列空是核心健康指标；不能默默 hold 到底
2. **hold 兜底**：空队列时保持上一步，而不是抛异常或跳到中位
3. **fps 换算**：数据集 fps 与控制频率不同时做线性插值
   （rdk 那里两者都是 30，所以它不需要）

跨块平滑（blend_steps）是可选项：两个动作块在交界处可能跳变，
RTC 是学术界解法但需要 autograd、静态 RKNN 图无法表达；
这里用便宜的线性过渡顶上。设 0 关闭。
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
    chunks: int = 0
    popped: int = 0
    starved: int = 0
    stale_dropped: int = 0
    blends: int = 0

    def as_dict(self) -> dict:
        total = self.popped or 1
        return {
            "chunks": self.chunks,
            "popped": self.popped,
            "starved": self.starved,
            "starve_ratio": round(self.starved / total, 5),
            "stale_dropped": self.stale_dropped,
            "blends": self.blends,
        }


class ActionQueue:
    """把一次推理产出的 (N, D) 动作块，按控制频率逐步喂出去。"""

    def __init__(
        self,
        action_dim: int,
        dataset_fps: float,
        control_freq_hz: float,
        max_chunk_age_s: float = 3.0,
        interpolate: bool = True,
        blend_steps: int = 0,
        on_empty: str = "hold",
        hold_value: np.ndarray | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.step = max(dataset_fps / control_freq_hz, 1e-6)
        self.max_chunk_age_s = max_chunk_age_s
        self.interpolate = interpolate
        self.blend_steps = blend_steps
        self.on_empty = on_empty

        self._q: deque[tuple[np.ndarray, float]] = deque()  # (actions, pushed_at)
        self._cursor = 0.0
        self._last: np.ndarray | None = None
        self._hold = hold_value
        self._blend_from: np.ndarray | None = None
        self._blend_left = 0
        self.stats = QueueStats()

    # ---------- 生产端 ----------

    def push(self, actions: np.ndarray, pushed_at: float | None = None) -> None:
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != self.action_dim:
            raise ValueError(f"动作块形状应为 (N,{self.action_dim})，实际 {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("动作块含 NaN/Inf，拒绝入队")

        self._q.append((a, pushed_at if pushed_at is not None else time.monotonic()))
        self.stats.chunks += 1

    @property
    def steps_available(self) -> int:
        if not self._q:
            return 0
        cur = self._q[0][0].shape[0]
        return int(max(0.0, cur - self._cursor)) + sum(a.shape[0] for a, _ in list(self._q)[1:])

    # ---------- 消费端 ----------

    def pop(self) -> np.ndarray:
        self.stats.popped += 1
        self._drop_stale()

        while self._q and self._q[0][0].shape[0] <= self._cursor:
            self._q.popleft()
            self._cursor = 0.0
            if self._last is not None and self.blend_steps > 0:
                self._blend_from = self._last.copy()
                self._blend_left = self.blend_steps
                self.stats.blends += 1

        if not self._q:
            return self._empty()

        actions = self._q[0][0]
        a = self._sample(actions, self._cursor)

        if self._blend_left > 0 and self._blend_from is not None:
            alpha = 1.0 - self._blend_left / max(self.blend_steps, 1)
            a = (1.0 - alpha) * self._blend_from + alpha * a
            self._blend_left -= 1
            if self._blend_left == 0:
                self._blend_from = None

        self._cursor += self.step
        self._last = a
        return a

    def _sample(self, actions: np.ndarray, pos: float) -> np.ndarray:
        n = actions.shape[0]
        if not self.interpolate or pos >= n - 1:
            return actions[min(int(pos), n - 1)].copy()
        i0 = int(pos)
        frac = float(pos - i0)
        return ((1.0 - frac) * actions[i0] + frac * actions[i0 + 1]).astype(np.float32)

    def _drop_stale(self) -> None:
        now = time.monotonic()
        while self._q and (now - self._q[0][1]) > self.max_chunk_age_s:
            self._q.popleft()
            self._cursor = 0.0
            self.stats.stale_dropped += 1
            log.warning("丢弃陈旧动作块（累计 %d）", self.stats.stale_dropped)

    def _empty(self) -> np.ndarray:
        self.stats.starved += 1
        if self.on_empty == "error":
            raise RuntimeError("动作队列饥饿 (on_empty=error)")
        if self._last is not None:
            return self._last.copy()
        if self._hold is not None:
            return self._hold.copy()
        return np.zeros(self.action_dim, dtype=np.float32)
