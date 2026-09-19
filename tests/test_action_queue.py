"""ActionQueue 逻辑测试。

队列刻意做得很薄（参照 D-Robotics/rdk_LeRobot_tools 用 deque + popleft 的做法），
所以这里只需要钉住几件真机上会出事的行为：
插值倍率、饥饿 hold 与计数、陈旧块丢弃、跨块平滑。
"""

from __future__ import annotations

import sys

import numpy as np

import _env  # noqa: F401  (路径注入)

from action_queue import ActionQueue  # noqa: E402


def _q(**kw) -> ActionQueue:
    base = dict(
        action_dim=6,
        dataset_fps=30.0,
        control_freq_hz=30.0,
        max_chunk_age_s=999.0,
        interpolate=True,
        blend_steps=0,
        on_empty="hold",
    )
    base.update(kw)
    return ActionQueue(**base)


def test_linear_consumption() -> None:
    q = _q()
    chunk = np.arange(60, dtype=np.float32).reshape(10, 6)
    q.push(chunk)
    for i in range(10):
        assert np.allclose(q.pop(), chunk[i]), f"第 {i} 步不符"
    assert q.stats.popped == 10


def test_interpolation_upsample() -> None:
    """15 fps 数据块以 30 Hz 执行 -> 每两拍插一个中间值。"""
    q = _q(dataset_fps=15.0, control_freq_hz=30.0)
    a = np.zeros(6, dtype=np.float32)
    b = np.full(6, 100.0, dtype=np.float32)
    q.push(np.stack([a, b]))
    assert np.allclose(q.pop(), a)
    assert np.allclose(q.pop(), 50.0), "中点插值"
    assert np.allclose(q.pop(), b), "到达第 2 帧"


def test_downsample() -> None:
    q = _q(dataset_fps=60.0, control_freq_hz=30.0)
    chunk = np.arange(60, dtype=np.float32).reshape(10, 6)
    q.push(chunk)
    assert np.allclose(q.pop(), chunk[0])
    assert np.allclose(q.pop(), chunk[2])


def test_starvation_holds_last_and_counts() -> None:
    q = _q()
    q.push(np.full((2, 6), 7.0, dtype=np.float32))
    q.pop()
    last = q.pop()
    held = q.pop()  # 空队列
    assert np.allclose(held, last), f"应保持 {last}，实际 {held}"
    assert q.stats.starved == 1
    q.pop()
    assert q.stats.starved == 2


def test_starvation_before_any_chunk() -> None:
    """从未收到动作时应给出 hold_value（或零），不能崩。"""
    q = _q(hold_value=np.full(6, 2047.0, dtype=np.float32))
    assert np.allclose(q.pop(), 2047.0)
    assert q.stats.starved == 1


def test_on_empty_error_raises() -> None:
    q = _q(on_empty="error")
    try:
        q.pop()
    except RuntimeError:
        assert q.stats.starved == 1
        return
    raise AssertionError("on_empty=error 时应抛错")


def test_stale_chunk_dropped() -> None:
    q = _q(max_chunk_age_s=0.0)
    q.push(np.full((5, 6), 123.0, dtype=np.float32), pushed_at=0.0)  # 远古时间戳
    got = q.pop()
    assert q.stats.stale_dropped >= 1
    assert not np.allclose(got, 123.0), "陈旧块被执行了"


def test_reject_bad_chunk() -> None:
    q = _q()
    for bad in (
        np.zeros((10, 5), dtype=np.float32),                 # 维度错
        np.full((10, 6), np.nan, dtype=np.float32),          # NaN
        np.full((10, 6), np.inf, dtype=np.float32),          # Inf
    ):
        try:
            q.push(bad)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝 {bad.shape} / {bad.flat[0]}")


def test_blend_across_chunks() -> None:
    """跨块平滑：新块生效时应从上一块末值渐变，而不是突跳。"""
    q = _q(blend_steps=4)
    q.push(np.full((2, 6), 0.0, dtype=np.float32))     # 块1: 全 0
    q.pop()
    q.pop()
    q.push(np.full((2, 6), 100.0, dtype=np.float32))   # 块2: 全 100
    first = q.pop()
    assert 0.0 <= first[0] < 100.0, f"切换第一步应介于中间，实际 {first[0]}"
    assert q.stats.blends == 1
    assert q.pop()[0] >= first[0], "应单调靠近新块"


def test_steps_available() -> None:
    q = _q()
    assert q.steps_available == 0
    q.push(np.zeros((10, 6), dtype=np.float32))
    assert q.steps_available == 10
    q.pop()
    assert q.steps_available == 9
    q.push(np.zeros((5, 6), dtype=np.float32))
    assert q.steps_available == 14


def test_stats_fields() -> None:
    q = _q()
    q.push(np.zeros((3, 6), dtype=np.float32))
    q.pop()
    d = q.stats.as_dict()
    for k in ("chunks", "popped", "starved", "starve_ratio", "stale_dropped", "blends"):
        assert k in d, f"缺少统计字段 {k}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"ActionQueue: {len(tests) - failed}/{len(tests)} 通过")
    return failed


if __name__ == "__main__":
    sys.exit(main())
