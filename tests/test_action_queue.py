"""ActionQueue 逻辑测试。

这是全项目最容易出隐性错误的地方：插值倍率、跨块平滑、陈旧丢弃、饥饿计数。
用纯 numpy 就能完整验证，不需要板卡。
"""

from __future__ import annotations

import sys

import numpy as np

import _env  # noqa: F401  (路径注入)

from action_queue import ActionQueue  # noqa: E402


def _q(**kw) -> ActionQueue:
    base = dict(
        dataset_fps=30.0,
        control_freq_hz=30.0,
        action_dim=6,
        queue_max_steps=200,
        max_chunk_age_s=999.0,  # 默认不触发陈旧丢弃
        interpolate=True,
        blend_steps=0,          # 默认关掉平滑，先验证基础语义
        on_queue_empty="hold",
    )
    base.update(kw)
    return ActionQueue(**base)


def test_linear_consumption() -> None:
    """同频时应逐帧原样出队。"""
    q = _q()
    chunk = np.arange(60, dtype=np.float32).reshape(10, 6)
    q.push(chunk)
    for i in range(10):
        got = q.pop()
        assert np.allclose(got, chunk[i]), f"第 {i} 步: {got} != {chunk[i]}"
    assert q.stats.popped_steps == 10


def test_interpolation_upsample() -> None:
    """15 fps 数据块以 30 Hz 执行 -> 每两拍插一个中间值。"""
    q = _q(dataset_fps=15.0, control_freq_hz=30.0)
    a = np.zeros(6, dtype=np.float32)
    b = np.full(6, 100.0, dtype=np.float32)
    q.push(np.stack([a, b]))  # 2 帧
    s0 = q.pop()
    s1 = q.pop()
    s2 = q.pop()
    # step = dataset_fps/control_freq = 0.5，所以游标依次落在 0 / 0.5 / 1.0
    assert np.allclose(s0, a), s0
    assert np.allclose(s1, 50.0), s1          # 中点
    assert np.allclose(s2, b), s2             # 到达第 2 帧


def test_downsample() -> None:
    """60 fps 数据块以 30 Hz 执行 -> 隔一帧取一个。"""
    q = _q(dataset_fps=60.0, control_freq_hz=30.0)
    chunk = np.arange(60, dtype=np.float32).reshape(10, 6)
    q.push(chunk)
    assert np.allclose(q.pop(), chunk[0])
    assert np.allclose(q.pop(), chunk[2])


def test_starvation_and_hold() -> None:
    """队列空时必须 hold 最后值，并计数。"""
    q = _q()
    q.push(np.full((2, 6), 7.0, dtype=np.float32))
    q.pop()
    last = q.pop()
    held = q.pop()  # 空队列
    assert np.allclose(held, last), f"应保持 {last}，实际 {held}"
    assert q.stats.starve_count == 1
    q.pop()
    assert q.stats.starve_count == 2


def test_starvation_before_any_chunk() -> None:
    """从未收到动作时返回中位(2047)，不能崩。"""
    q = _q()
    got = q.pop()
    assert np.allclose(got, 2047.0), got
    assert q.stats.starve_count == 1


def test_stale_chunk_dropped() -> None:
    """陈旧块必须被丢弃，不能执行过期动作。"""
    q = _q(max_chunk_age_s=0.0)
    q.push(np.full((5, 6), 123.0, dtype=np.float32), pushed_at=0.0)  # 远古时间戳
    got = q.pop()
    assert q.stats.dropped_stale >= 1
    assert not np.allclose(got, 123.0), "陈旧块被执行了"


def test_overflow_drops_oldest() -> None:
    """推理远快于消费时，超容量丢最旧块。"""
    q = _q(queue_max_steps=20)
    for k in range(5):
        q.push(np.full((10, 6), float(k), dtype=np.float32))
    assert q.steps_available <= 20, q.steps_available
    assert q.stats.dropped_overflow >= 1


def test_reject_bad_chunk() -> None:
    q = _q()
    for bad in (
        np.zeros((10, 5), dtype=np.float32),                       # 维度错
        np.full((10, 6), np.nan, dtype=np.float32),                # NaN
        np.full((10, 6), np.inf, dtype=np.float32),                # Inf
    ):
        try:
            q.push(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝 {bad.shape} / {bad.flat[0]}")


def test_blend_across_chunks() -> None:
    """跨块平滑：新块生效时应从上一块末值渐变，而不是突跳。"""
    q = _q(blend_steps=4, dataset_fps=30.0, control_freq_hz=30.0)
    q.push(np.full((2, 6), 0.0, dtype=np.float32))     # 块1: 全 0
    q.pop()
    q.pop()
    q.push(np.full((2, 6), 100.0, dtype=np.float32))   # 块2: 全 100
    first = q.pop()  # 切换瞬间
    assert 0.0 <= first[0] < 100.0, f"切换第一步应介于中间，实际 {first[0]}"
    assert q.stats.blend_events == 1
    later = q.pop()
    assert later[0] >= first[0], "应单调靠近新块"


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
    for k in ("pushed_chunks", "pushed_steps", "popped_steps", "starve_count", "starve_ratio"):
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
