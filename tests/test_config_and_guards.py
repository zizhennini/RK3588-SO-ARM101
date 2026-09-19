"""配置自洽性 + 归一化数学 + ActRKNN 防护逻辑测试。

重点验证两类问题：
  * 归一化算错（图像不做 z-score、通道顺序反了）—— 静默失效，动作整体偏
  * 防护失效（缺 manifest 还跑、输出全零还执行）—— 会动真机
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

import _env  # noqa: F401

_env.install_stubs()

from act_rknn import ActConfig, ActRKNN  # noqa: E402
from stats import NormStats  # noqa: E402


# ---------------- 配置自洽性 ----------------

def _load(name: str) -> dict:
    return yaml.safe_load((_env.CONFIGS / name).read_text(encoding="utf-8"))


def test_robot_config_consistency() -> None:
    cfg = _load("robot.yaml")
    joints = cfg["robot"]["joints"]
    pol = cfg["policy"]

    assert len(joints) == pol["action_dim"], "joints 数量必须等于 action_dim"
    assert len(joints) == pol["state_dim"], "joints 数量必须等于 state_dim"
    assert len(set(joints)) == len(joints), "关节名重复"
    assert joints[-1] == "gripper", "最后一个关节应为夹爪"

    # 相机槽位必须与 policy.image_slots 一致
    assert list(cfg["camera"]["slots"]) == list(pol["image_slots"]), \
        "camera.slots 与 policy.image_slots 必须一致"

    # 安全机制：两者至少要有一个开着
    assert cfg["robot"].get("max_relative_target") is not None or \
        cfg["robot"].get("action_limits") is not None, \
        "max_relative_target 与 action_limits 不能同时为空，否则没有安全限制"

    mrt = cfg["robot"].get("max_relative_target")
    if mrt is not None:
        assert 0 < mrt <= 100, f"max_relative_target 应在 (0,100]，实际 {mrt}"


def test_runtime_config_sane() -> None:
    cfg = _load("robot.yaml")
    rt = cfg["runtime"]
    pol = cfg["policy"]
    cam = cfg["camera"]

    assert rt["control_freq_hz"] > 0
    assert cam["fps"] > 0
    assert rt["on_queue_empty"] in ("hold", "error")
    assert pol["chunk_size"] == 100, "与已验证的 121ms 配置一致 (1,100,6)"
    assert rt["max_chunk_age_s"] > 0
    # 一个动作块覆盖的时长要大于单次推理耗时（121ms 量级）才有意义
    block_s = pol["chunk_size"] / pol["dataset_fps"]
    assert block_s > 0.5, f"动作块只覆盖 {block_s:.2f}s，太短"
    assert pol.get("image_pad_anchor", "top_left") in ("top_left", "bottom_right")


# ---------------- 归一化数学（stats.py） ----------------

def _write_stats(root: Path, cameras=("front",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for c in cameras:
        np.save(root / f"{c}_mean.npy", np.array([0.5, 0.4, 0.3], dtype=np.float32).reshape(3, 1, 1))
        np.save(root / f"{c}_std.npy", np.array([0.2, 0.1, 0.25], dtype=np.float32).reshape(3, 1, 1))
    np.save(root / "state_mean.npy", np.zeros(6, dtype=np.float32))
    np.save(root / "state_std.npy", np.ones(6, dtype=np.float32))
    np.save(root / "action_mean.npy", np.arange(6, dtype=np.float32) * 10)
    np.save(root / "action_std.npy", np.full(6, 2.0, dtype=np.float32))
    (root / "norm_stats.json").write_text(json.dumps({
        "format": "rkrobot.norm_stats.v1",
        "normalization": "MEAN_STD",
        "cameras": list(cameras),
    }), encoding="utf-8")
    return root


def test_norm_stats_image_is_mean_std_not_just_div255() -> None:
    """图像必须做 z-score。只 /255 是错的，而且不会报错。"""
    with _env.tmpdir() as d:
        st = NormStats.load(_write_stats(Path(d)))
        x = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)  # 三个通道都是 0.5
        out = st.normalize_image(x, "front")
        # 0.5 相对各通道 mean/std 的结果应各不相同（说明确实按通道做了 z-score）
        vals = [float(out[0, c].mean()) for c in range(3)]
        assert len(set(round(v, 6) for v in vals)) == 3, f"三个通道结果相同，说明没做逐通道 z-score: {vals}"
        assert np.allclose(vals[0], (0.5 - 0.5) / 0.2), vals


def test_norm_stats_action_roundtrip() -> None:
    with _env.tmpdir() as d:
        st = NormStats.load(_write_stats(Path(d)))
        a_norm = np.zeros((2, 6), dtype=np.float32)
        raw = st.denormalize_action(a_norm)
        assert np.allclose(raw, np.arange(6) * 10), raw  # 0 * std + mean = mean
        back = (raw - st.action_mean) / st.action_std
        assert np.allclose(back, a_norm, atol=1e-5)


def test_norm_stats_missing_file_refuses() -> None:
    with _env.tmpdir() as d:
        try:
            NormStats.load(Path(d))
        except FileNotFoundError as e:
            assert "norm_stats" in str(e)
            return
        raise AssertionError("缺少统计量时必须拒绝运行")


def test_norm_stats_dim_mismatch() -> None:
    with _env.tmpdir() as d:
        st = NormStats.load(_write_stats(Path(d)))
        try:
            st.normalize_state(np.zeros(5, dtype=np.float32))
        except ValueError:
            return
        raise AssertionError("state 维度不符应报错")


# ---------------- ActRKNN 防护 ----------------

def _bare(cfg: ActConfig) -> ActRKNN:
    obj = ActRKNN.__new__(ActRKNN)
    obj.cfg = cfg
    return obj


def _cfg(tmp: Path) -> ActConfig:
    return ActConfig(
        rknn_model=str(tmp / "x.rknn"),
        manifest=str(tmp / "x.manifest.json"),
        norm_stats_dir=str(tmp / "stats"),
        action_dim=6, state_dim=6, chunk_size=100,
    )


def test_missing_manifest_refuses_to_run() -> None:
    with _env.tmpdir() as d:
        cfg = _cfg(Path(d))
        try:
            _bare(cfg)._load_manifest(cfg.manifest)
        except FileNotFoundError as e:
            assert "manifest" in str(e).lower() or "输入顺序" in str(e)
            return
        raise AssertionError("缺少 manifest 时必须拒绝运行")


def test_manifest_missing_fields() -> None:
    with _env.tmpdir() as d:
        p = Path(d) / "m.json"
        p.write_text(json.dumps({"format": "x"}), encoding="utf-8")
        try:
            ActRKNN._load_manifest(str(p))
        except ValueError:
            return
        raise AssertionError("manifest 缺字段应报错")


def _full_setup(d: Path, layout: str = "nchw"):
    stats_dir = _write_stats(d / "stats")
    manifest = d / "m.json"
    manifest.write_text(json.dumps({
        "input_order": ["state", "front"],
        "image_slots": ["front"],
        "image_shapes": {"front": [1, 3, 4, 4]},
        "image_size": [4, 4],
        "image_layout": layout,
        "output_shape": [1, 100, 6],
    }), encoding="utf-8")
    return ActConfig(
        rknn_model=str(d / "fake.rknn"),
        manifest=str(manifest),
        norm_stats_dir=str(stats_dir),
        action_dim=6, state_dim=6, chunk_size=100,
    )


def test_infer_passes_nchw_data_format() -> None:
    """回归保护：rknn.inference 默认按 nhwc 解释输入，必须显式传 data_format。

    实测报错（未传时）：
        The input(ndarray) shape (1,3,480,640) is wrong, expect 'nhwc' like (1,480,640,3)
    """
    from rknnlite.api import RKNNLite

    with _env.tmpdir() as d:
        act = ActRKNN(_full_setup(Path(d)))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        out = act.infer({"front": frame}, np.zeros(6, dtype=np.float32))
        assert RKNNLite.last_inference_kwargs.get("data_format") == "nchw", \
            f"必须以 nchw 调用，实际: {RKNNLite.last_inference_kwargs}"
        assert out.shape == (100, 6), out.shape


def test_manifest_rejects_non_nchw_layout() -> None:
    with _env.tmpdir() as d:
        try:
            ActRKNN(_full_setup(Path(d), layout="nhwc"))
        except ValueError as e:
            assert "image_layout" in str(e)
            return
        raise AssertionError("nhwc layout 应被拒绝")


def test_prep_image_does_not_swap_channels() -> None:
    """LeRobot 的相机默认输出 RGB —— 板端**不能**再做 BGR->RGB 转换。

    这里喂纯红 RGB，经过预处理后红色通道必须仍在索引 0。
    通道反了不会报错，只会让策略完全失效。
    """
    with _env.tmpdir() as d:
        act = ActRKNN(_full_setup(Path(d)))
        # 归一化统计量里 front 的 mean/std 是各通道不同的，先归零以便判断
        act.stats.cameras["front"] = (
            np.zeros((3, 1, 1), dtype=np.float32), np.ones((3, 1, 1), dtype=np.float32)
        )
        red = np.zeros((4, 4, 3), dtype=np.uint8)
        red[:, :, 0] = 255  # R=255, G=0, B=0
        chw = act._prep_image(red, "front")
        assert chw[0, 0].max() > 0.99, "红色通道丢失 —— 通道顺序错了"
        assert chw[0, 1].max() < 1e-6 and chw[0, 2].max() < 1e-6, "绿/蓝通道不应有值"


def test_prep_image_pad_anchor() -> None:
    with _env.tmpdir() as d:
        cfg = _full_setup(Path(d))
        cfg.image_pad_anchor = "top_left"
        act = ActRKNN(cfg)
        act.image_size = (8, 8)
        act.stats.cameras["front"] = (
            np.zeros((3, 1, 1), dtype=np.float32), np.ones((3, 1, 1), dtype=np.float32)
        )
        frame = np.full((2, 8, 3), 255, dtype=np.uint8)  # 很扁的图
        out = act._prep_image(frame, "front")
        assert out.shape == (1, 3, 8, 8)
        # 2x8 等比缩放到 2x8（不需缩放），top_left 应把内容放在底部
        assert out[0, :, :6, :].max() < 1e-6, "top_left: 上部应为 padding"
        assert out[0, :, 6:, :].max() > 0.9, "top_left: 内容应在底部"


def test_validate_output_rejects() -> None:
    with _env.tmpdir() as d:
        obj = _bare(_full_setup(Path(d)))
        good = np.zeros((1, 100, 6), dtype=np.float32)
        good[0, :, 0] = 0.5
        obj._validate_output(good)  # 不应抛

        cases = {
            "非3维": np.zeros((100, 6), dtype=np.float32),
            "batch!=1": np.zeros((2, 100, 6), dtype=np.float32),
            "动作维度错": np.zeros((1, 100, 7), dtype=np.float32),
            "含NaN": np.full((1, 100, 6), np.nan, dtype=np.float32),
            "含Inf": np.full((1, 100, 6), np.inf, dtype=np.float32),
            "全零": np.zeros((1, 100, 6), dtype=np.float32),
        }
        for name, arr in cases.items():
            try:
                obj._validate_output(arr)
            except RuntimeError:
                continue
            raise AssertionError(f"应拒绝: {name}")


def test_infer_rejects_unknown_camera_slot() -> None:
    with _env.tmpdir() as d:
        act = ActRKNN(_full_setup(Path(d)))
        try:
            act.infer({}, np.zeros(6, dtype=np.float32))
        except ValueError as e:
            assert "front" in str(e)
            return
        raise AssertionError("缺少相机帧应报错")


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
    print(f"配置 + 归一化 + 防护: {len(tests) - failed}/{len(tests)} 通过")
    return failed


if __name__ == "__main__":
    sys.exit(main())
