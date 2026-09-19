"""配置与 ActRKNN 防护逻辑测试。

重点验证"静默失败"防线：
  * manifest 缺失时必须拒绝运行（而不是猜一个输入顺序）
  * 输出 NaN / 全零 / 形状错时必须报错（而不是照常驱动机械臂）
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

    ids = cfg["robot"]["motor_ids"]
    assert set(ids) == set(joints), f"motor_ids 与 joints 不匹配: {set(ids) ^ set(joints)}"
    assert sorted(ids.values()) == [1, 2, 3, 4, 5, 6], "舵机 ID 应为 1..6 且不重复"

    limits = cfg["robot"]["raw_limits"]
    assert set(limits) == set(joints), "raw_limits 与 joints 不匹配"
    for j, (lo, hi) in limits.items():
        assert 0 <= lo < hi <= 4095, f"{j} 限位非法: [{lo}, {hi}]"


def test_runtime_config_sane() -> None:
    cfg = _load("robot.yaml")
    rt = cfg["runtime"]
    assert rt["control_freq_hz"] > 0
    assert 0 < rt["refill_threshold"] <= cfg["policy"]["chunk_size"], \
        "refill_threshold 应小于等于一个块的步数，否则永远补不上"
    assert rt["queue_max_steps"] >= cfg["policy"]["chunk_size"], \
        "queue_max_steps 至少能装下一个完整块"
    assert rt["on_queue_empty"] in ("hold", "error")
    assert cfg["policy"]["chunk_size"] == 100, "与已验证的 121ms 配置一致 (1,100,6)"


def test_feetech_register_table() -> None:
    ft = _load("feetech_sts3215.yaml")
    regs = ft["registers"]
    assert regs["Goal_Position"] == [42, 2]
    assert regs["Present_Position"] == [56, 2]
    assert regs["Torque_Enable"] == [40, 1]
    assert regs["Present_Temperature"] == [63, 1]
    assert ft["model"]["resolution"] == 4096
    assert ft["model"]["baudrate"] == 1_000_000
    assert ft["model"]["model_number"] == 777


# ---------------- ActRKNN 防线 ----------------

def _bare(cfg: ActConfig) -> ActRKNN:
    """绕过 __init__（不需要真 RKNN），只为测防护逻辑。"""
    obj = ActRKNN.__new__(ActRKNN)
    obj.cfg = cfg
    return obj


def _cfg(tmp: Path) -> ActConfig:
    return ActConfig(
        rknn_model=str(tmp / "x.rknn"),
        manifest=str(tmp / "x.manifest.json"),
        denorm_json=str(tmp / "denorm.json"),
        action_dim=6,
        state_dim=6,
        chunk_size=100,
    )


def test_missing_manifest_refuses_to_run() -> None:
    with _env.tmpdir() as d:
        tmp = Path(d)
        cfg = _cfg(tmp)
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


def test_manifest_ok() -> None:
    with _env.tmpdir() as d:
        p = Path(d) / "m.json"
        p.write_text(json.dumps({
            "input_order": ["state", "front"],
            "image_slots": ["front"],
            "image_size": [480, 640],
        }), encoding="utf-8")
        m = ActRKNN._load_manifest(str(p))
        assert m["input_order"] == ["state", "front"]


def test_denorm_high_residual_warns(tmp_path_factory=None) -> None:
    with _env.tmpdir() as d:
        p = Path(d) / "d.json"
        p.write_text(json.dumps({
            "joints": list("abcdef"),
            "scale": [1.0] * 6,
            "offset": [0.0] * 6,
            "fit_residual_max": 0.5,   # 偏大
        }), encoding="utf-8")
        scale, offset, joints = ActRKNN._load_denorm(str(p))  # 只应打警告
        assert len(scale) == 6 and len(offset) == 6


def test_validate_output_rejects() -> None:
    cfg = _cfg(Path("."))
    obj = _bare(cfg)

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


def test_validate_output_allows_small_values() -> None:
    """不能把"数值很小"误判成"全零失效"。"""
    cfg = _cfg(Path("."))
    obj = _bare(cfg)
    a = np.full((1, 100, 6), 1e-6, dtype=np.float32)
    obj._validate_output(a)


def test_prep_image_shape_and_channel_order() -> None:
    """用 cv2 桩验证形状/值域/通道顺序（BGR 输入 -> RGB 输出）。"""
    cfg = _cfg(Path("."))
    cfg.image_norm = "zero_one"
    cfg.image_resize = "squash"
    obj = _bare(cfg)
    obj.image_size = (480, 640)

    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # 蓝色通道
    out = obj._prep_image(bgr)
    assert out.shape == (1, 3, 480, 640), out.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6
    # BGR -> RGB：原蓝色应落在第 3 个通道（索引 2）
    assert out[0, 2].max() > 0.9, "蓝色通道没有落到 RGB 的 B 位（通道顺序错）"
    assert out[0, 0].max() < 0.1, "红色通道不应有值"


def test_prep_image_pad_mode_geometry() -> None:
    """pad 模式：等比例缩放 + 补零，锚点必须可切换。

    默认 top_left —— 与 IB_Robot 板端实测记录一致：
        "480x640 -> 512x512 后顶部 128 行为零"
    方向错了不会报错，只会让策略失效，所以这里把两种锚点都钉住。
    """
    cfg = _cfg(Path("."))
    cfg.image_resize = "pad"

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)

    # --- 锚点 top_left：内容靠右下，顶部 128 行为零 ---
    cfg.image_pad_anchor = "top_left"
    obj = _bare(cfg)
    obj.image_size = (512, 512)
    out = obj._prep_image(frame)
    assert out.shape == (1, 3, 512, 512)
    assert np.allclose(out[0, :, :128, :], 0.0), "top_left: 顶部 128 行应为零"
    assert out[0, :, 128:, :].max() > 0.9, "top_left: 缩放后的图像内容丢失"

    # --- 锚点 bottom_right：内容靠左上，底部 128 行为零 ---
    cfg.image_pad_anchor = "bottom_right"
    obj2 = _bare(cfg)
    obj2.image_size = (512, 512)
    out2 = obj2._prep_image(frame)
    assert np.allclose(out2[0, :, 384:, :], 0.0), "bottom_right: 底部 128 行应为零"
    assert out2[0, :, :384, :].max() > 0.9, "bottom_right: 缩放后的图像内容丢失"

    # 两种锚点在垂直方向应是翻转关系
    assert not np.allclose(out, out2), "两种锚点产生了相同结果，锚点参数没生效"


def test_infer_passes_nchw_data_format() -> None:
    """回归保护：rknn.inference 默认按 nhwc 解释输入，必须显式传 data_format='nchw'。

    实测报错（未传时）：
        The input(ndarray) shape (1,3,480,640) is wrong,
        expect 'nhwc' like (1,480,640,3)
    """
    from rknnlite.api import RKNNLite

    with _env.tmpdir() as d:
        manifest = Path(d) / "m.json"
        manifest.write_text(json.dumps({
            "input_order": ["state", "front"],
            "image_slots": ["front"],
            "image_shapes": {"front": [1, 3, 480, 640]},
            "image_size": [480, 640],
            "image_layout": "nchw",
            "output_shape": [1, 100, 6],
        }), encoding="utf-8")
        denorm = Path(d) / "denorm.json"
        denorm.write_text(json.dumps({
            "joints": list("abcdef"),
            "scale": [100.0] * 6,
            "offset": [2047.0] * 6,
            "fit_residual_max": 0.0,
        }), encoding="utf-8")

        cfg = ActConfig(
            rknn_model=str(Path(d) / "fake.rknn"),
            manifest=str(manifest),
            denorm_json=str(denorm),
            action_dim=6, state_dim=6, chunk_size=100,
        )
        act = ActRKNN(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = act.infer({"front": frame}, np.full(6, 2047.0, dtype=np.float32))

        assert RKNNLite.last_inference_kwargs.get("data_format") == "nchw", \
            f"必须以 nchw 调用，实际: {RKNNLite.last_inference_kwargs}"
        assert out.shape == (100, 6), out.shape
        # 反归一化应生效：0.25 * 100 + 2047 = 2072
        assert np.allclose(out[0], 2072.0, atol=1e-3), out[0]


def test_manifest_rejects_non_nchw_layout() -> None:
    """manifest 声明 nhwc 时应当直接拒绝，而不是悄悄转置。"""
    with _env.tmpdir() as d:
        manifest = Path(d) / "m.json"
        manifest.write_text(json.dumps({
            "input_order": ["state", "front"],
            "image_slots": ["front"],
            "image_size": [480, 640],
            "image_layout": "nhwc",
        }), encoding="utf-8")
        denorm = Path(d) / "denorm.json"
        denorm.write_text(json.dumps({
            "joints": list("abcdef"), "scale": [1.0] * 6, "offset": [0.0] * 6,
        }), encoding="utf-8")
        cfg = ActConfig(
            rknn_model=str(Path(d) / "fake.rknn"),
            manifest=str(manifest), denorm_json=str(denorm),
            action_dim=6, state_dim=6, chunk_size=100,
        )
        act = ActRKNN(cfg)
        try:
            act.infer({"front": np.zeros((480, 640, 3), dtype=np.uint8)},
                      np.zeros(6, dtype=np.float32))
        except ValueError as e:
            assert "image_layout" in str(e)
            return
        raise AssertionError("nhwc layout 应被拒绝")


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
    print(f"配置 + ActRKNN 防护: {len(tests) - failed}/{len(tests)} 通过")
    return failed


if __name__ == "__main__":
    sys.exit(main())
