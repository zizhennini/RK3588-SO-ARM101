"""RKNN 工具链冒烟测试：确认 rknn-toolkit2 真的能转换，而不只是装上了。

不需要板卡。验证三件事：
  1. 基础算子（MatMul/Gemm/Add/Relu）能否转换成功
  2. 模型里出现 LayerNormalization 时，转换日志是否出现 `unsupport cpu <Op> op`
     —— 那是硬性阻断信号；官方算子表说 LayerNorm 支持，但它的分解体
        `exNorm:ReduceMean_0_2ln` 在历史案例里会 fallback 失败
  3. 转换产物大小与输入输出形状符合预期

在 `rknn` 环境里跑：
    conda activate rknn
    python pc/convert/toolchain_smoketest.py
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

OUT_DIR = Path(tempfile.gettempdir()) / "rknn_smoketest"


def make_mlp_onnx(path: Path) -> None:
    """y = relu(x @ W + b)，输入 (1, 64)。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.1, (64, 64)).astype(np.float32)
    b = rng.normal(0, 0.1, (64,)).astype(np.float32)

    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "b"], ["lin"]),
        helper.make_node("Relu", ["lin"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
        initializer=[
            numpy_helper.from_array(W, "W"),
            numpy_helper.from_array(b, "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def make_layernorm_onnx(path: Path) -> None:
    """带 LayerNormalization 的模型 —— 探测 transformer 相关算子的转换情况。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(1)
    scale = np.ones((64,), dtype=np.float32)
    bias = np.zeros((64,), dtype=np.float32)

    nodes = [
        helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["ln"], axis=-1),
        helper.make_node("MatMul", ["ln", "W"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "ln",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
        initializer=[
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(rng.normal(0, 0.1, (64, 64)).astype(np.float32), "W"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def convert(onnx_path: Path, rknn_path: Path) -> tuple[bool, str]:
    """返回 (是否成功, 日志)。"""
    # ⚠️ 必须在 import rknn 之前打垫片（onnx>=1.17 移除了 onnx.mapping）
    from rknn_onnx_compat import patch_onnx_mapping

    patch_onnx_mapping()

    from rknn.api import RKNN

    buf = io.StringIO()
    rknn = RKNN(verbose=False)
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            ret = rknn.config(
                target_platform="rk3588",
                float_dtype="float16",
                optimization_level=3,
                single_core_mode=False,
            )
            if ret != 0:
                return False, buf.getvalue() + f"\n[rknn.config ret={ret}]"
            if rknn.load_onnx(model=str(onnx_path)) != 0:
                return False, buf.getvalue() + "\n[load_onnx 失败]"
            if rknn.build(do_quantization=False) != 0:
                return False, buf.getvalue() + "\n[build 失败]"
            if rknn.export_rknn(str(rknn_path)) != 0:
                return False, buf.getvalue() + "\n[export_rknn 失败]"
        return True, buf.getvalue()
    except Exception as e:  # noqa: BLE001
        return False, buf.getvalue() + f"\n[异常] {type(e).__name__}: {e}"
    finally:
        try:
            rknn.release()
        except Exception:  # noqa: BLE001
            pass


def report(name: str, ok: bool, log: str, rknn_path: Path) -> bool:
    print(f"\n--- {name} ---")
    print(f"  转换结果: {'成功' if ok else '失败'}")
    if rknn_path.exists():
        print(f"  产物: {rknn_path.name}  {rknn_path.stat().st_size/1024:.1f} KB")

    bad = [ln for ln in log.splitlines()
           if "unsupport cpu" in ln.lower() or "fallback cpu failed" in ln.lower()]
    if bad:
        print("  ⚠️ 出现 CPU fallback 相关日志（硬性阻断信号）:")
        for ln in bad[:8]:
            print(f"     {ln.strip()}")
    else:
        print("  未出现 `unsupport cpu ...`（好迹象）")

    warn = [ln for ln in log.splitlines() if "warn" in ln.lower()][:5]
    for ln in warn:
        print(f"  warn: {ln.strip()}")
    return ok


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"rknn-toolkit2 冒烟测试  输出目录: {OUT_DIR}")

    # 让 `import rknn_onnx_compat` 在直接运行本文件时也能找到
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from rknn_onnx_compat import patch_onnx_mapping

    added = patch_onnx_mapping(verbose=True)
    print(f"onnx.mapping 垫片: {'已注入' if added else '无需'}")

    try:
        from rknn.api import RKNN  # noqa: F401

        print("rknn-toolkit2: 可导入")
    except Exception as e:  # noqa: BLE001
        print(f"rknn 导入失败: {e}")
        return 2

    import onnx
    import onnxruntime  # noqa: F401

    print(f"onnx {onnx.__version__}   numpy {np.__version__}")

    ok_all = True

    p1 = OUT_DIR / "mlp.onnx"
    make_mlp_onnx(p1)
    ok1, log1 = convert(p1, OUT_DIR / "mlp.rknn")
    ok_all &= report("基础算子 (MatMul + Add + Relu)", ok1, log1, OUT_DIR / "mlp.rknn")

    p2 = OUT_DIR / "layernorm.onnx"
    make_layernorm_onnx(p2)
    ok2, log2 = convert(p2, OUT_DIR / "layernorm.rknn")
    report("LayerNormalization（transformer 关键算子）", ok2, log2, OUT_DIR / "layernorm.rknn")

    # RKNN 无板卡时不能跑推理；这里只确认 ONNX 本身可用作参考
    import onnxruntime as ort

    sess = ort.InferenceSession(str(p1), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(2).normal(0, 1, (1, 64)).astype(np.float32)
    y = sess.run(None, {"x": x})[0]
    print(f"\nORT 参考输出: shape={y.shape} finite={bool(np.isfinite(y).all())}")

    print("\n=== 结论 ===")
    if ok_all:
        print("  工具链可用：基础算子转换成功。")
    else:
        print("  基础算子转换失败 —— 先解决这个再谈 ACT 模型。")
    print("  注意：无板卡时**无法**验证推理数值正确性，那一步必须在 RK3588 上做。")
    return 0 if ok_all else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
