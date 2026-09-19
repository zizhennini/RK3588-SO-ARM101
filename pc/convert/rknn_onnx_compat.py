"""onnx.mapping 兼容垫片 —— 必须在 import rknn 之前调用。

背景
----
`rknn-toolkit2==2.3.2` 内部使用 `onnx.mapping.TENSOR_TYPE_TO_NP_TYPE`
（见 `rknn/api/base_utils.py` 的 `to_np_type`）。
但 `onnx.mapping` 自 onnx 1.13 起废弃，**在 onnx>=1.17 已被移除**。

所以在新版 onnx 下转换会直接崩：

    AttributeError: module 'onnx' has no attribute 'mapping'

历史上这个坑被反复记录（IB_Robot 的 `convert_to_rknn.py` 就带了这个补丁，
SKILL.md 里也专门列了 "`onnx.mapping` AttributeError" 这一条故障）。

两种解法
--------
a) 降级 onnx 到 <=1.16.2（最简单，但可能牵连其他包）
b) **本垫片**：把缺失的 `onnx.mapping` 造回来，逻辑等同 onnx<=1.16 的实现

本仓库用 (b)，这样 onnx 可以保持新版。

用法
----
    from rknn_onnx_compat import patch_onnx_mapping
    patch_onnx_mapping()          # 必须在 `from rknn.api import RKNN` 之前
    from rknn.api import RKNN
"""

from __future__ import annotations

import sys
import types

_PATCHED = False


def _build_mapping_module():
    import numpy as np
    from onnx import TensorProto

    # 与 onnx<=1.16 的 onnx/mapping.py 保持一致
    tensor_type_to_np_type = {
        TensorProto.FLOAT: np.dtype("float32"),
        TensorProto.UINT8: np.dtype("uint8"),
        TensorProto.INT8: np.dtype("int8"),
        TensorProto.UINT16: np.dtype("uint16"),
        TensorProto.INT16: np.dtype("int16"),
        TensorProto.INT32: np.dtype("int32"),
        TensorProto.INT64: np.dtype("int64"),
        TensorProto.STRING: np.dtype("object"),
        TensorProto.BOOL: np.dtype("bool"),
        TensorProto.FLOAT16: np.dtype("float16"),
        TensorProto.DOUBLE: np.dtype("float64"),
        TensorProto.UINT32: np.dtype("uint32"),
        TensorProto.UINT64: np.dtype("uint64"),
        TensorProto.COMPLEX64: np.dtype("complex64"),
        TensorProto.COMPLEX128: np.dtype("complex128"),
    }

    m = types.ModuleType("onnx.mapping")
    m.TENSOR_TYPE_TO_NP_TYPE = tensor_type_to_np_type
    m.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in tensor_type_to_np_type.items()}
    m.__doc__ = "由 rkrobot 垫片提供的 onnx.mapping（onnx>=1.17 已移除）"
    return m


def patch_onnx_mapping(verbose: bool = False) -> bool:
    """若 `onnx.mapping` 缺失则补上。返回是否实际打了补丁。"""
    global _PATCHED
    if _PATCHED:
        return False

    import onnx

    if hasattr(onnx, "mapping"):
        _PATCHED = True
        if verbose:
            print("onnx.mapping 已存在，无需垫片")
        return False

    m = _build_mapping_module()
    # 同时满足 `import onnx.mapping` 与 `from onnx import mapping`
    sys.modules["onnx.mapping"] = m
    onnx.mapping = m  # type: ignore[attr-defined]
    _PATCHED = True

    if verbose:
        print(f"已注入 onnx.mapping 垫片（onnx {onnx.__version__} 已移除该模块）")
    return True


if __name__ == "__main__":
    import onnx

    added = patch_onnx_mapping(verbose=True)
    print("patch applied:", added)
    print("onnx:", onnx.__version__)
    print("TENSOR_TYPE_TO_NP_TYPE 条目数:", len(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE))
