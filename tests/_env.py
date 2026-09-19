"""测试共同环境：路径注入 + 缺失依赖的桩模块。

开发机没有 rknnlite / pyserial / opencv，用最小桩替代，
只为了让被测模块能 import。桩不参与被验证的逻辑（协议字节、队列数学、校验规则）。

跑法： python tests/run_all.py
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RK3588 = REPO / "rk3588"
CONFIGS = REPO / "configs"

for p in (str(RK3588), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 系统临时目录可能被沙箱挡住，统一用仓库内的临时目录
SCRATCH_ROOT = REPO.parent / ".testtmp"


class _Scratch:
    """极简临时目录上下文：只创建，不删除。

    不用 tempfile.TemporaryDirectory —— 它在退出时要 rmtree，
    在受沙箱限制的环境里会抛 WinError 5。测试不需要清理。
    """

    def __init__(self, p: Path):
        self.path = p

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *_exc) -> bool:
        return False


def tmpdir() -> _Scratch:
    import uuid

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    p = SCRATCH_ROOT / f"t{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return _Scratch(p)


def install_stubs() -> None:
    import logging

    # 测试输出保持干净：被测模块的 warning 会污染 PASS/FAIL 列表
    logging.disable(logging.CRITICAL)

    _stub_serial()
    _stub_cv2()
    _stub_rknnlite()


def _stub_serial() -> None:
    if "serial" in sys.modules:
        return
    m = types.ModuleType("serial")
    m.EIGHTBITS = 8
    m.PARITY_NONE = "N"
    m.STOPBITS_ONE = 1
    m.Serial = object  # 只用于类型标注
    sys.modules["serial"] = m


def _stub_cv2() -> None:
    """最小 cv2 桩：只需支持 resize / cvtColor，用于验证形状、通道顺序、值域。"""
    if "cv2" in sys.modules:
        return
    import numpy as np

    m = types.ModuleType("cv2")
    m.INTER_LINEAR = 1
    m.COLOR_BGR2RGB = 4

    def resize(img, size, interpolation=None):  # noqa: ARG001
        w, h = size
        ys = (np.arange(h) * img.shape[0] // h).clip(0, img.shape[0] - 1)
        xs = (np.arange(w) * img.shape[1] // w).clip(0, img.shape[1] - 1)
        return img[ys][:, xs]

    def cvtColor(img, code):  # noqa: ARG001
        if img.ndim == 3 and img.shape[2] == 3:
            return img[:, :, ::-1]
        return img

    m.resize = resize
    m.cvtColor = cvtColor
    sys.modules["cv2"] = m


def _stub_rknnlite() -> None:
    if "rknnlite" in sys.modules:
        return
    pkg = types.ModuleType("rknnlite")
    api = types.ModuleType("rknnlite.api")

    class RKNNLite:  # noqa: D401
        def load_rknn(self, *_a, **_k):
            return 0

        def init_runtime(self, *_a, **_k):
            return 0

        def inference(self, inputs=None):  # noqa: ARG002
            return []

        def release(self):
            return 0

    api.RKNNLite = RKNNLite
    pkg.api = api
    sys.modules["rknnlite"] = pkg
    sys.modules["rknnlite.api"] = api
