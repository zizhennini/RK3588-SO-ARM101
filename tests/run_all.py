"""跑所有测试： python tests/run_all.py

不需要 pytest。开发机不需要板卡 / 串口 / RKNN 运行时 ——
缺失的第三方依赖由 tests/_env.py 用最小桩替代。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _env  # noqa: E402

_env.install_stubs()

import test_action_queue  # noqa: E402
import test_config_and_guards  # noqa: E402
import test_feetech_protocol  # noqa: E402

MODULES = [test_action_queue, test_feetech_protocol, test_config_and_guards]


def main() -> int:
    print("=" * 60)
    total_failed = 0
    for m in MODULES:
        print(f"\n[{m.__name__}]")
        total_failed += m.main()
    print("\n" + "=" * 60)
    if total_failed:
        print(f"失败 {total_failed} 项")
    else:
        print("全部通过")
    print("=" * 60)
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
