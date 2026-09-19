#!/usr/bin/env bash
# RK3588 板端环境安装
#
# 前提：Ubuntu 22.04 arm64，Python 3.10
# 注意：不要在板端装 torch —— ACT 路线不需要。
#      反归一化只是一次仿射变换，numpy 读 configs/denorm.json 即可。

set -euo pipefail

echo "==> 1/4 系统依赖"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential \
    libopenblas-dev libjpeg-dev libgl1 \
    ffmpeg v4l-utils

echo "==> 2/4 Python 虚拟环境"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r rk3588/requirements.txt

echo "==> 3/4 rknnlite"
# ⚠️ 不要直接 pip install rknn_toolkit_lite2 —— 版本必须与主机 rknn-toolkit2 匹配（2.3.2）。
#    用板厂提供的 wheel，或从 rknn-toolkit2 仓库的 rknpu2/runtime 目录取。
if python -c "from rknnlite.api import RKNNLite" 2>/dev/null; then
    echo "    rknnlite 已可用"
else
    echo "    ⚠️ rknnlite 不可用。请手动安装与主机 rknn-toolkit2==2.3.2 匹配的版本："
    echo "      pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-linux_aarch64.whl"
    echo "    并确认板端 librknnrt.so 版本：strings \$(find / -name 'librknnrt.so' 2>/dev/null | head -1) | grep -i version"
fi

echo "==> 4/4 权限与设备检查"
if [ -e /dev/ttyACM0 ]; then
    sudo chmod 666 /dev/ttyACM* || true
    echo "    /dev/ttyACM0 存在，已放宽权限"
    echo "    （更稳妥的做法是把当前用户加入 dialout 组：sudo usermod -aG dialout \$USER 然后重新登录）"
else
    echo "    ⚠️ 未找到 /dev/ttyACM0。检查："
    echo "      1) 机械臂 USB 是否插好，dmesg | tail 看是否枚举"
    echo "      2) 内核是否启用 CONFIG_USB_ACM"
fi

echo
echo "完成。下一步："
echo "  source .venv/bin/activate"
echo "  python rk3588/main.py --once       # 单次推理，不碰舵机"
echo "  python rk3588/main.py --dry-run    # 闭环但不写舵机"
echo "  python rk3588/main.py              # 真机（会先 5 秒倒计时）"
