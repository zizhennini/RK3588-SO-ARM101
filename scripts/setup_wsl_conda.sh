#!/usr/bin/env bash
# 在 WSL2 (Ubuntu 22.04) 中用 Miniconda 搭建 rkrobot 的两个环境：
#   env `lerobot` —— 采集 / 训练 / ONNX 导出   (Python 3.10 + torch CUDA)
#   env `rknn`    —— ONNX -> RKNN 转换          (torch<=2.4.0, numpy<=1.26.4，必须与主环境隔离)
#
# 幂等：可重复执行，已存在的会跳过。
# 不中断：任一步失败只记录，继续后续步骤，最后统一汇总。
#
# 调用方式（只传一个脚本路径参数，避免 PowerShell->wsl.exe 吃引号）：
#     wsl.exe -e env SUDO_PW=xxx bash /mnt/d/.../setup_wsl_conda.sh
# sudo 免密时可省略 SUDO_PW。

set -uo pipefail

WORK="${WORK:-$HOME/work}"
CODE="$WORK/rkrobot"
MINI="$HOME/miniconda3"
CONDA="$MINI/bin/conda"
LOG="$WORK/setup.log"
WIN_SRC="/mnt/d/Project/RK3588/RKROBOT/rkrobot"

mkdir -p "$WORK"
# 所有输出进日志（后台任务里 stdout 可能抓不到，统一看日志）
: > "$LOG"
exec >>"$LOG" 2>&1

RESULTS=()
STEP_NO=0
step() { STEP_NO=$((STEP_NO+1)); echo; echo "##### [$STEP_NO] $*  [$(date '+%T')]"; }
ok()   { echo "  [OK]   $*"; RESULTS+=("OK   $*"); }
skip() { echo "  [SKIP] $*"; RESULTS+=("SKIP $*"); }
fail() { echo "  [FAIL] $*"; RESULTS+=("FAIL $*"); }

SUDO() {
  if sudo -n true 2>/dev/null; then sudo "$@"
  elif [ -n "${SUDO_PW:-}" ]; then printf '%s\n' "$SUDO_PW" | sudo -S -p '' "$@"
  else echo "  (无 sudo 且未提供 SUDO_PW)"; return 1; fi
}

# 静默执行；失败时把真实输出打出来（不要再吞错误）
run_show() {
  local desc="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    return 0
  else
    echo "  ---- 「$desc」失败，真实输出 ----"
    printf '%s\n' "$out" | tail -30
    echo "  ---------------------------------"
    return 1
  fi
}

# conda 26.x 使用默认源需要接受 ToS；这里既接受，也强制只用 conda-forge
conda_prepare() {
  "$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
  "$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1
  "$CONDA" config --set channel_priority flexible >/dev/null 2>&1
}

conda_make_env() {
  local name="$1"
  run_show "conda create $name" \
    "$CONDA" create -y -n "$name" --override-channels -c conda-forge python=3.10
}

echo "=========== rkrobot WSL2 + Miniconda 环境搭建  $(date '+%F %T') ==========="
echo "WORK=$WORK"
echo "kernel : $(uname -r)"
echo "python3: $(python3 -V 2>&1)"
echo "nproc  : $(nproc)"

# ---------------------------------------------------------- 1. GPU 直通
step "GPU 直通检查"
if nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null; then
  ok "WSL2 GPU 直通可用"
else
  fail "nvidia-smi 不可用 —— 训练会退化成 CPU"
fi

# ---------------------------------------------------------- 2. 系统依赖
step "系统依赖 (apt)"
if ! SUDO -v >/dev/null 2>&1; then
  fail "sudo 不可用，跳过 apt"
else
  SUDO apt-get update -qq -o Acquire::Retries=3 >/dev/null 2>&1
  # 注意：不再需要 python3-venv，因为改用 conda
  PKGS="ffmpeg libgl1 libglib2.0-0 build-essential curl git-lfs"
  MISSING=""
  for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
  if [ -n "$MISSING" ]; then
    echo "  安装:$MISSING"
    if SUDO apt-get install -y -qq -o Acquire::Retries=3 $MISSING >/dev/null 2>&1; then
      ok "apt 依赖已安装:$MISSING"
    else
      fail "apt 安装失败:$MISSING"
    fi
  else
    skip "apt 依赖已齐全"
  fi
fi

# ---------------------------------------------------------- 3. Miniconda
step "Miniconda"
if [ -x "$CONDA" ]; then
  skip "已安装: $($CONDA --version 2>&1)"
else
  echo "  下载 Miniconda 安装包……"
  if curl -fsSL -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; then
    if bash /tmp/miniconda.sh -b -p "$MINI" >/dev/null 2>&1; then
      ok "Miniconda 已安装到 $MINI ($($CONDA --version 2>&1))"
    else
      fail "Miniconda 安装失败"
    fi
    rm -f /tmp/miniconda.sh
  else
    fail "Miniconda 下载失败（检查网络）"
  fi
fi

if [ -x "$CONDA" ]; then
  conda_prepare
  ok "conda 已配置（ToS 已接受；强制 conda-forge）"
  # 写进 .bashrc，方便交互式使用
  if ! grep -q 'conda initialize' "$HOME/.bashrc" 2>/dev/null; then
    "$CONDA" init bash >/dev/null 2>&1 && ok "conda init bash 完成（新终端可用 conda activate）"
  else
    skip "conda init 已配置"
  fi
fi

# ---------------------------------------------------------- 4. 同步代码
step "同步代码到 $CODE"
if [ -d "$CODE/.git" ]; then
  git -C "$CODE" pull --ff-only -q 2>/dev/null && ok "已更新" || skip "已存在，无新提交"
elif [ -d "$WIN_SRC/.git" ]; then
  # Windows 侧仓库作为 origin；后续可换成 GitHub 远程
  git clone -q "$WIN_SRC" "$CODE" && ok "已克隆自 $WIN_SRC" || fail "克隆失败"
else
  fail "找不到源仓库 $WIN_SRC"
fi
[ -f "$CODE/tests/run_all.py" ] && ok "代码就位" || fail "代码不完整"

# ---------------------------------------------------------- 5. lerobot 环境
step "conda env: lerobot（采集 / 训练 / ONNX 导出）"
if [ -x "$CONDA" ]; then
  if "$CONDA" env list | grep -qE '^lerobot\s'; then
    skip "env lerobot 已存在"
  else
    echo "  创建 env（conda-forge, python=3.10）……"
    conda_make_env lerobot && ok "env lerobot 已创建" || fail "env lerobot 创建失败"
  fi

  LRPY="$MINI/envs/lerobot/bin/python"
  if [ -x "$LRPY" ]; then
    "$LRPY" -m pip install -q -U pip wheel setuptools >/dev/null 2>&1
    echo "  安装 lerobot[feetech]（含 torch CUDA，约 2-3GB，耐心等）……"
    if run_show "pip install lerobot[feetech]" "$LRPY" -m pip install "lerobot[feetech]"; then
      ok "lerobot 已安装"
    else
      fail "lerobot 安装失败"
    fi

    echo "  验证:"
    "$LRPY" - <<'PY' || true
import importlib
try:
    import torch
    print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "  <-- 警告: 会用 CPU"))
except Exception as e:
    print(f"    torch 导入失败: {e}")
for m in ("lerobot", "serial", "yaml", "numpy", "cv2", "torchvision"):
    try:
        mod = importlib.import_module(m)
        print(f"    {m} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"    {m} 缺失: {type(e).__name__}")
PY

    for cli in lerobot-train lerobot-record lerobot-calibrate lerobot-find-port lerobot-setup-motors lerobot-teleoperate; do
      [ -x "$MINI/envs/lerobot/bin/$cli" ] && ok "CLI $cli" || fail "CLI 缺失 $cli"
    done
  else
    fail "lerobot env 的 python 不存在"
  fi
else
  fail "conda 不可用，跳过 lerobot 环境"
fi

# ---------------------------------------------------------- 6. rknn 环境
step "conda env: rknn（ONNX -> RKNN 转换，必须与 lerobot 隔离）"
if [ -x "$CONDA" ]; then
  if "$CONDA" env list | grep -qE '^rknn\s'; then
    skip "env rknn 已存在"
  else
    echo "  创建 env（conda-forge, python=3.10）……"
    conda_make_env rknn && ok "env rknn 已创建" || fail "env rknn 创建失败"
  fi

  RKPY="$MINI/envs/rknn/bin/python"
  if [ -x "$RKPY" ]; then
    "$RKPY" -m pip install -q -U pip wheel setuptools >/dev/null 2>&1
    echo "  安装 rknn-toolkit2==2.3.2 与 onnx 工具链……"
    # rknn-toolkit2 要求 torch<=2.4.0 / numpy<=1.26.4
    if run_show "pip install rknn 工具链" "$RKPY" -m pip install \
          "rknn-toolkit2==2.3.2" onnx onnxruntime onnxsim onnx-graphsurgeon \
          "numpy<=1.26.4" "torch<=2.4.0"; then
      ok "rknn 工具链已安装"
    else
      fail "rknn 工具链安装失败"
    fi
    "$RKPY" - <<'PY' || true
import importlib
for m in ("rknn", "onnx", "onnxruntime", "numpy", "torch"):
    try:
        mod = importlib.import_module(m)
        print(f"    {m} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"    {m} 缺失: {type(e).__name__}")
PY
  else
    fail "rknn env 的 python 不存在"
  fi
else
  fail "conda 不可用，跳过 rknn 环境"
fi

# ---------------------------------------------------------- 7. 单测
step "运行 rkrobot 单元测试"
LRPY="$MINI/envs/lerobot/bin/python"
if [ -x "$LRPY" ] && [ -f "$CODE/tests/run_all.py" ]; then
  if (cd "$CODE" && "$LRPY" tests/run_all.py); then ok "单测全部通过"; else fail "单测有失败项"; fi
else
  fail "无法运行单测"
fi

# ---------------------------------------------------------- 汇总
echo
echo "=================== 汇总 ==================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
NF=$(printf '%s\n' "${RESULTS[@]}" | grep -c '^FAIL' || true)
echo "==========================================="
echo "失败项: $NF"
echo
echo "用法:"
echo "  conda activate lerobot     # 采集/训练/导出"
echo "  conda activate rknn        # RKNN 转换"
echo "代码:     $CODE"
echo "日志:     $LOG"
echo
[ "$NF" -eq 0 ] && echo "结果: 全部成功" || echo "结果: 有 $NF 项失败（见上方 [FAIL]）"
echo "=========== 结束 $(date '+%F %T') ==========="
exit 0
