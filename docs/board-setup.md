# 板端环境实录（RK3588，2026-09-19 全部实测）

本文件记录**在这块具体板子上验证过的事实**，不是通用建议。换板子请重新验证。

## 板子身份

| 项 | 值 |
|---|---|
| 主机名 | `elf2-desktop` |
| 用户 / 密码 | `elf` / `elf`（在 `sudo`、`dialout`、`video`、`plugdev` 组） |
| SSH | 无线 `10.1.27.9`（有线那个 `192.168.1.100` 不在当时网段） |
| 系统 | Ubuntu 22.04.5 LTS，kernel **5.10.209** aarch64 |
| CPU / 内存 | 8 核 / **7.7 GiB**（注意：不是方案里写的 16GB） |
| 系统盘 | 56G，清理后 **37G 可用** |
| TF 卡 | **30G** 挂在 `/media/elf/ROOT`（建议放数据集与 `.rknn`） |

## 已验证的硬件能力

### NPU ✅

用板子自带的 `resnet18_for_rk3588.rknn` 实测：

```
librknnrt version: 2.3.2 (429f97ae6b@2025-04-09)   ← 与主机 rknn-toolkit2 2.3.2 完全匹配
RKNN Driver version: 0.9.8
toolkit version: 2.3.2 / target: rk3588 / static_shape
NPU load: Core0 15%, Core1 0%, Core2 0%            ← 三核在线
cur_freq: 1.0 GHz（最高档）
推理延迟: mean=8.28ms  p50=8.21ms  max=9.36ms
```

**注意 resnet18 正是 ACT 的视觉骨干** —— 8.28ms 这个数字给 IB_Robot 记录的 ACT 端到端 121ms 提供了很强的侧证。

复现脚本在 `pc/convert/toolchain_smoketest.py` 的思路可以直接搬到板端。

### D435i ✅ —— **修正一个之前的错误判断**

我曾在文档里写"D435i 在 RK3588 上 V4L2 直读不行、必须用 libuvc 绕内核"。
**在这块板子上这是错的**：

- 内核 **`CONFIG_USB_VIDEO_CLASS=y`（uvcvideo 是 builtin，不是模块）** → 插上即认
- 内核 **`CONFIG_USB_ACM=y`** → 串口也是 builtin
- `uvcvideo: Found UVC 1.50 device Intel(R) RealSense(TM) Depth Camera 435i (8086:0b3a)`
- `pyrealsense2 2.58.2` 直接枚举成功，无需编译 librealsense

实测取流（固件 5.15.1.55）：

| 流 | 结果 |
|---|---|
| 彩色 640×480@30 bgr8 | **30.0 fps** |
| 彩色 640×480@30 rgb8 | **30.0 fps** |
| 深度 640×480@30 z16 | **30.0 fps** |
| 彩色 + 深度 同时 | **30.0 fps** |

→ **不需要 libuvc 编译，不需要换 UVC 相机。**

**两个序列号不一样是正常的**（RealSense 有两套）：

- USB descriptor serial（dmesg）：`260843064898`
- ASIC serial（librealsense）：`254322076620`

绑定时要用 **ASIC serial**（`rs.config().enable_device(...)` 或 `device` 参数），别用 sysfs 的。

节点：`/dev/video21`~`/dev/video26`（6 个 UVC interface），另有 `/dev/media2`、`/dev/media3`。
注意 `/dev/video0`~`/dev/video20` 是 **RK 自己的 ISP / HDMI-RX 硬件节点**（`rkisp_mainpath`、`rk_hdmirx` 等），**不是摄像头**，别选错。

### 机械臂 ✅

```
usb 1-1.4: idVendor=1a86, idProduct=55d3  Product: USB Single Serial
           SerialNumber: 5B41532950
cdc_acm 1-1.4:1.0: ttyACM0: USB ACM device
```

- 设备：`/dev/ttyACM0`，**波特率 1,000,000**
- 稳定路径：`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532950-if00`
- 权限：`root:dialout`，用户 `elf` 已在 `dialout` 组 → **不需要 `chmod 666`**
- **PING 扫描结果：ID 1~6 全部应答**（前作已经设好 ID）

```python
# STS3215 PING（只读，不动舵机）
body = [mid, 0x02, 0x01]
pkt = bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
```

## Python 环境

板端统一用 **conda env `rkvla`**（`/home/elf/work/miniconda/envs/rkvla`）：

```
Python 3.10.20    torch 2.12.1+cpu   torchvision 0.27.1+cpu
lerobot 0.4.4     numpy 2.2.6        cv2 4.13.0
rknnlite          pyrealsense2 2.58.2
```

`lerobot` 是从 `/home/elf/work/lerobot`（源码克隆，tag **v0.4.4**）以 editable 方式装的。

构造 `SOFollower` 验证通过，`observation_features` 为：

```
['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos',
 'wrist_roll.pos','gripper.pos','front']
```

—— 与 `rk3588/main.py` 期望的键完全一致。

## 踩过的坑（会重复出现，记下来）

这台板子上的环境是前作遗留的**混合体**：`~/.local/lib/python3.10/site-packages`（用户级 site，815M）
会**泄漏进所有 conda 环境**，而且 `sys.path` 里它排在 conda 前面。这导致一串问题：

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| B1 | `import lerobot` 突然失败 | `__editable__.lerobot-0.4.4.pth` 指向 `/home/elf/work/RK3588-EIA/lerobot/src`，而该目录被清理掉了 | 从 `/home/elf/work/lerobot` 重装：`pip install --no-deps -e /home/elf/work/lerobot` |
| B2 | `ModuleNotFoundError: No module named 'lazy_loader'` | `librosa` 被 `transformers` 惰性探测到，但它自身依赖不全 | `pip install lazy_loader` |
| B3 | `ImportError: libscipy_openblas-9778f98e.so` | rkvla 里的 scipy 装得不完整 | `pip install --force-reinstall --no-cache-dir scipy` |
| B4 | `OSError: Could not load this library: libtorchaudio.so` | `~/.local` 里的 **torchaudio 2.5.0 要求 torch==2.5.0**，而环境是 torch 2.12.1 → 原生库加载失败；`transformers.is_torchaudio_available()` 只看包在不在，于是硬导 | **`pip uninstall -y torchaudio`**（ACT 根本不需要音频） |
| B5 | `ModuleNotFoundError: Could not import module 'AutoProcessor'` | 上面 B2/B3/B4 的连锁表现 | 同上 |

**根源教训**：`lerobot.processor.__init__` 会连带导入 `tokenizer_processor` → `transformers`，
而 `lerobot.robots.robot` 只是想要两个类型别名。**为一个类型别名拉进整个 transformers 重依赖链**，
是 LeRobot 的一个设计瑕疵，也意味着**任何 transformers 依赖链上的破损都会阻断机械臂控制**。

→ 板端排查机械臂导入问题时，**先查 `transformers` 能否干净导入**：

```bash
python -c "import transformers; print(transformers.__version__)"
```

## 清理记录

已删除（前作遗留、与 ACT 无关，且 GitHub 有备份）：

| 项 | 体积 |
|---|---|
| `~/.cache/pip` | 2.5G |
| conda env `asr` / `qwen` / `rknn` / `lerobot` | 354M / 279M / 233M / 2.9G |
| `Qwen-Chat-Assistant` + zip | 892M |
| `RK3588-EIA` | 2.2G |

**删除前已备份到 TF 卡** `/media/elf/ROOT/backup_RK3588-EIA/`（58M，13 个文件）：
`recordings/`（55M 遥操作录像）、`models/so101_urdf`、`teleop_record_*.json`
—— 这几样前作 GitHub 仓库里**没有**，其余目录（camera/ config/ docs/ scripts/ vla/ voice_assistant/）仓库里都有。

保留：`rknn-toolkit2-2.3.2`（2.1G，含 librknnrt 与 examples）、`inteld435i`（librealsense 源码）、
`lerobot`（源码克隆）、`Seeed_RoboController`、`rkrobot`（本项目）。

清理后：系统盘 37G 可用，`/home/elf/work` 从 14G 降到 **7.3G**。
