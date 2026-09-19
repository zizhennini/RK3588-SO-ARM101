# rkrobot — RK3588 + SO-ARM101 端侧具身智能操作终端（ACT 路线）

> **为什么是 ACT，不是 SmolVLA**：同样在 RK3588 NPU 上，ACT 单次前向 **≈121 ms**（float16，114 MB，实测），
> SmolVLA 三模块 RKNN **≈5049 ms**（实测）——**相差 42 倍**。ACT 一次推理产出 100 步动作块，
> 20–30 Hz 控制下只占约 2–4% 占空比；SmolVLA 的推理比动作块可消费时长还长，队列必然饥饿。
> 详见 `docs/act-pipeline.md`。

## 架构

```
语音/文字指令
   ↓  ① 语言层（可选，复用前作 RKLLM Qwen3.5-0.8B，约 1 s/次）
   ↓     输出：技能 + 目标（"把笔放到右边" → pick(pen) → place(right)）
   ↓  ② 策略层：ACT on RKNN，单次前向 ≈121 ms，输出 [1, 100, 6] 动作块
   ↓  ③ 调度层：动作块队列 + 线性插值 + hold 兜底 + 关节限位钳制
   ↓  ④ 执行层：Feetech STS3215 串口 → SO-ARM101
```

ACT 本身**不吃语言条件**，所以第 ① 层是外挂的：先用固定 task 跑通，再接语言选技能。
也可以完全跳过 ①，直接跑固定任务——**"能实现"优先**。

## 目录

```
rkrobot/
├── README.md                       ← 本文件
├── docs/
│   └── act-pipeline.md             关键约束、踩坑清单、验收标准
├── configs/
│   ├── robot.yaml                  机械臂 / 相机 / 策略 / 调度 参数
│   └── feetech_sts3215.yaml        STS3215 寄存器表（摘自 LeRobot，权威）
├── pc/                             在 PC（Windows/Linux）上跑
│   ├── collect/                    数据采集（遥操作录制）
│   ├── train/                      ACT 微调
│   └── convert/
│       ├── export_denorm_params.py 生成"模型输出 → 舵机原始计数"仿射参数
│       └── convert_to_rknn.py      ONNX → RKNN（fp16）+ 输入顺序自动确认
├── rk3588/                         在 RK3588 上跑
│   ├── act_rknn.py                 RKNN 推理封装（含输入顺序严格校验 + 静默失效防护）
│   ├── action_queue.py             动作块队列 / 插值 / 跨块平滑 / hold / 饥饿统计
│   ├── feetech_bus.py              极简 STS3215 串口总线（仅 pyserial）
│   └── main.py                     闭环主程序（多线程 + 相机取流）
└── scripts/
    └── setup_rk3588.sh             板端环境安装
```

## 三种运行模式（必须按顺序放行）

```bash
python rk3588/main.py --once       # ① 单次推理：打印延迟/形状/数值范围，不碰舵机
python rk3588/main.py --dry-run    # ② 闭环但**不写舵机**：验证延迟、队列、饥饿率
python rk3588/main.py              # ③ 真机（5 秒倒计时）
```

**不要跳过 ① 和 ②。** 同类项目里"模型加载成功 + 输出 shape 正确"而机器人不动或乱动是常态。

## 已验证 / 未验证（重要）

**已通过单元测试（33/33）**——不需要板卡、串口、RKNN 运行时，缺失依赖用最小桩替代：

```bash
python tests/run_all.py
```

| 测试集 | 覆盖内容 |
|---|---|
| `test_action_queue.py` (11) | 线性消费、上/降采样插值、跨块平滑、饥饿 hold 与计数、陈旧块丢弃、容量溢出、非法块拒绝 |
| `test_feetech_protocol.py` (11) | 校验和（手算期望值）、READ/WRITE/SYNC_WRITE 字节级包结构、符号-幅值编解码、SYNC_READ 解析、坏校验和拒绝、寄存器地址回归保护（Goal_Position 必须是 42） |
| `test_config_and_guards.py` (11) | 配置自洽性（关节/ID/限位/维度）、manifest 缺失拒绝运行、输出 NaN/全零/形状错必须报错、图像预处理形状/通道顺序/padding 锚点 |

**尚未验证**：`feetech_bus.py` 的真实串口收发、`act_rknn.py` 的真实 NPU 推理、
`convert_to_rknn.py` 的真实 RKNN 转换、`export_denorm_params.py` 的 LeRobot API 对接点
（该文件里有两处标了 `⚠️ 适配点`，需要按你的 LeRobot 版本补齐）。
开发机没有板卡、没有串口设备、也无法访问 WSL2（沙箱拒绝），这些都必须在真机上验收。

## WSL2 环境（已搭好，2026-09-19）

`bash scripts/setup_wsl_conda.sh` 在 WSL2 (Ubuntu 22.04) 里建两个 conda env：

| env | 内容 | 用途 |
|---|---|---|
| `lerobot` | Python 3.10、**torch 2.10.0+cu128**、lerobot 0.4.4、numpy 2.2.6、opencv 4.12 | 采集 / 训练 / ONNX 导出 |
| `rknn` | Python 3.10、rknn-toolkit2 2.3.2、onnx 1.22.0、onnxruntime 1.23.2、**numpy 1.26.4**、torch 2.4.0 | ONNX → RKNN 转换 |

**必须分成两个环境**：lerobot 要 `numpy>=2.0`，rknn-toolkit2 要 `numpy<=1.26.4`，装一起直接冲突。

已验证：

- **WSL2 GPU 直通正常**：`torch.cuda.is_available() == True`，设备 `NVIDIA GeForce RTX 4060 Laptop GPU`
- 6 个 CLI 全部就位：`lerobot-train` / `lerobot-record` / `lerobot-calibrate` / `lerobot-find-port` / `lerobot-setup-motors` / `lerobot-teleoperate`
- `tests/run_all.py` 在真实环境里 **33/33 通过**

```bash
conda activate lerobot     # 采集 / 训练 / 导出
conda activate rknn        # RKNN 转换
```

代码在 `~/work/rkrobot`（origin 指向 Windows 侧仓库，`git pull` 即可同步）。

### 装机时踩到的三个坑（脚本已处理）

1. **PyPI 官方源不可达**：`files.pythonhosted.org` 实测 15 秒下 0 字节；pip 进程 12 分钟只消耗 5 秒 CPU（纯阻塞在网络，不是卡在编译）。
   脚本自动把 `~/.pip/pip.conf` 指向清华镜像（实测约 2 MB/s，装完 torch 全套约 10 分钟）。
2. **conda 26.x 需要先接受 ToS**，否则 `conda create` 会在 4 秒内静默失败。脚本用 `conda tos accept` + `--override-channels -c conda-forge` 规避。
3. **WSL2 不继承 Windows 代理**（NAT 模式；宿主 `127.0.0.1:7897` 从 WSL 内不可达）。本机代理未启用，直连镜像即可，无需配置。

## 全流程

### 阶段 1 · 硬件与环境

```bash
# 1) 按套件标签确认供电电压！（来源矛盾，照抄文档可能烧舵机）
#    Seeed 标准套件 5V4A 双臂 / Waveshare 12V5A / TheRobotStudio leader 恒 7.4V
#    Seeed 明确警告 12V 会烧 7.4V 舵机

# 2) PC 侧环境（采集 + 训练）
pip install "lerobot[feetech,smolvla]"     # smolvla 仅为备用，本路线不用
pip install pyrealsense2

# 3) 找串口 + 设 ID（倒序执行：先 gripper ID 6，最后 shoulder_pan ID 1）
lerobot-find-port
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5

# 4) 整臂校准（leader / follower 必须分别校准，各自 --id）
lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower
```

板端：

```bash
bash scripts/setup_rk3588.sh
sudo chmod 666 /dev/ttyACM*      # 或把用户加入 dialout 组
```

### 阶段 2 · 采集数据

```bash
# 遥操作录制。单条 episode < 30 秒；目标位置必须随机化
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=COM5 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=COM6 \
  --teleop.id=my_leader \
  --dataset.repo_id=你的HF用户名/so101-pen-place \
  --dataset.num_episodes=50 \
  --dataset.single_task="把笔放到右边" \
  --dataset.fps=30
```

> **语言指令必须一字不差**，且 `tasks.parquet` 必须包含该文本。
> **50 集是起点不是达标线**：官方明确"25 集不够，表现很差"；社区 100 集才约 70%。
> 轨迹质量优先于数量——筛掉碰撞、不平滑的。

### 阶段 3 · 训练 ACT

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=你的HF用户名/so101-pen-place \
  --output_dir=outputs/act_pen_place \
  --job_name=act_pen_place \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=100000
```

ACT 极小：BS4 仅需 **0.94 GB** 显存，本机 RTX 4060（8 GB）就够，**不需要云 GPU**。
（这是相对 SmolVLA 的另一个优势——SmolVLA 的 BS8+AdamW 要 10–16 GB。）

### 阶段 4 · 导出 ONNX → RKNN

```bash
# ⚠️ 必须在独立虚拟环境里做：rknn-toolkit2 要求 torch<=2.4.0 / numpy<=1.26.4
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install rknn-toolkit2==2.3.2 onnx onnxruntime

# 4a) 生成反归一化仿射参数（模型输出 → 舵机原始计数）
python pc/convert/export_denorm_params.py \
    --policy outputs/act_pen_place/checkpoints/last/pretrained_model \
    --dataset 你的HF用户名/so101-pen-place \
    --out configs/denorm.json

# 4b) ONNX → RKNN + 自动确认输入顺序
python pc/convert/convert_to_rknn.py \
    --policy outputs/act_pen_place/checkpoints/last/pretrained_model \
    --out models/act_pen_place.rknn
```

**推荐的替代方案**：IB_Robot 已经写好了这套（`src/model_utils/model_utils/export_onnx_rknn.py`，
自动处理 `onnx.mapping` 兼容、输出裁剪、fp16/int8/hybrid），实测 121 ms。**建议直接拿它当参考实现**，
本仓库的 `convert_to_rknn.py` 补齐了它缺的一环：**输入顺序的自动暴力确认**。

### 阶段 5 · 板端闭环

```bash
# 先只做推理复现，不接机械臂
python rk3588/main.py --config configs/robot.yaml --dry-run

# 确认 121 ms 量级 + 输入顺序正确后，接真机
python rk3588/main.py --config configs/robot.yaml
```

## 验收标准

| 阶段 | 判据 |
|---|---|
| 采集 | ≥50 集，单条 <30 s，无碰撞轨迹，指令文本已冻结 |
| 训练 | loss 收敛无 NaN；PC 端回放能完成任务的**基线成功率**（只记录，不设下限） |
| 转换 | 输入顺序自动确认通过；RKNN 与 ONNX Runtime 输出 `max|Δ| < 1e-2`（fp16） |
| 板端推理 | 单次推理 **< 300 ms**（目标 ≈121 ms）；连续 200 次无异常、无 NaN、非全零 |
| 闭环 | 队列饥饿率 < 5%；端到端任务成功率（20 次）记录实测 |

## 已知风险

| 风险 | 缓解 |
|---|---|
| RKNN 重排输入顺序 → 动作全错 | `convert_to_rknn.py` **暴力枚举排列**并与 ORT 比对，自动确认；板端启动时再断言一次 |
| 动作输出静默全零 | 板端 fail-fast：全零 velocity 立即报错 |
| 反归一化参数错 → 动作幅度错 | `export_denorm_params.py` 用最小二乘拟合 + 残差断言，不手写公式 |
| D435i 在 RK3588 驱动 | 首选 libuvc 源码编译；退路是 Windows 侧采集。见 `docs/act-pipeline.md` |
| 舵机寄存器写错 | 寄存器表直接摘自 LeRobot `feetech/tables.py`；写操作只覆盖 Goal_Position / Torque_Enable |
| 串口权限 / 热插拔 | 启动时检查设备存在；写失败立即停扭矩并抛错 |

## 未实测声明

本仓库的板端代码（`rk3588/`）与转换脚本（`pc/convert/`）**尚未在真实 RK3588 + SO-ARM101 上跑过**
（开发机没有板卡、没有串口设备、也没有可用的 Python 环境）。它们是把已验证的知识固化成结构，
**首次上板请按 `--dry-run` → 单关节 → 整臂 的顺序逐步放行**，不要一上来就跑闭环。
