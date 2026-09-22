# rkrobot — RK3588 + SO-ARM101 端侧具身智能操作终端（ACT 路线）

> **为什么是 ACT**：同样在 RK3588 NPU 上，ACT 单次前向 **≈121 ms**（float16，114 MB，实测），
> SmolVLA 三模块 RKNN **≈5049 ms**（实测）——相差 **42 倍**。ACT 一次推理产出 100 步动作块，
> 20–30 Hz 下只占 2–4% 占空比；SmolVLA 的推理比动作块可消费时长还长，队列必然饥饿。
> 依据与出处见 `docs/act-pipeline.md`。

## 设计原则：能复用就不自己写

本仓库**刻意写得很薄**。凡是成熟开源项目已经做好的，直接复用：

| 环节 | 复用什么 | 而不是自己写 |
|---|---|---|
| 机械臂驱动（Feetech STS3215 协议、标定、读写） | **LeRobot** `robots/so_follower/` + `motors/feetech/`（2000+ 行，久经考验） | ~~自己实现串口协议~~ |
| 相机取流 | **LeRobot** `cameras/opencv/`（还有 `realsense/` 后端） | ~~自己包 V4L2~~ |
| ACT → ONNX 导出 | **IB_Robot** `export_onnx_rknn.py` 的做法（wrapper + 输出裁剪 + onnxsim） | ~~自己猜 torch.onnx 参数~~ |
| RKNN 推理 | **IB_Robot** `RKNNSession` 的做法（data_format 来自 manifest、core_mask） | — |
| 动作队列 | **D-Robotics/rdk_LeRobot_tools** 的 `deque` + `popleft`，只补饥饿计数/hold/插值 | ~~190 行状态机~~ |
| 归一化参数 | **rdk** 存 `.npy`（`front_mean.npy` 等）的做法 | ~~自己拟合仿射~~ |

**关键前提**：LeRobot 的 `motors/` `robots/` `cameras/` **完全不依赖 torch**（已实测确认可 import），
所以板端可以 `pip install --no-deps lerobot` + 少量轻量依赖，不必装 ~8GB 的 torch。
见 `rk3588/requirements.txt`。

我们**只保留**这些参考项目没有的东西：

- **输入顺序的暴力确认**：RKNN 会重排输入顺序，顺序错且两路图像形状相同时**不会报错**，
  只会让动作全错。`convert_to_rknn.py` 枚举所有排列并与 ONNX Runtime 逐值比对。
  没有 manifest 就拒绝运行。
- **`onnx.mapping` 垫片**：onnx≥1.17 移除了它，rknn-toolkit2 2.3.2 依赖它。
- **融合规则自动回退**：`convert_layernorm_to_exnorm` 在标准 LayerNorm 上会崩（ACT 必然命中）。
- **真机安全网**：输出 finite/全零检查、队列饥饿计数、限速。

## 架构

```
机器人/相机（LeRobot 驱动）
   ↓  robot.get_observation()   ->  {"<motor>.pos": [-100,100], "<cam>": RGB HWC}
   ↓  归一化（MEAN_STD，numpy，见 stats.py）
   ↓  ACT on RKNN  —— 单次前向 ≈121 ms，输出 [1,100,6]
   ↓  反归一化 -> 动作块 -> ActionQueue（deque）
   ↓  robot.send_action({...})  ->  Feetech 串口 -> SO-ARM101
```

队列空才推理（inline），照 rdk 的做法——ACT 121 ms 相对 3.3 秒的动作块可以忽略，
不需要多线程。主循环在 `rk3588/main.py`。

## 目录

```
rkrobot/
├── docs/act-pipeline.md        决策依据、踩坑清单、工具链实测发现、验收清单
├── configs/robot.yaml          机械臂 / 相机 / 策略 / 调度 参数
├── pc/convert/                 在 PC 上跑（lerobot 环境 + rknn 环境）
│   ├── export_act_onnx.py      ACT -> ONNX（含与 PyTorch 的逐值校验）
│   ├── export_norm_stats.py    导出归一化统计量 .npy
│   ├── convert_to_rknn.py      ONNX -> RKNN + 输入顺序暴力确认 -> manifest
│   ├── rknn_onnx_compat.py     onnx.mapping 垫片
│   ├── make_dummy_act_policy.py 用真实 ACT 架构造随机策略（链路测试）
│   ├── make_mock_act_onnx.py    造假 ONNX（只测转换脚本）
│   └── toolchain_smoketest.py  只测 rknn-toolkit2 本身
├── rk3588/                     在板端跑
│   ├── stats.py                归一化统计量（numpy，不需要 torch）
│   ├── act_rknn.py             RKNN 推理 + 静默失效防护
│   ├── action_queue.py         薄队列：插值 / hold / 饥饿计数
│   ├── main.py                 闭环主程序（LeRobot 驱动 + inline 推理）
│   └── requirements.txt        --no-deps lerobot + 轻量依赖
├── scripts/setup_wsl_conda.sh  WSL2 环境一键搭建
└── tests/run_all.py            25 项单测，不需要板卡
```

## 全流程

### 0 · 环境

```bash
# WSL2（采集 / 训练 / 导出）
bash scripts/setup_wsl_conda.sh      # 建 lerobot 与 rknn 两个 conda env
conda activate lerobot               # 采集 / 训练 / ONNX 导出
conda activate rknn                  # ONNX -> RKNN 转换

# 板端（只推理，不装 torch）
pip install --no-deps lerobot
pip install -r rk3588/requirements.txt
# rknnlite 用板厂提供的 wheel，版本需与主机 rknn-toolkit2==2.3.2 匹配
```

### 1 · 采集（PC）

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=my_leader \
  --dataset.repo_id=你的用户名/so101-pen-place \
  --dataset.num_episodes=50 \
  --dataset.single_task="把笔放到右边" \
  --dataset.fps=30
```

> **供电按套件标签确认**（来源互相矛盾，照抄文档可能烧舵机）。
> `lerobot-setup-motors` 是**倒序执行**（先 gripper ID 6）。
> 相机槽位名 `front` 会决定数据集键 `observation.images.front`，**训练与部署必须一致**。

### 2 · 训练（PC / 云 GPU）

```bash
lerobot-train --policy.type=act \
  --dataset.repo_id=你的用户名/so101-pen-place \
  --output_dir=outputs/act_pen_place --policy.device=cuda \
  --batch_size=8 --steps=100000
```

ACT 很小：BS4 仅需 **0.94 GB** 显存，本机 RTX 4060 就够，**不需要云 GPU**。

### 3 · 导出与转换（PC）

```bash
conda activate lerobot
python pc/convert/export_act_onnx.py \
    --policy_path outputs/act_pen_place/checkpoints/last/pretrained_model \
    --output models/act_pen_place.onnx            # 内部会与 PyTorch 逐值比对
python pc/convert/export_norm_stats.py \
    --dataset 你的用户名/so101-pen-place --out models/norm_stats

conda activate rknn
python pc/convert/convert_to_rknn.py \
    --onnx models/act_pen_place.onnx --out models/act_pen_place.rknn
```

### 4 · 板端（按顺序放行）

```bash
python rk3588/main.py --once      # 单次推理：打印延迟/形状/数值范围，不碰舵机
python rk3588/main.py --dry-run   # 闭环但不写舵机：验证延迟、队列、饥饿率
python rk3588/main.py             # 真机（5 秒倒计时）
```

## 已验证 / 未验证

**已通过单元测试（25/25）**——不需要板卡、串口、RKNN 运行时（缺失依赖用桩替代）：

```bash
python tests/run_all.py
```

**工具链已端到端跑通**（WSL2 的 `rknn` 环境，用真实 ACT 架构的随机策略）：
ONNX → RKNN（含 LayerNorm 融合规则自动回退）→ 输入顺序暴力确认 → manifest。

**尚未验证**：真实训练产物（无数据）；真实 NPU 上跑 ACT 模型（无微调产物）。
**已实测通过**：板子 NPU（resnet18 8.28ms）、D435i 取流（640×480@30 满帧）、
机械臂串口（ID 1~6 全部应答）、`SOFollower` 构造。
详见 [`docs/board-setup.md`](docs/board-setup.md)。

## 工具链实测发现（都是真跑出来的）

| # | 现象 | 对策 |
|---|---|---|
| T1 | `module 'onnx' has no attribute 'mapping'` | onnx≥1.17 移除、rknn-toolkit2 依赖 → 垫片 `rknn_onnx_compat.py` |
| T2 | `No module named 'pkg_resources'` | setuptools≥81 移除 → rknn 环境钉 `setuptools<81` |
| T3 | 含 LayerNorm 的图 build 崩（`KeyError: 'LayerNormalization'`） | 融合规则有缺陷，**ACT 必然命中** → `disable_rules=[...]` 自动回退 |
| T4 | 回退不触发 | **`rknn.build()` 出错是抛异常，不是返回非零** |
| T5 | `shape (1,3,480,640) is wrong, expect 'nhwc'...` | **`rknn.inference` 的 data_format 默认是 nhwc** → 显式传 `'nchw'` |
| T6 | `runtime has not been initialized` | 转换顺序漏了 `init_runtime` |
| T7 | 图像归一化 | ACT 的 **VISUAL 也是 MEAN_STD**，只做 `/255` 是错的 → 用数据集 per-camera 统计量 |
| T8 | 通道顺序 | LeRobot 相机默认输出 **RGB**，板端不能再做 BGR→RGB（会反相） |
| T9 | torchvision 下载权重卡死 | 数据传完但连接不收尾，`.partial` 永不转正 → 用 curl 预置权重 |
| T10 | `huggingface.co` 不可达 | `hf-mirror.com` 可用 → 设 `HF_ENDPOINT` |
| T11 | `data_format='nchw'` 的准确语义 | 板端实测：RKNN 会**自动转换并给警告**（`need NHWC ... will be changed to NHWC`），结果正确但多一次内部转换 → 更优是板端直接产出 NHWC |
| T12 | 板端 `import lerobot.robots.so_follower` 失败 | `lerobot.processor` 为**一个类型别名**拉进 `transformers` 整条重依赖链，链上任何破损都会断掉机械臂控制（本次元凶是 `~/.local` 里 torch 版本不匹配的 torchaudio）→ `pip uninstall -y torchaudio`（同一个依赖链后来**又以 T14 的形式复发**，真正的总根因是 `~/.local` 越权，见 T14） |
| T13 | **D435i 在 RK3588 上开箱可用** | **修正此前的错误判断**：该板 `CONFIG_USB_VIDEO_CLASS=y`（uvcvideo builtin），`pyrealsense2 2.58.2` 直接枚举，彩色/深度/双流全 640×480@30 满帧，**不需要 libuvc 编译** |
| T14 | **三个 lerobot CLI 全部启动即崩**：`TypeError: non-default argument 'backbone_cfg' follows default argument` | `~/.local/lib/python3.10/site-packages` 在 `sys.path` 里**排在 conda 环境前面**，且其中的 `transformers 5.12.1` **超出** lerobot 0.4.4 的 `transformers<5.0.0` 约束；transformers 5.x 把 `PretrainedConfig` 变成 **kw_only dataclass**，踩中 Python 3.10 的 dataclass 规则。因为 env 里根本没有 transformers，**装包永远修不好**（装的进 env，生效的是 `~/.local`）→ 装合规版本进 env + 永久 `PYTHONNOUSERSITE=1`，详见 `docs/board-setup.md` **B6** |

板端环境的完整实录见 [`docs/board-setup.md`](docs/board-setup.md)：
板子身份、NPU/相机/机械臂实测数据、清理记录、5 个踩坑对策。
