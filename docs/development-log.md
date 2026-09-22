# 开发日志

格式见项目开发方案 §9.1。**任何延迟或成功率数字都必须同时记录配置基线**
（chunk_size / num_steps / control_freq / 数据集 fps / 相机数量 / 量化模式），
脱离配置的单点数字没有意义。

---

## [2026-09-19] [阶段〇·方案评审] 技术选型：ACT 而非 SmolVLA

**状态**：完成
**环境**：PC（桌面调研 + 一手资料核实）

**任务**：评审 v1.4 方案的可行性，确定端侧策略选型。

**产出**：
- 方案 v1.5（覆盖 v1.4，原件归档至 `archive/`）
- `docs/act-pipeline.md` 决策依据章节

**关键指标**（同一块 RK3588 NPU 上的实测对比）：

| | ACT | SmolVLA |
|---|---|---|
| RKNN 模块数 | **1** | 3（vision/prefill/action） |
| 模型大小 | **114 MB**（fp16） | 730 MB |
| **NPU 推理延迟** | **≈121 ms** | **≈5049 ms**（端到端 ≈6.4 s） |
| 采样方式 | **单次前向** | flow matching，10 步迭代去噪 |
| 输出 | `(1,100,6)` | `[1,50,32]` → `[50,6]` |
| 闭环可行性 | 100 步块 @20–30 Hz = 3.3–5 s，推理占 **2–4%** 占空比 → **可行** | 推理 4.2–7.8 s > 块可消费时长 2.5 s → **队列持续饥饿** |

**问题与解决**：
- SmolVLA 在 RK3588 上慢了 42 倍，根因**不是模型大，而是非单次前向**（flow matching 迭代采样）。
  RTC（Real-Time Chunking）是学术界的掩盖方案，但依赖 autograd，**静态 RKNN 图无法表达**，
  且可行性条件 `d ≤ s ≤ H−d` 要求折算延迟 ≤25 步，实测 `d ≈ 100–154`，**超出 4–6 倍**。
- 结论：换策略而非换硬件。

**下一步**：搭 WSL2 开发环境。

---

## [2026-09-19] [阶段一] WSL2 开发环境搭建（Miniconda 双环境）

**状态**：完成
**环境**：WSL2（Ubuntu 22.04）

**任务**：建采集/训练/转换所需的两个隔离环境。

**版本登记**：

| 环境 | 内容 |
|---|---|
| `lerobot` | Python 3.10、**torch 2.10.0+cu128**、lerobot 0.4.4、numpy 2.2.6、opencv 4.12 |
| `rknn` | Python 3.10、**rknn-toolkit2 2.3.2**、onnx 1.22.0、onnxruntime 1.23.2、**numpy 1.26.4**、torch 2.4.0、setuptools<81 |

**产出**：`scripts/setup_wsl_conda.sh`（幂等，失败不中断）
**关键指标**：`torch.cuda.is_available() == True`，设备 `NVIDIA GeForce RTX 4060 Laptop GPU`

**问题与解决**：
1. **PyPI 官方源不可达** —— `files.pythonhosted.org` 实测 15 秒下 0 字节；pip 进程跑 12 分钟只消耗 5 秒
   CPU（纯阻塞在网络，不是卡在编译）。→ 改用清华镜像（实测约 2 MB/s，装完 torch 全套约 10 分钟）。
2. **conda 26.x 需要先接受 ToS**，否则 `conda create` 在 4 秒内静默失败。
   → `conda tos accept` + `--override-channels -c conda-forge`。
3. **WSL2 不继承 Windows 代理**（NAT 模式，宿主 127.0.0.1:7897 从 WSL 内不可达）。
4. **torchvision 自己下载 resnet18 权重会卡死** —— 数据其实已传完（`.partial` 有 46MB），
   但连接停在 `FIN-WAIT-1` 不收尾，torchvision 永不把 `.partial` 转正；进程跑 29 分钟只耗 6 秒 CPU。
   → 用 curl 预置权重到 `~/.cache/torch/hub/checkpoints/`。
5. **`huggingface.co` 不可达**（http=000），`hf-mirror.com` 可用（http=200）→ 设 `HF_ENDPOINT`。

**下一步**：写代码骨架并验证 RKNN 工具链。

---

## [2026-09-19] [阶段三] RKNN 工具链端到端验证（用假 ACT）

**状态**：完成
**环境**：PC 转换（WSL2 的 `rknn` 环境）

**任务**：在采集任何数据之前，先证明工具链走得通。

**产出**：
- `pc/convert/export_act_onnx.py`（做法吸收自 IB_Robot 的 `export_onnx_rknn.py`）
- `pc/convert/convert_to_rknn.py`（含**输入顺序暴力确认**）
- `pc/convert/rknn_onnx_compat.py`（`onnx.mapping` 垫片）
- `pc/convert/make_dummy_act_policy.py`（用**真实 LeRobot ACT 架构**随机初始化）
- `pc/convert/toolchain_smoketest.py`、`make_mock_act_onnx.py`

**关键指标**：

```
假 ACT 策略：51.6 M 参数（真实 ACT ≈52 M），config.input_features =
             ['observation.images.front','observation.state']
导出 → 转换 → 输入顺序确认 → manifest
verified_max_abs_diff = 6.88e-4   （容差 1e-2）
```

**问题与解决**（6 项工具链缺陷，全部实测复现）：

| # | 现象 | 根因与对策 |
|---|---|---|
| T1 | `module 'onnx' has no attribute 'mapping'`，load_onnx 直接崩 | onnx≥1.17 移除了它，rknn-toolkit2 2.3.2 内部依赖 → 注入等价垫片 |
| T2 | `No module named 'pkg_resources'` | setuptools≥81 移除 → rknn 环境钉 `setuptools<81` |
| T3 | 含 LayerNorm 的图 build 崩（`KeyError: 'LayerNormalization'`） | 融合规则 `convert_layernorm_to_exnorm` 有缺陷，**ACT 必然命中** → `disable_rules=[...]` 自动回退（已验证有效） |
| T4 | 回退不触发 | **`rknn.build()` 出错是抛异常，不是返回非零** —— 只判断返回码会漏掉 |
| T5 | `shape (1,3,480,640) is wrong, expect 'nhwc'...` | **`rknn.inference` 的 `data_format` 默认是 nhwc** → 显式传 `'nchw'` |
| T6 | `runtime has not been initialized` | 漏了 `init_runtime` |

**下一步**：板端勘察。

---

## [2026-09-22] [阶段四] 板端环境勘察与硬件验证

**状态**：完成
**环境**：RK3588（`elf@10.1.27.9`，无线）

**任务**：摸清板子实际能力，验证 NPU / 相机 / 机械臂。

**版本登记**（板端）：

```
系统      Ubuntu 22.04.5 LTS，kernel 5.10.209 aarch64，8 核 / 7.7 GiB
librknnrt 2.3.2 (429f97ae6b@2025-04-09)   ← 与主机 rknn-toolkit2 2.3.2 完全匹配
NPU driver 0.9.8
rknnlite  已装（/usr/local/lib/python3.10/dist-packages）
rknn env  conda env `rkvla`：Python 3.10.20 + torch 2.12.1 + lerobot 0.4.4
          + rknnlite + pyrealsense2 2.58.2
```

**关键指标**（全部实测）：

| 项 | 结果 |
|---|---|
| **NPU**（板载 resnet18） | **mean 8.28 ms / p50 8.21 ms / max 9.36 ms**；三核在线；cur_freq 1.0 GHz |
| **D435i 彩色** 640×480@30 bgr8 | **30.0 fps** |
| **D435i 深度** 640×480@30 z16 | **30.0 fps** |
| **D435i 彩色+深度同时** | **30.0 fps** |
| **机械臂** PING 扫描 | **ID 1~6 全部应答**，`/dev/ttyACM0 @ 1,000,000` |
| 板端单测 | **25/25 通过** |

**注**：resnet18 **正是 ACT 的视觉骨干**，8.28 ms 这个数字给 IB_Robot 记录的 ACT 端到端 121 ms 提供了很强的侧证。

**问题与解决**：

1. **修正一处此前的错误判断** —— 我曾在文档里写"D435i 在 RK3588 上 V4L2 直读不行、
   必须用 libuvc 绕内核"。在这块板子上**不成立**：`CONFIG_USB_VIDEO_CLASS=y`（uvcvideo 是
   内核内置），`pyrealsense2 2.58.2` 直接枚举成功，彩色/深度/双流全满帧，**不需要编译任何驱动**。
   同样 `CONFIG_USB_ACM=y`（串口也是内置）。
2. **`/dev/video0`~`20` 是 RK 自己的 ISP/HDMI-RX 硬件节点**（`rkisp_mainpath`、`rk_hdmirx` 等），
   **不是摄像头**；D435i 在 `/dev/video21`~`26`。别选错。
3. **RealSense 有两个序列号**：USB descriptor serial（dmesg，`260843064898`）与 ASIC serial
   （librealsense，`254322076620`）。绑定时用 **ASIC serial**。
4. **T5 的准确语义在现场被澄清**：板端实测报
   `need NHWC data format, but NCHW set, the data format and data buffer will be changed to NHWC`
   —— 即 **RKNN 会自动转换并给警告**，结果正确但多一次内部转换。更优做法是板端直接产出 NHWC。
   这解释了 IB_Robot 的 MR #225 为什么要专门改成喂 NHWC。

**下一步**：修板端环境 + 清理前作遗留。

---

## [2026-09-22] [阶段四] 板端环境修复与前作遗留清理

**状态**：完成
**环境**：RK3588

**任务**：清理前作遗留，把板端环境修到能用。

**产出**：项目部署到 `/home/elf/work/rkrobot`（git 仓库，24 文件）

**关键指标**：
- 系统盘：29G → **37G 可用**；`/home/elf/work`：14G → **7.3G**
- 备份到 TF 卡：**58 MB / 13 个文件**（前作 GitHub 仓库里**没有**的部分）
- 保留的 conda 环境：`base` / `d435i` / `rkvla`

**问题与解决**（5 个连锁问题，B1~B5）：

| # | 现象 | 根因与对策 |
|---|---|---|
| B1 | `import lerobot` 突然失败 | `__editable__.lerobot-0.4.4.pth` 指向 `/home/elf/work/RK3588-EIA/lerobot/src`，而该目录被本次清理删掉了 → 从 `/home/elf/work/lerobot` 重装（`pip install --no-deps -e`） |
| B2 | `No module named 'lazy_loader'` | `librosa` 被 transformers 惰性探测到，自身依赖不全 → `pip install lazy_loader` |
| B3 | `ImportError: libscipy_openblas-*.so` | rkvla 里的 scipy 装得不完整 → `pip install --force-reinstall --no-cache-dir scipy` |
| B4 | `OSError: Could not load libtorchaudio.so` | `~/.local` 里的 **torchaudio 2.5.0 要求 torch==2.5.0**，环境是 2.12.1 → 原生库加载失败；`transformers.is_torchaudio_available()` 只看包在不在，于是硬导 → **`pip uninstall -y torchaudio`**（ACT 根本不需要音频） |
| B5 | `Could not import module 'AutoProcessor'` | B2/B3/B4 的连锁表现 |

**根因教训（值得记）**：
`lerobot.processor.__init__` 会连带导入 `tokenizer_processor` → `transformers`，
而 `lerobot.robots.robot` **只是想要两个类型别名**。为一个类型别名拉进整个 transformers
重依赖链，是 LeRobot 的设计瑕疵，也意味着**链上任何破损都会阻断机械臂控制**。
→ 板端排查机械臂导入问题时，**先查 `transformers` 能否干净导入**。

**修复后验证**：

```
✅ lerobot.robots.so_follower.so_follower 可导入
✅ SOFollower 构造成功
   observation_features: [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
                          wrist_flex.pos, wrist_roll.pos, gripper.pos, front]
   action_features     : [6 个 <joint>.pos]
```

与 `rk3588/main.py` 期望的键**完全一致**。

**清理明细**（前作遗留，与 ACT 无关，且 GitHub 有备份）：

| 删除 | 体积 |
|---|---|
| `~/.cache/pip` | 2.5G |
| conda env `lerobot` / `asr` / `qwen` / `rknn` | 2.9G / 354M / 279M / 233M |
| `Qwen-Chat-Assistant` + zip | 892M |
| `RK3588-EIA` | 2.2G |

**删除前已备份到 TF 卡** `/media/elf/ROOT/backup_RK3588-EIA/`：
`recordings/`（55M 遥操作录像）、`models/so101_urdf`、`teleop_record_*.json`
—— 这几样前作仓库里没有；其余目录（camera/ config/ docs/ scripts/ vla/ voice_assistant/）仓库里都有。

**产出沉淀**：`docs/board-setup.md`（板子实录）、README 新增 T11~T13
**提交**：`7193ed6`，已推送 GitHub（15 commits / 24 files）

**下一步**：采集数据。

---

## [2026-09-22] [阶段四·续] 板端 CLI 全线复活 + 自研校准工具

### 起点：三个 CLI 全部启动即崩

主臂接上、准备开始校准，`lerobot-calibrate` 失败。最初按"缺依赖"处理，
装了 `gymnasium` / `imageio` / `diffusers` 之后仍然失败，报错换成了
`huggingface-hub>=1.5.0,<2.0 is required ... but found 0.35.3`。

——装 `diffusers` 时把 `huggingface-hub` 从 1.20.1 降到 0.35.3，**把
`~/.local` 的 transformers 弄坏了**。这是一次典型的"打地鼠"，继续装包只会更糟。

**停下来做根因分析**，而不是接着装包。这是本次唯一的正确决定。

### 根因（两层，缺一不可）

**第一层：`~/.local` 在 `sys.path` 里排在 conda 环境前面**

```
1. .../rkvla/lib/python3.10.zip
2. .../rkvla/lib/python3.10
3. .../rkvla/lib/python3.10/lib-dynload
4. /home/elf/.local/lib/python3.10/site-packages   ← 用户级，抢先命中
5. .../rkvla/lib/python3.10/site-packages          ← conda 环境，被遮蔽
```

于是 `import transformers` **永远**拿到 `~/.local` 那份，而 env 里压根没有
transformers（`PYTHONNOUSERSITE=1` 时是 MISSING）。**装包永远修不好**：
装的都进 env，生效的始终是 `~/.local`。

**第二层：`~/.local` 的 transformers 5.12.1 超出 lerobot 0.4.4 的约束**

LeRobot 0.4.4 的 METADATA 白纸黑字：

```
Requires-Dist: transformers<5.0.0,>=4.57.1; extra == "transformers-dep"
```

而 transformers 5.x 把 `PretrainedConfig` 变成了 **kw_only dataclass**
（实测 `dataclasses.is_dataclass(PretrainedConfig) == True`）。
`lerobot/policies/groot/groot_n1.py:179` 是
`backbone_cfg: dict = field(init=False, ...)`，Python 3.10 的
`dataclasses._init_fn` 在 kw_only 分支上直接抛：

```
TypeError: non-default argument 'backbone_cfg' follows default argument
```

**报错链**（跟 SO-101 毫无关系，纯属 CLI 的 eager import 连坐）：

```
lerobot_{calibrate,record}.py 顶层 import 全部机器人类型
  → lerobot.robots.unitree_g1
  → lerobot.envs.factory
  → lerobot.policies.__init__        # 又是 eager import 全部策略
  → lerobot.policies.groot.groot_n1
  → TypeError
```

实测：`lerobot-calibrate` 和 `lerobot-record` 都是这个错，
`lerobot-teleoperate` 先卡在 `No module named 'rerun'`（同一个坑的前一站）。

### 修复

```bash
RK=/home/elf/work/miniconda/envs/rkvla
export PYTHONNOUSERSITE=1
$RK/bin/python -m pip install --no-cache-dir \
    "transformers>=4.57.1,<5.0.0" "huggingface-hub>=0.34.2,<0.36.0" \
    "tokenizers>=0.22.0,<=0.23.0" \
    "av>=15.0.0,<16.0.0" "datasets>=4.0.0,<5.0.0" \
    "draccus==0.10.0" "rerun-sdk>=0.24.0,<0.27.0" regex safetensors

mkdir -p $RK/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=1' > $RK/etc/conda/activate.d/zz_disable_usersite.sh
```

三个坑中坑：

1. **pip 会被 `~/.local` 骗** —— 不带 `PYTHONNOUSERSITE=1` 时 pip 认为
   `regex` / `tokenizers` "already satisfied"（其实是在 `~/.local` 里），不装进 env。
2. **`tokenizers` 不能装最新版** —— transformers 4.57.6 要求 `<=0.23.0`，
   直接 `pip install tokenizers` 拿到 0.23.2 → transformers 导入期就 `ImportError`。
   必须锁区间。
3. **`draccus` 的 `__version__` 是 `0.8.0` 但 dist 版本是 `0.10.0`**
   （包内属性忘了更新）—— 差点被这个假象带偏，要去查 `importlib.metadata.version`。

### 结果

| CLI | 修复前 | 修复后 |
|---|---|---|
| `lerobot-calibrate` | ❌ groot TypeError | ✅ rc=0 |
| `lerobot-teleoperate` | ❌ `No module named 'rerun'` | ✅ rc=0 |
| `lerobot-record` | ❌ groot TypeError | ✅ rc=0 |

**整套官方 CLI 复活 → 不需要自己重写 record/teleop**，原计划不用大改。
顺带确认了两个类型名：`--robot.type` 里有 `so101_follower`，
`--teleop.type` 里有 `so101_leader`（之前文档里还标着"待核对"）。

### 自研校准工具 `scripts/calibrate_so101.py`

排查期间（CLI 还没修好时）写了这个工具；修好后**仍然保留**，因为它更合适：

- 只 import `so_follower` / `so_leader` 两个模块，不碰 `unitree_g1`
  → 不会被别人的 eager import 连坐
- **校准文件写在仓库内** `configs/calibration/<id>.json`，跟随版本控制，
  彻底摆脱 HF 缓存里 `None.json` / `my_awesome_*.json` 的静默劫持
- 自带 `check` 子命令（只读，不写任何数据）先确认六个舵机应答

```
ports      列出 /dev/serial/by-id，并标注哪个是从臂哪个是主臂
check      只读自检（sync_read Present_Position，normalize=False）
calibrate  交互式校准（--force 可跳过"沿用旧文件"询问直接重标）
show       查看已保存的校准文件
```

写的时候踩到两个真 bug，都值得记：

1. **`Robot.__init__` 的 `calibration_dir` 语义**：
   `calibration_dir = config.calibration_dir or <HF缓存>/robots/<name>`。
   **显式传 `calibration_dir` 时不会再拼 `robots/so_follower/` 子目录**，
   文件就是 `<dir>/<id>.json`。第一版的路径推导因此是错的，
   已改为一律用实例的 `calibration_fpath` 权威路径。
2. **未校准时 `sync_read("Present_Position")` 会失败**：它的默认参数是
   `normalize=True`，会去查校准表做归一化，而那时校准表是空的。
   必须显式 `normalize=False`（顺便说，`range_min/max`、`homing_offset`
   本来就是**原始值**，读原始值才是对的）。

另外确认 `display_data` 在两个 CLI 里**默认都是 `False`**，
SSH 无显示器环境下保持默认即可（`rerun` 只是顶层 import，已装 0.26.2）。

### 硬件现状（两臂都已接上并确认）

| 设备 | 节点 | by-id 序列号 |
|---|---|---|
| 从臂 follower | `/dev/ttyACM0` | `usb-1a86_USB_Single_Serial_5B41532950-if00` |
| 主臂 leader | `/dev/ttyACM1` | `usb-1a86_USB_Single_Serial_5AAF262805-if00` |
| D435i | `/dev/video21` | ASIC 序列号 `254322076620` |

（`ttyACM0/1` 会随插拔顺序变化，序列号不会。）

**产出沉淀**：`docs/board-setup.md` 新增 **B6 详解**；
`docs/collection-guide.md` §2/§3/§4 改用新工具与新校准路径；
新增 `scripts/calibrate_so101.py`

**下一步**：`check` 自检两臂 → 正式校准 → `lerobot-teleoperate` 验证跟随 → 试录 1 集。

---

## 当前状态汇总（截至 2026-09-22）

### 已验证 ✅

| 项 | 证据 |
|---|---|
| 策略选型 | ACT 121 ms vs SmolVLA 5049 ms（差 42 倍） |
| WSL2 双环境 | cuda=True / 6 个 lerobot CLI 就位 |
| 代码单测 | **25/25 通过**（板端与 PC 端一致） |
| RKNN 工具链 | 假 ACT 端到端，`verified_max_abs_diff = 6.88e-4` |
| 板端 NPU | resnet18 **8.28 ms** |
| 板端 D435i | 彩色/深度/双流 **640×480@30 满帧** |
| 板端机械臂 | **两臂均已识别**：`ttyACM0`=follower / `ttyACM1`=leader，1 Mbaud，ID 1~6 全应答 |
| `SOFollower` | 构造成功，features 与 `main.py` 一致 |
| **板端 3 个 CLI** | **calibrate / teleoperate / record 全部 rc=0**（B6 修复后） |
| **校准工具** | `scripts/calibrate_so101.py` 四个子命令全部跑通（非硬件路径） |

### 未验证 ⬜

| 项 | 阻塞原因 |
|---|---|
| 板端两臂校准 | 未开始（命令已就绪，等你操作机械臂） |
| 遥操作跟随 | 待校准完成 |
| 真实 ACT 的导出→转换链路 | 无微调产物；且需 PC 侧 WSL 提权 |
| 板端真实 NPU 跑 ACT 模型 | 同上 |
| 真实数据采集与微调 | 待开始 |
| 端到端任务闭环 | 待开始 |

### 待处理的现实问题

1. **`~/.local` 仍是颗定时炸弹** —— 里面还留着越权包（如 torchaudio）。
   env 已用 `PYTHONNOUSERSITE=1` 隔离（conda 激活时自动生效），
   但**用裸 `python` 绕过 conda 激活时仍会中招**。排查任何导入怪问题时，
   第一件事是确认这个变量在不在。
2. `board` 是 WiFi 连接，SSH 会**间歇性超时/断连**（实测约 40% 的尝试会失败）
   —— 所有远程 scp/ssh 操作都要带重试。
3. 训练需要把数据集在板子/PC 之间搬运；`huggingface.co` 不可达，
   用 `HF_ENDPOINT=https://hf-mirror.com`。
4. 板子内存只有 **7.7 GiB**（方案写 16GB），别在板上跑训练。
5. 板端 `torchvision 0.27.1` / `diffusers 0.40.0` 等超出 lerobot 0.4.4 约束，
   属于**已知噪音**（板端只做 RKNN 推理，不碰这些路径），先不动。
