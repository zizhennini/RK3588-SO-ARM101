# ACT on RK3588 — 关键约束、踩坑与验收

## 1. 为什么选 ACT（决策依据）

同一块 RK3588 NPU 上的实测对比：

| | ACT | SmolVLA |
|---|---|---|
| RKNN 模块数 | **1** | 3（vision / prefill / action） |
| 模型大小 | **114 MB**（float16） | 730 MB |
| **NPU 推理延迟** | **≈121 ms** | **≈5049 ms**（首次；端到端约 6.4 s） |
| 采样方式 | **单次前向** | flow matching，10 步迭代去噪 |
| 输出 | `(1, 100, 6)` | `[1, 50, 32]` → 后处理 `[50, 6]` |
| 闭环可行性 | 100 步块 @20–30 Hz = 3.3–5 s，推理占 **2–4%** 占空比 → **可行** | 推理 4.2–7.8 s > 块可消费时长 2.5 s → 队列持续饥饿 |

- ACT 实测来源：`IB_Robot/docs/RKNN_NPU_inference_on_BQ3588HM.md`（Bearkey BQ3588HM，OpenHarmony，float16，
  `librknnrt` 2.4.1b0 / driver 0.9.5）。注意该文档在 IB_Robot 当前 master 上**已被移除**，
  只存在于历史提交 `a6ca9ae7`。
- SmolVLA 实测来源：`IB_Robot/docs/SmolVLA_RKNN_Troubleshooting_Record.md`，该文档自己写明
  "推理速度慢于动作消费速度……无法称为平滑闭环"，且"尚未解决"。

**核心结论**：问题不是"模型太大"，而是"**非单次前向**"。
ACT 是单次前向的轻量策略（约 50–80 M 参数），是目前 RK3588 上唯一"有实测延迟、有部署指南、有导出脚本"的策略。

### 代价：ACT 不吃语言

ACT 是纯模仿学习策略，输入是图像 + 关节状态，**没有语言条件**。要保留"自然语言指令"的对外形态，
需要在外面挂一层。三种做法，按工程量递增：

1. **固定任务**：一个模型一个任务，指令写死在配置里。最快能跑通。
2. **技能选择器**：小 LM/VLM 把指令映射到 `(技能, 目标)`，再选对应的 ACT 模型或参数。复用你前作的
   RKLLM（Qwen3.5-0.8B，实测 21.6 tok/s，约 1 s/次决策）。
3. **端到端 VLA**：回到 SmolVLA —— 但在 RK3588 NPU 上不可行（见上表）。

**建议先做 1，验证闭环后再做 2。**

---

## 2. 必须知道的坑

### 2.1 RKNN 会重排输入顺序（最容易静默失败的一条）

实测：ACT 的 ONNX 输入顺序是 `[cam_high, cam_left, state]`，**转换后的 RKNN 期望 `[state, cam_high, cam_left]`**。

典型报错：`input[0] need 2dims input, but 4dims`——因为 state 是 2 维、图像是 4 维，顺序错了维度就对不上。

**危险的是**：如果两个图像输入形状相同，顺序互换**不会报错**，只会让动作全错（静默失败）。

对策：`pc/convert/convert_to_rknn.py` 会**暴力枚举图像输入的所有排列**，逐个与 ONNX Runtime 的输出比对，
取误差最小的那个，并把确认结果写进 `models/*.rknn.manifest.json`。板端启动时再断言一次。

### 2.2 动作输出可能静默全零

历史故障：SmolVLA 的 action expert 含 INT64 `ReduceMin`，RK3588 上 CPU fallback 失败 → velocity 全零，
而"转换成功 + 加载成功"都发现不了。ACT 没这个算子，但**同类静默失效必须防**。

对策：板端对每次推理输出做 `np.isfinite()` + 非全零 + shape 检查；连续多步异常立即停扭矩并报错。

### 2.3 环境隔离（torch / numpy / onnx 版本三方冲突）

`rknn-toolkit2` 要求 `torch<=2.4.0` + `numpy<=1.26.4`，而 LeRobot 要 `torch>=2.7` + `numpy>=2.0`。
**必须用独立虚拟环境**，且不要在同一 shell 里先 source 主环境再调用 `.venv-rknn` 的解释器。

另外：新版 `onnx` 会触发 `rknn-toolkit2==2.3.2` 的 `onnx.mapping` AttributeError。IB_Robot 的
`convert_to_rknn.py` 带了这个补丁。

### 2.4 版本矩阵要登记，不要只记"RKNN 版本"

官方**没有** toolkit/runtime 兼容矩阵表。已知的失配症状是 `Invalid RKNN model version6`。
每次复现请分别记录：

```
主机 rknn-toolkit2 / 模型内 compiler / 板端 rknnlite package / 板端 librknnrt.so / NPU driver / 镜像
```

已验证可用的组合（IB_Robot）：toolkit 2.3.2 + rknnlite 2.3.2 + librknnrt 2.4.1b0 + driver 0.9.5。

### 2.5 不要做 ONNX 事后手术

`rknn-toolkit2` 不支持部分算子（如 `NonZero`）。**要在 exporter 层改 PyTorch 源码消掉它，不要事后改 ONNX**——
插入的显式 Constant 会被后续 `fold_constant` / `fuse_ops` 剥掉，补丁失效并重现崩溃。

判据：**转换日志里出现 `unsupport cpu <Op> op` 就是硬性阻断**，不要指望 CPU fallback 兜底。
也不要相信官方算子表——它是陈旧的，且"表里支持 ≠ 能跑"（LayerNorm 表内 Supported，其分解体
`exNorm:ReduceMean_0_2ln` 仍会 fallback 失败）。

### 2.6 相机：D435i 是负担，普通 UVC 相机更省事

**所有能找到的 RK3588/边缘 LeRobot 项目用的都是普通 USB UVC 摄像头**（`cv2.VideoCapture` + MJPG）。

D435i 在 RK3588 上的现实：
- RK3588 BSP 是 5.10.x 内核，librealsense 的内核补丁脚本只维护 Ubuntu LTS 5.4/5.8/5.11，直接报
  `Unsupported kernel version 5.10.160`
- Intel 官方要求 Rockchip 平台**从源码用 RSUSB / libuvc 后端编译**（`-DFORCE_LIBUVC=true`）绕过内核
- V4L2 直读即便可行也只有 RGB，无 depth/IMU，且 metadata 节点在 5.10 BSP 上不存在
- 固件版本是真实变量：有案例 FW 5.16.0.1 枚举失败，降到 5.13.0.55 后 RGB+depth+IMU 全正常

**先例**：IB_Robot 的 LeKiwi 抓取管线在用 D435i（`realsense2_camera 4.57.7`，**Color-first 启动顺序**，
640×360@30，`initial_reset=false`，`enable_sync=false`，关 pointcloud/IMU）。它明确记录：
"YAML 参数本身不能改变 sensor 启动顺序；系统仍加载原版 Depth-first wrapper 时不得继续抓取。"
→ 需要打过补丁的 RealSense wrapper。

**建议**：如果只是为了跑通闭环，**买一只 15 块钱的 UVC 相机**，能省掉整条驱动风险链。
D435i 留到需要深度的时候再上。

### 2.7 舵机侧

- 波特率 **1,000,000**，protocol 0，分辨率 4096，model number 777，model string `sts3215`
- `lerobot-setup-motors` 是**倒序执行**（先 gripper ID 6，最后 shoulder_pan ID 1）
- 重复 ID 会改整条总线——所有舵机建议拆下逐个重编
- leader / follower 传动比与行程不同，**必须分别校准、各自 `--id`**
- 换舵机后要删掉校准文件重来
- 供电来源互相矛盾（Seeed 5V4A 双臂 / Waveshare 12V5A / TheRobotStudio leader 恒 7.4V），
  **以套件标签为准**；Seeed 明确警告 12V 会烧 7.4V 舵机

---

## 3. 动作用户空间：反归一化怎么处理

ACT 的输出是**归一化动作**，要经过两步变换才能写进舵机：

```
模型输出 (归一化)
  → ×std + mean            (LeRobot 数据集统计量)
  → 标定变换 (homing_offset / drive_mode / 行程)
  → 舵机原始计数 (0–4095)
```

两步都是**逐关节仿射变换**，可以合并成 `raw = scale[j] * out[j] + offset[j]`。

**不要手写这两步的公式**——LeRobot 的内部实现随版本变化，抄错一处动作幅度就整体偏。
`pc/convert/export_denorm_params.py` 的做法是：

1. 采样若干组随机的归一化动作
2. 喂给**LeRobot 自己的** postprocessor + 机器人标定转换，得到对应的原始计数
3. 用最小二乘拟合出 `scale` / `offset`
4. **断言残差 ≈ 0**（残差不为 0 说明该关节不是仿射的，或采样区间不够）

这样得到的参数与 LeRobot 版本无关，且自带验证。

板端只做一次 `raw = scale * out + offset`，然后钳制到关节限位。

---

## 4. 调度：为什么必须有动作块队列

ACT 一次产出 100 步动作块。**绝不能"算一步、走一步"**。

```
控制线程 @ control_freq_hz
   ├─ 从 ActionQueue 取一步 → 钳制 → 写舵机
   ├─ 队列剩余 < 阈值  → 通知推理线程补块（异步，不阻塞控制）
   └─ 队列空          → hold 上一步（可选：报饥饿计数）

推理线程
   └─ 取最新帧 + 当前关节状态 → RKNN → push chunk（带时间戳）
```

要点：
- **推理异步**：`T_infer ≈ 121 ms` 远小于块时长（100/30 ≈ 3.3 s），可以提前算好下一块
- **插值**：数据集的 fps 与舵机更新率可能不同，队列出队时做线性插值
- **陈旧块丢弃**：时间戳过旧的块不进队列（`max_chunk_age_s`），否则机械臂会执行过期动作
- **hold 兜底**：队列空时保持最后位置，并计数饥饿次数（这是核心健康指标）
- **限位钳制**：任何情况下写出的原始计数都必须在 `[raw_min, raw_max]` 内

---

## 6. RKNN 工具链实测发现（2026-09-19，WSL2 真机验证）

以下 6 条都是**实际跑出来**的，不是推测。前 3 条是环境/工具链缺陷，后 3 条是我自己代码里的 bug
（其中 T5 如果没端到端跑一遍，会被原样带到板上）。

| # | 现象 | 根因 | 对策 | 状态 |
|---|---|---|---|---|
| **T1** | `AttributeError: module 'onnx' has no attribute 'mapping'`，`load_onnx` 直接崩 | onnx≥1.17 移除了 `onnx.mapping`，而 rknn-toolkit2 2.3.2 的 `base_utils.to_np_type` 依赖它 | `pc/convert/rknn_onnx_compat.py` 注入等价垫片（`patch_onnx_mapping()`，须在 `import rknn` 之前调用） | ✅ 已验证修复 |
| **T2** | `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools≥81 移除了 `pkg_resources`，rknn-toolkit2 内部 import 它 | rknn 环境钉 `setuptools<81`（setup 脚本已内置） | ✅ 已验证修复 |
| **T3** | 含 LayerNorm 的图 `build` 崩：`KeyError: 'LayerNormalization'`（栈顶 `rules/norm.py::_p_convert_layernorm_to_exnorm`） | rknn-toolkit2 2.3.2 的图融合规则有缺陷。**ACT 的 transformer 解码器含 LayerNorm，必然命中** | `rknn.config(disable_rules=['convert_layernorm_to_exnorm'])` | ✅ **已验证有效**（禁用后转换成功，28 KB 模型） |
| **T4** | 默认构建失败时脚本直接崩溃，回退逻辑不执行 | **`rknn.build()` 出错时是抛异常，不是返回非零**；只判断返回码会漏掉 | `_try_build()` 整段包 try/except，失败时 release context 并返回日志 | ✅ 已修 |
| **T5** | `The input(ndarray) shape (1,3,480,640) is wrong, expect 'nhwc' like (1,480,640,3)` | **`rknn.inference()` 的 `data_format` 默认是 `nhwc`**，而我们的 ONNX 是 NCHW | 显式传 `data_format='nchw'`（转换脚本与板端都已钉死）；manifest 记录 `image_layout`；新增 2 个回归测试 | ✅ 已修 + 测试覆盖 |
| **T6** | `inference: The runtime has not been initialized` | 转换顺序漏了一步 | 正确顺序：`config → load_onnx → build → export_rknn → **init_runtime** → inference` | ✅ 已修 |

**端到端验证结果**（假 ACT ONNX → RKNN）：

```
输入  observation.state [1,6] + observation.images.front [1,3,480,640]
输出  action [1,100,6]
→ 构建：默认失败 → 自动禁用 convert_layernorm_to_exnorm → 成功
→ 输入顺序：暴力枚举 + 与 ONNX Runtime 比对 → 确认 ["state","front"]
→ verified_max_abs_diff = 6.88e-4  （容差 1e-2）
→ manifest 已生成（含 image_layout / build_fallback_disabled_rules）
```

**可复现的命令**（在 WSL2 的 `rknn` 环境里）：

```bash
conda activate rknn
python pc/convert/make_mock_act_onnx.py --out /tmp/mock_act.onnx
python pc/convert/convert_to_rknn.py --onnx /tmp/mock_act.onnx --out /tmp/mock_act.rknn
python pc/convert/toolchain_smoketest.py      # 只测工具链本身
```

**仍未验证**：真实 NPU 上的推理数值（无板卡）、真实 ACT 模型（无微调产物）。
另外模拟器的算子重排行为**未必与真机一致**，板端最好再复核一次输入顺序。

---

## 7. 验收与排查清单

**每次上板，按顺序走：**

- [ ] `--dry-run`：只推理不写舵机，确认延迟 + 输出 shape + 非全零 + finite
- [ ] 单关节：只写一个关节，确认方向、范围、限位都对
- [ ] 整臂静态：读出真实关节状态喂给模型，确认输出动作方向合理（可以手动挪动机械臂看动作是否跟随）
- [ ] 整臂动态：接上队列，低速跑（把 `control_freq_hz` 减半）
- [ ] 连续 200 次推理：无异常、无 NaN、非全零、延迟分布稳定
- [ ] 端到端 20 次任务：记录成功率 + 队列饥饿次数

**不要只看"模型加载成功"和"输出 shape 正确"** —— 这两件事都成立而机器人不动/乱动的情况，
在同类项目里是常态。

**任何延迟或成功率数字，必须同时记录配置基线**：`chunk_size` / `n_steps` 执行步数 /
`control_freq_hz` / 数据集 fps / 相机数量 / 量化模式 / 图像分辨率。
脱离配置的单点延迟没有意义。
