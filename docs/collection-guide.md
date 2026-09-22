# 数据采集指南（路线 A：主臂遥操作）

板端环境：`conda activate rkvla`（或直接用 `/home/elf/work/miniconda/envs/rkvla/bin/python`）
项目路径：`/home/elf/work/rkrobot`
硬件：从臂 `/dev/ttyACM0`、主臂 `/dev/ttyACM1`、D435i `/dev/video21`
（都用 by-id 更稳，序列号见 §1）

---

## 0 · 版本锁定（重要决策）

**本项目锁定 LeRobot `v0.4.4`，不升级到 v0.6.x。**

| 版本 | 发布 | requires_python |
|---|---|---|
| **0.4.4** | 2026-02-27 | **>=3.10** ← 我们在用 |
| 0.5.0 | 2026-03-09 | >=3.12 |
| 0.6.1（最新） | 2026-08-03 | >=3.12 |

理由：
1. **Python 硬约束** —— 板子是 3.10.20，WSL 也是 3.10，0.5+ 要 3.12，两个环境都装不了。
2. **0.4.4 的环境、API、硬件已全部实测通过**（NPU 8.28ms、D435i 满帧、机械臂 ID 1~6 应答、
   `SOFollower` 构造成功）。
3. **前作资产是 0.4.4 时代** —— 校准文件、遥操作录制、`/home/elf/work/lerobot` 源码。
4. **0.6.1 的实际增益对我们几乎为零**：`intelrealsense` extra（我们已能用）、
   新策略（不用）、`lerobot-eval`（不需要）、数据集 v3（0.4.4 已支持）；
   而代价是板端重装 torch（板载镜像 **140 KB/s**）+ 重写 `main.py`
   （0.6.1 的 feetech 依赖从 `scservo_sdk` 换成 `feetech-servo-sdk`）。

**换了板子或决定升级时，必须重新验证本文档里的每一步。**

---

## 1 · 接线

| 设备 | 接口 | 预期节点 | by-id 序列号 |
|---|---|---|---|
| 从臂（follower） | USB 2.0 | `/dev/ttyACM0` | `..._5B41532950-if00` |
| 主臂（leader） | USB 2.0 | `/dev/ttyACM1` | `..._5AAF262805-if00` |
| D435i | **USB 3.0** | `/dev/video21`~`/dev/video26` | ASIC `254322076620` |

> ⚠️ D435i 必须插 USB3.0 口（蓝色），否则帧率砍半。
> ⚠️ 两个臂**必须分别确认端口**，不要靠猜——用 `ls /dev/serial/by-id/` 看序列号。
> ⚠️ **ttyACM0/1 会随插拔顺序变化**，序列号不会。要稳就用 by-id 全路径。

```bash
ls -la /dev/serial/by-id/
```

主臂和从臂的序列号不同，从这里能明确区分哪个是哪个。

---

## 2 · 校准（必须重做，且 id 必须唯一）

### 命令

**不再用 `lerobot-calibrate`，用仓库自带的 `scripts/calibrate_so101.py`。**

原因见 `board-setup.md` B6：官方 CLI 在这块板子上会因为
`~/.local` 的 transformers 5.12.1 而整条挂掉。那个问题已经修好了，
但我们的工具仍然更合适——**它把校准文件写在仓库内的固定路径**，
不会被 HF 缓存里的 `None.json` / `my_awesome_*.json` 静默劫持。

```bash
source /home/elf/work/miniconda/bin/activate rkvla
cd /home/elf/work/rkrobot

# 0) 先看串口，确认 by-id 对得上（防止 ttyACM0/1 换序）
python scripts/calibrate_so101.py ports

# 1) 只读自检：六个舵机是否全部应答（不写任何数据）
python scripts/calibrate_so101.py check --role follower --port /dev/ttyACM0
python scripts/calibrate_so101.py check --role leader   --port /dev/ttyACM1

# 2) 正式校准（交互式，按提示摆臂）
python scripts/calibrate_so101.py calibrate --role follower --port /dev/ttyACM0
python scripts/calibrate_so101.py calibrate --role leader   --port /dev/ttyACM1

# 3) 确认结果
python scripts/calibrate_so101.py show --role follower
python scripts/calibrate_so101.py show --role leader
```

习惯用 by-id 全路径也可以（更稳，不受插拔顺序影响）：

```bash
F=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532950-if00   # follower
L=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AAF262805-if00   # leader
python scripts/calibrate_so101.py check --role follower --port $F
python scripts/calibrate_so101.py check --role leader   --port $L
```

> ⚠️ **扭矩警告（重要）**
> `check` 是**纯只读**的，跑完保持扭矩状态不变——内部用的是
> `disconnect(disable_torque=False)`。这一点**必须显式写**：因为
> `MotorsBus.disconnect()` 的默认参数是 **`disable_torque=True`**，
> 不覆盖的话从臂会被当场松掉扭矩、直接瘫下来。
> 而 `calibrate` 会**主动 `disable_torque()`**：机械臂将失去支撑，
> **请先用手扶住或垫好**，别让它砸下来。

校准过程中（三个 `input()` 提示）：

1. 若已有同名校准文件，会问「回车沿用 / 输入 `c` 重标」→ 首次都该输入 **`c`**
   （或直接加 `--force` 跳过这个询问）
2. `把机械臂摆到各关节行程的中间位置` → **这是 homing offset 的来源，摆不准后面全偏**
3. `把除 wrist_roll 外的每个关节手动转完整个行程` → 慢慢转到底再转到底

> `wrist_roll` 会被**跳过**（LeRobot 硬编码 0–4095）。
> 主臂校准前记得**扭矩是断开的**（`configure()` 会 `disable_torque()`），可以直接手拖。

**校准文件落点**（`--id` 默认 `so101_follower` / `so101_leader`）：

```
/home/elf/work/rkrobot/configs/calibration/so101_follower.json
/home/elf/work/rkrobot/configs/calibration/so101_leader.json
```

**这不是 LeRobot 的默认位置**（默认在
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`）。
好处是跟随仓库、可版本化；代价是后面用 `lerobot-teleoperate` / `lerobot-record`
时**必须显式传** `--robot.calibration_dir` 和 `--teleop.calibration_dir`，
否则它找不到文件、会**重新触发一次校准**（下面 §3/§4 的命令里已经带了）。

### ⚠️ `wrist_roll` 的零点很特殊（必须知道）

5 个身体关节和 `wrist_roll` 的处理**根本不同**。`so_follower.py` / `so_leader.py`
里都是这么写的：

```python
full_turn_motor = "wrist_roll"
unknown_range_motors = [m for m in self.bus.motors if m != full_turn_motor]
range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
range_mins[full_turn_motor] = 0        # ← 硬编码，不是扫出来的
range_maxes[full_turn_motor] = 4095
```

而归一化公式是（`motors_bus.py:_normalize`）：

```
norm = (clamp(val, min, max) - min) / (max - min) * 200 - 100
val  = 物理编码器读数 + homing_offset
```

- **5 个身体关节**：`range_min/max` 是在设完 homing **之后**扫出来的，
  代入上式后 `homing_offset` **完全约掉** → 映射只由「扫过的物理行程」决定。
  **「按 ENTER 时摆的中间位姿」其实不影响结果**，别在那一动作上纠结。
- **`wrist_roll`**：分母是常数 `4095`，`homing_offset` **约不掉**
  → **按 ENTER 那一刻的物理转角就是它的零点**。

所以 `wrist_roll` 是唯一一个「两个臂必须转到**同一个物理角度**再按 ENTER」的关节。
官方 issue [#3193](https://github.com/huggingface/lerobot/issues/3193) 讲的就是这件事，
维护者原话：

> The position of the wrist motor you start the calibration with will be the "0",
> so make sure that they are quite well aligned.

两个臂的 `wrist_roll` 零点没对齐 → 遥操作时手腕有**恒定角度错位**，
典型表现是「转到一半突然跳一下」或者顶到机械限位。

**重标定时的正确做法**：先把两个臂摆成**同一个姿态**
（`wrist_roll` 的夹爪朝向也要一致），再分别跑 `calibrate --force`。

### 旧校准文件的处置

板上前作留下了**两份从臂校准文件，数值不一样**：

| 关节 | `my_awesome_follower_arm.json` | `None.json` |
|---|---|---|
| shoulder_pan | 1876 | 1888 |
| shoulder_lift | -1078 | -1099 |
| elbow_flex | 1889 | 1854 |
| wrist_flex | 1265 | **1205** |
| wrist_roll | -1514 | **-1458** |
| gripper | 1781 | 1779 |

用哪一份由 `--robot.id` 决定（不传 id 就用 `None.json`）。
采数据用一份、部署用另一份 → **所有动作整体偏，且不报任何错**。

因为我们把校准写进**仓库内的独立目录**，这些文件**根本不会被读到**，
歧义自动消失。不用特意去删它们——确认一下它们不在 `configs/calibration/` 里就行：

```bash
find /home/elf/work/rkrobot/configs/calibration -name '*.json'
```

应该只有你刚生成的 `so101_follower.json` 和 `so101_leader.json`。

---

## 3 · 遥操作验证（先不录数据）

```bash
source /home/elf/work/miniconda/bin/activate rkvla
CAL=/home/elf/work/rkrobot/configs/calibration

lerobot-teleoperate \
  --robot.type=so101_follower  --robot.port=/dev/ttyACM0  --robot.id=so101_follower \
  --robot.calibration_dir=$CAL \
  --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader \
  --teleop.calibration_dir=$CAL \
  --fps=60
```

**验收标准**：动主臂，从臂应实时跟随，无明显延迟、不抖动、方向正确。
**这一步不过就别录数据。**

要点：
- `--display_data` 默认就是 `False`，**SSH 无显示器环境下保持默认**
  （设 `true` 会尝试开 rerun 窗口，无 X11 会失败）
- 如果这时它**要求重新校准**，说明 `--robot.calibration_dir` / `--teleop.calibration_dir`
  没生效（路径写错，或 §2 的校准文件不存在）——**别顺手就重标**，先查路径
- `--robot.id` / `--teleop.id` 必须和校准时的 id 一致，否则又回到"加载错文件"的老坑

### 觉得姿态有偏差时：用 `compare` 量，不要靠眼睛

「主臂比从臂高了一点」这类观感**无法自查**，直接量：

```bash
python scripts/calibrate_so101.py compare \
  --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1
```

它会：

1. 先打印两个臂的**校准行程对比**（`span` 差得多 = 扫过的物理范围不一致，
   这是身体关节偏移的**唯一**来源）
2. 把两个臂**都松开扭矩**，让你把它们摆成同一个物理姿态，回车
3. 逐关节打印 `follower` / `leader` / **差值**，并给出结论

判定标准：身体关节偏差 **≤3° 算一致**，>10° 建议重标定。
`gripper` 是 0–100 量程，不参与角度判定。

---

## 4 · 试录 1 集（冒烟测试）

```bash
source /home/elf/work/miniconda/bin/activate rkvla
CAL=/home/elf/work/rkrobot/configs/calibration

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101_follower \
  --robot.calibration_dir=$CAL \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video21, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so101_leader \
  --teleop.calibration_dir=$CAL \
  --dataset.repo_id=local/so101-pen-place \
  --dataset.root=/media/elf/ROOT/datasets/so101-pen-place \
  --dataset.num_episodes=1 \
  --dataset.single_task="把笔放到右边" \
  --dataset.fps=30
```

**关键点**：
- `--dataset.root` 指向 **TF 卡**，别占系统盘（系统盘 37G）
- `--dataset.repo_id` 用 `local/...` 前缀，避免尝试推到 HuggingFace Hub
  （板上 `huggingface.co` 不可达；真要推就设 `HF_ENDPOINT=https://hf-mirror.com`）
- `--dataset.single_task` 文本**一经确定就一字不差**，训练和部署都要用同一个
- 相机 `index_or_path` 用 `/dev/video21`；**不要用 `.0`~`.20`**（那些是 RK 的 ISP/HDMI-RX 节点）

**冒烟测试要通过这几项检查**（详见 `scripts/check_dataset.py`）：
- 数据集目录结构正确（`data/`、`videos/`、`meta/`）
- 图像不是全黑、不是静止重复帧
- 关节值有变化（不是全程不动）
- episode 时长合理（<30 秒）
- `meta/tasks.parquet` 里有任务文本

---

## 5 · 批量采集

冒烟测试通过后，把 `--dataset.num_episodes` 改大，分批采（50 → 评估 → 补 50）。

**采集纪律**：
- 单条 episode **< 30 秒**
- 按 4 步示教法：粗调定位 → 低速微调 → 缓慢下探 → 果断执行
- **目标位置必须随机化**（否则策略只会背位置）
- 筛除碰撞、不平滑的轨迹
- 中途相机不能动（动了要重新标定视野，且与训练不一致）

**目标：50 集起步，100 集更好。** 官方明确"25 集不够，表现很差"。

---

## 6 · 采集前自检清单

每次开新会话采数据前，跑一遍 `scripts/check_dataset.py` 配套的预检
（或手工核对）：

- [ ] 两个臂都在 `/dev/serial/by-id/` 里能对上号
- [ ] 校准文件**只有一个** id，且与命令里的 id 一致
- [ ] `lerobot-teleoperate` 跟随正常
- [ ] D435i 在 `/dev/video21`，30fps 出图
- [ ] TF 卡挂载在 `/media/elf/ROOT` 且有空间
- [ ] `--dataset.single_task` 文本与最终训练/部署用的一致
- [ ] 采集前后**没人碰相机**

---

## 7 · 采完之后

1. 数据集拉到 PC（用于训练）
2. PC 上 `lerobot-train --policy.type=act`（RTX 4060，BS4 仅需 0.94GB 显存）
3. `export_act_onnx.py` → `convert_to_rknn.py`（含输入顺序暴力确认）
4. `export_norm_stats.py` 导出归一化统计量
5. 拷到板子 → `main.py --once` → `--dry-run` → 真机
