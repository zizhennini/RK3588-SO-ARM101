# 数据采集指南（路线 A：主臂遥操作）

板端环境：`conda activate rkvla`（或直接用 `/home/elf/work/miniconda/envs/rkvla/bin/python`）
项目路径：`/home/elf/work/rkrobot`
硬件：从臂 `/dev/ttyACM0`、主臂（待接）、D435i `/dev/video21`

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

| 设备 | 接口 | 预期节点 |
|---|---|---|
| 从臂（follower） | USB 2.0 | `/dev/ttyACM0` |
| 主臂（leader） | USB 2.0 | `/dev/ttyACM1`（第二个接入的） |
| D435i | **USB 3.0** | `/dev/video21`~`/dev/video26` |

> ⚠️ D435i 必须插 USB3.0 口（蓝色），否则帧率砍半。
> ⚠️ 两个臂**必须分别确认端口**，不要靠猜——用 `ls /dev/serial/by-id/` 看序列号。

```bash
ls -la /dev/serial/by-id/
```

主臂和从臂的序列号不同，从这里能明确区分哪个是哪个。

---

## 2 · 校准（必须重做，且 id 必须唯一）

### 为什么必须重做

板上前作留下了**两份从臂校准文件，数值不一样**：

| 关节 | `my_awesome_follower_arm.json` | `None.json` |
|---|---|---|
| shoulder_pan | 1876 | 1888 |
| shoulder_lift | -1078 | -1099 |
| elbow_flex | 1889 | 1854 |
| wrist_flex | 1265 | **1205** |
| wrist_roll | -1514 | **-1458** |
| gripper | 1781 | 1779 |

**用哪一份由 `--robot.id` 决定**（不传 id 就用 `None.json`）。
采数据用一份、部署用另一份 → **所有动作整体偏，且不报任何错**。

→ 所以：**重新校准，用全新的 id**，把旧的歧义彻底排除。

### 命令

```bash
source /home/elf/work/miniconda/bin/activate rkvla

# 从臂（id 用 rkrobot_follower）
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=rkrobot_follower

# 主臂（id 用 rkrobot_leader）
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=rkrobot_leader
```

> `so101_leader` 这个类型名请在 0.4.4 里核对一下（目录名是 `so_leader`，
> 注册名可能是 `so101_leader` / `so100_leader`）：
> ```bash
> python -c "from lerobot.teleoperators.utils import TeleopConfig; print(TeleopConfig.get_choice_class_names())"
> ```
> 或者直接看 `lerobot-record --help` 里 `--teleop.type` 的可选值。

校准过程中：
- 先**手动把两个臂摆到同一个中间姿态**（这一步决定了 homing offset，摆不准后面全偏）
- `wrist_roll` 关节会被**跳过**（LeRobot 硬编码 0–4095）
- 校准结果写到 `~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}/so*_*/<id>.json`

**校准完立即确认只有你新建的那份 id**：

```bash
find ~/.cache/huggingface/lerobot/calibration -name '*.json' -printf '  %p\n'
```

建议把旧的 `None.json` 和 `my_awesome_*.json` **移走备份**（不要留在原地），
免得后面忘记传 id 时静默加载了错的那份。

---

## 3 · 遥操作验证（先不录数据）

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=rkrobot_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=rkrobot_leader
```

**验收标准**：动主臂，从臂应实时跟随，无明显延迟、不抖动、方向正确。
**这一步不过就别录数据。**

---

## 4 · 试录 1 集（冒烟测试）

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=rkrobot_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video21, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=rkrobot_leader \
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
