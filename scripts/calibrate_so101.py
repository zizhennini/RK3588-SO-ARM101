#!/usr/bin/env python
"""SO-101 单臂校准 / 自检工具（不依赖 lerobot-calibrate CLI）。

为什么要自己写
--------------
官方 ``lerobot-calibrate`` 在模块顶层 eager import 了**全部**机器人类型
（``lerobot/scripts/lerobot_calibrate.py`` 第 36-60 行：bi_openarm、hope_jr、
lekiwi、omx、openarm、unitree_g1 …）。其中 unitree_g1 会拉进
``lerobot.envs.factory`` → ``lerobot.policies`` → ``groot``，最终在
Python 3.10 下必然抛::

    TypeError: non-default argument 'backbone_cfg' follows default argument

这是 LeRobot 的 ``GR00TN15Config(PretrainedConfig)`` 与 Python 3.10
dataclass 规则的不兼容，**与我们的 SO-101 毫无关系**，也永远不会因为我们
装包而修好。

我们真正需要的只有两个模块，直接实例化即可（已实测可导入）::

    lerobot.robots.so_follower.so_follower.SOFollower
    lerobot.teleoperators.so_leader.so_leader.SOLeader

用法
----
::

    # 列出串口（按 by-id，避免 ttyACM0/1 换序）
    python scripts/calibrate_so101.py ports

    # 自检：只读位置，不写任何数据
    python scripts/calibrate_so101.py check --role follower --port /dev/ttyACM0

    # 校准（交互式，会提示移动机械臂）
    python scripts/calibrate_so101.py calibrate --role follower --port /dev/ttyACM0
    python scripts/calibrate_so101.py calibrate --role leader   --port /dev/ttyACM1

    # 查看已保存的校准文件
    python scripts/calibrate_so101.py show --role follower

注意：``--id`` 不传时会用默认值。**不要留空**，否则 LeRobot 会把校准文件
存成 ``None.json``（我们之前清理时就见过这种残留文件）。

校准文件落在哪
--------------
默认写到仓库内的 ``configs/calibration/``，文件名就是 ``<id>.json``::

    configs/calibration/so101_follower.json
    configs/calibration/so101_leader.json

之所以不是 LeRobot 默认的 HF 缓存（``~/.cache/huggingface/lerobot/
calibration/robots/so_follower/<id>.json``）：放进仓库可以版本化、可复现，
也符合"文件都放在项目工作区"的约定。代价是以后用官方工具时要显式传
``--robot.calibration_dir=configs/calibration``（不传它会找不到、然后重新
触发一次校准）。用 ``show`` 子命令打印的是权威路径。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB_DIR = REPO_ROOT / "configs" / "calibration"

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DEFAULT_IDS = {"follower": "so101_follower", "leader": "so101_leader"}

# 板子上两块 CH340/CH343 的 by-id 名，用于提示（不是硬编码，只是打印参考）
KNOWN_BY_ID = {
    "5B41532950": "follower (从臂)",
    "5AAF262805": "leader (主臂)",
}


# --------------------------------------------------------------------------- #
# 构建（延迟 import，保证 --help 不需要 lerobot）
# --------------------------------------------------------------------------- #
def build_device(role: str, port: str, dev_id: str, calib_dir: Path):
    if role == "follower":
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        cfg = SOFollowerRobotConfig(
            port=port,
            id=dev_id,
            calibration_dir=calib_dir,
            # 校准阶段不需要相机；也避免 cameras={} 触发 __post_init__ 的宽高检查
            cameras={},
        )
        return SOFollower(cfg)

    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    cfg = SOLeaderTeleopConfig(port=port, id=dev_id, calibration_dir=calib_dir)
    return SOLeader(cfg)


def resolve_calib_dir(args) -> Path:
    d = Path(args.calibration_dir).expanduser().resolve() if args.calibration_dir else DEFAULT_CALIB_DIR
    return d


def device_id(args) -> str:
    return args.id or DEFAULT_IDS[args.role]


# --------------------------------------------------------------------------- #
# ports：列出串口
# --------------------------------------------------------------------------- #
def cmd_ports(args) -> int:
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    print("=== /dev/serial/by-id ===")
    if not by_id:
        print("  (空) —— 两条 USB 线都插好了吗？")
    for p in by_id:
        try:
            target = os.path.realpath(p)
        except OSError:
            target = "?"
        hint = ""
        for key, label in KNOWN_BY_ID.items():
            if key in p:
                hint = f"   <-- {label}"
                break
        print(f"  {p}\n      -> {target}{hint}")

    print("\n=== /dev/ttyACM* / ttyUSB* ===")
    tty = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    for p in tty:
        print(f"  {p}")

    print("\n提示：ttyACM 序号会随插拔顺序变化，脚本一律用 by-id 全路径最稳。")
    return 0


# --------------------------------------------------------------------------- #
# check：只读自检
# --------------------------------------------------------------------------- #
def cmd_check(args) -> int:
    calib_dir = resolve_calib_dir(args)
    dev_id = device_id(args)
    dev = build_device(args.role, args.port, dev_id, calib_dir)

    print("=" * 64)
    print(f"角色       : {args.role}")
    print(f"端口       : {args.port}")
    print(f"ID         : {dev_id}")
    print(f"校准目录   : {calib_dir}")
    print(f"校准文件   : {dev.calibration_fpath}")
    print(f"文件已存在 : {dev.calibration_fpath.exists()}")
    print("=" * 64)
    print("连接总线（只读，不写入任何校准数据）...")

    try:
        dev.bus.connect()
    except Exception as e:  # noqa: BLE001
        print(f"\n[失败] 总线连接失败: {type(e).__name__}: {e}")
        print("      检查：电源是否打开 / USB 线是否插紧 / 端口是否正确 / 是否被别的进程占用")
        return 2

    try:
        try:
            pos = dev.bus.sync_read("Present_Position")
        except Exception as e:  # noqa: BLE001
            print(f"\n[失败] 读取位置失败: {type(e).__name__}: {e}")
            print("      六个舵机都通电了吗？1Mbaud 下 PING 通不通？")
            return 3

        print(f"\n{'电机':<16}{'原始值(0-4095)':>16}   状态")
        print("-" * 46)
        missing = []
        for m in MOTORS:
            v = pos.get(m)
            if v is None:
                print(f"{m:<16}{'--':>16}   缺失")
                missing.append(m)
                continue
            if v in (0, 4095):
                note = "边界值，注意"
            else:
                note = "OK"
            print(f"{m:<16}{v:>16}   {note}")

        if missing:
            print(f"\n[警告] 缺少电机读数: {missing}")
            return 4
        print("\n[成功] 六个舵机全部响应。")
        if not dev.calibration_fpath.exists():
            print("       还没有校准文件 → 下一步跑 calibrate 子命令。")
        else:
            print("       已有校准文件 → 若更换了机械臂或想重来，加 --force。")
        return 0
    finally:
        # 关键：MotorsBus.disconnect() 的默认参数是 disable_torque=True，
        # 会把扭矩关掉 → 上电保持姿态的从臂会当场瘫下来。
        # 自检是纯只读操作，必须保持扭矩状态不变。
        try:
            dev.bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# calibrate：交互式校准
# --------------------------------------------------------------------------- #
def cmd_calibrate(args) -> int:
    calib_dir = resolve_calib_dir(args)
    dev_id = device_id(args)
    dev = build_device(args.role, args.port, dev_id, calib_dir)

    print("=" * 64)
    print(f"校准 {args.role}   端口 {args.port}   ID {dev_id}")
    print(f"目标文件: {dev.calibration_fpath}")
    print("=" * 64)
    print()
    print("!! 扭矩警告 !!")
    print("   校准过程会主动 disable_torque()：机械臂将失去支撑。")
    print("   从臂请先用手扶住或垫好，别让它砸下来。")
    print()

    if dev.calibration_fpath.exists() and not args.force:
        print("\n该 ID 已有校准文件。")
        print("  * 直接回车 = 把文件里的校准值写进舵机（沿用旧标定）")
        print("  * 输入 c 回车 = 重新走一遍完整校准")
        print("  * 想跳过这个询问、强制重标定，请 Ctrl+C 后加 --force 重跑\n")

    try:
        dev.bus.connect()
    except Exception as e:  # noqa: BLE001
        print(f"\n[失败] 总线连接失败: {type(e).__name__}: {e}")
        return 2

    try:
        if args.force:
            print("\n[--force] 直接进入完整校准流程。\n")
            _interactive_calibrate(dev)
        else:
            dev.calibrate()

        # 回读确认
        print("\n" + "=" * 64)
        print("回读校验")
        print("=" * 64)
        try:
            pos = dev.bus.sync_read("Present_Position")
            print(f"{'电机':<16}{'原始值':>10}{'归零偏移':>12}{'range_min':>12}{'range_max':>12}")
            print("-" * 62)
            for m in MOTORS:
                cal = (dev.calibration or {}).get(m)
                ho = getattr(cal, "homing_offset", "--")
                rmin = getattr(cal, "range_min", "--")
                rmax = getattr(cal, "range_max", "--")
                print(f"{m:<16}{pos.get(m, '--'):>10}{ho:>12}{rmin:>12}{rmax:>12}")
        except Exception as e:  # noqa: BLE001
            print(f"  (回读失败，可忽略: {type(e).__name__}: {e})")

        print(f"\n[成功] 校准文件: {dev.calibration_fpath}")
        print(f"       舵机内校准态: is_calibrated = {dev.is_calibrated}")
        return 0
    finally:
        try:
            dev.bus.disconnect()
        except Exception:  # noqa: BLE001
            pass


def _interactive_calibrate(dev) -> None:
    """完整校准流程，跳过‘是否沿用旧文件’的询问。

    与 LeRobot 的 Device.calibrate() 逻辑保持一致，只是不询问旧文件。
    """
    from lerobot.motors import MotorCalibration
    from lerobot.motors.feetech import OperatingMode

    print(f"\n开始校准 {dev}")
    dev.bus.disable_torque()
    for motor in dev.bus.motors:
        dev.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    input("把机械臂摆到各关节行程的**中间位置**，然后回车...")
    homing_offsets = dev.bus.set_half_turn_homings()

    full_turn_motor = "wrist_roll"
    unknown_range_motors = [m for m in dev.bus.motors if m != full_turn_motor]
    print(
        f"\n现在依次把除 '{full_turn_motor}' 以外的每个关节**手动转完整个行程**。\n"
        "脚本会持续记录，转完后按回车结束..."
    )
    range_mins, range_maxes = dev.bus.record_ranges_of_motion(unknown_range_motors)
    range_mins[full_turn_motor] = 0
    range_maxes[full_turn_motor] = 4095

    dev.calibration = {}
    for motor, m in dev.bus.motors.items():
        dev.calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )

    dev.bus.write_calibration(dev.calibration)
    dev._save_calibration()
    print(f"校准已保存到 {dev.calibration_fpath}")


# --------------------------------------------------------------------------- #
# show：查看已保存的校准文件
# --------------------------------------------------------------------------- #
def cmd_show(args) -> int:
    calib_dir = resolve_calib_dir(args)
    dev_id = device_id(args)
    # 实例化只为拿到权威路径。不会连硬件：__init__ 只做 mkdir + 读文件。
    # 注意 Robot.__init__ 是 `calibration_dir = config.calibration_dir or <HF缓存>/robots/<name>`，
    # 传入 calibration_dir 时**不会**再拼 robots/so_follower 子目录，文件就是 <dir>/<id>.json。
    # show 子命令没有 --port（不需要连硬件），getattr 兜底
    dev = build_device(
        args.role, getattr(args, "port", None) or "UNUSED-NO-HARDWARE", dev_id, calib_dir
    )
    fpath = dev.calibration_fpath

    print(f"路径: {fpath}")
    if not fpath.exists():
        print("  (不存在)")
        # 顺手看看同目录下有没有别的（比如 None.json）
        parent = fpath.parent
        if parent.exists():
            others = sorted(p.name for p in parent.glob("*.json"))
            if others:
                print(f"  同目录其它文件: {others}")
        return 1

    data = json.loads(fpath.read_text())
    print(f"{'电机':<16}{'id':>4}{'drive':>7}{'homing':>10}{'rmin':>8}{'rmax':>8}")
    print("-" * 54)
    for m in MOTORS:
        c = data.get(m, {})
        print(
            f"{m:<16}{c.get('id', '--'):>4}{c.get('drive_mode', '--'):>7}"
            f"{c.get('homing_offset', '--'):>10}{c.get('range_min', '--'):>8}"
            f"{c.get('range_max', '--'):>8}"
        )
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="SO-101 校准/自检（绕开 lerobot-calibrate 的依赖沼泽）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p, need_port=True):
        p.add_argument("--role", choices=["follower", "leader"], required=True,
                       help="follower=从臂(被控制) / leader=主臂(手拖)")
        p.add_argument("--id", default=None,
                       help="设备 ID，决定校准文件名。默认 so101_follower / so101_leader")
        p.add_argument("--calibration-dir", default=None,
                       help=f"校准文件根目录，默认 {DEFAULT_CALIB_DIR}")
        if need_port:
            p.add_argument("--port", required=True,
                           help="串口，如 /dev/ttyACM0 或 /dev/serial/by-id/usb-1a86_...")

    p_ports = sub.add_parser("ports", help="列出串口")
    p_ports.set_defaults(func=cmd_ports)

    p_check = sub.add_parser("check", help="只读自检（不写任何数据）")
    add_common(p_check)
    p_check.set_defaults(func=cmd_check)

    p_cal = sub.add_parser("calibrate", help="交互式校准")
    add_common(p_cal)
    p_cal.add_argument("--force", action="store_true",
                       help="跳过‘是否沿用旧文件’询问，直接完整重标定")
    p_cal.set_defaults(func=cmd_calibrate)

    p_show = sub.add_parser("show", help="查看已保存的校准文件")
    add_common(p_show, need_port=False)
    p_show.set_defaults(func=cmd_show)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
