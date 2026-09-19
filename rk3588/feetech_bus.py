"""极简 Feetech STS3215 串口总线。

只依赖 pyserial。只实现本项目需要的寄存器读写 + SYNC READ/WRITE。

设计约束
--------
* 寄存器地址全部来自 configs/feetech_sts3215.yaml（转自 LeRobot tables.py），
  不在代码里硬编码 —— 写错寄存器地址可能损坏舵机。
* 只写 Torque_Enable 与 Goal_Position；EPROM 一律不碰。
* 任何通信异常都必须能被上层捕获，以便立刻停扭矩。

未实测：本文件尚未在真实 SO-ARM101 上运行过。首次上板请先单独调用
    bus.read_present_positions()
确认能读到 6 个合理值（约 1000-3000 之间），再考虑写动作。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import serial
import yaml

log = logging.getLogger(__name__)

HEADER = (0xFF, 0xFF)
BROADCAST_ID = 0xFE

INSTR_PING = 1
INSTR_READ = 2
INSTR_WRITE = 3
INSTR_SYNC_READ = 0x82
INSTR_SYNC_WRITE = 0x83


def checksum(parts: list[int]) -> int:
    return (~sum(parts)) & 0xFF


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    """符号-幅值解码。非负值（bit 未置位）恒等返回。"""
    if value & (1 << sign_bit):
        return -(value & ((1 << sign_bit) - 1))
    return value


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    """符号-幅值编码。非负值（且小于 2**sign_bit）恒等返回。"""
    if value < 0:
        return (1 << sign_bit) | (-value)
    return value


@dataclass
class BusConfig:
    port: str
    baudrate: int = 1_000_000
    timeout_s: float = 0.05
    num_retries: int = 2
    # 寄存器地址
    addr_goal_position: int = 42
    addr_present_position: int = 56
    addr_torque_enable: int = 40
    addr_present_temperature: int = 63
    # 编码位
    sign_bit_position: int = 15
    resolution: int = 4096


def load_bus_config(robot_yaml: str, feetech_yaml: str) -> BusConfig:
    with open(robot_yaml, "r", encoding="utf-8") as f:
        robot = yaml.safe_load(f)
    with open(feetech_yaml, "r", encoding="utf-8") as f:
        ft = yaml.safe_load(f)

    regs = ft["registers"]
    enc = ft["encoding_bits"]
    return BusConfig(
        port=robot["robot"]["port"],
        baudrate=int(robot["robot"].get("baudrate", 1_000_000)),
        addr_goal_position=regs["Goal_Position"][0],
        addr_present_position=regs["Present_Position"][0],
        addr_torque_enable=regs["Torque_Enable"][0],
        addr_present_temperature=regs["Present_Temperature"][0],
        sign_bit_position=enc["Present_Position"],
        resolution=int(ft["model"]["resolution"]),
    )


class FeetechBus:
    def __init__(self, cfg: BusConfig):
        self.cfg = cfg
        self.ser: serial.Serial | None = None

    # ---------- 连接 ----------

    def open(self) -> None:
        self.ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baudrate,
            timeout=self.cfg.timeout_s,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        # 有些板子在打开串口后会有残留字节
        time.sleep(0.05)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        log.info("串口已打开 %s @ %d", self.cfg.port, self.cfg.baudrate)

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
            log.info("串口已关闭")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------- 底层收发 ----------

    def _build(self, motor_id: int, instruction: int, params: list[int]) -> bytes:
        length = len(params) + 2
        body = [motor_id, length, instruction, *params]
        return bytes([*HEADER, *body, checksum(body)])

    def _txrx(self, packet: bytes, expected_params: int) -> tuple[int, int, list[int]]:
        """发送请求并读取应答，返回 (motor_id, error, params)。"""
        assert self.ser is not None, "串口未打开"
        last_err: Exception | None = None

        for _ in range(self.cfg.num_retries + 1):
            try:
                self.ser.reset_input_buffer()
                self.ser.write(packet)
                self.ser.flush()

                # 应答: FF FF ID LEN ERR [params...] CHECKSUM
                resp_len = expected_params + 6
                raw = self.ser.read(resp_len)
                if len(raw) < 6:
                    raise IOError(f"应答过短 ({len(raw)} 字节): {raw.hex()}")

                # 容忍前面的噪声，找到 header
                idx = raw.find(bytes(HEADER))
                if idx < 0:
                    raise IOError(f"未找到包头: {raw.hex()}")
                raw = raw[idx:]
                if len(raw) < resp_len:
                    raise IOError(f"应答不完整 ({len(raw)}/{resp_len}): {raw.hex()}")

                motor_id = raw[2]
                length = raw[3]
                error = raw[4]
                params = list(raw[5 : 5 + length - 2])
                got_cs = raw[5 + length - 2]
                want_cs = checksum([motor_id, length, error, *params])
                if got_cs != want_cs:
                    raise IOError(f"校验和不符: got {got_cs:#x} want {want_cs:#x}")
                if error:
                    # error 位含义: bit0 电压 bit1 角度 bit2 过热 bit3 过流 bit4 过载
                    log.warning("舵机 %d 返回错误标志 %#x", motor_id, error)
                return motor_id, error, params

            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.002)

        raise IOError(f"通信失败 (重试 {self.cfg.num_retries} 次): {last_err}")

    # ---------- 单寄存器读写 ----------

    def write_u8(self, motor_id: int, addr: int, value: int) -> None:
        self._txrx(self._build(motor_id, INSTR_WRITE, [addr, value & 0xFF]), 0)

    def write_u16(self, motor_id: int, addr: int, value: int) -> None:
        v = value & 0xFFFF
        self._txrx(self._build(motor_id, INSTR_WRITE, [addr, v & 0xFF, (v >> 8) & 0xFF]), 0)

    def read_u8(self, motor_id: int, addr: int) -> int:
        _, _, params = self._txrx(self._build(motor_id, INSTR_READ, [addr, 1]), 1)
        return params[0]

    def read_u16(self, motor_id: int, addr: int) -> int:
        _, _, params = self._txrx(self._build(motor_id, INSTR_READ, [addr, 2]), 2)
        return params[0] | (params[1] << 8)

    # ---------- 批量（同步）读写 ----------

    def sync_write_u16(self, addr: int, values: dict[int, int]) -> None:
        """一条广播包写完所有舵机的同一寄存器。"""
        params: list[int] = [addr, 2]
        for motor_id, value in values.items():
            v = value & 0xFFFF
            params += [motor_id & 0xFF, v & 0xFF, (v >> 8) & 0xFF]
        assert self.ser is not None
        # SYNC_WRITE 是广播，没有应答
        self.ser.reset_input_buffer()
        self.ser.write(self._build(BROADCAST_ID, INSTR_SYNC_WRITE, params))
        self.ser.flush()

    def sync_read_u16(self, addr: int, ids: list[int]) -> dict[int, int]:
        """一条广播包读回所有舵机的同一寄存器。"""
        params: list[int] = [addr, 2, *[i & 0xFF for i in ids]]
        packet = self._build(BROADCAST_ID, INSTR_SYNC_READ, params)
        assert self.ser is not None

        expected_params = len(ids) * 3  # 每个舵机: ID + 2 字节
        resp_len = expected_params + 6

        last_err: Exception | None = None
        for _ in range(self.cfg.num_retries + 1):
            try:
                self.ser.reset_input_buffer()
                self.ser.write(packet)
                self.ser.flush()
                raw = self.ser.read(resp_len)
                idx = raw.find(bytes(HEADER))
                if idx < 0:
                    raise IOError(f"未找到包头: {raw.hex()}")
                raw = raw[idx:]
                if len(raw) < resp_len:
                    raise IOError(f"应答不完整 ({len(raw)}/{resp_len})")

                length = raw[3]
                error = raw[4]
                body = raw[5 : 5 + length - 2]
                got_cs = raw[5 + length - 2]
                if got_cs != checksum([raw[2], length, error, *body]):
                    raise IOError("校验和不符")

                out: dict[int, int] = {}
                for k in range(0, len(body) - 2, 3):
                    mid = body[k]
                    out[mid] = body[k + 1] | (body[k + 2] << 8)
                if len(out) != len(ids):
                    raise IOError(f"只读到 {len(out)}/{len(ids)} 个舵机")
                return out

            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.002)

        raise IOError(f"SYNC_READ 失败: {last_err}")

    # ---------- 语义化封装 ----------

    def enable_torque(self, ids: list[int], enable: bool = True) -> None:
        values = {i: (1 if enable else 0) for i in ids}
        for motor_id, v in values.items():
            self.write_u8(motor_id, self.cfg.addr_torque_enable, v)

    def read_present_positions(self, ids: list[int]) -> dict[int, int]:
        """读回原始计数（已做符号-幅值解码，一般为 0..4095）。"""
        raw = self.sync_read_u16(self.cfg.addr_present_position, ids)
        return {
            i: decode_sign_magnitude(v, self.cfg.sign_bit_position)
            for i, v in raw.items()
        }

    def write_goal_positions(self, targets: dict[int, int]) -> None:
        """写入目标位置（原始计数）。应在上层做过限位钳制。"""
        encoded = {
            i: encode_sign_magnitude(v, self.cfg.sign_bit_position)
            for i, v in targets.items()
        }
        self.sync_write_u16(self.cfg.addr_goal_position, encoded)

    def read_temperatures(self, ids: list[int]) -> dict[int, int]:
        # Present_Temperature 是 1 字节，SYNC_READ 按 2 字节读会错位，逐个读
        return {i: self.read_u8(i, self.cfg.addr_present_temperature) for i in ids}
