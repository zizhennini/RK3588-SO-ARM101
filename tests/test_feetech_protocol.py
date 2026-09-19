"""Feetech STS3215 协议字节级测试。

用假串口捕获实际写出的字节，与手算的期望包逐字节比对。
寄存器地址写错可能损坏舵机，所以这部分必须字节级验证。
"""

from __future__ import annotations

import sys

import numpy as np

import _env  # noqa: F401

_env.install_stubs()

from feetech_bus import (  # noqa: E402
    BROADCAST_ID,
    BusConfig,
    FeetechBus,
    checksum,
    decode_sign_magnitude,
    encode_sign_magnitude,
    load_bus_config,
)


class FakeSerial:
    """记录写出的字节；read() 返回预置的应答。"""

    def __init__(self, response: bytes = b""):
        self.response = response
        self.written = bytearray()
        self.is_open = True

    def write(self, data):
        self.written.extend(data)
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def read(self, n):
        out, self.response = self.response[:n], self.response[n:]
        return out

    def close(self):
        self.is_open = False


def _bus(response: bytes = b"", **kw) -> FeetechBus:
    cfg = BusConfig(port="FAKE", num_retries=0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    b = FeetechBus(cfg)
    b.ser = FakeSerial(response)
    return b


# ---------- 校验和 ----------

def test_checksum_known_values() -> None:
    # READ Present_Position(56), len 2, servo 1: body = [01,04,02,38,02]
    assert checksum([0x01, 0x04, 0x02, 0x38, 0x02]) == 0xBE
    # WRITE Torque_Enable(40)=1, servo 3: body = [03,04,03,28,01]
    assert checksum([0x03, 0x04, 0x03, 0x28, 0x01]) == 0xCC
    # 全零
    assert checksum([0, 0]) == 0xFF


# ---------- 包结构 ----------

def test_read_packet_bytes() -> None:
    b = _bus()
    b._build(1, 2, [56, 2])  # INSTR_READ
    assert bytes(b._build(1, 2, [56, 2])) == bytes([0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE])


def test_write_packet_bytes() -> None:
    b = _bus()
    assert bytes(b._build(3, 3, [40, 1])) == bytes([0xFF, 0xFF, 0x03, 0x04, 0x03, 0x28, 0x01, 0xCC])


def test_sync_write_packet_layout() -> None:
    b = _bus()
    b.sync_write_u16(42, {1: 2047, 2: 1000})
    w = bytes(b.ser.written)
    assert w[:2] == b"\xff\xff"
    assert w[2] == BROADCAST_ID, "SYNC_WRITE 必须用广播 ID 0xFE"
    assert w[4] == 0x83, "指令应为 SYNC_WRITE(0x83)"
    assert w[5] == 42 and w[6] == 2, "地址/字节长度不对"
    # 每个舵机 3 字节：ID + 2 字节小端
    assert w[7] == 1 and w[8] == 2047 & 0xFF and w[9] == 2047 >> 8
    assert w[10] == 2 and w[11] == 1000 & 0xFF and w[12] == 1000 >> 8
    assert w[-1] == checksum(list(w[2:-1])), "校验和不符"
    # 包长 = 2(header) + 1(id) + 1(len) + 1(instr) + 2(addr,len) + 2*3(舵机) + 1(cs) = 14
    assert len(w) == 14, f"包长应为 14，实际 {len(w)}: {bytes(w).hex()}"


def test_goal_position_address_is_42() -> None:
    """回归保护：Goal_Position 必须是 0x2A(42)。写错地址可能损坏舵机。"""
    b = _bus()
    assert b.cfg.addr_goal_position == 42
    assert b.cfg.addr_present_position == 56
    assert b.cfg.addr_torque_enable == 40


# ---------- 符号-幅值 ----------

def test_sign_magnitude_roundtrip() -> None:
    for v in (-4095, -2047, -1, 0, 1, 2047, 4095):
        enc = encode_sign_magnitude(v, 15)
        assert 0 <= enc <= 0xFFFF, (v, enc)
        assert decode_sign_magnitude(enc, 15) == v, (v, enc)


def test_sign_magnitude_known() -> None:
    assert encode_sign_magnitude(-2047, 15) == 34815
    assert decode_sign_magnitude(34815, 15) == -2047
    # 位置在 0..4095 内应为恒等
    assert decode_sign_magnitude(2047, 15) == 2047
    assert encode_sign_magnitude(2047, 15) == 2047


def test_present_load_sign_bit_is_10() -> None:
    """Present_Load 的符号位是 10，不是 15。"""
    assert decode_sign_magnitude(0x0400 | 100, 10) == -100
    assert decode_sign_magnitude(100, 10) == 100


# ---------- SYNC_READ 解析 ----------

def test_sync_read_parses_two_servos() -> None:
    ids = [1, 2]
    pos1, pos2 = 2047, 1000
    body = [0xFE, 0x08, 0x00,
            1, pos1 & 0xFF, pos1 >> 8,
            2, pos2 & 0xFF, pos2 >> 8]
    resp = bytes([0xFF, 0xFF, *body, checksum(body)])
    b = _bus(response=resp)
    got = b.sync_read_u16(56, ids)
    assert got == {1: pos1, 2: pos2}, got


def test_sync_read_rejects_bad_checksum() -> None:
    body = [0xFE, 0x08, 0x00, 1, 0xFF, 0x07, 2, 0xE8, 0x03]
    resp = bytes([0xFF, 0xFF, *body, 0x00])  # 故意错校验和
    b = _bus(response=resp)
    try:
        b.sync_read_u16(56, [1, 2])
    except IOError:
        return
    raise AssertionError("坏校验和应被拒绝")


# ---------- 配置加载 ----------

def test_load_bus_config() -> None:
    cfg = load_bus_config(str(_env.CONFIGS / "robot.yaml"), str(_env.CONFIGS / "feetech_sts3215.yaml"))
    assert cfg.baudrate == 1_000_000
    assert cfg.addr_goal_position == 42
    assert cfg.resolution == 4096
    assert cfg.sign_bit_position == 15
    assert "ttyACM0" in cfg.port


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"Feetech 协议: {len(tests) - failed}/{len(tests)} 通过")
    return failed


if __name__ == "__main__":
    sys.exit(main())
