"""
protocol_v2.py -- the RX node's 42-byte USB frames (from the advisor's Inertial_Sensors_Capture.py):

    A5 5A | 30-byte protocol-v2 payload | 8 diagnostic bytes | CRC16-CCITT (big-endian)
    921600 baud, 400 Hz, RX_Node_60B_True_400Hz_09_23_2026 with the 09_26 transmitters

The framing (header, length, CRC over payload + diagnostics) is implemented here. The contents
of the 30-byte payload are decoded by the advisor's `serial_protocol.py` (ProtocolV2Decoder),
which is not in this repo: copy it (and nothing else) into IMU/ on the Orin, or point
DCKFT_SERIAL_PROTOCOL at the folder that holds it. Its records are mapped onto ImuSample:

    tx_us      <- icm_samd_timestamp_unwrapped_us   (device time, already unwrapped: no wrap)
    accel      <- icm45686_accel_mps2               gyro <- icm45686_gyro_rad_s
    accel_hg   <- adxl375_accel_mps2                impact_flag <- impact_flag (firmware)

Frames the decoder marks duplicate / out-of-order, or with icm_valid false, are dropped.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import List, Optional

import numpy as np

HEADER = b"\xA5\x5A"
FRAME_SIZE = 42
PAYLOAD = slice(2, 32)
DIAG = slice(32, 40)
CRC = slice(40, 42)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF): fallback if serial_protocol is absent."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def load_serial_protocol():
    """Import the advisor's serial_protocol module from IMU/, the repo root, or $DCKFT_SERIAL_PROTOCOL."""
    for d in (os.environ.get("DCKFT_SERIAL_PROTOCOL"), os.path.join(_REPO, "IMU"), _REPO):
        if d and os.path.exists(os.path.join(d, "serial_protocol.py")) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        return importlib.import_module("serial_protocol")
    except ImportError:
        return None


class V2Parser:
    """Streaming parser for the 42-byte frames. Same interface as frames.FrameParser."""

    def __init__(self, cfg=None, protocol=None):
        self.sp = protocol if protocol is not None else load_serial_protocol()
        if self.sp is None:
            raise RuntimeError(
                "The IMU receiver is sending protocol-v2 (A5 5A, 42-byte) frames, but "
                "serial_protocol.py is missing. Copy the advisor's serial_protocol.py into the "
                "repo's IMU/ folder (or set DCKFT_SERIAL_PROTOCOL to its folder) and run again.")
        self.decoder = self.sp.ProtocolV2Decoder()
        self.crc = getattr(self.sp, "crc16_ccitt", _crc16_ccitt_false)
        self.valid_payload = getattr(self.sp, "is_valid_payload", lambda p: True)
        self.buf = bytearray()
        self.good = self.resyncs = self.skipped_bytes = 0
        self.crc_failures = self.payload_failures = 0

    def feed(self, data: bytes, ts_ns_host: int = 0) -> list:
        from .frames import ImuSample
        self.buf.extend(data)
        out: List[ImuSample] = []
        while len(self.buf) >= FRAME_SIZE:
            i = self.buf.find(HEADER)
            if i < 0:
                self.skipped_bytes += len(self.buf) - 1
                del self.buf[:-1]
                break
            if i:
                self.skipped_bytes += i
                self.resyncs += 1
                del self.buf[:i]
            if len(self.buf) < FRAME_SIZE:
                break
            fr = bytes(self.buf[:FRAME_SIZE])
            payload, diag = fr[PAYLOAD], fr[DIAG]
            if int.from_bytes(fr[CRC], "big") != self.crc(payload + diag):
                self.crc_failures += 1
                del self.buf[0]
                continue
            if not self.valid_payload(payload):
                self.payload_failures += 1
                del self.buf[0]
                continue
            del self.buf[:FRAME_SIZE]
            rec = self.decoder.decode(payload, diag)
            if getattr(rec, "tx_timestamp_event", None) in ("duplicate", "out_of_order"):
                continue
            if not getattr(rec, "icm_valid", True):
                continue
            out.append(ImuSample(
                tx_us=int(rec.icm_samd_timestamp_unwrapped_us),
                accel=np.asarray(rec.icm45686_accel_mps2, float),
                gyro=np.asarray(rec.icm45686_gyro_rad_s, float),
                accel_hg=np.asarray(getattr(rec, "adxl375_accel_mps2", (0.0, 0.0, 0.0)), float),
                hf=np.zeros(3),
                status=int(getattr(rec, "status_word", 0)),
                ts_ns_host=ts_ns_host,
                impact_flag=int(bool(getattr(rec, "impact_flag", 0)))))
            self.good += 1
        return out


def looks_like_v2(data: bytes) -> bool:
    """True if the bytes contain >= 3 A5 5A headers spaced exactly 42 bytes apart."""
    i = data.find(HEADER)
    hits = 0
    while 0 <= i and i + FRAME_SIZE <= len(data):
        if data[i:i + 2] == HEADER:
            hits += 1
            i += FRAME_SIZE
        else:
            j = data.find(HEADER, i + 1)
            hits, i = 0, j
        if hits >= 3:
            return True
    return False
