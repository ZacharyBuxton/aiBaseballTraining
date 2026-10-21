"""
frames.py -- decode the IMU's 30-byte binary frames (ICM45686 + ADXL375 board).

Frame layout (big-endian), unchanged from the previous team's firmware and carried as-is
over the 900 MHz radio link:

    bytes  0..3   uint32  tx timestamp, microseconds (wraps at 2^32)
    bytes  4..21  9x int16: accel x,y,z (ICM45686) | gyro x,y,z | accel2 (ADXL375) raw 0,1,2
    bytes 22..29  4x uint16: hf1, hf2, hf3, status   (status == 4095 marks a valid frame)

The axis signs / ADXL axis swap below are copied from IMU/legacy/imu_data_collection.py so
new CSVs are column-for-column comparable with the old ones.

Unlike the legacy reader (which threw away its whole buffer on one bad frame), FrameParser
re-synchronises by scanning for the status word, so a dropped radio byte costs one frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from ..config import ImuConfig

FRAME_SIZE = 30
_STRUCT = struct.Struct(">I9h4H")
assert _STRUCT.size == FRAME_SIZE

CSV_COLUMNS = [
    "tx_time_stamp", "imu_meas_time", "system_time", "status_var_value",
    "accel_x", "accel_y", "accel_z", "omega_x", "omega_y", "omega_z",
    "accel_x2", "accel_y2", "accel_z2", "accel_hf_1", "accel_hf_2", "accel_hf_3",
    "ts_ns_host",
]


@dataclass
class ImuSample:
    tx_us: int                 # raw device timestamp (wrapping uint32)
    accel: np.ndarray          # m/s^2, ICM45686
    gyro: np.ndarray           # rad/s
    accel_hg: np.ndarray       # m/s^2, ADXL375 high-g
    hf: np.ndarray             # extra high-g channels (legacy scaling)
    status: int
    ts_ns_host: int = 0        # host time.time_ns() when the frame was decoded

    def to_dict(self) -> dict:
        return {
            "tx_us": int(self.tx_us), "ts_ns_host": int(self.ts_ns_host),
            "accel": self.accel.tolist(), "gyro": self.gyro.tolist(),
            "accel_hg": self.accel_hg.tolist(), "hf": self.hf.tolist(),
        }


def decode_frame(frame: bytes, cfg: ImuConfig = ImuConfig(), ts_ns_host: int = 0) -> Optional[ImuSample]:
    if len(frame) != FRAME_SIZE:
        return None
    v = _STRUCT.unpack(frame)
    if v[13] != cfg.sync_word:
        return None
    a, g, h = cfg.accel_lsb, cfg.gyro_lsb, cfg.hg_lsb
    return ImuSample(
        tx_us=v[0],
        accel=np.array([-a * v[1], a * v[2], -a * v[3]]),
        gyro=np.array([-g * v[4], g * v[5], -g * v[6]]),
        accel_hg=np.array([h * v[8], -h * v[7], h * v[9]]),
        hf=np.array([h * v[10], h * v[11], h * v[12]]),
        status=v[13],
        ts_ns_host=ts_ns_host,
    )


def encode_frame(tx_us: int, accel, gyro, accel_hg=(0, 0, 0), hf=(0, 0, 0),
                 cfg: ImuConfig = ImuConfig()) -> bytes:
    """Inverse of decode_frame (used by the simulator and tests)."""
    a, g, h = cfg.accel_lsb, cfg.gyro_lsb, cfg.hg_lsb
    clip16 = lambda x: int(np.clip(round(x), -32768, 32767))
    clipu = lambda x: int(np.clip(round(x), 0, 65535))
    return _STRUCT.pack(
        int(tx_us) & 0xFFFFFFFF,
        clip16(-accel[0] / a), clip16(accel[1] / a), clip16(-accel[2] / a),
        clip16(-gyro[0] / g), clip16(gyro[1] / g), clip16(-gyro[2] / g),
        clip16(-accel_hg[1] / h), clip16(accel_hg[0] / h), clip16(accel_hg[2] / h),
        clipu(hf[0] / h), clipu(hf[1] / h), clipu(hf[2] / h),
        cfg.sync_word,
    )


class FrameParser:
    """Streaming byte parser with sync-word resynchronisation."""

    def __init__(self, cfg: ImuConfig = ImuConfig()):
        self.cfg = cfg
        self.buf = bytearray()
        self.sync = struct.pack(">H", cfg.sync_word)
        self.good = 0
        self.resyncs = 0
        self.skipped_bytes = 0

    def _aligned(self, off: int) -> bool:
        n = FRAME_SIZE
        if self.buf[off + n - 2: off + n] != self.sync:
            return False
        # If the next frame is already buffered, require it to line up too.
        nxt = off + 2 * n
        if len(self.buf) >= nxt and self.buf[nxt - 2: nxt] != self.sync:
            return False
        return True

    def feed(self, data: bytes, ts_ns_host: int = 0) -> List[ImuSample]:
        self.buf.extend(data)
        out: List[ImuSample] = []
        n = FRAME_SIZE
        while len(self.buf) >= n:
            if self.buf[n - 2:n] == self.sync:
                s = decode_frame(bytes(self.buf[:n]), self.cfg, ts_ns_host)
                del self.buf[:n]
                if s is not None:
                    out.append(s)
                    self.good += 1
                continue
            # Lost alignment: slide forward to the next plausible frame boundary.
            self.resyncs += 1
            found = None
            for off in range(1, len(self.buf) - n + 1):
                if self._aligned(off):
                    found = off
                    break
            if found is None:
                keep = n - 1
                self.skipped_bytes += len(self.buf) - keep
                del self.buf[:len(self.buf) - keep]
                break
            self.skipped_bytes += found
            del self.buf[:found]
        return out


def iter_frames(data: bytes, cfg: ImuConfig = ImuConfig()) -> Iterator[ImuSample]:
    p = FrameParser(cfg)
    yield from p.feed(data)


class TxClock:
    """Unwraps the device's uint32 microsecond counter into a monotonically increasing int."""

    def __init__(self):
        self.last = None
        self.wraps = 0

    def __call__(self, tx_us: int) -> int:
        if self.last is not None and tx_us < self.last and (self.last - tx_us) > 2 ** 31:
            self.wraps += 1
        self.last = tx_us
        return tx_us + self.wraps * 2 ** 32
