"""
frames.py -- decode the IMU's binary frames. The layout is a FrameFormat (FORMATS below), so the
new IMU firmware is a config change: imu.frame_format. "v1" = the current 30-byte frame
(ICM45686 + ADXL375 board), documented here:

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

import re
import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from ..config import ImuConfig

@dataclass(frozen=True)
class FrameFormat:
    """
    One firmware's binary frame layout. The decoder, parser and simulator are driven by this, so
    the NEW IMU firmware only needs a new entry in FORMATS (and imu.frame_format in the config).

      struct      big-endian struct string for the whole frame
      sync_index  index (into the unpacked tuple) of the status/sync word; its 2 bytes are the
                  LAST two bytes of the frame (what the resync scanner looks for)
      ts_index    index of the device timestamp; ts_bits 32 (wraps) or 64 (no wrap);
                  ts_unit_us = microseconds per timestamp tick
      accel/gyro/hg/hf   ((index, sign), ...) per axis, before LSB scaling; hg/hf may be ()
    """
    name: str
    struct: str
    sync_index: int
    ts_index: int
    ts_bits: int
    ts_unit_us: float
    accel: tuple
    gyro: tuple
    hg: tuple = ()
    hf: tuple = ()

    @property
    def size(self) -> int:
        return struct.calcsize(self.struct)


FORMATS = {
    # Previous team's firmware, carried unchanged over the 900 MHz radio (Oct 2026):
    #   uint32 tx_us | int16 ICM accel xyz | int16 ICM gyro xyz | int16 ADXL375 raw 0,1,2 |
    #   uint16 hf1..3 | uint16 status (4095)
    # Axis signs / ADXL swap copied from IMU/legacy/imu_data_collection.py.
    "v1": FrameFormat(
        name="v1", struct=">I9h4H", sync_index=13, ts_index=0, ts_bits=32, ts_unit_us=1.0,
        accel=((1, -1), (2, 1), (3, -1)), gyro=((4, -1), (5, 1), (6, -1)),
        hg=((8, 1), (7, -1), (9, 1)), hf=((10, 1), (11, 1), (12, 1))),
    # NEW FIRMWARE: add its layout here once Charles / Dr. Jupina fix it, e.g. an ICM45686-only
    # frame with a 64-bit microsecond timestamp (no 71-minute wrap, better host mapping):
    #   "v2": FrameFormat(name="v2", struct=">Q6hH", sync_index=7, ts_index=0, ts_bits=64,
    #                     ts_unit_us=1.0, accel=((1, -1), (2, 1), (3, -1)),
    #                     gyro=((4, -1), (5, 1), (6, -1))),
    # then set imu.frame_format = "v2" and imu.baud to match. Verify signs with a tilt test.
}


def frame_format(cfg: ImuConfig) -> FrameFormat:
    try:
        return FORMATS[cfg.frame_format]
    except KeyError:
        raise ValueError(f"Unknown imu.frame_format {cfg.frame_format!r}; known: {list(FORMATS)}")


FRAME_SIZE = FORMATS["v1"].size          # kept for older callers

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


def _axes(v, spec, lsb):
    return np.array([sign * lsb * v[i] for i, sign in spec]) if spec else np.zeros(3)


def decode_frame(frame: bytes, cfg: ImuConfig = ImuConfig(), ts_ns_host: int = 0) -> Optional[ImuSample]:
    fmt = frame_format(cfg)
    if len(frame) != fmt.size:
        return None
    v = struct.unpack(fmt.struct, frame)
    if v[fmt.sync_index] != cfg.sync_word:
        return None
    return ImuSample(
        tx_us=int(round(v[fmt.ts_index] * fmt.ts_unit_us)),
        accel=_axes(v, fmt.accel, cfg.accel_lsb),
        gyro=_axes(v, fmt.gyro, cfg.gyro_lsb),
        accel_hg=_axes(v, fmt.hg, cfg.hg_lsb),
        hf=_axes(v, fmt.hf, cfg.hg_lsb),
        status=v[fmt.sync_index],
        ts_ns_host=ts_ns_host,
    )


def encode_frame(tx_us: int, accel, gyro, accel_hg=(0, 0, 0), hf=(0, 0, 0),
                 cfg: ImuConfig = ImuConfig()) -> bytes:
    """Inverse of decode_frame for the configured format (simulator and tests)."""
    fmt = frame_format(cfg)
    codes = struct.Struct(fmt.struct)
    n = len(codes.unpack(bytes(fmt.size)))
    vals = [0] * n
    expanded = []                                         # one struct code per value
    for part in re.findall(r"(\d*)([a-zA-Z])", fmt.struct):
        expanded += [part[1]] * (int(part[0]) if part[0] else 1)

    def put(spec, xyz, lsb):
        for (i, sign), x in zip(spec, xyz):
            vals[i] = sign * x / lsb

    put(fmt.accel, accel, cfg.accel_lsb)
    put(fmt.gyro, gyro, cfg.gyro_lsb)
    put(fmt.hg, accel_hg, cfg.hg_lsb)
    put(fmt.hf, hf, cfg.hg_lsb)
    vals[fmt.ts_index] = int(round(tx_us / fmt.ts_unit_us)) & ((1 << fmt.ts_bits) - 1)
    vals[fmt.sync_index] = cfg.sync_word
    lim = {"h": (-32768, 32767), "H": (0, 65535), "i": (-2**31, 2**31 - 1), "I": (0, 2**32 - 1),
           "q": (-2**63, 2**63 - 1), "Q": (0, 2**64 - 1), "b": (-128, 127), "B": (0, 255)}
    out = []
    for code, x in zip(expanded, vals):
        lo, hi = lim.get(code, (-2**63, 2**63 - 1))
        out.append(int(np.clip(round(x), lo, hi)))
    return codes.pack(*out)


class FrameParser:
    """Streaming byte parser with sync-word resynchronisation."""

    def __init__(self, cfg: ImuConfig = ImuConfig()):
        self.cfg = cfg
        self.n = frame_format(cfg).size
        self.buf = bytearray()
        self.sync = struct.pack(">H", cfg.sync_word)
        self.good = 0
        self.resyncs = 0
        self.skipped_bytes = 0

    def _aligned(self, off: int) -> bool:
        n = self.n
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
        n = self.n
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
    """Unwraps a wrapping device counter (32-bit by default) into a monotonically increasing int.
    A 64-bit timestamp (new firmware) never wraps, so it passes through unchanged."""

    def __init__(self, bits: int = 32):
        self.last = None
        self.wraps = 0
        self.bits = bits

    def __call__(self, tx_us: int) -> int:
        if self.bits >= 64:
            return tx_us
        if self.last is not None and tx_us < self.last and (self.last - tx_us) > 2 ** 31:
            self.wraps += 1
        self.last = tx_us
        return tx_us + self.wraps * 2 ** 32
