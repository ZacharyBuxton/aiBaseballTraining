"""
reader.py -- cross-platform IMU / 900 MHz radio receiver ingestion.

Replaces IMU/legacy/imu_data_collection.py (Windows-only, hard-coded COM10, fixed 2400 samples).

    # list serial ports the receiver might be on
    python -m dckft.imu.reader --list-ports

    # stream forever: publish on ZMQ :5558 for dckft.run live, and log a CSV
    python -m dckft.imu.reader --port /dev/ttyUSB0 --csv sessions/imu_$(date +%s).csv

    # legacy-style fixed capture (2400 samples = 6 s at 400 Hz)
    python -m dckft.imu.reader --samples 2400 --csv reports/imu_test.csv

    # measure gyro bias with the sensor sitting still
    python -m dckft.imu.reader --calibrate 3 --calib-out imu_calib.json

    # re-publish a recorded CSV at real-time speed (tests the live pipeline without hardware)
    python -m dckft.imu.reader --replay reports/imu_test.csv

Every sample is stamped with time.time_ns() on the host at decode time. When this runs on the
Jetson, that is the same clock the ZED SDK uses for ts_ns_image, so camera and IMU share a
timebase (dckft.sync then removes host jitter by regressing host time on the device clock).

Linux: add yourself to the dialout group once (`sudo usermod -aG dialout $USER`, then log out/in)
or the port open fails with "Permission denied".
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from typing import Optional

import numpy as np

from ..config import ImuConfig, load_config
from .frames import CSV_COLUMNS, FrameParser, ImuSample, TxClock


def list_ports():
    from serial.tools import list_ports
    return list(list_ports.comports())


def autodetect_port() -> Optional[str]:
    """Prefer USB serial adapters (FTDI/CP210x/CH340/ACM). Returns None if nothing found."""
    ports = list_ports()
    scored = []
    for p in ports:
        desc = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        score = 0
        for key, s in (("ttyusb", 3), ("ttyacm", 3), ("usb", 2), ("ftdi", 2), ("cp210", 2),
                       ("ch340", 2), ("serial", 1), ("com", 1)):
            if key in desc:
                score += s
        if "zed" in desc or "bluetooth" in desc:
            score -= 5
        scored.append((score, p.device))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


class CsvSink:
    """Writes the legacy column set plus ts_ns_host (fuse/analysis code reads either)."""

    def __init__(self, path: str):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(CSV_COLUMNS)
        self.clock = TxClock()
        self.tx0 = None
        self.host0 = None

    def write(self, s: ImuSample) -> None:
        tx = self.clock(s.tx_us)
        if self.tx0 is None:
            self.tx0, self.host0 = tx, s.ts_ns_host
        self.w.writerow([
            s.tx_us, (tx - self.tx0) / 1e6, (s.ts_ns_host - self.host0) / 1e9, s.status,
            *np.round(s.accel, 6), *np.round(s.gyro, 6), *np.round(s.accel_hg, 4),
            *np.round(s.hf, 4), s.ts_ns_host,
        ])

    def close(self):
        self.f.close()


class ZmqSink:
    """PUSH socket, non-blocking. Sends small JSON batches to keep per-message overhead low."""

    def __init__(self, bind: str, batch: int = 8):
        import zmq
        self.zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 2000)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(bind)
        self.batch = batch
        self.pending = []
        self.dropped = 0

    def write(self, s: ImuSample) -> None:
        self.pending.append(s.to_dict())
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        try:
            self.sock.send_string(json.dumps({"type": "imu", "samples": self.pending}),
                                  flags=self.zmq.NOBLOCK)
        except self.zmq.Again:
            self.dropped += len(self.pending)
        self.pending = []

    def close(self):
        self.flush()
        self.sock.close()


def run_serial(cfg: ImuConfig, sinks, max_samples: int = 0, duration_s: float = 0.0,
               quiet: bool = False, on_sample=None) -> dict:
    import serial
    port = cfg.port or autodetect_port()
    if not port:
        raise SystemExit("No serial port found. Plug in the radio receiver or pass --port.")
    ser = serial.Serial(port, cfg.baud, timeout=0.02)
    ser.reset_input_buffer()
    parser = FrameParser(cfg)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    n = 0
    t_start = time.time()
    t_last_print = t_start
    if not quiet:
        print(f"Reading {port} @ {cfg.baud} baud. Ctrl+C to stop.")
    try:
        while not stop["flag"]:
            chunk = ser.read(max(ser.in_waiting, cfg.frame_size))
            if not chunk:
                continue
            now = time.time_ns()
            for s in parser.feed(chunk, now):
                for k in sinks:
                    k.write(s)
                if on_sample:
                    on_sample(s)
                n += 1
                if max_samples and n >= max_samples:
                    stop["flag"] = True
                    break
            if duration_s and time.time() - t_start >= duration_s:
                break
            if not quiet and time.time() - t_last_print >= 1.0:
                rate = n / (time.time() - t_start)
                print(f"\r{n} samples  {rate:6.1f} Hz  resyncs={parser.resyncs} "
                      f"skipped={parser.skipped_bytes}B", end="", flush=True)
                t_last_print = time.time()
    finally:
        ser.close()
        for k in sinks:
            k.close()
    elapsed = time.time() - t_start
    stats = {"port": port, "samples": n, "seconds": elapsed, "rate_hz": n / max(elapsed, 1e-9),
             "resyncs": parser.resyncs, "skipped_bytes": parser.skipped_bytes}
    if not quiet:
        print("\n" + json.dumps(stats))
    return stats


def calibrate(cfg: ImuConfig, seconds: float, out_path: Optional[str]) -> dict:
    """Sensor must be still. Gyro bias = mean rate; also reports noise and gravity magnitude."""
    gyro, acc = [], []

    def grab(s):
        gyro.append(s.gyro)
        acc.append(s.accel)

    run_serial(cfg, [], duration_s=seconds, quiet=True, on_sample=grab)
    if len(gyro) < 50:
        raise SystemExit(f"Only {len(gyro)} samples received; check the link.")
    g, a = np.array(gyro), np.array(acc)
    res = {"gyro_bias_rads": g.mean(0).tolist(), "gyro_std_rads": g.std(0).tolist(),
           "accel_mean_ms2": a.mean(0).tolist(), "accel_norm_ms2": float(np.linalg.norm(a.mean(0))),
           "samples": len(g)}
    print(json.dumps(res, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
    return res


def replay_csv(path: str, bind: str, speed: float = 1.0, rebase_time: bool = True) -> None:
    """Publish a recorded CSV over ZMQ, re-stamped with the current host clock."""
    import pandas as pd
    df = pd.read_csv(path)
    t = df["imu_meas_time"].to_numpy(float)
    sink = ZmqSink(bind)
    print(f"Replaying {len(df)} samples from {path} on {bind} (Ctrl+C to stop)")
    time.sleep(0.5)        # let consumers connect
    t0_host = time.time_ns()
    t0 = t[0]
    for i, row in enumerate(df.itertuples(index=False)):
        target = t0_host + int((t[i] - t0) / speed * 1e9)
        while time.time_ns() < target:
            time.sleep(0.0002)
        s = ImuSample(
            tx_us=int(row.tx_time_stamp),
            accel=np.array([row.accel_x, row.accel_y, row.accel_z]),
            gyro=np.array([row.omega_x, row.omega_y, row.omega_z]),
            accel_hg=np.array([row.accel_x2, row.accel_y2, row.accel_z2]),
            hf=np.array([row.accel_hf_1, row.accel_hf_2, row.accel_hf_3]),
            status=4095,
            ts_ns_host=target if rebase_time else int(getattr(row, "ts_ns_host", target)),
        )
        sink.write(s)
    sink.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="IMU / radio receiver reader")
    p.add_argument("--config", help="dckft JSON config (uses its 'imu' section)")
    p.add_argument("--port", help="Serial port, e.g. /dev/ttyUSB0 or COM10 (default: auto)")
    p.add_argument("--baud", type=int)
    p.add_argument("--csv", help="Write samples to this CSV")
    p.add_argument("--no-zmq", action="store_true", help="Don't publish on ZMQ")
    p.add_argument("--zmq-bind", help="PUSH bind address (default tcp://*:5558)")
    p.add_argument("--samples", type=int, default=0, help="Stop after N samples (0 = run until Ctrl+C)")
    p.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--calibrate", type=float, metavar="SECONDS", help="Measure gyro bias at rest")
    p.add_argument("--calib-out", default="imu_calib.json")
    p.add_argument("--replay", help="Publish a recorded CSV over ZMQ instead of reading hardware")
    p.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    cfg = load_config(a.config).imu
    if a.port:
        cfg.port = a.port
    if a.baud:
        cfg.baud = a.baud
    if a.zmq_bind:
        cfg.zmq_bind = a.zmq_bind

    if a.list_ports:
        for port in list_ports():
            print(f"{port.device:20s} {port.description}")
        print(f"auto-detect would pick: {autodetect_port()}")
        return 0
    if a.replay:
        replay_csv(a.replay, cfg.zmq_bind or "tcp://*:5558", a.speed)
        return 0
    if a.calibrate:
        calibrate(cfg, a.calibrate, a.calib_out)
        return 0

    sinks = []
    if a.csv:
        sinks.append(CsvSink(a.csv))
    if not a.no_zmq and cfg.zmq_bind:
        sinks.append(ZmqSink(cfg.zmq_bind))
    if not sinks:
        print("Nothing to do: pass --csv and/or leave ZMQ enabled.", file=sys.stderr)
        return 2
    run_serial(cfg, sinks, a.samples, a.seconds, a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
