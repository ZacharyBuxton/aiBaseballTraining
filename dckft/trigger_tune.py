"""
trigger_tune.py -- pick the IMU swing trigger (swing.trigger_source / trigger_accel_ms2) from data.

Record ~30 s with the IMU on the BOTTOM hand, wire side UP: stand still a few seconds, take 3-5
practice swings holding the finish still after each, and waggle / walk around a bit between
them (that's what the trigger must NOT fire on). Then:

    python -m dckft.imu.reader --seconds 30 --no-zmq --csv imu_tune.csv
    python -m dckft.trigger_tune --csv imu_tune.csv

or both in one go:

    python -m dckft.trigger_tune --record 30 --port /dev/ttyUSB0

For each candidate signal (|a|-g, |(a_x,a_y)|, each signed axis, |omega|) it prints the peak in
every swing and the largest value outside swings, then suggests the source with the clearest
gap and a threshold halfway between. Swings are found by wrist rate (> 15 rad/s), which needs no
tuning for real swings.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import G, SwingConfig, load_config
from .data import load_imu_csv
from .swing import omega_mag, trigger_signal

CANDIDATES = ["accel_mag", "accel_xy", "accel_-y", "accel_+y", "accel_-x", "accel_+x",
              "accel_-z", "accel_+z", "gyro"]


def analyse(t, gyro, accel, swing_rads: float = 15.0, pad_s: float = 0.4) -> dict:
    rate = 1.0 / np.median(np.diff(t))
    m = omega_mag(gyro, rate)
    hot = m > swing_rads
    # group swing samples into events, padded on both sides
    events, i, n = [], 0, len(t)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and m[j] > 4.0:
            j += 1
        a0 = np.searchsorted(t, t[i] - pad_s)
        b0 = np.searchsorted(t, t[min(j, n - 1)] + pad_s)
        if events and a0 <= events[-1][1]:
            events[-1] = (events[-1][0], b0)
        else:
            events.append((a0, b0))
        i = j + 1
    in_swing = np.zeros(n, bool)
    for a0, b0 in events:
        in_swing[a0:b0] = True

    report = {"swings_found": len(events), "rate_hz": round(rate, 1), "sources": {}}
    for src in CANDIDATES:
        cfg = SwingConfig(trigger_source=src)
        sig, _ = trigger_signal(gyro, accel, cfg, rate)
        peaks = [float(sig[a0:b0].max()) for a0, b0 in events]
        quiet = float(sig[~in_swing].max()) if (~in_swing).any() else float("nan")
        lo = min(peaks) if peaks else float("nan")
        gap = lo - quiet
        report["sources"][src] = {
            "swing_peaks": [round(p, 1) for p in peaks],
            "max_outside_swings": round(quiet, 1),
            "separation": round(gap / max(abs(lo), 1e-6), 3) if peaks else None,
            "suggested_threshold": round(quiet + 0.4 * gap, 1) if peaks and gap > 0 else None,
            "units": "rad/s" if src == "gyro" else "m/s^2",
        }
    ok = {s: r for s, r in report["sources"].items() if r["suggested_threshold"] is not None}
    if ok:
        best = max(ok, key=lambda s: ok[s]["separation"])
        report["recommended"] = {"trigger_source": best,
                                 ("trigger_rads" if best == "gyro" else "trigger_accel_ms2"):
                                     ok[best]["suggested_threshold"]}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Choose the IMU swing trigger from practice swings")
    ap.add_argument("--csv", help="IMU CSV from dckft.imu.reader")
    ap.add_argument("--record", type=float, help="record this many seconds first (needs the radio)")
    ap.add_argument("--port", help="serial port for --record")
    ap.add_argument("--config", help="dckft config (IMU format / scaling)")
    ap.add_argument("--json", action="store_true", help="print the raw report as JSON")
    a = ap.parse_args(argv)
    path = a.csv
    if a.record:
        from .imu.reader import CsvSink, run_serial
        cfg = load_config(a.config).imu
        if a.port:
            cfg.port = a.port
        path = path or "imu_trigger_tune.csv"
        print(f"Recording {a.record:.0f} s to {path}: stand still, then 3-5 swings holding the "
              "finish, with some waggles/walking in between.")
        run_serial(cfg, [CsvSink(path, cfg)], duration_s=a.record)
    if not path:
        ap.error("give --csv or --record")
    imu = load_imu_csv(path)
    rep = analyse(imu.t, imu.gyro, imu.accel)
    if a.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"\n{rep['swings_found']} swing(s) found at {rep['rate_hz']} Hz in {path}")
    if not rep["swings_found"]:
        print("No swings (wrist rate never passed 15 rad/s). Record again with full-speed swings.")
        return 1
    print(f"{'source':10s} {'min swing peak':>15s} {'max elsewhere':>14s} {'separation':>11s}  suggested")
    for src, r in rep["sources"].items():
        lo = min(r["swing_peaks"]) if r["swing_peaks"] else float("nan")
        print(f"{src:10s} {lo:15.1f} {r['max_outside_swings']:14.1f} "
              f"{(r['separation'] if r['separation'] is not None else float('nan')):11.2f}  "
              f"{r['suggested_threshold'] if r['suggested_threshold'] is not None else '-'} {r['units']}")
    if "recommended" in rep:
        print("\nPut this in the swing section of dckft_config.json:")
        print(json.dumps(rep["recommended"], indent=2))
    else:
        print("\nNo signal separates swings from the other motion; record again with fewer waggles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
