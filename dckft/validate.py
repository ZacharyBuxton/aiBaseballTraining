"""
validate.py -- Monte-Carlo accuracy + latency check against the simulator's ground truth.

    python -m dckft.validate --n 20            # prints a table and the pass/fail vs the <5% spec

Speeds are compared as % error; angles in degrees (a % of an angle near 0 is meaningless);
times in ms. The <5% tracking-error spec is applied to bat speed and hand speed, and the
<5 s render-latency spec to the processing time.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import FusionConfig
from .data import payloads_to_camera, gravity_from_meta
from .pipeline import process_swing
from .simulate import SimParams, simulate, write_session
from .data import load_camera_jsonl, load_imu_csv

PCT = ["bat_speed_mph", "peak_bat_speed_mph", "peak_hand_speed_mph"]
DEGS = ["attack_angle_deg", "vertical_bat_angle_deg", "swing_plane_tilt_deg"]
MS = ["time_to_contact_s"]


def run(n: int = 20, seed0: int = 100, cfg: FusionConfig = None, tmp_dir: str = None,
        params: dict = None, verbose: bool = True) -> dict:
    import tempfile
    cfg = cfg or FusionConfig()
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="dckft_val_")
    rows = []
    for i in range(n):
        sim = simulate(SimParams(seed=seed0 + i, **(params or {})), cfg)
        paths = write_session(sim, f"{tmp_dir}/s{i}")
        cam, imu = load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"])
        res = process_swing(cam, imu, cfg)
        m, tr = res.metrics.to_dict(), sim["truth"]["metrics"]
        row = {"seed": seed0 + i, "total_ms": res.timings_ms["total"]}
        for k in PCT:
            row[k] = 100 * (m[k] - tr[k]) / tr[k]
        for k in DEGS:
            row[k] = m[k] - tr[k]
        for k in MS:
            row[k] = 1e3 * (m[k] - tr[k])
        rows.append(row)
        if verbose:
            print(f"seed {row['seed']}: bat {row['bat_speed_mph']:+5.1f}%  peak {row['peak_bat_speed_mph']:+5.1f}%  "
                  f"hand {row['peak_hand_speed_mph']:+5.1f}%  AA {row['attack_angle_deg']:+5.1f}°  "
                  f"VBA {row['vertical_bat_angle_deg']:+5.1f}°  TTC {row['time_to_contact_s']:+5.1f}ms  "
                  f"{row['total_ms']:.0f} ms")
    keys = PCT + DEGS + MS
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "mae": float(np.mean([abs(r[k]) for r in rows])),
                   "p95_abs": float(np.percentile([abs(r[k]) for r in rows], 95))} for k in keys}
    summary["latency_ms"] = {"mean": float(np.mean([r["total_ms"] for r in rows])),
                             "max": float(np.max([r["total_ms"] for r in rows]))}
    summary["pass_speed_5pct"] = all(summary[k]["mae"] < 5.0 for k in PCT)
    summary["pass_latency_5s"] = summary["latency_ms"]["max"] < 5000
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate fusion accuracy on simulated swings")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", help="write the summary JSON here")
    a = ap.parse_args(argv)
    s = run(a.n, a.seed)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=2)


if __name__ == "__main__":
    main()
