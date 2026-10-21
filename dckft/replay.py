"""
replay.py -- publish a recorded (or simulated) session over ZMQ exactly like the live rig:

    camera payloads -> PUSH tcp://*:5557   (as Cameras/track_bat.py does)
    IMU samples     -> PUSH tcp://*:5558   (as dckft.imu.reader does)

Timestamps are re-based to "now" so dckft.run live sees fresh data. Use it to rehearse the
demo or debug live mode with no hardware:

    python -m dckft.run simulate --out sessions/sim --no-show      # make data (or use a real session)
    python -m dckft.run live --session-dir /tmp/live &               # consumer first
    python -m dckft.replay --cam sessions/sim/camera.jsonl --imu sessions/sim/imu.csv
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def replay(cam_path: str, imu_path: str, cam_bind="tcp://*:5557", imu_bind="tcp://*:5558",
           speed: float = 1.0, loops: int = 1, gap_s: float = 2.0):
    import pandas as pd
    import zmq

    ctx = zmq.Context.instance()
    cs = ctx.socket(zmq.PUSH); cs.setsockopt(zmq.SNDHWM, 2000); cs.bind(cam_bind)
    isock = ctx.socket(zmq.PUSH); isock.setsockopt(zmq.SNDHWM, 4000); isock.bind(imu_bind)

    with open(cam_path) as f:
        cams = [json.loads(l) for l in f if l.strip()]
    df = pd.read_csv(imu_path)
    if "ts_ns_host" in df:
        t_imu = df["ts_ns_host"].to_numpy(np.int64)
    else:
        t_imu = (df["imu_meas_time"].to_numpy(float) * 1e9).astype(np.int64) + cams[0]["ts_ns_image"]
    t_cam = np.array([c["ts_ns_image"] for c in cams], dtype=np.int64)
    t0 = min(t_cam[0], t_imu[0])

    events = [(int(tc - t0), "c", i) for i, tc in enumerate(t_cam)] + \
             [(int(ti - t0), "i", i) for i, ti in enumerate(t_imu)]
    events.sort()
    rows = df.to_dict("records")
    print(f"Replaying {len(cams)} camera frames + {len(rows)} IMU samples "
          f"({events[-1][0] / 1e9:.1f} s) x{loops}. Waiting 1 s for consumers...")
    time.sleep(1.0)
    for loop in range(loops):
        start = time.time_ns()
        batch = []
        for rel, kind, i in events:
            target = start + int(rel / speed)
            while time.time_ns() < target:
                time.sleep(0.0002)
            now_stamp = start + rel           # re-based capture time
            if kind == "c":
                p = dict(cams[i]); p["ts_ns_image"] = p["ts_ns_mono"] = now_stamp
                p["ts_ns_wall"] = time.time_ns()
                try:
                    cs.send_string(json.dumps(p), zmq.NOBLOCK)
                except zmq.Again:
                    pass
            else:
                r = rows[i]
                batch.append({"tx_us": int(r["tx_time_stamp"]), "ts_ns_host": now_stamp,
                              "accel": [r["accel_x"], r["accel_y"], r["accel_z"]],
                              "gyro": [r["omega_x"], r["omega_y"], r["omega_z"]],
                              "accel_hg": [r["accel_x2"], r["accel_y2"], r["accel_z2"]],
                              "hf": [r.get("accel_hf_1", 0), r.get("accel_hf_2", 0), r.get("accel_hf_3", 0)]})
                if len(batch) >= 8:
                    try:
                        isock.send_string(json.dumps({"type": "imu", "samples": batch}), zmq.NOBLOCK)
                    except zmq.Again:
                        pass
                    batch = []
        print(f"loop {loop + 1}/{loops} done")
        time.sleep(gap_s)
    cs.close(); isock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay a session over ZMQ as if live")
    ap.add_argument("--cam", required=True)
    ap.add_argument("--imu", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loops", type=int, default=1)
    a = ap.parse_args(argv)
    replay(a.cam, a.imu, speed=a.speed, loops=a.loops)


if __name__ == "__main__":
    main()
