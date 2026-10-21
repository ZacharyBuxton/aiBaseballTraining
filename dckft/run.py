"""
run.py -- the one entry point for the DC-KFT pipeline.

  # Offline: process a recorded session (camera JSONL from track_bat.py + IMU CSV from the reader)
  python -m dckft.run process --cam sessions/s1/camera.jsonl --imu sessions/s1/imu.csv --out sessions/s1/out

  # Live on the Jetson (start track_bat.py and dckft.imu.reader first; see docs/RUNBOOK.md)
  python -m dckft.run live --session-dir sessions

  # No hardware: simulate a swing and run it through the full pipeline
  python -m dckft.run simulate --out sessions/sim --seed 3

  # Accuracy / latency check against simulated ground truth
  python -m dckft.run validate --n 20

  # Write the default config so it can be edited (bat marker distances etc.)
  python -m dckft.run dump-config > dckft_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

from .config import FusionConfig, load_config
from .data import (CameraData, imu_from_samples, load_camera_jsonl, load_imu_csv,
                   payloads_to_camera, gravity_from_meta)
from .pipeline import NoSwingError, process_swing
from .swing import LiveSwingDetector, detect_swings


def _print_summary(res, out_dir=None):
    m = res.metrics
    print(f"\n=== Swing @ {time.strftime('%H:%M:%S', time.localtime(m.t_contact))} ===")
    print(f"  bat speed      {m.bat_speed_mph:6.1f} mph   (peak {m.peak_bat_speed_mph:.1f})")
    print(f"  hand speed     {m.peak_hand_speed_mph:6.1f} mph")
    print(f"  attack angle   {m.attack_angle_deg:+6.1f} deg")
    print(f"  vertical bat   {m.vertical_bat_angle_deg:+6.1f} deg")
    print(f"  time to contact {m.time_to_contact_s * 1e3:5.0f} ms   "
          f"{'(impact detected)' if m.contact_detected else '(dry swing)'}")
    for n in res.notes:
        print(f"  * {n}")
    d = res.diagnostics()
    print(f"  [sync {d['time_offset_ms']:+.1f} ms, coverage {d['camera_coverage']:.0%}, "
          f"calib {d['calib_rms_axis_deg']:.1f} deg, fused in {d['timings_ms']['total']:.0f} ms]")
    if out_dir:
        print(f"  saved -> {out_dir}")


def cmd_process(a, cfg: FusionConfig) -> int:
    cam = load_camera_jsonl(a.cam, list(cfg.bat.marker_s_m))
    imu = load_imu_csv(a.imu)
    swings = detect_swings(imu.t, imu.gyro, imu.accel_hg, cfg.swing)
    if not swings:
        print("No swings found in the IMU data.")
        return 1
    print(f"Found {len(swings)} swing(s).")
    idxs = range(len(swings)) if a.all else [a.swing]
    out_base = a.out or os.path.join(os.path.dirname(os.path.abspath(a.cam)), "out")
    for i in idxs:
        res = process_swing(cam, imu, cfg, swing_index=i)
        out_dir = out_base if len(idxs) == 1 else os.path.join(out_base, f"swing_{i:02d}")
        res.save(out_dir, cfg.bat)
        _print_summary(res, out_dir)
        if not a.no_render:
            from .render import render_swing
            cfg.render.show = not a.no_show
            cfg.render.save_mp4 = a.mp4
            fig = render_swing(res, cfg.bat, cfg.render, out_dir)
            if fig is not None:
                import matplotlib.pyplot as plt
                plt.show()
    return 0


def cmd_simulate(a, cfg: FusionConfig) -> int:
    from .simulate import SimParams, simulate, write_session
    sim = simulate(SimParams(seed=a.seed, contact=not a.no_contact), cfg)
    paths = write_session(sim, a.out)
    print(f"Simulated session written to {a.out}")
    print("Ground truth:", json.dumps(sim["truth"]["metrics"], indent=2))
    a.cam, a.imu, a.all, a.swing = paths["camera"], paths["imu"], False, -1
    a.out = os.path.join(a.out, "out")
    return cmd_process(a, cfg)


def cmd_validate(a, cfg: FusionConfig) -> int:
    from .validate import run
    s = run(a.n, a.seed, cfg)
    return 0 if (s["pass_speed_5pct"] and s["pass_latency_5s"]) else 1


class _Recorder:
    """Keeps raw live data on disk so every live swing can be re-processed offline."""

    def __init__(self, session_dir):
        os.makedirs(session_dir, exist_ok=True)
        self.dir = session_dir
        self.cam = open(os.path.join(session_dir, "camera.jsonl"), "a", buffering=1)
        from .imu.reader import CsvSink
        self.imu_path = os.path.join(session_dir, "imu.csv")
        self.imu = open(self.imu_path, "a", buffering=1)
        if self.imu.tell() == 0:
            from .imu.frames import CSV_COLUMNS
            self.imu.write(",".join(CSV_COLUMNS) + "\n")
        self.tx0 = self.h0 = None

    def camera(self, line: str):
        self.cam.write(line.strip() + "\n")

    def imu_sample(self, s: dict):
        if self.tx0 is None:
            self.tx0, self.h0 = s["tx_us"], s["ts_ns_host"]
        a, g, h, hf = s["accel"], s["gyro"], s["accel_hg"], s.get("hf", [0, 0, 0])
        self.imu.write(",".join(str(v) for v in [
            s["tx_us"], ((s["tx_us"] - self.tx0) % 2 ** 32) / 1e6, (s["ts_ns_host"] - self.h0) / 1e9,
            4095, *a, *g, *h, *hf, s["ts_ns_host"]]) + "\n")

    def close(self):
        self.cam.close(); self.imu.close()


def cmd_live(a, cfg: FusionConfig) -> int:
    import zmq
    ctx = zmq.Context.instance()
    cam_sock = ctx.socket(zmq.PULL); cam_sock.connect(a.camera_zmq or cfg.camera_zmq)
    imu_sock = ctx.socket(zmq.PULL); imu_sock.connect(a.imu_zmq or cfg.imu_zmq)
    poller = zmq.Poller()
    poller.register(cam_sock, zmq.POLLIN); poller.register(imu_sock, zmq.POLLIN)

    session = os.path.join(a.session_dir, time.strftime("session_%Y%m%d_%H%M%S"))
    rec = _Recorder(session)
    print(f"Live: camera <- {a.camera_zmq or cfg.camera_zmq}, IMU <- {a.imu_zmq or cfg.imu_zmq}")
    print(f"Recording raw data to {session}. Swing when ready; Ctrl+C to quit.")

    # FIFO buffers (~12 s) -- the queues that interleave/align the two streams
    cam_buf = deque(maxlen=120 * 12)
    imu_buf = deque(maxlen=400 * 12)
    det = LiveSwingDetector(cfg.swing)
    gravity = None
    show = not a.no_show
    if show:
        import matplotlib.pyplot as plt
        plt.ion()
    last_check = 0.0
    n_swings = 0
    last_status = time.time()
    try:
        while True:
            events = dict(poller.poll(20))
            if cam_sock in events:
                while True:
                    try:
                        msg = cam_sock.recv_string(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    rec.camera(msg)
                    cam_buf.append(json.loads(msg))
            if imu_sock in events:
                while True:
                    try:
                        msg = imu_sock.recv_string(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    for s in json.loads(msg).get("samples", []):
                        rec.imu_sample(s)
                        imu_buf.append(s)

            now = time.time()
            if now - last_status > 2.0:
                last_status = now
                print(f"\r  buffers: camera {len(cam_buf):5d}  imu {len(imu_buf):5d}  swings {n_swings}",
                      end="", flush=True)
            if now - last_check < 0.05 or len(imu_buf) < 400:
                if show:
                    plt.pause(0.001)
                continue
            last_check = now
            recent = list(imu_buf)[-400 * 4:]
            imu = imu_from_samples(recent)
            sw = det.update(now, imu.t, imu.gyro, imu.accel_hg)
            if sw is None:
                continue
            t_detect = time.perf_counter()
            w0 = sw.t_onset - cfg.swing.pre_roll_s - 0.3
            w1 = sw.t_end + cfg.swing.post_roll_s + 0.3
            imu_w = imu.slice(w0, w1)
            cams = [p for p in cam_buf if w0 - 0.3 <= p["ts_ns_image"] * 1e-9 <= w1 + 0.3]
            if len(cams) < 10:
                print(f"\n[live] swing detected but only {len(cams)} camera frames in the window -- "
                      "is track_bat.py running and publishing?")
                continue
            cam = payloads_to_camera(cams, list(cfg.bat.marker_s_m))
            try:
                res = process_swing(cam, imu_w, cfg)
            except Exception as e:      # keep the session alive on a bad swing
                print(f"\n[live] could not process swing: {e}")
                continue
            n_swings += 1
            out_dir = os.path.join(session, f"swing_{n_swings:02d}")
            res.save(out_dir, cfg.bat)
            _print_summary(res, out_dir)
            if not a.no_render:
                from .render import render_swing
                cfg.render.show = show
                if show:
                    plt.close("all")
                render_swing(res, cfg.bat, cfg.render, out_dir, title=f"Swing {n_swings}")
                if show:
                    plt.pause(0.05)
            latency = time.perf_counter() - t_detect + cfg.swing.post_roll_s
            print(f"  swing end -> render: {latency:.2f} s (spec < 5 s)")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        rec.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DC-KFT pipeline")
    ap.add_argument("--config", help="JSON config (python -m dckft.run dump-config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="process a recorded session")
    p.add_argument("--cam", required=True, help="camera JSONL from Cameras/track_bat.py --jsonl")
    p.add_argument("--imu", required=True, help="IMU CSV from dckft.imu.reader --csv")
    p.add_argument("--out", help="output folder (default: <cam dir>/out)")
    p.add_argument("--swing", type=int, default=-1, help="which swing (default: last)")
    p.add_argument("--all", action="store_true", help="process every swing in the files")

    s = sub.add_parser("simulate", help="simulate a swing and process it")
    s.add_argument("--out", default="sessions/sim")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-contact", action="store_true")

    lv = sub.add_parser("live", help="live: consume track_bat + IMU reader over ZMQ")
    lv.add_argument("--session-dir", default="sessions")
    lv.add_argument("--camera-zmq", help="default tcp://127.0.0.1:5557")
    lv.add_argument("--imu-zmq", help="default tcp://127.0.0.1:5558")

    for q in (p, s, lv):
        q.add_argument("--no-show", action="store_true", help="don't open a window (still saves PNG)")
        q.add_argument("--no-render", action="store_true", help="skip rendering entirely")
        q.add_argument("--mp4", action="store_true", help="also save swing.mp4 (needs ffmpeg)")

    v = sub.add_parser("validate", help="accuracy/latency vs simulated ground truth")
    v.add_argument("--n", type=int, default=20)
    v.add_argument("--seed", type=int, default=100)

    sub.add_parser("dump-config", help="print the default config JSON")

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "dump-config":
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    try:
        return {"process": cmd_process, "simulate": cmd_simulate, "validate": cmd_validate,
                "live": cmd_live}[a.cmd](a, cfg)
    except NoSwingError as e:
        print(f"[dckft] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
