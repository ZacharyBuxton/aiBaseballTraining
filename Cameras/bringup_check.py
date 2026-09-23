"""
bringup_check.py

First-power-on check for the ZED X + ZED Link Duo. Run this before anything else.

It opens the camera, prints what the SDK actually configured, applies manual
exposure, measures the real frame rate from hardware image timestamps, counts
dropped frames, estimates gravity from the camera IMU, and prints the stereo
depth-resolution budget from the factory calibration. Results go to
reports/bringup_<time>/ (reports/ is gitignored).

Examples:
    python Cameras/bringup_check.py                      # 10 s at SVGA/120, 500 us exposure
    python Cameras/bringup_check.py --seconds 20 --exposure-us 800 --preview
    python Cameras/bringup_check.py --record --codec lossless-cpu   # Orin Nano
    python Cameras/bringup_check.py --svo reports/.../test.svo2      # re-check a recording
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config                      # noqa: E402
from stereo import print_depth_budget               # noqa: E402
from zed_camera import CameraError, ZedCamera       # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ZED X bring-up check")
    p.add_argument("--config", help="JSON config (see config.py)")
    p.add_argument("--svo", help="Check a recorded .svo2 instead of the live camera")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--resolution", help="Override: SVGA | HD1080 | HD1200")
    p.add_argument("--fps", type=int, help="Override frame rate")
    p.add_argument("--exposure-us", type=int, help="Manual exposure in microseconds")
    p.add_argument("--auto-exposure", action="store_true", help="Use auto exposure/gain")
    p.add_argument("--analog-gain", type=int, help="Analog gain in mdB (default range 1000-16000)")
    p.add_argument("--record", action="store_true", help="Record an SVO during the test")
    p.add_argument("--codec", default="h265", choices=sorted(ZedCamera.RECORD_CODECS))
    p.add_argument("--preview", action="store_true", help="Show the left/right images")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cam_cfg = cfg.camera
    cam_cfg.depth_mode = "NONE"          # bring-up measures capture only
    if args.resolution:
        cam_cfg.resolution = args.resolution
    if args.fps:
        cam_cfg.fps = args.fps
    if args.exposure_us is not None:
        cam_cfg.exposure_us = args.exposure_us
    if args.auto_exposure:
        cam_cfg.exposure_us = cam_cfg.analog_gain_mdb = cam_cfg.digital_gain = None
    if args.analog_gain is not None:
        cam_cfg.analog_gain_mdb = args.analog_gain

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo_root, "reports",
                                           time.strftime("bringup_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    report = {"config": cfg.to_dict()}

    cam = ZedCamera(cam_cfg, svo_path=args.svo)
    try:
        cam.open()
    except CameraError as e:
        print(f"[FAIL] {e}")
        return 1

    try:
        info = cam.info()
        report["camera"] = info
        print("\n== Camera ==")
        for k, v in info.items():
            print(f"  {k:14s}: {v}")

        if not args.svo:
            settings = cam.read_settings()
            report["settings_readback"] = settings
            print("\n== Exposure/gain readback ==")
            for k, v in settings.items():
                print(f"  {k:14s}: {v}")

        calib = cam.calibration()
        report["calibration"] = calib.to_dict()
        print("\n== Stereo calibration / depth budget ==")
        print_depth_budget(calib)

        if not args.svo:
            print("\n== Camera IMU (keep the camera still) ==")
            grav = cam.estimate_gravity(1.0)
            report["gravity"] = grav
            if grav:
                a = np.array(grav["accel_mean_mps2"])
                print(f"  accel mean: {np.round(a, 3)} m/s^2  |a|={np.linalg.norm(a):.3f}  "
                      f"rate~{grav['rate_hz']:.0f} Hz")
            else:
                print("  [WARN] no IMU samples")

        if args.record:
            svo_path = os.path.join(out_dir, "bringup.svo2")
            cam.start_recording(svo_path, codec=args.codec)
            report["recording"] = {"path": svo_path, "codec": args.codec}
            print(f"\nRecording to {svo_path} ({args.codec})")

        print(f"\n== Capturing for {args.seconds:.0f} s ==")
        drops_start = cam.dropped_frames()
        timestamps = []
        first = None
        t_end = time.monotonic() + args.seconds
        while args.svo or time.monotonic() < t_end:
            frame = cam.grab()
            if frame is None:
                break
            timestamps.append(frame.ts_ns)
            if first is None:
                first = frame
            if args.preview:
                cv.imshow("left | right", np.hstack([frame.left, frame.right]))
                if (cv.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
        drops = cam.dropped_frames() - drops_start

        if first is not None:
            cv.imwrite(os.path.join(out_dir, "left.png"), first.left)
            cv.imwrite(os.path.join(out_dir, "right.png"), first.right)
            sbs = np.hstack([first.left, first.right])
            for y in range(0, sbs.shape[0], 40):     # rows should line up across both halves
                cv.line(sbs, (0, y), (sbs.shape[1], y), (0, 255, 0), 1)
            cv.imwrite(os.path.join(out_dir, "rectification_check.png"), sbs)

        ts = np.asarray(timestamps, dtype=np.int64)
        stats = {"frames": int(ts.size), "sdk_dropped_frames": int(drops)}
        if ts.size > 2:
            dt = np.diff(ts) / 1e9
            expected = 1.0 / info["fps"]
            stats.update({
                "measured_fps": float(1.0 / np.median(dt)),
                "mean_fps": float((ts.size - 1) / ((ts[-1] - ts[0]) / 1e9)),
                "dt_ms_median": float(np.median(dt) * 1e3),
                "dt_ms_p99": float(np.percentile(dt, 99) * 1e3),
                "dt_ms_max": float(dt.max() * 1e3),
                "timestamp_gaps": int(np.sum(dt > 1.5 * expected)),
                "nonmonotonic_timestamps": int(np.sum(dt <= 0)),
            })
        report["capture"] = stats

        print("\n== Result ==")
        for k, v in stats.items():
            print(f"  {k:24s}: {v:.3f}" if isinstance(v, float) else f"  {k:24s}: {v}")

        ok = ts.size > 2
        if ok and not args.svo:
            ok = stats["measured_fps"] >= 0.95 * cam_cfg.fps and stats["timestamp_gaps"] == 0
        report["pass"] = bool(ok)
        print(f"\n{'[PASS]' if ok else '[CHECK]'} results saved to {out_dir}")
        if not ok and ts.size > 2:
            print("  Lower fps or gaps usually mean: wrong resolution for 120 fps, recording "
                  "load (try --codec lossless-cpu or no --record), or GMSL cable/power issues.")
    finally:
        cam.close()
        if args.preview:
            cv.destroyAllWindows()
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)

    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    sys.exit(main())
