"""
track_bat.py

Optical capture subsystem entry point: tracks colored markers on the bat with the
ZED X and publishes their 3D positions to the fusion engine.

Pipeline per frame:
    grab rectified L/R  ->  detect markers in both  ->  3D position per marker
    ->  JSON payload  ->  ZeroMQ PUSH (tcp://*:5557)  [+ optional JSONL log]

3D positions come from one of two sources:
    --depth-source stereo (default)  triangulate the marker centroids directly.
                                     No neural depth, so it's cheap enough for 120 fps.
    --depth-source sdk               look up the ZED SDK point cloud at each left-image
                                     centroid (NEURAL_LIGHT unless the config says otherwise).
                                     Heavier; mainly useful to cross-check the stereo path.

The same code runs on the live camera or on a recorded .svo2 (--svo). If live
processing can't keep up with 120 fps, record first and process the recording:

    python Cameras/track_bat.py --record reports/swing01.svo2 --no-zmq     # capture
    python Cameras/track_bat.py --svo reports/swing01.svo2 --jsonl reports/swing01.jsonl

Other examples:
    python Cameras/track_bat.py --preview                    # live, publish on :5557
    python Cameras/track_bat.py --preview --exposure-us 800 --save-config cage.json
    python Cameras/track_bat.py --config cage.json --jsonl reports/session.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, save_config                                # noqa: E402
from marker_detector import ColorMarkerDetector, draw_detections            # noqa: E402
from payload import FramePublisher, JsonlWriter, build_frame_payload        # noqa: E402
from stereo import match_and_triangulate, print_depth_budget, sample_xyz    # noqa: E402
from zed_camera import CameraError, ZedCamera                               # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ZED X bat-marker tracker")
    p.add_argument("--config", help="JSON config (see config.py)")
    p.add_argument("--save-config", help="Write the effective config to this path and continue")
    p.add_argument("--svo", help="Process a recorded .svo2 instead of the live camera")
    p.add_argument("--record", help="Record the live session to this .svo2 path")
    p.add_argument("--codec", default="h265", choices=sorted(ZedCamera.RECORD_CODECS),
                   help="SVO codec (use lossless-cpu on an Orin Nano)")
    p.add_argument("--depth-source", choices=("stereo", "sdk"), default="stereo")
    p.add_argument("--exposure-us", type=int, help="Manual exposure in microseconds")
    p.add_argument("--analog-gain", type=int, help="Analog gain in mdB")
    p.add_argument("--zmq-bind", help="Override the PUSH bind address")
    p.add_argument("--no-zmq", action="store_true", help="Don't open the ZMQ socket")
    p.add_argument("--jsonl", help="Append every payload to this JSONL file")
    p.add_argument("--camera-id", help="camera_id field in payloads")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    p.add_argument("--preview", action="store_true", help="Show annotated left/right images")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.exposure_us is not None:
        cfg.camera.exposure_us = args.exposure_us
    if args.analog_gain is not None:
        cfg.camera.analog_gain_mdb = args.analog_gain
    if args.depth_source == "sdk" and cfg.camera.depth_mode.upper() == "NONE":
        cfg.camera.depth_mode = "NEURAL_LIGHT"
    if args.depth_source == "stereo":
        cfg.camera.depth_mode = "NONE"
    if args.zmq_bind:
        cfg.output.zmq_bind = args.zmq_bind
    if args.no_zmq:
        cfg.output.zmq_bind = None
    if args.jsonl:
        cfg.output.jsonl_path = args.jsonl
    if args.camera_id:
        cfg.output.camera_id = args.camera_id
    return cfg


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    if args.save_config:
        save_config(cfg, args.save_config)
        print(f"Saved config to {args.save_config}")

    cam = ZedCamera(cfg.camera, svo_path=args.svo)
    try:
        cam.open()
    except CameraError as e:
        print(f"[FAIL] {e}")
        return 1

    publisher = jsonl = None
    n_frames = n_published = 0
    hits = {m: 0 for m in cfg.markers.hues}
    try:
        info = cam.info()
        calib = cam.calibration()
        print(f"Camera: {info['model']} S/N {info['serial_number']} | "
              f"{info['resolution'][0]}x{info['resolution'][1]} @ {info['fps']:.0f} fps | "
              f"source={info['source']} | 3D from {args.depth_source}")
        if not args.svo:
            print(f"Exposure/gain: {cam.read_settings()}")
        if not args.quiet:
            print_depth_budget(calib, distances=(2.0, 3.0, 4.0))

        gravity = cam.estimate_gravity(0.5) if not args.svo else None

        if cfg.output.zmq_bind:
            publisher = FramePublisher(cfg.output.zmq_bind)
            print(f"Publishing on {cfg.output.zmq_bind} (PUSH)")
        if cfg.output.jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(cfg.output.jsonl_path)), exist_ok=True)
            jsonl = JsonlWriter(cfg.output.jsonl_path)
            with open(cfg.output.jsonl_path + ".meta.json", "w") as f:
                json.dump({"camera": info, "calibration": calib.to_dict(), "gravity": gravity,
                           "config": cfg.to_dict(), "depth_source": args.depth_source,
                           "svo": args.svo, "record": args.record}, f, indent=2, default=str)
            print(f"Logging payloads to {cfg.output.jsonl_path}")

        if args.record:
            if args.svo:
                print("[WARN] --record is ignored when replaying an SVO")
            else:
                os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
                cam.start_recording(args.record, codec=args.codec)
                print(f"Recording to {args.record} ({args.codec})")

        det_left = ColorMarkerDetector(cfg.markers)
        det_right = ColorMarkerDetector(cfg.markers)
        period_s = 1.0 / max(1.0, info["fps"])
        proc_ms = deque(maxlen=240)
        ts_hist = deque(maxlen=240)
        drops_start = cam.dropped_frames()
        last_report = time.monotonic()
        print("Running. Ctrl+C (or q in the preview) to stop.")

        while True:
            frame = cam.grab()
            if frame is None:
                print("End of SVO.")
                break
            t0 = time.perf_counter()
            n_frames += 1
            ts_hist.append(frame.ts_ns)

            left_dets = det_left.detect(frame.left)
            right_dets = det_right.detect(frame.right)

            points = {}
            if args.depth_source == "stereo":
                matched = match_and_triangulate(left_dets, right_dets, calib,
                                                cfg.markers.max_row_mismatch_px,
                                                cfg.camera.depth_min_m, cfg.camera.depth_max_m)
                for m, sp in matched.items():
                    points[m] = {"xyz": sp.xyz,
                                 "disparity_px": round(sp.disparity_px, 3),
                                 "row_err_px": round(sp.row_mismatch_px, 3),
                                 "uv_left": [round(sp.uv_left[0], 2), round(sp.uv_left[1], 2)],
                                 "uv_right": [round(sp.uv_right[0], 2), round(sp.uv_right[1], 2)]}
            else:
                for m, d in left_dets.items():
                    xyz = sample_xyz(frame.xyz, d.u, d.v)
                    if xyz is not None and cfg.camera.depth_min_m <= xyz[2] <= cfg.camera.depth_max_m:
                        points[m] = {"xyz": xyz, "uv_left": [round(d.u, 2), round(d.v, 2)]}

            for m in points:
                hits[m] += 1

            if len(points) >= cfg.output.min_points_to_publish:
                payload = build_frame_payload(cfg.output.camera_id, frame.frame_id, frame.ts_ns, points,
                                              extra={"depth_source": args.depth_source})
                if publisher:
                    publisher.publish(payload)
                if jsonl:
                    jsonl.write(payload)
                n_published += 1

            proc_ms.append((time.perf_counter() - t0) * 1e3)

            if args.preview:
                vis_l, vis_r = frame.left.copy(), frame.right.copy()
                draw_detections(vis_l, left_dets)
                draw_detections(vis_r, right_dets)
                for m, p in points.items():
                    u, v = p["uv_left"]
                    cv.putText(vis_l, f"z={p['xyz'][2]:.2f}m", (int(u) + 6, int(v) + 12),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA)
                vis = np.hstack([vis_l, vis_r])
                cv.putText(vis, f"frame {frame.frame_id}  proc {np.mean(proc_ms):.1f} ms  "
                                f"markers {len(points)}", (10, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                cv.imshow("track_bat  (left | right)", vis)
                if (cv.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

            now = time.monotonic()
            if not args.quiet and now - last_report >= 1.0:
                last_report = now
                fps = 0.0
                if len(ts_hist) > 1:
                    fps = (len(ts_hist) - 1) / ((ts_hist[-1] - ts_hist[0]) / 1e9)
                mean_proc = float(np.mean(proc_ms))
                msg = (f"fps {fps:6.1f} | proc {mean_proc:5.1f} ms | "
                       f"dropped {cam.dropped_frames() - drops_start} | published {n_published}")
                if publisher:
                    msg += f" (zmq sent {publisher.sent}, no-consumer drops {publisher.dropped})"
                print(msg)
                if not args.svo and not args.preview and mean_proc > period_s * 1e3:
                    print(f"  [WARN] processing ({mean_proc:.1f} ms) is slower than the frame "
                          f"period ({period_s * 1e3:.1f} ms); record first and process the SVO.")

            if args.max_frames and n_frames >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if publisher:
            publisher.close()
        if jsonl:
            jsonl.close()
        if args.preview:
            cv.destroyAllWindows()

    if n_frames:
        print(f"\nFrames: {n_frames} | published: {n_published}")
        for m, h in hits.items():
            print(f"  {m:12s} 3D fixes: {h:6d} ({100.0 * h / n_frames:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
