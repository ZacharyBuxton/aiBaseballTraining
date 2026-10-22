#!/usr/bin/env python3
"""Record a swing, extract camera positions, fuse the IMU, and open a 3D replay.

Run from the Jetson desktop: python3 run_schema.py
First run asks for measured bat geometry (centimetres); subsequent runs reuse it.
Defaults: 600 frames, SVGA/120 FPS, no preview during recording, automatic serial port.
Use --mode 1080p60 for the quality comparison; --frames 1200 for a longer capture.
Use --session PATH to retry processing an existing run_schema session without recording.
Python executables are probed independently. Override --camera-python, --imu-python,
or --fusion-python when the ZED SDK and analysis dependencies live in different environments.
No packages, drivers, power modes, or permissions are changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
MODES = {"120fps": ("SVGA", 120), "1080p60": ("HD1080", 60)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def choose_python(override, probe, label):
    candidates = [override] if override else [
        sys.executable, str(ROOT / ".venv-dckft/bin/python"),
        str(ROOT / ".venv/bin/python"), "python3", "python"]
    errors = []
    for candidate in dict.fromkeys(candidates):
        executable = shutil.which(candidate)
        if not executable:
            continue
        try:
            result = subprocess.run([executable, "-c", probe], cwd=str(ROOT),
                                    text=True, capture_output=True, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
            continue
        if result.returncode == 0:
            return executable
        errors.append(executable + ": " + result.stderr.strip()[-700:])
    raise RuntimeError("No working %s Python.\n%s\nUse --%s-python PATH; "
                       "raw files are not deleted." % (label, "\n".join(errors), label))


def python_json(executable, code, *args):
    result = subprocess.run([executable, "-c", code, *map(str, args)],
                            cwd=str(ROOT), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def validate_geometry(config):
    bat = config["bat"]
    length = float(bat["length_m"])
    markers = bat["marker_s_m"]
    if not math.isfinite(length) or length <= 0 or len(markers) < 2:
        raise ValueError("Positive bat length and at least two marker locations are required.")
    positions = [float(v) for v in markers.values()]
    if any(not math.isfinite(v) or not 0 <= v <= length for v in positions):
        raise ValueError("Marker positions must be between the knob and the bat tip.")
    if len(set(positions)) != len(positions):
        raise ValueError("Markers must have distinct distances along the bat.")
    for key in ("sweet_spot_s_m", "hands_s_m", "imu_s_m"):
        value = float(bat[key])
        if not math.isfinite(value) or not 0 <= value <= length:
            raise ValueError(key + " must lie between knob and tip.")


def get_config(args, fusion_python):
    if args.config.exists():
        config = python_json(fusion_python,
            "import json,sys; from dckft.config import load_config; "
            "print(json.dumps(load_config(sys.argv[1]).to_dict()))", args.config)
    else:
        config = python_json(fusion_python,
            "import json; from dckft.config import FusionConfig; "
            "print(json.dumps(FusionConfig().to_dict()))")
        if args.example_geometry:
            print("EXAMPLE GEOMETRY: metrics are experimental, not calibrated bat measurements.")
        else:
            if not sys.stdin.isatty():
                raise RuntimeError("Supply --config with measured bat geometry, or explicitly "
                                   "use --example-geometry for a software-only trial.")
            print("First-time setup: measure from the END OF THE KNOB, in centimetres.")
            def measure(label):
                while True:
                    try:
                        value = float(input(label + " (cm): ")) / 100.0
                        if not math.isfinite(value) or value < 0:
                            raise ValueError()
                        return value
                    except ValueError:
                        print("Enter a nonnegative number, for example 83.8.")
            bat = config["bat"]
            bat["length_m"] = measure("Bat total length")
            for marker in bat["marker_s_m"]:
                bat["marker_s_m"][marker] = measure(marker + " band CENTRE from knob")
            bat["sweet_spot_s_m"] = measure("Barrel point where speed should be measured")
            bat["hands_s_m"] = measure("Hand reference point from knob")
            bat["imu_s_m"] = measure("Approximate wrist IMU projection along bat from knob")
            validate_geometry(config)
            args.config.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.config, config)
            print("Saved measured configuration:", args.config)
    validate_geometry(config)
    return config


class Worker:
    def __init__(self, name, command, session):
        self.name = name
        self.lines = queue.Queue()
        self.log = open(session / (name + ".log"), "w", buffering=1)
        try:
            self.proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        except OSError:
            self.log.close()
            raise
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.log.write(line)
                self.lines.put(line)
                print("[%s] %s" % (self.name, line.rstrip()), flush=True)
        finally:
            self.proc.stdout.close()
            self.log.close()

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        self.thread.join(timeout=5)
        return self.proc.returncode


def imu_count(path):
    if not path.exists():
        return 0
    with path.open() as f:
        return max(0, sum(1 for _ in f) - 1)


def capture(session, args, camera_python, imu_python, config):
    workers = []
    try:
        imu = Worker("imu", [imu_python, "-u", "-m", "dckft.imu.reader",
            "--port", args.port, "--config", str(session / "fusion_config.json"),
            "--no-zmq", "--csv", str(session / "imu.csv")], session)
        workers.append(imu)
        deadline = time.monotonic() + args.startup_timeout
        while imu_count(session / "imu.csv") < 50:
            if imu.proc.poll() is not None:
                raise RuntimeError("IMU reader exited; see imu.log. Check USB permissions/receiver.")
            if time.monotonic() > deadline:
                raise RuntimeError("No usable IMU samples before timeout; check transmitter/radio.")
            time.sleep(0.1)
        print("IMU samples confirmed. Get into position; camera is opening.", flush=True)
        camera = Worker("camera", [camera_python, "-u", str(ROOT / "Cameras/track_bat.py"),
            "--config", str(session / "camera_config.json"), "--record-only", "--no-zmq",
            "--record", str(session / "camera.svo2"), "--codec", args.codec,
            "--max-frames", str(args.frames)], session)
        workers.append(camera)
        ready = False
        deadline = time.monotonic() + args.startup_timeout
        while camera.proc.poll() is None:
            if imu.proc.poll() is not None:
                raise RuntimeError("IMU stopped during capture; keeping partial recordings.")
            try:
                line = camera.lines.get(timeout=0.1)
                if line.startswith("Recording ") and not ready:
                    ready = True
                    deadline = time.monotonic() + args.frames / MODES[args.mode][1] * 3 + 20
                    print("\nRECORDING NOW: hold stance still ~1 second, then swing.\n", flush=True)
            except queue.Empty:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Camera stalled/timed out; stopping and keeping raw data.")
        if camera.proc.returncode:
            raise RuntimeError("Camera failed; see camera.log.")
    finally:
        for worker in reversed(workers):
            worker.stop()
    if not (session / "camera.svo2").exists() or imu_count(session / "imu.csv") < 50:
        raise RuntimeError("Incomplete recording; raw files retained for inspection.")


SVO_AUDIT = r'''
import json, sys
import pyzed.sl as sl
c = sl.Camera(); p = sl.InitParameters()
p.set_from_svo_file(sys.argv[1]); p.svo_real_time_mode = False
p.depth_mode = sl.DEPTH_MODE.NONE
err = c.open(p)
if err != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(str(err))
ts = []
try:
    fps = float(c.get_camera_information().camera_configuration.fps)
    while True:
        err = c.grab()
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED: break
        if err > sl.ERROR_CODE.SUCCESS: raise RuntimeError(str(err))
        ts.append(int(c.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()))
finally:
    c.close()
with open(sys.argv[2], 'w') as f:
    json.dump({'nominal_fps': fps, 'timestamps_ns': ts}, f)
'''


def timestamp_stats(timestamps, fps):
    gaps = [(b - a) / 1e9 for a, b in zip(timestamps, timestamps[1:])]
    positive = [g for g in gaps if g > 0]
    duration = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0
    return {"saved_frames": len(timestamps), "span_s": duration,
        "effective_fps": (len(timestamps) - 1) / duration if duration > 0 else None,
        "max_gap_ms": max(positive) * 1000 if positive else None,
        "nonincreasing_timestamps": sum(g <= 0 for g in gaps),
        "estimated_missing_slots_within_recording": sum(max(0, round(g * fps) - 1) for g in positive)}


def run_stage(name, command, session):
    worker = Worker(name, command, session)
    try:
        code = worker.proc.wait()
    finally:
        worker.stop()
    if code:
        raise RuntimeError(name + " failed; see " + name + ".log. Raw data is preserved.")


def summarize(session):
    audit = json.loads((session / "camera_timestamps.json").read_text())
    ts = audit["timestamps_ns"]
    stats = {"capture": timestamp_stats(ts, audit["nominal_fps"])}
    counts, paired, observed = {}, 0, 0
    with (session / "camera.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            ids = {p["id"] for p in row["points"]}
            observed += 1
            paired += len(ids) >= 2
            for marker in ids:
                counts[marker] = counts.get(marker, 0) + 1
    with (session / "imu.csv").open() as f:
        rows = list(csv.DictReader(f))
    stats["tracking"] = {"frames_with_positions": observed, "frames_with_two_or_more_markers": paired,
        "positions_per_marker": counts, "fraction_of_saved_frames": observed / len(ts) if ts else 0}
    stats["imu"] = {"samples": len(rows)}
    if len(rows) > 1:
        host = [int(row["ts_ns_host"]) for row in rows]
        duration = (host[-1] - host[0]) / 1e9
        stats["imu"]["host_span_s"] = duration
        stats["imu"]["host_rate_hz"] = (len(host) - 1) / duration if duration > 0 else None
        stats["imu"]["overlap_with_camera_s"] = max(0, (min(host[-1], ts[-1]) -
            max(host[0], ts[0])) / 1e9) if ts else 0
    write_json(session / "quality.json", stats)
    print("\nRecording / tracking quality (not a fusion accuracy guarantee):")
    print(json.dumps(stats, indent=2))
    if paired == 0:
        raise RuntimeError("No frames with two markers: cannot establish the optical bat axis. "
                           "Improve detection and retry --session PATH; data has been preserved.")
    if stats["imu"].get("overlap_with_camera_s", 0) <= 0:
        raise RuntimeError("Camera and IMU timestamps do not overlap; inspect timing before fusion.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="120fps")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--port", help="Default: auto-detect USB receiver")
    ap.add_argument("--config", type=Path, default=ROOT / "run_schema_config.json")
    ap.add_argument("--cam-config", type=Path, help="Optional marker/exposure config; mode overrides resolution/FPS")
    ap.add_argument("--session-root", type=Path, default=ROOT / "sessions")
    ap.add_argument("--session", type=Path, help="Reprocess an existing run_schema session; no capture")
    ap.add_argument("--example-geometry", action="store_true", help="Explicitly use unmeasured defaults for a trial")
    ap.add_argument("--codec", choices=("h265", "h265-lossless", "lossless-cpu"), default="h265")
    ap.add_argument("--mp4", action="store_true", help="Also export 3D animation; requires ffmpeg")
    ap.add_argument("--no-show", action="store_true", help="Save render without opening desktop window")
    ap.add_argument("--startup-timeout", type=float, default=30)
    for role in ("camera", "imu", "fusion"):
        ap.add_argument("--" + role + "-python")
    args = ap.parse_args(argv)
    if args.frames < 2 or args.startup_timeout <= 0:
        ap.error("--frames must be >= 2; --startup-timeout must be positive")
    manifest, session = {}, None
    try:
        camera_python = choose_python(args.camera_python,
            "import pyzed.sl, cv2, numpy; import sys; sys.path.insert(0,'Cameras'); import zed_camera", "camera")
        imu_python = choose_python(args.imu_python,
            "import serial; import dckft.imu.reader", "imu")
        fusion_python = choose_python(args.fusion_python,
            "import dckft.run; import dckft.render; from scipy.interpolate import make_smoothing_spline", "fusion")
        if args.mp4 and not shutil.which("ffmpeg"):
            raise RuntimeError("MP4 needs ffmpeg. Install it or omit --mp4; PNG/replay remain available.")
        if not args.no_show and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("Open Terminal on the Jetson desktop, or use --no-show to save the render.")
        if args.session:
            session = args.session.resolve()
            if not session.is_dir():
                raise RuntimeError("Session folder does not exist.")
            # Preserve original snapshot unless a different measured config is explicitly supplied.
            arguments = argv if argv is not None else sys.argv[1:]
            if not any(v == "--config" or v.startswith("--config=") for v in arguments):
                args.config = session / "fusion_config.json"
            if not args.config.exists():
                raise RuntimeError("Session configuration missing; supply --config PATH.")
            config = get_config(args, fusion_python)
            manifest_path = session / "run_schema.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        else:
            config = get_config(args, fusion_python)
            if not args.port:
                args.port = python_json(imu_python,
                    "import json; from dckft.imu.reader import autodetect_port; print(json.dumps(autodetect_port()))")
            if not args.port:
                raise RuntimeError("No receiver detected. Connect USB and use --port /dev/ttyACM0 if needed.")
            args.session_root.mkdir(parents=True, exist_ok=True)
            session = (args.session_root / datetime.now().strftime("schema_%Y%m%d_%H%M%S_%f")).resolve()
            session.mkdir()
            write_json(session / "fusion_config.json", config)
            camera_config = python_json(camera_python,
                "import json,sys; sys.path.insert(0,'Cameras'); from config import load_config; "
                "print(json.dumps(load_config(sys.argv[1] or None).to_dict()))", args.cam_config or "")
            resolution, fps = MODES[args.mode]
            camera_config["camera"].update(resolution=resolution, fps=fps, depth_mode="NONE")
            # Match our successful auto-exposure trials unless a camera config was supplied.
            if not args.cam_config:
                camera_config["camera"]["exposure_us"] = None
            write_json(session / "camera_config.json", camera_config)
            manifest = {"status": "capturing", "mode": args.mode, "requested_frames": args.frames,
                "port": args.port, "example_geometry": args.example_geometry,
                "interpreters": {"camera": camera_python, "imu": imu_python, "fusion": fusion_python}}
            write_json(session / "run_schema.json", manifest)
            print("Session:", session, flush=True)
            capture(session, args, camera_python, imu_python, config)
        manifest["status"] = "processing"
        write_json(session / "run_schema.json", manifest)
        # New output directory for every attempt: never append duplicate positions or overwrite replay results.
        attempt = session / datetime.now().strftime("analysis_%Y%m%d_%H%M%S_%f")
        attempt.mkdir()
        write_json(attempt / "fusion_config.json", config)
        run_stage("audit", [camera_python, "-u", "-c", SVO_AUDIT,
            str(session / "camera.svo2"), str(attempt / "camera_timestamps.json")], attempt)
        run_stage("detection", [camera_python, "-u", str(ROOT / "Cameras/track_bat.py"),
            "--svo", str(session / "camera.svo2"), "--config", str(session / "camera_config.json"),
            "--no-zmq", "--jsonl", str(attempt / "camera.jsonl")], attempt)
        # summarize expects both raw IMU and analysis files together; read via an explicit argument.
        shutil.copy2(session / "imu.csv", attempt / "imu.csv")
        summarize(attempt)
        command = [fusion_python, "-u", "-m", "dckft.run", "--config", str(attempt / "fusion_config.json"),
            "process", "--cam", str(attempt / "camera.jsonl"), "--imu", str(attempt / "imu.csv"),
            "--all", "--out", str(attempt / "out")]
        if args.no_show:
            command.append("--no-show")
        if args.mp4:
            command.append("--mp4")
        print("\nFusing recorded observations and IMU estimates, then rendering. "
              "Filled gaps are estimates, not measured camera positions.", flush=True)
        run_stage("fusion", command, attempt)
        manifest.update(status="complete", latest_analysis=str(attempt))
        write_json(session / "run_schema.json", manifest)
        print("\nDONE:", attempt / "out")
        print("Metrics: swing.json; trajectory: fused.csv; 3D render: swing.png" +
              (" and swing.mp4" if args.mp4 else ""))
        print("Check sync correlation, calibration error and measured marker coverage before trusting speeds.")
        return 0
    except KeyboardInterrupt:
        message = "Interrupted. Existing recordings are preserved."
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        message = str(exc)
    if session:
        manifest.update(status="incomplete", error=message)
        write_json(session / "run_schema.json", manifest)
        print("Session preserved:", session)
        print("Retry after fixing the issue: python3 run_schema.py --session", session)
    print("run_schema:", message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
