"""
backend.py -- what the app's buttons drive.

Two sources feed the SAME ring buffer + fusion + render path:

  hardware  starts the IMU reader and the camera tracker (live pink detection) as background
            processes. IMU samples go through the advisor's ring buffer (dckft.ring); each
            completed window (3 s before + 3 s after the |omega| trigger) is cut together with
            the camera frames over the same span, fused, rendered and saved as a swing.
  demo      "Simulate swing" makes a synthetic swing (dckft.simulate) and pushes its IMU samples
            through the same ring buffer, so the app can be tested with no camera or IMU.

Each swing is a folder in app_data/swings/<id>/: the raw window (imu.csv, camera.jsonl), the
fusion outputs (swing.json, fused.csv, markers*.csv, camera_orientation.csv) and swing.png.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("MPLBACKEND", "Agg")

from dckft.config import FusionConfig, load_config          # noqa: E402
from dckft.data import imu_from_samples                       # noqa: E402
from dckft.imu.frames import TxClock                          # noqa: E402
from dckft.ring import CameraRing                             # noqa: E402
from dckft.run import _Recorder, make_ring, process_window    # noqa: E402

DATA = os.path.join(REPO, "app_data")
SWINGS = os.path.join(DATA, "swings")


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:                    # git missing / offline
        return f"(git unavailable: {e})"


def git_info() -> dict:
    return {"commit": git("rev-parse", "--short", "HEAD"), "subject": git("log", "-1", "--format=%s"),
            "date": git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD")}


def git_pull() -> str:
    out = subprocess.run(["git", "pull", "--ff-only"], cwd=REPO, capture_output=True, text=True,
                         timeout=120)
    return (out.stdout + out.stderr).strip() or "(no output)"


class Backend:
    def __init__(self, config_path: Optional[str] = None, imu_port: Optional[str] = None,
                 cam_config: Optional[str] = None):
        os.makedirs(SWINGS, exist_ok=True)
        self.config_path = config_path
        self.cfg: FusionConfig = load_config(config_path)
        self.imu_port, self.cam_config = imu_port, cam_config
        self.lock = threading.Lock()
        self.mode = "idle"                 # idle | hardware | demo
        self.message = "Ready."
        self.log = deque(maxlen=200)
        self.busy = False                  # processing a swing
        self.procs = []
        self.stop_evt = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.ring = None
        self.n_frames = {"imu": 0, "camera": 0}

    # ------------------------------------------------------------------ status
    def say(self, msg: str) -> None:
        self.message = msg
        self.log.append(time.strftime("%H:%M:%S ") + msg)
        print(msg, flush=True)

    def status(self) -> dict:
        return {"mode": self.mode, "message": self.message, "busy": self.busy,
                "ring": self.ring.progress if self.ring is not None else None,
                "frames": self.n_frames, "log": list(self.log)[-12:],
                "config": self.config_path or "(defaults: example bat geometry)",
                "trigger": self._trigger_text()}

    def _trigger_text(self) -> str:
        s = self.cfg.swing
        if s.trigger_source == "gyro":
            v = f"|omega| > {s.trigger_rads:g} rad/s"
        else:
            v = f"{s.trigger_source} > {s.trigger_accel_ms2:g} m/s^2"
        return (f"{v}; {s.ring_pre_frames} frames before + {s.ring_post_frames} after "
                f"({(s.ring_pre_frames + s.ring_post_frames) / self.cfg.imu.nominal_rate_hz:.0f} s)")

    # ------------------------------------------------------------------ swings
    def list_swings(self) -> list:
        out = []
        for sid in sorted(os.listdir(SWINGS), reverse=True):
            d = os.path.join(SWINGS, sid)
            try:
                with open(os.path.join(d, "meta.json")) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            m = {}
            try:
                with open(os.path.join(d, "swing.json")) as f:
                    m = json.load(f).get("metrics", {})
            except (OSError, ValueError):
                pass
            out.append({"id": sid, **meta, "attack_angle_deg": m.get("attack_angle_deg"),
                        "vertical_bat_angle_deg": m.get("vertical_bat_angle_deg"),
                        "bat_speed_mph": m.get("bat_speed_mph")})
        return out

    def swing_detail(self, sid: str) -> Optional[dict]:
        d = os.path.join(SWINGS, os.path.basename(sid))
        if not os.path.isdir(d):
            return None
        detail = {"id": sid, "files": sorted(os.listdir(d))}
        for name in ("meta.json", "swing.json"):
            try:
                with open(os.path.join(d, name)) as f:
                    detail[name.split(".")[0]] = json.load(f)
            except (OSError, ValueError):
                pass
        return detail

    def delete_swing(self, sid: str) -> bool:
        d = os.path.join(SWINGS, os.path.basename(sid))
        if os.path.isdir(d):
            shutil.rmtree(d)
            return True
        return False

    def _save_window(self, imu_samples: list, payloads: list, t_trigger: float, source: str,
                     gravity=None) -> Optional[str]:
        """Fuse + render one ring-buffer window and store it as a swing folder."""
        from dckft.render import render_swing
        self.busy = True
        try:
            sid = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            d = os.path.join(SWINGS, sid)
            rec = _Recorder(d)                       # raw window, same formats as a session
            for p in payloads:
                rec.camera(json.dumps(p))
            for s in imu_samples:
                rec.imu_sample(s)
            rec.close()
            meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source,
                    "imu_frames": len(imu_samples), "camera_frames": len(payloads),
                    "git": git_info()["commit"], "config": self.config_path or "defaults",
                    "status": "processing"}
            self._write_meta(d, meta)
            imu_w = imu_from_samples(imu_samples)
            res = process_window(imu_w, payloads, self.cfg, gravity, t_trigger)
            if res is None:
                meta["status"] = "failed: not enough camera frames or no usable swing (see log)"
                self._write_meta(d, meta)
                self.say(f"Swing {sid}: could not be processed")
                return sid
            res.save(d, self.cfg.bat)
            self.cfg.render.show = False
            render_swing(res, self.cfg.bat, self.cfg.render, d, title=f"Swing {sid}")
            m = res.metrics
            meta["status"] = "ok"
            self._write_meta(d, meta)
            fmt = lambda v: f"{v:+.1f}°" if v is not None and np.isfinite(v) else "n/a"
            self.say(f"Swing {sid} saved: attack angle {fmt(m.attack_angle_deg)}, "
                     f"vertical bat angle {fmt(m.vertical_bat_angle_deg)}")
            return sid
        except Exception as e:
            self.say(f"Processing failed: {e}")
            traceback.print_exc()
            return None
        finally:
            self.busy = False

    @staticmethod
    def _write_meta(d, meta):
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------ demo
    def demo_swing(self) -> Optional[str]:
        """Simulated swing through the real ring buffer -> fusion -> render path."""
        from dckft.simulate import SimParams, simulate
        if self.mode == "hardware":
            self.say("Stop hardware recording before simulating.")
            return None
        if self.busy:
            self.say("Still processing the previous swing.")
            return None
        self.mode = "demo"
        try:
            pre = self.cfg.swing.ring_pre_frames / self.cfg.imu.nominal_rate_hz
            post = self.cfg.swing.ring_post_frames / self.cfg.imu.nominal_rate_hz
            seed = int(time.time()) % 100000
            self.say(f"Simulating a swing (seed {seed}) ...")
            sim = simulate(SimParams(seed=seed, stance_s=pre + 0.6, tail_s=post + 0.6), self.cfg)
            ring = make_ring(self.cfg)
            self.ring = ring
            cam = CameraRing(seconds=pre + post + 10)
            for p in sim["payloads"]:
                cam.push(p)
            clock = TxClock()
            win = None
            for s, h in zip(sim["imu_rows"], sim["imu_host"]):
                d = s.to_dict()
                d["ts_ns_host"] = int(round(h * 1e9))
                d["t_dev_s"] = clock(int(d["tx_us"])) * 1e-6
                win = ring.push(d) or win
            if win is None:
                self.say("Simulated swing did not trigger the ring buffer.")
                return None
            imu_w = imu_from_samples(win.samples)
            sid = self._save_window(win.samples, cam.cut(imu_w.t[0] - 0.3, imu_w.t[-1] + 0.3),
                                    float(imu_w.t[win.trigger_index]), "demo (simulated)")
            return sid
        finally:
            self.mode = "idle"

    # ------------------------------------------------------------------ hardware
    def start_hardware(self) -> bool:
        if self.mode != "idle":
            self.say(f"Already running ({self.mode}).")
            return False
        self.stop_evt.clear()
        self.mode = "hardware"
        self.thread = threading.Thread(target=self._hardware_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        if self.mode == "hardware":
            self.stop_evt.set()
            if self.thread:
                self.thread.join(timeout=30)
        self.mode = "idle"

    def _spawn(self, log_dir):
        py = sys.executable
        imu_cmd = [py, "-u", "-m", "dckft.imu.reader", "--zmq-bind", "tcp://*:5558", "--quiet",
                   "--csv", os.path.join(log_dir, "imu_session.csv")]
        cam_cmd = [py, "-u", os.path.join(REPO, "Cameras", "track_bat.py"), "--quiet",
                   "--zmq-bind", "tcp://*:5557", "--jsonl", os.path.join(log_dir, "camera_session.jsonl")]
        if self.imu_port:
            imu_cmd += ["--port", self.imu_port]
        if self.config_path:
            imu_cmd += ["--config", self.config_path]
            cam_cmd += ["--bat-config", self.config_path]
        if self.cam_config:
            cam_cmd += ["--config", self.cam_config]
        for name, cmd in (("imu", imu_cmd), ("camera", cam_cmd)):
            log = open(os.path.join(log_dir, name + ".log"), "w")
            self.procs.append((name, subprocess.Popen(cmd, cwd=REPO, stdout=log,
                                                      stderr=subprocess.STDOUT,
                                                      start_new_session=True), log))

    def _hardware_loop(self):
        import zmq
        from dckft.data import gravity_from_meta
        session = os.path.join(DATA, "sessions", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(session, exist_ok=True)
        self.n_frames = {"imu": 0, "camera": 0}
        try:
            self._spawn(session)
            self.say("Starting IMU reader and camera ... (logs in app_data/sessions/)")
            ctx = zmq.Context.instance()
            cam_sock = ctx.socket(zmq.PULL); cam_sock.connect("tcp://127.0.0.1:5557")
            imu_sock = ctx.socket(zmq.PULL); imu_sock.connect("tcp://127.0.0.1:5558")
            poller = zmq.Poller()
            poller.register(cam_sock, zmq.POLLIN); poller.register(imu_sock, zmq.POLLIN)
            ring = make_ring(self.cfg)
            self.ring = ring
            sw = self.cfg.swing
            cam_ring = CameraRing(seconds=(sw.ring_pre_frames + sw.ring_post_frames) /
                                  self.cfg.imu.nominal_rate_hz + 6)
            clock = TxClock()
            gravity = None
            meta_path = os.path.join(session, "camera_session.jsonl.meta.json")
            pending = []
            t_start = time.time()
            said_ready = False
            while not self.stop_evt.is_set():
                dead = [n for n, p, _ in self.procs if p.poll() is not None]
                if dead:
                    self.say(f"{' and '.join(dead)} stopped -- see its log in {session}")
                    break
                events = dict(poller.poll(50))
                if cam_sock in events:
                    while True:
                        try:
                            cam_ring.push(json.loads(cam_sock.recv_string(zmq.NOBLOCK)))
                            self.n_frames["camera"] += 1
                        except zmq.Again:
                            break
                if imu_sock in events:
                    while True:
                        try:
                            msg = imu_sock.recv_string(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        for smp in json.loads(msg).get("samples", []):
                            self.n_frames["imu"] += 1
                            smp["t_dev_s"] = clock(int(smp["tx_us"])) * 1e-6
                            win = ring.push(smp)
                            if win is not None:
                                self.say("Swing triggered -- processing")
                                pending.append((win, time.time()))
                if not said_ready and self.n_frames["imu"] > 400:
                    said_ready = True
                    self.say("Recording. Hold the stance ~1 s, swing, hold the finish.")
                if not said_ready and time.time() - t_start > 15:
                    self.say("No IMU data after 15 s -- check the receiver / port (imu.log).")
                    t_start = time.time()
                if gravity is None and os.path.exists(meta_path):
                    try:
                        with open(meta_path) as f:
                            gravity = gravity_from_meta(json.load(f))
                    except (OSError, ValueError):
                        pass
                if pending:
                    win, t_done = pending[0]
                    imu_w = imu_from_samples(win.samples)
                    if cam_ring.latest_t() >= imu_w.t[-1] or time.time() - t_done > 1.5:
                        pending.pop(0)
                        self._save_window(win.samples, cam_ring.cut(imu_w.t[0] - 0.3, imu_w.t[-1] + 0.3),
                                          float(imu_w.t[win.trigger_index]), "hardware", gravity)
        except Exception as e:
            self.say(f"Hardware loop error: {e}")
            traceback.print_exc()
        finally:
            for name, p, log in self.procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            for name, p, log in self.procs:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.terminate()
                log.close()
            self.procs = []
            self.ring = None
            self.mode = "idle"
            self.say("Recording stopped.")
