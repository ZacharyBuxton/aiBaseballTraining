"""
backend.py -- what the app's buttons drive.

Hardware records first and processes after capture. Demo data uses the ring buffer:

  hardware  records SVO + IMU CSV, closes both writers, detects every saved camera frame,
            then aligns and fuses one swing. No live detection or fusion competes with capture.
  demo      "Simulate swing" makes a synthetic swing (dckft.simulate) and pushes its IMU samples
            through the same ring buffer, so the app can be tested with no camera or IMU.

Each swing is a folder in app_data/swings/<id>/: the raw window (imu.csv, camera.jsonl), the
fusion outputs (swing.json, fused.csv, markers*.csv, camera_orientation.csv) and swing.png.
"""

from __future__ import annotations

import csv
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
        self.capture_mode = "120fps"
        self.capture_seconds = 5.0
        self.session = None
        self.n_frames = {"imu": 0, "camera": 0}

    # ------------------------------------------------------------------ status
    def say(self, msg: str) -> None:
        self.message = msg
        self.log.append(time.strftime("%H:%M:%S ") + msg)
        print(msg, flush=True)

    def status(self) -> dict:
        return {"mode": self.mode, "message": self.message, "busy": self.busy,
                "ring": (f"{self.ring.progress}, resets from IMU gaps: {self.ring.discontinuities}"
                         if self.ring is not None else None),
                "frames": self.n_frames, "log": list(self.log)[-12:],
                "config": self.config_path or "(defaults: example bat geometry)",
                "trigger": self._trigger_text(), "session": self.session,
                "capture_mode": self.capture_mode, "capture_seconds": self.capture_seconds}

    def _trigger_text(self) -> str:
        s = self.cfg.swing
        if s.trigger_source == "gyro":
            v = f"|omega| > {s.trigger_rads:g} rad/s"
        else:
            v = f"{s.trigger_source} > {s.trigger_accel_ms2:g} m/s^2"
        return f"{v}; record first, process after capture; no uninterrupted ring window required"

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
    def start_hardware(self, capture_mode="120fps", seconds=5.0) -> bool:
        """Capture one bounded clip; detection and fusion run only after writers close."""
        if capture_mode not in ("120fps", "1080p60"):
            raise ValueError("Choose 120fps or 1080p60")
        seconds = float(seconds)
        if not np.isfinite(seconds) or not 3 <= seconds <= 15:
            raise ValueError("Recording duration must be between 3 and 15 seconds")
        with self.lock:
            if self.mode != "idle" or self.busy:
                self.say(f"Already running ({self.mode}).")
                return False
            self.capture_mode, self.capture_seconds = capture_mode, seconds
            self.stop_evt.clear()
            self.mode = "hardware"
            self.thread = threading.Thread(target=self._hardware_loop, daemon=True)
            self.thread.start()
        return True

    def stop(self) -> None:
        if self.mode == "hardware":
            self.stop_evt.set()
            self.say("Finishing the clip, then processing the saved data ...")
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=30)
        # The worker owns mode/busy until processing has actually finished.

    def _start_child(self, name, cmd, session):
        log = open(os.path.join(session, name + ".log"), "w")
        try:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except Exception:
            log.close()
            raise
        self.procs.append((name, proc, log))
        return proc

    def _stop_children(self):
        children, self.procs = self.procs, []
        for _, proc, _ in children:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for _, proc, log in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            finally:
                log.close()

    def _capture_commands(self, session):
        from Cameras.config import load_config as load_camera_config
        camera = load_camera_config(self.cam_config).to_dict()
        fps = 120 if self.capture_mode == "120fps" else 60
        camera["camera"].update(resolution="SVGA" if fps == 120 else "HD1080",
                                 fps=fps, depth_mode="NONE")
        camera["output"].update(zmq_bind=None, jsonl_path=None)
        camera_path = os.path.join(session, "camera_config.json")
        fusion_path = os.path.join(session, "fusion_config.json")
        with open(camera_path, 'w') as f:
            json.dump(camera, f, indent=2)
        with open(fusion_path, 'w') as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        py = sys.executable
        imu = [py, '-u', '-m', 'dckft.imu.reader', '--no-zmq', '--quiet',
               '--config', fusion_path, '--csv', os.path.join(session, 'imu_session.csv')]
        if self.imu_port:
            imu += ['--port', self.imu_port]
        camera_cmd = [py, '-u', os.path.join(REPO, 'Cameras', 'track_bat.py'),
                      '--config', camera_path, '--record-only', '--no-zmq',
                      '--record', os.path.join(session, 'camera.svo2'),
                      '--frame-log', os.path.join(session, 'capture_frames.csv'),
                      '--max-frames', str(round(self.capture_seconds*fps))]
        return imu, camera_cmd

    def _hardware_loop(self):
        session = os.path.join(DATA, 'sessions', time.strftime('%Y%m%d_%H%M%S')
                               + f'_{time.time_ns() % 1000000:06d}')
        self.session = session
        os.makedirs(session, exist_ok=True)
        self.n_frames = {'imu': 0, 'camera': 0}
        try:
            imu_cmd, camera_cmd = self._capture_commands(session)
            self.say('Opening the camera and IMU. Wait for RECORDING NOW before swinging.')
            self._start_child('imu', imu_cmd, session)
            proc = self._start_child('camera', camera_cmd, session)
            begun, start = False, time.monotonic()
            while proc.poll() is None:
                if self.stop_evt.wait(.1):
                    break
                with open(os.path.join(session, 'camera.log'), errors='replace') as f:
                    lines = f.read()
                if not begun and 'Recording ' in lines:
                    begun, start = True, time.monotonic()
                    self.say('RECORDING NOW: hold still 1 second, swing once, hold the finish. '
                             'Capture stops automatically; Stop can end it earlier.')
                if time.monotonic()-start > (self.capture_seconds+20 if begun else 30):
                    raise RuntimeError('Camera timed out; see camera.log. Raw files are preserved.')
            code = proc.poll()
            self._stop_children()  # finalise SVO and CSV before any offline readers start
            if code not in (None, 0):
                raise RuntimeError('Camera recording failed; see camera.log')
            svo = os.path.join(session, 'camera.svo2')
            if not os.path.isfile(svo) or os.path.getsize(svo) == 0:
                raise RuntimeError('No camera video was saved; see camera.log')
            self.mode, self.busy = 'processing', True
            self.say('Clip saved. Detecting pink bands in every saved frame ...')
            self._detect_recording(session)
            self.say('Camera processing finished. Aligning IMU and building camera-led replay ...')
            self._save_recorded_session(session)
        except Exception as e:
            self.say(f'Capture/processing failed: {e}. Raw session kept: {session}')
            traceback.print_exc()
        finally:
            self._stop_children()
            self.busy, self.mode = False, 'idle'

    def _detect_recording(self, session):
        cmd = [sys.executable, '-u', os.path.join(REPO, 'Cameras', 'track_bat.py'),
               '--svo', os.path.join(session, 'camera.svo2'),
               '--config', os.path.join(session, 'camera_config.json'),
               '--bat-config', os.path.join(session, 'fusion_config.json'),
               '--depth-source', 'stereo', '--no-zmq',
               '--frame-log', os.path.join(session, 'camera_frames.csv'),
               '--jsonl', os.path.join(session, 'camera_session.jsonl')]
        proc = self._start_child('detection', cmd, session)
        try:
            code = proc.wait(timeout=300)
            if code:
                raise RuntimeError('Offline detection failed; see detection.log')
        finally:
            self._stop_children()

    def _save_recorded_session(self, session):
        from dckft.data import load_camera_jsonl, load_imu_csv
        from dckft.pipeline import process_swing
        from dckft.swing import detect_swings_imu
        from dckft.render import render_swing
        cam_path = os.path.join(session, 'camera_session.jsonl')
        cam = load_camera_jsonl(cam_path, list(self.cfg.bat.marker_s_m))
        if len(cam) < 10:
            raise ValueError('Not enough optical positions; inspect detection.log and camera_frames.csv')
        imu = load_imu_csv(os.path.join(session, 'imu_session.csv'))
        self.n_frames = {'imu': len(imu), 'camera': len(cam)}
        if len(imu) < 10:
            raise ValueError('No usable IMU stream. Camera video is saved, but no IMU gap fill is possible')
        swings = [s for s in detect_swings_imu(imu, self.cfg)
                  if s.t_end >= cam.t[0]-self.cfg.sync.max_offset_s
                  and s.t_onset <= cam.t[-1]+self.cfg.sync.max_offset_s]
        if not swings:
            raise ValueError('No swing found during the camera clip. Check IMU mounting/trigger and logs')
        swing = max(swings, key=lambda s: s.peak_rads)
        sid = os.path.basename(session)
        d = os.path.join(SWINGS, sid)
        os.makedirs(d, exist_ok=True)
        for src, dst in [('imu_session.csv', 'imu.csv'), ('camera_session.jsonl', 'camera.jsonl'),
                         ('camera_session.jsonl.meta.json', 'camera.jsonl.meta.json'),
                         ('camera_frames.csv', 'camera_frames.csv'),
                         ('capture_frames.csv', 'capture_frames.csv'),
                         ('camera_config.json', 'camera_config.json'),
                         ('fusion_config.json', 'fusion_config.json')]:
            source = os.path.join(session, src)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(d, dst))
        meta = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'source': 'hardware (record first)',
                'session': session, 'imu_frames': len(imu), 'camera_frames': len(cam),
                'git': git_info()['commit'], 'config': self.config_path or 'defaults',
                'status': 'processing'}
        self._write_meta(d, meta)
        try:
            res = process_swing(cam, imu, self.cfg, t_trigger=swing.t_peak)
            quality = self._camera_quality(session)
            with open(os.path.join(d, 'camera_quality.json'), 'w') as f:
                json.dump(quality, f, indent=2)
            self.n_frames['camera'] = quality['saved_frames']
            res.save(d, self.cfg.bat)
            self.cfg.render.show = False
            render_swing(res, self.cfg.bat, self.cfg.render, d, title=f'Swing {sid}')
            meta['status'] = 'ok' if res.metrics.speed_reliable else 'saved; review tracking quality'
            meta['camera_quality'] = quality
            self._write_meta(d, meta)
            self.say(f'Swing {sid} saved. Camera coverage during swing: {res.camera_coverage:.0%}. '
                     'Pink = camera-supported; orange = IMU/model estimate. '
                     + ('' if res.metrics.speed_reliable else res.metrics.speed_note))
            return sid
        except Exception as e:
            meta['status'] = 'failed: '+str(e)
            self._write_meta(d, meta)
            raise

    @staticmethod
    def _camera_quality(session):
        with open(os.path.join(session, 'camera_frames.csv')) as f:
            frames = list(csv.DictReader(f))
        if not frames:
            raise ValueError('No saved SVO frames decoded')
        times = np.array([int(r['ts_ns_image']) for r in frames], dtype=np.int64)
        n = len(times)
        span = float((times[-1]-times[0])/1e9)
        points = np.array([int(r['accepted_3d']) for r in frames])
        return {'saved_frames': n, 'span_s': span,
                'effective_saved_fps': (n-1)/span if span > 0 else None,
                'max_frame_gap_ms': float(np.diff(times).max()/1e6) if n > 1 else None,
                'nonincreasing_timestamps': int((np.diff(times) <= 0).sum()),
                'frames_with_positions': int((points > 0).sum()),
                'frames_with_two_or_more_markers': int((points >= 2).sum()),
                'tracking_fraction_entire_clip': float((points > 0).mean())}
