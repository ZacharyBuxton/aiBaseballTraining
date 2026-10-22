"""
markers.py -- per-marker tracks with the camera's gaps filled from the IMU.

The camera sees each tape marker only in some frames: fast markers blur out near contact, the
batter's body occludes them, the color detector misses some. After fusion we know, at IMU rate
(400 Hz), where the bat is (knob + axis from the Kalman smoother), so each marker's position
follows from its measured distance along the bat:

    p_i(t) = knob(t) + s_i * u(t)

fill_markers() returns, for every marker:
  * a 400 Hz filled track (camera where the camera saw it, IMU-driven fusion everywhere)
  * the same track at every camera frame time, flagged "camera" or "imu_fill"
  * how well the fused track agrees with the camera where both exist (residual, mm) and the
    longest gap the IMU had to bridge

write_outputs() saves:
  markers.csv            400 Hz:   t_s, <marker>_x/_y/_z ...
  markers_camera.csv     per frame: t_s, <marker>_x/_y/_z, <marker>_src (camera | imu_fill)
  camera_filled.jsonl    track_bat.py payload format, every marker present in every frame,
                         each point tagged "source": "camera" | "imu_fill"
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config import BatConfig
from .trajectory import Trajectory


@dataclass
class MarkerTrack:
    marker_id: str
    s_m: float
    filled: np.ndarray            # (N,3) at trajectory (IMU) times
    cam_filled: np.ndarray        # (M,3) at camera times
    cam_seen: np.ndarray          # (M,) bool: camera measured it in that frame
    cam_measured: np.ndarray      # (M,3) camera measurement, NaN when not seen
    residual_mm: np.ndarray       # (M,) |measured - fused| where seen, NaN otherwise

    def summary(self, t_cam, window) -> dict:
        m = (t_cam >= window[0]) & (t_cam <= window[1])
        seen = self.cam_seen[m]
        res = self.residual_mm[m][seen]
        return {
            "seen_pct": round(100.0 * float(seen.mean()), 1) if len(seen) else 0.0,
            "frames_filled": int((~seen).sum()),
            "longest_gap_ms": round(float(_longest_gap_ms(t_cam[m], seen)), 1),
            "residual_rms_mm": round(float(np.sqrt(np.mean(res ** 2))), 2) if len(res) else None,
        }


@dataclass
class MarkerFill:
    t: np.ndarray
    t_cam: np.ndarray
    tracks: Dict[str, MarkerTrack] = field(default_factory=dict)

    def summary(self, window) -> dict:
        return {m: tr.summary(self.t_cam, window) for m, tr in self.tracks.items()}


def _longest_gap_ms(t: np.ndarray, seen: np.ndarray) -> float:
    if len(t) == 0 or seen.all():
        return 0.0
    best, start = 0.0, None
    for i, s in enumerate(seen):
        if not s and start is None:
            start = i - 1 if i > 0 else i
        if s and start is not None:
            best = max(best, t[i] - t[start])
            start = None
    if start is not None:
        best = max(best, t[-1] - t[start])
    return best * 1e3


def fill_markers(traj: Trajectory, bat: BatConfig, t_cam: np.ndarray,
                 cam_points: Dict[str, np.ndarray]) -> MarkerFill:
    # camera frames inside the fused window only (no extrapolation)
    inside = (t_cam >= traj.t[0]) & (t_cam <= traj.t[-1])
    tc = t_cam[inside]
    out = MarkerFill(t=traj.t, t_cam=tc)
    for mid, s in bat.marker_s_m.items():
        filled = traj.point(s)
        cam_f = np.column_stack([np.interp(tc, traj.t, filled[:, j]) for j in range(3)])
        meas = cam_points.get(mid, np.full((len(t_cam), 3), np.nan))[inside]
        seen = np.isfinite(meas).all(axis=1)
        res = np.full(len(tc), np.nan)
        res[seen] = np.linalg.norm(meas[seen] - cam_f[seen], axis=1) * 1e3
        out.tracks[mid] = MarkerTrack(mid, s, filled, cam_f, seen, meas, res)
    return out


def write_outputs(fill: MarkerFill, out_dir: str, t0: float, camera_id: str = "fused") -> None:
    os.makedirs(out_dir, exist_ok=True)
    ids = list(fill.tracks)
    with open(os.path.join(out_dir, "markers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [f"{m}_{a}" for m in ids for a in "xyz"])
        for k in range(len(fill.t)):
            w.writerow([round(fill.t[k] - t0, 6)] +
                       [round(float(v), 5) for m in ids for v in fill.tracks[m].filled[k]])
    with open(os.path.join(out_dir, "markers_camera.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [c for m in ids for c in (f"{m}_x", f"{m}_y", f"{m}_z", f"{m}_src")])
        for i in range(len(fill.t_cam)):
            row = [round(fill.t_cam[i] - t0, 6)]
            for m in ids:
                tr = fill.tracks[m]
                p = tr.cam_measured[i] if tr.cam_seen[i] else tr.cam_filled[i]
                row += [round(float(v), 5) for v in p] + ["camera" if tr.cam_seen[i] else "imu_fill"]
            w.writerow(row)
    with open(os.path.join(out_dir, "camera_filled.jsonl"), "w") as f:
        for i in range(len(fill.t_cam)):
            ts = int(round(fill.t_cam[i] * 1e9))
            pts = []
            for m in ids:
                tr = fill.tracks[m]
                p = tr.cam_measured[i] if tr.cam_seen[i] else tr.cam_filled[i]
                pts.append({"id": m, "pos": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                            "source": "camera" if tr.cam_seen[i] else "imu_fill"})
            f.write(json.dumps({"schema_version": "1.1", "coord_sys": "zedx_left_rect_opencv_meters_v1",
                                "camera_id": camera_id, "frame_id": i, "ts_ns_image": ts,
                                "ts_ns_mono": ts, "points": pts}) + "\n")
