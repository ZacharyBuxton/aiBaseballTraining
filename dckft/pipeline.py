"""
pipeline.py -- one function from raw recordings to metrics: process_swing().

    camera JSONL + IMU CSV  ->  bat line  ->  swing detect  ->  time sync  ->  self-calibration
                            ->  quaternion MEKF  ->  trajectory  ->  metrics  ->  coaching
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import quat as Q
from .calib import Calibration, calibrate
from .coaching import RuleBasedCoach
from .config import FusionConfig, G
from .data import CameraData, ImuData, bat_line_from_markers
from .ekf import EkfResult, run_ekf
from .metrics import SwingMetrics, compute_metrics
from .swing import Swing, detect_swings
from .sync import axis_rate, estimate_offset
from .trajectory import Trajectory, build_trajectory


class NoSwingError(RuntimeError):
    pass


@dataclass
class SwingResult:
    swing: Swing
    metrics: SwingMetrics
    notes: List[str]
    traj: Trajectory
    ekf: EkfResult
    calib: Calibration
    up: np.ndarray
    time_offset_s: float
    sync_corr: float
    camera_coverage: float           # fraction of camera frames in the swing with a bat line
    timings_ms: dict = field(default_factory=dict)
    cam_t: Optional[np.ndarray] = None
    cam_knob: Optional[np.ndarray] = None
    cam_axis: Optional[np.ndarray] = None

    def diagnostics(self) -> dict:
        e = self.ekf
        return {
            "time_offset_ms": round(self.time_offset_s * 1e3, 2),
            "sync_correlation": round(self.sync_corr, 3),
            "camera_coverage": round(self.camera_coverage, 3),
            "calib_rms_axis_deg": round(self.calib.rms_axis_deg, 2),
            "calib_rms_gravity_deg": (None if np.isnan(self.calib.rms_gravity_deg)
                                      else round(self.calib.rms_gravity_deg, 2)),
            "grip_axis_imu": np.round(self.calib.grip_axis, 4).tolist(),
            "gyro_bias_rads": np.round(e.bias[-1], 5).tolist(),
            "ekf_camera_updates": e.n_cam_updates,
            "ekf_camera_rejected": e.n_cam_rejected,
            "ekf_gravity_updates": e.n_grav_updates,
            "ekf_median_innovation_deg": (round(float(np.median(e.cam_innov_deg)), 2)
                                          if len(e.cam_innov_deg) else None),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }

    def save(self, out_dir: str, bat_cfg) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        summary = {"metrics": self.metrics.to_dict(), "coaching": self.notes,
                   "diagnostics": self.diagnostics()}
        with open(os.path.join(out_dir, "swing.json"), "w") as f:
            json.dump(summary, f, indent=2)
        tr = self.traj
        ss = tr.point(bat_cfg.sweet_spot_s_m)
        tip = tr.point(bat_cfg.length_m)
        spd = tr.speed(bat_cfg.sweet_spot_s_m)
        with open(os.path.join(out_dir, "fused.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "qw", "qx", "qy", "qz", "knob_x", "knob_y", "knob_z",
                        "axis_x", "axis_y", "axis_z", "tip_x", "tip_y", "tip_z",
                        "sweet_x", "sweet_y", "sweet_z", "sweet_speed_mps", "att_std_deg"])
            t0 = self.swing.t_onset
            for k in range(len(tr.t)):
                w.writerow([round(tr.t[k] - t0, 6), *np.round(self.ekf.q[k], 6),
                            *np.round(tr.knob[k], 5), *np.round(tr.axis[k], 5),
                            *np.round(tip[k], 5), *np.round(ss[k], 5), round(float(spd[k]), 4),
                            round(float(self.ekf.att_std_deg[k]), 3)])
        return summary


def process_swing(cam: CameraData, imu: ImuData, cfg: FusionConfig = FusionConfig(),
                  swing_index: int = -1, coach=None, verbose: bool = False) -> SwingResult:
    tm = {}
    t_start = time.perf_counter()

    knob_c, axis_c, count = bat_line_from_markers(cam, cfg.bat)
    g = cam.gravity_cam if cam.gravity_cam is not None else np.asarray(cfg.gravity_cam, float)
    up = -np.asarray(g, float) / np.linalg.norm(g)
    tm["bat_line"] = (time.perf_counter() - t_start) * 1e3

    swings = detect_swings(imu.t, imu.gyro, imu.accel_hg, cfg.swing)
    if not swings:
        raise NoSwingError(f"No swing found in the IMU data (peak |omega| "
                           f"{np.linalg.norm(imu.gyro, axis=1).max():.1f} rad/s < trigger "
                           f"{cfg.swing.trigger_rads} rad/s)")
    sw = swings[swing_index]

    # ---- time sync
    t1 = time.perf_counter()
    offset, corr = cfg.sync.fixed_offset_s, 0.0
    if cfg.sync.use_xcorr:
        cam_rate = axis_rate(cam.t, axis_c)
        imu_rate = np.linalg.norm(imu.gyro, axis=1)
        if imu.has_host_clock:
            win = (sw.t_onset - 0.3, sw.t_end + 0.3)
            offset, corr = estimate_offset(cam.t, cam_rate, imu.t, imu_rate,
                                           cfg.sync.max_offset_s, window=win)
        else:
            # Legacy CSV: IMU time starts at 0, so search the whole camera span.
            span = (cam.t[-1] - cam.t[0]) + (imu.t[-1] - imu.t[0])
            imu0 = imu.shifted(cam.t[0] - imu.t[0])
            off2, corr = estimate_offset(cam.t, cam_rate, imu0.t, imu_rate, span)
            offset = (cam.t[0] - imu.t[0]) + off2
    imu = imu.shifted(offset)
    sw = Swing(sw.t_onset + offset, sw.t_peak + offset, sw.t_end + offset,
               None if sw.t_contact is None else sw.t_contact + offset, sw.peak_rads)
    tm["sync"] = (time.perf_counter() - t1) * 1e3

    # ---- window
    w0, w1 = sw.t_onset - cfg.swing.pre_roll_s, sw.t_end + cfg.swing.post_roll_s
    win_imu = imu.slice(w0, w1)
    cm = (cam.t >= w0 - 0.2) & (cam.t <= w1 + 0.2)
    t_cam, knob_w, axis_w = cam.t[cm], knob_c[cm], axis_c[cm]
    in_swing = (cam.t >= sw.t_onset) & (cam.t <= sw.t_end)
    coverage = float((count[in_swing] >= 2).mean()) if in_swing.any() else 0.0

    # ---- calibration + EKF
    t1 = time.perf_counter()
    t_axis_end = sw.t_contact if sw.t_contact is not None else sw.t_peak
    cal = calibrate(win_imu.t, win_imu.gyro, win_imu.accel, t_cam, axis_w, up, t_axis_end, cfg.ekf,
                    cfg.sync.refine_max_s)
    if cal.dt_refine:
        d = cal.dt_refine
        offset += d
        win_imu = win_imu.shifted(d)
        cal.t_ref += d
        sw = Swing(sw.t_onset + d, sw.t_peak + d, sw.t_end + d,
                   None if sw.t_contact is None else sw.t_contact + d, sw.peak_rads)
    tm["calibration"] = (time.perf_counter() - t1) * 1e3
    t1 = time.perf_counter()
    ekf = run_ekf(win_imu.t, win_imu.gyro, win_imu.accel, t_cam, axis_w, up, cal, cfg.ekf,
                  cfg.imu.gyro_saturation_rads)
    tm["ekf"] = (time.perf_counter() - t1) * 1e3

    # ---- trajectory, metrics, coaching
    t1 = time.perf_counter()
    acc_world = acc_valid = None
    if cfg.knob_track == "kalman":
        acc_world = Q.rotate(ekf.q, win_imu.accel) + np.asarray(g, float)
        # lever arm: the IMU sits imu_s_m up the handle from the knob, so remove the
        # tangential + centripetal terms to get the knob's acceleration
        r_arm = cfg.bat.imu_s_m * ekf.axis
        w = ekf.omega_world
        alpha = np.gradient(w, ekf.t, axis=0)
        acc_world = acc_world - np.cross(alpha, r_arm) - np.cross(w, np.cross(w, r_arm))
        acc_valid = np.all(np.abs(win_imu.accel) < 0.95 * cfg.imu.accel_saturation_ms2, axis=1)
        if sw.t_contact is not None:      # impact ringing: don't integrate it
            acc_valid &= ~((win_imu.t >= sw.t_contact - 0.005) & (win_imu.t <= sw.t_contact + 0.05))
    traj = build_trajectory(ekf.t, ekf.axis, ekf.omega_world, t_cam, knob_w,
                            lam=cfg.knob_spline_lam, acc_world=acc_world, axis_cam=axis_w,
                            flex_lam=cfg.flex_lam, acc_valid=acc_valid)
    metrics = compute_metrics(traj, ekf.omega_world, sw, cfg.bat, up)
    notes = (coach or RuleBasedCoach()).notes(metrics)
    tm["metrics"] = (time.perf_counter() - t1) * 1e3
    tm["total"] = (time.perf_counter() - t_start) * 1e3

    res = SwingResult(swing=sw, metrics=metrics, notes=notes, traj=traj, ekf=ekf, calib=cal, up=up,
                      time_offset_s=offset, sync_corr=corr, camera_coverage=coverage,
                      timings_ms=tm, cam_t=t_cam, cam_knob=knob_w, cam_axis=axis_w)
    if verbose:
        print(json.dumps(res.diagnostics(), indent=2))
    return res
