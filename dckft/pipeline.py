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
from .cam_orientation import CamOrientation, camera_angles, camera_orientation, compare_with_imu
from .markers import MarkerFill, fill_markers, write_outputs as write_marker_outputs
from .metrics import BLAST_REFERENCE, SwingMetrics, compute_metrics
from .quality import clean_camera, optical_support
from .swing import Swing, detect_swings, detect_swings_imu
from .sync import axis_rate, estimate_offset
from .trajectory import Trajectory, build_trajectory


class NoSwingError(RuntimeError):
    pass


def merge_accels(accel, accel_hg, cfg: FusionConfig):
    """
    The ICM45686 accelerometer clips at +-32 g, and the wrist exceeds that late in the swing --
    exactly where the camera tends to lose the markers. The board's ADXL375 reads to +-200 g, so
    where the ICM is saturated its samples are swapped in (noisier, but not clipped).
    Returns (accel, valid): valid is False only where neither sensor is usable.
    """
    sat = np.any(np.abs(accel) >= 0.95 * cfg.imu.accel_saturation_ms2, axis=1)
    out = accel.copy()
    valid = ~sat
    if cfg.imu.use_highg_when_saturated and accel_hg is not None and np.any(accel_hg):
        hg_ok = sat & np.all(np.abs(accel_hg) < 0.95 * cfg.imu.hg_saturation_ms2, axis=1)
        out[hg_ok] = accel_hg[hg_ok]
        valid = valid | hg_ok
    return out, valid


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
    markers: Optional[MarkerFill] = None
    cam_orient: Optional[CamOrientation] = None
    cam_vs_imu: dict = field(default_factory=dict)
    camera_rejected: int = 0

    def diagnostics(self) -> dict:
        e = self.ekf
        return {
            "time_offset_ms": round(self.time_offset_s * 1e3, 2),
            "sync_correlation": round(self.sync_corr, 3),
            "camera_coverage": round(self.camera_coverage, 3),
            "camera_frames_rejected": self.camera_rejected,
            "camera_quaternion_vs_imu": self.cam_vs_imu,
            "swing_end_s": self.swing.t_end,
            "speed_reliable": self.metrics.speed_reliable,
            "speed_note": self.metrics.speed_note,
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
            "ekf_smoothed": bool(e.smoothed),
            "markers": (self.markers.summary((self.swing.t_onset, self.swing.t_end))
                        if self.markers is not None else None),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }

    def save(self, out_dir: str, bat_cfg) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        summary = {"metrics": self.metrics.to_dict(), "coaching": self.notes,
                   "blast_reference": BLAST_REFERENCE,
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
                        "sweet_x", "sweet_y", "sweet_z", "sweet_speed_mps", "att_std_deg",
                        "speed_valid", "in_swing"])
            t0 = self.swing.t_onset
            for k in range(len(tr.t)):
                w.writerow([round(tr.t[k] - t0, 6), *np.round(self.ekf.q[k], 6),
                            *np.round(tr.knob[k], 5), *np.round(tr.axis[k], 5),
                            *np.round(tip[k], 5), *np.round(ss[k], 5), round(float(spd[k]), 4),
                            round(float(self.ekf.att_std_deg[k]), 3),
                            int(tr.valid[k]) if tr.valid is not None else 1,
                            int(self.swing.t_onset <= tr.t[k] <= self.swing.t_end)])
        if self.markers is not None:
            write_marker_outputs(self.markers, out_dir, self.swing.t_onset)
        if self.cam_orient is not None:
            co = self.cam_orient
            with open(os.path.join(out_dir, "camera_orientation.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "qw", "qx", "qy", "qz", "omega_x", "omega_y", "omega_z",
                            "omega_mag_rads", "axis_x", "axis_y", "axis_z"])
                for k in range(len(co.t)):
                    w.writerow([round(co.t[k] - self.swing.t_onset, 6), *np.round(co.q[k], 6),
                                *np.round(co.omega[k], 4),
                                round(float(np.linalg.norm(co.omega[k])), 4), *np.round(co.axis[k], 5)])
        return summary


def process_swing(cam: CameraData, imu: ImuData, cfg: FusionConfig = FusionConfig(),
                  swing_index: int = -1, coach=None, verbose: bool = False) -> SwingResult:
    tm = {}
    t_start = time.perf_counter()

    cam, rejected = clean_camera(cam, cfg.bat.marker_s_m)
    knob_c, axis_c, count = bat_line_from_markers(cam, cfg.bat)
    g = cam.gravity_cam if cam.gravity_cam is not None else np.asarray(cfg.gravity_cam, float)
    up = -np.asarray(g, float) / np.linalg.norm(g)
    tm["bat_line"] = (time.perf_counter() - t_start) * 1e3

    swings = detect_swings_imu(imu, cfg)
    if not swings:
        raise NoSwingError(f"No swing found in the IMU data (trigger {cfg.swing.trigger_source}; "
                           f"peak |omega| {np.linalg.norm(imu.gyro, axis=1).max():.1f} rad/s, peak "
                           f"| |a|-g | {np.abs(np.linalg.norm(imu.accel, axis=1) - 9.80665).max():.1f} "
                           f"m/s^2). Tune with: python -m dckft.trigger_tune --csv <imu.csv>")
    sw = swings[swing_index]

    # ---- time sync
    t1 = time.perf_counter()
    offset, corr = cfg.sync.fixed_offset_s, 0.0
    if cfg.sync.use_xcorr:
        imu_rate = np.linalg.norm(imu.gyro, axis=1)
        if imu.has_host_clock:
            sync_camera = (cam.t >= sw.t_onset - .3 - cfg.sync.max_offset_s) & (cam.t <= sw.t_end)
            cam_rate = axis_rate(cam.t[sync_camera], axis_c[sync_camera])
            win = (sw.t_onset - 0.3, sw.t_end)
            offset, corr = estimate_offset(cam.t[sync_camera], cam_rate, imu.t, imu_rate,
                                           cfg.sync.max_offset_s, window=win)
        else:
            cam_rate = axis_rate(cam.t, axis_c)
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
    # Keep stance for calibration, but STOP fusion/smoothing at swing end. Later held-bat
    # observations must not pull the RTS solution or inflate the dry-swing speed peak.
    w0, w1 = sw.t_onset - cfg.swing.pre_roll_s, sw.t_end
    win_imu = imu.slice(w0, w1)
    cm = (cam.t >= w0) & (cam.t <= w1)
    t_cam, knob_w, axis_w = cam.t[cm], knob_c[cm], axis_c[cm]
    in_swing = (cam.t >= sw.t_onset) & (cam.t <= sw.t_end)
    # JSONL contains only published frames; its row count alone can report 100% despite
    # a long complete dropout. Include absent frame slots in the denominator.
    fps = float(cam.meta.get("camera", {}).get("fps", 0) or 0)
    if fps <= 0 and len(cam.t)>1:
        fps = 1.0/float(np.median(np.diff(cam.t)))
    expected = max(1, int(round((sw.t_end-sw.t_onset)*fps))+1)
    coverage = min(1.0, float(np.isfinite(axis_c[in_swing]).all(axis=1).sum())/expected)

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
        # Refinement shifts IMU times; still exclude all later optical observations.
        keep = t_cam <= sw.t_end
        t_cam, knob_w, axis_w = t_cam[keep], knob_w[keep], axis_w[keep]
        cm &= cam.t <= sw.t_end
    tm["calibration"] = (time.perf_counter() - t1) * 1e3
    t1 = time.perf_counter()
    ekf = run_ekf(win_imu.t, win_imu.gyro, win_imu.accel, t_cam, axis_w, up, cal, cfg.ekf,
                  cfg.imu.gyro_saturation_rads, smooth=cfg.ekf.smooth)
    tm["ekf"] = (time.perf_counter() - t1) * 1e3

    # ---- trajectory, metrics, coaching
    t1 = time.perf_counter()
    acc_world = acc_valid = None
    if cfg.knob_track == "kalman":
        acc_body, acc_valid = merge_accels(win_imu.accel,
                                           win_imu.accel_hg if cfg.imu.use_adxl375 else None, cfg)
        acc_world = Q.rotate(ekf.q, acc_body) + np.asarray(g, float)
        # lever arm: the IMU sits imu_s_m up the handle from the knob, so remove the
        # tangential + centripetal terms to get the knob's acceleration
        r_arm = cfg.bat.imu_s_m * ekf.axis
        w = ekf.omega_world
        alpha = np.gradient(w, ekf.t, axis=0)
        acc_world = acc_world - np.cross(alpha, r_arm) - np.cross(w, np.cross(w, r_arm))
        if sw.t_contact is not None:      # impact ringing: don't integrate it
            acc_valid &= ~((win_imu.t >= sw.t_contact - 0.005) & (win_imu.t <= sw.t_contact + 0.05))
    traj = build_trajectory(ekf.t, ekf.axis, ekf.omega_world, t_cam, knob_w,
                            lam=cfg.knob_spline_lam, acc_world=acc_world, axis_cam=axis_w,
                            flex_lam=cfg.flex_lam, acc_valid=acc_valid,
                            marker_pts={m: p[cm] for m, p in cam.points.items()},
                            marker_s=cfg.bat.marker_s_m, max_gap_s=cfg.swing.max_camera_gap_s)
    traj.valid = optical_support(traj.t, t_cam[np.isfinite(axis_w).all(axis=1)],
                                 cfg.swing.max_camera_gap_s, cfg.swing.reacquire_guard_s)
    metrics = compute_metrics(traj, ekf.omega_world, sw, cfg.bat, up)
    # Camera-only orientation from the pink centroids: quaternion + its rate (independent check)
    cam_orient, cam_vs_imu = None, {}
    try:
        cam_orient = camera_orientation(t_cam, axis_w)
        cam_vs_imu = compare_with_imu(cam_orient, ekf.t, ekf.omega_world, traj.axis,
                                      (sw.t_onset, sw.t_end))
        metrics.attack_angle_cam_deg, metrics.vertical_bat_angle_cam_deg = camera_angles(
            t_cam, knob_w, axis_w, metrics.t_contact, up, cfg.bat.sweet_spot_s_m,
            cfg.swing.max_camera_gap_s, cam_orient)
    except ValueError:
        pass
    notes = ((coach or RuleBasedCoach()).notes(metrics) if metrics.speed_reliable
             else [metrics.speed_note])
    tm["metrics"] = (time.perf_counter() - t1) * 1e3
    marker_fill = fill_markers(traj, cfg.bat, cam.t[cm], {m: p[cm] for m, p in cam.points.items()})
    tm["total"] = (time.perf_counter() - t_start) * 1e3

    res = SwingResult(swing=sw, metrics=metrics, notes=notes, traj=traj, ekf=ekf, calib=cal, up=up,
                      time_offset_s=offset, sync_corr=corr, camera_coverage=coverage,
                      timings_ms=tm, cam_t=t_cam, cam_knob=knob_w, cam_axis=axis_w,
                      markers=marker_fill, camera_rejected=rejected,
                      cam_orient=cam_orient, cam_vs_imu=cam_vs_imu)
    if verbose:
        print(json.dumps(res.diagnostics(), indent=2))
    return res

