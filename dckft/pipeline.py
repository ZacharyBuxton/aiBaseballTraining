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
from .markers import MarkerFill, fill_markers, write_outputs as write_marker_outputs
from .metrics import SwingMetrics, compute_metrics
from .quality import clean_camera, optical_support
from .replay_fill import ReplayFill, camera_guided_replay
from .swing import Swing, detect_swings
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
    camera_rejected: int = 0
    replay: Optional[ReplayFill] = None

    def diagnostics(self) -> dict:
        e = self.ekf
        active = (self.traj.t >= self.swing.t_onset) & (self.traj.t <= self.swing.t_end)
        replay_residuals = {}
        if self.replay is not None and self.markers is not None:
            rt = self.replay.traj
            for mid, marker in self.markers.tracks.items():
                pos = rt.point(marker.s_m)
                sampled = np.column_stack([np.interp(self.markers.t_cam,rt.t,pos[:,j]) for j in range(3)])
                seen = marker.cam_seen & (self.markers.t_cam >= self.swing.t_onset)
                residual = sampled[seen]-marker.cam_measured[seen]
                replay_residuals[mid] = (round(float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))*1000,2)
                                         if len(residual) else None)
        return {
            "time_offset_ms": round(self.time_offset_s * 1e3, 2),
            "sync_correlation": round(self.sync_corr, 3),
            "camera_coverage": round(self.camera_coverage, 3),
            "camera_frames_rejected": self.camera_rejected,
            "swing_end_s": self.swing.t_end,
            "speed_reliable": self.metrics.speed_reliable,
            "speed_note": self.metrics.speed_note,
            "replay": ({"estimated_samples": int(self.replay.estimated.sum()),
                        "total_samples": len(self.replay.traj.t),
                        "active_swing_samples": int(active.sum()),
                        "active_camera_pose_fraction": (round(float(self.replay.pose_supported[active].mean()), 4)
                                                        if active.any() else None),
                        "active_speed_valid_fraction": (round(float(self.traj.valid[active].mean()), 4)
                                                        if active.any() and self.traj.valid is not None else None),
                        "max_distance_to_camera_support_s": round(float(self.replay.nearest_support_s.max()), 4),
                        "camera_marker_residual_rms_mm": replay_residuals,
                        "method": "cleaned camera bat-line anchors and short-interval interpolation; wrist IMU/model fills optical gaps; speed validity is separate"}
                       if self.replay is not None else None),
            "calib_rms_axis_deg": round(self.calib.rms_axis_deg, 2),
            "analysis_time_range_s": [float(self.traj.t[0]),float(self.traj.t[-1])],
            "camera_window_time_range_s": ([float(self.cam_t[0]),float(self.cam_t[-1])]
                                            if self.cam_t is not None and len(self.cam_t) else None),
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
        if self.replay is not None:
            rp = self.replay
            rt = rp.traj
            tip, sweet = rt.point(bat_cfg.length_m), rt.point(bat_cfg.sweet_spot_s_m)
            with open(os.path.join(out_dir, "replay.csv"), "w", newline="") as f:
                w = csv.writer(f)
                ids = list(bat_cfg.marker_s_m)
                marker_pos = {m: rt.point(bat_cfg.marker_s_m[m]) for m in ids}
                w.writerow(["t_s", "knob_x", "knob_y", "knob_z", "axis_x", "axis_y", "axis_z",
                            "tip_x", "tip_y", "tip_z", "sweet_x", "sweet_y", "sweet_z"] +
                           [f"{m}_{d}" for m in ids for d in "xyz"] +
                           ["pose_source", "nearest_camera_support_s", "imu_accel_used", "speed_valid", "in_swing"])
                for k in range(len(rt.t)):
                    w.writerow([round(rt.t[k]-self.swing.t_onset, 6),
                                *np.round(rt.knob[k], 5), *np.round(rt.axis[k], 5),
                                *np.round(tip[k], 5), *np.round(sweet[k], 5),
                                *[round(float(x), 5) for m in ids for x in marker_pos[m][k]],
                                "imu_estimate" if rp.estimated[k] else "camera_supported_fusion",
                                round(float(rp.nearest_support_s[k]), 6), int(rp.accel_used[k]),
                                int(rt.valid[k]), int(self.swing.t_onset <= rt.t[k] <= self.swing.t_end)])
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
        acc_body, acc_valid = merge_accels(win_imu.accel, win_imu.accel_hg, cfg)
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
    notes = ((coach or RuleBasedCoach()).notes(metrics) if metrics.speed_reliable
             else [metrics.speed_note])
    tm["metrics"] = (time.perf_counter() - t1) * 1e3
    marker_fill = fill_markers(traj, cfg.bat, cam.t[cm], {m: p[cm] for m, p in cam.points.items()})
    # Keep the conservative metric trajectory intact. Replay may fill its unsupported
    # gaps, but doing so must never manufacture optical support or a trusted speed.
    replay_fill = camera_guided_replay(traj, t_cam, knob_w, axis_w, acc_world, acc_valid,
                                      cfg.swing.max_camera_gap_s)
    tm["total"] = (time.perf_counter() - t_start) * 1e3

    res = SwingResult(swing=sw, metrics=metrics, notes=notes, traj=traj, ekf=ekf, calib=cal, up=up,
                      time_offset_s=offset, sync_corr=corr, camera_coverage=coverage,
                      timings_ms=tm, cam_t=t_cam, cam_knob=knob_w, cam_axis=axis_w,
                      markers=marker_fill, camera_rejected=rejected, replay=replay_fill)
    if verbose:
        print(json.dumps(res.diagnostics(), indent=2))
    return res

