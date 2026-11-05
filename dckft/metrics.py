"""
metrics.py -- the handful of numbers a hitter/coach actually uses (no raw 4D-vector dumps).

All angles use `up` = -gravity in the camera/world frame.

  bat_speed_mph           sweet-spot speed at contact
  peak_bat_speed_mph      max sweet-spot speed between onset and contact
  peak_hand_speed_mph     max speed of the hands point (BatConfig.hands_s_m)
  attack_angle_deg        sweet-spot velocity direction vs horizontal at contact (+ = upward)
  vertical_bat_angle_deg  bat axis (knob -> tip) vs horizontal at contact (- = barrel below hands)
                          (both as Blast Motion defines them; ranges in BLAST_REFERENCE)
  *_cam_deg               the same two angles from the camera alone (pink centroids), a cross-check
  time_to_contact_s       swing onset -> contact
  swing_plane_tilt_deg    tilt from horizontal of the plane best fitting the sweet-spot path
  peak_wrist_rate_dps     peak wrist angular rate (IMU)
  peak_rot_accel_rads2    peak |d omega / dt| of the wrist before contact
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from .config import BatConfig
from .swing import Swing
from .trajectory import Trajectory

MPS_TO_MPH = 2.2369362920544

# Blast Motion's definitions and typical ranges (blastmotion.com, "What is attack angle" May 2026,
# "What is vertical bat angle" Jul 2026; Blast metric reference v4.7):
#   Attack angle: angle of the bat's PATH at impact relative to horizontal; + = swinging up.
#   Vertical bat angle: angle of the BAT (knob -> barrel) relative to horizontal at impact;
#     negative when the barrel is below the knob.
#   Bat speed: sweet spot, 6 in from the tip, at impact.  Peak hand speed: 6 in from the knob.
BLAST_REFERENCE = {
    "attack_angle_deg": {"target": [6, 14], "by_level": {
        "Professional": [2, 16], "MiLB": [1, 15], "College": [0, 14],
        "High School": [0, 14], "Youth": [0, 14]}},
    "vertical_bat_angle_deg": {"by_level": {
        "Professional": [-35, -25], "MiLB": [-34, -24], "College": [-33, -23],
        "HS Varsity": [-31, -21], "HS JV": [-30, -20], "Middle School": [-27, -17],
        "Youth": [-25, -15]}},
}


@dataclass
class SwingMetrics:
    bat_speed_mph: float
    peak_bat_speed_mph: float
    peak_hand_speed_mph: float
    attack_angle_deg: float
    vertical_bat_angle_deg: float
    time_to_contact_s: float
    swing_plane_tilt_deg: float
    peak_wrist_rate_dps: float
    peak_rot_accel_rads2: float
    t_onset: float
    t_contact: float
    contact_detected: bool
    speed_reliable: bool = True
    speed_note: str = ""
    speed_valid_fraction: float = 1.0
    # Angles are judged on their own: they need the bat POSE seen at contact, not the speed
    # peak (advisor: attack angle and vertical bat angle are the end-of-project goal).
    angles_reliable: bool = True
    attack_angle_cam_deg: float = float("nan")
    vertical_bat_angle_cam_deg: float = float("nan")

    def to_dict(self) -> dict:
        return {k: ((round(v, 4) if np.isfinite(v) else None) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _angle_to_horizontal(v: np.ndarray, up: np.ndarray) -> float:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arcsin(np.clip(np.dot(v, up) / n, -1, 1))))


def compute_metrics(traj: Trajectory, omega_world: np.ndarray, swing: Swing, bat: BatConfig,
                    up: np.ndarray, t_contact: Optional[float] = None) -> SwingMetrics:
    up = np.asarray(up, float) / np.linalg.norm(up)
    t = traj.t
    ss_speed = traj.speed(bat.sweet_spot_s_m)
    valid = np.isfinite(ss_speed)
    if traj.valid is not None:
        valid &= traj.valid
    active = (t >= swing.t_onset) & (t <= swing.t_end)

    contact_detected = t_contact is not None or swing.t_contact is not None
    if t_contact is None:
        t_contact = swing.t_contact
    if t_contact is None:
        # dry swing: contact := time of peak sweet-spot speed between onset and swing end
        m = active & valid
        t_contact = float(t[m][np.argmax(ss_speed[m])]) if m.any() else swing.t_peak

    k_c = int(np.clip(np.searchsorted(t, t_contact), 0, len(t) - 1))
    # Impact itself rattles the IMU, so read contact kinematics 2.5 ms *before* the spike.
    k_c_pre = (max(0, k_c - max(1, int(round(0.0025 / np.median(np.diff(t))))))
               if contact_detected else k_c)

    pre = (t >= swing.t_onset) & (t <= t[k_c_pre]) & valid
    fraction = float(valid[active].mean()) if active.any() else 0.0
    peak_window = active & (abs(t-swing.t_peak) <= .08)
    peak_support = float(valid[peak_window].mean()) if peak_window.any() else 0.0
    reliable = bool(valid[k_c_pre] and (active & valid).sum() >= 5 and peak_support >= .8)
    note = ("Model estimate with optical support; not independently validated." if reliable else
            "Speed unavailable: insufficient optical support at contact/wrist peak. "
            "Occlusion and reacquisition are excluded; no speed is inferred across a long gap.")
    v_ss = traj.velocity(bat.sweet_spot_s_m)
    hands_speed = traj.speed(bat.hands_s_m)

    path = traj.point(bat.sweet_spot_s_m)[(t >= swing.t_onset) & (t <= t_contact)]
    if len(path) >= 3:
        c = path - path.mean(0)
        normal = np.linalg.svd(c, full_matrices=False)[2][-1]
        tilt = float(np.degrees(np.arccos(np.clip(abs(np.dot(normal, up)), 0, 1))))
    else:
        tilt = float("nan")

    dt = np.median(np.diff(t))
    win = max(5, int(round(0.015 / dt)) | 1)
    if len(t) > win:
        dw = savgol_filter(omega_world, win, 2, deriv=1, delta=dt, axis=0)
        rot_acc = np.linalg.norm(dw, axis=1)
        pre_c = (t >= swing.t_onset) & (t <= t_contact - 0.005)
        peak_acc = float(rot_acc[pre_c].max()) if pre_c.any() else float("nan")
    else:
        peak_acc = float("nan")

    wrist = np.linalg.norm(omega_world, axis=1)
    # Angles: the pose at contact must be optically supported, and for the attack angle (a
    # direction of motion) a few supported samples must sit around contact.
    near = (t >= t[k_c_pre] - 0.02) & (t <= t[k_c_pre] + 0.02)
    angles_ok = bool(valid[k_c_pre]) and int((near & valid).sum()) >= 3
    return SwingMetrics(
        bat_speed_mph=float(ss_speed[k_c_pre] * MPS_TO_MPH) if reliable else float("nan"),
        peak_bat_speed_mph=float(ss_speed[pre].max() * MPS_TO_MPH) if reliable else float("nan"),
        peak_hand_speed_mph=float(hands_speed[pre].max() * MPS_TO_MPH) if reliable else float("nan"),
        attack_angle_deg=_angle_to_horizontal(v_ss[k_c_pre], up) if angles_ok else float("nan"),
        vertical_bat_angle_deg=(_angle_to_horizontal(traj.axis[k_c_pre], up) if valid[k_c_pre]
                                else float("nan")),
        time_to_contact_s=float(t_contact - swing.t_onset),
        swing_plane_tilt_deg=tilt,
        peak_wrist_rate_dps=float(np.degrees(wrist[pre].max())) if pre.any() else float("nan"),
        peak_rot_accel_rads2=peak_acc,
        t_onset=float(swing.t_onset),
        t_contact=float(t_contact),
        contact_detected=bool(contact_detected),
        speed_reliable=reliable,
        speed_note=note,
        speed_valid_fraction=fraction,
        angles_reliable=angles_ok,
    )

