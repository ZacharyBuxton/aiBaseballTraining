"""
swing.py -- find swings in the wrist-IMU stream and estimate ball contact.

Trigger (swing.trigger_source): the wrist IMU's ACCELERATION by default -- |a| - g, one
signed axis such as -y (wire side up on the bottom hand), or |(a_x, a_y)| -- or the original
|omega| > trigger_rads. Onset is the last time before the trigger that |omega| was below
onset_rads. The swing ENDS when the bat is held still (low wrist rate and |a| ~ 1 g for
end_hold_s): hold the finish. Contact is the first sharp spike on the impact accelerometer
(ICM45686 by default; ADXL375 if imu.use_adxl375); with none (a dry swing), contact is the time
of peak bat speed, filled in later by dckft.metrics.

LiveSwingDetector does the same thing incrementally for dckft.run live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import SwingConfig


@dataclass
class Swing:
    t_onset: float
    t_peak: float
    t_end: float
    t_contact: Optional[float]        # None if no impact spike found
    peak_rads: float

    @property
    def contact_detected(self) -> bool:
        return self.t_contact is not None


def omega_mag(gyro: np.ndarray, rate_hz: float = 400.0) -> np.ndarray:
    m = np.linalg.norm(gyro, axis=1)
    k = max(1, int(round(rate_hz * 0.01)))           # 10 ms box filter
    return uniform_filter1d(m, k)


def impact_indices(accel_hg: np.ndarray, rate_hz: float, thresh: float) -> np.ndarray:
    """
    Ball impact = a sharp spike on the high-g accelerometer. The swing itself already puts
    hundreds of m/s^2 of smooth centripetal load on the wrist, so threshold the *high-passed*
    signal (minus a 25 ms median) rather than the raw magnitude.
    """
    from scipy.ndimage import median_filter
    if len(accel_hg) < 3:
        return np.array([], dtype=int)
    k = max(3, int(round(rate_hz * 0.025)) | 1)
    base = median_filter(accel_hg, size=(k, 1), mode="nearest")
    hp = np.linalg.norm(accel_hg - base, axis=1)
    return np.where(hp > thresh)[0]


G = 9.80665
_AXES = {"x": 0, "y": 1, "z": 2}


def trigger_signal(gyro: np.ndarray, accel: Optional[np.ndarray], cfg: SwingConfig,
                   rate_hz: float) -> tuple:
    """
    The signal the swing trigger watches, and its threshold.

      gyro         |omega| (rad/s) vs trigger_rads                     (original behaviour)
      accel_mag    | |a| - g | (m/s^2) vs trigger_accel_ms2             orientation-free
      accel_xy     |(a_x, a_y)| with gravity left in, vs trigger_accel_ms2
      accel_-y, accel_+y, accel_-x, ...  one signed axis (advisor: with the wire side UP on the
                   bottom hand it is "prolly -y"), vs trigger_accel_ms2

    Accelerations are the IMU's own axes (specific force). 10 ms box smoothing either way.
    Run `python -m dckft.trigger_tune` on a few practice swings to pick the source/threshold.
    """
    k = max(1, int(round(rate_hz * 0.01)))
    src = cfg.trigger_source
    if src == "gyro" or accel is None:
        return omega_mag(gyro, rate_hz), cfg.trigger_rads
    a = np.asarray(accel, float)
    if src == "accel_mag":
        sig = np.abs(np.linalg.norm(a, axis=1) - G)
    elif src == "accel_xy":
        sig = np.linalg.norm(a[:, :2], axis=1)
    elif src.startswith("accel_") and len(src) == 8 and src[6] in "+-" and src[7] in _AXES:
        sig = (1.0 if src[6] == "+" else -1.0) * a[:, _AXES[src[7]]]
    else:
        raise ValueError(f"Unknown swing.trigger_source {src!r}")
    return uniform_filter1d(sig, k), cfg.trigger_accel_ms2


def still_mask(gyro: np.ndarray, accel: Optional[np.ndarray], cfg: SwingConfig,
               rate_hz: float) -> np.ndarray:
    """Bat held still: low wrist rate AND (when accel is given) |a| close to 1 g."""
    still = omega_mag(gyro, rate_hz) <= cfg.end_rads
    if accel is not None and cfg.end_requires_still_accel:
        k = max(1, int(round(rate_hz * 0.02)))
        dev = uniform_filter1d(np.abs(np.linalg.norm(accel, axis=1) - G), k)
        still &= dev <= cfg.still_accel_ms2
    return still


def detect_swings(t: np.ndarray, gyro: np.ndarray, accel_hg: np.ndarray,
                  cfg: SwingConfig, accel: Optional[np.ndarray] = None,
                  impact_thresh: Optional[float] = None) -> List[Swing]:
    """
    accel_hg   signal used for IMPACT detection (ADXL375, or the ICM45686 accel in ICM-only mode)
    accel      ICM45686 accelerometer for the trigger and the "held still at the end" check;
               None keeps the gyro-only behaviour.
    """
    if len(t) < 10:
        return []
    rate = 1.0 / np.median(np.diff(t))
    m = omega_mag(gyro, rate)
    trig, thresh = trigger_signal(gyro, accel, cfg, rate)
    above = trig > thresh
    quiet = still_mask(gyro, accel, cfg, rate)
    impact_thresh = cfg.contact_hg_ms2 if impact_thresh is None else impact_thresh
    swings: List[Swing] = []
    i = 0
    n = len(t)
    while i < n:
        if not above[i]:
            i += 1
            continue
        # End the event at the START of a sustained low-rate period. Brief dips during a
        # swing do not split it, but picking the bat up again must not extend its end.
        j, quiet_start = i, None
        while j < n and t[j] - t[i] < cfg.max_duration_s:
            if quiet[j]:
                if quiet_start is None:
                    quiet_start = j
                if t[j] - t[quiet_start] >= cfg.end_hold_s:
                    break
            else:
                quiet_start = None
            j += 1
        event_end = quiet_start if quiet_start is not None else min(j, n - 1)
        seg = slice(i, max(i + 1, event_end + 1))
        # an accel trigger can fire before the wrist rate builds: take the peak from the trigger
        # point to the end, and search back for onset from the trigger point
        k_peak = i + int(np.argmax(m[seg]))
        k_on = min(i, k_peak)
        while k_on > 0 and m[k_on] > cfg.onset_rads:
            k_on -= 1
        k_end = event_end
        hits = impact_indices(accel_hg[k_on:k_end + 1], rate, impact_thresh)
        t_contact = float(t[k_on + hits[0]]) if len(hits) else None
        sw = Swing(float(t[k_on]), float(t[k_peak]), float(t[k_end]), t_contact, float(m[k_peak]))
        if swings and sw.t_onset - swings[-1].t_end < cfg.min_gap_s:
            # Cooldown suppresses return-to-stance movement; never merge it into the
            # completed swing (that used to move the speed peak and replay end).
            pass
        else:
            swings.append(sw)
        i = j + 1
    return swings


class LiveSwingDetector:
    """Feed IMU samples as they arrive; `poll()` returns a Swing once its post-roll has passed."""

    def __init__(self, cfg: SwingConfig):
        self.cfg = cfg
        self.armed_peak_t = None
        self.last_swing_end = -1e18

    def update(self, t_latest: float, t: np.ndarray, gyro: np.ndarray, accel_hg: np.ndarray,
               accel: Optional[np.ndarray] = None, impact_thresh: Optional[float] = None):
        # look only at the last few seconds the caller passes in
        sw = detect_swings(t, gyro, accel_hg, self.cfg, accel, impact_thresh)
        sw = [s for s in sw if s.t_onset > self.last_swing_end]
        if not sw:
            return None
        s = sw[0]
        # wait until the swing has finished and post-roll data is in
        if s.t_end < t[-1] - 1e-9 and t_latest - s.t_end >= self.cfg.post_roll_s:
            self.last_swing_end = s.t_end
            return s
        return None



def impact_source(imu, fcfg):
    """(signal, threshold) for impact detection: ICM45686 accel unless imu.use_adxl375."""
    if fcfg.imu.use_adxl375 and np.any(imu.accel_hg):
        return imu.accel_hg, fcfg.swing.contact_hg_ms2
    return imu.accel, fcfg.swing.contact_icm_ms2


def detect_swings_imu(imu, fcfg) -> List[Swing]:
    """
    detect_swings() on an ImuData with the full FusionConfig (trigger + impact source).

    Ball contact, in order of preference:
      1. the firmware's impact flag (protocol-v2 receiver), when the frames carry it
      2. a spike on the ICM45686 accelerometer (the main sensor)
      3. a spike on the ADXL375, when its data are present. The ICM45686 clips at +-32 g and is
         often near that late in the swing, so the impact spike can be flattened; the ADXL375
         (+-200 g) is then used for contact TIMING only (swing.impact_fallback_adxl).
    """
    sig, thr = impact_source(imu, fcfg)
    swings = detect_swings(imu.t, imu.gyro, sig, fcfg.swing, imu.accel, thr)
    for sw in swings:
        win = (imu.t >= sw.t_onset) & (imu.t <= sw.t_end)
        if imu.impact_flag is not None and np.any(imu.impact_flag[win]):
            sw.t_contact = float(imu.t[win][np.argmax(imu.impact_flag[win])])
            continue
        if sw.t_contact is None and fcfg.swing.impact_fallback_adxl and np.any(imu.accel_hg[win]):
            rate = 1.0 / np.median(np.diff(imu.t))
            hits = impact_indices(imu.accel_hg[win], rate, fcfg.swing.contact_hg_ms2)
            if len(hits):
                sw.t_contact = float(imu.t[win][hits[0]])
    return swings
