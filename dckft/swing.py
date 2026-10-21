"""
swing.py -- find swings in the wrist-IMU stream and estimate ball contact.

A swing is confirmed when the smoothed |omega| exceeds trigger_rads (~860 dps). Its onset is
the last time before the peak that |omega| was below onset_rads, its end the first time after
the peak it falls below end_rads. Contact is the first ADXL375 high-g spike in the window
(the wrist feels the impact); if there is none (a dry swing), contact is the time of peak
bat speed, filled in later by dckft.metrics.

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


def detect_swings(t: np.ndarray, gyro: np.ndarray, accel_hg: np.ndarray,
                  cfg: SwingConfig) -> List[Swing]:
    if len(t) < 10:
        return []
    rate = 1.0 / np.median(np.diff(t))
    m = omega_mag(gyro, rate)
    above = m > cfg.trigger_rads
    swings: List[Swing] = []
    i = 0
    n = len(t)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and m[j] > cfg.end_rads:
            j += 1
        seg = slice(i, j)
        k_peak = i + int(np.argmax(m[seg]))
        k_on = i
        while k_on > 0 and m[k_on] > cfg.onset_rads:
            k_on -= 1
        k_end = min(j, n - 1)
        hits = impact_indices(accel_hg[k_on:k_end + 1], rate, cfg.contact_hg_ms2)
        t_contact = float(t[k_on + hits[0]]) if len(hits) else None
        sw = Swing(float(t[k_on]), float(t[k_peak]), float(t[k_end]), t_contact, float(m[k_peak]))
        if swings and sw.t_onset - swings[-1].t_end < cfg.min_gap_s:
            # merge with the previous (same swing, e.g. follow-through re-acceleration)
            prev = swings[-1]
            if sw.peak_rads > prev.peak_rads:
                prev.t_peak, prev.peak_rads = sw.t_peak, sw.peak_rads
            prev.t_end = sw.t_end
            prev.t_contact = prev.t_contact or sw.t_contact
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

    def update(self, t_latest: float, t: np.ndarray, gyro: np.ndarray, accel_hg: np.ndarray):
        # look only at the last ~3 s buffer that the caller passes in
        sw = detect_swings(t, gyro, accel_hg, self.cfg)
        sw = [s for s in sw if s.t_onset > self.last_swing_end]
        if not sw:
            return None
        s = sw[0]
        # wait until the swing has finished and post-roll data is in
        if s.t_end < t[-1] - 1e-9 and t_latest - s.t_end >= self.cfg.post_roll_s:
            self.last_swing_end = s.t_end
            return s
        return None
