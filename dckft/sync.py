"""
sync.py -- camera <-> IMU time alignment.

On the Jetson both streams carry host time.time_ns() stamps, so they already share a clock;
what remains is the radio + USB latency of the IMU path (a few ms, roughly constant).
We estimate it the same way the previous team did (cross_correlation.m / alignsignals), but
bounded to +-max_offset_s and on signals that are comparable for a wrist-worn IMU:

  camera:  |du/dt|   rate of change of the bat axis   (= |omega_perp|, roll is invisible)
  IMU:     |omega|   wrist angular-rate magnitude

The two differ in amplitude (forearm roll shows up only in the IMU) but peak together, so both
are z-scored before correlating. Returns the offset to ADD to IMU times.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags, savgol_filter


def fill_nans(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN rows (per column). Edges are held."""
    x = np.array(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        if ok.sum() >= 2:
            x[:, j] = np.interp(t, t[ok], x[ok, j])
    return x.squeeze()


def axis_rate(t_cam: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """|du/dt| from the camera bat-axis series (rad/s)."""
    if len(t_cam) < 7:
        return np.zeros(len(t_cam))
    u = fill_nans(t_cam, axis)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    dt = np.median(np.diff(t_cam))
    du = savgol_filter(u, 7, 2, deriv=1, delta=dt, axis=0)
    return np.linalg.norm(du, axis=1)


def _z(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def estimate_offset(t_cam, cam_rate, t_imu, imu_rate, max_offset_s: float = 0.15,
                    grid_hz: float = 1000.0, window=None) -> tuple:
    """
    Returns (offset_s, peak_correlation). offset is added to IMU times so they line up with
    camera times. `window` = (t0, t1) restricts the comparison (e.g. around the swing).
    """
    t0 = max(t_cam[0], t_imu[0] - max_offset_s)
    t1 = min(t_cam[-1], t_imu[-1] + max_offset_s)
    if window is not None:
        t0, t1 = max(t0, window[0]), min(t1, window[1])
    if t1 - t0 < 0.1:
        return 0.0, 0.0
    grid = np.arange(t0, t1, 1.0 / grid_hz)
    a = _z(np.interp(grid, t_cam, cam_rate))
    # IMU grid extends by max_offset on both sides so every lag sees full overlap
    grid_i = np.arange(t0 - max_offset_s, t1 + max_offset_s, 1.0 / grid_hz)
    b = _z(np.interp(grid_i, t_imu, imu_rate, left=0.0, right=0.0))
    corr = correlate(b, a, mode="valid")
    lags = np.arange(len(corr)) / grid_hz - max_offset_s     # IMU sample at t+lag matches cam t
    k = int(np.argmax(corr))
    peak = corr[k] / len(a)
    lag = lags[k]
    if 0 < k < len(corr) - 1:                      # parabolic sub-sample refinement
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            lag += 0.5 * (y0 - y2) / den / grid_hz
    return float(-lag), float(peak)
