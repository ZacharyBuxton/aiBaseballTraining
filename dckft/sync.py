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
    t_cam, axis = np.asarray(t_cam), np.asarray(axis)
    out = np.full(len(t_cam), np.nan)
    indices = np.flatnonzero(np.isfinite(axis).all(axis=1))
    groups = np.split(indices, np.where(np.diff(t_cam[indices]) > .06)[0]+1)
    for ix in groups:
        if len(ix) < 3:
            continue
        tc, u = t_cam[ix], axis[ix]
        if np.any(np.diff(tc) <= 0):
            continue
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        # Differentiate on the real timestamps. A frame-count derivative exaggerates
        # motion when published frames are irregular or absent.
        du = np.gradient(u, tc-tc[0], axis=0, edge_order=2)
        out[ix] = np.linalg.norm(du, axis=1)
    return out


def _supported_interp(times, values, grid, max_gap):
    times, values = np.asarray(times), np.asarray(values)
    ok = np.isfinite(times) & np.isfinite(values)
    times, unique = np.unique(times[ok], return_index=True)
    values = values[ok][unique]
    if len(times) < 2:
        return np.full(len(grid), np.nan)
    result = np.interp(grid, times, values, left=np.nan, right=np.nan)
    right = np.clip(np.searchsorted(times, grid), 1, len(times)-1)
    result[(times[right]-times[right-1]) > max_gap] = np.nan
    return result


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
    a = _supported_interp(t_cam, cam_rate, grid, .06)
    lags = np.arange(-round(max_offset_s*grid_hz), round(max_offset_s*grid_hz)+1) / grid_hz
    scores = np.full(len(lags), -np.inf)
    required = max(50, int(np.isfinite(a).sum()*.5))
    for i, lag in enumerate(lags):
        b = _supported_interp(t_imu, imu_rate, grid+lag, .02)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= required and a[ok].std() > 1e-6 and b[ok].std() > 1e-6:
            scores[i] = float(np.mean(_z(a[ok])*_z(b[ok])))
    if not np.isfinite(scores).any():
        return 0.0, 0.0
    k = int(np.argmax(scores))
    return float(-lags[k]), float(np.clip(scores[k], -1, 1))

