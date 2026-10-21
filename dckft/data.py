"""
data.py -- load camera (track_bat JSONL) and IMU (reader CSV) recordings into arrays.

Times are float seconds on the host clock (time_ns / 1e9) when the recording has host
stamps, which is the case for everything recorded on the Jetson with the new code.
Legacy IMU CSVs (no ts_ns_host column) get device-relative times, and has_host_clock=False
tells dckft.sync to rely on cross-correlation alone.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import G, BatConfig


# --------------------------------------------------------------------------- IMU
@dataclass
class ImuData:
    t: np.ndarray            # (N,) s
    gyro: np.ndarray         # (N,3) rad/s, IMU body frame
    accel: np.ndarray        # (N,3) m/s^2 specific force, IMU body frame
    accel_hg: np.ndarray     # (N,3) m/s^2 ADXL375
    has_host_clock: bool = True

    def __len__(self):
        return len(self.t)

    def slice(self, t0: float, t1: float) -> "ImuData":
        m = (self.t >= t0) & (self.t <= t1)
        return ImuData(self.t[m], self.gyro[m], self.accel[m], self.accel_hg[m], self.has_host_clock)

    def shifted(self, dt: float) -> "ImuData":
        return ImuData(self.t + dt, self.gyro, self.accel, self.accel_hg, self.has_host_clock)


def unwrap_tx_us(tx: np.ndarray) -> np.ndarray:
    tx = np.asarray(tx, dtype=np.int64)
    d = np.diff(tx)
    wraps = np.concatenate([[0], np.cumsum(d < -(2 ** 31))])
    return tx + wraps * (2 ** 32)


def map_device_clock(tx_us: np.ndarray, host_ns: np.ndarray) -> np.ndarray:
    """
    Map the IMU's microsecond counter onto the host clock.

    Host stamps carry USB/radio batching jitter (several frames share one read time), while the
    device counter is smooth. Fit host = a + b * tx, then shift so the fit hugs the *earliest*
    arrivals (5th percentile of residuals) -- the best estimate of minimum transport latency.
    Returns seconds.
    """
    tx = unwrap_tx_us(tx_us).astype(float) * 1e-6
    host = np.asarray(host_ns, dtype=np.float64) * 1e-9
    h0 = host[0]
    b, a = np.polyfit(tx - tx[0], host - h0, 1)
    fit = a + b * (tx - tx[0])
    resid = (host - h0) - fit
    return h0 + fit + np.percentile(resid, 5)


def load_imu_csv(path: str) -> ImuData:
    import pandas as pd
    df = pd.read_csv(path)
    gyro = df[["omega_x", "omega_y", "omega_z"]].to_numpy(float)
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy(float)
    hg_cols = ["accel_x2", "accel_y2", "accel_z2"]
    accel_hg = df[hg_cols].to_numpy(float) if all(c in df for c in hg_cols) else np.zeros_like(accel)
    if "ts_ns_host" in df and df["ts_ns_host"].notna().all() and (df["ts_ns_host"] > 0).all():
        t = map_device_clock(df["tx_time_stamp"].to_numpy(), df["ts_ns_host"].to_numpy())
        host = True
    else:
        t = df["imu_meas_time"].to_numpy(float)
        host = False
    order = np.argsort(t, kind="stable")
    return ImuData(t[order], gyro[order], accel[order], accel_hg[order], host)


def imu_from_samples(samples: List[dict]) -> ImuData:
    """From dckft.imu.reader ZMQ sample dicts (live mode)."""
    tx = np.array([s["tx_us"] for s in samples])
    host = np.array([s["ts_ns_host"] for s in samples])
    t = map_device_clock(tx, host) if len(samples) > 10 else host * 1e-9
    return ImuData(t, np.array([s["gyro"] for s in samples], float),
                   np.array([s["accel"] for s in samples], float),
                   np.array([s["accel_hg"] for s in samples], float), True)


# ------------------------------------------------------------------------ Camera
@dataclass
class CameraData:
    t: np.ndarray                                  # (M,) s
    points: Dict[str, np.ndarray]                  # marker id -> (M,3), NaN when missing
    gravity_cam: Optional[np.ndarray] = None       # from the ZED IMU, if recorded
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.t)


def payloads_to_camera(payloads: List[dict], marker_ids=None) -> CameraData:
    payloads = sorted(payloads, key=lambda p: p["ts_ns_image"])
    if marker_ids is None:
        marker_ids = sorted({pt["id"] for p in payloads for pt in p.get("points", [])})
    t = np.array([p["ts_ns_image"] for p in payloads], dtype=np.float64) * 1e-9
    pts = {m: np.full((len(payloads), 3), np.nan) for m in marker_ids}
    for i, p in enumerate(payloads):
        for pt in p.get("points", []):
            if pt["id"] in pts:
                pos = pt["pos"]
                pts[pt["id"]][i] = (pos["x"], pos["y"], pos["z"])
    return CameraData(t, pts)


def gravity_from_meta(meta: dict) -> Optional[np.ndarray]:
    """
    Gravity vector in the left-camera OpenCV frame from track_bat's .meta.json.
    The ZED reports specific force (points *up* at rest), so gravity = -accel, rotated by the
    camera<->IMU transform. Returns None if absent or implausible.
    """
    g = (meta or {}).get("gravity")
    if not g or "accel_mean_mps2" not in g:
        return None
    acc = np.asarray(g["accel_mean_mps2"], float)
    T = np.asarray(g.get("camera_imu_transform", np.eye(4)), float)
    R = T[:3, :3] if T.shape == (4, 4) else np.eye(3)
    grav = -(R @ acc)
    n = np.linalg.norm(grav)
    if not (0.8 * G < n < 1.2 * G):
        warnings.warn(f"ZED gravity magnitude {n:.2f} m/s^2 looks wrong; ignoring it")
        return None
    return grav / n * G


def load_camera_jsonl(path: str, marker_ids=None) -> CameraData:
    payloads = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
    cam = payloads_to_camera(payloads, marker_ids)
    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            cam.meta = json.load(f)
        cam.gravity_cam = gravity_from_meta(cam.meta)
    return cam


# --------------------------------------------------------------------- bat line
def bat_line_from_markers(cam: CameraData, bat: BatConfig):
    """
    Per camera frame, solve for the knob point k and unit bat axis u (knob -> tip) from
    collinear markers p_i = k + s_i * u, with s_i the measured distances from the knob.

    >= 2 markers: least squares. Returns (knob (M,3), axis (M,3), n_markers (M,)) with NaN rows
    where fewer than two markers were seen. Roll about the axis is NOT observable from collinear
    markers -- the wrist IMU supplies it through the EKF.
    """
    ids = [m for m in bat.marker_s_m if m in cam.points]
    if len(ids) < 2:
        raise ValueError(f"Need >= 2 configured markers in the camera data; have {ids}. "
                         f"Configured: {list(bat.marker_s_m)}; seen: {list(cam.points)}")
    P = np.stack([cam.points[m] for m in ids], axis=1)         # (M, K, 3)
    s = np.array([bat.marker_s_m[m] for m in ids])             # (K,)
    M = len(cam.t)
    knob = np.full((M, 3), np.nan)
    axis = np.full((M, 3), np.nan)
    count = np.zeros(M, dtype=int)
    for i in range(M):
        ok = np.isfinite(P[i]).all(axis=1)
        n = int(ok.sum())
        count[i] = n
        if n < 2:
            continue
        si, pi = s[ok], P[i][ok]
        # Linear LS on [k; v] with p_i = k + s_i v, then u = v/|v|
        A = np.zeros((3 * n, 6))
        for j in range(n):
            A[3 * j:3 * j + 3, :3] = np.eye(3)
            A[3 * j:3 * j + 3, 3:] = si[j] * np.eye(3)
        sol, *_ = np.linalg.lstsq(A, pi.reshape(-1), rcond=None)
        v = sol[3:]
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            continue
        u = v / nv
        axis[i] = u
        # Re-solve k with unit-length u (the marker spacing is known, not a free scale)
        knob[i] = np.mean(pi - si[:, None] * u, axis=0)
    return knob, axis, count
