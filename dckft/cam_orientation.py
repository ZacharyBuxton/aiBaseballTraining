"""
cam_orientation.py -- bat orientation and angular velocity from the camera alone.

Per camera frame, the pink band centroids (pink_1..N, triangulated) give the bat line, so its
unit axis u(t) (knob -> tip). That is turned into a quaternion q(t) and differentiated:

    q_k   = shortest_arc(u_{k-1} -> u_k) (x) q_{k-1}      (q_0 = shortest_arc(x_bat -> u_0))
    q(t)  : smoothing spline through the q_k components, renormalised
    omega = 2 * qdot (x) q*      (vector part, world frame, rad/s)

because a body->world quaternion obeys qdot = 1/2 omega (x) q. Building q by shortest arcs
adds no spin about the bat's own axis -- roll about the bat is invisible to collinear bands,
so omega here is the bat's angular velocity PERPENDICULAR to its axis. The wrist IMU supplies
roll; this camera-only estimate is an independent check on the fused/IMU rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_smoothing_spline

from . import quat as Q


@dataclass
class CamOrientation:
    t: np.ndarray            # (M,) camera frame times with a bat line
    q: np.ndarray            # (M,4) bat orientation (body x = bat axis), world <- bat
    omega: np.ndarray        # (M,3) world-frame angular velocity from the quaternion rate, rad/s
    axis: np.ndarray         # (M,3) smoothed bat axis = R(q) x

    def omega_at(self, t):
        return np.column_stack([np.interp(t, self.t, self.omega[:, j]) for j in range(3)])


def quaternions_from_axes(axes: np.ndarray) -> np.ndarray:
    """Minimal-twist quaternion sequence whose body x-axis follows the given unit axes."""
    ex = np.array([1.0, 0.0, 0.0])
    q = np.empty((len(axes), 4))
    q[0] = Q.from_two_vectors(ex, axes[0])
    for k in range(1, len(axes)):
        q[k] = Q.normalize(Q.multiply(Q.from_two_vectors(axes[k - 1], axes[k]), q[k - 1]))
        if np.dot(q[k], q[k - 1]) < 0:          # keep the sign continuous for the spline
            q[k] = -q[k]
    return q


def camera_orientation(t_cam: np.ndarray, axis_cam: np.ndarray, lam: float = 2e-7) -> CamOrientation:
    ok = np.isfinite(axis_cam).all(axis=1)
    t, u = np.asarray(t_cam)[ok], np.asarray(axis_cam)[ok]
    if len(t) < 6:
        raise ValueError(f"Only {len(t)} camera bat lines: not enough for a camera orientation")
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    q_raw = quaternions_from_axes(u)
    t0 = t[0]
    q = np.empty_like(q_raw)
    qd = np.empty_like(q_raw)
    for j in range(4):
        spl = make_smoothing_spline(t - t0, q_raw[:, j], lam=lam)
        q[:, j] = spl(t - t0)
        qd[:, j] = spl.derivative()(t - t0)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / n
    qd = qd / n - q * np.sum(q * qd / n, axis=1, keepdims=True)   # derivative of the normalised q
    omega = 2.0 * Q.multiply(qd, Q.conj(q))[:, 1:]
    axis = Q.rotate(q, np.broadcast_to([1.0, 0.0, 0.0], (len(q), 3)))
    return CamOrientation(t=t, q=q, omega=omega, axis=axis)


def compare_with_imu(co: CamOrientation, t_imu: np.ndarray, omega_imu_world: np.ndarray,
                     axis_imu: np.ndarray, window=None) -> dict:
    """
    Camera-quaternion rate vs the fused wrist-IMU rate, both reduced to the component
    perpendicular to the bat (the only part the camera sees). Returns RMS difference (rad/s),
    the correlation of the magnitudes, and the sample count.
    """
    m = np.ones(len(co.t), bool) if window is None else (co.t >= window[0]) & (co.t <= window[1])
    m &= (co.t >= t_imu[0]) & (co.t <= t_imu[-1])
    if m.sum() < 5:
        return {"n": int(m.sum())}
    w_i = np.column_stack([np.interp(co.t[m], t_imu, omega_imu_world[:, j]) for j in range(3)])
    u_i = np.column_stack([np.interp(co.t[m], t_imu, axis_imu[:, j]) for j in range(3)])
    u_i /= np.linalg.norm(u_i, axis=1, keepdims=True)
    w_i_perp = w_i - np.sum(w_i * u_i, axis=1, keepdims=True) * u_i
    w_c = co.omega[m]
    w_c_perp = w_c - np.sum(w_c * u_i, axis=1, keepdims=True) * u_i
    a, b = np.linalg.norm(w_c_perp, axis=1), np.linalg.norm(w_i_perp, axis=1)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"n": int(m.sum()),
            "rms_diff_rads": round(float(np.sqrt(np.mean(np.sum((w_c_perp - w_i_perp) ** 2, 1)))), 3),
            "magnitude_correlation": round(corr, 3),
            "peak_camera_rads": round(float(a.max()), 2), "peak_imu_rads": round(float(b.max()), 2)}


def camera_angles(t_cam, knob_cam, axis_cam, t_contact: float, up, sweet_s: float,
                  max_gap_s: float = 0.06, co: "CamOrientation" = None) -> tuple:
    """
    Attack angle and vertical bat angle at contact from the camera ALONE (pink centroids),
    as a cross-check on the fused values. Needs camera bat lines within max_gap_s on both
    sides of contact; returns (attack_deg, vba_deg), NaN where the camera can't support it.
      VBA: bat axis from the camera quaternion at contact.
      AoA: sweet-spot velocity = knob velocity + s * (omega x u), with omega from the camera
           quaternion rate (2 qdot q*) -- the bat turns ~100 deg in 60 ms near contact, so
           fitting the sweet-spot path directly would cut the corner.
    """
    up = np.asarray(up, float) / np.linalg.norm(up)
    ok = np.isfinite(axis_cam).all(axis=1) & np.isfinite(knob_cam).all(axis=1)
    t, u, k = np.asarray(t_cam)[ok], np.asarray(axis_cam)[ok], np.asarray(knob_cam)[ok]
    nan = float("nan")
    before, after = t[t <= t_contact], t[t >= t_contact]
    if not len(before) or not len(after) or t_contact - before[-1] > max_gap_s \
            or after[0] - t_contact > max_gap_s:
        return nan, nan
    ang = lambda v: float(np.degrees(np.arcsin(np.clip(np.dot(v, up) / np.linalg.norm(v), -1, 1))))
    if co is None:
        co = camera_orientation(t, u)
    u_c = np.array([np.interp(t_contact, co.t, co.axis[:, j]) for j in range(3)])
    u_c /= np.linalg.norm(u_c)
    vba = ang(u_c)
    m = np.abs(t - t_contact) <= 0.06
    if m.sum() < 4:
        return nan, vba
    tt = t[m] - t_contact
    k_vel = np.array([np.polyfit(tt, k[m][:, j], 2)[1] for j in range(3)])   # knob d/dt at contact
    w = co.omega_at(np.array([t_contact]))[0]
    v = k_vel + sweet_s * np.cross(w, u_c)
    return ang(v), vba
