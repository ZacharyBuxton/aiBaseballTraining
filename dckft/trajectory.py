"""
trajectory.py -- bat point positions/velocities at IMU rate.

  knob(t):  smoothing spline through the camera's knob estimates (120 Hz -> any rate);
            the spline derivative gives knob velocity.
  axis(t):  fused bat axis from the EKF (400 Hz), so blur/occlusion gaps in the camera
            are bridged by the wrist gyro.

  p(s, t) = knob(t) + s * u(t)
  v(s, t) = knob'(t) + s * (omega_world(t) x u(t))       (rigid-grip assumption)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_smoothing_spline


@dataclass
class Trajectory:
    t: np.ndarray
    knob: np.ndarray
    knob_vel: np.ndarray
    axis: np.ndarray
    axis_rate: np.ndarray     # du/dt

    def point(self, s: float) -> np.ndarray:
        return self.knob + s * self.axis

    def velocity(self, s: float) -> np.ndarray:
        return self.knob_vel + s * self.axis_rate

    def speed(self, s: float) -> np.ndarray:
        return np.linalg.norm(self.velocity(s), axis=1)


def smooth_positions(t_cam, xyz, t_out, lam=None):
    """Per-axis smoothing spline (GCV-chosen smoothing if lam is None). Returns pos, vel."""
    ok = np.isfinite(xyz).all(axis=1)
    tc, pc = t_cam[ok], xyz[ok]
    if len(tc) < 6:
        raise ValueError(f"Only {len(tc)} valid knob positions -- not enough to build a trajectory")
    # Normalise time for numerical stability of the spline solve
    t0 = tc[0]
    pos = np.empty((len(t_out), 3))
    vel = np.empty((len(t_out), 3))
    for j in range(3):
        spl = make_smoothing_spline(tc - t0, pc[:, j], lam=lam)
        pos[:, j] = spl(t_out - t0)
        vel[:, j] = spl.derivative()(t_out - t0)
    # Don't extrapolate wildly beyond the camera coverage
    outside = (t_out < tc[0]) | (t_out > tc[-1])
    if outside.any():
        pos[outside] = np.array([np.interp(t_out[outside], tc, pc[:, j]) for j in range(3)]).T
        vel[outside] = 0.0
    return pos, vel


def kalman_knob(t_imu, acc_world, t_cam, knob_cam, meas_std_m=0.006, acc_std=25.0,
                p0_std=0.05, acc_valid=None, invalid_acc_std=400.0):
    """
    Accel-aided knob track: constant-velocity model driven by the wrist IMU's world-frame
    linear acceleration (R(q) f + g), corrected by the camera knob at 120 Hz, then an RTS
    smoother pass. The IMU bridges blur/occlusion gaps; acc_std absorbs the wrist-to-knob
    lever-arm difference. Samples with acc_valid False (impact ringing, accelerometer
    saturation) are not used as input -- the filter coasts on the camera instead.
    Returns (pos, vel) at t_imu.
    """
    if acc_valid is None:
        acc_valid = np.ones(len(t_imu), bool)
    N = len(t_imu)
    ok = np.isfinite(knob_cam).all(axis=1)
    tc, zc = t_cam[ok], knob_cam[ok]
    if len(tc) < 3:
        raise ValueError("Not enough knob fixes for the trajectory")
    # measurement index per IMU step: nearest IMU sample at/after each camera time
    meas = {}
    for i, ti in enumerate(tc):
        k = int(np.searchsorted(t_imu, ti))
        if 0 <= k < N:
            meas[k] = zc[i]
    x = np.zeros((N, 6)); Pp = np.zeros((N, 6, 6))
    xf = np.zeros((N, 6)); Pf = np.zeros((N, 6, 6)); Fs = np.zeros((N, 6, 6))
    xk = np.concatenate([zc[0], np.zeros(3)])
    Pk = np.diag([p0_std ** 2] * 3 + [2.0 ** 2] * 3)
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    Rm = np.eye(3) * meas_std_m ** 2
    I3 = np.eye(3)
    for k in range(N):
        if k > 0:
            dt = t_imu[k] - t_imu[k - 1]
            F = np.eye(6); F[:3, 3:] = I3 * dt
            good = acc_valid[k] and acc_valid[k - 1]
            a = 0.5 * (acc_world[k] + acc_world[k - 1]) if good else np.zeros(3)
            xk = F @ xk + np.concatenate([0.5 * a * dt * dt, a * dt])
            G = np.vstack([0.5 * dt * dt * I3, dt * I3])
            Pk = F @ Pk @ F.T + G @ G.T * (acc_std if good else invalid_acc_std) ** 2
            Fs[k] = F
        x[k], Pp[k] = xk, Pk
        if k in meas:
            y = meas[k] - H @ xk
            S = H @ Pk @ H.T + Rm
            K = Pk @ H.T @ np.linalg.inv(S)
            if y @ np.linalg.solve(S, y) < 50.0:
                xk = xk + K @ y
                Pk = (np.eye(6) - K @ H) @ Pk
        xf[k], Pf[k] = xk, Pk
    xs = xf.copy()
    for k in range(N - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - x[k + 1])
    return xs[:, :3], xs[:, 3:]


def flex_correction(t_imu, u_imu, t_cam, axis_cam, lam=2e-6, max_gap_s=0.02):
    """
    Complementary smoother for the wrist-to-bat flex.

    The EKF axis is u_imu = R(q) a with a fixed grip vector, so when the wrist hinges/releases
    the bat turns relative to the IMU and u_imu lags/leads the real bat. The camera sees the real
    bat axis (noisy, 120 Hz, gaps). The small rotation d(t) taking u_imu onto the camera axis is
    smooth (it's a joint angle), so: rotation vectors at camera times -> smoothing spline ->
    evaluate d(t) and d'(t) at IMU rate. High frequencies come from the gyro, the slowly
    varying flex from the camera.

    Returns (u_corrected, extra_rate) where extra_rate = d'(t) (world rad/s) to add to omega.
    """
    ok = np.isfinite(axis_cam).all(axis=1) & (t_cam >= t_imu[0]) & (t_cam <= t_imu[-1])
    if ok.sum() < 8:
        return u_imu, np.zeros_like(u_imu)
    tc, uc = t_cam[ok], axis_cam[ok]
    ui = np.column_stack([np.interp(tc, t_imu, u_imu[:, j]) for j in range(3)])
    ui /= np.linalg.norm(ui, axis=1, keepdims=True)
    cr = np.cross(ui, uc)
    s = np.linalg.norm(cr, axis=1)
    ang = np.arctan2(s, np.sum(ui * uc, axis=1))
    rv = cr * (ang / np.where(s > 1e-12, s, 1.0))[:, None]
    d, dd = smooth_positions(tc, rv, t_imu, lam)
    # Inside camera gaps (blur/occlusion near contact) a spline can bow; bridge each gap with a
    # straight line between its edges instead (constant mean flex rate across the gap).
    gaps = np.where(np.diff(tc) > max_gap_s)[0]
    for g in gaps:
        a0, a1 = tc[g], tc[g + 1]
        m = (t_imu > a0) & (t_imu < a1)
        if not m.any():
            continue
        k0 = int(np.searchsorted(t_imu, a0)); k1 = min(int(np.searchsorted(t_imu, a1)), len(t_imu) - 1)
        d0, d1 = d[max(k0 - 1, 0)], d[k1]
        frac = ((t_imu[m] - a0) / (a1 - a0))[:, None]
        d[m] = d0 + frac * (d1 - d0)
        dd[m] = (d1 - d0) / (a1 - a0)
    from . import quat as Q
    u_corr = Q.rotate(Q.from_rotvec(d), u_imu)
    return u_corr, dd


def build_trajectory(t_imu, axis, omega_world, t_cam, knob_cam, lam=None,
                     acc_world=None, axis_cam=None, flex_lam=2e-6, acc_valid=None) -> Trajectory:
    if acc_world is not None:
        knob, knob_vel = kalman_knob(t_imu, acc_world, t_cam, knob_cam, acc_valid=acc_valid)
    else:
        knob, knob_vel = smooth_positions(t_cam, knob_cam, t_imu, lam)
    u = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    omega = np.asarray(omega_world, float)
    if axis_cam is not None and flex_lam:
        u, extra = flex_correction(t_imu, u, t_cam, axis_cam, flex_lam)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        omega = omega + extra
    du = np.cross(omega, u)
    return Trajectory(t=np.asarray(t_imu), knob=knob, knob_vel=knob_vel, axis=u, axis_rate=du)
