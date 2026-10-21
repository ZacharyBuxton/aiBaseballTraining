"""
ekf.py -- multiplicative (error-state) quaternion EKF for the wrist IMU.

Nominal state:  q  (world <- IMU, unit quaternion),  b  (gyro bias, rad/s)
Error state:    dx = [dtheta (3), db (3)],  q_true = q (x) exp(dtheta)

Predict (every IMU sample, ~400 Hz):
    q <- q (x) exp((omega - b) dt)
    F  = [[I - [w]x dt, -I dt], [0, I]]

Updates:
  * camera bat axis (120 Hz):  z = u_cam,  h = R(q) a          H = [-R [a]x , 0]
      a = grip vector from dckft.calib. Noise covers triangulation and wrist-to-bat flex.
      This is what replaces a magnetometer (the ICM45686 has none): the camera pins yaw,
      so chain-link-fence magnetic interference is a non-issue by construction.
  * gravity (accelerometer, only when nearly still):  z = f/|f|,  h = R(q)^T up
                                                      H = [[R^T up]x , 0]
  * innovation gating (chi-square, 3 dof) rejects marker mis-detections.

Quaternions only -- no Euler angles anywhere (gimbal-lock constraint).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat as Q
from .calib import Calibration
from .config import G, EkfConfig

CHI2_3DOF_999 = 16.27


@dataclass
class EkfResult:
    t: np.ndarray              # (N,) IMU times
    q: np.ndarray              # (N,4) world <- IMU
    bias: np.ndarray           # (N,3)
    omega_world: np.ndarray    # (N,3) bias-corrected angular velocity in world frame
    axis: np.ndarray           # (N,3) fused bat axis in world frame
    att_std_deg: np.ndarray    # (N,) sqrt(trace(P_att)) in degrees
    n_cam_updates: int
    n_cam_rejected: int
    n_grav_updates: int
    cam_innov_deg: np.ndarray  # angles of accepted camera innovations


def run_ekf(t_imu, gyro, accel, t_cam, axis_cam, up, calib: Calibration, cfg: EkfConfig,
            gyro_sat: float = 69.0) -> EkfResult:
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = calib.grip_axis
    N = len(t_imu)
    q = calib.q0.copy()
    b = calib.gyro_bias.copy()
    P = np.diag([np.radians(3.0) ** 2] * 3 + [0.01 ** 2] * 3)

    # Camera samples in time order (valid only)
    ok = np.isfinite(axis_cam).all(axis=1)
    tc, uc = t_cam[ok], axis_cam[ok]
    ci = int(np.searchsorted(tc, t_imu[0]))

    sig_u2 = cfg.grip_flex_rad ** 2 + cfg.camera_axis_rad ** 2
    R_cam = np.eye(3) * sig_u2
    R_grav = np.eye(3) * (cfg.gravity_std_ms2 / G) ** 2
    I6 = np.eye(6)

    qs = np.empty((N, 4)); bs = np.empty((N, 3)); stds = np.empty(N); ow = np.empty((N, 3))
    n_cam = n_rej = n_grav = 0
    innov = []

    def update(z, h, H, Rm):
        nonlocal q, b, P
        y = z - h
        S = H @ P @ H.T + Rm
        Si = np.linalg.inv(S)
        d2 = float(y @ Si @ y)
        if d2 > CHI2_3DOF_999 * 4:          # generous: flex can be large mid-swing
            return False
        K = P @ H.T @ Si
        dx = K @ y
        q = Q.normalize(Q.multiply(q, Q.from_rotvec(dx[:3])))
        b = b + dx[3:]
        IKH = I6 - K @ H
        P = IKH @ P @ IKH.T + K @ Rm @ K.T   # Joseph form
        return True

    for k in range(N):
        if k > 0:
            dt = t_imu[k] - t_imu[k - 1]
            w = 0.5 * (gyro[k] + gyro[k - 1]) - b
            q = Q.normalize(Q.multiply(q, Q.from_rotvec(w * dt)))
            F = np.eye(6)
            F[:3, :3] -= Q.skew(w) * dt
            F[:3, 3:] = -np.eye(3) * dt
            sat = np.any(np.abs(gyro[k]) > gyro_sat)
            gn = cfg.gyro_noise * (20.0 if sat else 1.0)
            Qd = np.diag([gn ** 2 * dt] * 3 + [cfg.gyro_bias_walk ** 2 * dt] * 3)
            P = F @ P @ F.T + Qd

            # camera updates whose timestamps fall in (t[k-1], t[k]]
            while ci < len(tc) and tc[ci] <= t_imu[k]:
                R = Q.to_matrix(q)
                h = R @ a
                H = np.zeros((3, 6)); H[:, :3] = -R @ Q.skew(a)
                if update(uc[ci], h, H, R_cam):
                    n_cam += 1
                    innov.append(np.degrees(np.arccos(np.clip(np.dot(uc[ci], h), -1, 1))))
                else:
                    n_rej += 1
                ci += 1

        # gravity update when quasi-static
        fn = np.linalg.norm(accel[k])
        if abs(fn - G) < cfg.gravity_gate_ms2 and np.linalg.norm(gyro[k] - b) < cfg.gravity_gate_rads:
            R = Q.to_matrix(q)
            h = R.T @ up
            H = np.zeros((3, 6)); H[:, :3] = Q.skew(h)
            if update(accel[k] / fn, h, H, R_grav):
                n_grav += 1

        qs[k] = q; bs[k] = b
        stds[k] = np.degrees(np.sqrt(max(np.trace(P[:3, :3]), 0.0)))
        ow[k] = Q.rotate(q, gyro[k] - b)

    axis = Q.rotate(qs, np.broadcast_to(a, (N, 3)))
    return EkfResult(t=np.asarray(t_imu), q=qs, bias=bs, omega_world=ow, axis=axis,
                     att_std_deg=stds, n_cam_updates=n_cam, n_cam_rejected=n_rej,
                     n_grav_updates=n_grav, cam_innov_deg=np.array(innov))
