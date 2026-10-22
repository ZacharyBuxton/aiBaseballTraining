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

Backward pass (smooth=True, the default for recorded data): a Rauch-Tung-Striebel smoother
re-runs the estimates from the end of the window to the start, so each sample also uses the
camera fixes that came after it. See rts_smooth().

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
    smoothed: bool = False     # True when the RTS backward pass was applied


def run_ekf(t_imu, gyro, accel, t_cam, axis_cam, up, calib: Calibration, cfg: EkfConfig,
            gyro_sat: float = 69.0, smooth: bool = True) -> EkfResult:
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
    # history for the backward (RTS) pass
    q_pred = np.empty((N, 4)); b_pred = np.empty((N, 3))
    P_pred = np.empty((N, 6, 6)); P_filt = np.empty((N, 6, 6)); Fs = np.zeros((N, 6, 6))
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
        if k == 0:
            q_pred[0], b_pred[0], P_pred[0] = q, b, P
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
            q_pred[k], b_pred[k], P_pred[k], Fs[k] = q, b, P, F

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

        qs[k] = q; bs[k] = b; P_filt[k] = P

    smoothed = False
    if smooth and N > 1:
        qs, bs, P_s = rts_smooth(qs, bs, P_filt, q_pred, b_pred, P_pred, Fs)
        smoothed = True
    else:
        P_s = P_filt
    for k in range(N):
        stds[k] = np.degrees(np.sqrt(max(np.trace(P_s[k][:3, :3]), 0.0)))
        ow[k] = Q.rotate(qs[k], gyro[k] - bs[k])

    axis = Q.rotate(qs, np.broadcast_to(a, (N, 3)))
    return EkfResult(t=np.asarray(t_imu), q=qs, bias=bs, omega_world=ow, axis=axis,
                     att_std_deg=stds, n_cam_updates=n_cam, n_cam_rejected=n_rej,
                     n_grav_updates=n_grav, cam_innov_deg=np.array(innov), smoothed=smoothed)


def rts_smooth(q_f, b_f, P_f, q_p, b_p, P_p, F):
    """
    Rauch-Tung-Striebel backward pass for the error-state filter.

    Recorded data is processed after the fact, so every estimate can use the measurements
    that came AFTER it too. Going backwards from the last sample:
        C_k   = P_k|k F_k+1^T P_k+1|k^-1
        dx    = [ 2 vec(q_pred_k+1^-1 (x) q_smooth_k+1) ,  b_smooth_k+1 - b_pred_k+1 ]
        x_k^s = x_k|k (+) C_k dx                 (quaternion: right-multiply by exp(dtheta))
        P_k^s = P_k|k + C_k (P_k+1^s - P_k+1|k) C_k^T
    Removes the lag/overshoot a forward-only filter shows right after each camera fix and
    through camera gaps (blur, occlusion), where the forward filter can only coast on the gyro.
    """
    N = len(q_f)
    q_s = q_f.copy(); b_s = b_f.copy(); P_s = P_f.copy()
    for k in range(N - 2, -1, -1):
        Pp = P_p[k + 1]
        C = P_f[k] @ F[k + 1].T @ np.linalg.inv(Pp + 1e-15 * np.eye(6))
        dth = Q.to_rotvec(Q.multiply(Q.conj(q_p[k + 1]), q_s[k + 1]))
        dx = C @ np.concatenate([dth, b_s[k + 1] - b_p[k + 1]])
        q_s[k] = Q.normalize(Q.multiply(q_f[k], Q.from_rotvec(dx[:3])))
        b_s[k] = b_f[k] + dx[3:]
        P_s[k] = P_f[k] + C @ (P_s[k + 1] - Pp) @ C.T
    return q_s, b_s, P_s
