"""
calib.py -- per-swing self-calibration of the wrist IMU against the camera.

Unknowns (the IMU is on the wrist, not the bat, so these change every time the batter re-grips):
  q0  orientation of the IMU in the camera/world frame at the start of the window
  a   the bat axis (knob -> tip) expressed in the IMU frame  ("grip vector")
  b   gyro bias

Measurements used:
  * gyro, integrated from the window start:   R(t) = R0 * dR(t)
  * camera bat axis (collinear markers):      u_cam(t) ~= R(t) a
  * accelerometer while nearly still (stance): R(t) f(t)/|f| ~= up
  * plus a small residual camera/IMU time offset, solved jointly (the xcorr estimate compares
    different signals -- wrist |omega| vs bat-axis rate -- so it can be biased by a few ms,
    which at ~1000 rad/s^2 near contact is several % of bat speed)

Solved by robust nonlinear least squares (multi-start, since yaw and grip are only weakly
constrained by gravity alone). The result seeds dckft.ekf, which then tracks q(t) and b(t)
through the swing while still treating a as fixed (grip flex is absorbed as measurement noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import quat as Q
from .config import G, EkfConfig


@dataclass
class Calibration:
    q0: np.ndarray            # world <- IMU at t_ref
    t_ref: float
    grip_axis: np.ndarray     # bat axis in IMU frame (unit)
    gyro_bias: np.ndarray
    rms_axis_deg: float
    rms_gravity_deg: float
    n_axis: int
    n_gravity: int
    dt_refine: float = 0.0    # extra seconds to ADD to IMU times (joint time-offset estimate)


def still_mask(gyro, accel, cfg: EkfConfig):
    return ((np.abs(np.linalg.norm(accel, axis=1) - G) < cfg.gravity_gate_ms2)
            & (np.linalg.norm(gyro, axis=1) < cfg.gravity_gate_rads))


def _interp_quats(t_src, q_src, t_dst):
    """Component-wise linear interpolation + renormalise (fine at 400 Hz)."""
    out = np.stack([np.interp(t_dst, t_src, q_src[:, i]) for i in range(4)], axis=1)
    return Q.normalize(out)


def _sph(a):
    a = a / np.linalg.norm(a)
    return np.array([np.arccos(np.clip(a[2], -1, 1)), np.arctan2(a[1], a[0])])


def _unsph(p):
    th, ph = p
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def calibrate(t_imu, gyro, accel, t_cam, axis_cam, up, t_end_axis, cfg: EkfConfig,
              max_dt_s: float = 0.015) -> Calibration:
    """
    t_imu/gyro/accel: IMU window (already time-aligned to the camera clock).
    t_cam/axis_cam:   camera times and unit bat-axis rows (NaN rows allowed).
    up:               unit up vector in world (camera) frame.
    t_end_axis:       only camera samples before this time are used (keep gyro drift small;
                      typically the contact time).
    """
    up = np.asarray(up, float) / np.linalg.norm(up)
    still = still_mask(gyro, accel, cfg)
    bias = np.median(gyro[still], axis=0) if (cfg.init_bias_from_stance and still.sum() >= 20) \
        else np.zeros(3)

    dt = np.diff(t_imu, prepend=t_imu[0])
    q_rel = Q.integrate_gyro(np.array([1.0, 0, 0, 0]), gyro - bias, dt)

    cam_ok = np.isfinite(axis_cam).all(axis=1) & (t_cam >= t_imu[0]) & (t_cam <= min(t_end_axis, t_imu[-1]))
    tc, uc = t_cam[cam_ok], axis_cam[cam_ok]
    if len(tc) < 5:
        raise ValueError("Not enough camera bat-axis samples to calibrate the IMU "
                         f"({len(tc)} usable). Check marker visibility / time sync.")
    def Rc_at(dt_s):
        # IMU times are shifted by +dt_s, so camera time tc sees IMU state at tc - dt_s
        return Q.to_matrix(_interp_quats(t_imu, q_rel, tc - dt_s))

    gi = np.where(still)[0]
    if len(gi) > 200:
        gi = gi[np.linspace(0, len(gi) - 1, 200).astype(int)]
    Rg = Q.to_matrix(q_rel[gi])
    fg = accel[gi] / np.linalg.norm(accel[gi], axis=1, keepdims=True)

    sig_u = np.hypot(cfg.grip_flex_rad, cfg.camera_axis_rad)
    sig_g = cfg.gravity_std_ms2 / G
    w_u, w_g = 1.0 / sig_u, 1.0 / sig_g
    # Down-weight gravity so a long stance doesn't swamp the (fewer) camera samples
    w_g *= np.sqrt(len(tc) / max(len(gi), 1)) if len(gi) else 0.0

    def resid(x):
        return resid_with(x, Rc_cache)

    def resid_with(x, Rc):
        R0 = Q.to_matrix(Q.from_rotvec(x[:3]))
        a = _unsph(x[3:5])
        pred_u = np.einsum("ij,njk,k->ni", R0, Rc, a)
        r = [(pred_u - uc).ravel() * w_u]
        if len(gi):
            pred_up = np.einsum("ij,njk,nk->ni", R0, Rg, fg)
            r.append((pred_up - up).ravel() * w_g)
        return np.concatenate(r)

    Rc_cache = Rc_at(0.0)
    if len(gi):
        # Gravity fixes tilt; scan yaw. For a given R0 the best grip vector has a closed form
        # (mean of the camera axes mapped into the IMU frame), so the scan is cheap.
        f_w = np.mean(np.einsum("nij,nj->ni", Rg, fg), axis=0)   # mean 'up' in the dR frame
        q_tilt = Q.from_two_vectors(f_w, up)
        scored = []
        for yaw in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            q0_c = Q.multiply(Q.from_rotvec(up * yaw), q_tilt)
            RR = np.einsum("ij,njk->nik", Q.to_matrix(q0_c), Rc_cache)   # world <- IMU, cam times
            v = np.einsum("nji,nj->ni", RR, uc)                         # camera axes, IMU frame
            a0 = v.mean(0)
            if np.linalg.norm(a0) < 1e-6:
                continue
            a0 /= np.linalg.norm(a0)
            c = float(np.sum((np.einsum("nij,j->ni", RR, a0) - uc) ** 2))
            scored.append((c, np.concatenate([Q.to_rotvec(q0_c), _sph(a0)])))
        scored.sort(key=lambda z: z[0])
        cands = [z[1] for z in scored[:3]]
    else:
        # No stance data: generic multi-start over yaw x grip directions.
        starts = []
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            r0 = up * yaw
            for a0 in np.vstack([np.eye(3), -np.eye(3)]):
                starts.append(np.concatenate([r0, _sph(a0 + 1e-3)]))
        coarse = sorted((least_squares(resid, x0, loss="soft_l1", f_scale=2.0, max_nfev=25)
                         for x0 in starts), key=lambda s: s.cost)
        cands = [c.x for c in coarse[:3]]
    best = min((least_squares(resid, x0, loss="soft_l1", f_scale=2.0, max_nfev=200)
                for x0 in cands), key=lambda s: s.cost)
    x = best.x
    dt_ref = 0.0
    if max_dt_s > 0:
        # 1-D bounded search on the time offset, re-solving orientation/grip at each step
        from scipy.optimize import minimize_scalar
        cache = {}

        def cost(d):
            Rd = Rc_at(d)
            f = lambda xx: resid_with(xx, Rd)
            sol = least_squares(f, x, loss="soft_l1", f_scale=2.0, max_nfev=12)
            cache[d] = sol.x
            return sol.cost

        r = minimize_scalar(cost, bounds=(-max_dt_s, max_dt_s), method="bounded",
                            options={"xatol": 3e-4, "maxiter": 14})
        if r.fun < best.cost:
            dt_ref, x = float(r.x), cache[r.x]
    q0 = Q.from_rotvec(x[:3])
    a = _unsph(x[3:5])

    R0 = Q.to_matrix(q0)
    Rc = Rc_at(dt_ref)
    pu = np.einsum("ij,njk,k->ni", R0, Rc, a)
    ang_u = np.degrees(np.arccos(np.clip(np.sum(pu * uc, 1), -1, 1)))
    if len(gi):
        pg = np.einsum("ij,njk,nk->ni", R0, Rg, fg)
        ang_g = np.degrees(np.arccos(np.clip(pg @ up, -1, 1)))
        rms_g = float(np.sqrt(np.mean(ang_g ** 2)))
    else:
        rms_g = float("nan")
    return Calibration(q0=q0, t_ref=float(t_imu[0]), grip_axis=a, gyro_bias=bias,
                       rms_axis_deg=float(np.sqrt(np.mean(ang_u ** 2))), rms_gravity_deg=rms_g,
                       n_axis=len(tc), n_gravity=len(gi), dt_refine=dt_ref)
