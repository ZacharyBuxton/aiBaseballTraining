"""
simulate.py -- synthetic swing generator with ground truth.

Produces exactly what the real rig records, so the whole pipeline (and the live runner via
replay) can be developed, tested and validated without the Jetson, camera or radio:

    <out>/camera.jsonl            track_bat.py payloads (120 Hz, collinear markers, noise, blur dropouts)
    <out>/camera.jsonl.meta.json  ZED gravity estimate in the same shape track_bat writes
    <out>/imu.csv                 dckft.imu.reader CSV (400 Hz wrist IMU, bias, noise, quantisation,
                                  saturation, contact spike on the ADXL375, radio latency + batching)
    <out>/truth.json              true metrics + simulation parameters

    python -m dckft.simulate --out sessions/sim01 --seed 1

Physical model (batter frame: z up):
  * bat axis = elevation phi(t), azimuth psi(t); roll rho(t) about the bat (forearm rotation)
  * swing progress p(t) accelerates quadratically to contact, then decelerates in follow-through
  * knob moves on a ~0.25 m arc around the body and drops as the swing starts
  * wrist IMU = bat orientation (x) grip rotation (x) time-varying wrist "flex" (non-rigid grip)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from . import quat as Q
from .config import G, FusionConfig
from .imu.frames import CSV_COLUMNS, decode_frame, encode_frame
from .metrics import compute_metrics
from .swing import detect_swings
from .trajectory import Trajectory

DEG = np.pi / 180


@dataclass
class SimParams:
    seed: int = 0
    t_contact_from_start: float = 0.15     # motion start -> contact
    follow_s: float = 0.30
    stance_s: float = 1.2
    tail_s: float = 0.6
    psi0_deg: float = 165.0                # azimuth at stance (bat back over the shoulder)
    psic_deg: float = -5.0                 # azimuth at contact
    phi0_deg: float = 55.0                 # elevation at stance
    phic_deg: float = -22.0                # elevation at contact (vertical bat angle)
    roll_deg: float = 80.0                 # forearm roll over the swing
    uphill_deg: float = 25.0               # barrel rise rate into contact (sets attack angle)
    knob_radius_m: float = 0.24
    knob_drop_m: float = 0.25
    flex_deg: float = 6.0                  # wrist-to-bat non-rigidity amplitude
    cam_noise_xy_m: float = 0.002
    cam_noise_z_m: float = 0.006
    cam_dropout: float = 0.03
    blur_speed_mps: float = 28.0           # markers faster than this drop out 50% of the time
    occlusion_frames: int = 3              # all markers lost for N frames just before contact
    gyro_noise: float = 0.01
    gyro_bias_std: float = 0.03
    accel_noise: float = 0.05
    imu_latency_s: float = 0.006
    imu_batch: int = 4
    contact: bool = True
    camera_distance_m: float = 2.6


def _q_axis(axis, ang):
    axis = np.asarray(axis, float)
    return Q.from_rotvec(np.outer(np.atleast_1d(ang), axis).squeeze())


def simulate(params: SimParams = SimParams(), cfg: FusionConfig = FusionConfig()):
    rng = np.random.default_rng(params.seed)
    P = params
    fs = 2000.0
    T, Tf = P.t_contact_from_start, P.follow_s
    t_rel = np.arange(-P.stance_s, T + Tf + P.tail_s, 1.0 / fs)
    t_base = 1.79e9 + rng.uniform(0, 1000)          # host clock epoch (s)
    t = t_base + t_rel + P.stance_s

    # progress p(t): p' = 2t/T^2 on [0,T], then linear decay to 0 over Tf
    pdot = np.where((t_rel >= 0) & (t_rel <= T), 2 * t_rel / T ** 2, 0.0)
    m = (t_rel > T) & (t_rel <= T + Tf)
    pdot[m] = (2 / T) * (1 - (t_rel[m] - T) / Tf)
    p = np.cumsum(pdot) / fs

    sway = 2.0 * DEG * np.sin(2 * np.pi * 0.9 * t_rel) * (t_rel < 0)
    psi = P.psi0_deg * DEG + (P.psic_deg - P.psi0_deg) * DEG * p + sway
    # elevation drops fast early, then the barrel works back up into contact (+ attack angle)
    pc = np.clip(p, 0, None)
    phi = (P.phic_deg * DEG + (P.phi0_deg - P.phic_deg) * DEG * (1 - np.clip(pc, 0, 1)) ** 2
           + P.uphill_deg * DEG * (pc - 1) + 0.5 * sway)
    rho = P.roll_deg * DEG * p

    ez, ey, ex = np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
    q_bat_b = Q.multiply(Q.multiply(_q_axis(ez, psi), _q_axis(ey, -phi)), _q_axis(ex, rho))

    # batter frame -> camera frame (camera: x right, y down, z forward; batter z = up = -y_cam)
    M = np.column_stack([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    q_M = Q.from_matrix(M)
    q_bat = Q.multiply(q_M, q_bat_b)
    centre = np.array([rng.uniform(-0.2, 0.2), 0.15, P.camera_distance_m])

    alpha = psi - 1.3
    knob_b = np.column_stack([P.knob_radius_m * np.cos(alpha), P.knob_radius_m * np.sin(alpha),
                              0.25 - P.knob_drop_m * (2 * np.clip(p, 0, 1) - np.clip(p, 0, 1) ** 2)])
    knob_b[:, 0] += 0.01 * np.sin(2 * np.pi * 0.7 * t_rel) * (t_rel < 0)
    knob = centre + knob_b @ M.T
    u = Q.rotate(q_bat, np.broadcast_to(ex, (len(t), 3)))

    # wrist IMU orientation: bat (x) grip (x) flex(t)
    grip_rv = rng.normal(size=3)
    grip_rv *= rng.uniform(0.5, 2.5) / np.linalg.norm(grip_rv)
    q_grip = Q.from_rotvec(grip_rv)
    flex_axis = rng.normal(size=3); flex_axis /= np.linalg.norm(flex_axis)
    flex_ang = P.flex_deg * DEG * np.sin(np.pi * np.clip(p, 0, 1.5) / 1.5)
    q_flex = _q_axis(flex_axis, flex_ang)
    q_imu = Q.multiply(Q.multiply(q_bat, q_grip), q_flex)

    # true body rates and specific force at the wrist
    dq = Q.multiply(Q.conj(q_imu[:-1]), q_imu[1:])
    w_body = np.vstack([Q.to_rotvec(dq) * fs, np.zeros((1, 3))])
    w_body = 0.5 * (w_body + np.vstack([w_body[:1], w_body[:-1]]))
    pos_imu = knob + 0.05 * u
    acc_w = np.gradient(np.gradient(pos_imu, 1 / fs, axis=0), 1 / fs, axis=0)
    g_cam = np.array([0.0, G, 0.0])
    f_body = Q.rotate(Q.conj(q_imu), acc_w - g_cam)
    w_world_true = Q.rotate(q_imu, w_body)

    # ---------------------------------------------------------------- IMU stream
    imu_cfg = cfg.imu
    t_dev = np.arange(t[0] + 0.05, t[-1] - 0.05, 1.0 / imu_cfg.nominal_rate_hz)
    gi = lambda arr: np.column_stack([np.interp(t_dev, t, arr[:, j]) for j in range(arr.shape[1])])
    gyro = gi(w_body)
    acc = gi(f_body)
    bias = rng.normal(scale=P.gyro_bias_std, size=3)
    gyro_m = gyro + bias + rng.normal(scale=P.gyro_noise, size=gyro.shape)
    acc_m = acc + rng.normal(scale=P.accel_noise, size=acc.shape)
    hg = acc + rng.normal(scale=2.0, size=acc.shape)
    t_c_true = t_base + P.stance_s + T
    if P.contact:
        hit = (t_dev >= t_c_true) & (t_dev < t_c_true + 0.004)
        spike_dir = rng.normal(size=3); spike_dir /= np.linalg.norm(spike_dir)
        hg[hit] += 700.0 * spike_dir
        acc_m[hit] += 400.0 * spike_dir
        after = (t_dev >= t_c_true) & (t_dev < t_c_true + 0.02)
        gyro_m[after] += rng.normal(scale=1.5, size=(after.sum(), 3))

    # through the real binary frame format (quantisation + saturation), device clock w/ drift
    tx0 = int(rng.uniform(0, 2 ** 32 - 1))
    drift = 1 + rng.normal(scale=30e-6)
    rows = []
    for k in range(len(t_dev)):
        tx = (tx0 + int(round((t_dev[k] - t_dev[0]) * 1e6 * drift))) & 0xFFFFFFFF
        fr = encode_frame(tx, acc_m[k], gyro_m[k], hg[k], (0, 0, 0), imu_cfg)
        rows.append(decode_frame(fr, imu_cfg))
    batch_end = np.minimum((np.arange(len(t_dev)) // P.imu_batch + 1) * P.imu_batch - 1, len(t_dev) - 1)
    host = t_dev[batch_end] + P.imu_latency_s + np.abs(rng.normal(scale=0.001, size=len(t_dev)))
    host = np.maximum.accumulate(host)

    # ---------------------------------------------------------------- camera stream
    t_img = np.arange(t[0] + 0.02, t[-1] - 0.02, 1 / 120.0)
    payloads = []
    gc = lambda arr, tt: np.column_stack([np.interp(tt, t, arr[:, j]) for j in range(3)])
    knob_c, u_c = gc(knob, t_img), gc(u, t_img)
    u_c /= np.linalg.norm(u_c, axis=1, keepdims=True)
    vel_k = np.gradient(knob, 1 / fs, axis=0)
    vel_u = np.gradient(u, 1 / fs, axis=0)
    k_occ = int(np.searchsorted(t_img, t_c_true))
    occluded = set(range(k_occ - P.occlusion_frames, k_occ))
    for i, ti in enumerate(t_img):
        pts = []
        if i not in occluded:
            for mid, s in cfg.bat.marker_s_m.items():
                pos = knob_c[i] + s * u_c[i]
                speed = np.linalg.norm(np.interp(ti, t, np.linalg.norm(vel_k + s * vel_u, axis=1)))
                if rng.random() < P.cam_dropout:
                    continue
                if speed > P.blur_speed_mps and rng.random() < 0.5:
                    continue
                z = pos[2]
                sc = (z / 2.5) ** 2
                pos = pos + rng.normal(scale=[P.cam_noise_xy_m * sc, P.cam_noise_xy_m * sc,
                                              P.cam_noise_z_m * sc])
                pts.append({"id": mid, "pos": {"x": pos[0], "y": pos[1], "z": pos[2]}})
        ts = int(round(ti * 1e9))
        payloads.append({"schema_version": "1.1", "coord_sys": "zedx_left_rect_opencv_meters_v1",
                         "camera_id": "sim", "frame_id": i, "ts_ns_image": ts, "ts_ns_mono": ts,
                         "ts_ns_wall": ts + 3_000_000, "depth_source": "sim", "points": pts})

    # ---------------------------------------------------------------- truth
    truth_swings = detect_swings(t_dev, gyro, hg if P.contact else np.zeros_like(hg), cfg.swing)
    sw = truth_swings[-1]
    sw.t_contact = t_c_true if P.contact else None
    traj = Trajectory(t=t, knob=knob, knob_vel=vel_k, axis=u, axis_rate=vel_u)
    up = -g_cam / G
    truth_metrics = compute_metrics(traj, w_world_true, sw, cfg.bat, up,
                                    t_contact=t_c_true if P.contact else None)
    truth = {"params": asdict(P), "metrics": truth_metrics.to_dict(),
             "gyro_bias": bias.tolist(), "t_contact": t_c_true,
             "q_grip": q_grip.tolist()}

    return {"payloads": payloads, "imu_rows": rows, "imu_host": host, "truth": truth,
            "truth_traj": traj, "g_cam": g_cam}


def write_session(sim: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cam_path = os.path.join(out_dir, "camera.jsonl")
    with open(cam_path, "w") as f:
        for p in sim["payloads"]:
            f.write(json.dumps(p) + "\n")
    with open(cam_path + ".meta.json", "w") as f:
        json.dump({"camera": {"model": "simulated"},
                   "gravity": {"accel_mean_mps2": (-sim["g_cam"]).tolist(),
                               "camera_imu_transform": np.eye(4).tolist()}}, f, indent=2)
    imu_path = os.path.join(out_dir, "imu.csv")
    rows, host = sim["imu_rows"], sim["imu_host"]
    with open(imu_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        tx0 = rows[0].tx_us
        wraps, last = 0, rows[0].tx_us
        for k, s in enumerate(rows):
            if s.tx_us < last:
                wraps += 1
            last = s.tx_us
            tx = s.tx_us + wraps * 2 ** 32
            w.writerow([s.tx_us, (tx - tx0) / 1e6, host[k] - host[0], s.status,
                        *np.round(s.accel, 6), *np.round(s.gyro, 6), *np.round(s.accel_hg, 4),
                        *np.round(s.hf, 4), int(round(host[k] * 1e9))])
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump(sim["truth"], f, indent=2)
    return {"camera": cam_path, "imu": imu_path}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a synthetic swing session with ground truth")
    ap.add_argument("--out", default="sessions/sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-contact", action="store_true", help="dry swing (no ball impact)")
    a = ap.parse_args(argv)
    sim = simulate(SimParams(seed=a.seed, contact=not a.no_contact))
    paths = write_session(sim, a.out)
    print(json.dumps({**paths, "truth": sim["truth"]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
