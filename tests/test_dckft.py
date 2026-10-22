"""Tests for the dckft fusion package. No hardware needed: the simulator supplies ground truth."""
import json
import os

import numpy as np
import pytest

from dckft import quat as Q
from dckft.config import FusionConfig, ImuConfig, load_config, save_config
from dckft.data import (CameraData, bat_line_from_markers, load_camera_jsonl, load_imu_csv,
                        map_device_clock, unwrap_tx_us)
from dckft.imu.frames import FrameParser, TxClock, decode_frame, encode_frame
from dckft.pipeline import NoSwingError, process_swing
from dckft.simulate import SimParams, simulate, write_session
from dckft.sync import estimate_offset

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ quaternions
def test_quat_roundtrips():
    rv = RNG.normal(size=(50, 3))
    q = Q.from_rotvec(rv)
    assert np.allclose(np.linalg.norm(q, axis=1), 1)
    rv2 = Q.to_rotvec(q)
    q2 = Q.from_rotvec(rv2)
    assert np.allclose(Q.angle_between(q, q2), 0, atol=1e-7)
    for k in range(5):
        R = Q.to_matrix(q[k])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert Q.angle_between(Q.from_matrix(R), q[k]) < 1e-7


def test_quat_multiply_matches_matrices():
    a, b = Q.from_rotvec(RNG.normal(size=3)), Q.from_rotvec(RNG.normal(size=3))
    assert np.allclose(Q.to_matrix(Q.multiply(a, b)), Q.to_matrix(a) @ Q.to_matrix(b))


def test_integrate_gyro_constant_rate():
    w = np.array([0.0, 0.0, 2.0])
    t = np.linspace(0, 1, 401)
    q = Q.integrate_gyro(np.array([1.0, 0, 0, 0]), np.tile(w, (len(t), 1)), np.diff(t, prepend=0))
    assert np.allclose(Q.to_rotvec(q[-1]), [0, 0, 2.0], atol=1e-6)


def test_horn_recovers_pose():
    B = RNG.normal(size=(4, 3))
    q = Q.from_rotvec([0.3, -1.1, 0.4]); tvec = np.array([0.1, 0.2, 2.5])
    W = Q.rotate(q, B) + tvec
    q2, t2 = Q.horn(B, W)
    assert Q.angle_between(q, q2) < 1e-9 and np.allclose(t2, tvec)


# ------------------------------------------------------------------ IMU frames
def test_frame_encode_decode():
    cfg = ImuConfig()
    fr = encode_frame(123456, [1.0, -2.0, 9.8], [0.5, -0.25, 3.0], [50.0, -20.0, 10.0], cfg=cfg)
    assert len(fr) == 30
    s = decode_frame(fr, cfg)
    assert s.tx_us == 123456
    assert np.allclose(s.accel, [1.0, -2.0, 9.8], atol=cfg.accel_lsb)
    assert np.allclose(s.gyro, [0.5, -0.25, 3.0], atol=cfg.gyro_lsb)
    assert np.allclose(s.accel_hg, [50.0, -20.0, 10.0], atol=cfg.hg_lsb)


def test_parser_resyncs_after_garbage():
    frames = [encode_frame(1000 * k, [0, 0, 9.8], [0.1 * k, 0, 0]) for k in range(20)]
    stream = b"".join(frames[:5]) + b"\x01\x02\x03" + b"".join(frames[5:12]) + frames[12][:7] + \
        b"".join(frames[13:])
    p = FrameParser()
    out = []
    for i in range(0, len(stream), 13):          # arbitrary chunking like a serial port
        out += p.feed(stream[i:i + 13])
    txs = [s.tx_us for s in out]
    assert txs[:5] == [0, 1000, 2000, 3000, 4000]
    assert 11000 in txs and 19000 in txs          # recovered after both corruptions
    assert 12000 not in txs                       # the truncated frame is dropped
    assert p.resyncs >= 1


def test_tx_unwrap():
    tx = np.array([2 ** 32 - 2000, 2 ** 32 - 1000, 500, 1500])
    assert np.all(np.diff(unwrap_tx_us(tx)) > 0)
    c = TxClock()
    assert [c(v) for v in tx][-1] == 1500 + 2 ** 32


def test_device_clock_mapping_removes_batch_jitter():
    t_true = 1.79e9 + np.arange(2000) / 400.0
    tx = ((t_true - t_true[0]) * 1e6).astype(np.int64) + 5
    batch_end = np.minimum((np.arange(2000) // 4 + 1) * 4 - 1, 1999)
    host = (t_true[batch_end] + 0.004) * 1e9
    t = map_device_clock(tx, host)
    err = t - (t_true + 0.004)
    assert np.abs(err).max() < 5e-4


# ------------------------------------------------------------------ geometry / sync
def test_bat_line_from_collinear_markers():
    cfg = FusionConfig()
    u = np.array([0.6, -0.48, 0.64]); u /= np.linalg.norm(u)
    k = np.array([0.1, 0.2, 2.5])
    pts = {m: np.array([k + s * u]) for m, s in cfg.bat.marker_s_m.items()}
    cam = CameraData(np.array([0.0]), pts)
    knob, axis, n = bat_line_from_markers(cam, cfg.bat)
    assert n[0] == 3 and np.allclose(knob[0], k) and np.allclose(axis[0], u)
    pts["magenta"][:] = np.nan          # two markers still enough
    knob, axis, n = bat_line_from_markers(CameraData(np.array([0.0]), pts), cfg.bat)
    assert n[0] == 2 and np.allclose(axis[0], u)


def test_xcorr_offset():
    t = np.arange(0, 3, 1 / 400)
    sig = np.exp(-((t - 1.5) / 0.05) ** 2) * 30
    tc = np.arange(0, 3, 1 / 120)
    cam = np.interp(tc, t, sig)
    off, corr = estimate_offset(tc, cam, t + 0.012, sig, max_offset_s=0.1)
    assert abs(off + 0.012) < 1e-3 and corr > 0.9


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def sim_session(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    sim = simulate(SimParams(seed=7))
    paths = write_session(sim, str(d))
    return sim, paths


def test_pipeline_accuracy_on_simulated_swing(sim_session):
    sim, paths = sim_session
    cam, imu = load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"])
    assert imu.has_host_clock and cam.gravity_cam is not None
    res = process_swing(cam, imu)
    m, tr = res.metrics, sim["truth"]["metrics"]
    assert abs(m.bat_speed_mph - tr["bat_speed_mph"]) / tr["bat_speed_mph"] < 0.05
    assert abs(m.peak_hand_speed_mph - tr["peak_hand_speed_mph"]) / tr["peak_hand_speed_mph"] < 0.10
    assert abs(m.attack_angle_deg - tr["attack_angle_deg"]) < 4
    assert abs(m.vertical_bat_angle_deg - tr["vertical_bat_angle_deg"]) < 3
    assert m.contact_detected and abs(m.t_contact - sim["truth"]["t_contact"]) < 0.006
    assert abs(res.time_offset_s + sim["truth"]["params"]["imu_latency_s"]) < 0.004
    assert res.timings_ms["total"] < 5000
    assert res.notes


def test_pipeline_saves_outputs(sim_session, tmp_path):
    _, paths = sim_session
    res = process_swing(load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"]))
    cfg = FusionConfig()
    summary = res.save(str(tmp_path), cfg.bat)
    assert os.path.exists(tmp_path / "fused.csv")
    with open(tmp_path / "swing.json") as f:
        assert json.load(f)["metrics"]["bat_speed_mph"] == summary["metrics"]["bat_speed_mph"]


def test_render_smoke(sim_session, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from dckft.render import render_swing
    _, paths = sim_session
    res = process_swing(load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"]))
    cfg = FusionConfig(); cfg.render.show = False
    render_swing(res, cfg.bat, cfg.render, str(tmp_path))
    assert os.path.getsize(tmp_path / "swing.png") > 10_000


def test_dry_swing_uses_peak_speed(tmp_path):
    sim = simulate(SimParams(seed=3, contact=False))
    p = write_session(sim, str(tmp_path))
    res = process_swing(load_camera_jsonl(p["camera"]), load_imu_csv(p["imu"]))
    assert not res.metrics.contact_detected
    assert res.metrics.bat_speed_mph == pytest.approx(res.metrics.peak_bat_speed_mph, rel=0.02)


def test_legacy_csv_without_host_clock(sim_session, tmp_path):
    import pandas as pd
    sim, paths = sim_session
    df = pd.read_csv(paths["imu"]).drop(columns=["ts_ns_host"])
    legacy = tmp_path / "legacy.csv"
    df.to_csv(legacy, index=False)
    imu = load_imu_csv(str(legacy))
    assert not imu.has_host_clock
    res = process_swing(load_camera_jsonl(paths["camera"]), imu)
    tr = sim["truth"]["metrics"]
    assert abs(res.metrics.bat_speed_mph - tr["bat_speed_mph"]) / tr["bat_speed_mph"] < 0.08


def test_no_swing_raises(sim_session, tmp_path):
    _, paths = sim_session
    imu = load_imu_csv(paths["imu"])
    imu.gyro[:] *= 0.05
    with pytest.raises(NoSwingError):
        process_swing(load_camera_jsonl(paths["camera"]), imu)


def test_config_roundtrip(tmp_path):
    cfg = FusionConfig()
    cfg.bat.marker_s_m["neon_pink"] = 0.123
    save_config(cfg, str(tmp_path / "c.json"))
    cfg2 = load_config(str(tmp_path / "c.json"))
    assert cfg2.bat.marker_s_m["neon_pink"] == 0.123
    with open(tmp_path / "bad.json", "w") as f:
        json.dump({"bat": {"nope": 1}}, f)
    with pytest.raises(KeyError):
        load_config(str(tmp_path / "bad.json"))


def test_legacy_sample_capture_loads():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "IMU", "reports", "ICM45686_and_ADXL375_test_data_2026-09-30_16-10-23.csv")
    imu = load_imu_csv(path)
    assert len(imu) == 2400 and not imu.has_host_clock
    # first ~1.5 s the sensor is still: |accel| ~ g, tiny rates (then it gets moved by hand)
    still = imu.t < 1.5
    assert abs(np.median(np.linalg.norm(imu.accel[still], axis=1)) - 9.81) < 0.5
    assert np.median(np.linalg.norm(imu.gyro[still], axis=1)) < 0.1
    assert abs(np.median(np.diff(imu.t)) - 0.0025) < 1e-4      # 400 Hz


# ------------------------------------------------------------------ smoother / markers / capture
def test_smoother_helps_inside_camera_gaps(tmp_path):
    """Long occlusion before contact: the RTS pass should beat forward-only coasting there."""
    errs = {False: [], True: []}
    for seed in (20, 21, 22):
        sim = simulate(SimParams(seed=seed, flex_deg=0.0, occlusion_frames=8, gyro_bias_std=0.08))
        p = write_session(sim, str(tmp_path / f"s{seed}"))
        cam, imu = load_camera_jsonl(p["camera"]), load_imu_csv(p["imu"])
        tr, tc = sim["truth_traj"], sim["truth"]["t_contact"]
        for smooth in (False, True):
            cfg = FusionConfig(); cfg.ekf.smooth = smooth; cfg.flex_lam = None
            res = process_swing(cam, imu, cfg)
            assert res.ekf.smoothed == smooth
            t = res.ekf.t
            m = (t >= tc - 8 / 120) & (t <= tc)
            ut = np.column_stack([np.interp(t[m], tr.t, tr.axis[:, j]) for j in range(3)])
            errs[smooth].append(np.degrees(np.arccos(np.clip(np.sum(res.ekf.axis[m] * ut, 1), -1, 1))).mean())
    assert np.mean(errs[True]) < np.mean(errs[False])


def test_marker_gap_filling(sim_session, tmp_path):
    sim, paths = sim_session
    cfg = FusionConfig()
    res = process_swing(load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"]), cfg)
    mf = res.markers
    assert set(mf.tracks) == set(cfg.bat.marker_s_m)
    tr = sim["truth_traj"]
    win = (mf.t_cam >= res.swing.t_onset) & (mf.t_cam <= res.swing.t_end)
    n_gap = 0
    for mid, track in mf.tracks.items():
        true = tr.knob + cfg.bat.marker_s_m[mid] * tr.axis
        tt = np.column_stack([np.interp(mf.t_cam, tr.t, true[:, j]) for j in range(3)])
        gap = win & ~track.cam_seen
        n_gap += int(gap.sum())
        if gap.any():   # IMU-filled positions: a few cm while the bat moves ~25 cm per frame
            e = np.linalg.norm(track.cam_filled[gap] - tt[gap], axis=1)
            assert np.median(e) < 0.03 and e.max() < 0.05
        assert np.isfinite(track.filled).all()
    assert n_gap > 0       # the simulator always occludes frames before contact
    summ = res.diagnostics()["markers"]
    assert all(s["frames_filled"] >= 0 and s["residual_rms_mm"] < 30 for s in summ.values())
    res.save(str(tmp_path), cfg.bat)
    lines = (tmp_path / "camera_filled.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert len(first["points"]) == 3 and {p["source"] for p in first["points"]} <= {"camera", "imu_fill"}
    assert sum('"imu_fill"' in l for l in lines) > 0
    assert (tmp_path / "markers.csv").exists() and (tmp_path / "markers_camera.csv").exists()


def test_postprocess_session_folder(sim_session, tmp_path):
    """postprocess on a session folder that already has camera.jsonl (SVO step skipped)."""
    import shutil
    from dckft.run import main
    _, paths = sim_session
    sess = tmp_path / "capture_test"
    sess.mkdir()
    shutil.copy(paths["camera"], sess / "camera.jsonl")
    shutil.copy(paths["camera"] + ".meta.json", sess / "camera.jsonl.meta.json")
    shutil.copy(paths["imu"], sess / "imu.csv")
    assert main(["postprocess", "--session", str(sess), "--no-render"]) == 0
    assert (sess / "out" / "swing_00" / "swing.json").exists() or (sess / "out" / "swing.json").exists()
