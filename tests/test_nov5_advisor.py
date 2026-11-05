"""
Nov 5 advisor items: ICM45686-only IMU, pluggable frame format (new firmware / better
timestamps), IMU acceleration trigger with a held finish, camera quaternion from the pink
centroids (omega = 2 qdot q*), attack angle / vertical bat angle as primary outputs.
"""
import numpy as np
import pytest

from dckft import quat as Q
from dckft.cam_orientation import camera_angles, camera_orientation, quaternions_from_axes
from dckft.config import FusionConfig, ImuConfig, SwingConfig
from dckft.data import bat_line_from_markers, load_camera_jsonl, load_imu_csv, payloads_to_camera
from dckft.imu import frames as F
from dckft.metrics import BLAST_REFERENCE
from dckft.pipeline import process_swing
from dckft.simulate import SimParams, simulate, write_session
from dckft.swing import LiveSwingDetector, detect_swings, detect_swings_imu, trigger_signal
from dckft.trigger_tune import analyse


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    s = simulate(SimParams(seed=11))
    return s, write_session(s, str(tmp_path_factory.mktemp("nov5")))


# ---------------------------------------------------------------- IMU / firmware
def test_v1_format_roundtrip_through_registry():
    cfg = ImuConfig()
    assert F.frame_format(cfg).size == 30
    fr = F.encode_frame(4_000_000_000, [1, -2, 9.8], [0.5, -0.2, 3.0], [40, -10, 5], cfg=cfg)
    s = F.decode_frame(fr, cfg)
    assert s.tx_us == 4_000_000_000 and np.allclose(s.gyro, [0.5, -0.2, 3.0], atol=cfg.gyro_lsb)


def test_new_firmware_format_is_a_config_change():
    """A 64-bit-timestamp, ICM45686-only frame (what the new firmware may look like)."""
    F.FORMATS["test_v2"] = F.FrameFormat(name="test_v2", struct=">Q6hH", sync_index=7, ts_index=0,
                                         ts_bits=64, ts_unit_us=1.0, accel=((1, 1), (2, 1), (3, 1)),
                                         gyro=((4, 1), (5, 1), (6, 1)))
    try:
        cfg = ImuConfig(frame_format="test_v2")
        big = 2 ** 40                                         # far past the old 32-bit wrap
        frames = [F.encode_frame(big + 2500 * k, [0, 0, 9.8], [0.1 * k, 0, 0], cfg=cfg) for k in range(6)]
        p = F.FrameParser(cfg)
        out = p.feed(b"".join(frames[:3]) + b"\x00\x01" + b"".join(frames[3:]))
        assert [s.tx_us for s in out][:3] == [big, big + 2500, big + 5000]
        assert len(out) >= 5 and np.allclose(out[0].accel_hg, 0)      # no ADXL in this frame
        clock = F.TxClock(64)
        assert clock(big) == big and clock(5) == 5                     # 64-bit: never "wraps"
    finally:
        del F.FORMATS["test_v2"]


def test_icm_only_is_the_default_and_finds_impact(sim):
    s, paths = sim
    cfg = FusionConfig()
    assert cfg.imu.use_adxl375 is False
    imu = load_imu_csv(paths["imu"])
    imu.accel_hg[:] = 0.0                                     # board without the ADXL375
    sw = detect_swings_imu(imu, cfg)
    assert sw and sw[-1].t_contact is not None


# ---------------------------------------------------------------- trigger
def test_trigger_signal_modes():
    a = np.tile([0.0, 0.0, 9.80665], (50, 1))
    a[20:30] = [5.0, -40.0, 9.8]
    gyro = np.zeros((50, 3))
    for src, expect in (("accel_mag", 30.0), ("accel_-y", 30.0), ("accel_xy", 30.0)):
        sig, thr = trigger_signal(gyro, a, SwingConfig(trigger_source=src), 400.0)
        assert sig.max() > expect and thr == SwingConfig().trigger_accel_ms2
    sig, _ = trigger_signal(gyro, a, SwingConfig(trigger_source="accel_+y"), 400.0)
    assert sig.max() < 1.0                                    # wrong sign never fires
    with pytest.raises(ValueError):
        trigger_signal(gyro, a, SwingConfig(trigger_source="accel_w"), 400.0)


def test_accel_trigger_and_held_finish(sim):
    s, paths = sim
    cfg = FusionConfig()
    imu = load_imu_csv(paths["imu"])
    sw = detect_swings_imu(imu, cfg)
    assert len(sw) == 1
    tc = s["truth"]["t_contact"]
    assert sw[0].t_onset < tc < sw[0].t_end
    # the end is where the bat is held still: rate low and |a| ~ g right after it
    k = np.searchsorted(imu.t, sw[0].t_end)
    assert np.linalg.norm(imu.gyro[k:k + 20], axis=1).max() < 6.0


def test_live_detector_with_accel_trigger(sim):
    _, paths = sim
    cfg = FusionConfig()
    imu = load_imu_csv(paths["imu"])
    det = LiveSwingDetector(cfg.swing)
    found = None
    for k in range(400, len(imu.t), 20):                      # feed it as if live, 50 ms at a time
        r = det.update(imu.t[k - 1], imu.t[:k], imu.gyro[:k], imu.accel[:k], imu.accel[:k],
                       cfg.swing.contact_icm_ms2)
        found = found or r
    assert found is not None


def test_trigger_tune_recommends_something(sim):
    _, paths = sim
    imu = load_imu_csv(paths["imu"])
    rep = analyse(imu.t, imu.gyro, imu.accel)
    assert rep["swings_found"] == 1 and "recommended" in rep


# ---------------------------------------------------------------- camera quaternion
def test_quaternions_follow_the_axes_without_twist():
    th = np.linspace(0, 2.5, 50)
    axes = np.column_stack([np.cos(th), np.sin(th), 0.3 * np.ones_like(th)])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    q = quaternions_from_axes(axes)
    assert np.allclose(Q.rotate(q, np.tile([1.0, 0, 0], (50, 1))), axes, atol=1e-9)


def test_camera_quaternion_rate_matches_true_bat_rate(sim):
    s, paths = sim
    cfg = FusionConfig()
    cam = payloads_to_camera(s["payloads"])
    _, u, _ = bat_line_from_markers(cam, cfg.bat)
    co = camera_orientation(cam.t, u)
    tr = s["truth_traj"]
    w_true = np.cross(tr.axis, tr.axis_rate)                  # omega perpendicular to the bat
    wt = np.column_stack([np.interp(co.t, tr.t, w_true[:, j]) for j in range(3)])
    tc = s["truth"]["t_contact"]
    m = (co.t > tc - 0.15) & (co.t < tc - 0.03)
    err = np.linalg.norm(co.omega[m] - wt[m], axis=1) / np.maximum(np.linalg.norm(wt[m], axis=1), 1)
    assert np.median(err) < 0.10


# ---------------------------------------------------------------- angles
def test_attack_and_vertical_bat_angle_fused_and_camera_only(sim):
    s, paths = sim
    res = process_swing(load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"]))
    m, t = res.metrics, s["truth"]["metrics"]
    assert m.angles_reliable
    assert abs(m.attack_angle_deg - t["attack_angle_deg"]) < 4
    assert abs(m.vertical_bat_angle_deg - t["vertical_bat_angle_deg"]) < 3
    assert abs(m.attack_angle_cam_deg - t["attack_angle_deg"]) < 5
    assert abs(m.vertical_bat_angle_cam_deg - t["vertical_bat_angle_deg"]) < 4
    assert res.cam_vs_imu["magnitude_correlation"] > 0.9


def test_blast_reference_and_hand_point():
    assert BLAST_REFERENCE["attack_angle_deg"]["target"] == [6, 14]
    assert BLAST_REFERENCE["vertical_bat_angle_deg"]["by_level"]["College"] == [-33, -23]
    assert FusionConfig().bat.hands_s_m == pytest.approx(0.152)     # 6 in from the knob


def test_outputs_include_camera_orientation(sim, tmp_path):
    import json
    _, paths = sim
    res = process_swing(load_camera_jsonl(paths["camera"]), load_imu_csv(paths["imu"]))
    res.save(str(tmp_path), FusionConfig().bat)
    assert (tmp_path / "camera_orientation.csv").exists()
    j = json.loads((tmp_path / "swing.json").read_text())
    assert "blast_reference" in j and "attack_angle_cam_deg" in j["metrics"]
