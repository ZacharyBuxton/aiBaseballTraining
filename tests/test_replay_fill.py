"""Continuous replay must fill display gaps without inventing metric validity."""
import csv

import numpy as np
import pytest

from dckft.config import FusionConfig
from dckft.data import load_camera_jsonl, load_imu_csv
from dckft.pipeline import process_swing
from dckft.replay_fill import build_replay, camera_guided_replay
from dckft.simulate import SimParams, simulate, write_session
from dckft.trajectory import Trajectory


def track():
    t = np.arange(0, 1.001, .005)
    p = np.column_stack([t*t, t*0, t*0])
    v = np.column_stack([2*t, t*0, t*0])
    axis = np.tile([0., 1., 0.], (len(t), 1))
    good = ~((t>.3) & (t<.7))
    return Trajectory(t, p, v, axis, np.zeros_like(axis), good)


def test_acceleration_bridges_gap_without_teleport_or_mutation():
    tr = track()
    original_p, original_v = tr.knob.copy(), tr.knob_vel.copy()
    # Simulate the metric track's reset to the reacquired segment in the gap.
    tr.knob[~tr.valid] = 123
    tr.knob_vel[~tr.valid] = 900
    raw = tr.knob.copy()
    a = np.zeros_like(tr.knob); a[:, 0] = 2
    rp = build_replay(tr, a)
    assert np.allclose(rp.traj.knob, original_p, atol=1e-10)
    assert np.allclose(rp.traj.knob_vel, original_v, atol=1e-10)
    assert np.array_equal(tr.knob, raw)
    assert np.array_equal(rp.traj.valid, tr.valid)
    assert rp.estimated.sum() == (~tr.valid).sum()
    assert np.max(np.linalg.norm(np.diff(rp.traj.knob, axis=0), axis=1)) < .011
    assert rp.nearest_support_s.max() == pytest.approx(.2)


def test_reacquisition_endpoint_constraints_and_invalid_accel():
    tr = track()
    tr.knob[tr.t>=.7, 0] += .2
    rp = build_replay(tr, np.full_like(tr.knob, np.nan))
    assert np.isfinite(rp.traj.knob).all()
    assert np.isfinite(rp.traj.knob_vel).all()
    assert np.allclose(rp.traj.knob[tr.valid], tr.knob[tr.valid])
    assert not rp.accel_used.any()
    # Endpoint correction distributes the displacement instead of a midpoint jump.
    assert np.max(np.linalg.norm(np.diff(rp.traj.knob, axis=0), axis=1)) < .025


def test_edges_dead_reckon_and_unanchored_replay_is_rejected():
    tr = track()
    tr.valid[:] = (tr.t>=.2) & (tr.t<=.8)
    a = np.zeros_like(tr.knob); a[:, 0] = 2
    rp = build_replay(tr, a)
    assert np.allclose(rp.traj.knob[:, 0], tr.t**2)
    tr.valid[:] = False
    with pytest.raises(ValueError, match="anchor"):
        build_replay(tr, a)


def test_gap_replay_export_and_render_keep_speed_unavailable(tmp_path):
    paths = write_session(simulate(SimParams(seed=7, occlusion_frames=20)), str(tmp_path/'in'))
    r = process_swing(load_camera_jsonl(paths['camera']), load_imu_csv(paths['imu']))
    assert r.replay is not None and r.replay.estimated.any()
    assert np.isfinite(r.replay.traj.knob).all()
    assert not r.metrics.speed_reliable
    cfg = FusionConfig(); cfg.render.show = False
    summary = r.save(str(tmp_path/'out'), cfg.bat)
    with open(tmp_path/'out/replay.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(r.traj.t)
    estimated = [x for x in rows if x['pose_source']=='imu_estimate']
    assert estimated and all(x['speed_valid']=='0' for x in estimated)
    assert summary['metrics']['bat_speed_mph'] is None
    assert summary['diagnostics']['replay']['estimated_samples'] > 0
    from dckft.render import render_swing
    render_swing(r, cfg.bat, cfg.render, str(tmp_path/'out'))
    assert (tmp_path/'out/swing.png').stat().st_size > 10000


def test_camera_poses_override_biased_fusion_without_validating_speed():
    tr = track()
    tr.knob[:, 2] += .08  # mimic the observed 8 cm fusion/camera disagreement
    tr.valid[:] = False  # a conservative speed mask must not hide camera poses
    tc = np.arange(0, 1.001, .02)
    pc = np.column_stack([tc*tc, tc*0, tc*0])
    uc = np.tile([1., 0., 0.], (len(tc), 1))
    rp = camera_guided_replay(tr, tc, pc, uc)
    assert rp.pose_supported.all()
    assert not rp.estimated.any()
    assert not rp.traj.valid.any()
    assert np.max(abs(rp.traj.knob[:, 2])) < 1e-12
    assert np.allclose(rp.traj.axis, uc[0], atol=1e-10)
    for i, time in enumerate(tc):
        k = np.argmin(abs(tr.t-time))
        assert np.allclose(rp.traj.knob[k], pc[i], atol=1e-10)
    assert np.all(tr.knob[:, 2] == .08)


def test_camera_gap_has_pose_support_up_to_edges_but_keeps_speed_guard():
    tr = track()
    tc = np.r_[np.arange(0,.301,.02), np.arange(.7,1.001,.02)]
    pc = np.column_stack([tc*tc, tc*0, tc*0])
    uc = np.tile([0., 1., 0.], (len(tc), 1))
    tr.valid[(tr.t>.26) & (tr.t<.74)] = False
    a = np.zeros_like(tr.knob); a[:, 0] = 2
    rp = camera_guided_replay(tr, tc, pc, uc, a)
    assert rp.pose_supported[(tr.t>.26) & (tr.t<=.3)].all()
    assert rp.estimated[(tr.t>.31) & (tr.t<.69)].all()
    assert not rp.traj.valid[(tr.t>.26) & (tr.t<.74)].any()
    assert np.isfinite(rp.traj.knob).all()
    assert np.max(np.linalg.norm(np.diff(rp.traj.knob,axis=0),axis=1)) < .012


def test_exact_camera_times_and_isolated_anchor_are_used():
    tr = track()
    # Camera and IMU grids need not coincide. Exact camera times anchor the internal solve.
    tc = np.array([.103, .123, .143, .503, .803, .823, .843])
    pc = np.column_stack([tc, tc*0, tc*0])
    uc = np.tile([0., 1., 0.], (len(tc), 1))
    tr.knob_vel[:, 0] = 1
    rp = camera_guided_replay(tr, tc, pc, uc)
    k = np.argmin(abs(tr.t-.503))
    assert rp.pose_supported[k]
    assert rp.nearest_support_s[k] == pytest.approx(abs(tr.t[k]-.503))
    assert np.allclose(rp.traj.knob[k], [tr.t[k],0,0], atol=.001)
    assert np.array_equal(rp.traj.valid, tr.valid)
