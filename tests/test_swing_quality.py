"""Regression tests for reset motion, long occlusions, and false speed peaks."""
import json

import numpy as np
import pytest

from dckft.config import BatConfig, FusionConfig, SwingConfig
from dckft.data import CameraData, load_camera_jsonl, load_imu_csv
from dckft.metrics import compute_metrics, MPS_TO_MPH
from dckft.pipeline import process_swing
from dckft.quality import clean_camera, optical_support
from dckft.simulate import SimParams, simulate, write_session
from dckft.swing import Swing, detect_swings
from dckft.trajectory import Trajectory, build_trajectory


def test_reset_movement_does_not_extend_finished_swing():
    t = np.arange(0, 4, .0025)
    w = np.zeros((len(t),3))
    w[(t>=1)&(t<=1.4),0] = 20
    w[(t>=1.8)&(t<=2.1),0] = 40  # faster return to stance
    swings = detect_swings(t,w,np.zeros_like(w),SwingConfig())
    assert len(swings) == 1
    assert swings[0].t_end < 1.45
    assert swings[0].peak_rads < 25


def test_short_dip_does_not_cut_swing():
    t = np.arange(0,3,.0025)
    w = np.zeros((len(t),3)); w[(t>=1)&(t<=1.5),0] = 20
    w[(t>=1.2)&(t<=1.23),0] = 0
    swings = detect_swings(t,w,np.zeros_like(w),SwingConfig())
    assert len(swings)==1 and 1.49 < swings[0].t_end < 1.55


def test_continuous_motion_has_finite_window():
    t = np.arange(0,5,.0025)
    w = np.zeros((len(t),3)); w[t>=1,0]=20
    swings=detect_swings(t,w,np.zeros_like(w),SwingConfig())
    assert swings[0].t_end-swings[0].t_onset < 1.25


def test_long_gap_and_reacquisition_are_not_supported():
    t=np.arange(0,1,.0025)
    tc=np.r_[np.arange(0,.3,.01),np.arange(.65,1,.01)]
    support=optical_support(t,tc,.06,.04)
    assert not support[(t>=.26)&(t<=.68)].any()
    assert support[(t>=.05)&(t<=.2)].all()
    assert support[(t>=.72)&(t<=.9)].all()


def make_traj():
    t=np.arange(0,1,.0025)
    velocity=np.zeros((len(t),3)); velocity[:,0]=20
    axis=np.tile([1.,0.,0.],(len(t),1))
    tr=Trajectory(t,np.column_stack([20*t,t*0,t*0]),velocity,axis,np.zeros_like(axis))
    return tr,Swing(.1,.4,.8,None,20)


def test_occlusion_spike_cannot_become_reported_peak():
    tr,sw=make_traj(); tr.knob_vel[(tr.t>.55)&(tr.t<.65),0]=200
    tr.valid=~((tr.t>=.5)&(tr.t<=.7))
    m=compute_metrics(tr,np.zeros_like(tr.axis),sw,BatConfig(),[0,0,1])
    assert m.speed_reliable
    assert m.bat_speed_mph == pytest.approx(20*MPS_TO_MPH)
    assert m.peak_bat_speed_mph == pytest.approx(20*MPS_TO_MPH)


def test_held_bat_tail_cannot_change_dry_swing_peak():
    tr,sw=make_traj(); tr.knob_vel[tr.t>.85,0]=200
    m=compute_metrics(tr,np.zeros_like(tr.axis),sw,BatConfig(),[0,0,1])
    assert m.bat_speed_mph == pytest.approx(20*MPS_TO_MPH)
    assert m.t_contact <= sw.t_end


def test_missing_peak_is_unavailable_in_json():
    tr,sw=make_traj(); tr.valid=abs(tr.t-sw.t_peak)>.15
    m=compute_metrics(tr,np.zeros_like(tr.axis),sw,BatConfig(),[0,0,1])
    assert not m.speed_reliable and np.isnan(m.bat_speed_mph)
    data=m.to_dict()
    assert data['bat_speed_mph'] is None
    json.dumps(data,allow_nan=False)


def test_geometry_and_spike_rejection_preserves_raw_data():
    t=np.arange(20)/120
    a=np.column_stack([t,t*0,t*0]); b=a+[.2,0,0]
    a[10]+=1; b[10]+=1
    cam=CameraData(t,{'a':a,'b':b})
    clean,rejected=clean_camera(cam,{'a':0,'b':.2})
    assert rejected>=1 and np.isnan(clean.points['a'][10]).all()
    assert np.isfinite(cam.points['a']).all()


def test_reacquisition_does_not_pull_earlier_translation():
    ti=np.arange(0,1.2,.0025); tc=np.r_[np.arange(0,.4,.01),np.arange(.8,1.2,.01)]
    p=np.column_stack([tc,tc*0,tc*0]); axis=np.tile([1.,0.,0.],(len(ti),1))
    kwargs=dict(acc_world=np.zeros_like(axis),flex_lam=None)
    before=build_trajectory(ti,axis,np.zeros_like(axis),tc,p,**kwargs)
    p[tc>=.8]+=5
    after=build_trajectory(ti,axis,np.zeros_like(axis),tc,p,**kwargs)
    assert np.allclose(before.knob_vel[ti<.35],after.knob_vel[ti<.35])


def test_pipeline_and_render_stop_at_swing_end(tmp_path):
    sim=simulate(SimParams(seed=7))
    paths=write_session(sim,str(tmp_path/'input'))
    cam,imu=load_camera_jsonl(paths['camera']),load_imu_csv(paths['imu'])
    result=process_swing(cam,imu)
    assert result.traj.t[-1] <= result.swing.t_end+1e-6
    assert result.cam_t[-1] <= result.swing.t_end+1e-6
    original=result.metrics.bat_speed_mph
    # Whole bat teleports AFTER completion; must not change the completed swing.
    tail=cam.t>result.swing.t_end+.08
    for p in cam.points.values():
        p[tail]+=20
    replay=process_swing(cam,imu)
    assert replay.metrics.bat_speed_mph == pytest.approx(original,rel=.001)
    summary=replay.save(str(tmp_path/'out'),FusionConfig().bat)
    assert 'speed_reliable' in summary['diagnostics']
    assert 'speed_valid' in (tmp_path/'out/fused.csv').read_text().splitlines()[0]
    from dckft.render import render_swing
    cfg=FusionConfig(); cfg.render.show=False
    render_swing(replay,cfg.bat,cfg.render,str(tmp_path/'out'))
    assert (tmp_path/'out/swing.png').stat().st_size>10000


def test_pipeline_hides_speed_when_peak_is_occluded(tmp_path):
    sim=simulate(SimParams(seed=7,occlusion_frames=20))
    paths=write_session(sim,str(tmp_path/'input'))
    result=process_swing(load_camera_jsonl(paths['camera']),load_imu_csv(paths['imu']))
    assert not result.metrics.speed_reliable
    assert result.metrics.to_dict()['bat_speed_mph'] is None
    assert 'insufficient optical support' in result.notes[0]
    from dckft.render import render_swing
    cfg=FusionConfig(); cfg.render.show=False
    render_swing(result,cfg.bat,cfg.render,str(tmp_path/'out'))
    result.save(str(tmp_path/'out'),cfg.bat)
    assert 'unsupported' in (tmp_path/'out/camera_filled.jsonl').read_text()
