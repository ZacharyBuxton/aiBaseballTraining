"""Record-first web acquisition and camera-led display regressions (no ZED hardware)."""
import csv
import json
from pathlib import Path
import signal
import subprocess

import numpy as np
import pytest

from dckft.data import CameraData
from dckft.quality import clean_camera
from dckft.sync import axis_rate, estimate_offset


def test_irregular_camera_rate_uses_actual_timestamps():
    t = np.r_[0., np.cumsum(np.tile([1, 1, 4], 100)/120)]
    axes = np.column_stack([np.cos(3*t), np.sin(3*t), t*0])
    rate = axis_rate(t, axes)
    assert np.median(rate[10:-10]) == pytest.approx(3., rel=.02)


def test_axis_rate_does_not_bridge_a_long_occlusion():
    t = np.r_[np.arange(0, .3, .01), np.arange(.6, 1., .01)]
    phase = 2*t + (t >= .6)*1.5
    u = np.column_stack([np.cos(phase), np.sin(phase), t*0])
    assert np.nanmax(axis_rate(t, u)) < 2.1


def test_sync_preserves_real_delay_with_gaps_and_irregular_samples():
    t = np.arange(0, 2., .0025)
    signal = lambda x: np.exp(-((x-.8)/.08)**2)+.6*np.exp(-((x-1.2)/.12)**2)
    tc = np.arange(0, 2., 1/120)
    tc = tc[~((tc > .9) & (tc < 1.06))]
    tc = tc[np.arange(len(tc)) % 4 != 0]
    off, corr = estimate_offset(tc, signal(tc), t+.012, signal(t), .1)
    assert off == pytest.approx(-.012, abs=.003)
    assert corr > .98


def test_good_marker_pair_survives_one_outlier():
    t = np.arange(5)/120
    s = {'a': .3, 'b': .45, 'c': .635}
    pts = {m: np.tile([v, 0, 2.], (5, 1)) for m, v in s.items()}
    pts['c'][2] += [1, 1, 1]
    cleaned, rejected = clean_camera(CameraData(t, pts), s)
    assert rejected == 1
    assert np.isfinite(cleaned.points['a'][2]).all()
    assert np.isfinite(cleaned.points['b'][2]).all()
    assert np.isnan(cleaned.points['c'][2]).all()
    assert np.isfinite(pts['c'][2]).all()  # preserve raw measurements


@pytest.fixture
def backend(tmp_path, monkeypatch):
    import app.backend as be
    monkeypatch.setattr(be, 'DATA', str(tmp_path))
    monkeypatch.setattr(be, 'SWINGS', str(tmp_path/'swings'))
    return be.Backend(imu_port='/dev/fake')


def test_capture_commands_preserve_marker_settings_and_record_only(backend, tmp_path):
    source = tmp_path/'camera.json'
    source.write_text(json.dumps({'markers': {'s_min': 123, 'hue_tol': 9}}))
    backend.cam_config = str(source)
    backend.capture_mode = '1080p60'
    imu, cam = backend._capture_commands(str(tmp_path))
    settings = json.loads((tmp_path/'camera_config.json').read_text())
    assert settings['camera']['resolution'] == 'HD1080'
    assert settings['camera']['fps'] == 60
    assert settings['markers']['s_min'] == 123
    assert '--record-only' in cam and '--preview' not in cam
    assert '--no-zmq' in cam and '--no-zmq' in imu
    assert cam[cam.index('--max-frames')+1] == '300'


def test_camera_writers_close_before_detection_and_fusion(backend, monkeypatch):
    import app.backend as be
    events = []
    class Proc:
        returncode = 0
        def poll(self):
            return 0
        def wait(self, **kwargs):
            events.append('closed')
            return 0
    def child(name, cmd, session):
        p = Proc()
        log = open(Path(session)/(name+'.log'), 'w')
        if name == 'camera':
            (Path(session)/'camera.svo2').write_bytes(b'fake SVO')
        backend.procs.append((name, p, log))
        return p
    monkeypatch.setattr(backend, '_start_child', child)
    monkeypatch.setattr(backend, '_detect_recording', lambda session: events.append('detect'))
    monkeypatch.setattr(backend, '_save_recorded_session', lambda session: events.append('fuse'))
    backend.mode = 'hardware'
    backend._hardware_loop()
    assert events == ['closed', 'closed', 'detect', 'fuse']
    assert backend.mode == 'idle' and not backend.busy


def test_child_that_ignores_interrupt_is_terminated_and_waited(backend, tmp_path):
    events = []
    class Proc:
        def poll(self):
            return None
        def send_signal(self, sig):
            assert sig == signal.SIGINT
        def wait(self, timeout=None):
            events.append(timeout)
            if timeout == 10:
                raise subprocess.TimeoutExpired('child', timeout)
            return 0
        def terminate(self):
            events.append('terminate')
    log = open(tmp_path/'log', 'w')
    backend.procs = [('camera', Proc(), log)]
    backend._stop_children()
    assert events == [10, 'terminate', 3]
    assert log.closed and backend.procs == []


def test_empty_imu_does_not_invent_a_fused_swing(backend, tmp_path):
    from dckft.simulate import SimParams, simulate
    sim = simulate(SimParams(seed=7))
    (tmp_path/'camera_session.jsonl').write_text('\n'.join(json.dumps(p) for p in sim['payloads']))
    from dckft.imu.frames import CSV_COLUMNS
    (tmp_path/'imu_session.csv').write_text(','.join(CSV_COLUMNS)+'\n')
    with pytest.raises(ValueError, match='empty|usable|polyfit|non-empty'):
        backend._save_recorded_session(str(tmp_path))
    assert backend.list_swings() == []


def test_browser_replay_is_in_page_and_reads_the_exported_pose_stream():
    from app.server import swing_page
    page = swing_page({'id': 'example', 'files': ['swing.png', 'replay.json']})
    assert 'canvas' in page and '/replay.js' in page and 'replay.json' in page
    assert 'IMU/model estimate' in page


def test_recorded_pair_uses_full_csv_clock_mapping_and_exports_browser_replay(backend, tmp_path):
    from dckft.simulate import SimParams, simulate, write_session
    import shutil
    sim = simulate(SimParams(seed=7))
    paths = write_session(sim, str(tmp_path/'raw'))
    session = tmp_path/'session'
    session.mkdir()
    shutil.copy2(paths['camera'], session/'camera_session.jsonl')
    shutil.copy2(paths['camera']+'.meta.json', session/'camera_session.jsonl.meta.json')
    shutil.copy2(paths['imu'], session/'imu_session.csv')
    with (session/'camera_frames.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_id', 'ts_ns_image', 'accepted_3d'])
        for p in sim['payloads']:
            w.writerow([p['frame_id'], p['ts_ns_image'], len(p['points'])])
    sid = backend._save_recorded_session(str(session))
    target = Path(backend.session or tmp_path)/'swings'/sid
    assert (target/'replay.json').is_file()
    meta = json.loads((target/'meta.json').read_text())
    assert meta['source'] == 'hardware (record first)'
    assert meta['camera_quality']['saved_frames'] == len(sim['payloads'])
    saved = json.loads((target/'swing.json').read_text())
    assert saved['diagnostics']['replay']['active_swing_estimated_fraction'] is not None


def test_filled_poses_visible_without_forcing_a_speed(tmp_path):
    from dckft.simulate import SimParams, simulate, write_session
    from dckft.data import load_camera_jsonl, load_imu_csv
    from dckft.pipeline import process_swing
    from dckft.config import FusionConfig
    from dckft.render import render_swing
    paths = write_session(simulate(SimParams(seed=7, occlusion_frames=20)), str(tmp_path/'in'))
    r = process_swing(load_camera_jsonl(paths['camera']), load_imu_csv(paths['imu']))
    cfg = FusionConfig()
    k = np.flatnonzero(r.replay.estimated & (r.traj.t >= r.swing.t_onset)
                       & (r.traj.t <= r.swing.t_end))[0]
    fig = render_swing(r, cfg.bat, cfg.render)
    fig._dckft_refs[0].set_val(float(r.traj.t[k]-r.swing.t_onset))
    assert any(line.get_color() == 'darkorange' and len(line.get_data_3d()[0]) == 2
               for line in fig.axes[0].lines)
    assert not r.metrics.speed_reliable
    r.save(str(tmp_path/'out'), cfg.bat)
    replay = json.loads((tmp_path/'out/replay.json').read_text())
    estimated = [f for f in replay['frames'] if f['source'] == 'imu_estimate']
    assert estimated and all(f['speed'] is None for f in estimated)
    import matplotlib.pyplot as plt
    plt.close(fig)
