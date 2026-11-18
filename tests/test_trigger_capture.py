"""Real elapsed-time windows, closed-file handoff, and camera continuity boundaries."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dckft.time_ring import TimeSwingRing
from app.trigger_capture import CsvTail, TriggerCapture, sample_from_row, write_json


def sample(t, above=False):
    return {'t_dev_s': float(t), 'gyro': [12. if above else 0., 0., 0.]}


@pytest.mark.parametrize('hz', [250, 380, 400])
def test_elapsed_window_is_six_seconds_at_actual_radio_rate(hz):
    ring = TimeSwingRing()
    done = []
    for t in np.arange(0, 8, 1/hz):
        w = ring.push(sample(t, 4 <= t < 4.3))
        if w:
            done.append(w)
    assert len(done) == 1
    w = done[0]
    assert w.t_first == pytest.approx(w.t_trigger-3)
    assert w.t_last == pytest.approx(w.t_trigger+3)
    assert w.samples[-1]['t_dev_s']-w.samples[0]['t_dev_s'] == pytest.approx(6, abs=2/hz)
    assert len(w.samples) == pytest.approx(6*hz, abs=2)


def test_packet_gaps_are_preserved_and_repeated_swings_rearm():
    ring = TimeSwingRing()
    done = []
    times = np.arange(0, 15, .0025)
    # Reproduce the 20--150 ms outages that defeated the old 1200-frame ring.
    times = times[~(((times > 2) & (times < 2.15)) | ((times > 5) & (times < 5.08)))]
    for t in times:
        w = ring.push(sample(t, (4 <= t < 4.2) or (10 <= t < 10.2)))
        if w:
            done.append(w)
    assert len(done) == 2 and ring.gaps == 2
    assert ring.restarts == 0
    assert max(np.diff([s['t_dev_s'] for s in done[0].samples])) > .07
    assert all(not 5.01 < s['t_dev_s'] < 5.07 for s in done[0].samples)


def test_sustained_above_threshold_does_not_retrigger_and_requires_warmup():
    ring = TimeSwingRing()
    windows = []
    for t in np.arange(0, 15, .01):
        w = ring.push(sample(t, t > 1))
        if w:
            windows.append(w)
    assert not windows  # rising edge occurred before the pre-roll was ready
    ring.push(sample(15, False))
    ring.push(sample(15.01, True))
    assert ring.pending is not None
    assert ring.finish_if_due(18.01) is not None
    for t in np.arange(18.02, 24, .01):
        assert ring.push(sample(t, True)) is None


def test_silent_radio_can_finish_without_inventing_samples():
    ring = TimeSwingRing()
    for t in np.arange(0, 4.3, .01):
        ring.push(sample(t, t >= 4))
    count = len(ring.window)
    w = ring.finish_if_due(7.01)
    assert len(w.samples) == count
    assert w.samples[-1]['t_dev_s'] < 4.31
    assert w.t_last == pytest.approx(7.)


def test_device_reboot_requires_new_preroll_and_duplicate_is_ignored():
    ring = TimeSwingRing()
    for t in np.arange(0, 4.2, .01):
        ring.push(sample(t, t >= 4))
    assert ring.pending is not None
    ring.push(sample(0))
    assert ring.pending is None and ring.restarts == 1
    n = len(ring.pre)
    ring.push(sample(0))
    assert len(ring.pre) == n
    assert ring.push(sample(.1, True)) is None and ring.pending is None


def test_csv_tail_waits_for_complete_row(tmp_path):
    path = tmp_path/'imu.csv'
    tail = CsvTail(path)
    assert tail.read() == []
    path.write_text('one,two\n1,')
    assert tail.read() == []
    with path.open('a') as f:
        f.write('2\n3,4\n')
    assert tail.read() == [{'one': '1', 'two': '2'}, {'one': '3', 'two': '4'}]
    assert tail.read() == []


@dataclass
class Frame:
    frame_id: int
    ts_ns: int


def test_svo_sequence_preserves_every_unique_frame_and_exact_window(tmp_path):
    from Cameras.svo_sequence import SvoSequence
    data = {'a.svo2': [0, 10, 20, 30], 'b.svo2': [30, 40, 50, 60]}
    closed = []
    class Camera:
        def __init__(self, cfg, svo_path):
            self.name = Path(svo_path).name
            self.frames = iter(data[self.name])
        def open(self):
            pass
        def close(self):
            closed.append(self.name)
        def grab(self):
            t = next(self.frames, None)
            return None if t is None else Frame(123, t)
    manifest = tmp_path/'camera_manifest.json'
    manifest.write_text(json.dumps({'start_ns': 10, 'end_ns': 50,
                                    'segments': [{'path': n} for n in data]}))
    cam = SvoSequence(None, manifest, Camera)
    cam.open()
    frames = []
    while True:
        f = cam.grab()
        if f is None:
            break
        frames.append(f)
    cam.close()
    assert [f.ts_ns for f in frames] == [10, 20, 30, 40, 50]
    assert [f.frame_id for f in frames] == [1, 2, 3, 4, 5]
    assert closed == list(data)


def test_continuous_recorder_closes_parts_before_publishing_manifest(tmp_path, monkeypatch):
    from Cameras.rolling_record import record_segments
    import Cameras.rolling_record as recording
    clock = iter(np.arange(0, 100, .1))
    monkeypatch.setattr(recording.time, 'monotonic', lambda: next(clock))
    closed = set()
    class Camera:
        t = 0
        active = None
        def info(self):
            return {'resolution': [960, 600], 'fps': 120}
        def calibration(self):
            return SimpleNamespace(to_dict=lambda: {})
        def estimate_gravity(self, seconds):
            return None
        def start_recording(self, path, codec):
            assert self.active is None
            self.active = path
            Path(path).write_bytes(b'video')
        def stop_recording(self):
            assert self.active is not None
            closed.add(Path(self.active).name)
            self.active = None
        def grab_timestamp_only(self):
            # Every listed part must already have had the recorder closed.
            manifest = tmp_path/'segments.jsonl'
            if manifest.exists():
                assert all(json.loads(l)['path'] in closed for l in manifest.read_text().splitlines())
            self.t += 100000000
            if self.t == 2500000000:
                write_json(tmp_path/'seal_request.json', {'sequence': 1, 'through_ns': self.t})
            return self.t if self.t <= 11000000000 else None
        def dropped_frames(self):
            return 0
        def close(self):
            assert self.active is None
    record_segments(Camera(), SimpleNamespace(to_dict=lambda: {}), tmp_path)
    parts = [json.loads(l) for l in (tmp_path/'segments.jsonl').read_text().splitlines()]
    assert len(parts) >= 3
    assert sum(p['frames'] for p in parts) == 110
    assert parts[0]['last_ns'] == 2500000000
    assert json.loads((tmp_path/'recorder_status.json').read_text())['running'] is False


def test_trigger_window_queues_only_closed_overlapping_parts_and_keeps_capture_alive(tmp_path):
    from dckft.config import FusionConfig
    cfg = FusionConfig()
    backend = SimpleNamespace(cfg=cfg, say=lambda _: None)
    capture = TriggerCapture(backend, tmp_path)
    capture.parts.mkdir(exist_ok=True)
    for name in ('camera_config.json', 'fusion_config.json'):
        (tmp_path/name).write_text('{}')
    rows = []
    for t in np.arange(0, 7.01, .01):
        row = {'imu_meas_time': str(t), 'tx_time_stamp': str(round(t*1e6)),
               'ts_ns_host': str(round((100+t)*1e9)),
               **{'omega_'+a: str(12 if a == 'x' and 4 <= t < 4.2 else 0) for a in 'xyz'},
               **{'accel_'+a: '0' for a in 'xyz'}}
        w = capture.ring.push(sample_from_row(row))
        if w:
            capture.persist_window(w)
    folder, details = capture.pending[0]
    assert details['start_ns'] == pytest.approx(101e9, abs=1000)
    assert details['end_ns'] == pytest.approx(107e9, abs=1000)
    parts = [{'path': 'before.svo2', 'first_ns': 90e9, 'last_ns': 100e9},
             {'path': 'a.svo2', 'first_ns': 100e9, 'last_ns': 104e9},
             {'path': 'b.svo2', 'first_ns': 104.01e9, 'last_ns': 107.1e9}]
    for p in parts:
        source = capture.parts/p['path']
        source.write_bytes(b'video')
        Path(str(source)+'.meta.json').write_text('{}')
        with source.with_suffix('.frames.csv').open('w') as f:
            f.write('frame_id,ts_ns_image\n1,%d\n' % p['first_ns'])
    manifest = capture.parts/'segments.jsonl'
    manifest.write_text('\n'.join(json.dumps(p) for p in parts[:2])+'\n')
    capture.queue_ready()
    assert capture.work.empty()
    with manifest.open('a') as f:
        f.write(json.dumps(parts[2])+'\n')
    capture.queue_ready()
    assert not capture.pending and capture.work.get_nowait() == folder
    saved = json.loads((folder/'camera_manifest.json').read_text())
    assert [p['path'] for p in saved['segments']] == ['a.svo2', 'b.svo2']
    assert not (folder/'before.svo2').exists()
    assert (capture.parts/'a.svo2').read_bytes() == (folder/'a.svo2').read_bytes()


def test_offline_detection_never_stops_live_children(tmp_path, monkeypatch):
    import app.backend as be
    monkeypatch.setattr(be, 'SWINGS', str(tmp_path/'swings'))
    b = be.Backend()
    live = object()
    b.procs = [live]
    (tmp_path/'camera_manifest.json').write_text('{}')
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(be.subprocess, 'run', run)
    b._detect_recording(str(tmp_path))
    assert b.procs == [live]
    assert '--svo-manifest' in calls[0] and '--preview' not in calls[0]


def test_imu_seconds_timeout_is_checked_even_when_serial_is_silent(monkeypatch):
    import serial
    import dckft.imu.reader as reader
    from dckft.config import ImuConfig
    clock = iter([0, 0, .2, .4, .6, .8, 1.01, 1.02])
    monkeypatch.setattr(reader.time, 'time', lambda: next(clock))
    monkeypatch.setattr(reader.signal, 'signal', lambda *a: None)
    class Port:
        in_waiting = 0
        def reset_input_buffer(self):
            pass
        def read(self, n):
            return b''
        def close(self):
            pass
    monkeypatch.setattr(serial, 'Serial', lambda *a, **kw: Port())
    cfg = ImuConfig(port='/dev/fake')
    result = reader.run_serial(cfg, [], duration_s=1, quiet=True)
    assert result['samples'] == 0 and result['seconds'] >= 1


def test_web_default_is_continuous_and_render_has_xyz_controls():
    from app.server import INDEX, swing_page
    assert 'Continuous: IMU trigger' in INDEX
    assert "options.get('seconds'" not in INDEX
    page = swing_page({'id': 'test', 'files': ['replay.json']})
    assert 'replay-coordinates' in page and 'YZ (side)' in page
    assert 'Z up' in page and 'speedplot' in page


def test_triggered_recording_produces_camera_led_replay_and_keeps_six_second_bounds(tmp_path, monkeypatch):
    import shutil
    import app.backend as be
    from dckft.simulate import SimParams, simulate, write_session
    sim = simulate(SimParams(seed=7, stance_s=4., tail_s=4.))
    raw = write_session(sim, str(tmp_path/'simulation'))
    monkeypatch.setattr(be, 'DATA', str(tmp_path))
    monkeypatch.setattr(be, 'SWINGS', str(tmp_path/'swings'))
    b = be.Backend()
    session = tmp_path/'continuous'
    session.mkdir()
    b._capture_commands(str(session))
    capture = TriggerCapture(b, session)
    with open(raw['imu']) as f:
        for row in csv.DictReader(f):
            done = capture.ring.push(sample_from_row(row))
            if done:
                capture.persist_window(done)
    assert len(capture.pending) == 1
    folder, window = capture.pending[0]
    payloads = [p for p in sim['payloads'] if window['start_ns'] <= p['ts_ns_image'] <= window['end_ns']]
    (folder/'camera_session.jsonl').write_text('\n'.join(json.dumps(p) for p in payloads)+'\n')
    shutil.copy2(raw['camera']+'.meta.json', folder/'camera_session.jsonl.meta.json')
    with (folder/'camera_frames.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'ts_ns_image', 'accepted_3d'])
        writer.writerows([p['frame_id'], p['ts_ns_image'], len(p['points'])] for p in payloads)
    # The SDK decoder is exercised with timestamp-boundary fakes above; use its
    # resulting observations here to exercise the real fusion/export path.
    write_json(folder/'camera_manifest.json', dict(window, segments=[]))
    sid = b._save_recorded_session(str(folder))
    saved = tmp_path/'swings'/sid
    meta = json.loads((saved/'meta.json').read_text())
    assert meta['source'] == 'hardware (IMU trigger)'
    assert meta['capture_window']['end_ns']-meta['capture_window']['start_ns'] == 6000000000
    replay = json.loads((saved/'replay.json').read_text())
    assert {'camera_supported', 'imu_estimate'} <= {f['source'] for f in replay['frames']}
    assert (saved/'swing.png').is_file()
    assert (saved/'window.json').is_file()
    assert max(f['t'] for f in replay['frames']) < 2  # replay is the swing, not the held finish


def test_processing_failure_does_not_prevent_the_next_trigger_window(tmp_path):
    from dckft.config import FusionConfig
    calls = []
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir(); second.mkdir()
    def detect(path):
        calls.append(path)
        if path == str(first):
            raise ValueError('bad optical window')
    backend = SimpleNamespace(cfg=FusionConfig(), say=lambda _: None, _detect_recording=detect,
                              _save_recorded_session=lambda p: calls.append('saved '+p))
    capture = TriggerCapture(backend, tmp_path)
    capture.work.put(first); capture.work.put(second); capture.work.put(None)
    capture.process_queue()
    assert capture.failed == capture.saved == 1
    assert calls == [str(first), str(second), 'saved '+str(second)]
    assert (first/'processing_error.txt').is_file()
