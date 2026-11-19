"""Continuous raw capture, elapsed-time IMU trigger, and queued offline analysis."""
import csv
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback

import numpy as np

from dckft.data import imu_from_samples
from dckft.run import make_ring
from dckft.time_ring import TimeSwingRing


class CsvTail:
    """Read complete rows only; a concurrent writer may stop midway through a line."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = ''
        self.header = None

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open() as f:
            f.seek(self.offset)
            text = self.pending+f.read()
            self.offset = f.tell()
        lines = text.split('\n')
        self.pending = lines.pop()
        out = []
        for line in lines:
            if not line.strip():
                continue
            values = next(csv.reader([line]))
            if self.header is None:
                self.header = values
            elif len(values) == len(self.header):
                out.append(dict(zip(self.header, values)))
        return out


def sample_from_row(row):
    return {'t_dev_s': float(row['imu_meas_time']),
            'tx_us': int(row['tx_time_stamp']), 'ts_ns_host': int(row['ts_ns_host']),
            'gyro': [float(row['omega_'+a]) for a in 'xyz'],
            'accel': [float(row['accel_'+a]) for a in 'xyz'],
            'accel_hg': [float(row.get('accel_'+a+'2', 0)) for a in 'xyz'],
            'impact_flag': bool(float(row.get('impact_flag', 0))),
            'raw_row': row}


def link_file(source, target):
    """Closed video parts are immutable; hard links preserve raw data without duplication."""
    if target.exists():
        return
    try:
        os.link(str(source), str(target))
    except OSError:
        shutil.copy2(str(source), str(target))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


class TriggerCapture:
    def __init__(self, backend, session):
        self.backend, self.session = backend, Path(session)
        self.parts = self.session/'camera_parts'
        self.parts.mkdir(parents=True, exist_ok=True)
        strict = make_ring(backend.cfg)
        sw = backend.cfg.swing
        self.ring = TimeSwingRing(strict.threshold, strict.signal_of,
                                  sw.capture_pre_s, sw.capture_post_s)
        self.tail = CsvTail(self.session/'imu_session.csv')
        self.pending = []
        self.work = queue.Queue()
        self.count = self.saved = self.failed = self.processing = 0
        self.camera_first = None
        self.finished = False
        self.worker = threading.Thread(target=self.process_queue, daemon=True)

    def progress(self):
        return '%s | saved %d | processing %d | queued %d | IMU gaps %d' % (
            self.ring.progress, self.saved, self.processing, self.work.qsize(), self.ring.gaps)

    def run(self):
        b = self.backend
        imu_cmd, _ = b._capture_commands(str(self.session))
        camera_cmd = [os.sys.executable, '-u', os.path.join(b.repo, 'Cameras', 'track_bat.py'),
                      '--config', str(self.session/'camera_config.json'), '--record-only',
                      '--no-zmq', '--record-dir', str(self.parts)]
        self.worker.start()
        started = last_warning = time.monotonic()
        try:
            imu_proc = b._start_child('imu', imu_cmd, str(self.session))
            camera_proc = b._start_child('camera', camera_cmd, str(self.session))
            b.say('Continuous capture opening. Wait for the three-second pre-roll to arm.')
            while not b.stop_evt.wait(.05):
                if imu_proc.poll() is not None:
                    raise RuntimeError('IMU reader stopped; see imu.log')
                if camera_proc.poll() is not None:
                    raise RuntimeError('Camera recorder stopped; see camera.log')
                self.poll()
                now = time.monotonic()
                if self.camera_first is None and now-started > 30:
                    raise RuntimeError('Camera did not start; see camera.log')
                if now-last_warning > 15:
                    last_warning = now
                    if self.ring.last_t is None or time.time_ns()-self.last_imu_host > 2e9:
                        b.say('Waiting for live IMU samples; camera recording is preserved. Check the transmitter and imu.log.')
        finally:
            b._stop_children()
            self.poll()  # drain the final CSV flush and closed SVO manifest
            if self.ring.pending is not None:
                self.persist_window(self.ring.finish_if_due(float('inf')), partial=True)
            self.queue_ready(final=True)
            self.finished = True
            b.mode, b.busy = 'processing', True
            b.say('Capture stopped. Finishing %d queued swing(s); raw session: %s' % (
                self.work.qsize()+self.processing, self.session))
            self.work.put(None)
            self.worker.join()

    def poll(self):
        status_path = self.parts/'recorder_status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            self.backend.n_frames['camera'] = status['frames']
            if self.camera_first is None and status.get('first_ns'):
                self.camera_first = status['first_ns']
                self.backend.say('Camera is recording. Hold still while the IMU pre-roll fills, then swing when armed.')
        for row in self.tail.read():
            sample = sample_from_row(row)
            self.backend.n_frames['imu'] += 1
            self.last_imu_host = sample['ts_ns_host']
            if self.camera_first is None or sample['ts_ns_host'] < self.camera_first:
                continue
            was_pending = self.ring.pending
            was_ready = self.ring.first_t is not None and self.ring.last_t-self.ring.first_t >= self.ring.pre_s
            done = self.ring.push(sample)
            if (not was_ready and self.ring.first_t is not None
                    and self.ring.last_t-self.ring.first_t >= self.ring.pre_s):
                self.backend.say('ARMED: swing once. Saving %.0f s before and %.0f s after the IMU threshold.' % (
                    self.ring.pre_s, self.ring.post_s))
            if was_pending is None and self.ring.pending is not None:
                self.backend.say('SWING TRIGGERED: collecting %.0f seconds after threshold ...' % self.ring.post_s)
            if done is not None:
                self.persist_window(done)
        if self.ring.pending is not None:
            trigger = self.ring.pending
            elapsed = (time.time_ns()-trigger['ts_ns_host'])*1e-9
            # Finish on elapsed time even if the radio stops. Keep only actual rows.
            if elapsed >= self.ring.post_s+.25:
                self.persist_window(self.ring.finish_if_due(trigger['t_dev_s']+elapsed))
        self.queue_ready()

    def persist_window(self, window, partial=False):
        self.count += 1
        folder = self.session/('swing_%03d' % self.count)
        folder.mkdir()
        rows = [s['raw_row'] for s in window.samples]
        with (folder/'imu_session.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        imu = imu_from_samples(window.samples)
        k = min(range(len(window.samples)), key=lambda i: abs(window.samples[i]['t_dev_s']-window.t_trigger))
        trigger_ns = int(round(imu.t[k]*1e9))
        raw_t = np.array([s['t_dev_s'] for s in window.samples])
        gaps = np.diff(raw_t)
        details = {'trigger_ns': trigger_ns, 'start_ns': trigger_ns-int(window.pre_s*1e9),
                   'end_ns': trigger_ns+int(window.post_s*1e9),
                   'pre_s': window.pre_s, 'post_s': window.post_s,
                   'threshold': self.ring.threshold, 'trigger_source': self.backend.cfg.swing.trigger_source,
                   'observed_imu_samples': len(rows), 'imu_gaps_over_10ms': int((gaps > .01).sum()),
                   'largest_imu_gap_ms': float(gaps.max()*1000) if len(gaps) else None,
                   'actual_imu_pre_s': float(window.t_trigger-raw_t[0]),
                   'actual_imu_post_s': float(raw_t[-1]-window.t_trigger),
                   'interrupted_post_roll': partial, 'raw_session': str(self.session)}
        write_json(folder/'window.json', details)
        for name in ('camera_config.json', 'fusion_config.json'):
            shutil.copy2(str(self.session/name), str(folder/name))
        self.pending.append((folder, details))
        write_json(self.parts/'seal_request.json', {'sequence': self.count,
                   'through_ns': details['end_ns']+20000000})
        self.backend.say('Swing window %d captured%s. Camera continues recording; analysis queued.' % (
            self.count, ' (Stop interrupted the post-roll)' if partial else ''))

    def queue_ready(self, final=False):
        manifest = self.parts/'segments.jsonl'
        if not manifest.exists():
            return
        text = manifest.read_text()
        # Ignore an append that has not finished its newline yet.
        closed = [json.loads(line) for line in text.splitlines(keepends=True) if line.endswith('\n')]
        for folder, details in list(self.pending):
            parts = [p for p in closed if p['last_ns'] >= details['start_ns'] and p['first_ns'] <= details['end_ns']]
            if not parts or (not final and max(p['last_ns'] for p in closed) < details['end_ns']):
                continue
            for part in parts:
                src = self.parts/part['path']
                link_file(src, folder/src.name)
                shutil.copy2(str(src)+'.meta.json', str(folder/src.name)+'.meta.json')
            shutil.copy2(str(folder/parts[0]['path'])+'.meta.json', str(folder/'camera_manifest.json.meta.json'))
            write_json(folder/'camera_manifest.json', dict(details, segments=parts))
            with (folder/'capture_frames.csv').open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=['frame_id', 'ts_ns_image'])
                writer.writeheader()
                for part in parts:
                    with (self.parts/part['path']).with_suffix('.frames.csv').open() as f:
                        for row in csv.DictReader(f):
                            if details['start_ns'] <= int(row['ts_ns_image']) <= details['end_ns']:
                                writer.writerow(row)
            self.work.put(folder)
            self.pending.remove((folder, details))
        if final and self.pending:
            self.backend.say('Some trigger windows have no closed camera data. Raw windows remain in '+str(self.session))

    def process_queue(self):
        while True:
            folder = self.work.get()
            if folder is None:
                self.work.task_done()
                return
            self.processing = 1
            try:
                self.backend._detect_recording(str(folder))
                self.backend._save_recorded_session(str(folder))
                self.saved += 1
            except Exception as exc:
                self.failed += 1
                (folder/'processing_error.txt').write_text(traceback.format_exc())
                self.backend.say('Swing processing failed: %s. Raw window kept: %s' % (exc, folder))
            finally:
                self.processing = 0
                self.work.task_done()
