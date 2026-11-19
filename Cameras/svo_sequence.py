"""Decode a timestamp-bounded swing from closed SVO parts without filling frames."""
from dataclasses import replace
import json
from pathlib import Path


class SvoSequence:
    def __init__(self, cfg, manifest_path, factory):
        self.cfg, self.factory = cfg, factory
        self.svo_path = str(manifest_path)
        manifest = json.loads(Path(manifest_path).read_text())
        self.paths = [str(Path(manifest_path).parent/p['path']) for p in manifest['segments']]
        if not self.paths:
            raise ValueError('No camera SVO parts in this swing')
        self.start_ns, self.end_ns = manifest['start_ns'], manifest['end_ns']
        self.index = 0
        self.cam = None
        self.frame_id = 0
        self.last_ns = None

    def open(self):
        self.cam = self.factory(self.cfg, svo_path=self.paths[self.index])
        self.cam.open()

    def close(self):
        if self.cam is not None:
            self.cam.close()
            self.cam = None

    def grab(self):
        while self.cam is not None:
            frame = self.cam.grab()
            if frame is None:
                self.close()
                self.index += 1
                if self.index == len(self.paths):
                    return None
                self.open()
                continue
            if frame.ts_ns < self.start_ns or (self.last_ns is not None and frame.ts_ns <= self.last_ns):
                continue
            if frame.ts_ns > self.end_ns:
                return None
            self.frame_id += 1
            self.last_ns = frame.ts_ns
            return replace(frame, frame_id=self.frame_id)
        return None

    def info(self):
        return dict(self.cam.info(), source=self.svo_path)

    def calibration(self):
        return self.cam.calibration()

    def dropped_frames(self):
        # File playback is not a capture-drop audit; use decoded image timestamps.
        return 0
