"""
ring.py -- the advisor's ring-buffer swing trigger (from Inertial_Sensors_Capture.py, Nov 2026).

The IMU streams continuously. A rolling pre-roll of PRE frames is always kept; when the trigger
signal crosses ABOVE the threshold (a rising edge, after at least one quiet sample) the window
becomes [PRE frames before] + [the trigger frame and the following POST-1 frames]. Defaults:
1200 + 1200 frames = 3 s + 3 s at 400 Hz, trigger |omega| > 10 rad/s on the ICM45686 gyro.

States, exactly as in the example:
  arm       filling the pre-roll; a rising edge with a full pre-roll starts a window
  post      collecting the post-trigger frames; when complete the window is returned, and the
            last PRE frames of it seed the next pre-roll
  cooldown  the window ended while still above threshold: wait for a quiet sample

A window never spans a discontinuity: a duplicate / out-of-order frame is ignored, and a gap
(device time step > max_dt_us, i.e. missing frames), a device reboot or an invalid sample resets
the ring and requires a quiet sample before the next edge -- so a saved window is always a
continuous 6 s of frames.

Camera frames are kept in a time-indexed buffer alongside (CameraRing) and cut to the same span.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

import numpy as np


@dataclass
class RingWindow:
    samples: list               # PRE + POST samples, in order
    trigger_index: int          # index of the frame that crossed the threshold (== PRE)
    t_trigger: float            # its time (s, same clock as time_of)
    t_first: float
    t_last: float


class SwingRing:
    def __init__(self, pre_frames: int = 1200, post_frames: int = 1200, threshold: float = 10.0,
                 signal_of: Callable = None, time_of: Callable = None, valid_of: Callable = None,
                 max_dt_s: float = 0.00375):
        if not (math.isfinite(threshold) and threshold > 0):
            raise ValueError("trigger threshold must be finite and positive")
        self.PRE, self.POST = int(pre_frames), int(post_frames)
        self.threshold = float(threshold)
        self.signal_of = signal_of or (lambda s: float(np.linalg.norm(s["gyro"])))
        self.time_of = time_of or (lambda s: s["t_dev_s"])
        self.valid_of = valid_of or (lambda s: True)
        self.max_dt_s = max_dt_s
        self.pre: Deque = deque(maxlen=self.PRE)
        self.window: list = []
        self.state = "arm"
        self.below = True
        self.last_t: Optional[float] = None
        self.windows_done = 0
        self.discontinuities = 0

    def reset(self) -> None:
        self.pre.clear()
        self.window = []
        self.state = "arm"
        self.below = False              # require an observed quiet sample before a new edge
        self.discontinuities += 1

    def push(self, sample) -> Optional[RingWindow]:
        t = self.time_of(sample)
        if self.last_t is not None:
            dt = t - self.last_t
            if dt <= 0:                 # duplicate / out of order
                return None
            if dt > self.max_dt_s:      # missing frames or reboot: never stitch across it
                self.reset()
        self.last_t = t
        sig = self.signal_of(sample)
        if not self.valid_of(sample) or not math.isfinite(sig):
            self.reset()
            return None
        above = sig > self.threshold
        if self.state == "post":
            self.window.append(sample)
            if len(self.window) >= self.PRE + self.POST:
                done = self.window
                self.window = []
                self.pre.clear()
                self.pre.extend(done[-self.PRE:])
                self.below = not above
                self.state = "cooldown" if above else "arm"
                self.windows_done += 1
                return RingWindow(done, self.PRE, self.time_of(done[self.PRE]),
                                  self.time_of(done[0]), self.time_of(done[-1]))
            return None
        if self.state == "cooldown":
            self.pre.append(sample)
            if not above:
                self.state = "arm"
                self.below = True
            return None
        if above and self.below and len(self.pre) >= self.PRE:
            self.window = list(self.pre)
            self.window.append(sample)
            self.state = "post"
            self.below = False
            return None
        self.pre.append(sample)
        self.below = not above
        return None

    @property
    def progress(self) -> str:
        if self.state == "post":
            return f"recording swing {len(self.window) - self.PRE}/{self.POST}"
        return f"{self.state} (pre-roll {len(self.pre)}/{self.PRE})"


class CameraRing:
    """Camera payloads kept by capture time (ts_ns_image), long enough to cover a ring window."""

    def __init__(self, seconds: float = 10.0, fps: float = 150.0):
        self.buf: Deque[dict] = deque(maxlen=int(seconds * fps))

    def push(self, payload: dict) -> None:
        self.buf.append(payload)

    def latest_t(self) -> float:
        return self.buf[-1]["ts_ns_image"] * 1e-9 if self.buf else -math.inf

    def cut(self, t0: float, t1: float) -> List[dict]:
        return [p for p in self.buf if t0 <= p["ts_ns_image"] * 1e-9 <= t1]
