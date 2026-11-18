"""Elapsed-time trigger windows for recordings with real radio packet gaps.

The strict sample-count ring used by the streaming CLI is unchanged. This ring
keeps observed samples only, and records gaps instead of resetting on each loss.
"""
from collections import deque
from dataclasses import dataclass
import math


@dataclass
class TimedWindow:
    samples: list
    trigger: dict
    t_first: float
    t_last: float
    pre_s: float
    post_s: float

    @property
    def t_trigger(self):
        return self.trigger['t_dev_s']


class TimeSwingRing:
    def __init__(self, threshold=10., signal_of=None, pre_s=3., post_s=3., gap_s=.01):
        if not all(math.isfinite(v) and v > 0 for v in (threshold, pre_s, post_s, gap_s)):
            raise ValueError('Positive finite trigger, window durations and gap size required')
        self.threshold, self.pre_s, self.post_s = threshold, pre_s, post_s
        self.signal_of = signal_of or (lambda sample: math.sqrt(sum(v*v for v in sample['gyro'])))
        self.gap_s = gap_s
        self.pre = deque()
        self.pending = None
        self.window = []
        self.last_t = self.first_t = None
        self.below = False
        self.gaps = self.restarts = self.windows_done = 0
        self.largest_gap_s = 0.

    def push(self, sample):
        t = float(sample['t_dev_s'])
        value = float(self.signal_of(sample))
        if not math.isfinite(t) or not math.isfinite(value):
            return None
        if self.last_t is not None:
            dt = t-self.last_t
            if dt < -.5:  # device reboot: never combine different device clocks
                self.pre.clear()
                self.pending, self.window, self.first_t = None, [], None
                self.below = False
                self.restarts += 1
            elif dt <= 0:
                return None
            elif dt > self.gap_s:
                self.gaps += 1
                self.largest_gap_s = max(self.largest_gap_s, dt)
        self.last_t = t
        if self.first_t is None:
            self.first_t = t
        self.pre.append(sample)
        while len(self.pre) > 1 and self.pre[1]['t_dev_s'] < t-self.pre_s:
            self.pre.popleft()
        above = value > self.threshold
        if self.pending is not None:
            if t <= self.pending['t_dev_s']+self.post_s+1e-8:
                self.window.append(sample)
            self.below = not above
            return self.finish_if_due(t)
        ready = t-self.first_t >= self.pre_s
        if ready and above and self.below:
            self.pending = sample
            self.window = [s for s in self.pre if s['t_dev_s'] >= t-self.pre_s-1e-8]
        self.below = not above
        return None

    def finish_if_due(self, device_time):
        """A wall-clock watchdog can finish a window even if the radio goes silent."""
        if self.pending is None or device_time < self.pending['t_dev_s']+self.post_s-1e-8:
            return None
        trigger, samples = self.pending, self.window
        self.pending, self.window = None, []
        self.windows_done += 1
        return TimedWindow(samples, trigger, trigger['t_dev_s']-self.pre_s,
                           trigger['t_dev_s']+self.post_s, self.pre_s, self.post_s)

    @property
    def progress(self):
        if self.pending is not None:
            elapsed = min(self.post_s, max(0., self.last_t-self.pending['t_dev_s']))
            return 'triggered: %.1f / %.1f s after threshold' % (elapsed, self.post_s)
        elapsed = 0. if self.first_t is None else min(self.pre_s, self.last_t-self.first_t)
        return ('armed' if elapsed >= self.pre_s else 'warming pre-roll') + ' (%.1f / %.1f s)' % (elapsed, self.pre_s)
