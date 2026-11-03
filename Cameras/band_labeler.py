"""
band_labeler.py -- name same-color pink points by their place along the bat.

All bands are the same neon pink, so color can't say which is which. After triangulation the
points of one frame lie (nearly) on the bat line, so their ORDER along that line is known; what
is unknown is (a) which end is the knob and (b), when some bands are hidden, which ones are
missing. Both are settled by scoring every order-preserving assignment of the seen points to
the labels, in both directions:

  * continuity   distance from each point to where its label was predicted (constant
                 velocity from the last two frames) -- the main cue while tracking
  * spacing      if the band distances along the bat are known (band_s_m), how well the gaps
                 between the points match them -- why UNEQUAL spacing is recommended
  * gravity      with no history: the knob end is the lower end (bat held up in the stance).
                 Needs the gravity direction in the camera frame.

Without band_s_m, labelling starts only on a frame that shows EVERY band (a partial frame could
be any subset), and the spacing is then learned from such frames so later partial frames can
be identified too. Single-band frames are named by continuity only and never update history.

Labels: "bands" layout -> pink_1 (nearest the knob) ... pink_N; "stripe" layout ->
stripe_start (knob side), stripe_end. The fusion engine (dckft) uses the same names in
bat.marker_s_m.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np


def band_labels(layout: str, n_bands: int) -> List[str]:
    if layout == "stripe":
        return ["stripe_start", "stripe_end"]
    return [f"pink_{i + 1}" for i in range(n_bands)]


class BandLabeler:
    def __init__(self, labels: Sequence[str], s_m: Optional[Sequence[float]] = None,
                 gravity: Optional[Sequence[float]] = None, gate_m: float = 0.25,
                 max_age_s: float = 0.1, spacing_tol_m: float = 0.05):
        self.labels = list(labels)
        self.s = None if s_m is None else np.asarray(sorted(s_m), float)
        if self.s is not None and len(self.s) != len(self.labels):
            raise ValueError(f"{len(self.labels)} labels but {len(self.s)} band distances")
        g = np.array([0.0, 1.0, 0.0]) if gravity is None else np.asarray(gravity, float)
        self.down = g / np.linalg.norm(g)          # OpenCV camera: +y is down for a level camera
        self.gate = gate_m
        self.spacing_tol = spacing_tol_m
        self.max_age = max_age_s
        self.tracks: Dict[str, dict] = {}           # label -> {t, p, v}: last sighting + velocity
        self._gaps: List[np.ndarray] = []           # measured gaps from full frames (no s_m given)

    def reset(self) -> None:
        self.tracks = {}
        self._gaps = []

    def _predict(self, t: float) -> Dict[str, np.ndarray]:
        """Per-band constant-velocity prediction from each band's own last sightings."""
        pred = {}
        for name, tr in self.tracks.items():
            if t - tr["t"] > self.max_age:
                continue
            pred[name] = tr["p"] + tr["v"] * (t - tr["t"])
        return pred

    def _update_tracks(self, t: float, named: Dict[str, np.ndarray]) -> None:
        for name, p in named.items():
            tr = self.tracks.get(name)
            v = np.zeros(3)
            if tr is not None and 0 < t - tr["t"] <= self.max_age:
                v = (p - tr["p"]) / (t - tr["t"])
            self.tracks[name] = {"t": t, "p": np.asarray(p, float), "v": v}

    def label(self, t: float, points: Sequence[Sequence[float]]) -> Dict[str, np.ndarray]:
        """points: list of xyz (m) from one frame, any order. Returns {label: xyz}."""
        P = np.asarray(points, float).reshape(-1, 3)
        n_lab = len(self.labels)
        if len(P) == 0:
            return {}
        if len(P) > n_lab:                              # spurious blob: keep the most collinear set
            P = self._best_subset(P, n_lab)
        k = len(P)
        pred = self._predict(t)
        spacing = self.s if self.s is not None else self._learned_spacing()
        if not pred and spacing is None and k < n_lab:
            # nothing to track and no spacing to go on: a partial frame is ambiguous (which
            # bands are missing?), so wait for a frame that shows every band
            return {}

        # order along the line (single point: trivially ordered)
        if k >= 2:
            c = P.mean(0)
            d = np.linalg.svd(P - c, full_matrices=False)[2][0]
            P = P[np.argsort((P - c) @ d)]

        best, best_cost = None, np.inf
        for direction in (1, -1):
            Q = P if direction == 1 else P[::-1]          # Q[0] is taken as the knob-side point
            for idx in combinations(range(n_lab), k):     # which labels these points are
                cost = 0.0
                # 1) spacing: with known band distances the gaps identify WHICH bands these
                #    are (unequal spacing -> every pair/triple has its own gap pattern)
                if spacing is not None and k >= 2:
                    gaps = np.linalg.norm(np.diff(Q, axis=0), axis=1)
                    e = float(np.mean(np.abs(gaps - np.diff(spacing[list(idx)]))))
                    cost += e if e < self.spacing_tol else 100.0 + e
                # 2) continuity with the last confidently labelled frame
                if pred:
                    errs = [np.linalg.norm(Q[i] - pred[self.labels[j]]) if self.labels[j] in pred
                            else self.gate for i, j in enumerate(idx)]
                    cost += float(np.mean(errs)) + (1.0 if max(errs) > self.gate else 0.0)
                # 3) gravity, only when there is nothing to track: knob end is the lower end
                if not pred and k >= 2:
                    lower = Q[0] @ self.down >= Q[-1] @ self.down
                    cost += 0.0 if lower else (1.0 if spacing is None else 0.02)
                if cost < best_cost:
                    best_cost, best = cost, {self.labels[j]: Q[i] for i, j in enumerate(idx)}

        if k == 1 and n_lab > 1:
            # a lone point can only be named by continuity, and never updates the tracks (one
            # mistake there would propagate). It must be clearly closest to ONE prediction.
            if not pred:
                return {}
            d = sorted((np.linalg.norm(P[0] - q), n) for n, q in pred.items())
            if d[0][0] > 0.5 * self.gate or (len(d) > 1 and d[1][0] < 2.5 * d[0][0]):
                return {}
            return {d[0][1]: P[0]}
        if best_cost >= 100.0:                         # no assignment fits the band spacing
            return {}
        self._update_tracks(t, best)
        if k == n_lab and self.s is None and best_cost < 1.0:
            # every band seen and confidently named: remember the measured spacing
            Q = np.array([best[l] for l in self.labels])
            self._gaps.append(np.linalg.norm(np.diff(Q, axis=0), axis=1))
            self._gaps = self._gaps[-200:]
        return best

    def _learned_spacing(self) -> Optional[np.ndarray]:
        """Band distances learned from frames that showed every band (needs >= 10 of them)."""
        if len(self._gaps) < 10:
            return None
        return np.concatenate([[0.0], np.cumsum(np.median(self._gaps, axis=0))])

    @staticmethod
    def _best_subset(P: np.ndarray, n: int) -> np.ndarray:
        best, best_r = None, np.inf
        for idx in combinations(range(len(P)), n):
            S = P[list(idx)]
            if n < 3:
                r = 0.0
            else:
                c = S.mean(0)
                sv = np.linalg.svd(S - c, compute_uv=False)
                r = sv[1] + sv[2]                        # off-line spread
            if r < best_r:
                best_r, best = r, S
        return best
