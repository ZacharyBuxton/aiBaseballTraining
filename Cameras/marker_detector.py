"""
marker_detector.py

Color-marker detection for bat tracking, adapted from the previous team's
HSV blob detector (legacy/Device_Tracking_2_16.py) with three changes:

  1. Two-stage detection. A coarse search runs on a downscaled frame for speed,
     then each hit is re-measured on a small full-resolution patch. Stereo depth
     comes from the left/right pixel difference, so centroid precision matters
     far more here than it did with the Kinect's depth map.
  2. If a marker isn't found inside its predicted search window, the detector
     falls back to a full-frame search on the same frame instead of waiting a frame.
  3. No camera or display dependencies, so it can be unit tested and run on
     recorded SVO files.

Use one detector instance per image stream (one for left, one for right), since
each instance keeps its own motion prediction state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from config import MarkerConfig


@dataclass
class Detection:
    marker_id: str
    u: float        # full-resolution column, sub-pixel
    v: float        # full-resolution row, sub-pixel
    area: float     # full-resolution pixel area
    bbox: Tuple[int, int, int, int]   # x, y, w, h at full resolution


def _hue_ranges(center: int, tol: int, s_min: int, s_max: int, v_min: int, v_max: int):
    """HSV inRange bounds for one hue center, splitting at the 0/180 wrap."""
    lo_h, hi_h = center - tol, center + tol
    if lo_h < 0:
        return [((0, s_min, v_min), (hi_h, s_max, v_max)),
                ((180 + lo_h, s_min, v_min), (179, s_max, v_max))]
    if hi_h > 179:
        return [((lo_h, s_min, v_min), (179, s_max, v_max)),
                ((0, s_min, v_min), (hi_h - 180, s_max, v_max))]
    return [((lo_h, s_min, v_min), (hi_h, s_max, v_max))]


class ColorMarkerDetector:
    def __init__(self, cfg: MarkerConfig):
        self.cfg = cfg
        self.ranges: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for marker_id, centers in cfg.hues.items():
            spans = []
            for c in centers:
                for lo, hi in _hue_ranges(c, cfg.hue_tol, cfg.s_min, cfg.s_max, cfg.v_min, cfg.v_max):
                    spans.append((np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
            self.ranges[marker_id] = spans
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        self.last_pos: Dict[str, Optional[Tuple[float, float]]] = {m: None for m in cfg.hues}
        self.last_vel: Dict[str, Tuple[float, float]] = {m: (0.0, 0.0) for m in cfg.hues}

    # ------------------------------------------------------------------ helpers
    def _mask(self, hsv: np.ndarray, marker_id: str) -> np.ndarray:
        mask = None
        for lo, hi in self.ranges[marker_id]:
            m = cv.inRange(hsv, lo, hi)
            mask = m if mask is None else cv.bitwise_or(mask, m)
        return mask

    def _blobs(self, mask: np.ndarray, min_area: float):
        n, _, stats, cents = cv.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            area = float(stats[i, cv.CC_STAT_AREA])
            if area >= min_area:
                x, y, w, h = (int(s) for s in stats[i, :4])
                out.append((float(cents[i][0]), float(cents[i][1]), area, (x, y, w, h)))
        return out

    def _refine(self, bgr: np.ndarray, marker_id: str, bbox_full) -> Optional[Tuple[float, float, float]]:
        """Re-measure the centroid on a full-resolution patch around the coarse bbox."""
        H, W = bgr.shape[:2]
        x, y, w, h = bbox_full
        p = self.cfg.refine_pad_px
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(W, x + w + p), min(H, y + h + p)
        if x1 <= x0 or y1 <= y0:
            return None
        hsv = cv.cvtColor(bgr[y0:y1, x0:x1], cv.COLOR_BGR2HSV)
        mask = self._mask(hsv, marker_id)
        mom = cv.moments(mask, binaryImage=True)
        if mom["m00"] <= 0:
            return None
        return x0 + mom["m10"] / mom["m00"], y0 + mom["m01"] / mom["m00"], float(mom["m00"])

    # --------------------------------------------------------------------- main
    def detect(self, bgr: np.ndarray) -> Dict[str, Detection]:
        """Find each configured marker in a BGR frame. Returns {marker_id: Detection}."""
        cfg = self.cfg
        H, W = bgr.shape[:2]
        s = cfg.detection_scale
        small = bgr if s == 1.0 else cv.resize(bgr, (int(W * s), int(H * s)), interpolation=cv.INTER_AREA)
        hsv_small = cv.cvtColor(small, cv.COLOR_BGR2HSV)
        hs, ws = hsv_small.shape[:2]
        min_area_small = max(1.0, cfg.min_area_px * s * s)

        results: Dict[str, Detection] = {}
        for marker_id in self.ranges:
            mask = self._mask(hsv_small, marker_id)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)

            # Predict where the marker should be (constant velocity, full-res pixels).
            pred = None
            if self.last_pos[marker_id] is not None:
                lx, ly = self.last_pos[marker_id]
                vx, vy = self.last_vel[marker_id]
                pred = (lx + vx, ly + vy)

            blobs = []
            if pred is not None:
                r = int(cfg.roi_radius_px * s)
                px, py = int(pred[0] * s), int(pred[1] * s)
                x0, y0 = max(0, px - r), max(0, py - r)
                x1, y1 = min(ws, px + r), min(hs, py + r)
                if x1 > x0 and y1 > y0:
                    blobs = [(cx + x0, cy + y0, a, (bx + x0, by + y0, bw, bh))
                             for cx, cy, a, (bx, by, bw, bh) in self._blobs(mask[y0:y1, x0:x1], min_area_small)]
            if not blobs:
                blobs = self._blobs(mask, min_area_small)

            if not blobs:
                self._lost(marker_id)
                continue

            if pred is None:
                best = max(blobs, key=lambda b: b[2])
            else:
                best = min(blobs, key=lambda b: (b[0] / s - pred[0]) ** 2 + (b[1] / s - pred[1]) ** 2)

            cx, cy, area, (bx, by, bw, bh) = best
            bbox_full = (int(bx / s), int(by / s), int(np.ceil(bw / s)), int(np.ceil(bh / s)))
            refined = self._refine(bgr, marker_id, bbox_full)
            if refined is not None:
                u, v, area_full = refined
            else:
                u, v, area_full = cx / s, cy / s, area / (s * s)

            if area_full < cfg.min_area_px:
                self._lost(marker_id)
                continue

            if self.last_pos[marker_id] is not None:
                lx, ly = self.last_pos[marker_id]
                self.last_vel[marker_id] = (u - lx, v - ly)
            self.last_pos[marker_id] = (u, v)
            results[marker_id] = Detection(marker_id, u, v, area_full, bbox_full)

        return results

    # ------------------------------------------------- same-color bands / stripe
    def detect_blobs(self, bgr: np.ndarray, color_id: str, max_blobs: int) -> List[Detection]:
        """
        All pink bands in one image: the max_blobs largest blobs of color_id, each re-measured at
        full resolution. Same-color blobs can't be told apart here, so every Detection carries
        marker_id=color_id; band_labeler.py names them after triangulation (order along the bat).
        A full-frame coarse search every frame -- no per-band prediction window to mix up.
        """
        cfg = self.cfg
        H, W = bgr.shape[:2]
        s = cfg.detection_scale
        small = bgr if s == 1.0 else cv.resize(bgr, (int(W * s), int(H * s)), interpolation=cv.INTER_AREA)
        hsv_small = cv.cvtColor(small, cv.COLOR_BGR2HSV)
        mask = self._mask(hsv_small, color_id)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.kernel)
        blobs = sorted(self._blobs(mask, max(1.0, cfg.min_area_px * s * s)),
                       key=lambda b: -b[2])[:max_blobs]
        out: List[Detection] = []
        for cx, cy, area, (bx, by, bw, bh) in blobs:
            bbox_full = (int(bx / s), int(by / s), int(np.ceil(bw / s)), int(np.ceil(bh / s)))
            refined = self._refine(bgr, color_id, bbox_full)
            u, v, a = refined if refined is not None else (cx / s, cy / s, area / (s * s))
            if a >= cfg.min_area_px:
                out.append(Detection(color_id, u, v, a, bbox_full))
        return out

    def detect_stripe_ends(self, bgr: np.ndarray, color_id: str) -> List[Detection]:
        """
        Both ends of one long pink stripe: the largest blob's pixels are projected onto their
        principal axis and the 2nd / 98th percentiles taken as the ends (robust to ragged tape
        edges). Returns [] or two Detections (unordered; band_labeler.py decides which end is
        the knob side).
        """
        dets = self.detect_blobs(bgr, color_id, 1)
        if not dets:
            return []
        x, y, w, h = dets[0].bbox
        p = self.cfg.refine_pad_px
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        hsv = cv.cvtColor(bgr[y0:y1, x0:x1], cv.COLOR_BGR2HSV)
        ys, xs = np.nonzero(self._mask(hsv, color_id))
        if len(xs) < self.cfg.min_area_px:
            return []
        pts = np.column_stack([xs + x0, ys + y0]).astype(float)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        proj = (pts - c) @ vt[0]
        lo, hi = np.percentile(proj, [2, 98])
        if hi - lo < 4.0:                    # not elongated: can't define two ends
            return []
        ends = []
        for t in (lo, hi):
            near = np.abs(proj - t) <= max(1.5, 0.03 * (hi - lo))
            u, v = pts[near].mean(0)          # average across the stripe's width at that end
            ends.append(Detection(color_id, float(u), float(v), float(near.sum()), dets[0].bbox))
        return ends

    def _lost(self, marker_id: str) -> None:
        self.last_pos[marker_id] = None
        self.last_vel[marker_id] = (0.0, 0.0)

    def reset(self) -> None:
        for m in self.last_pos:
            self._lost(m)


def draw_detections(bgr: np.ndarray, detections: Dict[str, Detection]) -> None:
    """Overlay detections in place (for the preview window)."""
    for det in detections.values():
        x, y, w, h = det.bbox
        cv.rectangle(bgr, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv.circle(bgr, (int(round(det.u)), int(round(det.v))), 4, (0, 0, 255), -1)
        cv.putText(bgr, det.marker_id, (x, max(0, y - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.4,
                   (255, 255, 255), 1, cv.LINE_AA)
