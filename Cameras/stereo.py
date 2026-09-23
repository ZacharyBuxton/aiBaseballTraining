"""
stereo.py

Triangulates marker centroids from the ZED X's rectified left/right images.

The ZED SDK returns rectified images by default, so a marker lands on the same
row in both views and its depth follows directly from the column offset
(disparity):

    Z = fx * B / d        X = (uL - cx) * Z / fx        Y = (vL - cy) * Z / fy

Coordinates are in the left camera's rectified frame, OpenCV convention
(x right, y down, z forward), in meters. This matches sl.COORDINATE_SYSTEM.IMAGE
with sl.UNIT.METER, so SDK depth and our own triangulation are interchangeable.

Depth resolution degrades with the square of distance:

    dZ ~= Z^2 / (fx * B) * dd

where dd is the disparity error in pixels. print_depth_budget() evaluates this
with the camera's real calibration, which is why marker centroids are refined to
sub-pixel precision in marker_detector.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from marker_detector import Detection


@dataclass(frozen=True)
class StereoCalib:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "baseline_m": self.baseline_m, "width": self.width, "height": self.height}


@dataclass
class StereoPoint:
    marker_id: str
    xyz: Tuple[float, float, float]
    disparity_px: float
    row_mismatch_px: float
    uv_left: Tuple[float, float]
    uv_right: Tuple[float, float]


def triangulate(u_left: float, v_left: float, u_right: float,
                calib: StereoCalib) -> Optional[Tuple[float, float, float]]:
    """Rectified-stereo triangulation. Returns None for non-positive disparity."""
    d = u_left - u_right
    if d <= 0:
        return None
    z = calib.fx * calib.baseline_m / d
    x = (u_left - calib.cx) * z / calib.fx
    y = (v_left - calib.cy) * z / calib.fy
    return float(x), float(y), float(z)


def depth_error(z_m: float, calib: StereoCalib, disparity_err_px: float = 1.0) -> float:
    """Approximate depth uncertainty (m) at distance z for a given disparity error."""
    return (z_m ** 2) / (calib.fx * calib.baseline_m) * disparity_err_px


def match_and_triangulate(left: Dict[str, Detection], right: Dict[str, Detection],
                          calib: StereoCalib, max_row_mismatch_px: float,
                          z_min_m: float, z_max_m: float) -> Dict[str, StereoPoint]:
    """Pair detections by marker id, reject bad pairs, and triangulate the rest."""
    out: Dict[str, StereoPoint] = {}
    for marker_id, dl in left.items():
        dr = right.get(marker_id)
        if dr is None:
            continue
        row_err = abs(dl.v - dr.v)
        if row_err > max_row_mismatch_px:
            continue
        # Average the rows: both are measurements of the same epipolar line.
        v = 0.5 * (dl.v + dr.v)
        xyz = triangulate(dl.u, v, dr.u, calib)
        if xyz is None or not (z_min_m <= xyz[2] <= z_max_m):
            continue
        out[marker_id] = StereoPoint(marker_id, xyz, dl.u - dr.u, row_err,
                                     (dl.u, dl.v), (dr.u, dr.v))
    return out


def sample_xyz(xyz_map: np.ndarray, u: float, v: float, win: int = 2) -> Optional[Tuple[float, float, float]]:
    """
    Robust lookup into an SDK XYZ point cloud (H x W x 4, meters).
    Takes the median of finite values in a (2*win+1)^2 window around (u, v).
    """
    H, W = xyz_map.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return None
    patch = xyz_map[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1, :3].reshape(-1, 3)
    patch = patch[np.isfinite(patch).all(axis=1)]
    if patch.shape[0] == 0:
        return None
    x, y, z = np.median(patch, axis=0)
    return float(x), float(y), float(z)


def print_depth_budget(calib: StereoCalib, distances=(1.5, 2.0, 3.0, 4.0, 5.0),
                       disparity_errs=(1.0, 0.25, 0.1)) -> None:
    print(f"  fx={calib.fx:.1f}px  baseline={calib.baseline_m * 1000:.1f}mm  "
          f"image={calib.width}x{calib.height}")
    header = "  Z (m) | " + " | ".join(f"dZ @ {e:g}px" for e in disparity_errs)
    print(header)
    for z in distances:
        cells = " | ".join(f"{depth_error(z, calib, e) * 100:8.2f} cm" for e in disparity_errs)
        print(f"  {z:5.1f} | {cells}")
