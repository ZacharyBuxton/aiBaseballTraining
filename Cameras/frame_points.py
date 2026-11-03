"""
frame_points.py -- one camera frame -> named 3D bat points (no ZED SDK needed, so it is testable).

Used by track_bat.py for every frame, live or from an SVO. See detect_points().
"""

from __future__ import annotations

import numpy as np

from stereo import match_and_triangulate, match_unlabeled, sample_xyz


def gravity_in_camera(gravity):
    """ZED IMU estimate -> gravity direction in the left camera frame (None if unavailable)."""
    if not gravity or "accel_mean_mps2" not in gravity:
        return None
    acc = np.asarray(gravity["accel_mean_mps2"], float)
    T = np.asarray(gravity.get("camera_imu_transform", np.eye(4)), float)
    g = -(T[:3, :3] @ acc)          # accelerometer reads specific force (up); gravity is down
    n = np.linalg.norm(g)
    return g / n if 7.0 < n < 12.5 else None


def _point_entry(sp):
    return {"xyz": sp.xyz, "disparity_px": round(sp.disparity_px, 3),
            "row_err_px": round(sp.row_mismatch_px, 3),
            "uv_left": [round(sp.uv_left[0], 2), round(sp.uv_left[1], 2)],
            "uv_right": [round(sp.uv_right[0], 2), round(sp.uv_right[1], 2)]}


def detect_points(frame, cfg, calib, det_left, det_right, labeler, depth_source):
    """
    One frame -> (left detections, right detections, {label: point dict}).
      colors : one blob per color, labelled by color (legacy scheme)
      bands  : up to n_bands pink blobs per image, matched along epipolar rows, triangulated,
               then named pink_1..pink_N by band_labeler (order along the bat)
      stripe : the two ends of one pink stripe, named stripe_start / stripe_end
    """
    mk = cfg.markers
    zmin, zmax = cfg.camera.depth_min_m, cfg.camera.depth_max_m
    points = {}
    if mk.layout == "colors":
        left_dets = det_left.detect(frame.left)
        right_dets = det_right.detect(frame.right)
        if depth_source == "stereo":
            for m, sp in match_and_triangulate(left_dets, right_dets, calib, mk.max_row_mismatch_px,
                                               zmin, zmax).items():
                points[m] = _point_entry(sp)
        else:
            for m, d in left_dets.items():
                xyz = sample_xyz(frame.xyz, d.u, d.v)
                if xyz is not None and zmin <= xyz[2] <= zmax:
                    points[m] = {"xyz": xyz, "uv_left": [round(d.u, 2), round(d.v, 2)]}
        return left_dets, right_dets, points

    if mk.layout == "stripe":
        L = det_left.detect_stripe_ends(frame.left, mk.color_id)
        R = det_right.detect_stripe_ends(frame.right, mk.color_id)
    else:
        L = det_left.detect_blobs(frame.left, mk.color_id, mk.n_bands)
        R = det_right.detect_blobs(frame.right, mk.color_id, mk.n_bands)
    entries = []
    if depth_source == "stereo":
        for sp in match_unlabeled(L, R, calib, mk.max_row_mismatch_px, zmin, zmax):
            entries.append(_point_entry(sp))
    else:
        for d in L:
            xyz = sample_xyz(frame.xyz, d.u, d.v)
            if xyz is not None and zmin <= xyz[2] <= zmax:
                entries.append({"xyz": xyz, "uv_left": [round(d.u, 2), round(d.v, 2)]})
    named = labeler.label(frame.ts_ns * 1e-9, [e["xyz"] for e in entries])
    for label, xyz in named.items():
        e = min(entries, key=lambda e: float(np.sum((np.asarray(e["xyz"]) - xyz) ** 2)))
        points[label] = dict(e, color=mk.color_id)
    left_dets = {f"{mk.color_id}_{i}": d for i, d in enumerate(L)}
    right_dets = {f"{mk.color_id}_{i}": d for i, d in enumerate(R)}
    return left_dets, right_dets, points
