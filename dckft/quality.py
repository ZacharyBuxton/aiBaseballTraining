"""Optical support checks. Missing observations are uncertainty, not measured motion."""
import numpy as np

from .data import CameraData


def clean_camera(cam, marker_s):
    """Reject inconsistent rigid geometry and isolated position spikes; retain raw files."""
    pts = {m: p.copy() for m, p in cam.points.items()}
    ids = [m for m in marker_s if m in pts]
    bad = np.zeros(len(cam.t), bool)
    for a, mid in enumerate(ids):
        for other in ids[a + 1:]:
            distance = np.linalg.norm(pts[mid] - pts[other], axis=1)
            expected = abs(marker_s[mid] - marker_s[other])
            bad |= np.isfinite(distance) & (abs(distance - expected) > max(.05, .2 * expected))
    # A single erroneous stereo fix should not become a derivative spike. Test a point
    # against its two close neighbours, without enforcing a maximum bat speed.
    for p in pts.values():
        for i in range(1, len(cam.t) - 1):
            dt = cam.t[i + 1] - cam.t[i - 1]
            if 0 < dt <= .05 and np.isfinite(p[i - 1:i + 2]).all():
                f = (cam.t[i] - cam.t[i - 1]) / dt
                residual = np.linalg.norm(p[i] - ((1-f)*p[i-1] + f*p[i+1]))
                if residual > .15:
                    bad[i] = True
    for p in pts.values():
        p[bad] = np.nan
    return CameraData(cam.t.copy(), pts, cam.gravity_cam, cam.meta), int(bad.sum())


def optical_support(t, t_camera, max_gap_s=.06, guard_s=.04):
    """True only inside short, bracketed camera intervals, away from long-gap edges."""
    t = np.asarray(t)
    tc = np.unique(np.asarray(t_camera))
    valid = np.zeros(len(t), bool)
    if len(tc) < 2:
        return valid
    for left, right in zip(tc[:-1], tc[1:]):
        if right-left <= max_gap_s:
            valid |= (t >= left) & (t <= right)
    for left, right in zip(tc[:-1], tc[1:]):
        if right-left > max_gap_s:
            valid[(t >= left-guard_s) & (t <= right+guard_s)] = False
    return valid
