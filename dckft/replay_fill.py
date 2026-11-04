"""Continuous replay estimates, kept separate from optically supported speed metrics.

The wrist IMU predicts orientation in the fusion engine. Here its corrected world
acceleration bridges missing translation. A cubic endpoint correction joins the next
supported camera segment without the midpoint reset used by the conservative track.
This is an offline, camera-constrained estimate, not recovered camera measurements.
"""
from dataclasses import dataclass

import numpy as np

from .trajectory import Trajectory


@dataclass
class ReplayFill:
    traj: Trajectory
    estimated: np.ndarray
    nearest_support_s: np.ndarray
    accel_used: np.ndarray


def build_replay(traj, acc_world=None, acc_valid=None):
    """Fill each unsupported run at existing IMU times; never change the metric track.

    Bounded gaps use IMU integration plus endpoint position/velocity constraints.
    Unbounded edges dead-reckon from the nearest supported sample. Invalid acceleration
    coasts at constant velocity. Wrist flex and sensor drift limit accuracy throughout.
    """
    t = np.asarray(traj.t, float)
    good = (np.ones(len(t), bool) if traj.valid is None else traj.valid.copy())
    good &= np.isfinite(traj.knob).all(axis=1) & np.isfinite(traj.knob_vel).all(axis=1)
    anchors = np.flatnonzero(good)
    if not len(anchors):
        raise ValueError("No supported camera pose to anchor the IMU replay")
    a = np.zeros_like(traj.knob)
    usable = np.zeros(len(t), bool)
    if acc_world is not None:
        usable = np.isfinite(acc_world).all(axis=1)
        if acc_valid is not None:
            usable &= np.asarray(acc_valid, bool)
        a[usable] = np.asarray(acc_world)[usable]
    p, v = traj.knob.copy(), traj.knob_vel.copy()

    def propagate(indices, anchor):
        pp = np.empty((len(indices), 3)); vv = np.empty_like(pp)
        pk, vk, prev = p[anchor].copy(), v[anchor].copy(), anchor
        for j, k in enumerate(indices):
            dt = t[k]-t[prev]
            aa = .5*(a[k]+a[prev]) if usable[k] and usable[prev] else np.zeros(3)
            pk = pk + vk*dt + .5*aa*dt*dt
            vk = vk + aa*dt
            pp[j], vv[j], prev = pk, vk, k
        return pp, vv

    for left, right in zip(anchors[:-1], anchors[1:]):
        if right == left+1:
            continue
        ix = np.arange(left+1, right+1)
        pp, vv = propagate(ix, left)
        duration = t[right]-t[left]
        u = ((t[ix]-t[left])/duration)[:, None]
        dp, dv = p[right]-pp[-1], v[right]-vv[-1]
        # Zero correction and derivative at the left edge; match both at the right.
        pp += (3*u*u-2*u*u*u)*dp + (u*u*u-u*u)*duration*dv
        vv += (6*u-6*u*u)/duration*dp + (3*u*u-2*u)*dv
        p[ix[:-1]], v[ix[:-1]] = pp[:-1], vv[:-1]
    for ix, anchor in [(np.arange(anchors[0]-1, -1, -1), anchors[0]),
                       (np.arange(anchors[-1]+1, len(t)), anchors[-1])]:
        if len(ix):
            p[ix], v[ix] = propagate(ix, anchor)
    j = np.searchsorted(t[anchors], t)
    left = t[anchors[np.clip(j-1, 0, len(anchors)-1)]]
    right = t[anchors[np.clip(j, 0, len(anchors)-1)]]
    nearest = np.minimum(abs(t-left), abs(t-right))
    replay = Trajectory(t.copy(), p, v, traj.axis.copy(), traj.axis_rate.copy(), good.copy())
    return ReplayFill(replay, ~good, nearest, usable & ~good)
