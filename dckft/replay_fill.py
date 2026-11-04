"""Continuous replay estimates, kept separate from optically supported speed metrics.

The wrist IMU predicts orientation in the fusion engine. Here its corrected world
acceleration bridges missing translation. A cubic endpoint correction joins the next
supported camera segment without the midpoint reset used by the conservative track.
This is an offline, camera-constrained estimate, not recovered camera measurements.
"""
from dataclasses import dataclass

import numpy as np

from .trajectory import Trajectory
from .quality import optical_support
from . import quat as Q


@dataclass
class ReplayFill:
    traj: Trajectory
    estimated: np.ndarray
    nearest_support_s: np.ndarray
    accel_used: np.ndarray
    pose_supported: np.ndarray = None


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
    return ReplayFill(replay, ~good, nearest, usable & ~good, good.copy())


def camera_guided_replay(traj, t_camera, knob_camera, axis_camera,
                         acc_world=None, acc_valid=None, max_gap_s=.06):
    """Camera-first display poses, with IMU motion only across missing optical intervals.

    Use exact camera timestamps internally so optical anchors are not rounded to an IMU
    sample. Return at the original IMU times. Metric validity stays entirely independent.
    The bat line is fitted to the cleaned marker positions, not to the wrist prediction.
    """
    t = np.asarray(traj.t)
    tc = np.asarray(t_camera)
    ok = (np.isfinite(knob_camera).all(axis=1) & np.isfinite(axis_camera).all(axis=1)
          & (tc >= t[0]) & (tc <= t[-1]))
    tc, pc, uc = tc[ok], np.asarray(knob_camera)[ok], np.asarray(axis_camera)[ok]
    if not len(tc):
        return None
    tc, unique = np.unique(tc, return_index=True)
    pc, uc = pc[unique], uc[unique]
    uc = uc / np.linalg.norm(uc, axis=1, keepdims=True)
    joint = np.unique(np.r_[t, tc])
    interp = lambda times, values, out: np.column_stack(
        [np.interp(out, times, values[:, j]) for j in range(3)])
    pos = interp(t, traj.knob, joint)
    vel = interp(t, traj.knob_vel, joint)
    fused_axis = interp(t, traj.axis, joint)
    fused_axis /= np.linalg.norm(fused_axis, axis=1, keepdims=True)
    camera_idx = np.searchsorted(joint, tc)
    # Apply a smoothly changing correction to the IMU axis, matching both optical ends
    # of each gap. No claim is made that wrist motion observes hidden bat flex.
    rv = np.array([Q.to_rotvec(Q.from_two_vectors(a, b))
                   for a, b in zip(fused_axis[camera_idx], uc)])
    axis = Q.rotate(Q.from_rotvec(interp(tc, rv, joint)), fused_axis)
    pos[camera_idx] = pc
    for i, k in enumerate(camera_idx):
        # Local optical velocity for the gap boundary; never fit across an occlusion.
        lo = i
        while lo > 0 and tc[lo]-tc[lo-1] <= max_gap_s and tc[i]-tc[lo-1] <= .04:
            lo -= 1
        hi = i+1
        while hi < len(tc) and tc[hi]-tc[hi-1] <= max_gap_s and tc[hi]-tc[i] <= .04:
            hi += 1
        if hi-lo >= 2:
            dt = tc[lo:hi]-tc[i]
            vel[k] = np.linalg.lstsq(np.column_stack([np.ones(len(dt)), dt]),
                                     pc[lo:hi], rcond=None)[0][1]
    anchor_mask = np.zeros(len(joint), bool)
    anchor_mask[camera_idx] = True
    anchors = Trajectory(joint, pos, vel, axis, np.zeros_like(axis), anchor_mask)
    acceleration = None if acc_world is None else interp(t, acc_world, joint)
    acceleration_ok = None
    if acc_valid is not None:
        acceleration_ok = np.interp(joint, t, np.asarray(acc_valid, float)) >= .999
    filled = build_replay(anchors, acceleration, acceleration_ok)
    pos, vel = filled.traj.knob, filled.traj.knob_vel
    for i in range(len(tc)-1):
        if tc[i+1]-tc[i] > max_gap_s:
            continue
        inside = (joint >= tc[i]) & (joint <= tc[i+1])
        f = ((joint[inside]-tc[i])/(tc[i+1]-tc[i]))[:, None]
        pos[inside] = (1-f)*pc[i]+f*pc[i+1]
        vel[inside] = (pc[i+1]-pc[i])/(tc[i+1]-tc[i])
        axis[inside] = Q.rotate(Q.from_rotvec(f*Q.to_rotvec(Q.from_two_vectors(uc[i],uc[i+1]))), uc[i])
    at = np.searchsorted(joint, t)
    axis = axis[at]
    support = optical_support(t, tc, max_gap_s, guard_s=0)
    j = np.searchsorted(tc, t)
    nearest = np.minimum(abs(t-tc[np.clip(j-1,0,len(tc)-1)]),
                         abs(t-tc[np.clip(j,0,len(tc)-1)]))
    # An isolated optical anchor still constrains its nearest IMU sample. Do not erase
    # it with the 40 ms guard used exclusively for speed reliability.
    support |= nearest <= .5*np.median(np.diff(t))+1e-8
    metric_valid = np.ones(len(t), bool) if traj.valid is None else traj.valid.copy()
    replay = Trajectory(t.copy(),pos[at],vel[at],axis,np.gradient(axis,t,axis=0),metric_valid)
    return ReplayFill(replay,~support,nearest,filled.accel_used[at] & ~support,support)
