"""
quat.py -- quaternion helpers (scalar-first [w, x, y, z], Hamilton product).

Every function accepts a single quaternion of shape (4,) or a stack of shape (N, 4)
where that makes sense. Orientation quaternions q map body-frame vectors into the
world frame:  v_world = rotate(q, v_body).

No Euler angles anywhere in the fusion path (project constraint: avoid gimbal lock).
"""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n == 0, 1.0, n)


def conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p (x) q (broadcasts over leading dims)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pw, px, py, pz = np.moveaxis(p, -1, 0)
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    return np.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], axis=-1)


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) for unit quaternion(s)."""
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion from a single 3x3 rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = normalize(np.array(q))
    return q if q[0] >= 0 else -q


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q (body -> world)."""
    return np.einsum("...ij,...j->...i", to_matrix(q), np.asarray(v, dtype=float))


def from_rotvec(rv: np.ndarray) -> np.ndarray:
    """Quaternion exp map of a rotation vector (axis * angle, radians)."""
    rv = np.asarray(rv, dtype=float)
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * theta
    # sin(x/2)/x with a Taylor fallback near zero
    k = np.where(theta > 1e-9, np.sin(half) / np.where(theta > 1e-9, theta, 1.0), 0.5 - theta ** 2 / 48.0)
    return np.concatenate([np.cos(half), rv * k], axis=-1)


def to_rotvec(q: np.ndarray) -> np.ndarray:
    """Quaternion log map -> rotation vector (shortest rotation)."""
    q = normalize(q)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    k = np.where(s > 1e-12, angle / np.where(s > 1e-12, s, 1.0), 2.0)
    return v * k


def angle_between(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between orientations p and q."""
    d = np.abs(np.sum(normalize(p) * normalize(q), axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = np.dot(a, b)
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    q = np.array([1.0 + c, *np.cross(a, b)])
    return normalize(q)


def integrate_gyro(q0: np.ndarray, omega: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
    Integrate body-frame angular velocity omega (N, 3) [rad/s] with step sizes dt (N,)
    starting from q0. Returns (N, 4); out[0] == q0 and out[k] uses omega[k-1] over dt[k].
    """
    n = len(omega)
    out = np.empty((n, 4))
    out[0] = normalize(q0)
    for k in range(1, n):
        w = 0.5 * (omega[k - 1] + omega[k])          # trapezoid on the rate
        out[k] = normalize(multiply(out[k - 1], from_rotvec(w * dt[k])))
    return out


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def horn(model_pts: np.ndarray, world_pts: np.ndarray):
    """
    Horn / Davenport q-method absolute orientation for N >= 3 non-collinear points.
    Returns (q, t) with world = rotate(q, model) + t.
    """
    B = np.asarray(model_pts, float)
    W = np.asarray(world_pts, float)
    cb, cw = B.mean(0), W.mean(0)
    M = (B - cb).T @ (W - cw)
    Sxx, Sxy, Sxz = M[0]
    Syx, Syy, Syz = M[1]
    Szx, Szy, Szz = M[2]
    N = np.array([
        [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
        [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
        [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
    ])
    vals, vecs = np.linalg.eigh(N)
    q = normalize(vecs[:, np.argmax(vals)])
    if q[0] < 0:
        q = -q
    t = cw - rotate(q, cb)
    return q, t
