"""Quaternion-based attitude utilities and pointing computations.

Quaternion convention: scalar-LAST [qx, qy, qz, qw], rotates body->inertial (q_BN).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def dcm_to_quat(dcm: np.ndarray) -> np.ndarray:
    """3x3 DCM (columns = body axes in inertial) -> quaternion [x,y,z,w]."""
    return R.from_matrix(dcm).as_quat()  # scipy returns scalar-last


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(q, dtype=float)).as_matrix()


def compute_pointing_quat(
    r_sat_eci: np.ndarray,
    v_sat_eci: np.ndarray,
    target_eci: np.ndarray,
) -> np.ndarray:
    """Compute q_BN such that +Z_body points from satellite to target.

    Uses velocity vector to disambiguate roll (cross-track aligned with orbit normal).
    """
    r_sat = np.asarray(r_sat_eci, dtype=float)
    v_sat = np.asarray(v_sat_eci, dtype=float)
    target = np.asarray(target_eci, dtype=float)

    z_b = _norm(target - r_sat)
    # x_b roughly along cross-track: v x z
    cross = np.cross(v_sat, z_b)
    if np.linalg.norm(cross) < 1e-9:
        # Fall back to using arbitrary perpendicular
        helper = np.array([1.0, 0.0, 0.0]) if abs(z_b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        cross = np.cross(helper, z_b)
    x_b = _norm(cross)
    y_b = np.cross(z_b, x_b)
    # Columns of DCM are body axes expressed in inertial
    dcm = np.column_stack([x_b, y_b, z_b])
    q = dcm_to_quat(dcm)
    return quat_normalize(q)


def compute_off_nadir(r_sat_eci: np.ndarray, target_eci: np.ndarray) -> float:
    r_sat = np.asarray(r_sat_eci, dtype=float)
    nadir = -_norm(r_sat)
    look = _norm(np.asarray(target_eci, dtype=float) - r_sat)
    cos_a = float(np.clip(np.dot(nadir, look), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.dot(q1, q2))


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest geodesic angle (radians) between two attitudes."""
    d = abs(quat_dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    return 2.0 * math.acos(d)


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation. Always takes the short path."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    d = float(np.dot(q1, q2))
    if d < 0.0:
        q2 = -q2
        d = -d
    if d > 0.9995:
        out = q1 + t * (q2 - q1)
        return quat_normalize(out)
    theta_0 = math.acos(min(1.0, max(-1.0, d)))
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    s1 = math.sin(theta_0 - theta) / sin_theta_0
    s2 = math.sin(theta) / sin_theta_0
    return quat_normalize(s1 * q1 + s2 * q2)


def generate_attitude_trajectory(
    waypoints: Sequence[Tuple[float, np.ndarray]],
    dt: float = 0.020,
) -> List[Tuple[float, np.ndarray]]:
    """SLERP between (t, q) waypoints at fixed dt sample spacing.

    Output includes one sample at each waypoint time and at dt-spaced grid in between.
    Sign-continuity is enforced (no 360° spins).
    """
    if not waypoints:
        return []
    out: List[Tuple[float, np.ndarray]] = []
    # Enforce continuity: ensure consecutive waypoint quats have positive dot
    fixed: List[Tuple[float, np.ndarray]] = []
    prev_q = None
    for t, q in waypoints:
        q = quat_normalize(np.asarray(q, dtype=float))
        if prev_q is not None and float(np.dot(prev_q, q)) < 0:
            q = -q
        fixed.append((float(t), q))
        prev_q = q

    for i in range(len(fixed) - 1):
        t0, q0 = fixed[i]
        t1, q1 = fixed[i + 1]
        seg = max(0.0, t1 - t0)
        n = max(1, int(round(seg / dt)))
        for k in range(n):
            t = t0 + k * dt
            if t > t1:
                t = t1
            u = 0.0 if seg == 0 else (t - t0) / seg
            q = slerp(q0, q1, u)
            out.append((t, q))
    # Append final waypoint
    out.append(fixed[-1])
    # Deduplicate by time (keep last)
    seen = {}
    for t, q in out:
        seen[round(t, 9)] = q
    out_sorted = [(t, seen[t]) for t in sorted(seen.keys())]
    return out_sorted


def estimate_body_rate(q1: np.ndarray, q2: np.ndarray, dt: float) -> float:
    """Approximate |omega| in deg/s from consecutive attitudes."""
    if dt <= 0:
        return 0.0
    ang = quaternion_angular_distance(q1, q2)
    return math.degrees(ang / dt)


def nadir_pointing_quat(r_sat_eci: np.ndarray, v_sat_eci: np.ndarray) -> np.ndarray:
    """Quaternion that points +Z_body at nadir, +X along velocity (roughly)."""
    r_sat = np.asarray(r_sat_eci, dtype=float)
    v_sat = np.asarray(v_sat_eci, dtype=float)
    z_b = _norm(-r_sat)
    cross = np.cross(v_sat, z_b)
    x_b = _norm(cross)
    y_b = np.cross(z_b, x_b)
    return quat_normalize(dcm_to_quat(np.column_stack([x_b, y_b, z_b])))
