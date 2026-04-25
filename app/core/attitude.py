"""Attitude (quaternion) math and target-pointing.

Quaternion convention: scalar-last [qx, qy, qz, qw]; q rotates Body -> Inertial.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v
    return v / n


def dcm_to_quat(R_BN: np.ndarray) -> np.ndarray:
    """Body->Inertial DCM (columns are body axes in inertial) -> quaternion xyzw."""
    return R.from_matrix(R_BN).as_quat().astype(float)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    return R.from_quat(np.asarray(q, dtype=float)).as_matrix()


def compute_pointing_quat(
    r_sat_eci: np.ndarray,
    v_sat_eci: np.ndarray,
    target_eci: np.ndarray,
) -> np.ndarray:
    """Quaternion (xyzw) so that +Z_body in inertial = look(sat -> target)."""
    r_sat = np.asarray(r_sat_eci, dtype=float)
    v_sat = np.asarray(v_sat_eci, dtype=float)
    target = np.asarray(target_eci, dtype=float)

    look = normalize(target - r_sat)
    z_B = look
    cross_vz = np.cross(v_sat, z_B)
    norm = np.linalg.norm(cross_vz)
    if norm < 1e-9:
        # Degenerate: pick any vector perpendicular to z_B
        helper = np.array([1.0, 0.0, 0.0]) if abs(z_B[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_B = normalize(np.cross(helper, z_B))
    else:
        x_B = cross_vz / norm
    y_B = np.cross(z_B, x_B)
    R_BN = np.column_stack([x_B, y_B, z_B])
    return dcm_to_quat(R_BN)


def compute_off_nadir(r_sat_eci: np.ndarray, target_eci: np.ndarray) -> float:
    """Angle in degrees between nadir and (sat->target)."""
    r_sat = np.asarray(r_sat_eci, dtype=float)
    target = np.asarray(target_eci, dtype=float)
    nadir = -r_sat / np.linalg.norm(r_sat)
    look = normalize(target - r_sat)
    c = float(np.clip(np.dot(nadir, look), -1.0, 1.0))
    return math.degrees(math.acos(c))


def nadir_quat(r_sat_eci: np.ndarray, v_sat_eci: np.ndarray) -> np.ndarray:
    """Quaternion pointing +Z body at nadir, +X cross-track."""
    r_sat = np.asarray(r_sat_eci, dtype=float)
    v_sat = np.asarray(v_sat_eci, dtype=float)
    z_B = -r_sat / np.linalg.norm(r_sat)
    cross_vz = np.cross(v_sat, z_B)
    n = np.linalg.norm(cross_vz)
    if n < 1e-9:
        helper = np.array([1.0, 0.0, 0.0])
        x_B = normalize(np.cross(helper, z_B))
    else:
        x_B = cross_vz / n
    y_B = np.cross(z_B, x_B)
    R_BN = np.column_stack([x_B, y_B, z_B])
    return dcm_to_quat(R_BN)


# --- SLERP ----------------------------------------------------------------

def _ensure_short_path(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if np.dot(q1, q2) < 0.0:
        return -q2
    return q2


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q2 = _ensure_short_path(q1, q2)
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_t0 = math.sin(theta_0)
    s1 = math.sin(theta_0 - theta) / sin_t0
    s2 = math.sin(theta) / sin_t0
    return s1 * q1 + s2 * q2


def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle in radians between two attitudes."""
    d = abs(float(np.dot(np.asarray(q1), np.asarray(q2))))
    d = min(1.0, max(-1.0, d))
    return 2.0 * math.acos(d)


def estimate_body_rate(q1: np.ndarray, q2: np.ndarray, dt: float) -> float:
    """|omega| in deg/s between consecutive attitudes."""
    if dt <= 0:
        return 0.0
    return math.degrees(quaternion_angular_distance(q1, q2)) / dt


def generate_attitude_trajectory(
    waypoints: Sequence[Tuple[float, np.ndarray]],
    dt: float = 0.020,
    hold_dt: float = 0.100,
    hold_intervals: Sequence[Tuple[float, float]] = (),
) -> List[Tuple[float, np.ndarray]]:
    """Generate a smooth quaternion timeline from (t, q) waypoints.

    Between consecutive waypoints uses SLERP at `dt`. During `hold_intervals`
    (start_t, end_t) the quaternion is held constant at the most recent
    waypoint and sampled at `hold_dt`.

    Sign-flip-free: each emitted quaternion is brought to the short-arc side of
    the previous one.
    """
    if len(waypoints) == 0:
        return []

    holds = sorted([(float(a), float(b)) for a, b in hold_intervals])

    def in_hold(t: float) -> bool:
        for a, b in holds:
            if a - 1e-9 <= t <= b + 1e-9:
                return True
        return False

    out: List[Tuple[float, np.ndarray]] = []

    def push(t: float, q: np.ndarray):
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        if out and np.dot(out[-1][1], q) < 0.0:
            q = -q
        out.append((t, q))

    # Always emit first waypoint
    t0, q0 = waypoints[0]
    push(t0, q0)

    for i in range(len(waypoints) - 1):
        t_a, q_a = waypoints[i]
        t_b, q_b = waypoints[i + 1]
        seg = max(0.0, t_b - t_a)
        if seg < 1e-9:
            push(t_b, q_b)
            continue
        # Choose dt depending on whether segment is a "hold" or a slew
        is_hold = abs(quaternion_angular_distance(q_a, q_b)) < 1e-6 and in_hold(t_a)
        step = hold_dt if is_hold else dt
        # Use floor so the resulting per-sample dt is >= `step` (>= 20ms).
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            frac = k / n
            t_k = t_a + frac * seg
            q_k = slerp(q_a, q_b, frac)
            push(t_k, q_k)
    return out
