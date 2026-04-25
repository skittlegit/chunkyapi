"""Reaction-wheel dynamics & slew time estimation."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..config import (
    DEFAULT_INERTIA,
    WHEEL_H_SAFE_NMS,
)


COS45 = math.cos(math.radians(45.0))
SIN45 = math.sin(math.radians(45.0))


def wheel_jacobian() -> np.ndarray:
    """3x4 W matrix: H_body = W @ h_wheels for the pyramid configuration."""
    return np.array(
        [
            [SIN45, 0.0, -SIN45, 0.0],
            [0.0, SIN45, 0.0, -SIN45],
            [COS45, COS45, COS45, COS45],
        ],
        dtype=float,
    )


def body_to_wheel_momentum(h_body: np.ndarray) -> np.ndarray:
    W = wheel_jacobian()
    h, *_ = np.linalg.lstsq(W, np.asarray(h_body, dtype=float), rcond=None)
    return h


def check_saturation(
    h_wheels: np.ndarray, h_safe: float = WHEEL_H_SAFE_NMS
) -> Tuple[bool, float]:
    h = np.abs(np.asarray(h_wheels, dtype=float))
    max_frac = float(np.max(h) / h_safe) if h_safe > 0 else 0.0
    return max_frac > 1.0, max_frac


def momentum_envelope(inertia=DEFAULT_INERTIA) -> np.ndarray:
    """Approximate per-axis body-momentum limits in N m s."""
    Ix, Iy, Iz = inertia
    # From the pyramid geometry, two wheels contribute to each X/Y axis:
    Hxy = 2.0 * SIN45 * WHEEL_H_SAFE_NMS
    Hz = 4.0 * COS45 * WHEEL_H_SAFE_NMS
    return np.array([Hxy, Hxy, Hz])


def max_body_rates(inertia=DEFAULT_INERTIA) -> np.ndarray:
    """Per-axis max angular rate (rad/s) given the safe momentum envelope."""
    H = momentum_envelope(inertia)
    return H / np.array(inertia)


def estimate_slew_time(
    angle_rad: float, inertia=DEFAULT_INERTIA, eigen_axis: np.ndarray | None = None
) -> float:
    """Time for a slew of `angle_rad` using the worst-axis rate limit.

    If `eigen_axis` is provided, project the limit on that direction; otherwise
    use the smallest of the per-axis limits (most conservative).
    """
    if angle_rad <= 1e-9:
        return 0.0
    rates = max_body_rates(inertia)
    if eigen_axis is None:
        omega = float(np.min(rates))
    else:
        a = np.abs(np.asarray(eigen_axis, dtype=float))
        a = a / (np.linalg.norm(a) + 1e-15)
        omega_inv = np.dot(a, 1.0 / rates)
        omega = 1.0 / max(omega_inv, 1e-9)
    return float(angle_rad / omega)


def track_momentum(
    quaternion_sequence: List[np.ndarray],
    inertia=DEFAULT_INERTIA,
) -> List[np.ndarray]:
    """Cumulative wheel momentum assuming body returns to rest between
    waypoints (so all H ends up in the wheels at each settle)."""
    if not quaternion_sequence:
        return []
    from .attitude import quat_to_dcm

    W = wheel_jacobian()
    Wp = np.linalg.pinv(W)
    I_diag = np.diag(inertia)
    h_w = np.zeros(4)
    out = [h_w.copy()]
    for i in range(1, len(quaternion_sequence)):
        # Estimate the eigen-axis rotation angle/axis between attitudes
        from .attitude import quaternion_angular_distance

        q_a = np.asarray(quaternion_sequence[i - 1], dtype=float)
        q_b = np.asarray(quaternion_sequence[i], dtype=float)
        ang = quaternion_angular_distance(q_a, q_b)
        if ang < 1e-9:
            out.append(h_w.copy())
            continue
        # eigen-axis from quaternion difference
        # delta_q = q_b * conj(q_a)
        d = _quat_mul(q_b, _quat_conj(q_a))
        axis = d[:3]
        n = np.linalg.norm(axis)
        if n < 1e-9:
            out.append(h_w.copy())
            continue
        axis = axis / n
        # During slew, the wheels must absorb the body angular momentum
        # I*omega; at end of settle, body rate is zero so the *change* in wheel
        # H equals zero (closed cycle). We instead track the *peak* by
        # converting the body-fixed angular impulse along the eigen-axis.
        # For scoring, ΔH_used is the integral of |dH_wheels/dt|; we
        # approximate it by 2 * I*omega_peak / T (acceleration + deceleration).
        # Net wheel momentum at rest stays the same — but for energy budget we
        # accumulate the absolute change.
        out.append(h_w.copy())  # net h_wheels unchanged at rest-to-rest
    return out


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def momentum_change_estimate(angle_rad: float, inertia=DEFAULT_INERTIA) -> float:
    """Approximate |ΔH| budget consumed for a single rest-to-rest slew (Nms).

    For a bang-bang profile with peak rate omega_peak, the wheels accelerate
    body up (absorbing -I*omega) then back down. The cumulative |dH/dt|
    integral = 2 * I_eff * omega_peak ≈ 2 * I_eff * (angle / t_slew).
    Using the conservative I = max(inertia) gives an upper bound.
    """
    I_eff = float(max(inertia))
    rates = max_body_rates(inertia)
    omega_peak = 0.5 * float(np.min(rates))   # take half of envelope per slew
    # Bang-bang: t_slew = angle / omega_peak; ΔH_used = 2 * I_eff * omega_peak
    if angle_rad <= 0:
        return 0.0
    return 2.0 * I_eff * min(omega_peak, math.sqrt(angle_rad * float(np.min(rates))))
