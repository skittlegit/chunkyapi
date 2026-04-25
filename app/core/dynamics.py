"""Reaction wheel momentum & slew dynamics."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..config import settings


def wheel_jacobian() -> np.ndarray:
    """3x4 W matrix mapping wheel momentum -> body momentum.

    Pyramid of 4 wheels canted at 45°, evenly spaced in azimuth.
    """
    cant = math.radians(45.0)
    s = math.sin(cant)
    c = math.cos(cant)
    azimuths = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    cols = []
    for az in azimuths:
        cols.append([s * math.cos(az), s * math.sin(az), c])
    return np.array(cols).T  # shape (3,4)


W_MATRIX = wheel_jacobian()
W_PINV = np.linalg.pinv(W_MATRIX)


def body_to_wheel_momentum(h_body: np.ndarray) -> np.ndarray:
    return W_PINV @ np.asarray(h_body, dtype=float)


def wheel_to_body_momentum(h_wheels: np.ndarray) -> np.ndarray:
    return W_MATRIX @ np.asarray(h_wheels, dtype=float)


def check_saturation(
    h_wheels: np.ndarray, h_max: float | None = None
) -> Tuple[bool, float]:
    if h_max is None:
        h_max = settings.wheel_h_safe_nms
    max_abs = float(np.max(np.abs(h_wheels)))
    return max_abs > h_max, max_abs / h_max


def estimate_slew_time(angle_deg: float, axis: str = "xy") -> float:
    """Approximate bang-coast-bang slew time for a momentum-limited slew.

    Uses the smaller of the X/Y body rate limit (~20 deg/s) for arbitrary
    cross-track slews, or the Z limit for yaw.
    """
    if angle_deg <= 0.0:
        return 0.0
    # From momentum budget: omega_max = h_body_max / I
    Ix, Iy, Iz = settings.inertia_diag
    if axis == "z":
        omega_max = 84.9e-3 / Iz
    else:
        omega_max = 42.4e-3 / Ix
    omega_max_dps = math.degrees(omega_max)
    # Use 50% of max as a practical operational rate
    omega_op = 0.5 * omega_max_dps
    return angle_deg / omega_op


def momentum_change_for_slew(angle_rad: float, inertia: float = 0.12) -> float:
    """Peak body-frame momentum needed for a slew of given angle (very rough).

    Assumes a triangular rate profile peaking at omega_peak; momentum needed
    is I * omega_peak. For our purposes we model |delta_h| ≈ I * omega_peak,
    where omega_peak is chosen to complete the slew in the budgeted time.
    """
    return inertia * angle_rad
