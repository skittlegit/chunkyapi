"""Coordinate frame transformations.

Frames:
    TEME   — output of SGP4
    J2000  — Earth-centered inertial (ECI)
    ECEF   — Earth-centered Earth-fixed (WGS-84)
    Geodetic (lat, lon, alt)

All vectors in km, angles in radians unless suffixed `_deg`.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ..config import WGS84_A_KM, WGS84_E2, WGS84_B_KM


ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


# --- Basic rotation matrices ---------------------------------------------

def Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)


# --- TEME -> J2000 (precession only, IAU-76) -----------------------------

def precession_matrix(jd_tt: float) -> np.ndarray:
    """IAU-76 precession matrix from J2000 to date.

    Returns the matrix P such that r_date = P @ r_J2000.
    """
    T = (jd_tt - 2451545.0) / 36525.0
    zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T ** 3) * ARCSEC_TO_RAD
    theta = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T ** 3) * ARCSEC_TO_RAD
    z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T ** 3) * ARCSEC_TO_RAD
    # P = Rz(-z) Ry(theta) Rz(-zeta)
    return Rz(-z) @ Ry(theta) @ Rz(-zeta)


def teme_to_j2000(
    r_teme: np.ndarray, v_teme: np.ndarray, jd_tt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert TEME -> J2000 (ECI). Skips nutation (~0.01 deg error)."""
    P = precession_matrix(jd_tt)
    # P maps J2000 -> date; TEME is approximately the date frame, so inverse
    # transports TEME -> J2000.
    R = P.T
    return R @ np.asarray(r_teme, dtype=float), R @ np.asarray(v_teme, dtype=float)


# --- GMST and ECI <-> ECEF ----------------------------------------------

def gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU-82)."""
    T = (jd_ut1 - 2451545.0) / 36525.0
    # Seconds of time
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T ** 3
    )
    # Wrap to [0, 86400)
    gmst_sec = gmst_sec % 86400.0
    if gmst_sec < 0:
        gmst_sec += 86400.0
    return (gmst_sec / 86400.0) * 2.0 * math.pi


def eci_to_ecef(r_eci: np.ndarray, jd_ut1: float) -> np.ndarray:
    g = gmst_rad(jd_ut1)
    return Rz(g) @ np.asarray(r_eci, dtype=float)


def ecef_to_eci(r_ecef: np.ndarray, jd_ut1: float) -> np.ndarray:
    g = gmst_rad(jd_ut1)
    return Rz(-g) @ np.asarray(r_ecef, dtype=float)


# --- Geodetic <-> ECEF ---------------------------------------------------

def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_km: float) -> np.ndarray:
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    N = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_km) * cos_lat * math.cos(lon_rad)
    y = (N + alt_km) * cos_lat * math.sin(lon_rad)
    z = (N * (1.0 - WGS84_E2) + alt_km) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """Bowring's iterative method. Returns (lat_rad, lon_rad, alt_km)."""
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p < 1e-9:
        lat = math.copysign(math.pi / 2, z)
        alt = abs(z) - WGS84_B_KM
        return lat, lon, alt
    # Initial guess
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        N = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - N
        lat_new = math.atan2(z, p * (1.0 - WGS84_E2 * N / (N + alt)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = math.sin(lat)
    N = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return lat, lon, alt


def geodetic_to_eci(
    lat_rad: float, lon_rad: float, alt_km: float, jd_ut1: float
) -> np.ndarray:
    return ecef_to_eci(geodetic_to_ecef(lat_rad, lon_rad, alt_km), jd_ut1)


def eci_to_geodetic(r_eci: np.ndarray, jd_ut1: float) -> Tuple[float, float, float]:
    return ecef_to_geodetic(eci_to_ecef(r_eci, jd_ut1))
