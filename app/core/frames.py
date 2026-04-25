"""Coordinate frame transformations.

Implements TEME->J2000, ECI<->ECEF, geodetic conversions for WGS-84.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# WGS-84 constants
WGS84_A = 6378.137  # equatorial radius (km)
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # eccentricity squared
WGS84_B = WGS84_A * (1.0 - WGS84_F)

J2000_JD = 2451545.0
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def precession_matrix(jd_tt: float) -> np.ndarray:
    """IAU-76 precession matrix from J2000 to date (P maps J2000 -> date frame).

    For TEME->J2000 we need the inverse, which is P.T.
    """
    T = (jd_tt - J2000_JD) / 36525.0
    zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T ** 3) * ARCSEC_TO_RAD
    z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T ** 3) * ARCSEC_TO_RAD
    theta = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T ** 3) * ARCSEC_TO_RAD
    # Rotation: P = Rz(-z) * Ry(theta) * Rz(-zeta) maps mean-of-J2000 to mean-of-date
    return _rz(-z) @ _ry(theta) @ _rz(-zeta)


def teme_to_j2000(
    r_teme: np.ndarray, v_teme: np.ndarray, jd_tt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert TEME (SGP4 output) to J2000/ECI.

    Skips nutation and equation-of-equinoxes terms (sub-arcsecond effect),
    which is acceptable for a 17 km FOV at 500 km altitude.
    """
    # P_J2000_to_date = precession; inverse is its transpose
    P = precession_matrix(jd_tt)
    R = P.T  # date -> J2000
    return R @ np.asarray(r_teme), R @ np.asarray(v_teme)


def gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time in radians, IAU-82 model."""
    T = (jd_ut1 - J2000_JD) / 36525.0
    # GMST in seconds of time
    gmst_s = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T ** 3
    )
    gmst_s = gmst_s % 86400.0
    if gmst_s < 0:
        gmst_s += 86400.0
    return (gmst_s / 86400.0) * 2.0 * math.pi


def eci_to_ecef(r_eci: np.ndarray, jd_ut1: float) -> np.ndarray:
    theta = gmst_rad(jd_ut1)
    return _rz(theta) @ np.asarray(r_eci)


def ecef_to_eci(r_ecef: np.ndarray, jd_ut1: float) -> np.ndarray:
    theta = gmst_rad(jd_ut1)
    return _rz(-theta) @ np.asarray(r_ecef)


def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """ECEF (km) -> (lat_rad, lon_rad, alt_km) using Bowring's method."""
    x, y, z = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    a, b = WGS84_A, WGS84_B
    e2 = WGS84_E2
    ep2 = (a * a - b * b) / (b * b)

    p = math.hypot(x, y)
    if p < 1e-9:
        lat = math.copysign(math.pi / 2, z)
        lon = 0.0
        alt = abs(z) - b
        return lat, lon, alt

    theta = math.atan2(z * a, p * b)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    lat = math.atan2(z + ep2 * b * sin_t ** 3, p - e2 * a * cos_t ** 3)
    lon = math.atan2(y, x)
    N = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - N
    return lat, lon, alt


def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_km: float) -> np.ndarray:
    a, e2 = WGS84_A, WGS84_E2
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + alt_km) * cos_lat * math.cos(lon_rad)
    y = (N + alt_km) * cos_lat * math.sin(lon_rad)
    z = (N * (1.0 - e2) + alt_km) * sin_lat
    return np.array([x, y, z])


def geodetic_to_eci(
    lat_rad: float, lon_rad: float, alt_km: float, jd_ut1: float
) -> np.ndarray:
    return ecef_to_eci(geodetic_to_ecef(lat_rad, lon_rad, alt_km), jd_ut1)


def eci_to_geodetic(r_eci: np.ndarray, jd_ut1: float) -> Tuple[float, float, float]:
    return ecef_to_geodetic(eci_to_ecef(r_eci, jd_ut1))


def ray_ellipsoid_intersect(
    origin: np.ndarray, direction: np.ndarray
) -> np.ndarray | None:
    """Intersect ray (origin + t*direction) with WGS-84 ellipsoid (in ECI/ECEF-shape).

    Returns the near-side intersection point in the same frame as origin, or None.
    Note: the ellipsoid is symmetric about its principal axes, so this works
    in any frame whose axes align with the WGS-84 ellipsoid (i.e., ECEF).
    For ECI inputs the result is approximate — small error from polar flattening.
    """
    a = WGS84_A
    b = WGS84_B
    # Scale to a sphere
    s = np.array([1.0 / a, 1.0 / a, 1.0 / b])
    o = origin * s
    d = direction * s
    A = float(np.dot(d, d))
    B = 2.0 * float(np.dot(o, d))
    C = float(np.dot(o, o)) - 1.0
    disc = B * B - 4.0 * A * C
    if disc < 0 or A == 0:
        return None
    sq = math.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)
    # Take the smallest positive root
    t = min(x for x in (t1, t2) if x > 0) if (t1 > 0 or t2 > 0) else None
    if t is None:
        return None
    return origin + t * direction
