"""FOV projection and ground footprint computation."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .attitude import quat_to_dcm
from .frames import eci_to_geodetic, ray_ellipsoid_intersect, eci_to_ecef


def _project_ray_to_ground(
    r_sat_eci: np.ndarray, dir_eci: np.ndarray, jd_ut1: float
) -> Tuple[float, float] | None:
    """Project a ray from r_sat in direction dir to the WGS-84 surface.

    Returns (lat_deg, lon_deg) or None if no intersection.
    Uses ECEF for accurate ellipsoid intersection.
    """
    r_ecef = eci_to_ecef(r_sat_eci, jd_ut1)
    # Approx: rotate direction by GMST too. But over short window, the
    # small instantaneous rotation of the direction vector is what matters.
    # Easier: convert direction to ECEF using same rotation.
    d_ecef = eci_to_ecef(dir_eci, jd_ut1)
    hit = ray_ellipsoid_intersect(r_ecef, d_ecef)
    if hit is None:
        return None
    # ECEF -> geodetic
    from .frames import ecef_to_geodetic

    lat, lon, _alt = ecef_to_geodetic(hit)
    return math.degrees(lat), math.degrees(lon)


def project_boresight(
    r_sat_eci: np.ndarray, q_BN: np.ndarray, jd_ut1: float
) -> Tuple[float, float] | None:
    """Where +Z_body hits the Earth surface."""
    dcm = quat_to_dcm(q_BN)
    z_b_eci = dcm[:, 2]
    return _project_ray_to_ground(r_sat_eci, z_b_eci, jd_ut1)


def compute_footprint(
    r_sat_eci: np.ndarray,
    q_BN: np.ndarray,
    fov_deg: float,
    jd_ut1: float,
) -> List[Tuple[float, float]]:
    """Project the four FOV corners (±fov/2 in body X and Y) to ground.

    Returns a list of (lat_deg, lon_deg) tuples in CCW-ish order; entries
    are dropped if they don't intersect the Earth.
    """
    dcm = quat_to_dcm(q_BN)
    half = math.radians(fov_deg / 2.0)
    tan_h = math.tan(half)
    # Corner directions in body frame: (+/-tan_h, +/-tan_h, 1) normalized
    corners_body = [
        np.array([+tan_h, +tan_h, 1.0]),
        np.array([-tan_h, +tan_h, 1.0]),
        np.array([-tan_h, -tan_h, 1.0]),
        np.array([+tan_h, -tan_h, 1.0]),
    ]
    pts: List[Tuple[float, float]] = []
    for cb in corners_body:
        d_eci = dcm @ cb
        d_eci = d_eci / np.linalg.norm(d_eci)
        hit = _project_ray_to_ground(r_sat_eci, d_eci, jd_ut1)
        if hit is not None:
            pts.append(hit)
    return pts


def footprint_area_km2(corners_latlon: List[Tuple[float, float]]) -> float:
    """Approximate area of small ground polygon using local equirectangular projection."""
    if len(corners_latlon) < 3:
        return 0.0
    lat0 = sum(p[0] for p in corners_latlon) / len(corners_latlon)
    cos_lat = math.cos(math.radians(lat0))
    KM_PER_DEG_LAT = 110.574
    KM_PER_DEG_LON = 111.320 * cos_lat
    xy = [
        ((lon - corners_latlon[0][1]) * KM_PER_DEG_LON, (lat - corners_latlon[0][0]) * KM_PER_DEG_LAT)
        for (lat, lon) in corners_latlon
    ]
    # Shoelace
    s = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5
