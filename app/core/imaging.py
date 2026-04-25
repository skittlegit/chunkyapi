"""FOV / footprint projection onto WGS-84 ellipsoid."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..config import WGS84_A_KM, WGS84_B_KM
from .attitude import quat_to_dcm
from .frames import ecef_to_geodetic, eci_to_ecef


def ray_ellipsoid_intersection(
    origin: np.ndarray, direction: np.ndarray
) -> np.ndarray | None:
    """Intersect a ray (origin + t*direction) with the WGS-84 ellipsoid.

    Returns the *near* (smallest positive t) intersection point in the same
    frame as `origin`, or None if no intersection.
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    a = WGS84_A_KM
    b = WGS84_B_KM
    inv = np.array([1.0 / a, 1.0 / a, 1.0 / b])
    ox, oy, oz = o * inv
    dx, dy, dz = d * inv
    A = dx * dx + dy * dy + dz * dz
    B = 2.0 * (ox * dx + oy * dy + oz * dz)
    C = ox * ox + oy * oy + oz * oz - 1.0
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t1 = (-B - sq) / (2.0 * A)
    t2 = (-B + sq) / (2.0 * A)
    t = t1 if t1 > 1e-9 else t2
    if t <= 1e-9:
        return None
    return o + t * d


def project_boresight(
    r_sat_eci: np.ndarray, q_BN: np.ndarray, jd_ut1: float
) -> Tuple[float, float] | None:
    """Project +Z_body to the ground. Returns (lat_deg, lon_deg)."""
    R_BN = quat_to_dcm(q_BN)
    z_inertial = R_BN[:, 2]
    hit = ray_ellipsoid_intersection(np.asarray(r_sat_eci, dtype=float), z_inertial)
    if hit is None:
        return None
    ecef = eci_to_ecef(hit, jd_ut1)
    lat, lon, _ = ecef_to_geodetic(ecef)
    return math.degrees(lat), math.degrees(lon)


def compute_footprint(
    r_sat_eci: np.ndarray,
    q_BN: np.ndarray,
    jd_ut1: float,
    fov_deg: float = 2.0,
) -> List[Tuple[float, float]]:
    """Return the 4 corner (lat_deg, lon_deg) of the FOV footprint.

    Order: TL, TR, BR, BL in body coordinates (cross-track x along-track).
    """
    R_BN = quat_to_dcm(q_BN)
    half = math.radians(fov_deg / 2.0)
    # Corners in body: combine tilts about X (along-track) and Y (cross-track)
    signs = [(+1, +1), (+1, -1), (-1, -1), (-1, +1)]
    out: List[Tuple[float, float]] = []
    r_sat = np.asarray(r_sat_eci, dtype=float)
    for sx, sy in signs:
        # tilt vector in body frame
        bx = math.tan(sx * half)
        by = math.tan(sy * half)
        bz = 1.0
        v_body = np.array([bx, by, bz])
        v_body /= np.linalg.norm(v_body)
        v_eci = R_BN @ v_body
        hit = ray_ellipsoid_intersection(r_sat, v_eci)
        if hit is None:
            return []
        ecef = eci_to_ecef(hit, jd_ut1)
        lat, lon, _ = ecef_to_geodetic(ecef)
        out.append((math.degrees(lat), math.degrees(lon)))
    return out


# --- Polygon area + clipping helpers (planar approximation) --------------

def _equirect_xy(lat_lon: List[Tuple[float, float]], lat0: float) -> np.ndarray:
    """Project (lat, lon) deg list to local x,y km using equirectangular."""
    arr = np.asarray(lat_lon, dtype=float)
    coslat0 = math.cos(math.radians(lat0))
    x = (arr[:, 1]) * (math.pi / 180.0) * 6371.0 * coslat0
    y = (arr[:, 0]) * (math.pi / 180.0) * 6371.0
    return np.column_stack([x, y])


def polygon_area_latlon_km2(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    lat0 = sum(p[0] for p in poly) / len(poly)
    xy = _equirect_xy(poly, lat0)
    x = xy[:, 0]
    y = xy[:, 1]
    return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def footprint_area_km2(corners_latlon: List[Tuple[float, float]]) -> float:
    return polygon_area_latlon_km2(corners_latlon)


def sutherland_hodgman_clip(
    subject: List[Tuple[float, float]], clip: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Clip `subject` polygon against convex `clip` polygon. Both are (x,y)."""
    output = list(subject)
    if not output:
        return []
    cp1 = clip[-1]
    for cp2 in clip:
        if not output:
            break
        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    output.append(_intersect(s, e, cp1, cp2))
                output.append(e)
            elif _inside(s, cp1, cp2):
                output.append(_intersect(s, e, cp1, cp2))
            s = e
        cp1 = cp2
    return output


def _inside(p, cp1, cp2) -> bool:
    return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= 0.0


def _intersect(s, e, cp1, cp2):
    dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
    dp = (s[0] - e[0], s[1] - e[1])
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-12:
        return e
    n3 = 1.0 / denom
    return (
        (n1 * dp[0] - n2 * dc[0]) * n3,
        (n1 * dp[1] - n2 * dc[1]) * n3,
    )
