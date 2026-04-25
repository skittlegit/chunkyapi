"""Imaging mission planner.

Boustrophedon raster scan with momentum-aware sequencing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from ..config import settings
from .attitude import (
    compute_off_nadir,
    compute_pointing_quat,
    generate_attitude_trajectory,
    nadir_pointing_quat,
    quaternion_angular_distance,
)
from .dynamics import (
    body_to_wheel_momentum,
    check_saturation,
    estimate_slew_time,
)
from .frames import geodetic_to_eci
from .imaging import compute_footprint, project_boresight
from .propagator import EphemerisPoint, propagate_pass
from .tiling import (
    TileCenter,
    adaptive_tile_size_km,
    boustrophedon_order,
    polygon_area_km2,
    tile_aoi,
)


LatLon = Tuple[float, float]


@dataclass
class TileAccess:
    tile: TileCenter
    best_t: float
    best_off_nadir: float
    best_idx: int  # index into ephemeris


@dataclass
class PlannedImage:
    tile_id: str
    t_image: float
    q_BN: np.ndarray
    off_nadir_deg: float
    footprint: List[LatLon] = field(default_factory=list)


@dataclass
class PlanResult:
    schedule: dict
    diagnostics: dict
    ephemeris_summary: dict


def _tile_eci(lat_deg: float, lon_deg: float, jd: float) -> np.ndarray:
    return geodetic_to_eci(math.radians(lat_deg), math.radians(lon_deg), 0.0, jd)


def find_closest_approach(
    ephemeris: List[EphemerisPoint], aoi_center_latlon: LatLon
) -> int:
    lat_c, lon_c = aoi_center_latlon
    best_i = 0
    best_d = float("inf")
    for i, ep in enumerate(ephemeris):
        target = _tile_eci(lat_c, lon_c, ep.jd)
        d = float(np.linalg.norm(target - ep.r_eci))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def evaluate_tile_access(
    tile: TileCenter,
    ephemeris: List[EphemerisPoint],
    off_nadir_max_deg: float,
) -> TileAccess | None:
    best_idx = -1
    best_off_nadir = float("inf")
    for i, ep in enumerate(ephemeris):
        target = _tile_eci(tile.lat_deg, tile.lon_deg, ep.jd)
        off = compute_off_nadir(ep.r_eci, target)
        if off < best_off_nadir:
            best_off_nadir = off
            best_idx = i
    if best_idx < 0 or best_off_nadir > off_nadir_max_deg:
        return None
    ep = ephemeris[best_idx]
    return TileAccess(tile=tile, best_t=ep.t_offset_s, best_off_nadir=best_off_nadir, best_idx=best_idx)


def _ephemeris_at_t(ephemeris: List[EphemerisPoint], t: float) -> EphemerisPoint:
    idx = max(0, min(len(ephemeris) - 1, int(round(t))))
    # If dt of ephemeris is 1.0 s this is exact-ish
    return ephemeris[idx]


def plan_imaging(
    tle1: str,
    tle2: str,
    aoi_polygon: Sequence[LatLon],
    pass_start_utc: str,
    pass_end_utc: str,
    settle_margin_s: float = 3.0,
    off_nadir_margin_deg: float = 5.0,
    strategy: str = "boustrophedon",
) -> PlanResult:
    fov = settings.fov_deg
    off_nadir_limit = settings.off_nadir_limit_deg - off_nadir_margin_deg
    integration = settings.integration_time_s

    # 1) Propagate
    ephemeris = propagate_pass(tle1, tle2, pass_start_utc, pass_end_utc, dt=1.0)
    if not ephemeris:
        raise ValueError("Empty ephemeris from propagator")

    pass_duration = ephemeris[-1].t_offset_s - ephemeris[0].t_offset_s

    # 2) Closest approach
    lat_c = sum(p[0] for p in aoi_polygon) / len(aoi_polygon)
    lon_c = sum(p[1] for p in aoi_polygon) / len(aoi_polygon)
    ca_idx = find_closest_approach(ephemeris, (lat_c, lon_c))
    ca_ep = ephemeris[ca_idx]
    target_ca = _tile_eci(lat_c, lon_c, ca_ep.jd)
    off_at_ca = compute_off_nadir(ca_ep.r_eci, target_ca)

    # 3) Tile the AOI based on expected off-nadir at CA
    tile_size_km = adaptive_tile_size_km(off_at_ca, fov, settings.altitude_km_nominal)
    tiles = tile_aoi(aoi_polygon, tile_size_km)

    # 4) Determine accessible tiles
    accesses: List[TileAccess] = []
    for tile in tiles:
        a = evaluate_tile_access(tile, ephemeris, off_nadir_limit)
        if a is not None:
            accesses.append(a)

    # 5) Order tiles boustrophedon
    accessible_tiles = [a.tile for a in accesses]
    if strategy == "boustrophedon":
        ordered_tiles = boustrophedon_order(accessible_tiles)
    elif strategy == "center_first":
        ordered_tiles = sorted(
            accessible_tiles,
            key=lambda t: (t.lat_deg - lat_c) ** 2 + (t.lon_deg - lon_c) ** 2,
        )
    else:
        # greedy: by best access time
        access_by_id = {a.tile.id: a for a in accesses}
        ordered_tiles = [access_by_id[t.id].tile for t in sorted(
            accessible_tiles,
            key=lambda t: access_by_id[t.id].best_t,
        )]
    access_by_id = {a.tile.id: a for a in accesses}

    # 6) Sequence: assign times, drop tiles that don't fit budget
    planned: List[PlannedImage] = []
    current_t = 5.0  # initial settle margin from start of pass
    current_q = nadir_pointing_quat(ephemeris[0].r_eci, ephemeris[0].v_eci)
    cumulative_dh = 0.0  # sum of |delta h_body| in N*m*s
    Ix = settings.inertia_diag[0]

    for tile in ordered_tiles:
        access = access_by_id[tile.id]
        # Schedule at max(current_t, access.best_t - small margin), clamped
        desired_t = max(current_t, access.best_t)
        if desired_t + integration > pass_duration - 5.0:
            # No more time
            break
        # Get ephemeris at the desired image time
        ep = _ephemeris_at_t(ephemeris, desired_t)
        target = _tile_eci(tile.lat_deg, tile.lon_deg, ep.jd)
        off_nadir = compute_off_nadir(ep.r_eci, target)
        if off_nadir > off_nadir_limit:
            continue
        q_target = compute_pointing_quat(ep.r_eci, ep.v_eci, target)

        # Slew time estimate
        slew_angle_rad = quaternion_angular_distance(current_q, q_target)
        slew_angle_deg = math.degrees(slew_angle_rad)
        t_slew = estimate_slew_time(slew_angle_deg)
        t_settle = settle_margin_s

        t_start_slew = current_t
        t_end_slew = t_start_slew + t_slew
        t_image_start = t_end_slew + t_settle
        t_image_end = t_image_start + integration

        if t_image_end > pass_duration - 2.0:
            break

        # Momentum bookkeeping (rough): peak body momentum during slew
        delta_h_body = Ix * slew_angle_rad / max(t_slew, 0.5)  # I*omega_peak
        # Apply both for accel and decel
        cumulative_dh += 2.0 * abs(delta_h_body)

        # Saturation check (very approximate): treat the slew as cross-track
        h_body = np.array([delta_h_body, 0.0, 0.0])
        h_wheels = body_to_wheel_momentum(h_body)
        sat, frac = check_saturation(h_wheels)
        if sat:
            continue

        # Footprint
        footprint = compute_footprint(ep.r_eci, q_target, fov, ep.jd)

        planned.append(
            PlannedImage(
                tile_id=tile.id,
                t_image=t_image_start,
                q_BN=q_target,
                off_nadir_deg=off_nadir,
                footprint=footprint,
            )
        )
        current_q = q_target
        current_t = t_image_end

    # 7) Build attitude trajectory waypoints
    waypoints: List[Tuple[float, np.ndarray]] = []
    # Start: nadir at t=0
    q_start = nadir_pointing_quat(ephemeris[0].r_eci, ephemeris[0].v_eci)
    waypoints.append((0.0, q_start))

    prev_t = 0.0
    for img in planned:
        # Slew from prev to image attitude across the (t_prev_end, t_image_start) window
        # Slew completes at t_image_start - settle (but the controller settles in real life;
        # for the schedule we just need the attitude at the shutter to be q_BN).
        slew_arrive = max(prev_t + 0.5, img.t_image - 0.1)
        if slew_arrive > prev_t:
            waypoints.append((slew_arrive, img.q_BN))
        # Hold during shutter
        waypoints.append((img.t_image + settings.integration_time_s, img.q_BN))
        prev_t = img.t_image + settings.integration_time_s

    # End: return toward nadir at end of pass
    last_ep = ephemeris[-1]
    q_end = nadir_pointing_quat(last_ep.r_eci, last_ep.v_eci)
    if waypoints[-1][0] < pass_duration:
        waypoints.append((pass_duration, q_end))

    attitude_samples = generate_attitude_trajectory(waypoints, dt=0.020)

    # 8) Schedule dict
    attitude_list = [
        {
            "t": float(t),
            "q_BN": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        }
        for t, q in attitude_samples
    ]
    shutter_list = [
        {
            "t_start": float(img.t_image),
            "t_end": float(img.t_image + integration),
            "duration_s": float(integration),
            "tile_id": img.tile_id,
        }
        for img in planned
    ]

    schedule = {
        "pass_start_utc": pass_start_utc,
        "pass_end_utc": pass_end_utc,
        "attitude": attitude_list,
        "shutters": shutter_list,
        "metadata": {
            "n_images": len(planned),
            "fov_deg": fov,
            "tile_size_km": tile_size_km,
        },
    }

    # Diagnostics
    aoi_area = polygon_area_km2(list(aoi_polygon))
    estimated_coverage = min(1.0, len(planned) * (tile_size_km ** 2) / max(aoi_area, 1e-6))
    diagnostics = {
        "n_tiles_total": len(tiles),
        "n_tiles_accessible": len(accesses),
        "n_tiles_imaged": len(planned),
        "estimated_coverage": estimated_coverage,
        "imaging_window_s": [
            float(planned[0].t_image) if planned else 0.0,
            float(planned[-1].t_image + integration) if planned else 0.0,
        ],
        "closest_approach_s": float(ca_ep.t_offset_s),
        "off_nadir_at_ca_deg": float(off_at_ca),
        "tile_size_km": tile_size_km,
        "cumulative_delta_h_nms": cumulative_dh,
        "footprints": [
            {"tile_id": img.tile_id, "corners_latlon": img.footprint, "off_nadir_deg": img.off_nadir_deg}
            for img in planned
        ],
    }
    ephemeris_summary = {
        "closest_approach_t": float(ca_ep.t_offset_s),
        "min_off_nadir_deg": float(off_at_ca),
        "sub_sat_lat_at_ca": float(ca_ep.lat_deg),
        "sub_sat_lon_at_ca": float(ca_ep.lon_deg),
        "altitude_km_at_ca": float(ca_ep.alt_km),
    }

    return PlanResult(
        schedule=schedule,
        diagnostics=diagnostics,
        ephemeris_summary=ephemeris_summary,
    )
