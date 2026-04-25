"""Mission planner: build a schedule (attitude trajectory + shutter list)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (
    DEFAULT_FOV_DEG,
    DEFAULT_INERTIA,
    OFF_NADIR_SAFE_LIMIT_DEG,
    PASS_DURATION_S,
    SHUTTER_DURATION_S,
)
from .attitude import (
    compute_off_nadir,
    compute_pointing_quat,
    estimate_body_rate,
    generate_attitude_trajectory,
    nadir_quat,
    quaternion_angular_distance,
)
from .dynamics import estimate_slew_time, momentum_change_estimate
from .frames import geodetic_to_eci
from .imaging import compute_footprint
from .propagator import EphemerisPoint, ephemeris_arrays, propagate_pass
from .tiling import (
    TileCenter,
    adaptive_tile_size_km,
    point_in_polygon,
    tile_aoi,
)


@dataclass
class _TileWindow:
    tile: TileCenter
    t_best: float            # time of minimum off-nadir (sec offset)
    off_nadir_min: float     # at t_best (deg)
    accessible: bool


def _index_at_time(t_array: np.ndarray, t: float) -> int:
    return int(np.clip(np.argmin(np.abs(t_array - t)), 0, len(t_array) - 1))


def _aoi_centroid(aoi: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(aoi)
    return sum(p[0] for p in aoi) / n, sum(p[1] for p in aoi) / n


def _find_closest_approach(
    ephem: List[EphemerisPoint], aoi: List[Tuple[float, float]]
) -> Tuple[int, float]:
    """Index and off-nadir(deg) at AOI-center closest approach."""
    lat_c, lon_c = _aoi_centroid(aoi)
    best_i = 0
    best_off = 1e9
    for i, p in enumerate(ephem):
        if math.isnan(p.lat_deg):
            continue
        target = geodetic_to_eci(math.radians(lat_c), math.radians(lon_c), 0.0, p.jd)
        off = compute_off_nadir(np.array(p.r_eci), target)
        if off < best_off:
            best_off = off
            best_i = i
    return best_i, best_off


def _scan_tile_windows(
    ephem: List[EphemerisPoint],
    tiles: List[TileCenter],
    off_nadir_limit_deg: float,
) -> List[_TileWindow]:
    out: List[_TileWindow] = []
    for tile in tiles:
        best_i = -1
        best_off = 1e9
        lat_r = math.radians(tile.lat_deg)
        lon_r = math.radians(tile.lon_deg)
        for i, p in enumerate(ephem):
            if math.isnan(p.lat_deg):
                continue
            target = geodetic_to_eci(lat_r, lon_r, 0.0, p.jd)
            off = compute_off_nadir(np.array(p.r_eci), target)
            if off < best_off:
                best_off = off
                best_i = i
        accessible = best_off <= off_nadir_limit_deg and best_i >= 0
        out.append(
            _TileWindow(
                tile=tile,
                t_best=ephem[best_i].t_offset_s if best_i >= 0 else 0.0,
                off_nadir_min=best_off,
                accessible=accessible,
            )
        )
    return out


def _serpentine_order(tiles: Sequence[TileCenter]) -> List[TileCenter]:
    """Sort tiles in a boustrophedon raster (rows = lat, cols = lon)."""
    if not tiles:
        return []
    # Group by row
    by_lat: Dict[float, List[TileCenter]] = {}
    for t in tiles:
        key = round(t.lat_deg, 4)
        by_lat.setdefault(key, []).append(t)
    rows = sorted(by_lat.keys())
    out: List[TileCenter] = []
    for k, lat in enumerate(rows):
        row = sorted(by_lat[lat], key=lambda x: x.lon_deg)
        if k % 2 == 1:
            row = list(reversed(row))
        out.extend(row)
    return out


def plan_imaging(
    tle_line1: str,
    tle_line2: str,
    aoi_polygon: List[Tuple[float, float]],
    pass_start_utc: str,
    pass_end_utc: str,
    sc_params: Optional[Dict] = None,
    *,
    strategy: str = "boustrophedon",
    settle_margin_s: float = 3.0,
    off_nadir_margin_deg: float = 5.0,
) -> Dict:
    """Compute a full schedule. Returns a dict ready to serve via the API."""
    sc_params = sc_params or {}
    fov_deg = float(sc_params.get("fov_deg", DEFAULT_FOV_DEG))
    inertia = tuple(sc_params.get("inertia", DEFAULT_INERTIA))
    off_nadir_limit = OFF_NADIR_SAFE_LIMIT_DEG - off_nadir_margin_deg

    # 1. Propagate
    ephem = propagate_pass(tle_line1, tle_line2, pass_start_utc, pass_end_utc, dt=1.0)
    t_arr, jd_arr, r_arr, v_arr, lla_arr = ephemeris_arrays(ephem)
    n_steps = len(ephem)
    if n_steps < 2:
        raise ValueError("Pass window too short")

    # 2. Closest approach
    ca_idx, ca_off = _find_closest_approach(ephem, aoi_polygon)
    ca_t = ephem[ca_idx].t_offset_s
    alt_km = ephem[ca_idx].alt_km if not math.isnan(ephem[ca_idx].alt_km) else 500.0

    # 3. Tile the AOI
    tile_size_km = max(8.0, adaptive_tile_size_km(max(ca_off, 1.0), fov_deg, alt_km))
    tiles = tile_aoi(aoi_polygon, tile_size_km)

    # 4. Tile windows
    windows = _scan_tile_windows(ephem, tiles, off_nadir_limit)
    accessible = [w for w in windows if w.accessible]

    # 5. Order tiles (boustrophedon by default)
    ordered_windows = sorted(accessible, key=lambda w: (w.tile.lat_deg, w.tile.lon_deg))
    if strategy == "boustrophedon":
        ordered_tiles = _serpentine_order([w.tile for w in accessible])
        # Realign by tile id
        tile_to_window = {w.tile.id: w for w in accessible}
        ordered_windows = [tile_to_window[t.id] for t in ordered_tiles]
    elif strategy == "center_first":
        lat_c, lon_c = _aoi_centroid(aoi_polygon)
        ordered_windows = sorted(
            accessible,
            key=lambda w: (w.tile.lat_deg - lat_c) ** 2 + (w.tile.lon_deg - lon_c) ** 2,
        )
    elif strategy == "greedy":
        ordered_windows = sorted(accessible, key=lambda w: w.t_best)

    # 6. Build the time line: assign each tile an imaging time
    schedule_entries: List[Dict] = []
    delta_h_total = 0.0
    t_cursor = max(2.0, ca_t - PASS_DURATION_S * 0.45)  # start ~3min before CA
    t_cursor = min(t_cursor, PASS_DURATION_S - 10.0)
    prev_q: Optional[np.ndarray] = None

    # Initial nadir attitude at t=0
    init_q = nadir_quat(np.array(ephem[0].r_eci), np.array(ephem[0].v_eci))
    waypoints: List[Tuple[float, np.ndarray]] = [(0.0, init_q)]
    hold_intervals: List[Tuple[float, float]] = []
    shutters: List[Dict] = []
    skipped: List[str] = []
    body_rate_estimates: List[float] = []

    prev_q = init_q
    prev_t = 0.0

    for w in ordered_windows:
        # Sample the satellite state at the best observation time
        i_obs = _index_at_time(t_arr, w.t_best)
        r_sat = r_arr[i_obs]
        v_sat = v_arr[i_obs]
        jd_obs = jd_arr[i_obs]
        target = geodetic_to_eci(
            math.radians(w.tile.lat_deg),
            math.radians(w.tile.lon_deg),
            0.0,
            jd_obs,
        )
        q_obs = compute_pointing_quat(r_sat, v_sat, target)
        ang = quaternion_angular_distance(prev_q, q_obs)
        slew = estimate_slew_time(ang, inertia)
        # Time to start observing this tile
        t_obs = max(t_cursor, prev_t) + slew + settle_margin_s
        # Don't image after the time the geometry is good for (use later if needed)
        t_obs = max(t_obs, w.t_best)  # don't image earlier than the access window center
        # Check budget
        dh = momentum_change_estimate(ang, inertia)
        if delta_h_total + dh > 0.180:    # leave some margin from 0.200
            skipped.append(w.tile.id)
            continue
        if t_obs + SHUTTER_DURATION_S > PASS_DURATION_S - 1.0:
            skipped.append(w.tile.id)
            continue

        # Insert slew start waypoint and hold start waypoint
        # - end of slew = t_obs - 0   (settle is implicit; the trajectory will
        #   smoothly arrive at q_obs at t_obs)
        t_slew_start = max(prev_t, t_obs - slew - settle_margin_s)
        t_settle_start = t_obs - settle_margin_s
        if t_settle_start < t_slew_start + 1e-3:
            t_settle_start = t_slew_start + max(slew, 0.05)
        if t_obs < t_settle_start + 1e-3:
            t_obs = t_settle_start + 0.1

        # During [t_slew_start, t_settle_start] we do the slew
        # During [t_settle_start, t_obs + SHUTTER]: hold q_obs
        waypoints.append((t_slew_start, prev_q))
        waypoints.append((t_settle_start, q_obs))
        waypoints.append((t_obs, q_obs))
        waypoints.append((t_obs + SHUTTER_DURATION_S, q_obs))
        hold_intervals.append((t_settle_start, t_obs + SHUTTER_DURATION_S))

        shutters.append(
            {
                "t_start": float(t_obs),
                "t_end": float(t_obs + SHUTTER_DURATION_S),
                "tile_id": w.tile.id,
                "tile_lat_deg": float(w.tile.lat_deg),
                "tile_lon_deg": float(w.tile.lon_deg),
                "off_nadir_deg": float(w.off_nadir_min),
                "q_BN": [float(x) for x in q_obs],
            }
        )

        delta_h_total += dh
        body_rate_estimates.append(0.0)  # held attitude during shutter
        prev_q = q_obs
        prev_t = t_obs + SHUTTER_DURATION_S
        t_cursor = prev_t

    # Final return-to-nadir at end of pass
    t_end_eph = ephem[-1].t_offset_s
    if prev_t < t_end_eph - 0.5:
        final_q = nadir_quat(np.array(ephem[-1].r_eci), np.array(ephem[-1].v_eci))
        ang = quaternion_angular_distance(prev_q, final_q)
        slew = estimate_slew_time(ang, inertia)
        t_slew_start = min(prev_t + 0.5, t_end_eph - max(slew, 0.5))
        waypoints.append((t_slew_start, prev_q))
        waypoints.append((t_end_eph, final_q))

    # Generate trajectory
    trajectory = generate_attitude_trajectory(
        waypoints, dt=0.020, hold_dt=0.100, hold_intervals=hold_intervals
    )

    # Compute footprints
    footprints: List[List[Tuple[float, float]]] = []
    for s in shutters:
        i = _index_at_time(t_arr, s["t_start"])
        fp = compute_footprint(r_arr[i], np.array(s["q_BN"]), jd_arr[i], fov_deg)
        s["footprint"] = [list(c) for c in fp]
        footprints.append(fp)

    # Diagnostics
    t_active = sum(s["t_end"] - s["t_start"] for s in shutters)
    if shutters:
        t_active += shutters[-1]["t_end"] - shutters[0]["t_start"]

    schedule = {
        "meta": {
            "pass_start_utc": pass_start_utc,
            "pass_end_utc": pass_end_utc,
            "fov_deg": fov_deg,
            "inertia": list(inertia),
        },
        "attitude": [
            {"t": float(t), "q_BN": [float(x) for x in q]} for t, q in trajectory
        ],
        "shutters": shutters,
    }
    diagnostics = {
        "n_tiles_total": len(tiles),
        "n_tiles_imaged": len(shutters),
        "n_tiles_skipped": len(skipped),
        "skipped_tile_ids": skipped,
        "estimated_delta_h_used_nms": float(delta_h_total),
        "imaging_window_s": [
            float(shutters[0]["t_start"]) if shutters else None,
            float(shutters[-1]["t_end"]) if shutters else None,
        ],
        "closest_approach_s": float(ca_t),
        "closest_approach_off_nadir_deg": float(ca_off),
        "tile_size_km": float(tile_size_km),
    }
    return {
        "schedule": schedule,
        "diagnostics": diagnostics,
        "ephemeris_summary": {
            "closest_approach_t": float(ca_t),
            "min_off_nadir_deg": float(ca_off),
            "sub_sat_lat_at_ca": float(ephem[ca_idx].lat_deg),
            "sub_sat_lon_at_ca": float(ephem[ca_idx].lon_deg),
            "n_ephem_points": len(ephem),
        },
        "tiles": [t.to_dict() for t in tiles],
        "footprints": [[list(c) for c in fp] for fp in footprints],
    }
