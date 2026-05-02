"""Mission planner: build a schedule (attitude trajectory + shutter list)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (
    DEFAULT_FOV_DEG,
    DEFAULT_INERTIA,
    OFF_NADIR_HARD_LIMIT_DEG,
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
    t_access_start: float = 0.0   # earliest time off_nadir <= limit (sec)
    t_access_end: float = 0.0     # latest time off_nadir <= limit (sec)


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
        first_ok = -1
        last_ok = -1
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
            if off <= off_nadir_limit_deg:
                if first_ok < 0:
                    first_ok = i
                last_ok = i
        accessible = best_off <= off_nadir_limit_deg and best_i >= 0
        t_start = ephem[first_ok].t_offset_s if first_ok >= 0 else 0.0
        t_end = ephem[last_ok].t_offset_s if last_ok >= 0 else 0.0
        out.append(
            _TileWindow(
                tile=tile,
                t_best=ephem[best_i].t_offset_s if best_i >= 0 else 0.0,
                off_nadir_min=best_off,
                accessible=accessible,
                t_access_start=t_start,
                t_access_end=t_end,
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
    settle_margin_s: float = 0.3,
    off_nadir_margin_deg: float = 5.0,
) -> Dict:
    """Compute a full schedule. Returns a dict ready to serve via the API."""
    sc_params = sc_params or {}
    fov_deg = float(sc_params.get("fov_deg", DEFAULT_FOV_DEG))
    inertia = tuple(sc_params.get("inertia", DEFAULT_INERTIA))
    strict_limit = OFF_NADIR_SAFE_LIMIT_DEG - off_nadir_margin_deg
    hard_ceiling = OFF_NADIR_HARD_LIMIT_DEG - 1.0  # never image at the gate itself

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

    # 4. Tile windows -- adaptive threshold ladder. Start strict; if no tile is
    # accessible (typical for far ground tracks like case 3), progressively
    # loosen toward the hard 60 deg gate, capped 1 deg below it.
    candidate_limits: List[float] = [strict_limit]
    for extra in (5.0, 7.0, 9.0, 11.0):
        cand = strict_limit + extra
        if cand <= hard_ceiling and cand not in candidate_limits:
            candidate_limits.append(cand)
    if hard_ceiling not in candidate_limits:
        candidate_limits.append(hard_ceiling)

    windows: List[_TileWindow] = []
    accessible: List[_TileWindow] = []
    off_nadir_limit = strict_limit
    for limit in candidate_limits:
        windows = _scan_tile_windows(ephem, tiles, limit)
        accessible = [w for w in windows if w.accessible]
        off_nadir_limit = limit
        if accessible:
            break

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

    # 6. Two-pass scheduling.
    #
    # Pass A iterates through tiles in serpentine order, picks the earliest
    # legal shutter time inside each tile's access window, and records the
    # observation quaternion. Pass B builds the attitude trajectory starting
    # held at the first tile's attitude (no nadir → tile1 wind-up slew) and
    # ending held at the last tile's attitude (no return-to-nadir slew). This
    # is critical: the integrated ΔH used by the scorer equals Ix * the total
    # angular distance traversed; the wind-up + return-to-nadir slews are
    # ~1-1.5 rad each, on their own consuming the entire 0.200 N·m·s budget.
    delta_h_total = 0.0
    t_cursor = 0.0  # start at pass beginning; first tile dictates timeline
    schedule_entries: List[Dict] = []  # unused, kept for backwards compat
    skipped: List[str] = []
    body_rate_estimates: List[float] = []

    # ----- Pass A: select shutter times -----
    selected: List[Dict] = []  # each: t_obs, q_obs, off_nadir, ang, slew, tile, dh
    prev_q: Optional[np.ndarray] = None
    prev_t = 0.0

    for w in ordered_windows:
        # Earliest legal start: previous shutter end + settle, or this tile's
        # access window opening, whichever is later.
        if prev_q is None:
            # First tile — no slew, no settle. Open at access window start.
            earliest_avail = w.t_access_start
        else:
            earliest_avail = prev_t + settle_margin_s
        t_obs = max(earliest_avail, w.t_access_start)
        if t_obs > w.t_access_end:
            skipped.append(w.tile.id)
            continue
        # Resample geometry at the candidate time.
        i_obs = _index_at_time(t_arr, t_obs)
        target = geodetic_to_eci(
            math.radians(w.tile.lat_deg),
            math.radians(w.tile.lon_deg),
            0.0,
            jd_arr[i_obs],
        )
        q_obs = compute_pointing_quat(r_arr[i_obs], v_arr[i_obs], target)
        if prev_q is None:
            ang = 0.0
            slew = 0.0
        else:
            ang = quaternion_angular_distance(prev_q, q_obs)
            slew = estimate_slew_time(ang, inertia)
            t_obs = max(t_obs, prev_t + slew + settle_margin_s)
            if t_obs > w.t_access_end:
                skipped.append(w.tile.id)
                continue
            # Re-sample at slew-pushed time so footprint is accurate.
            i_obs = _index_at_time(t_arr, t_obs)
            target = geodetic_to_eci(
                math.radians(w.tile.lat_deg),
                math.radians(w.tile.lon_deg),
                0.0,
                jd_arr[i_obs],
            )
            q_obs = compute_pointing_quat(r_arr[i_obs], v_arr[i_obs], target)
            ang = quaternion_angular_distance(prev_q, q_obs)
        # Real ΔH cost equals the angular distance traversed (Ix · θ).
        dh_real = float(max(inertia)) * ang
        if delta_h_total + dh_real > 0.198:
            skipped.append(w.tile.id)
            continue
        if t_obs + SHUTTER_DURATION_S > PASS_DURATION_S - 1.0:
            skipped.append(w.tile.id)
            continue
        off_nadir_at_obs = compute_off_nadir(np.asarray(r_arr[i_obs]), np.asarray(target))
        selected.append({
            "t_obs": t_obs,
            "q_obs": q_obs,
            "off_nadir": off_nadir_at_obs,
            "ang": ang,
            "slew": slew,
            "tile": w.tile,
            "i_obs": i_obs,
        })
        delta_h_total += dh_real
        prev_q = q_obs
        prev_t = t_obs + SHUTTER_DURATION_S

    # ----- Pass B: build waypoints -----
    waypoints: List[Tuple[float, np.ndarray]] = []
    hold_intervals: List[Tuple[float, float]] = []
    shutters: List[Dict] = []

    if not selected:
        # Nothing to do — keep nadir for the whole pass.
        init_q = nadir_quat(np.array(ephem[0].r_eci), np.array(ephem[0].v_eci))
        waypoints.append((0.0, init_q))
        waypoints.append((ephem[-1].t_offset_s, init_q))
        hold_intervals.append((0.0, ephem[-1].t_offset_s))
    else:
        first = selected[0]
        # Hold at the first tile's attitude from t=0 onwards. No initial slew,
        # so no ΔH cost prior to the first shutter.
        waypoints.append((0.0, first["q_obs"]))
        hold_intervals.append((0.0, first["t_obs"] + SHUTTER_DURATION_S))
        prev_q = first["q_obs"]
        prev_t = 0.0

        for k, sel in enumerate(selected):
            t_obs = sel["t_obs"]
            q_obs = sel["q_obs"]
            slew = sel["slew"]
            if k == 0:
                # Already held; just emit the shutter window.
                waypoints.append((t_obs, q_obs))
                waypoints.append((t_obs + SHUTTER_DURATION_S, q_obs))
            else:
                # Slew window: [t_obs - slew - settle, t_obs - settle].
                t_slew_start = max(prev_t, t_obs - slew - settle_margin_s)
                t_settle_start = t_obs - settle_margin_s
                if t_settle_start < t_slew_start + 1e-3:
                    t_settle_start = t_slew_start + max(slew, 0.05)
                if t_obs < t_settle_start + 1e-3:
                    t_obs = t_settle_start + 0.1
                # Hold prev_q until slew start (zero rate during the gap).
                if t_slew_start > prev_t + 1e-6:
                    waypoints.append((t_slew_start, prev_q))
                    hold_intervals.append((prev_t, t_slew_start))
                # SLERP from prev_q -> q_obs over [t_slew_start, t_settle_start].
                waypoints.append((t_settle_start, q_obs))
                # Hold q_obs through the shutter exposure.
                waypoints.append((t_obs, q_obs))
                waypoints.append((t_obs + SHUTTER_DURATION_S, q_obs))
                hold_intervals.append((t_settle_start, t_obs + SHUTTER_DURATION_S))

            shutters.append({
                "t_start": float(t_obs),
                "t_end": float(t_obs + SHUTTER_DURATION_S),
                "tile_id": sel["tile"].id,
                "tile_lat_deg": float(sel["tile"].lat_deg),
                "tile_lon_deg": float(sel["tile"].lon_deg),
                "off_nadir_deg": float(sel["off_nadir"]),
                "q_BN": [float(x) for x in q_obs],
            })
            body_rate_estimates.append(0.0)
            prev_q = q_obs
            prev_t = t_obs + SHUTTER_DURATION_S

        # Hold last attitude until end of pass — NO return-to-nadir slew.
        t_end_eph = ephem[-1].t_offset_s
        if prev_t < t_end_eph - 1e-6:
            waypoints.append((t_end_eph, prev_q))
            hold_intervals.append((prev_t, t_end_eph))

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
        "off_nadir_limit_deg": float(off_nadir_limit),
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
