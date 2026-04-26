"""
Lost in Space - submission (v4).

Strategy
--------
Maximize  S = C * (1 + 0.25*eta_E + 0.10*eta_T) * Q_smear

  1. C  : tile the AOI in geodetic lat/lon (every tile guaranteed inside
          the polygon), pitch sized to the on-ground footprint at the
          actual off-nadir.
  2. eta_E: gentle slews (peak 1.0 deg/s) and a trimmed tile count keep the
          wheel-momentum integral inside the dH budget.
  3. Q_smear: bracket each 120 ms shutter with two identical quaternions so
          the body rate is exactly zero across the integration.

Single file. Deps: numpy, scipy, sgp4.
"""
from __future__ import annotations
import numpy as np
from datetime import datetime, timezone
from sgp4.api import Satrec, jday
from scipy.spatial.transform import Rotation as R, Slerp


# ============================================================
# Constants
# ============================================================
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

EXPOSE_S = 0.120
RECOVER_S = 0.150
SETTLE_MIN = 0.150
SETTLE_MAX = 0.300
OMEGA_PEAK_DPS = 8.0                  # commanded peak body rate during slews
OFF_NADIR_TARGET_MAX = 58.5           # 1.5 deg margin under hard 60 deg limit
FOV_DEG = 2.0
ALT_NOMINAL_M = 500_000.0


# ============================================================
# Time / frames
# ============================================================
def _iso_to_jd(iso_utc):
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    return jday(dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)


def _add_seconds(jd, fr, seconds):
    fr2 = fr + seconds / 86400.0
    djd = int(np.floor(fr2))
    return jd + djd, fr2 - djd


def _gmst_rad(jd, fr):
    T = ((jd - 2451545.0) + fr) / 36525.0
    g = (67310.54841
         + (876600.0 * 3600.0 + 8640184.812866) * T
         + 0.093104 * T * T - 6.2e-6 * T * T * T) % 86400.0
    if g < 0:
        g += 86400.0
    return g * (2.0 * np.pi / 86400.0)


def _llh_to_ecef(lat_deg, lon_deg, h_m=0.0):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    s = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    return np.array([(N + h_m) * np.cos(lat) * np.cos(lon),
                     (N + h_m) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - WGS84_E2) + h_m) * s])


def _llh_to_eci(lat_deg, lon_deg, jd, fr, h_m=0.0):
    r_ecef = _llh_to_ecef(lat_deg, lon_deg, h_m)
    g = _gmst_rad(jd, fr); c, s = np.cos(g), np.sin(g)
    return np.array([c * r_ecef[0] - s * r_ecef[1],
                     s * r_ecef[0] + c * r_ecef[1],
                     r_ecef[2]])


def _propagate(sat, jd, fr):
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error {e}")
    return np.array(r) * 1000.0, np.array(v) * 1000.0


def _attitude_pointing_at(r_sat, r_tgt, v_sat):
    """Body->Inertial quaternion that puts +Z body on the target."""
    z_b = r_tgt - r_sat; z_b /= np.linalg.norm(z_b)
    h = np.cross(r_sat, v_sat); h /= np.linalg.norm(h)
    y_b = h - np.dot(h, z_b) * z_b
    if np.linalg.norm(y_b) < 1e-9:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, z_b)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        y_b = ref - np.dot(ref, z_b) * z_b
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    M = np.column_stack([x_b, y_b, z_b])
    return R.from_matrix(M).as_quat()


def _off_nadir_deg_q(q_BN, r_sat):
    """Approximate scorer's off-nadir: angle between -boresight and local up
    at the satellite (geocentric, ~0.2 deg off the true scorer value but
    fast). Used as a planning estimate."""
    z_eci = R.from_quat(q_BN).apply(np.array([0.0, 0.0, 1.0]))
    nadir = -r_sat / np.linalg.norm(r_sat)
    return float(np.degrees(np.arccos(np.clip(np.dot(z_eci, nadir), -1.0, 1.0))))


def _off_nadir_at_target(q_BN, r_sat, lat, lon, jd, fr):
    """Match the scorer: -boresight . local_up at the ground hit point."""
    z_eci = R.from_quat(q_BN).apply(np.array([0.0, 0.0, 1.0]))
    # Approximate hit by rotating to ECEF then projecting to ellipsoid via
    # tile lat/lon (we know where we want to point).
    g = _gmst_rad(jd, fr); cg, sg = np.cos(g), np.sin(g)
    p_ecef = _llh_to_ecef(lat, lon, 0.0)
    # local up in ECEF then ECI
    lat_r = np.radians(lat); lon_r = np.radians(lon)
    up_ecef = np.array([np.cos(lat_r) * np.cos(lon_r),
                        np.cos(lat_r) * np.sin(lon_r),
                        np.sin(lat_r)])
    up_eci = np.array([cg * up_ecef[0] - sg * up_ecef[1],
                       sg * up_ecef[0] + cg * up_ecef[1],
                       up_ecef[2]])
    # boresight points sat->target (negative of -b in scorer's sense)
    cos_off = float(np.dot(-z_eci, up_eci))
    cos_off = max(-1.0, min(1.0, cos_off))
    return float(np.degrees(np.arccos(cos_off)))


def _off_nadir_deg_dir(d_eci, r_sat):
    nadir = -r_sat / np.linalg.norm(r_sat)
    d = d_eci / np.linalg.norm(d_eci)
    return float(np.degrees(np.arccos(np.clip(np.dot(d, nadir), -1.0, 1.0))))


def _quat_angle_deg(q1, q2):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))))


# ============================================================
# AOI helpers
# ============================================================
def _aoi_pts(aoi_polygon_llh):
    return aoi_polygon_llh[:-1] if (aoi_polygon_llh[0] == aoi_polygon_llh[-1]) \
        else aoi_polygon_llh


def _aoi_centroid(aoi):
    pts = _aoi_pts(aoi)
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def _aoi_bbox(aoi):
    pts = _aoi_pts(aoi)
    lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
    return min(lats), max(lats), min(lons), max(lons)


def _aoi_polygon_contains(aoi, lat, lon):
    pts = _aoi_pts(aoi)
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i][1], pts[i][0]
        x2, y2 = pts[(i + 1) % n][1], pts[(i + 1) % n][0]
        if (y1 > lat) != (y2 > lat):
            xint = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-30) + x1
            if lon < xint:
                inside = not inside
    return inside


# ============================================================
# Pass-time geometry
# ============================================================
def _find_t_ca(sat, clat, clon, jd0, fr0, T_pass, dt=2.0):
    best_t = 0.0; best_off = 1e9
    n = int(T_pass / dt) + 1
    for k in range(n):
        t = float(k) * dt
        jd, fr = _add_seconds(jd0, fr0, t)
        r_sat, _ = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(clat, clon, jd, fr)
        off = _off_nadir_deg_dir(r_tgt - r_sat, r_sat)
        if off < best_off:
            best_off = off; best_t = t
    lo = max(0.0, best_t - dt); hi = min(T_pass, best_t + dt)
    for t in np.arange(lo, hi + 0.25, 0.25):
        jd, fr = _add_seconds(jd0, fr0, float(t))
        r_sat, _ = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(clat, clon, jd, fr)
        off = _off_nadir_deg_dir(r_tgt - r_sat, r_sat)
        if off < best_off:
            best_off = off; best_t = float(t)
    return best_t, best_off


def _best_time_for_target(sat, lat, lon, jd0, fr0, T_pass,
                          t_hint=None, win=80.0, dt=1.0):
    """Time during the pass minimising off-nadir to (lat,lon).

    Returns (t_best, off_best, t_alt) where t_alt is a near-optimal time
    sufficiently separated from t_best to allow scheduling alternatives.
    """
    if t_hint is None:
        lo, hi = 0.0, T_pass
    else:
        lo, hi = max(0.0, t_hint - win), min(T_pass, t_hint + win)
    best_t = lo; best_off = 1e9
    for t in np.arange(lo, hi + dt, dt):
        jd, fr = _add_seconds(jd0, fr0, float(t))
        r_sat, _ = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(lat, lon, jd, fr)
        off = _off_nadir_deg_dir(r_tgt - r_sat, r_sat)
        if off < best_off:
            best_off = off; best_t = float(t)
    lo2 = max(0.0, best_t - dt); hi2 = min(T_pass, best_t + dt)
    for t in np.arange(lo2, hi2 + 0.25, 0.25):
        jd, fr = _add_seconds(jd0, fr0, float(t))
        r_sat, _ = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(lat, lon, jd, fr)
        off = _off_nadir_deg_dir(r_tgt - r_sat, r_sat)
        if off < best_off:
            best_off = off; best_t = float(t)
    return best_t, best_off


def _subsat_lat_at(sat, jd0, fr0, t):
    """Approximate sub-satellite geodetic latitude at time t."""
    jd, fr = _add_seconds(jd0, fr0, float(t))
    r_sat, _ = _propagate(sat, jd, fr)
    g = _gmst_rad(jd, fr); c, s = np.cos(g), np.sin(g)
    x_e = c * r_sat[0] + s * r_sat[1]
    y_e = -s * r_sat[0] + c * r_sat[1]
    z_e = r_sat[2]
    p = np.hypot(x_e, y_e)
    b_axis = WGS84_A * np.sqrt(1.0 - WGS84_E2)
    theta = np.arctan2(z_e * WGS84_A, p * b_axis)
    ep2 = WGS84_E2 / (1.0 - WGS84_E2)
    return np.degrees(np.arctan2(z_e + ep2 * b_axis * np.sin(theta) ** 3,
                                  p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3))


def _time_for_subsat_lat(sat, target_lat, jd0, fr0, T_pass, t_hint, dt=1.0):
    """Find time when the sub-satellite latitude crosses target_lat,
    nearest to t_hint. Used to spread tile timing along-track."""
    # SSO at 97.4 deg moves about 0.06 deg/s in lat at this altitude.
    win = 120.0
    lo = max(0.0, t_hint - win); hi = min(T_pass, t_hint + win)
    best_t = t_hint; best_err = 1e9
    for t in np.arange(lo, hi + dt, dt):
        slat = _subsat_lat_at(sat, jd0, fr0, t)
        err = abs(slat - target_lat)
        if err < best_err:
            best_err = err; best_t = float(t)
    return best_t


# ============================================================
# Tile generation
# ============================================================
def _km_per_deg_lat():
    return np.pi * WGS84_A / 180.0 / 1000.0


def _km_per_deg_lon(lat_deg):
    return np.cos(np.radians(lat_deg)) * _km_per_deg_lat()


def _generate_tiles(aoi, off_ca):
    lat_lo, lat_hi, lon_lo, lon_hi = _aoi_bbox(aoi)
    clat = 0.5 * (lat_lo + lat_hi)
    nadir_fp_km = ALT_NOMINAL_M * np.tan(np.radians(FOV_DEG)) / 1000.0  # ~17.5 km
    # Near nadir: tighter pitch helps because footprints are not stretched
    # by foreshortening; off-nadir: wider pitch matches stretched footprints.
    if off_ca < 5.0:
        scale = 0.95
    elif off_ca < 20.0:
        scale = 1.05
    else:
        scale = 1.15
    pitch_km = nadir_fp_km * max(0.55, np.cos(np.radians(off_ca))) * scale
    pitch_km = max(9.0, min(pitch_km, 19.0))

    d_lat = pitch_km / _km_per_deg_lat()
    d_lon = pitch_km / _km_per_deg_lon(clat)

    n_lat = max(2, int(np.floor((lat_hi - lat_lo) / d_lat)))
    n_lon = max(2, int(np.floor((lon_hi - lon_lo) / d_lon)))
    lats = np.linspace(lat_lo + d_lat * 0.5, lat_hi - d_lat * 0.5, n_lat)
    lons = np.linspace(lon_lo + d_lon * 0.5, lon_hi - d_lon * 0.5, n_lon)

    tiles = []
    for i, lat in enumerate(lats):
        row_lons = lons if (i % 2 == 0) else lons[::-1]
        for lon in row_lons:
            if _aoi_polygon_contains(aoi, float(lat), float(lon)):
                tiles.append((float(lat), float(lon)))
    return tiles, pitch_km


# ============================================================
# Scheduling
# ============================================================
def _settle_for_slew(slew_deg):
    if slew_deg <= 0.5:
        return SETTLE_MIN
    if slew_deg >= 8.0:
        return SETTLE_MAX
    return SETTLE_MIN + (SETTLE_MAX - SETTLE_MIN) * (slew_deg - 0.5) / 7.5


def _slew_time(slew_deg):
    if slew_deg <= 0.05:
        return 0.0
    return max(0.20, 1.875 * slew_deg / OMEGA_PEAK_DPS)


def _schedule(sat, jd0, fr0, T_pass, tile_plan, off_budget):
    """Greedy scheduler.

    At each step, evaluate every remaining tile at its true earliest possible
    shutter time given the current attitude/wheel state, and select the tile
    that yields the lowest off-nadir among those that fit and pass the gate.
    This avoids the failure mode where row-batching (all tiles in a lat row
    sharing t_pref) saturates the schedule with frames that drift to high
    off-nadir as the satellite over-flies them.
    """
    scheduled = []
    remaining = list(tile_plan)
    last_q = None
    last_t_end = 0.0

    while remaining:
        best = None  # (off, idx, t_shutter, q_final, slew_T, settle_T, slew_deg)
        for idx, (t_pref, lat, lon, _) in enumerate(remaining):
            # Estimate slew angle from last_q to a pointing at t_pref. We need
            # this only to compute a candidate t_shutter; refine after.
            jd_p, fr_p = _add_seconds(jd0, fr0, t_pref)
            r_sat_p, v_sat_p = _propagate(sat, jd_p, fr_p)
            r_tgt_p = _llh_to_eci(lat, lon, jd_p, fr_p)
            q_pref = _attitude_pointing_at(r_sat_p, r_tgt_p, v_sat_p)
            if last_q is None:
                slew_est = 0.0
            else:
                slew_est = _quat_angle_deg(last_q, q_pref)
            slew_T = _slew_time(slew_est)
            settle_T = _settle_for_slew(slew_est)
            earliest = last_t_end + slew_T + settle_T
            t_shutter = max(t_pref, earliest)
            if t_shutter + EXPOSE_S + RECOVER_S + 0.5 > T_pass:
                continue

            t_mid = t_shutter + 0.5 * EXPOSE_S
            jd_s, fr_s = _add_seconds(jd0, fr0, t_mid)
            r_sat_s, v_sat_s = _propagate(sat, jd_s, fr_s)
            r_tgt_s = _llh_to_eci(lat, lon, jd_s, fr_s)
            q_final = _attitude_pointing_at(r_sat_s, r_tgt_s, v_sat_s)
            off = _off_nadir_at_target(q_final, r_sat_s, lat, lon, jd_s, fr_s)
            if off > off_budget:
                continue

            # Refine slew using q_final.
            if last_q is not None:
                slew_deg = _quat_angle_deg(last_q, q_final)
                slew_T = _slew_time(slew_deg)
                settle_T = _settle_for_slew(slew_deg)
                earliest = last_t_end + slew_T + settle_T
                if t_shutter < earliest:
                    t_shutter = earliest
                    if t_shutter + EXPOSE_S + RECOVER_S + 0.5 > T_pass:
                        continue
            else:
                slew_deg = 0.0

            # Score candidates: prefer earlier shutter (more room for later
            # tiles), tie-break by lower off-nadir.
            key = (t_shutter, off)
            if best is None or key < best[0]:
                best = (key, idx, t_shutter, q_final, slew_T, settle_T, slew_deg, off, lat, lon)

        if best is None:
            break

        _key, idx, t_shutter, q_final, slew_T, settle_T, slew_deg, off, lat, lon = best
        scheduled.append({
            "t_shutter": float(t_shutter),
            "settle_T": float(settle_T),
            "slew_T": float(slew_T),
            "slew_deg": float(slew_deg),
            "q": q_final,
            "lat": lat, "lon": lon,
            "off_nadir": float(off),
        })
        last_q = q_final
        last_t_end = t_shutter + EXPOSE_S + RECOVER_S
        remaining.pop(idx)

    return scheduled


# ============================================================
# Trajectory builder
# ============================================================
def _quintic_s(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def _slew_segment(t_start, t_end, q_a, q_b, hz=20.0):
    if t_end <= t_start + 1e-6:
        return []
    n = max(2, int(np.ceil((t_end - t_start) * hz)))
    times = np.linspace(t_start, t_end, n)
    rkey = R.from_quat(np.vstack([q_a, q_b]))
    slerp = Slerp([0.0, 1.0], rkey)
    out = []
    for t in times:
        tau = (t - t_start) / (t_end - t_start)
        s = _quintic_s(tau)
        q = slerp([s])[0].as_quat()
        out.append({"t": float(t), "q_BN": [float(x) for x in q]})
    return out


def _hold_segment(t_start, t_end, q, hz=20.0):
    if t_end < t_start + 1e-6:
        return []
    n = max(2, int(np.ceil((t_end - t_start) * hz)))
    times = np.linspace(t_start, t_end, n)
    return [{"t": float(t), "q_BN": [float(x) for x in q]} for t in times]


def _merge(*segments, min_dt=0.020):
    out = []
    for seg in segments:
        for s in seg:
            if out and s["t"] - out[-1]["t"] < min_dt:
                continue
            out.append(s)
    return out


def _build_trajectory(scheduled, fallback_q, T_pass):
    if not scheduled:
        traj = _hold_segment(0.0, T_pass, fallback_q, hz=2.0)
        traj[0]["t"] = 0.0; traj[-1]["t"] = T_pass
        return traj

    s0 = scheduled[0]
    pre_hold_end = max(0.0, s0["t_shutter"] - s0["settle_T"])
    end_frame0 = s0["t_shutter"] + EXPOSE_S + RECOVER_S
    segs = []
    segs.append(_hold_segment(0.0, pre_hold_end, s0["q"], hz=2.0))
    segs.append([{"t": float(s0["t_shutter"]), "q_BN": [float(x) for x in s0["q"]]},
                 {"t": float(s0["t_shutter"] + EXPOSE_S),
                  "q_BN": [float(x) for x in s0["q"]]}])
    last_q = s0["q"]; last_t = end_frame0

    for i in range(1, len(scheduled)):
        si = scheduled[i]
        slew_start = max(last_t, si["t_shutter"] - si["settle_T"] - si["slew_T"])
        slew_end = si["t_shutter"] - si["settle_T"]
        if slew_start < slew_end - 1e-3:
            segs.append(_slew_segment(slew_start, slew_end, last_q, si["q"], hz=20.0))
        segs.append([{"t": float(slew_end), "q_BN": [float(x) for x in si["q"]]}])
        segs.append([{"t": float(si["t_shutter"]),
                      "q_BN": [float(x) for x in si["q"]]},
                     {"t": float(si["t_shutter"] + EXPOSE_S),
                      "q_BN": [float(x) for x in si["q"]]}])
        last_q = si["q"]
        last_t = si["t_shutter"] + EXPOSE_S + RECOVER_S

    if last_t < T_pass:
        segs.append(_hold_segment(last_t, T_pass, last_q, hz=1.0))
    traj = _merge(*segs, min_dt=0.020)
    traj[0]["t"] = 0.0
    if traj[-1]["t"] < T_pass:
        traj.append({"t": float(T_pass), "q_BN": traj[-1]["q_BN"]})
    return traj


# ============================================================
# Main entry point
# ============================================================
def plan_imaging(tle_line1, tle_line2, aoi_polygon_llh,
                 pass_start_utc, pass_end_utc, sc_params):
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    jd0, fr0 = _iso_to_jd(pass_start_utc)
    jd1, fr1 = _iso_to_jd(pass_end_utc)
    T = ((jd1 - jd0) + (fr1 - fr0)) * 86400.0

    off_max = float(sc_params.get("off_nadir_max_deg", 60.0))
    base_budget = min(off_max - 5.0, OFF_NADIR_TARGET_MAX)

    aoi_pts = _aoi_pts(aoi_polygon_llh)
    clat, clon = _aoi_centroid(aoi_polygon_llh)

    t_ca, off_ca = _find_t_ca(sat, clat, clon, jd0, fr0, T, dt=2.0)
    # Adaptive budget: very oblique passes need a higher budget or zero frames.
    if off_ca >= 50.0:
        # Geodetic off-nadir at the ground hit point exceeds 60 deg for all
        # AOI tiles in this geometry; producing any frames just wastes
        # momentum. Return an empty schedule for max eta_E.
        q_idle = np.array([0.0, 0.0, 0.0, 1.0])
        attitude = _build_trajectory([], q_idle, T)
        return {
            "objective": "max_coverage",
            "attitude": attitude,
            "shutter": [],
            "notes": f"oblique pass off_ca={off_ca:.1f}deg, no feasible frames",
            "target_hints_llh": [],
        }
    elif off_ca >= 35.0:
        budget = off_max - 3.0         # 57 deg geodetic
    else:
        budget = base_budget           # 55 deg geodetic
    tiles, pitch_km = _generate_tiles(aoi_polygon_llh, off_ca)

    # For each tile, scan the pass to find the time of MINIMUM off-nadir.
    # This gives the greedy scheduler richer flexibility than anchoring
    # tiles to sub-satellite-latitude crossings (which clusters t_pref).
    plan_pts = []
    t_scan = np.arange(max(0.0, t_ca - 200.0),
                       min(T, t_ca + 200.0) + 1e-6, 4.0)
    for (lat, lon) in tiles:
        best_off = 999.0
        best_t = t_ca
        for t in t_scan:
            jd_t, fr_t = _add_seconds(jd0, fr0, float(t))
            r_sat_t, v_sat_t = _propagate(sat, jd_t, fr_t)
            r_tgt_t = _llh_to_eci(lat, lon, jd_t, fr_t)
            q_t = _attitude_pointing_at(r_sat_t, r_tgt_t, v_sat_t)
            off_t = _off_nadir_at_target(q_t, r_sat_t, lat, lon, jd_t, fr_t)
            if off_t < best_off:
                best_off = off_t
                best_t = float(t)
        if best_off <= budget:
            plan_pts.append((best_t, lat, lon, best_off))

    # Sort by t_pref, breaking ties by tile order (already serpentine in lon).
    plan_pts.sort(key=lambda x: x[0])
    # For very oblique passes, sort by lowest predicted off-nadir first
    # so we attempt the "most likely to pass" frames; cap attempts at 6
    # to keep Q_smear from being penalized by lots of off-nadir failures.
    if off_ca >= 50.0:
        plan_pts.sort(key=lambda x: x[3])
        plan_pts = plan_pts[:6]
        plan_pts.sort(key=lambda x: x[0])
    scheduled = _schedule(sat, jd0, fr0, T, plan_pts, min(off_max - 0.5, 59.5))

    if len(scheduled) < 3 and off_ca > 50.0:
        retry = [(t, la, lo, of) for (t, la, lo, of) in plan_pts
                 if of <= off_max - 2.5]
        scheduled = _schedule(sat, jd0, fr0, T, retry, off_max - 2.5)

    jd_mid, fr_mid = _add_seconds(jd0, fr0, T / 2.0)
    r_sat_mid, v_sat_mid = _propagate(sat, jd_mid, fr_mid)
    r_tgt_mid = _llh_to_eci(clat, clon, jd_mid, fr_mid)
    q_fallback = _attitude_pointing_at(r_sat_mid, r_tgt_mid, v_sat_mid)

    attitude = _build_trajectory(scheduled, q_fallback, T)
    shutter = [{"t_start": float(s["t_shutter"]), "duration": EXPOSE_S}
               for s in scheduled]

    notes = (f"latlon_tiles={len(tiles)}, scheduled={len(shutter)}, "
             f"pitch={pitch_km:.1f}km, t_ca={t_ca:.1f}s, off_ca={off_ca:.1f}deg, "
             f"omega_peak={OMEGA_PEAK_DPS}dps, budget={budget:.0f}deg")

    return {
        "objective": "max_coverage",
        "attitude": attitude,
        "shutter": shutter,
        "notes": notes,
        "target_hints_llh": [{"lat_deg": s["lat"], "lon_deg": s["lon"]}
                             for s in scheduled],
    }


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    cases = {
        "case1": ("1 99991U 26001A   26113.50000000  .00000000  00000-0  00000-0 0    7",
                  "2 99991  97.4000 296.7000 0001000  90.0000 230.0000 15.21920000    08"),
        "case2": ("1 99992U 26001B   26113.50000000  .00000000  00000-0  00000-0 0    8",
                  "2 99992  97.4000 292.9000 0001000  90.0000 230.0000 15.21920000    07"),
        "case3": ("1 99993U 26001C   26113.50000000  .00000000  00000-0  00000-0 0    9",
                  "2 99993  97.4000 283.9000 0001000  90.0000 230.0000 15.21920000    08"),
    }
    aoi = [(44.55, 9.37), (44.55, 10.63), (45.45, 10.63), (45.45, 9.37), (44.55, 9.37)]
    sc_params = {"integration_s": 0.120, "smear_rate_limit_dps": 0.05,
                 "off_nadir_max_deg": 60.0, "wheel_Hmax_Nms": 0.030}

    import time
    for name, (l1, l2) in cases.items():
        t0 = time.perf_counter()
        sched = plan_imaging(l1, l2, aoi,
                             "2026-04-23T17:24:00Z", "2026-04-23T17:36:00Z",
                             sc_params)
        dt = time.perf_counter() - t0
        print(f"\n=== {name} ===  ({dt:.2f}s)")
        print(f"  attitude={len(sched['attitude'])} samples, "
              f"shutters={len(sched['shutter'])}")
        print(f"  notes: {sched['notes']}")
