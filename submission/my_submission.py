"""
Lost in Space Track - submission (v3, body-angular raster mosaic).

Strategy
--------
Real agile-EO satellites tessellate the AOI in BODY angular coordinates at a
reference time near closest approach, then raster through the grid in
serpentine order. That gives:
  * FOV-matched tile spacing at any off-nadir angle (FOV is 2 deg in body frame
    regardless of geometry).
  * Tiny inter-tile slews (1-2 deg), short settle, low momentum.
  * One contiguous imaging block packed around closest approach (efficient).

Per-frame logic
---------------
  1. Probe the pass to find t_ca = time of min off-nadir to AOI centroid.
  2. Build the AOI's body-angular bounding box at t_ca relative to the
     centroid-pointing attitude.
  3. Tile with footprint-matched spacing (default 1.7 deg <= FOV = 2.0 deg
     for ~15% overlap).
  4. Serpentine ordering -> small inter-tile slews.
  5. Pack shutters around t_ca at ~1.0 s/tile.
  6. Recompute the pointing quaternion at the ACTUAL shutter time (geometry
     moves while we image - this was the v2 bug that capped the score).
  7. Adaptive settle (60-250 ms) based on slew magnitude.
  8. Final gate check at shutter midtime; drop frames that would fail.

Exports plan_imaging(...) per Section 7 of the problem statement.
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
RECOVER_S = 0.05            # short post-shutter pad
SETTLE_MIN = 0.060          # for tiny (<1 deg) slews
SETTLE_MAX = 0.250          # for big (>10 deg) slews
OMEGA_PEAK_DPS = 4.0        # peak commanded body rate (smear is during shutter only)
OFF_NADIR_TARGET_MAX = 55.0 # margin below 60 deg hard limit
DT_PER_TILE_S = 1.05        # tile cadence around closest approach
FOV_DEG = 2.0               # imager FOV (body frame)


# ============================================================
# Time helpers
# ============================================================
def _iso_to_jd(iso_utc):
    s = iso_utc.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s).astimezone(timezone.utc)
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


# ============================================================
# Frames
# ============================================================
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
    """q_BN (scalar-last) putting body +Z on target."""
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
    q = R.from_matrix(M).as_quat()
    return q / np.linalg.norm(q)


def _off_nadir_deg_q(q_BN, r_sat):
    z_eci = R.from_quat(q_BN).apply(np.array([0.0, 0.0, 1.0]))
    nadir = -r_sat / np.linalg.norm(r_sat)
    return float(np.degrees(np.arccos(np.clip(np.dot(z_eci, nadir), -1.0, 1.0))))


def _off_nadir_deg_dir(d_eci, r_sat):
    nadir = -r_sat / np.linalg.norm(r_sat)
    d = d_eci / np.linalg.norm(d_eci)
    return float(np.degrees(np.arccos(np.clip(np.dot(d, nadir), -1.0, 1.0))))


# ============================================================
# Ray-ellipsoid intersection (for tile -> ground point)
# ============================================================
def _ray_ellipsoid_intersect(o, d):
    a = WGS84_A; b = a * np.sqrt(1.0 - WGS84_E2)
    M = np.diag([1.0 / a**2, 1.0 / a**2, 1.0 / b**2])
    od = o @ M @ d; dd = d @ M @ d; oo = o @ M @ o
    disc = od * od - dd * (oo - 1.0)
    if disc < 0:
        return None
    t = (-od - np.sqrt(disc)) / dd
    if t < 0:
        return None
    return o + t * d


def _ecef_to_geodetic(x_e, y_e, z_e):
    p = np.hypot(x_e, y_e)
    lon = np.degrees(np.arctan2(y_e, x_e))
    b_axis = WGS84_A * np.sqrt(1.0 - WGS84_E2)
    theta = np.arctan2(z_e * WGS84_A, p * b_axis)
    ep2 = WGS84_E2 / (1.0 - WGS84_E2)
    lat = np.degrees(np.arctan2(z_e + ep2 * b_axis * np.sin(theta) ** 3,
                                 p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3))
    return float(lat), float(lon)


# ============================================================
# AOI helpers
# ============================================================
def _aoi_pts(aoi_polygon_llh):
    return aoi_polygon_llh[:-1] if (aoi_polygon_llh[0] == aoi_polygon_llh[-1]) \
        else aoi_polygon_llh


def _aoi_centroid(aoi_polygon_llh):
    pts = _aoi_pts(aoi_polygon_llh)
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


# ============================================================
# Closest-approach probe + body-angular raster
# ============================================================
def _find_t_ca(sat, clat, clon, jd0, fr0, T_pass, dt=2.0):
    best_t = None; best_off = 1e9
    n = int(T_pass / dt) + 1
    for k in range(n):
        t = float(k) * dt
        jd, fr = _add_seconds(jd0, fr0, t)
        r_sat, v_sat = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(clat, clon, jd, fr)
        off = _off_nadir_deg_dir(r_tgt - r_sat, r_sat)
        if off < best_off:
            best_off = off; best_t = t
    lo = max(0.0, best_t - dt); hi = min(T_pass, best_t + dt)
    for t in np.arange(lo, hi + 0.5, 0.5):
        jd, fr = _add_seconds(jd0, fr0, float(t))
        r_sat, v_sat = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(clat, clon, jd, fr)
        off = _off_nadir_deg_dir(r_tgt - r_sat, r_sat)
        if off < best_off:
            best_off = off; best_t = float(t)
    return best_t, best_off


def _aoi_angular_extent_at_tca(sat, aoi_pts_llh, jd0, fr0, t_ca):
    clat = float(np.mean([p[0] for p in aoi_pts_llh]))
    clon = float(np.mean([p[1] for p in aoi_pts_llh]))
    jd, fr = _add_seconds(jd0, fr0, t_ca)
    r_sat, v_sat = _propagate(sat, jd, fr)
    r_centroid = _llh_to_eci(clat, clon, jd, fr)
    q_ref = _attitude_pointing_at(r_sat, r_centroid, v_sat)
    DCM_BN = R.from_quat(q_ref).as_matrix()         # B -> N
    alphas = []; betas = []
    for (lat, lon) in aoi_pts_llh:
        r_corner = _llh_to_eci(lat, lon, jd, fr)
        d = r_corner - r_sat; d /= np.linalg.norm(d)
        d_b = DCM_BN.T @ d
        alphas.append(np.degrees(np.arctan2(d_b[0], d_b[2])))
        betas.append(np.degrees(np.arctan2(d_b[1], d_b[2])))
    return q_ref, min(alphas), max(alphas), min(betas), max(betas), \
           r_sat, v_sat, jd, fr


def _make_raster(alpha_lo, alpha_hi, beta_lo, beta_hi, spacing_deg):
    pad = spacing_deg * 0.5
    a_lo, a_hi = alpha_lo - pad, alpha_hi + pad
    b_lo, b_hi = beta_lo - pad, beta_hi + pad
    n_a = max(2, int(np.ceil((a_hi - a_lo) / spacing_deg)) + 1)
    n_b = max(2, int(np.ceil((b_hi - b_lo) / spacing_deg)) + 1)
    a_vals = np.linspace(a_lo, a_hi, n_a)
    b_vals = np.linspace(b_lo, b_hi, n_b)
    out = []
    for j, b in enumerate(b_vals):
        row = list(a_vals) if (j % 2 == 0) else list(reversed(a_vals))
        for a in row:
            out.append((float(a), float(b)))
    return out, n_a, n_b


def _ground_target_for_body_dir(r_sat, q_ref, alpha_deg, beta_deg):
    DCM_BN = R.from_quat(q_ref).as_matrix()
    a, b = np.radians(alpha_deg), np.radians(beta_deg)
    d_b = np.array([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)])
    d_eci = DCM_BN @ d_b
    return _ray_ellipsoid_intersect(r_sat, d_eci)


# ============================================================
# Scheduling: body-angular tiles, attitude recomputed at firing time
# ============================================================
def _settle_for_slew(slew_deg):
    if slew_deg <= 1.0:
        return SETTLE_MIN
    if slew_deg >= 10.0:
        return SETTLE_MAX
    return SETTLE_MIN + (SETTLE_MAX - SETTLE_MIN) * (slew_deg - 1.0) / 9.0


def _quintic_s(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def _quat_angle_deg(q1, q2):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2)))))))


def _schedule_mosaic(sat, jd0, fr0, T_pass, t_ca, q_ref, tiles_ab,
                     off_nadir_budget=OFF_NADIR_TARGET_MAX):
    n = len(tiles_ab)
    t_first = max(1.0, t_ca - (n / 2.0) * DT_PER_TILE_S)

    # tile -> lat/lon (footprint center on the ellipsoid at t_ca)
    jd_ca, fr_ca = _add_seconds(jd0, fr0, t_ca)
    r_sat_ca, _ = _propagate(sat, jd_ca, fr_ca)
    g = _gmst_rad(jd_ca, fr_ca); cg, sg = np.cos(g), np.sin(g)

    tile_targets = []
    for (alpha, beta) in tiles_ab:
        p_eci = _ground_target_for_body_dir(r_sat_ca, q_ref, alpha, beta)
        if p_eci is None:
            tile_targets.append(None); continue
        x_e = cg * p_eci[0] + sg * p_eci[1]
        y_e = -sg * p_eci[0] + cg * p_eci[1]
        z_e = p_eci[2]
        lat, lon = _ecef_to_geodetic(x_e, y_e, z_e)
        tile_targets.append((lat, lon))

    scheduled = []
    last_q = None
    last_t_end = max(0.0, t_first - 1.0)

    for i, target in enumerate(tile_targets):
        if target is None:
            continue
        lat, lon = target
        t_pref = t_first + i * DT_PER_TILE_S
        if t_pref < 0.5 or t_pref > T_pass - 0.5:
            continue

        # Initial pointing at t_pref to size the slew gap
        jd_p, fr_p = _add_seconds(jd0, fr0, t_pref)
        r_sat_p, v_sat_p = _propagate(sat, jd_p, fr_p)
        r_tgt_p = _llh_to_eci(lat, lon, jd_p, fr_p)
        q_pref = _attitude_pointing_at(r_sat_p, r_tgt_p, v_sat_p)

        slew_deg = 0.0 if last_q is None else _quat_angle_deg(last_q, q_pref)
        slew_T = max(0.30, 1.875 * slew_deg / OMEGA_PEAK_DPS) if slew_deg > 0.05 else 0.0
        settle_T = _settle_for_slew(slew_deg)
        earliest = last_t_end + slew_T + settle_T
        t_shutter = max(t_pref, earliest)
        if t_shutter + EXPOSE_S + RECOVER_S + 0.05 > T_pass:
            break

        # Recompute pointing at the ACTUAL shutter midtime
        t_mid = t_shutter + 0.5 * EXPOSE_S
        jd_s, fr_s = _add_seconds(jd0, fr0, t_mid)
        r_sat_s, v_sat_s = _propagate(sat, jd_s, fr_s)
        r_tgt_s = _llh_to_eci(lat, lon, jd_s, fr_s)
        q_final = _attitude_pointing_at(r_sat_s, r_tgt_s, v_sat_s)

        off = _off_nadir_deg_q(q_final, r_sat_s)
        if off > off_nadir_budget:
            continue

        # Re-evaluate slew gap using final q
        if last_q is not None:
            slew_deg = _quat_angle_deg(last_q, q_final)
            slew_T = max(0.30, 1.875 * slew_deg / OMEGA_PEAK_DPS) if slew_deg > 0.05 else 0.0
            settle_T = _settle_for_slew(slew_deg)
            earliest = last_t_end + slew_T + settle_T
            if t_shutter < earliest:
                t_shutter = earliest
                if t_shutter + EXPOSE_S + RECOVER_S + 0.05 > T_pass:
                    break

        scheduled.append({
            "t_shutter": float(t_shutter),
            "slew_from_t": float(last_t_end),
            "slew_to_t": float(t_shutter - settle_T),
            "settle_T": float(settle_T),
            "slew_T": float(slew_T),
            "slew_deg": float(slew_deg),
            "q": q_final,
            "lat": lat, "lon": lon,
            "off_nadir": float(off),
        })
        last_q = q_final
        last_t_end = t_shutter + EXPOSE_S + RECOVER_S

    return scheduled


# ============================================================
# Trajectory builder
# ============================================================
def _slew_segment(t_start, t_end, q_a, q_b, hz=25.0):
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


def _merge(*segments, min_dt=0.025):
    out = []
    for seg in segments:
        for s in seg:
            if out and s["t"] - out[-1]["t"] < min_dt:
                continue
            out.append(s)
    return out


def _build_trajectory(scheduled, fallback_q, T_pass):
    if not scheduled:
        traj = _hold_segment(0.0, T_pass, fallback_q, hz=10.0)
        traj[0]["t"] = 0.0; traj[-1]["t"] = T_pass
        return traj

    s0 = scheduled[0]
    pre_hold_end = max(0.0, s0["t_shutter"] - s0["settle_T"])
    end_frame0 = s0["t_shutter"] + EXPOSE_S + RECOVER_S
    segs = []
    segs.append(_hold_segment(0.0, pre_hold_end, s0["q"], hz=10.0))
    segs.append(_hold_segment(pre_hold_end, end_frame0, s0["q"], hz=20.0))
    last_q = s0["q"]; last_t = end_frame0

    for i in range(1, len(scheduled)):
        si = scheduled[i]
        slew_start = si["slew_from_t"]
        slew_end = si["slew_to_t"]
        end_frame = si["t_shutter"] + EXPOSE_S + RECOVER_S
        segs.append(_slew_segment(slew_start, slew_end, last_q, si["q"], hz=25.0))
        segs.append(_hold_segment(slew_end, end_frame, si["q"], hz=20.0))
        last_q = si["q"]; last_t = end_frame

    if last_t < T_pass:
        segs.append(_hold_segment(last_t, T_pass, last_q, hz=5.0))
    traj = _merge(*segs, min_dt=0.025)
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

    aoi_pts = _aoi_pts(aoi_polygon_llh)
    clat, clon = _aoi_centroid(aoi_polygon_llh)

    # 1. Closest approach to AOI centroid
    t_ca, off_ca = _find_t_ca(sat, clat, clon, jd0, fr0, T, dt=2.0)

    # 2. AOI angular extent in body frame at t_ca
    q_ref, a_lo, a_hi, b_lo, b_hi, *_ = \
        _aoi_angular_extent_at_tca(sat, aoi_pts, jd0, fr0, t_ca)

    # 3. Body-angular raster (tighter spacing for large off-nadir to keep
    #    ground footprint overlap healthy as footprint elongates).
    if off_ca >= 45.0:
        spacing = 1.4
    elif off_ca >= 25.0:
        spacing = 1.6
    else:
        spacing = 1.7
    tiles_ab, n_a, n_b = _make_raster(a_lo, a_hi, b_lo, b_hi, spacing)

    # 4-7. Schedule: per-tile shutter, attitude recomputed at firing time.
    #      For very-far passes, escalate the off-nadir budget so we still get
    #      frames (case 3 is intentionally near the 60 deg limit).
    if off_ca >= 50.0:
        budgets = [55.0, 57.0, 59.0]
    elif off_ca >= 35.0:
        budgets = [55.0, 57.0]
    else:
        budgets = [55.0]
    scheduled = []
    for budget in budgets:
        scheduled = _schedule_mosaic(sat, jd0, fr0, T, t_ca, q_ref, tiles_ab,
                                      off_nadir_budget=budget)
        if len(scheduled) >= 5:
            break

    # Fallback (only if nothing scheduled - shouldn't happen for these cases)
    jd_mid, fr_mid = _add_seconds(jd0, fr0, T / 2.0)
    r_sat_mid, v_sat_mid = _propagate(sat, jd_mid, fr_mid)
    r_tgt_mid = _llh_to_eci(clat, clon, jd_mid, fr_mid)
    q_fallback = _attitude_pointing_at(r_sat_mid, r_tgt_mid, v_sat_mid)

    attitude = _build_trajectory(scheduled, q_fallback, T)
    shutter = [{"t_start": float(s["t_shutter"]), "duration": EXPOSE_S}
               for s in scheduled]

    return {
        "objective": "max_coverage",
        "attitude": attitude,
        "shutter": shutter,
        "notes": (f"body-angular raster {n_a}x{n_b} @ {spacing:.2f}deg, "
                  f"t_ca={t_ca:.1f}s, off_ca={off_ca:.1f}deg, "
                  f"frames={len(shutter)}, omega_peak={OMEGA_PEAK_DPS}dps"),
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

        # Self-check: smear + off-nadir at every shutter
        att = sched["attitude"]
        if len(att) >= 2:
            times = np.array([s["t"] for s in att])
            quats = np.array([s["q_BN"] for s in att])
            slerp = Slerp(times, R.from_quat(quats))
            sat_o = Satrec.twoline2rv(l1, l2)
            jd0, fr0 = _iso_to_jd("2026-04-23T17:24:00Z")

            n_smear_fail = 0; n_off_fail = 0; max_rate = 0.0; max_off = 0.0
            for sh in sched["shutter"]:
                t0_s = sh["t_start"]; t1_s = t0_s + sh["duration"]
                ts = np.linspace(t0_s, t1_s, 7)
                rs = slerp(ts)
                fail = False
                for i in range(len(ts) - 1):
                    dr = rs[i].inv() * rs[i + 1]
                    qd = dr.as_quat()
                    ang = 2.0 * np.arctan2(np.linalg.norm(qd[:3]), abs(qd[3]))
                    rate = np.degrees(ang / (ts[i + 1] - ts[i]))
                    max_rate = max(max_rate, rate)
                    if rate > 0.05:
                        fail = True; break
                if fail: n_smear_fail += 1
                tm = 0.5 * (t0_s + t1_s)
                jd, fr = _add_seconds(jd0, fr0, tm)
                r_sat, _ = _propagate(sat_o, jd, fr)
                q = slerp([tm])[0].as_quat()
                off = _off_nadir_deg_q(q, r_sat)
                max_off = max(max_off, off)
                if off > 60.0: n_off_fail += 1
            print(f"  max smear rate = {max_rate:.4f} deg/s (limit 0.05; fails {n_smear_fail}/{len(sched['shutter'])})")
            print(f"  max off-nadir  = {max_off:.2f} deg (limit 60; fails {n_off_fail}/{len(sched['shutter'])})")
