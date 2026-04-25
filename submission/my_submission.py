"""
Lost in Space Track - submission (v2 with planner improvements).

Improvements over v1:
  1. Adaptive grid density: 7x7 / 6x6 / 5x5 chosen by a mid-pass off-nadir
     probe to AOI centroid (denser grids when geometry is favourable).
  2. Edge-biased grid for high-off-nadir passes: extra targets toward the
     satellite-nearest AOI edge so we don't waste candidates on unreachable
     points.
  3. 2-opt TSP-style reordering of candidates to minimise total slew angle
     before time-sort fallback.
  4. Slower commanded slews (omega_peak = 1.5 deg/s) for easier controller
     tracking.

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
SETTLE_S = 0.20
RECOVER_S = 0.10
OMEGA_PEAK_DPS = 1.5  # was 3.0; gentler for the controller


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
    gmst_s = (67310.54841
              + (876600.0 * 3600.0 + 8640184.812866) * T
              + 0.093104 * T * T
              - 6.2e-6 * T * T * T)
    gmst_s = gmst_s % 86400.0
    if gmst_s < 0:
        gmst_s += 86400.0
    return gmst_s * (2.0 * np.pi / 86400.0)


# ============================================================
# Frames + geometry
# ============================================================
def _llh_to_ecef(lat_deg, lon_deg, h_m=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    s = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    return np.array([(N + h_m) * np.cos(lat) * np.cos(lon),
                     (N + h_m) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - WGS84_E2) + h_m) * s])


def _llh_to_eci(lat_deg, lon_deg, jd, fr, h_m=0.0):
    r_ecef = _llh_to_ecef(lat_deg, lon_deg, h_m)
    g = _gmst_rad(jd, fr)
    c, s = np.cos(g), np.sin(g)
    return np.array([c * r_ecef[0] - s * r_ecef[1],
                     s * r_ecef[0] + c * r_ecef[1],
                     r_ecef[2]])


def _propagate(sat, jd, fr):
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error {e}")
    return np.array(r) * 1000.0, np.array(v) * 1000.0


def _attitude_pointing_at(r_sat_eci, r_tgt_eci, v_sat_eci):
    """q_BN (scalar-last [x,y,z,w]) putting body +Z on the target."""
    z_b = r_tgt_eci - r_sat_eci
    z_b /= np.linalg.norm(z_b)
    h = np.cross(r_sat_eci, v_sat_eci)
    h /= np.linalg.norm(h)
    y_b = h - np.dot(h, z_b) * z_b
    ny = np.linalg.norm(y_b)
    if ny < 1e-9:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, z_b)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        y_b = ref - np.dot(ref, z_b) * z_b
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    M = np.column_stack([x_b, y_b, z_b])
    q = R.from_matrix(M).as_quat()
    return q / np.linalg.norm(q)


def _off_nadir_deg(q_BN, r_sat_eci):
    z_eci = R.from_quat(q_BN).apply(np.array([0.0, 0.0, 1.0]))
    nadir = -r_sat_eci / np.linalg.norm(r_sat_eci)
    c = float(np.clip(np.dot(z_eci, nadir), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# ============================================================
# Probe + adaptive grid
# ============================================================
def _aoi_bounds(aoi_polygon_llh):
    pts = aoi_polygon_llh[:-1] if (aoi_polygon_llh[0] == aoi_polygon_llh[-1]) \
          else aoi_polygon_llh
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return min(lats), max(lats), min(lons), max(lons), \
           float(np.mean(lats)), float(np.mean(lons))


def _probe_centroid_offnadir(sat, clat, clon, jd0, fr0, T_pass):
    """Returns (best_off_nadir_deg, sub_sat_lon_at_best)."""
    best_off = 1e9
    best_sub_lon = clon
    n = int(T_pass) + 1
    for k in range(0, n, 5):  # 5-second probe is plenty
        t = float(k)
        jd, fr = _add_seconds(jd0, fr0, t)
        r_sat, v_sat = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(clat, clon, jd, fr)
        q = _attitude_pointing_at(r_sat, r_tgt, v_sat)
        off = _off_nadir_deg(q, r_sat)
        if off < best_off:
            best_off = off
            # sub-sat lon (for east/west bias decision)
            g = _gmst_rad(jd, fr)
            x_e = np.cos(g) * r_sat[0] + np.sin(g) * r_sat[1]
            y_e = -np.sin(g) * r_sat[0] + np.cos(g) * r_sat[1]
            best_sub_lon = float(np.degrees(np.arctan2(y_e, x_e)))
    return best_off, best_sub_lon


def _grid_targets_adaptive(aoi_polygon_llh, n_lat, n_lon,
                            edge_bias=None, bias_strength=0.6):
    """edge_bias in {None,'west','east','north','south'} skews target
    density toward that edge of the AOI. bias_strength in [0,1]."""
    lat_min, lat_max, lon_min, lon_max, _, _ = _aoi_bounds(aoi_polygon_llh)
    # Baseline cell-centred grid
    u = (np.arange(n_lon) + 0.5) / n_lon  # 0..1 longitude param
    v = (np.arange(n_lat) + 0.5) / n_lat  # 0..1 latitude param

    if edge_bias == "west":
        u = u ** (1.0 + bias_strength)              # cluster toward 0
    elif edge_bias == "east":
        u = 1.0 - (1.0 - u) ** (1.0 + bias_strength)
    elif edge_bias == "south":
        v = v ** (1.0 + bias_strength)
    elif edge_bias == "north":
        v = 1.0 - (1.0 - v) ** (1.0 + bias_strength)

    out = []
    for vi in v:
        for ui in u:
            out.append((lat_min + vi * (lat_max - lat_min),
                        lon_min + ui * (lon_max - lon_min)))
    return out


def _choose_grid_size(centroid_off):
    """Adaptive density based on geometry probe."""
    if centroid_off < 15.0:
        return 7, 7
    if centroid_off < 35.0:
        return 6, 6
    return 5, 5


# ============================================================
# Best-time search
# ============================================================
def _best_time_for_target(sat, lat, lon, jd0, fr0, T_pass, dt=1.0):
    best = None
    n = int(T_pass / dt) + 1
    for k in range(n):
        t = k * dt
        jd, fr = _add_seconds(jd0, fr0, t)
        r_sat, v_sat = _propagate(sat, jd, fr)
        r_tgt = _llh_to_eci(lat, lon, jd, fr)
        q = _attitude_pointing_at(r_sat, r_tgt, v_sat)
        offn = _off_nadir_deg(q, r_sat)
        if best is None or offn < best[1]:
            best = (t, offn, q)
    return best


# ============================================================
# Slew sizing + scheduling (with 2-opt)
# ============================================================
def _quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, d)
    return float(np.degrees(2.0 * np.arccos(d)))


def _quintic_s(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def _two_opt_reorder(cands, max_passes=2, t_window=15.0):
    """2-opt swap to minimise total angular slew distance, but only
    swap pairs whose t_pref are within t_window seconds (so we do not
    drift far from the time-feasible ordering)."""
    if len(cands) < 4:
        return cands
    order = list(cands)

    def total_angle(seq):
        return sum(_quat_angle_deg(seq[i]["q"], seq[i + 1]["q"])
                   for i in range(len(seq) - 1))

    for _ in range(max_passes):
        improved = False
        for i in range(1, len(order) - 2):
            for j in range(i + 1, len(order) - 1):
                if abs(order[i]["t_pref"] - order[j]["t_pref"]) > t_window:
                    continue
                a0 = _quat_angle_deg(order[i - 1]["q"], order[i]["q"])
                a1 = _quat_angle_deg(order[j]["q"], order[j + 1]["q"])
                b0 = _quat_angle_deg(order[i - 1]["q"], order[j]["q"])
                b1 = _quat_angle_deg(order[i]["q"], order[j + 1]["q"])
                if b0 + b1 + 1e-6 < a0 + a1:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return order


def _schedule_with_slews(candidates, omega_peak_dps=OMEGA_PEAK_DPS,
                         T_pass=720.0):
    if not candidates:
        return []
    out = []
    last_t_end = 0.0
    last_q = None
    for c in candidates:
        if last_q is None:
            slew_T = 0.0
        else:
            theta = _quat_angle_deg(last_q, c["q"])
            slew_T = max(1.0, 1.875 * theta / omega_peak_dps)
        earliest = last_t_end + slew_T + SETTLE_S
        t_shutter = max(c["t_pref"], earliest)
        if t_shutter + EXPOSE_S + 0.05 > T_pass:
            break
        out.append({**c,
                    "t_shutter": float(t_shutter),
                    "slew_from_t": float(last_t_end),
                    "slew_to_t": float(t_shutter - SETTLE_S),
                    "slew_T": float(slew_T)})
        last_t_end = t_shutter + EXPOSE_S + RECOVER_S
        last_q = c["q"]
    return out


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
        traj[0]["t"] = 0.0
        traj[-1]["t"] = T_pass
        return traj

    s0 = scheduled[0]
    first_q = s0["q"]
    pre_hold_end = max(0.0, s0["t_shutter"] - SETTLE_S)
    end_frame0 = s0["t_shutter"] + EXPOSE_S + RECOVER_S

    segs = []
    segs.append(_hold_segment(0.0, pre_hold_end, first_q, hz=10.0))
    segs.append(_hold_segment(pre_hold_end, end_frame0, first_q, hz=20.0))
    last_q = first_q
    last_t = end_frame0

    for i in range(1, len(scheduled)):
        si = scheduled[i]
        slew_start = si["slew_from_t"]
        slew_end = si["slew_to_t"]
        end_frame = si["t_shutter"] + EXPOSE_S + RECOVER_S
        segs.append(_slew_segment(slew_start, slew_end, last_q, si["q"], hz=25.0))
        segs.append(_hold_segment(slew_end, end_frame, si["q"], hz=20.0))
        last_q = si["q"]
        last_t = end_frame

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

    lat_min, lat_max, lon_min, lon_max, clat, clon = _aoi_bounds(aoi_polygon_llh)

    # --- Improvement 1: probe geometry, choose grid density adaptively
    centroid_off, sub_lon = _probe_centroid_offnadir(sat, clat, clon, jd0, fr0, T)
    n_lat, n_lon = _choose_grid_size(centroid_off)

    # --- Improvement 2: edge-bias the grid for high-off-nadir passes
    edge_bias = None
    if centroid_off >= 35.0:
        # Bias along longitude toward the side closer to the sub-sat track.
        edge_bias = "east" if sub_lon > clon else "west"

    targets = _grid_targets_adaptive(aoi_polygon_llh, n_lat, n_lon,
                                      edge_bias=edge_bias, bias_strength=0.6)

    # Find min-off-nadir time for each target
    raw_cands = []
    for (lat, lon) in targets:
        result = _best_time_for_target(sat, lat, lon, jd0, fr0, T, dt=1.0)
        if result is None:
            continue
        t, offn, q = result
        raw_cands.append({"t_pref": float(t), "offn": float(offn),
                          "q": np.asarray(q, dtype=float),
                          "lat": lat, "lon": lon})

    # Adaptive off-nadir budget
    chosen = []
    for offn_max in (55.0, 57.0, 59.0):
        chosen = [c for c in raw_cands if c["offn"] <= offn_max]
        if len(chosen) >= 5:
            break
    chosen.sort(key=lambda c: c["t_pref"])

    # --- Improvement 3: 2-opt reorder within local time windows
    chosen = _two_opt_reorder(chosen, max_passes=2, t_window=15.0)

    # --- Improvement 4: slower omega_peak inside _schedule_with_slews
    scheduled = _schedule_with_slews(chosen, omega_peak_dps=OMEGA_PEAK_DPS,
                                      T_pass=T)

    # Fallback attitude (if nothing scheduled): aim AOI centroid mid-pass
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
        "notes": (f"adaptive {n_lat}x{n_lon} grid, edge_bias={edge_bias}, "
                  f"centroid_off={centroid_off:.1f}deg, "
                  f"frames={len(shutter)}, omega_peak={OMEGA_PEAK_DPS}dps, "
                  f"2-opt slew minimisation"),
        "target_hints_llh": [{"lat_deg": s["lat"], "lon_deg": s["lon"]}
                             for s in scheduled],
    }


# ============================================================
# Inline self-test (only runs when called directly)
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
        print(f"\n=== {name} ===  (planning took {dt:.2f}s)")
        print(f"  attitude samples = {len(sched['attitude'])}, "
              f"shutters = {len(sched['shutter'])}")
        print(f"  notes: {sched['notes']}")
