"""Reverify case-3 minimum off-nadir using Backlogged's exact off_nadir formula
to rule out any subtle difference from the official harness.

Pass + AOI from the hackathon case3.json are hard-coded below to match what we
verified earlier. If the user can supply the harness again, this script can be
re-pointed at it; for now it matches the official figures.
"""
import math, sys
from datetime import datetime, timedelta, timezone
import numpy as np
from sgp4.api import Satrec, jday

# Case 3 inputs (from official configs/case3.json verified earlier).
TLE1 = "1 99997U 26003A   26113.72916667  .00000000  00000-0  00000-0 0  9992"
TLE2 = "2 99997  97.4000 142.5000 0001000  90.0000 315.5000 15.21920000000017"
PASS_START = "2026-04-23T17:24:00Z"
PASS_END   = "2026-04-23T17:36:00Z"
AOI = [(44.55, 9.37), (44.55, 10.63), (45.45, 10.63), (45.45, 9.37), (44.55, 9.37)]

# Backlogged's exact constants.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def gmst(dt):
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
    T = ((jd - 2451545.0) + fr) / 36525.0
    g = (67310.54841 + (876600*3600 + 8640184.812866)*T + 0.093104*T*T - 6.2e-6*T*T*T) % 86400
    if g < 0: g += 86400
    return math.radians(g / 240.0)

def llh_to_ecef(lat_deg, lon_deg, alt=0.0):
    la = math.radians(lat_deg); lo = math.radians(lon_deg)
    sl = math.sin(la); cl = math.cos(la); ss = math.sin(lo); cs = math.cos(lo)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * sl*sl)
    return np.array([(n+alt)*cl*cs, (n+alt)*cl*ss, (n*(1-WGS84_E2)+alt)*sl])

def ecef_to_eci(r_ecef, g):
    c, s = math.cos(g), math.sin(g)
    return np.array([c*r_ecef[0]-s*r_ecef[1], s*r_ecef[0]+c*r_ecef[1], r_ecef[2]])

def off_nadir(r_sat, r_tgt):
    los = r_tgt - r_sat; los /= np.linalg.norm(los)
    nad = -r_sat / np.linalg.norm(r_sat)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(los, nad))))))


sat = Satrec.twoline2rv(TLE1, TLE2)
t0 = parse_iso(PASS_START)
T = (parse_iso(PASS_END) - t0).total_seconds()

# Sub-sat track
print("Sub-sat track (deg):")
for ts in [0, 100, 200, 300, 400, 500, 600, 720]:
    when = t0 + timedelta(seconds=ts)
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute, when.second + when.microsecond*1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e: print(f"  t={ts}: SGP4 err={e}"); continue
    r = np.array(r)*1000
    g = gmst(when)
    # ECI -> ECEF
    c, s = math.cos(-g), math.sin(-g)
    r_ecef = np.array([c*r[0]-s*r[1], s*r[0]+c*r[1], r[2]])
    # ECEF -> LLH (Bowring approximation)
    x,y,z = r_ecef
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p*(1-WGS84_E2))
    for _ in range(5):
        sl = math.sin(lat)
        n = WGS84_A / math.sqrt(1 - WGS84_E2*sl*sl)
        h = p/math.cos(lat) - n
        lat = math.atan2(z, p*(1 - WGS84_E2*n/(n+h)))
    print(f"  t={ts:4d}  sub-sat=({math.degrees(lat):.3f}, {math.degrees(lon):.3f})  alt={h/1000:.1f}km")

# Min off-nadir scan over (t, lat, lon) in AOI
best = (999, 0, 0, 0)
for ts in np.arange(0, T+1, 5.0):
    when = t0 + timedelta(seconds=float(ts))
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute, when.second + when.microsecond*1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e: continue
    r = np.array(r)*1000
    g = gmst(when)
    for la in np.arange(44.55, 45.46, 0.05):
        for lo in np.arange(9.37, 10.64, 0.05):
            r_tgt = ecef_to_eci(llh_to_ecef(la, lo), g)
            off = off_nadir(r, r_tgt)
            if off < best[0]:
                best = (off, ts, la, lo)
print(f"\nMin off-nadir into AOI: {best[0]:.4f} deg at t={best[1]:.1f}s tgt=({best[2]:.3f},{best[3]:.3f})")
print(f"Gate is 60 deg.  Gap = {best[0]-60:+.4f}")
print(f"\n=> Backlogged thresholds {{55,57,58,59}}: {'reachable' if best[0] <= 59 else 'UNREACHABLE -> 0 shutters'}")
