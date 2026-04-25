"""SGP4 orbit propagation + ephemeris generation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple

import math
import numpy as np
from sgp4.api import Satrec, WGS84, jday

from .frames import teme_to_j2000, eci_to_geodetic


@dataclass
class EphemerisPoint:
    t_offset_s: float
    jd: float
    r_eci: Tuple[float, float, float]      # km
    v_eci: Tuple[float, float, float]      # km/s
    lat_deg: float
    lon_deg: float
    alt_km: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["r_eci"] = list(self.r_eci)
        d["v_eci"] = list(self.v_eci)
        return d


def parse_iso_utc(ts: str) -> datetime:
    """Parse ISO-8601 string ('Z' or offset) into a tz-aware UTC datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_jd(dt: datetime) -> Tuple[float, float]:
    """Returns (jd, fr) suitable for sgp4.api.Satrec.sgp4()."""
    return jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond * 1e-6,
    )


def propagate_pass(
    tle_line1: str,
    tle_line2: str,
    t_start_utc: str,
    t_end_utc: str,
    dt: float = 1.0,
) -> List[EphemerisPoint]:
    """Propagate a satellite over a pass window at `dt` second cadence."""
    sat = Satrec.twoline2rv(tle_line1, tle_line2, WGS84)
    t0 = parse_iso_utc(t_start_utc)
    t1 = parse_iso_utc(t_end_utc)
    duration = (t1 - t0).total_seconds()
    n_steps = int(round(duration / dt)) + 1

    jd0, fr0 = datetime_to_jd(t0)

    points: List[EphemerisPoint] = []
    for k in range(n_steps):
        offset_s = k * dt
        # Add seconds to (jd, fr) -- keep precision in fr.
        fr = fr0 + offset_s / 86400.0
        # Normalize so that fr stays in [0, 1)
        extra_days = math.floor(fr)
        jd = jd0 + extra_days
        fr = fr - extra_days

        e, r_teme, v_teme = sat.sgp4(jd, fr)
        if e != 0:
            # SGP4 error code; still record but mark with NaNs.
            r_eci = (math.nan, math.nan, math.nan)
            v_eci = (math.nan, math.nan, math.nan)
            lat_deg = lon_deg = alt_km = math.nan
        else:
            jd_full = jd + fr
            r_j, v_j = teme_to_j2000(np.array(r_teme), np.array(v_teme), jd_full)
            lat, lon, alt = eci_to_geodetic(r_j, jd_full)
            r_eci = (float(r_j[0]), float(r_j[1]), float(r_j[2]))
            v_eci = (float(v_j[0]), float(v_j[1]), float(v_j[2]))
            lat_deg = math.degrees(lat)
            lon_deg = math.degrees(lon)
            alt_km = float(alt)

        points.append(
            EphemerisPoint(
                t_offset_s=offset_s,
                jd=jd + fr,
                r_eci=r_eci,
                v_eci=v_eci,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                alt_km=alt_km,
            )
        )
    return points


def ephemeris_arrays(points: List[EphemerisPoint]):
    """Convenience: stack ephemeris into numpy arrays for vectorized work."""
    n = len(points)
    t = np.empty(n)
    jd = np.empty(n)
    r = np.empty((n, 3))
    v = np.empty((n, 3))
    lla = np.empty((n, 3))
    for i, p in enumerate(points):
        t[i] = p.t_offset_s
        jd[i] = p.jd
        r[i] = p.r_eci
        v[i] = p.v_eci
        lla[i] = (p.lat_deg, p.lon_deg, p.alt_km)
    return t, jd, r, v, lla
