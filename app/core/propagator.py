"""SGP4 orbit propagation and ephemeris generation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import math
import numpy as np
from sgp4.api import Satrec, jday

from .frames import eci_to_geodetic, teme_to_j2000


@dataclass
class EphemerisPoint:
    t_offset_s: float
    jd: float  # full Julian date (jd + fr)
    r_eci: np.ndarray  # km
    v_eci: np.ndarray  # km/s
    lat_deg: float
    lon_deg: float
    alt_km: float

    def to_dict(self) -> dict:
        return {
            "t": float(self.t_offset_s),
            "jd": float(self.jd),
            "r_eci_km": [float(x) for x in self.r_eci],
            "v_eci_kms": [float(x) for x in self.v_eci],
            "lat_deg": float(self.lat_deg),
            "lon_deg": float(self.lon_deg),
            "alt_km": float(self.alt_km),
        }


def _parse_iso(t_iso: str) -> datetime:
    s = t_iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_jd(dt: datetime) -> tuple[float, float]:
    """Return (jd, fr) tuple from sgp4.api.jday."""
    seconds = dt.second + dt.microsecond * 1e-6
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds)


def propagate_pass(
    tle1: str,
    tle2: str,
    t_start_utc: str,
    t_end_utc: str,
    dt: float = 1.0,
) -> List[EphemerisPoint]:
    """Propagate the orbit from t_start to t_end at dt-second intervals.

    Times are UTC ISO-8601 strings. Output positions are in J2000/ECI (km).
    """
    sat = Satrec.twoline2rv(tle1, tle2)
    start = _parse_iso(t_start_utc)
    end = _parse_iso(t_end_utc)
    duration_s = (end - start).total_seconds()
    n_steps = int(math.floor(duration_s / dt)) + 1

    jd0, fr0 = datetime_to_jd(start)

    points: List[EphemerisPoint] = []
    for i in range(n_steps):
        t_off = i * dt
        # Add t_off seconds to start
        fr = fr0 + t_off / 86400.0
        jd = jd0
        # Let sgp4 normalize the fraction
        e, r_teme, v_teme = sat.sgp4(jd, fr)
        if e != 0:
            # Skip bad samples but continue
            continue
        jd_full = jd + fr
        r_eci, v_eci = teme_to_j2000(np.asarray(r_teme), np.asarray(v_teme), jd_full)
        lat, lon, alt = eci_to_geodetic(r_eci, jd_full)
        points.append(
            EphemerisPoint(
                t_offset_s=t_off,
                jd=jd_full,
                r_eci=r_eci,
                v_eci=v_eci,
                lat_deg=math.degrees(lat),
                lon_deg=math.degrees(lon),
                alt_km=alt,
            )
        )
    return points
