from __future__ import annotations

import math
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ..core.attitude import compute_off_nadir
from ..core.frames import geodetic_to_eci
from ..core.propagator import propagate_pass
from ..data import get_case, list_cases

router = APIRouter(prefix="/api", tags=["cases"])


@router.get("/cases")
def list_cases_route() -> Dict[str, Any]:
    return {"cases": list_cases()}


@router.get("/cases/{case_id}")
def get_case_route(case_id: str) -> Dict[str, Any]:
    c = get_case(case_id)
    if c is None:
        raise HTTPException(404, f"Unknown case_id: {case_id}")
    return c


@router.get("/cases/{case_id}/ephemeris")
def get_case_ephemeris(case_id: str, dt_s: float = 1.0) -> Dict[str, Any]:
    c = get_case(case_id)
    if c is None:
        raise HTTPException(404, f"Unknown case_id: {case_id}")
    eph = propagate_pass(
        c["tle_line1"],
        c["tle_line2"],
        c["pass_start_utc"],
        c["pass_end_utc"],
        dt=dt_s,
    )
    aoi = c["aoi_polygon"]
    lat_c = sum(p[0] for p in aoi) / len(aoi)
    lon_c = sum(p[1] for p in aoi) / len(aoi)
    samples = []
    for ep in eph:
        target = geodetic_to_eci(math.radians(lat_c), math.radians(lon_c), 0.0, ep.jd)
        off = compute_off_nadir(ep.r_eci, target)
        samples.append(
            {
                "t": ep.t_offset_s,
                "lat_deg": ep.lat_deg,
                "lon_deg": ep.lon_deg,
                "alt_km": ep.alt_km,
                "r_eci_km": list(ep.r_eci),
                "v_eci_kms": list(ep.v_eci),
                "off_nadir_to_aoi_center_deg": off,
            }
        )
    return {"case_id": case_id, "aoi_center": [lat_c, lon_c], "samples": samples}
