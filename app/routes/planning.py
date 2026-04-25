from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core.planner import plan_imaging
from ..data import get_case
from ..models.requests import PlanRequest
from ..models.responses import PlanResponse

router = APIRouter(prefix="/api", tags=["planning"])


@router.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    tle1 = req.tle_line1
    tle2 = req.tle_line2
    aoi = req.aoi_polygon
    t_start = req.pass_start_utc
    t_end = req.pass_end_utc

    if req.case_id:
        case = get_case(req.case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"Unknown case_id: {req.case_id}")
        tle1 = tle1 or case["tle_line1"]
        tle2 = tle2 or case["tle_line2"]
        aoi = aoi or case["aoi_polygon"]
        t_start = t_start or case["pass_start_utc"]
        t_end = t_end or case["pass_end_utc"]

    if not (tle1 and tle2 and aoi and t_start and t_end):
        raise HTTPException(
            status_code=400,
            detail="Must provide either case_id or full (tle_line1, tle_line2, aoi_polygon, pass_start_utc, pass_end_utc).",
        )

    aoi_tuples = [(float(p[0]), float(p[1])) for p in aoi]

    try:
        result = plan_imaging(
            tle1=tle1,
            tle2=tle2,
            aoi_polygon=aoi_tuples,
            pass_start_utc=t_start,
            pass_end_utc=t_end,
            settle_margin_s=req.settle_margin_s,
            off_nadir_margin_deg=req.off_nadir_margin_deg,
            strategy=req.strategy,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Planning failed: {e}") from e

    return PlanResponse(
        schedule=result.schedule,
        diagnostics=result.diagnostics,
        ephemeris_summary=result.ephemeris_summary,
    )
