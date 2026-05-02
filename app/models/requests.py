"""Pydantic request models."""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    case_id: Optional[str] = None
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None
    aoi_polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="List of [lat_deg, lon_deg] pairs defining the AOI polygon.",
    )
    pass_start_utc: Optional[str] = None
    pass_end_utc: Optional[str] = None
    sc_params: Optional[dict] = None
    strategy: str = "boustrophedon"
    settle_margin_s: float = 0.3
    off_nadir_margin_deg: float = 5.0


class ScheduleEnvelope(BaseModel):
    schedule: dict


class SimulateRequest(ScheduleEnvelope):
    case_id: Optional[str] = None
    aoi_polygon: Optional[List[Tuple[float, float]]] = None


class ValidateRequest(ScheduleEnvelope):
    pass
