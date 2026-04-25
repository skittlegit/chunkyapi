"""Request models."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    case_id: Optional[str] = Field(None, description="Built-in case id, e.g. 'case1'")
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None
    aoi_polygon: Optional[List[List[float]]] = Field(
        None, description="List of [lat_deg, lon_deg] pairs"
    )
    pass_start_utc: Optional[str] = None
    pass_end_utc: Optional[str] = None
    strategy: str = "boustrophedon"
    settle_margin_s: float = 3.0
    off_nadir_margin_deg: float = 5.0


class ValidateRequest(BaseModel):
    schedule: dict


class SimulateRequest(BaseModel):
    schedule: dict
    aoi_polygon: Optional[List[List[float]]] = None
    case_id: Optional[str] = None
