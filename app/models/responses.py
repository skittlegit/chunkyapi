"""Pydantic response models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class PlanResponse(BaseModel):
    schedule: Dict[str, Any]
    diagnostics: Dict[str, Any]
    ephemeris_summary: Dict[str, Any]
    tiles: List[Dict[str, Any]]
    footprints: List[List[List[float]]]


class ValidationViolation(BaseModel):
    code: str
    message: str
    location: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None


class ValidateResponse(BaseModel):
    ok: bool
    violations: List[ValidationViolation]


class SimulateResponse(BaseModel):
    score: float
    coverage: float
    eta_E: float
    eta_T: float
    Q_smear: float
    delta_h_used_nms: float
    t_active_s: float
    n_shutters: int
    body_rates_deg_per_s: List[float]
    diagnostics: Dict[str, Any]
