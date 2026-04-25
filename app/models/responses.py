"""Response models."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class PlanResponse(BaseModel):
    schedule: Dict[str, Any]
    diagnostics: Dict[str, Any]
    ephemeris_summary: Dict[str, Any]


class ValidationViolation(BaseModel):
    code: str
    message: str
    location: str | None = None


class ValidateResponse(BaseModel):
    ok: bool
    violations: List[ValidationViolation]


class SimulateResponse(BaseModel):
    score: float
    coverage: float
    eta_E: float
    eta_T: float
    Q_smear: float
    telemetry: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
