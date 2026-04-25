from __future__ import annotations

import math
from typing import List

import numpy as np
from fastapi import APIRouter

from ..config import SHUTTER_DURATION_S
from ..models.requests import ValidateRequest
from ..models.responses import ValidateResponse, ValidationViolation

router = APIRouter(prefix="/api", tags=["validation"])


@router.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    s = req.schedule
    violations: List[ValidationViolation] = []

    attitude = s.get("attitude", [])
    shutters = s.get("shutters", [])

    if not attitude:
        violations.append(
            ValidationViolation(code="EMPTY_ATTITUDE", message="No attitude samples")
        )
        return ValidateResponse(ok=False, violations=violations)

    prev_t = -math.inf
    for i, a in enumerate(attitude):
        try:
            t = float(a["t"])
            q = a["q_BN"]
            if len(q) != 4:
                raise ValueError("q_BN length != 4")
            qn = float(np.linalg.norm(q))
        except Exception as e:  # noqa: BLE001
            violations.append(
                ValidationViolation(
                    code="ATT_FORMAT",
                    message=f"Bad attitude entry: {e}",
                    location=f"attitude[{i}]",
                )
            )
            continue
        if abs(qn - 1.0) > 1e-6:
            violations.append(
                ValidationViolation(
                    code="QUAT_NORM",
                    message=f"|q|={qn} (expected 1.0+/-1e-6)",
                    location=f"attitude[{i}]",
                )
            )
        if t < prev_t:
            violations.append(
                ValidationViolation(
                    code="TIME_NOT_MONOTONIC",
                    message=f"t={t} < previous {prev_t}",
                    location=f"attitude[{i}]",
                )
            )
        prev_t = t

    if attitude and float(attitude[0]["t"]) > 1e-6:
        violations.append(
            ValidationViolation(
                code="FIRST_SAMPLE_NOT_ZERO",
                message=f"First attitude at t={attitude[0]['t']} (expected 0)",
            )
        )

    for i in range(1, len(attitude)):
        dt = float(attitude[i]["t"]) - float(attitude[i - 1]["t"])
        if 0.0 < dt < 0.019:
            violations.append(
                ValidationViolation(
                    code="SAMPLE_SPACING_TOO_TIGHT",
                    message=f"dt={dt}s < 20ms",
                    location=f"attitude[{i}]",
                )
            )
            break

    prev_end = -math.inf
    for i, sh in enumerate(shutters):
        try:
            t0 = float(sh["t_start"])
            t1 = float(sh["t_end"])
            dur = t1 - t0
        except Exception as e:  # noqa: BLE001
            violations.append(
                ValidationViolation(
                    code="SHUTTER_FORMAT",
                    message=f"Bad shutter entry: {e}",
                    location=f"shutters[{i}]",
                )
            )
            continue
        if abs(dur - SHUTTER_DURATION_S) > 1e-6:
            violations.append(
                ValidationViolation(
                    code="SHUTTER_DURATION",
                    message=f"duration={dur} (expected {SHUTTER_DURATION_S})",
                    location=f"shutters[{i}]",
                )
            )
        if t0 < prev_end:
            violations.append(
                ValidationViolation(
                    code="SHUTTER_OVERLAP",
                    message=f"shutter starts at {t0} but previous ended at {prev_end}",
                    location=f"shutters[{i}]",
                )
            )
        prev_end = t1

    if shutters:
        final_end = float(shutters[-1]["t_end"])
        last_att_t = float(attitude[-1]["t"])
        if last_att_t < final_end - 1e-6:
            violations.append(
                ValidationViolation(
                    code="ATTITUDE_TOO_SHORT",
                    message=f"Last attitude t={last_att_t} < final shutter end {final_end}",
                )
            )

    return ValidateResponse(ok=len(violations) == 0, violations=violations)
