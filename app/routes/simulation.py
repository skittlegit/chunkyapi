from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from ..config import DEFAULT_INERTIA
from ..core.attitude import estimate_body_rate
from ..core.scorer import (
    compute_coverage,
    compute_effort_efficiency,
    compute_score,
    compute_smear_quality,
    compute_time_efficiency,
)
from ..data import get_case
from ..models.requests import SimulateRequest
from ..models.responses import SimulateResponse

router = APIRouter(prefix="/api", tags=["simulation"])


def _sample_attitude_at(attitude: List[Dict[str, Any]], t: float) -> np.ndarray:
    if not attitude:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if t <= float(attitude[0]["t"]):
        return np.array(attitude[0]["q_BN"], dtype=float)
    if t >= float(attitude[-1]["t"]):
        return np.array(attitude[-1]["q_BN"], dtype=float)
    lo, hi = 0, len(attitude) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(attitude[mid]["t"]) <= t:
            lo = mid
        else:
            hi = mid
    t0 = float(attitude[lo]["t"])
    t1 = float(attitude[hi]["t"])
    q0 = np.array(attitude[lo]["q_BN"], dtype=float)
    q1 = np.array(attitude[hi]["q_BN"], dtype=float)
    if t1 == t0:
        return q0
    u = (t - t0) / (t1 - t0)
    if float(np.dot(q0, q1)) < 0:
        q1 = -q1
    q = (1.0 - u) * q0 + u * q1
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else q


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    schedule = req.schedule
    if not schedule:
        raise HTTPException(400, "Empty schedule")

    aoi = req.aoi_polygon
    if aoi is None and req.case_id:
        case = get_case(req.case_id)
        if case is None:
            raise HTTPException(404, f"Unknown case_id: {req.case_id}")
        aoi = case["aoi_polygon"]
    aoi_tuples: List[Tuple[float, float]] = (
        [(float(p[0]), float(p[1])) for p in aoi] if aoi else []
    )

    attitude = schedule.get("attitude", [])
    shutters = schedule.get("shutters", [])

    # Sample body rate at multiple points within each shutter exposure so a
    # single mis-aligned waypoint cannot dominate Q_smear. We measure the rate
    # over the full shutter interval as the worst-case proxy for image smear.
    body_rates: List[float] = []
    for sh in shutters:
        t0_sh = float(sh["t_start"])
        t1_sh = float(sh["t_end"])
        n_samples = 5
        for k in range(n_samples):
            u0 = k / n_samples
            u1 = (k + 1) / n_samples
            ta = t0_sh + u0 * (t1_sh - t0_sh)
            tb = t0_sh + u1 * (t1_sh - t0_sh)
            q_a = _sample_attitude_at(attitude, ta)
            q_b = _sample_attitude_at(attitude, tb)
            body_rates.append(estimate_body_rate(q_a, q_b, max(tb - ta, 1e-6)))

    # Time efficiency uses the active integration time (sum of shutter
    # durations), not the whole imaging window. Matches the hackathon scorer.
    t_active = sum(float(sh["t_end"]) - float(sh["t_start"]) for sh in shutters)

    inertia = schedule.get("meta", {}).get("inertia") or DEFAULT_INERTIA
    Ix = float(inertia[0])
    delta_h = 0.0
    prev_q, prev_t = None, None
    for a in attitude:
        t = float(a["t"])
        q = np.array(a["q_BN"], dtype=float)
        if prev_q is not None and prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                rate = estimate_body_rate(prev_q, q, dt)
                delta_h += Ix * math.radians(rate) * dt
        prev_q, prev_t = q, t

    fp_polys: List[List[Tuple[float, float]]] = []
    for sh in shutters:
        fp = sh.get("footprint") or []
        if fp:
            fp_polys.append([(float(c[0]), float(c[1])) for c in fp])

    if aoi_tuples and fp_polys:
        coverage = compute_coverage(aoi_tuples, fp_polys)
    else:
        coverage = 0.0

    eta_E = compute_effort_efficiency(delta_h)
    eta_T = compute_time_efficiency(t_active)
    Q_smear = compute_smear_quality(body_rates) if body_rates else (1.0 if not shutters else 0.0)
    score = compute_score(coverage, eta_E, eta_T, Q_smear)

    return SimulateResponse(
        score=score,
        coverage=coverage,
        eta_E=eta_E,
        eta_T=eta_T,
        Q_smear=Q_smear,
        delta_h_used_nms=delta_h,
        t_active_s=t_active,
        n_shutters=len(shutters),
        body_rates_deg_per_s=body_rates,
        diagnostics={"n_footprints_used": len(fp_polys), "aoi_provided": bool(aoi_tuples)},
    )
