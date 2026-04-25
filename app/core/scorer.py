"""Coverage scoring and S_orbit computation."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from ..config import settings
from .tiling import point_in_polygon, polygon_bbox, polygon_area_km2

LatLon = Tuple[float, float]


def compute_coverage(
    aoi_polygon: Sequence[LatLon],
    footprints: Sequence[Sequence[LatLon]],
    grid_n: int = 60,
) -> float:
    """Fraction of AOI area covered by union of footprints (grid sampling)."""
    if not footprints:
        return 0.0
    lat_min, lon_min, lat_max, lon_max = polygon_bbox(aoi_polygon)
    lats = np.linspace(lat_min, lat_max, grid_n)
    lons = np.linspace(lon_min, lon_max, grid_n)
    inside = 0
    covered = 0
    for la in lats:
        for lo in lons:
            if not point_in_polygon((la, lo), aoi_polygon):
                continue
            inside += 1
            for fp in footprints:
                if len(fp) >= 3 and point_in_polygon((la, lo), fp):
                    covered += 1
                    break
    if inside == 0:
        return 0.0
    return covered / inside


def compute_effort_efficiency(delta_h_used_nms: float, budget_nms: float | None = None) -> float:
    if budget_nms is None:
        budget_nms = settings.delta_h_budget_nms
    return max(0.0, 1.0 - delta_h_used_nms / budget_nms)


def compute_time_efficiency(t_active_s: float, t_pass_s: float | None = None) -> float:
    if t_pass_s is None:
        t_pass_s = settings.pass_window_s
    return max(0.0, 1.0 - t_active_s / t_pass_s)


def compute_smear_quality(body_rates_dps: Sequence[float], limit_dps: float | None = None) -> float:
    if not body_rates_dps:
        return 0.0
    if limit_dps is None:
        limit_dps = settings.body_rate_limit_dps
    n_pass = sum(1 for r in body_rates_dps if r <= limit_dps)
    return n_pass / len(body_rates_dps)


def compute_score(C: float, eta_E: float, eta_T: float, Q_smear: float) -> float:
    return C * (1.0 + 0.25 * eta_E + 0.10 * eta_T) * Q_smear
