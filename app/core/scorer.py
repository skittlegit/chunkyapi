"""Coverage scoring."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .imaging import _equirect_xy, sutherland_hodgman_clip
from .tiling import _signed_area


def compute_coverage(
    aoi_polygon: List[Tuple[float, float]],
    footprints: List[List[Tuple[float, float]]],
    grid_n: int = 80,
) -> float:
    """Fraction of AOI area covered by the union of footprints.

    Uses a fine equirectangular sample grid for robustness with arbitrarily
    overlapping footprints.
    """
    if not footprints or not aoi_polygon:
        return 0.0

    from .tiling import point_in_polygon

    lats = [p[0] for p in aoi_polygon]
    lons = [p[1] for p in aoi_polygon]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    n_in = 0
    n_cov = 0
    for i in range(grid_n):
        lat = lat_min + (lat_max - lat_min) * (i + 0.5) / grid_n
        for j in range(grid_n):
            lon = lon_min + (lon_max - lon_min) * (j + 0.5) / grid_n
            if not point_in_polygon(lat, lon, aoi_polygon):
                continue
            n_in += 1
            for fp in footprints:
                if len(fp) >= 3 and point_in_polygon(lat, lon, fp):
                    n_cov += 1
                    break
    if n_in == 0:
        return 0.0
    return n_cov / n_in


def compute_effort_efficiency(
    delta_h_used_nms: float, budget_nms: float = 0.200
) -> float:
    return max(0.0, 1.0 - delta_h_used_nms / budget_nms)


def compute_time_efficiency(t_active_s: float, t_pass_s: float = 720.0) -> float:
    return max(0.0, 1.0 - t_active_s / t_pass_s)


def compute_smear_quality(
    body_rates_deg_per_s: List[float], limit_deg_per_s: float = 0.05
) -> float:
    if not body_rates_deg_per_s:
        return 1.0
    n = len(body_rates_deg_per_s)
    n_ok = sum(1 for r in body_rates_deg_per_s if r <= limit_deg_per_s)
    return n_ok / n


def compute_score(C: float, eta_E: float, eta_T: float, Q_smear: float) -> float:
    return C * (1.0 + 0.25 * eta_E + 0.10 * eta_T) * Q_smear
