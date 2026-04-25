"""Application configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    version: str = "0.1.0"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))
    allowed_origins: List[str] = field(
        default_factory=lambda: _split_csv(os.getenv("ALLOWED_ORIGINS"))
        or [
            "https://chunkyweb.vercel.app",
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )

    # Spacecraft defaults (from problem statement)
    fov_deg: float = 2.0
    altitude_km_nominal: float = 500.0
    off_nadir_limit_deg: float = 60.0
    off_nadir_target_deg: float = 55.0
    body_rate_limit_dps: float = 0.05  # deg/s during shutter
    integration_time_s: float = 0.120
    pass_window_s: float = 720.0

    # Inertia (kg*m^2) - small sat
    inertia_diag: tuple = (0.12, 0.12, 0.08)

    # Wheel limits
    wheel_h_max_nms: float = 0.030  # per-wheel max momentum
    wheel_h_safe_nms: float = 0.025  # safe operating limit
    delta_h_budget_nms: float = 0.200  # for scoring


settings = Settings()
