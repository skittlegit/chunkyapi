"""Runtime configuration (env vars + physical constants)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _parse_origins(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Settings:
    version: str = "0.1.0"
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    allowed_origins: List[str] = field(
        default_factory=lambda: _parse_origins(
            os.getenv(
                "ALLOWED_ORIGINS",
                "https://chunkyweb.vercel.app,http://localhost:3000",
            )
        )
    )


# --- Physical / mission constants -----------------------------------------

# WGS-84
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Spacecraft (defaults — can be overridden via sc_params)
DEFAULT_INERTIA = (0.12, 0.12, 0.08)  # kg m^2 (diagonal Ix, Iy, Iz)
DEFAULT_FOV_DEG = 2.0
WHEEL_H_MAX_NMS = 0.030       # absolute hard limit per wheel
WHEEL_H_SAFE_NMS = 0.025      # safe target
SHUTTER_DURATION_S = 0.120
RATE_LIMIT_DEG_PER_S = 0.05
OFF_NADIR_HARD_LIMIT_DEG = 60.0
OFF_NADIR_SAFE_LIMIT_DEG = 55.0
PASS_DURATION_S = 720.0


settings = Settings()
