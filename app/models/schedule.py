"""Schedule data model (free-form dict — kept here for future tightening)."""
from __future__ import annotations

from typing import Any, Dict


def schedule_summary(schedule: Dict[str, Any]) -> Dict[str, Any]:
    att = schedule.get("attitude", []) or []
    sh = schedule.get("shutters", []) or []
    return {
        "n_attitude_samples": len(att),
        "n_shutters": len(sh),
        "t_first": att[0]["t"] if att else None,
        "t_last": att[-1]["t"] if att else None,
    }
