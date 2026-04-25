"""Loader for built-in test cases."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

DATA_PATH = Path(__file__).parent / "test_cases.json"


@lru_cache(maxsize=1)
def load_cases() -> List[Dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def get_case(case_id: str) -> Dict[str, Any] | None:
    for c in load_cases():
        if c["case_id"] == case_id:
            return c
    return None
