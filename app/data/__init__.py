"""Loader for built-in test cases."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).parent / "test_cases.json"


@lru_cache(maxsize=1)
def _load() -> Dict[str, Dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["id"]: c for c in data["cases"]}


def list_cases() -> List[Dict[str, Any]]:
    return list(_load().values())


def load_cases() -> List[Dict[str, Any]]:
    return list_cases()


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    return _load().get(case_id)
