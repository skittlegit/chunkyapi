"""End-to-end API smoke test."""
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_cases():
    r = client.get("/api/cases")
    assert r.status_code == 200
    assert len(r.json()["cases"]) >= 3


def test_plan_validate_simulate_case1():
    r = client.post("/api/plan", json={"case_id": "case1"})
    assert r.status_code == 200, r.text
    body = r.json()
    schedule = body["schedule"]
    assert len(schedule["attitude"]) > 0

    r2 = client.post("/api/validate", json={"schedule": schedule})
    assert r2.status_code == 200
    # Validation may surface minor warnings; assert no critical violations
    violations = r2.json()["violations"]
    critical = [v for v in violations if v["code"] in {"ATT_FORMAT", "QUAT_NORM", "TIME_NOT_MONOTONIC"}]
    assert critical == []

    r3 = client.post("/api/simulate", json={"schedule": schedule, "case_id": "case1"})
    assert r3.status_code == 200
    s = r3.json()
    assert "score" in s
