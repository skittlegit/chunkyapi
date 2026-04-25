"""Propagator + scorer sanity tests."""
from app.core.propagator import propagate_pass
from app.core.scorer import compute_score, compute_smear_quality
from app.data import get_case


def test_propagate_case1_altitude():
    c = get_case("case1")
    eph = propagate_pass(
        c["tle_line1"],
        c["tle_line2"],
        c["pass_start_utc"],
        c["pass_end_utc"],
        dt=10.0,
    )
    assert len(eph) > 50
    alts = [p.alt_km for p in eph]
    assert min(alts) > 400 and max(alts) < 700


def test_score_formula():
    s = compute_score(0.8, 0.6, 0.9, 1.0)
    assert abs(s - 0.8 * (1 + 0.25 * 0.6 + 0.10 * 0.9)) < 1e-12


def test_smear_quality_all_pass():
    assert compute_smear_quality([0.01, 0.02, 0.03]) == 1.0
    assert compute_smear_quality([0.01, 0.10, 0.03]) == 2 / 3
