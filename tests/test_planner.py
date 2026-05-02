from app.core.planner import plan_imaging
from app.data import get_case


def test_plan_case1_runs():
    c = get_case("case1")
    assert c is not None
    res = plan_imaging(
        c["tle_line1"],
        c["tle_line2"],
        [(p[0], p[1]) for p in c["aoi_polygon"]],
        c["pass_start_utc"],
        c["pass_end_utc"],
        sc_params=c.get("sc_params"),
    )
    schedule = res["schedule"]
    assert "attitude" in schedule
    assert len(schedule["attitude"]) > 0
    for a in schedule["attitude"][:50]:
        q = a["q_BN"]
        n2 = sum(x * x for x in q)
        assert abs(n2 - 1.0) < 1e-9
    assert res["diagnostics"]["n_tiles_total"] >= 1