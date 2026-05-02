"""Coordinate frame tests."""
import math

import numpy as np
import pytest

from app.core.frames import (
    ecef_to_geodetic,
    geodetic_to_ecef,
    eci_to_ecef,
    ecef_to_eci,
    gmst_rad,
)


def test_geodetic_roundtrip():
    cases = [(0.0, 0.0, 0.0), (45.0, 10.0, 0.5), (-30.0, -120.0, 5.0), (89.0, 50.0, 0.0)]
    for lat_d, lon_d, alt in cases:
        r = geodetic_to_ecef(math.radians(lat_d), math.radians(lon_d), alt)
        lat, lon, a = ecef_to_geodetic(r)
        assert math.degrees(lat) == pytest.approx(lat_d, abs=1e-6)
        assert math.cos(math.radians(lon_d) - lon) == pytest.approx(1.0, abs=1e-9)
        assert a == pytest.approx(alt, abs=1e-6)


def test_eci_ecef_roundtrip():
    r = np.array([7000.0, 1000.0, 500.0])
    jd = 2461000.5
    r2 = ecef_to_eci(eci_to_ecef(r, jd), jd)
    assert np.allclose(r, r2, atol=1e-9)


def test_gmst_monotonic():
    g0 = gmst_rad(2451545.0)
    g1 = gmst_rad(2451545.0 + 1.0 / 86400.0)
    assert g1 > g0