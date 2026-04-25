import math

import numpy as np

from app.core.attitude import (
    compute_off_nadir,
    compute_pointing_quat,
    quat_to_dcm,
    quaternion_angular_distance,
    slerp,
)


def test_pointing_quat_aligns_z_with_target():
    r_sat = np.array([6878.0, 0.0, 0.0])
    v_sat = np.array([0.0, 7.6, 0.0])
    target = np.array([6378.0, 0.0, 0.0])
    q = compute_pointing_quat(r_sat, v_sat, target)
    dcm = quat_to_dcm(q)
    z_b = dcm[:, 2]
    look = (target - r_sat) / np.linalg.norm(target - r_sat)
    assert np.allclose(z_b, look, atol=1e-9)


def test_off_nadir_zero_for_subsat():
    r_sat = np.array([6878.0, 0.0, 0.0])
    target = np.array([6378.0, 0.0, 0.0])
    assert compute_off_nadir(r_sat, target) < 1e-6


def test_slerp_endpoints():
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q2 = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    a = slerp(q1, q2, 0.0)
    b = slerp(q1, q2, 1.0)
    assert np.allclose(a, q1, atol=1e-9)
    assert np.allclose(b, q2, atol=1e-9)


def test_quat_angular_distance():
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q2 = np.array([0.0, 0.0, math.sin(math.pi / 6), math.cos(math.pi / 6)])
    ang = quaternion_angular_distance(q1, q2)
    assert abs(ang - math.pi / 3) < 1e-9
