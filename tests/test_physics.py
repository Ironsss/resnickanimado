import numpy as np
import pytest

from physics import ch01_measurement as c1
from physics import ch02_motion_1d as c2
from physics import ch03_vectors as c3
from physics import ch04_motion_2d as c4
from physics import ch05_newton as c5


def test_pendulum_exponents():
    e = c1.pendulum_exponents()
    assert e["L"] == pytest.approx(0.5, abs=1e-12)
    assert e["g"] == pytest.approx(-0.5, abs=1e-12)
    assert e["m"] == pytest.approx(0.0, abs=1e-12)


def test_pendulum_constant_is_two_pi():
    r = c1.pendulum_dimensionless_constant(theta0=0.01)
    assert r.k_numeric == pytest.approx(2 * np.pi, rel=1e-4)


def test_density_uncertainty_monte_carlo_matches_propagation():
    an = c1.density_analytic()
    mc = c1.density_monte_carlo()
    assert mc["std"] == pytest.approx(an["sigma_quadrature"], rel=0.02)
    assert an["sigma_quadrature"] < an["sigma_worst_case"]


def test_secant_converges_to_derivative():
    tab = c2.secant_table()
    errs = [r["error"] for r in tab]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 1e-3
    assert c2.central_second_difference(2.0, 1e-2) == pytest.approx(c2.a_analytic(2.0), abs=1e-6)


def test_area_under_v_is_displacement():
    d = c2.displacement_from_velocity(0.0, 4.0, 4000)
    assert d["trapezoid"] == pytest.approx(d["exact"], rel=1e-6)
    assert abs(d["riemann"] - d["exact"]) > abs(d["trapezoid"] - d["exact"])


def test_free_fall_rk4_exact_euler_first_order():
    rows = c2.free_fall_convergence()
    assert all(r["err_rk4"] < 1e-9 for r in rows)
    ratio = rows[0]["err_euler"] / rows[-1]["err_euler"]
    assert ratio == pytest.approx(rows[-1]["n"] / rows[0]["n"], rel=0.1)


def test_vector_resultant_analytic_equals_components():
    an, nu = c3.resultant_analytic(), c3.resultant_numeric()
    assert an["magnitude"] == pytest.approx(nu["magnitude"], abs=1e-9)
    assert an["angle_deg"] == pytest.approx(nu["angle_deg"], abs=1e-9)


def test_products_and_invariance():
    p = c3.products_analytic()
    assert p["dot_by_angle"] == pytest.approx(c3.dot())
    assert p["cross_by_angle"] == pytest.approx(c3.cross_z())
    area, _, _ = c3.parallelogram_area_monte_carlo()
    assert area == pytest.approx(abs(c3.cross_z()), rel=0.01)
    _, _, _, dots = c3.invariance_sweep()
    assert np.ptp(dots) < 1e-9


def test_projectile():
    an = c4.projectile_analytic()
    t, s = c4.projectile_numeric(dt=0.002)
    assert s[-1, 0] == pytest.approx(an["range"], rel=1e-6)
    assert t[-1] == pytest.approx(an["t_flight"], rel=1e-6)
    assert s[:, 1].max() == pytest.approx(an["height"], rel=1e-3)
    assert c4.optimal_angle_numeric() == pytest.approx(45.0, abs=0.05)


def test_circular_motion():
    ca = c4.circular_analytic()
    assert ca["a_c"] == pytest.approx(ca["a_c_from_v"])
    row = c4.fd_error_table()[-1]
    assert row["a_c"] == pytest.approx(ca["a_c"], rel=1e-3)
    assert row["speed"] == pytest.approx(ca["speed"], rel=1e-3)


def test_static_equilibrium():
    an, nu = c5.tensions_analytic(), c5.tensions_numeric()
    assert an == pytest.approx(nu)
    assert np.linalg.norm(c5.net_force(*nu)) < 1e-9


def test_pulley():
    an, nu = c5.pulley_analytic(), c5.pulley_numeric()
    assert an == pytest.approx(nu)
    t, s = c5.pulley_trajectory()
    assert s[-1, 0] == pytest.approx(0.5 * nu[0] * 3.0**2, rel=1e-9)
    assert c5.balance_angle_numeric() == pytest.approx(c5.balance_angle_analytic(), abs=1e-6)
