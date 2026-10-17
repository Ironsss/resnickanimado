import numpy as np
import pytest

from physics import ch06_dynamics as c6


def test_critical_angle():
    assert c6.critical_angle_numeric() == pytest.approx(c6.critical_angle_analytic(), abs=0.05)


def test_incline_sliding_acceleration():
    t, s, v = c6.simulate_incline(c6.THETA_DEG, 2.0, dt=1e-3)
    a = c6.incline_acceleration_analytic(c6.THETA_DEG)
    assert s[-1] == pytest.approx(0.5 * a * 4.0, rel=1e-6)
    assert v[-1] == pytest.approx(a * 2.0, rel=1e-6)


def test_block_stays_below_critical_angle():
    _, s, _ = c6.simulate_incline(20.0, 2.0)
    assert s[-1] == 0.0


def test_drag_terminal_and_convergence():
    assert c6.drag_velocity_analytic(50.0) == pytest.approx(c6.terminal_speed(), rel=1e-12)
    rows = c6.drag_convergence()
    assert all(r["err_rk4"] < 1e-4 for r in rows)
    assert rows[0]["err_euler"] / rows[-1]["err_euler"] == pytest.approx(16, rel=0.25)
    assert c6.time_to_fraction_numeric() == pytest.approx(c6.time_to_fraction_analytic(), rel=1e-6)


from physics import ch07_work_energy as c7
from physics import ch08_energy_conservation as c8
from physics import ch09_particle_systems as c9
from physics import ch10_collisions as c10


def test_spring_work_quadratures():
    ex = c7.spring_work_analytic()
    assert ex == pytest.approx(8.0)
    assert c7.spring_work_numeric(2, "simpson") == pytest.approx(ex, rel=1e-12)  # exacto para polinomio grado 1
    rows = c7.work_convergence()
    assert rows[-1]["riemann"] < rows[0]["riemann"]
    assert rows[0]["riemann"] / rows[-1]["riemann"] == pytest.approx(16, rel=0.1)
    assert rows[-1]["trapezoid"] < 1e-9


def test_work_energy_theorem():
    assert c7.final_speed_numeric() == pytest.approx(c7.final_speed_analytic(), rel=1e-4)
    assert c7.work_integral_numeric() == pytest.approx(30.0, rel=1e-12)


def test_pendulum_energy():
    assert c8.max_speed_numeric() == pytest.approx(c8.bottom_speed_analytic(), rel=1e-6)
    assert c8.energy_drift("rk4") < 1e-6
    assert c8.energy_drift("euler") > 100 * c8.energy_drift("rk4")
    assert c8.energy_drift("verlet") < c8.energy_drift("euler")


def test_spring_friction_turning_points():
    an = c8.turning_points_analytic()
    _, _, _, turn = c8.spring_friction_numeric()
    assert len(turn) == len(an)
    assert turn == pytest.approx(an, abs=2e-4)


def test_center_of_mass():
    assert c9.cm_discrete_numeric() == pytest.approx(c9.cm_discrete_analytic())
    assert c9.rod_cm_numeric() == pytest.approx(c9.rod_cm_analytic(), rel=1e-9)
    assert c9.rod_cm_monte_carlo()[0] == pytest.approx(c9.rod_cm_analytic(), rel=2e-3)


def test_explosion_cm_follows_parabola():
    an, nu = c9.fragments_analytic(), c9.fragments_numeric()
    assert nu["xb"] == pytest.approx(an["xb"], rel=1e-5)
    assert nu["x_cm"] == pytest.approx(c9.range_analytic(), rel=1e-5)


def test_collision_restitution():
    for zeta in (0.0, 0.3, 0.7):
        e = c10.restitution_from_damping(zeta)
        assert c10.restitution_measured(zeta) == pytest.approx(e, abs=2e-3)
    v1p, v2p = c10.outcome_analytic(1.0)
    assert (v1p, v2p) == pytest.approx((1.0, 4.0))
    assert c10.outcome_analytic(0.0)[0] == pytest.approx(2.0)


def test_ballistic_pendulum():
    an, nu = c10.bullet_block_analytic(), c10.bullet_block_numeric()
    assert nu["h"] == pytest.approx(an["h"], rel=1e-5)
    assert nu["loss_fraction"] == pytest.approx(an["loss_fraction"], rel=1e-9)
    assert an["v_from_h"] == pytest.approx(c10.BULLET_V)
