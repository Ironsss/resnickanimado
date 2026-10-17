import numpy as np
import pytest

from physics import ch11_rotational_kinematics as c11
from physics import ch12_rotational_dynamics as c12
from physics import ch13_angular_momentum as c13
from physics import ch14_equilibrium as c14
from physics import ch15_gravitation as c15
from physics import ch16_fluid_statics as c16
from physics import ch17_fluid_dynamics as c17


def test_spin_up_kinematics():
    t, w, th = c11.spin_up_numeric()
    assert w[-1] == pytest.approx(c11.omega_analytic(8.0), rel=1e-12)
    assert th[-1] == pytest.approx(c11.theta_analytic(8.0), rel=1e-4)
    t2, w2, th2 = c11.spin_up_numeric_variable()
    assert th2[-1] == pytest.approx(8.0**3 / 24, rel=1e-5)


def test_cycloid():
    assert c11.arch_length_numeric() == pytest.approx(c11.arch_length_analytic(), rel=1e-5)
    assert c11.arch_length_polyline() == pytest.approx(c11.arch_length_analytic(), rel=1e-5)
    assert c11.arch_area_numeric() == pytest.approx(c11.arch_area_analytic(), rel=1e-8)


def test_inertia():
    for axis in ("end", "center"):
        assert c12.rod_inertia_numeric(axis) == pytest.approx(c12.rod_inertia_analytic(axis), rel=1e-9)
    assert c12.disc_inertia_numeric() == pytest.approx(c12.disc_inertia_analytic(), rel=1e-9)
    assert c12.parallel_axis(c12.rod_inertia_analytic("center"), c12.M_ROD, c12.L_ROD / 2) == pytest.approx(c12.rod_inertia_analytic("end"))


def test_hanging_mass_disc():
    an, nu = c12.hanging_analytic(), c12.hanging_numeric()
    assert nu["a"] == pytest.approx(an["a"], rel=1e-12)
    assert nu["T"] == pytest.approx(an["T"], rel=1e-12)
    assert nu["energy_error"] < 1e-6


def test_angular_momentum():
    t, w, inertia, ang, k = c13.skater_numeric()
    assert w[-1] == pytest.approx(c13.omega_final_analytic(), rel=1e-6)
    assert np.ptp(ang) < 1e-6
    assert k[-1] - k[0] == pytest.approx(c13.kinetic_energy_change_analytic(), rel=1e-6)


def test_gyroscope():
    _, _, _, wp = c13.precession_numeric()
    assert wp == pytest.approx(c13.precession_analytic(), rel=1e-3)
    assert c13.precession_numeric(w_spin=400.0)[3] == pytest.approx(c13.precession_analytic(400.0), rel=1e-3)


def test_ladder():
    for d in (0.0, 2.5, 5.0):
        an, nu = c14.ladder_forces_analytic(d), c14.ladder_forces_numeric(d)
        assert nu["f"] == pytest.approx(an["f"])
        assert nu["N_wall"] == pytest.approx(an["N_wall"])
    assert c14.d_max_numeric() == pytest.approx(c14.d_max_analytic(), abs=1e-9)
    assert c14.min_angle_numeric() == pytest.approx(c14.min_angle_analytic(), abs=1e-6)


def test_beam_pivot_independence():
    an = c14.beam_reactions_analytic(2.0)
    for pivot in (0.0, 1.0, 6.0):
        assert c14.beam_reactions_numeric(2.0, pivot) == pytest.approx(an)


def test_kepler():
    assert c15.period_numeric() == pytest.approx(c15.period_analytic(), rel=1e-5)
    t, y = c15.orbit_numeric()
    assert np.ptp(c15.areal_velocity(t, y)) / abs(c15.areal_velocity(t, y)[0]) < 1e-6
    e = c15.energy(y)
    assert np.ptp(e) / abs(e[0]) < 1e-6
    for row in c15.kepler_third_law_table():
        assert row["T_nu"] == pytest.approx(row["T_an"], rel=1e-5)


def test_escape_velocity():
    assert c15.rmax_numeric(8000.0) == pytest.approx(c15.rmax_analytic(8000.0), rel=1e-4)
    assert c15.escape_speed_numeric() == pytest.approx(c15.escape_speed_analytic(), rel=2e-4)


def test_pressure():
    h, p = c16.atmosphere_numeric()
    assert p == pytest.approx(c16.atmosphere_analytic(h), rel=1e-9)
    assert c16.water_pressure(100.0) == pytest.approx(1.0823e6, rel=1e-3)


def test_floating_cylinder():
    assert c16.equilibrium_bisection() == pytest.approx(c16.submerged_depth_analytic(), abs=1e-12)
    assert c16.period_numeric() == pytest.approx(2 * np.pi / c16.omega_analytic(), rel=1e-4)
    t, s = c16.bobbing_numeric(damping=8.0, t_end=6.0)
    assert s[-1, 0] == pytest.approx(c16.submerged_depth_analytic(), abs=2e-3)


def test_venturi_and_drain():
    an = c17.venturi_analytic()
    a1, a2 = c17.areas()
    assert an["v1"] * a1 == pytest.approx(an["v2"] * a2)
    dp = c17.P1 - an["p2"]
    assert c17.flow_from_pressure_drop_numeric(dp) == pytest.approx(c17.Q, rel=1e-9)
    assert c17.flow_from_pressure_drop_analytic(dp) == pytest.approx(c17.Q, rel=1e-9)
    assert c17.drain_time_numeric() == pytest.approx(c17.drain_time_analytic(), rel=2e-3)
    t, y = c17.height_numeric(2000)
    mid = len(t) // 2
    assert y[mid, 0] == pytest.approx(float(c17.height_analytic(t[mid])), rel=1e-5)
