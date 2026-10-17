import numpy as np
import pytest

from physics import v2ch01_coulomb as e1
from physics import v2ch02_field as e2
from physics import v2ch03_gauss as e3
from physics import v2ch04_potential as e4
from physics import v2ch05_capacitance as e5
from physics import v2ch06_current as e6
from physics import v2ch07_circuits as e7


def test_coulomb_force_from_gradient():
    assert e1.force_numeric() == pytest.approx(e1.force_analytic(), rel=1e-5)
    assert e1.newton_third_law_residual() < 1e-6


def test_equilibrium_points():
    assert e1.zero_force_point_numeric() == pytest.approx(e1.zero_force_point_analytic(), abs=1e-9)
    assert e1.zero_force_point_analytic() == pytest.approx(0.4)
    assert e1.opposite_charges_point_numeric() == pytest.approx(e1.opposite_charges_point_analytic(), rel=1e-8)


def test_dipole_and_lines():
    z = 0.5
    assert e2.dipole_field_axis_far(z) == pytest.approx(float(e2.dipole_field_axis_exact(z)), rel=2e-3)
    ex, ey = e2.field_from_potential_numeric(0.3, 0.2), e2.field_dipole_vector(0.3, 0.2)
    assert ex == pytest.approx(ey, rel=1e-4)
    pts, closed = e2.trace_field_line(100.0)
    assert closed


def test_ring_field():
    for z in (0.02, 0.07, 0.3):
        assert e2.ring_axis_field_numeric(z) == pytest.approx(float(e2.ring_axis_field_analytic(z)), rel=1e-9)
    assert e2.ring_max_field_position_numeric() == pytest.approx(e2.ring_max_field_position_analytic(), rel=2e-3)


def test_gauss_flux():
    assert e3.flux_sphere_numeric() == pytest.approx(e3.flux_sphere_analytic(), rel=1e-4)
    assert e3.flux_sphere_numeric(charge_pos=(0.1, 0.05, 0.0)) == pytest.approx(e3.flux_sphere_analytic(), rel=1e-3)
    assert e3.flux_cube_numeric() == pytest.approx(e3.flux_sphere_analytic(), rel=1e-4)
    assert abs(e3.flux_sphere_numeric(radius=0.1, charge_pos=(0.3, 0.0, 0.0))) < 1e-3 * e3.flux_sphere_analytic()


def test_sphere_field_monte_carlo():
    for r in (0.05, 0.2):
        mc, sig = e3.sphere_field_monte_carlo(r)
        assert mc == pytest.approx(float(e3.sphere_field_analytic(r)), abs=5 * sig + 1e-9)


def test_potential_disk_and_pair():
    assert e4.disk_potential_numeric(0.05) == pytest.approx(float(e4.disk_potential_analytic(0.05)), rel=1e-9)
    fv = e4.field_from_potential_numeric(lambda z: float(e4.disk_potential_analytic(z)), 0.05)
    assert fv == pytest.approx(float(e4.disk_field_analytic(0.05)), rel=1e-6)
    assert e4.field_grid_error() < 5e-3
    assert e4.interaction_energy() < 0


def test_capacitance():
    num = e5.coax_capacitance_numeric(n=121, iterations=2500)
    assert num == pytest.approx(e5.coax_capacitance_per_length_analytic(), rel=0.06)
    nodal = e5.network_nodal_numeric()
    assert nodal["c_eq"] == pytest.approx(e5.network_ceq_analytic())
    d = e5.dielectric_energies()
    assert d["u_const_v"] == pytest.approx(4 * d["u0"]) and d["u_const_q"] == pytest.approx(d["u0"] / 4)


def test_drude_and_wire():
    mu, sigma_mu, _ = e6.drift_mobility_monte_carlo()
    assert mu == pytest.approx(e6.drift_mobility_analytic(), abs=5 * sigma_mu)
    assert sigma_mu < 0.03 * mu
    assert e6.tapered_resistance_numeric() == pytest.approx(e6.tapered_resistance_analytic(), rel=1e-9)
    assert e6.tapered_resistance_analytic() < e6.uniform_resistance()


def test_circuits():
    i_n, i_a = e7.mesh_currents_numeric(), e7.mesh_currents_analytic()
    assert i_n == pytest.approx(i_a)
    res = e7.kirchhoff_residuals(i_n)
    assert all(abs(v) < 1e-12 for v in res.values())
    t, y = e7.rc_numeric()
    assert np.max(np.abs(y[:, 0] - e7.q_analytic(t))) < 1e-9
    en = e7.rc_energies_numeric()
    assert en["dissipated"] == pytest.approx(en["capacitor"], rel=1e-4)
    assert en["battery"] == pytest.approx(e7.C * e7.EPS**2, rel=1e-4)
    assert e7.time_to_fraction_numeric() == pytest.approx(-e7.R * e7.C * np.log(1 - 0.632), rel=1e-4)
