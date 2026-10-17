import numpy as np
import pytest

from physics import v2ch08_magnetic_field as m8
from physics import v2ch09_biot_savart as m9
from physics import v2ch10_faraday as m10
from physics import v2ch11_inductance as m11
from physics import v2ch12_magnetic_matter as m12
from physics import v2ch13_ac as m13
from physics import v2ch14_maxwell as m14


def test_helix():
    t, y = m8.helix_rk4()
    assert m8.measured_radius(y) == pytest.approx(m8.radius_analytic(), rel=2e-3)
    assert m8.measured_period(t, y) == pytest.approx(m8.period_analytic(), rel=5e-3)
    assert y[-1, 2] == pytest.approx(m8.V_PAR * t[-1], rel=1e-9)
    tb, yb = m8.helix_boris(steps_per_turn=20)
    assert m8.speed_drift(yb) < 1e-12


def test_hall():
    assert m8.hall_voltage_numeric() == pytest.approx(m8.hall_voltage_analytic(), rel=1e-4)
    assert m8.carrier_density_from_hall(m8.hall_voltage_numeric()) == pytest.approx(m8.N_CU, rel=1e-4)


def test_biot_savart_loop_wire_solenoid():
    for z in (0.0, 0.03, 0.1):
        assert m9.loop_axis_numeric(z) == pytest.approx(float(m9.loop_axis_analytic(z)), rel=1e-4)
    assert m9.wire_field_numeric() == pytest.approx(m9.wire_field_analytic(), rel=1e-4)
    assert m9.wire_field_analytic() < m9.infinite_wire()
    assert m9.solenoid_axis_numeric(0.0, seg=120) == pytest.approx(float(m9.solenoid_axis_exact(0.0)), rel=1e-3)
    assert float(m9.solenoid_axis_exact(0.0)) == pytest.approx(m9.solenoid_center_infinite(), rel=5e-3)


def test_faraday():
    t = np.linspace(0, 0.02, 200)
    assert m10.emf_numeric(t) == pytest.approx(m10.emf_analytic(t), abs=1e-3)
    assert m10.emf_rms_numeric() == pytest.approx(m10.emf_peak_analytic() / np.sqrt(2), rel=1e-6)
    tt, y = m10.rod_numeric()
    assert np.max(np.abs(y[:, 1] - m10.rod_velocity_analytic(tt))) < 1e-8
    assert m10.rod_energy_dissipated_numeric() == pytest.approx(0.5 * m10.M_ROD * m10.V0_ROD**2, rel=1e-5)
    assert y[-1, 0] == pytest.approx(m10.rod_distance_analytic() * (1 - np.exp(-12 / m10.tau_rod())), rel=1e-8)


def test_inductance():
    t, y = m11.rl_numeric()
    assert np.max(np.abs(y[:, 0] - m11.i_growth_analytic(t))) < 1e-9
    assert m11.time_to_half_numeric() == pytest.approx(m11.tau_rl() * np.log(2), rel=1e-6)
    assert m11.energy_integral_numeric() == pytest.approx(m11.energy_magnetic(), rel=1e-3)
    t, y = m11.rlc_numeric()
    assert np.max(np.abs(y[:, 0] - m11.q_analytic(t))) < 1e-9
    assert m11.period_numeric() == pytest.approx(2 * np.pi / m11.omega_d(), rel=1e-5)


def test_magnetic_matter():
    for a in (10, 60, 90):
        assert m12.period_numeric(a) == pytest.approx(m12.period_exact(a), rel=1e-4)
    assert m12.energy_conservation_drift() < 1e-6
    for x in (0.1, 1.0, 2.0):
        mm, err = m12.magnetization_metropolis(x)
        assert mm == pytest.approx(float(np.tanh(x)), abs=5 * err + 1e-3)


def test_ac():
    for w in (200.0, 316.2, 500.0):
        amp, phi = m13.steady_state_numeric(w)
        assert amp == pytest.approx(float(m13.current_amplitude_analytic(w)), rel=2e-3)
        assert phi == pytest.approx(float(m13.phase_analytic(w)), abs=2e-3)
    assert m13.rms_numeric(170.0) == pytest.approx(170 / np.sqrt(2), rel=1e-6)
    assert m13.power_average_numeric() == pytest.approx(m13.power_average_analytic(), rel=1e-6)
    tr = m13.transformer_numeric()
    an = m13.transformer_analytic()
    assert tr["vs"] == pytest.approx(an["vs"], rel=1e-6) and tr["is"] == pytest.approx(an["is"], rel=1e-6)
    assert tr["p_s"] == pytest.approx(tr["p_p"], rel=1e-9)


def test_maxwell():
    assert m14.c_from_constants() == pytest.approx(299792458.0, rel=1e-8)
    assert m14.speed_numeric(0.5) == pytest.approx(299792458.0, rel=3e-3)
    assert m14.speed_numeric(0.99) == pytest.approx(299792458.0, rel=1e-3)
    assert m14.intensity_numeric() == pytest.approx(m14.intensity_analytic(), rel=1e-6)
    assert m14.displacement_current_numeric() == pytest.approx(m14.conduction_current_analytic(), rel=1e-6)
