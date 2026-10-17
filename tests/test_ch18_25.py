import numpy as np
import pytest

from physics import ch18_oscillations as c18
from physics import ch19_waves as c19
from physics import ch20_sound as c20
from physics import ch21_relativity as c21
from physics import ch22_temperature as c22
from physics import ch23_first_law as c23
from physics import ch24_kinetic_theory as c24
from physics import ch25_second_law as c25


def test_shm():
    t, y = c18.shm_numeric()
    assert np.max(np.abs(y[:, 0] - c18.x_analytic(t))) < 1e-9
    assert c18.period_numeric() == pytest.approx(2 * np.pi / c18.omega0(), rel=1e-6)
    for row in c18.pendulum_table():
        assert row["numeric"] == pytest.approx(row["exact"], rel=1e-5)
    assert c18.pendulum_period_exact(60) > c18.pendulum_period_small()


def test_resonance():
    ws = np.array([6.0, 9.0, 9.9, 12.0])
    num = c18.resonance_curve_numeric(ws)
    assert num == pytest.approx(c18.driven_amplitude_analytic(ws), rel=2e-3)
    grid = np.linspace(8, 11, 61)
    assert grid[np.argmax(c18.driven_amplitude_analytic(grid))] == pytest.approx(c18.resonance_frequency_analytic(), abs=0.03)


def test_wave_fdtd_matches_dalembert():
    assert c19.fdtd_error(courant=1.0) < 1e-8
    assert c19.fdtd_error(courant=0.9) < 5e-3
    assert c19.fdtd_error(courant=0.5, n=1600) < c19.fdtd_error(courant=0.9, n=800) * 1.5


def test_string_modes():
    freqs, x, shapes = c19.modes_numeric()
    for n in range(1, 6):
        assert freqs[n - 1] == pytest.approx(c19.mode_frequency_analytic(n), rel=2e-4)
    assert np.max(np.abs(shapes[0] - np.sin(np.pi * x / c19.STRING_L))) < 1e-3


def test_doppler_and_beats():
    assert c20.doppler_numeric() == pytest.approx(c20.doppler_analytic(), rel=1e-9)
    assert c20.doppler_numeric(vs=0.0, vo=20.0) == pytest.approx(c20.doppler_analytic(vs=0.0, vo=20.0), rel=1e-9)
    assert c20.beat_frequency_numeric() == pytest.approx(4.0, abs=0.05)
    assert c20.beat_envelope_frequency_numeric() == pytest.approx(4.0, rel=0.05)


def test_pipes():
    for kind in ("open", "closed"):
        assert c20.pipe_frequencies_numeric(kind) == pytest.approx(c20.pipe_frequencies_analytic(kind), rel=2e-3)


def test_muons():
    p, err = c21.survival_monte_carlo()
    assert p == pytest.approx(c21.survival_analytic(), abs=4 * err)
    assert c21.survival_analytic(relativistic=False) < 1e-5


def test_velocity_addition_and_energy():
    assert c21.velocity_addition_numeric(0.8, 0.8) == pytest.approx(c21.velocity_addition_analytic(0.8, 0.8))
    assert c21.rapidity_addition(0.8, 0.8) == pytest.approx(0.9756097560975611)
    for beta in (0.1, 0.5, 0.9):
        assert c21.kinetic_numeric(beta) == pytest.approx(c21.kinetic_analytic(beta), rel=1e-9)
    assert c21.kinetic_classical(0.9) / c21.kinetic_analytic(0.9) < 0.5


def test_thermal_expansion_and_equilibrium():
    assert c22.length_numeric() == pytest.approx(c22.length_analytic(), rel=1e-12)
    assert c22.length_analytic() - c22.L0 > c22.length_linear() - c22.L0
    t, y = c22.temperatures_numeric()
    ta, tb = c22.temperatures_analytic(t)
    assert np.max(np.abs(y[:, 0] - ta)) < 1e-6
    assert y[-1, 0] == pytest.approx(float(ta[-1]), abs=1e-6)
    t2, y2 = c22.temperatures_numeric(t_end=200.0, n=4000)
    assert y2[-1, 0] == pytest.approx(c22.final_temperature_analytic(), abs=1e-3)
    assert c22.to_kelvin(0) == 273.15 and c22.to_fahrenheit(100) == 212


def test_gas_processes_and_otto():
    for kind in ("isothermal", "isobaric", "adiabatic"):
        assert c23.work_numeric(kind) == pytest.approx(c23.work_analytic(kind), rel=1e-6)
    for row in c23.first_law_table():
        assert row["Q"] == pytest.approx(row["dU"] + row["W"])
    assert abs(c23.first_law_table()[2]["Q"]) < 1e-3  # adiabatica
    o = c23.otto_numeric()
    assert o["eta"] == pytest.approx(c23.otto_efficiency_analytic(), rel=1e-6)


def test_kinetic_theory():
    p_sim, t_sim = c24.pressure_simulation(n=2000, duration=1.0e-8)
    assert p_sim == pytest.approx(c24.pressure_analytic(2000, t_sim), rel=0.1)
    sp, an = c24.speeds_sampled(), c24.speeds_analytic()
    assert sp["v_mean"] == pytest.approx(an["v_mean"], rel=3e-3)
    assert sp["v_rms"] == pytest.approx(an["v_rms"], rel=3e-3)
    q = c24.moments_by_quadrature()
    assert q["norm"] == pytest.approx(1.0, rel=1e-6)
    assert q["v_mean"] == pytest.approx(an["v_mean"], rel=1e-6)


def test_carnot_and_entropy():
    c = c25.carnot_numeric()
    assert c["eta"] == pytest.approx(c25.carnot_efficiency_analytic(), rel=1e-6)
    assert c["pa_end"] == pytest.approx(c["pa"], rel=1e-6)
    assert c25.heating_entropy_numeric() == pytest.approx(c25.heating_entropy_analytic(), rel=1e-9)
    assert c25.contact_entropy_numeric() == pytest.approx(c25.contact_entropy_analytic(), rel=1e-4)
    assert c25.contact_entropy_analytic() > 0
