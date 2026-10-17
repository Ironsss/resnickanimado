"""Volumen 2, capitulos 15 a 25: verificacion de cada metodo numerico contra su resultado analitico."""
import numpy as np
import pytest

from physics import v2ch15_light as m15
from physics import v2ch16_lenses as m16
from physics import v2ch17_interference as m17
from physics import v2ch18_diffraction as m18
from physics import v2ch19_photons as m19
from physics import v2ch20_matter_waves as m20
from physics import v2ch21_schrodinger as m21
from physics import v2ch22_atom as m22
from physics import v2ch23_solids as m23
from physics import v2ch24_nuclear as m24
from physics import v2ch25_particles as m25


def test_light():
    d = m15.target_point()[0]
    assert m15.fermat_numeric(d) == pytest.approx(m15.refraction_point_analytic(), rel=1e-6)
    assert m15.critical_angle_numeric() == pytest.approx(m15.critical_angle_analytic(), abs=1e-9)
    for lam in (400, 550, 700):
        n = float(m15.cauchy(lam))
        assert m15.min_deviation_numeric(n) == pytest.approx(float(m15.min_deviation_analytic(n)), abs=1e-6)
    assert m15.cauchy(400) > m15.cauchy(700)


def test_lenses():
    f = m16.focal_lensmaker()
    assert f == pytest.approx(0.1)
    assert m16.lens_image_numeric(0.3) == pytest.approx(m16.image_distance_analytic(0.3, f), rel=2e-4)
    assert m16.mirror_image_numeric(0.6, 0.4) == pytest.approx(m16.mirror_image_analytic(0.6, 0.4), rel=1e-5)
    za, ma = m16.two_lens_analytic()
    zn, mn = m16.two_lens_matrix()
    assert zn == pytest.approx(za, rel=1e-9) and mn == pytest.approx(ma, rel=1e-9)


def test_interference():
    for m in (1, 3, 5):
        assert m17.bright_fringe_numeric(m) == pytest.approx(m17.fringe_position_analytic(m), rel=1e-6)
    y = np.linspace(-0.01, 0.01, 201)
    assert np.max(np.abs(m17.intensity_numeric(y) - m17.intensity_analytic(y))) < 5e-3
    lam = np.linspace(400e-9, 700e-9, 31)
    assert np.max(np.abs(m17.reflectance_matrix(lam) - m17.reflectance_analytic(lam))) < 1e-12
    assert m17.reflectance_analytic(550e-9) < m17.bare_reflectance() / 3


def test_diffraction():
    s = np.linspace(0, 0.3, 61)
    assert np.max(np.abs(m18.single_slit_numeric(s) - m18.single_slit_analytic(s))) < 1e-6
    s2 = np.linspace(0, 0.3, 61)
    assert np.max(np.abs(m18.grating_numeric(s2) - m18.grating_analytic(s2))) < 2e-4
    _, ang = m18.principal_maxima_analytic()
    num = m18.principal_maxima_numeric()
    assert len(num) == len(ang)
    assert np.max(np.abs(num - ang)) < 0.02
    for a in (0.0, 0.4, 0.785, 1.2):
        assert m18.jones_intensity([0, a]) == pytest.approx(float(m18.malus_analytic(a)), rel=1e-12)
    assert m18.jones_intensity([0, 0.5, np.pi / 2]) == pytest.approx(m18.three_polarizers_analytic(0.5), rel=1e-12)


def test_photons():
    for t in (3000.0, 5800.0, 10000.0):
        assert m19.wien_peak_numeric(t) == pytest.approx(m19.wien_peak_analytic(t), rel=1e-7)
        assert m19.stefan_numeric(t) == pytest.approx(m19.stefan_analytic(t), rel=1e-5)
    h, phi, dh, dphi = m19.millikan_fit()
    assert abs(h - 6.62607015e-34) < 3 * dh
    assert abs(phi - m19.PHI_EV) < 3 * dphi
    for th in (np.pi / 6, np.pi / 2, np.pi):
        assert m19.compton_shift_numeric(th) == pytest.approx(float(m19.compton_shift_analytic(th)), rel=1e-6)


def test_matter_waves():
    assert m20.de_broglie(100.0) == pytest.approx(1.2264e-10, rel=1e-3)
    hits = m20.electron_hits()
    sp, _ = m20.fringe_spacing_from_hits(hits)
    assert sp == pytest.approx(m20.fringe_spacing_analytic(), rel=5e-3)
    assert m20.bragg_angle_numeric() == pytest.approx(m20.bragg_angle_analytic(), abs=1e-9)
    sx, sp_ = m20.packet_numeric(0.0)
    assert sx * sp_ == pytest.approx(m20.HBAR / 2, rel=1e-6)
    for t in (1e-16, 1e-15):
        assert m20.packet_numeric(t)[0] == pytest.approx(float(m20.packet_width_analytic(t)), rel=1e-6)


def test_schrodinger():
    e, _, _ = m21.well_numeric(4)
    assert np.allclose(e, m21.well_energy_analytic([1, 2, 3, 4]), rtol=1e-5)
    assert m21.wavefunction_overlap(1) == pytest.approx(1.0, abs=1e-6)
    for ev in (3.0, 5.0, 8.0, 12.0):
        assert m21.transmission_numeric(ev) == pytest.approx(m21.transmission_analytic(ev), rel=1e-6)


def test_atom():
    e, r, u = m22.radial_numeric(4)
    assert np.allclose(e, m22.level_analytic([1, 2, 3, 4]), rtol=1e-3)
    assert m22.balmer_wavelength_numeric(3, e) == pytest.approx(m22.balmer_wavelength_analytic(3), rel=1e-4)
    assert m22.mean_radius_1s_numeric() == pytest.approx(1.5, rel=1e-4)
    assert m22.most_probable_radius_numeric() == pytest.approx(1.0, rel=1e-3)


def test_solids():
    ana = m23.band_edges_analytic(3)
    num = m23.band_edges_numeric(3)
    assert np.allclose(ana, num, rtol=1e-2)
    for k in (0.5, 1.5, 2.5):
        assert m23.band_energy_analytic(k, 0) == pytest.approx(m23.bands_plane_waves(k)[0], rel=1e-2)
    for e in (0.3, 0.7, 2.0, 5.0):
        assert m23.diode_voltage_numeric(e) == pytest.approx(m23.diode_voltage_analytic(e), rel=1e-9)


def test_nuclear():
    assert m24.chain_numeric(20.0, 4000)[1][-1, 1] == pytest.approx(float(m24.bateman_b(20.0)), abs=1e-9)
    assert m24.peak_time_numeric() == pytest.approx(m24.peak_time_analytic(), rel=1e-6)
    mean, err = m24.mean_life_monte_carlo()
    assert abs(mean - 1 / m24.LAM_B) < 4 * err
    assert abs(m24.z_stable_numeric(56) - m24.z_stable_analytic(56)) < 1.0
    assert 50 <= m24.peak_a_numeric() <= 66


def test_particles():
    assert m25.p_star_numeric() == pytest.approx(m25.p_star_analytic(), rel=1e-12)
    mean, std, pmax, pmin = m25.decay_monte_carlo()
    assert mean == pytest.approx(m25.M_PARENT, rel=1e-12) and std < 1e-9
    hi, lo = m25.lab_momentum_limits_analytic()
    assert pmax == pytest.approx(hi, rel=1e-3) and pmin == pytest.approx(lo, rel=1e-3)
    assert m25.age_numeric() == pytest.approx(m25.age_analytic(), rel=1e-6)
    t, y = m25.scale_factor_numeric()
    assert y[-1, 0] == pytest.approx(float(m25.scale_factor_analytic(25.0)), rel=1e-4)


def test_extras():
    assert abs(m24.z_stable_numeric(56, False) - m24.z_stable_analytic(56)) < 0.6
    a = np.arange(30, 230, 20)
    assert np.max(np.abs(np.array([m24.b_per_a(int(x), False) for x in a]) - m24.b_per_a_analytic(a))) < 0.02
    lo, hi = m25.lab_energy_range_analytic()
    e = m25.lab_energy_samples()
    assert e.min() >= lo - 1e-9 and e.max() <= hi + 1e-9
    hist, _ = np.histogram(e, bins=10, range=(lo, hi))
    assert hist.std() / hist.mean() < 0.03
