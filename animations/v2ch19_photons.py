"""Volumen 2, capitulo 19. Fotones y naturaleza ondulatoria de la luz."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch19_photons as m19

CT = "Fotones y naturaleza ondulatoria de la luz"


class BlackbodySpectrum(PhysicsScene):
    volume = 2
    chapter_number = 19
    chapter_title = CT

    def construct(self):
        lam = np.linspace(100e-9, 2500e-9, 400)
        temps = (4000.0, 5800.0)
        t = 5800.0
        ln = m19.wien_peak_numeric(t)
        curve_scene(
            self,
            found=[r"u(\lambda,T) = \dfrac{2\pi hc^2}{\lambda^5}\,\dfrac{1}{e^{hc/\lambda kT}-1}\qquad(\text{Planck})", r"\lambda_{max}T = b,\qquad P = \sigma A T^4\quad(\text{Wien y Stefan-Boltzmann})"],
            note="La cuantización de la energía de los osciladores elimina la divergencia ultravioleta de la teoría clásica.",
            fcap="Radiación de cuerpo negro.",
            problem=["Un cuerpo negro a T = 5800 K (superficie solar) emite según la ley de Planck.",
                     "Halle la longitud de onda del máximo y la potencia emitida por unidad de área."],
            pcap="La integral se calcula sobre todo el espectro.",
            analytic=[(rf"\lambda_{{max}} = \dfrac{{b}}{{T}} = {m19.wien_peak_analytic(t) * 1e9:.4f}\ \mathrm{{nm}}", ACCENT), rf"\dfrac{{P}}{{A}} = \sigma T^4 = {m19.stefan_analytic(t) / 1e6:.4f}\ \mathrm{{MW/m^2}}"],
            acap="El máximo cae en el verde, y la potencia crece como T⁴.",
            numeric="El máximo se busca por sección áurea sobre la función de Planck; la potencia se integra con Simpson (20 000 subintervalos) desde 0.02 λ_max hasta 200 λ_max.",
            xr=(0, 2500), yr=(0, 90), xticks=(0, 500, 1000, 1500, 2000, 2500), yticks=(0, 30, 60, 90), xlabel="λ [nm]", ylabel="u [kW/m²/nm]",
            curves=[(lam * 1e9, m19.planck(lam, temps[0]) * 1e-9 / 1e3, ERROR), (lam * 1e9, m19.planck(lam, temps[1]) * 1e-9 / 1e3, ANALYTIC)],
            dots=([ln * 1e9], [float(m19.planck(ln, t)) * 1e-9 / 1e3]), legend=[(ANALYTIC, "5800 K"), (ERROR, "4000 K"), (NUMERIC, "máximo (sección áurea)")],
            table=(["magnitud", "numérico", "analítico"],
                   [("λ_max [nm]", f"{ln * 1e9:.5f}", f"{m19.wien_peak_analytic(t) * 1e9:.5f}"), ("P/A [MW/m²]", f"{m19.stefan_numeric(t) / 1e6:.5f}", f"{m19.stefan_analytic(t) / 1e6:.5f}"),
                    ("λ_max 3000 K [nm]", f"{m19.wien_peak_numeric(3000.0) * 1e9:.5f}", f"{m19.wien_peak_analytic(3000.0) * 1e9:.5f}")]),
            final="Wien y Stefan-Boltzmann son consecuencias de la distribución de Planck.", fsize=32, asize=34)


class ComptonPhotoelectric(PhysicsScene):
    volume = 2
    chapter_number = 19
    chapter_title = CT

    def construct(self):
        th = np.linspace(0, np.pi, 100)
        tn = np.linspace(0, np.pi, 13)
        h, phi, dh, dphi = m19.millikan_fit()
        curve_scene(
            self,
            found=[r"\lambda' - \lambda = \dfrac{h}{m_ec}\,(1-\cos\theta)\qquad(\text{Compton})", r"eV_0 = hf - \phi\qquad(\text{Einstein, efecto fotoeléctrico})"],
            note="El efecto Compton confirma que el fotón porta momento p = h/λ; el fotoeléctrico, que porta energía hf.",
            fcap="Efecto Compton y efecto fotoeléctrico.",
            problem=["Rayos X de 71 pm se dispersan por electrones libres. Halle el corrimiento de longitud de onda en función del ángulo.",
                     "Además, a partir de potenciales de frenado del sodio, determine h y la función de trabajo."],
            pcap="Los datos fotoeléctricos son sintéticos, con 10 mV de ruido gaussiano.",
            analytic=[(rf"\lambda_C = h/m_ec = {m19.LAMBDA_C * 1e12:.5f}\ \mathrm{{pm}},\qquad \Delta\lambda(90^\circ) = {float(m19.compton_shift_analytic(np.pi / 2)) * 1e12:.5f}\ \mathrm{{pm}}", ACCENT),
                      rf"h = {m19.H_PLANCK:.6e}\ \mathrm{{J\,s}},\qquad \phi = {m19.PHI_EV:.2f}\ \mathrm{{eV}}"],
            acap="El corrimiento es independiente de la longitud de onda incidente.",
            numeric="Conservación de energía y momento relativistas: bisección de la energía del fotón dispersado. Ajuste lineal por mínimos cuadrados de V₀(f) para h y φ.",
            xr=(0, 180), yr=(0, 5), xticks=(0, 60, 120, 180), yticks=(0, 1, 2, 3, 4, 5), xlabel="θ [°]", ylabel="Δλ [pm]",
            curves=[(np.degrees(th), m19.compton_shift_analytic(th) * 1e12, ANALYTIC)], dots=(np.degrees(tn), [m19.compton_shift_numeric(float(x)) * 1e12 for x in tn]),
            legend=[(ANALYTIC, "λ_C (1 − cos θ)"), (NUMERIC, "conservación 4-momento")],
            table=(["magnitud", "numérico", "analítico"],
                   [("Δλ(90°) [pm]", f"{m19.compton_shift_numeric(np.pi / 2) * 1e12:.6f}", f"{float(m19.compton_shift_analytic(np.pi / 2)) * 1e12:.6f}"),
                    ("Δλ(180°) [pm]", f"{m19.compton_shift_numeric(np.pi) * 1e12:.6f}", f"{float(m19.compton_shift_analytic(np.pi)) * 1e12:.6f}"),
                    ("h [10⁻³⁴ J s]", f"{h * 1e34:.4f} ± {dh * 1e34:.4f}", f"{m19.H_PLANCK * 1e34:.4f}"), ("φ [eV]", f"{phi:.4f} ± {dphi:.4f}", f"{m19.PHI_EV:.4f}")]),
            final="El ajuste recupera h y φ dentro de su incertidumbre estadística.", fsize=32, asize=32)
