"""Volumen 2, capitulo 25. Particulas elementales y cosmologia."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch25_particles as m25

CT = "Partículas elementales y cosmología"


class TwoBodyDecay(PhysicsScene):
    volume = 2
    chapter_number = 25
    chapter_title = CT

    def construct(self):
        lo, hi = m25.lab_energy_range_analytic()
        e = m25.lab_energy_samples()
        hist, edges = np.histogram(e, bins=20, range=(lo, hi))
        centers = 0.5 * (edges[1:] + edges[:-1])
        dens = hist / hist.sum() / (edges[1] - edges[0])
        mean, std, pmax, pmin = m25.decay_monte_carlo()
        level = 1 / (hi - lo)
        curve_scene(
            self,
            found=[r"M^2 = (E_1 + E_2)^2 - |\vec p_1 + \vec p_2|^2,\qquad p^* = \dfrac{\sqrt{[M^2-(m_1+m_2)^2][M^2-(m_1-m_2)^2]}}{2M}",
                   r"E_{lab} = \gamma(E^* + \beta p^*\cos\theta^*)\ \Rightarrow\ \text{distribución uniforme en } E_{lab}"],
            note="Con decaimiento isótropo, la energía de la hija en el laboratorio se reparte uniformemente entre dos extremos.",
            fcap="Cinemática de decaimientos y masa invariante.",
            problem=["Un Λ (1115.683 MeV) con p = 2000 MeV/c decae en p + π⁻. Halle p* y el rango de energía del protón en el laboratorio.",
                     "Verifique con una simulación de 200 000 decaimientos isótropos y reconstruya la masa invariante."],
            pcap="Unidades c = 1; boost a lo largo de z.",
            analytic=[(rf"p^* = {m25.p_star_analytic():.4f}\ \mathrm{{MeV}}/c", ACCENT), rf"E_p \in [{lo:.2f},\,{hi:.2f}]\ \mathrm{{MeV}},\qquad f(E) = \dfrac{{1}}{{E_{{max}}-E_{{min}}}}"],
            acap="La distribución uniforme en E es una consecuencia de la isotropía en el sistema en reposo.",
            numeric="Bisección de E₁(p) + E₂(p) = M para p*; Monte Carlo con boost de Lorentz y reconstrucción de la masa invariante.",
            xr=(lo, hi), yr=(0, 2.2 * level * 1e3), xticks=(round(lo), round((lo + hi) / 2), round(hi)), yticks=(0, 1, 2), xlabel="E del protón [MeV]", ylabel="densidad [10⁻³/MeV]",
            curves=[(np.array([lo, hi]), np.array([level, level]) * 1e3, ANALYTIC)], dots=(centers, dens * 1e3), legend=[(ANALYTIC, "uniforme"), (NUMERIC, "Monte Carlo")],
            table=(["magnitud", "numérico", "analítico"],
                   [("p* [MeV/c]", f"{m25.p_star_numeric():.6f}", f"{m25.p_star_analytic():.6f}"), ("M reconstruida", f"{mean:.6f}", f"{m25.M_PARENT:.6f}"), ("σ(M) [MeV]", f"{std:.1e}", "0"),
                    ("p máx. lab", f"{pmax:.2f}", f"{m25.lab_momentum_limits_analytic()[0]:.2f}")]),
            final="La masa invariante se conserva evento por evento y las energías llenan el rango esperado.", fsize=28, asize=30)


class FriedmannAge(PhysicsScene):
    volume = 2
    chapter_number = 25
    chapter_title = CT

    def construct(self):
        t = np.linspace(0.05, 25, 200)
        tt, y = m25.scale_factor_numeric()
        age_a, age_n = m25.age_analytic(), m25.age_numeric()
        a_end = float(m25.scale_factor_analytic(25.0))
        curve_scene(
            self,
            found=[r"\left(\dfrac{\dot a}{a}\right)^2 = H_0^2\left(\dfrac{\Omega_m}{a^3} + \Omega_\Lambda\right)\qquad(\text{Friedmann, universo plano})",
                   r"a(t) = \left(\dfrac{\Omega_m}{\Omega_\Lambda}\right)^{1/3}\sinh^{2/3}\!\left(\dfrac{3}{2}\sqrt{\Omega_\Lambda}H_0t\right)"],
            note="La expansión desacelera mientras domina la materia y se acelera cuando domina la energía oscura.",
            fcap="Cosmología de Friedmann.",
            problem=["Un universo plano con Ω_m = 0.31, Ω_Λ = 0.69 y H₀ = 0.0703 Gyr⁻¹ (≈ 68.7 km/s/Mpc).",
                     "Halle su edad y el factor de escala a(t)."],
            pcap="Se normaliza a(hoy) = 1.",
            analytic=[(rf"t_0 = \dfrac{{2}}{{3H_0\sqrt{{\Omega_\Lambda}}}}\,\mathrm{{arcsinh}}\sqrt{{\Omega_\Lambda/\Omega_m}} = {age_a:.4f}\ \mathrm{{Gyr}}", ACCENT)],
            acap="El resultado supera a 2/(3H₀) = 9.5 Gyr del modelo de solo materia.",
            numeric="Simpson de la integral t₀ = ∫ da/(aH) con a = u² para eliminar la singularidad; RK4 de da/dt = H₀ a √(Ω_m/a³ + Ω_Λ).",
            xr=(0, 25), yr=(0, 2.6), xticks=(0, 5, 10, 15, 20, 25), yticks=(0, 1, 2), xlabel="t [Gyr]", ylabel="a(t)",
            curves=[(t, m25.scale_factor_analytic(t), ANALYTIC)], dots=(tt[::250], y[::250, 0]), legend=[(ANALYTIC, "a(t) analítico"), (NUMERIC, "RK4")],
            table=(["magnitud", "numérico", "analítico"], [("t₀ [Gyr]", f"{age_n:.6f}", f"{age_a:.6f}"), ("a(25 Gyr)", f"{y[-1, 0]:.6f}", f"{a_end:.6f}")]),
            final="Las dos integrales coinciden; la edad resulta ≈ 13.6 mil millones de años.", fsize=30)
