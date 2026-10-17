"""Volumen 2, capitulo 10. Ley de induccion de Faraday."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch10_faraday as m10


class RotatingLoop(PhysicsScene):
    volume = 2
    chapter_number = 10
    chapter_title = "Ley de inducción de Faraday"

    def construct(self):
        t = np.linspace(0, 0.04, 300)
        tn = np.linspace(0, 0.04, 33)
        e0 = m10.emf_peak_analytic()
        curve_scene(
            self,
            found=[r"\varepsilon = -\dfrac{d\Phi_B}{dt},\qquad \Phi_B = NBA\cos\omega t", r"\varepsilon(t) = NBA\omega\,\sin\omega t,\qquad \varepsilon_{rms} = \varepsilon_0/\sqrt{2}"],
            note="La ley de Lenz fija el signo: la fem se opone a la variación del flujo.",
            fcap="Ley de Faraday y generador.",
            problem=["Una bobina de 100 vueltas y 0.020 m² gira a 50 Hz en B = 0.50 T.",
                     "Halle la fem inducida, su valor máximo y su valor eficaz."],
            pcap="El eje de giro es perpendicular al campo.",
            analytic=[(rf"\varepsilon_0 = NBA\omega = {e0:.4f}\ \mathrm{{V}}", ACCENT), rf"\varepsilon_{{rms}} = {e0 / np.sqrt(2):.4f}\ \mathrm{{V}}"],
            acap="El máximo ocurre cuando el plano es paralelo al campo.",
            numeric="La fem se obtiene derivando numéricamente el flujo (diferencias centrales); el valor eficaz con Simpson.",
            xr=(0, 0.04), yr=(-350, 350), xticks=(0, 0.02, 0.04), yticks=(-300, 0, 300), xlabel="t [s]", ylabel="ε [V]",
            curves=[(t, m10.emf_analytic(t), ANALYTIC)], dots=(tn, m10.emf_numeric(tn)),
            legend=[(ANALYTIC, "NBAω sen ωt"), (NUMERIC, "−dΦ/dt")],
            table=(["magnitud", "numérico", "analítico"],
                   [("ε(5 ms) [V]", f"{float(m10.emf_numeric(np.array([0.005]))[0]):.5f}", f"{float(m10.emf_analytic(0.005)):.5f}"),
                    ("ε_rms [V]", f"{m10.emf_rms_numeric():.5f}", f"{e0 / np.sqrt(2):.5f}")]),
            final="La derivada numérica del flujo reproduce la fem sinusoidal.", fsize=36)


class SlidingRod(PhysicsScene):
    volume = 2
    chapter_number = 10
    chapter_title = "Ley de inducción de Faraday"

    def construct(self):
        t, y = m10.rod_numeric()
        tt = np.linspace(0, 12, 200)
        tau = m10.tau_rod()
        curve_scene(
            self,
            found=[r"\varepsilon = B\ell v,\qquad I = \dfrac{B\ell v}{R},\qquad F = -\dfrac{B^2\ell^2}{R}\,v", r"v(t) = v_0\,e^{-t/\tau},\qquad \tau = \dfrac{mR}{B^2\ell^2}"],
            note="La energía cinética de la barra se disipa íntegramente por efecto Joule en el circuito.",
            fcap="Fem de movimiento.",
            problem=["Una barra de 0.20 kg y 0.50 m desliza sin fricción sobre rieles unidos por R = 2.0 Ω en B = 0.80 T, con v₀ = 4.0 m/s.",
                     "Halle v(t), la distancia recorrida y la energía disipada."],
            pcap="No hay fuerzas externas salvo la magnética.",
            analytic=[(rf"\tau = {tau:.3f}\ \mathrm{{s}},\qquad x_\infty = v_0\tau = {m10.rod_distance_analytic():.4f}\ \mathrm{{m}}", ACCENT),
                      rf"E_{{Joule}} = \tfrac12 m v_0^2 = {0.5 * m10.M_ROD * m10.V0_ROD**2:.4f}\ \mathrm{{J}}"],
            acap="Decaimiento exponencial de la velocidad.",
            numeric="RK4 de la ecuación de movimiento acoplada; la energía disipada se integra como ∫ I²R dt con Simpson.",
            xr=(0, 12), yr=(0, 4.5), xticks=(0, 4, 8, 12), yticks=(0, 2, 4), xlabel="t [s]", ylabel="v [m/s]",
            curves=[(tt, m10.rod_velocity_analytic(tt), ANALYTIC)], dots=(t[::100], y[::100, 1]),
            legend=[(ANALYTIC, "v₀ e^(−t/τ)"), (NUMERIC, "RK4")],
            table=(["magnitud", "numérico", "analítico"],
                   [("v(4 s) [m/s]", f"{float(np.interp(4.0, t, y[:, 1])):.5f}", f"{float(m10.rod_velocity_analytic(4.0)):.5f}"),
                    ("x(12 s) [m]", f"{y[-1, 0]:.5f}", f"{m10.rod_distance_analytic() * (1 - np.exp(-12 / tau)):.5f}"),
                    ("E Joule [J]", f"{m10.rod_energy_dissipated_numeric():.5f}", f"{0.5 * m10.M_ROD * m10.V0_ROD**2:.5f}")]),
            final="La energía disipada coincide con la cinética inicial.", fsize=34)
