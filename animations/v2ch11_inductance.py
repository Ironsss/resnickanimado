"""Volumen 2, capitulo 11. Inductancia."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch11_inductance as m11


class RLCircuit(PhysicsScene):
    volume = 2
    chapter_number = 11
    chapter_title = "Inductancia"

    def construct(self):
        t, y = m11.rl_numeric()
        tt = np.linspace(0, 0.4, 200)
        tau = m11.tau_rl()
        curve_scene(
            self,
            found=[r"\varepsilon_L = -L\dfrac{di}{dt},\qquad U_B = \tfrac12 L i^2", r"i(t) = \dfrac{\varepsilon}{R}\left(1 - e^{-t/\tau_L}\right),\qquad \tau_L = \dfrac{L}{R}"],
            note="El inductor se opone a los cambios de corriente y almacena energía en su campo magnético.",
            fcap="Circuito RL.",
            problem=["Un inductor de 0.20 H en serie con 4.0 Ω se conecta a 12 V en t = 0.",
                     "Halle i(t), la constante de tiempo y la energía almacenada final."],
            pcap="La corriente inicial es nula.",
            analytic=[(rf"\tau_L = {tau * 1e3:.2f}\ \mathrm{{ms}},\qquad i_\infty = {m11.EPS / m11.R:.2f}\ \mathrm{{A}}", ACCENT),
                      rf"U_B = \tfrac12 L i_\infty^2 = {m11.energy_magnetic():.4f}\ \mathrm{{J}},\qquad t_{{1/2}} = \tau\ln 2 = {tau * np.log(2) * 1e3:.4f}\ \mathrm{{ms}}"],
            acap="Tras retirar la fuente, la corriente decae con la misma τ.",
            numeric="RK4 de L di/dt = ε − Ri; energía como ∫ i v_L dt (Simpson); tiempo de decaimiento por interpolación.",
            xr=(0, 0.4), yr=(0, 3.3), xticks=(0, 0.1, 0.2, 0.3, 0.4), yticks=(0, 1, 2, 3), xlabel="t [s]", ylabel="i [A]",
            curves=[(tt, m11.i_growth_analytic(tt), ANALYTIC)], dots=(t[::100], y[::100, 0]),
            legend=[(ANALYTIC, "analítico"), (NUMERIC, "RK4")],
            table=(["magnitud", "numérico", "analítico"],
                   [("t½ [ms]", f"{m11.time_to_half_numeric() * 1e3:.5f}", f"{tau * np.log(2) * 1e3:.5f}"),
                    ("U_B [J]", f"{m11.energy_integral_numeric():.5f}", f"{m11.energy_magnetic():.5f}")]),
            final="La energía integrada coincide con ½Li².", fsize=34)


class RLCOscillator(PhysicsScene):
    volume = 2
    chapter_number = 11
    chapter_title = "Inductancia"

    def construct(self):
        t, y = m11.rlc_numeric()
        tt = np.linspace(0, 0.15, 400)
        curve_scene(
            self,
            found=[r"L\ddot q + R\dot q + \dfrac{q}{C} = 0,\qquad \omega_0 = \dfrac{1}{\sqrt{LC}}", r"q(t) = Q_0 e^{-\gamma t}\left(\cos\omega_d t + \dfrac{\gamma}{\omega_d}\sin\omega_d t\right),\quad \gamma = \dfrac{R}{2L}"],
            note="La energía oscila entre el capacitor y el inductor mientras la resistencia la disipa.",
            fcap="Circuito RLC libre.",
            problem=["Un capacitor de 100 µF cargado con 1.0 mC se conecta a L = 0.10 H con R = 5.0 Ω.",
                     "Halle q(t), la frecuencia amortiguada, el factor de calidad y la energía restante tras un periodo."],
            pcap="Régimen subamortiguado.",
            analytic=[(rf"\omega_0 = {m11.omega0():.3f}\ \mathrm{{rad/s}},\qquad \omega_d = {m11.omega_d():.3f}\ \mathrm{{rad/s}}", ACCENT),
                      rf"Q = \dfrac{{\omega_0 L}}{{R}} = {m11.quality_factor():.3f},\qquad U(T)/U(0) \approx e^{{-2\gamma T}} = {np.exp(-2 * m11.gamma_() * 2 * np.pi / m11.omega_d()):.4f}"],
            acap="La amplitud decae con e^(−γt).",
            numeric="RK4 del sistema (q, i); el periodo se mide entre pasos descendentes por cero y la energía se evalúa en t = T.",
            xr=(0, 0.15), yr=(-1.2, 1.2), xticks=(0, 0.05, 0.1, 0.15), yticks=(-1, 0, 1), xlabel="t [s]", ylabel="q [mC]",
            curves=[(tt, m11.q_analytic(tt) * 1e3, ANALYTIC)], dots=(t[::150], y[::150, 0] * 1e3),
            legend=[(ANALYTIC, "analítico"), (NUMERIC, "RK4")],
            table=(["magnitud", "numérico", "analítico"],
                   [("T [ms]", f"{m11.period_numeric() * 1e3:.5f}", f"{2 * np.pi / m11.omega_d() * 1e3:.5f}"),
                    ("U(T)/U(0)", f"{m11.energy_ratio_after_period():.5f}", f"{np.exp(-2 * m11.gamma_() * 2 * np.pi / m11.omega_d()):.5f}")]),
            final="Tras un periodo la energía vale e^(−2γT) de la inicial.", fsize=34)
