"""Volumen 2, capitulo 13. Circuitos de corriente alterna."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch13_ac as m13


class SeriesRLC(PhysicsScene):
    volume = 2
    chapter_number = 13
    chapter_title = "Circuitos de corriente alterna"

    def construct(self):
        w = np.linspace(500, 1500, 300)
        wn = np.array([600, 750, 850, 920, 970, 1000, 1030, 1080, 1150, 1250, 1400], dtype=float)
        pk = m13.current_amplitude_analytic(m13.omega0())
        an = np.array([m13.steady_state_numeric(float(x)) for x in wn])
        a0 = m13.steady_state_numeric(m13.omega0())
        curve_scene(
            self,
            found=[r"Z = \sqrt{R^2 + (\omega L - 1/\omega C)^2},\qquad I_0 = \dfrac{V_0}{Z},\qquad \tan\phi = \dfrac{\omega L - 1/\omega C}{R}",
                   r"\omega_0 = \dfrac{1}{\sqrt{LC}},\qquad Q = \dfrac{\omega_0 L}{R}"],
            note="En resonancia la impedancia es mínima (Z = R) y la corriente está en fase con la tensión.",
            fcap="Impedancia y resonancia.",
            problem=["Un circuito RLC en serie (R = 20 Ω, L = 0.10 H, C = 10 µF) se alimenta con V₀ = 50 V de frecuencia variable.",
                     "Halle la frecuencia de resonancia, el factor de calidad y la curva I₀(ω)."],
            pcap="Se mide en régimen estacionario.",
            analytic=[(rf"\omega_0 = {m13.omega0():.3f}\ \mathrm{{rad/s}},\qquad Q = {m13.q_factor():.3f},\qquad I_{{max}} = V_0/R = {pk:.3f}\ \mathrm{{A}}", ACCENT)],
            acap="Curva de resonancia y fase de la corriente.",
            numeric="RK4 de la ecuación diferencial en el dominio del tiempo y detección síncrona (proyección sobre sen y cos) en 6 ciclos tras 25 de transitorio.",
            xr=(500, 1500), yr=(0, 3), xticks=(500, 1000, 1500), yticks=(0, 1, 2, 3), xlabel="ω [rad/s]", ylabel="I₀ [A]",
            curves=[(w, m13.current_amplitude_analytic(w), ANALYTIC)], dots=(wn, an[:, 0]),
            legend=[(ANALYTIC, "V₀/Z"), (NUMERIC, "RK4 + detección síncrona")],
            table=(["magnitud", "numérico", "analítico"],
                   [("I₀(ω₀) [A]", f"{a0[0]:.5f}", f"{pk:.5f}"), ("φ(ω₀) [rad]", f"{a0[1]:.5f}", "0"),
                    ("φ(1250) [rad]", f"{m13.steady_state_numeric(1250.0)[1]:.5f}", f"{float(m13.phase_analytic(1250.0)):.5f}")]),
            final="La simulación temporal reproduce amplitud y fase de la teoría fasorial.", fsize=34)


class PowerTransformer(PhysicsScene):
    volume = 2
    chapter_number = 13
    chapter_title = "Circuitos de corriente alterna"

    def construct(self):
        t = np.linspace(0, 1, 300)
        tn = np.linspace(0, 1, 41)
        ph = np.deg2rad(m13.PHI_DEG)
        v = lambda x: m13.V_PEAK * np.sin(2 * np.pi * x)  # noqa: E731
        i = lambda x: m13.I_PEAK * np.sin(2 * np.pi * x - ph)  # noqa: E731
        pa = m13.power_average_analytic()
        tr_a, tr_n = m13.transformer_analytic(), m13.transformer_numeric()
        curve_scene(
            self,
            found=[r"P = \tfrac12 V_0 I_0\cos\phi = V_{rms}I_{rms}\cos\phi,\qquad X_{rms} = X_0/\sqrt2", r"\dfrac{V_s}{V_p} = \dfrac{N_s}{N_p},\qquad V_pI_p = V_sI_s\ \ (\text{ideal})"],
            note="Solo la componente de corriente en fase con la tensión entrega potencia media; el factor cos φ es el factor de potencia.",
            fcap="Potencia en CA y transformador ideal.",
            problem=["Una carga recibe v = 170 sen ωt V con i = 8.0 sen(ωt − 40°) A. Halle la potencia media y los valores eficaces.",
                     "Un transformador ideal de 200:50 vueltas con 120 V y 2.0 A eficaces en el primario: halle el secundario."],
            pcap="Se compara p(t) con el promedio.",
            analytic=[(rf"\bar P = {pa:.4f}\ \mathrm{{W}},\quad V_{{rms}} = {m13.V_PEAK / np.sqrt(2):.3f}\ \mathrm{{V}}", ACCENT),
                      rf"V_s = {tr_a['vs']:.2f}\ \mathrm{{V}},\qquad I_s = {tr_a['is']:.2f}\ \mathrm{{A}}"],
            acap="El promedio de cos(2ωt − φ) en un periodo es nulo.",
            numeric="Simpson sobre un periodo de p(t) = v(t)·i(t); el transformador se verifica muestreando ondas y comprobando v·i en ambos lados.",
            xr=(0, 1), yr=(-700, 1000), xticks=(0, 0.5, 1), yticks=(-500, 0, 500, 1000), xlabel="t / T", ylabel="p [W]",
            curves=[(t, v(t) * i(t), ANALYTIC), (t, np.full_like(t, pa), ERROR)], dots=(tn, v(tn) * i(tn)),
            legend=[(ANALYTIC, "p(t) = v i"), (ERROR, "potencia media"), (NUMERIC, "muestreo")],
            table=(["magnitud", "numérico", "analítico"],
                   [("P media [W]", f"{m13.power_average_numeric():.5f}", f"{pa:.5f}"), ("V_rms [V]", f"{m13.rms_numeric(m13.V_PEAK):.5f}", f"{m13.V_PEAK / np.sqrt(2):.5f}"),
                    ("V_s [V]", f"{tr_n['vs']:.5f}", f"{tr_a['vs']:.5f}"), ("I_s [A]", f"{tr_n['is']:.5f}", f"{tr_a['is']:.5f}"),
                    ("P_p − P_s [W]", f"{tr_n['p_p'] - tr_n['p_s']:.1e}", "0")]),
            final="La potencia se conserva en el transformador ideal y P = Vrms·Irms·cos φ.", fsize=34)
