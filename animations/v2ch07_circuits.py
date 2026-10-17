"""Volumen 2, capitulo 7. Circuitos de corriente directa."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, battery_symbol, legend_block, mono_table,
    polyline, resistor_symbol, text_lines, wire,
)
from physics import v2ch07_circuits as e7


class MeshCircuit(PhysicsScene):
    volume = 2
    chapter_number = 7
    chapter_title = "Circuitos de corriente directa"

    def construct(self):
        i_a, i_n = e7.mesh_currents_analytic(), e7.mesh_currents_numeric()
        res = e7.kirchhoff_residuals(i_n)
        br = e7.branch_currents(i_n)
        self.foundations(
            [r"\sum I_{nodo} = 0\qquad\text{(Kirchhoff, nodos)},\qquad \sum \Delta V_{malla} = 0\qquad\text{(Kirchhoff, mallas)}",
             r"\varepsilon - IR\text{ al recorrer una resistencia en el sentido de la corriente}"],
            note="Las leyes de Kirchhoff expresan la conservación de la carga y de la energía en un circuito.",
            caption="Leyes de Kirchhoff y corrientes de malla.", size=32)
        self.problem(
            ["Un circuito de dos mallas tiene ε₁ = 12 V con R₁ = 4 Ω a la izquierda, ε₂ = 6 V con R₂ = 2 Ω a la derecha y R₃ = 6 Ω compartida.",
             "Halle las corrientes de malla y de cada rama, y verifique el balance de potencia."],
            caption="Se toman positivas las corrientes en sentido horario.")
        self.analytic(
            [r"(R_1 + R_3)\,i_1 - R_3\,i_2 = \varepsilon_1,\qquad -R_3\,i_1 + (R_2 + R_3)\,i_2 = -\varepsilon_2",
             (rf"i_1 = {i_a[0]:.4f}\ \mathrm{{A}},\qquad i_2 = {i_a[1]:.4f}\ \mathrm{{A}}", ACCENT),
             rf"I_{{R_3}} = i_1 - i_2 = {i_a[0] - i_a[1]:.4f}\ \mathrm{{A}}"],
            size=32, caption="Resolviendo el sistema de dos ecuaciones por determinantes.")

        self.numeric_phase("El sistema lineal 2×2 se resuelve con numpy.linalg.solve y se verifican las leyes de Kirchhoff y la potencia.")
        x0, y0 = -6.0, -1.8
        w, hh = 4.6, 3.4
        xm = x0 + w / 2
        parts = VGroup(
            wire((x0, y0), (x0, y0 + hh), (xm - 0.5, y0 + hh)), wire((xm + 0.5, y0 + hh), (x0 + w, y0 + hh), (x0 + w, y0)), wire((x0 + w, y0), (x0, y0)),
            wire((xm, y0), (xm, y0 + 1.0)), wire((xm, y0 + 2.4), (xm, y0 + hh)))
        r3 = resistor_symbol((xm, y0 + 1.0), (xm, y0 + 2.4))
        bat1 = battery_symbol((x0, y0 + 1.7), vertical=True)
        bat2 = battery_symbol((x0 + w, y0 + 1.7), vertical=True)
        r1 = resistor_symbol((x0 + 0.5, y0), (x0 + 2.0, y0))
        r2 = resistor_symbol((xm + 0.5, y0), (xm + 2.0, y0))
        labs = VGroup(T("ε₁ = 12 V", 15, MUTED).next_to(bat1, LEFT, buff=0.15), T("ε₂ = 6 V", 15, MUTED).next_to(bat2, RIGHT, buff=0.15),
                      T("R₁ = 4 Ω", 15, MUTED).next_to(r1, DOWN, buff=0.15), T("R₂ = 2 Ω", 15, MUTED).next_to(r2, DOWN, buff=0.15),
                      T("R₃ = 6 Ω", 15, MUTED).next_to(r3, RIGHT, buff=0.15))
        self.play(Create(parts), FadeIn(r1), FadeIn(r2), FadeIn(r3), FadeIn(bat1), FadeIn(bat2), FadeIn(labs), run_time=2.0)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("i₁ [A]", f"{i_n[0]:.6f}", f"{i_a[0]:.6f}"), ("i₂ [A]", f"{i_n[1]:.6f}", f"{i_a[1]:.6f}"),
                          ("I en R₃ [A]", f"{br['R3']:.6f}", f"{i_a[0] - i_a[1]:.6f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 3.0 + UP * 1.4)
        chk = mono_table(["verificación", "residuo"],
                         [("malla izquierda [V]", f"{res['loop_left']:.1e}"), ("malla derecha [V]", f"{res['loop_right']:.1e}"), ("potencia [W]", f"{res['power_balance']:.1e}")],
                         size=15, colors=[MUTED, FG], gap=0.4).move_to(RIGHT * 3.0 + DOWN * 0.9)
        self.say("Los residuos de las leyes de Kirchhoff y del balance de potencia son del orden del redondeo.", FadeIn(tab), FadeIn(chk), run_time=1.3)
        self.wait(2.5)


class RCCharging(PhysicsScene):
    volume = 2
    chapter_number = 7
    chapter_title = "Circuitos de corriente directa"

    def construct(self):
        eps, R, C = e7.EPS, e7.R, e7.C
        tau = R * C
        en = e7.rc_energies_numeric()
        t, y = e7.rc_numeric(1.2, 2400)
        self.foundations(
            [r"\varepsilon = IR + \dfrac{q}{C}\;\Rightarrow\; R\dfrac{dq}{dt} = \varepsilon - \dfrac{q}{C}",
             r"q(t) = C\varepsilon\left(1 - e^{-t/RC}\right),\qquad I(t) = \dfrac{\varepsilon}{R}\,e^{-t/RC}"],
            note="La constante de tiempo τ = RC es el tiempo en que el capacitor alcanza el 63 % de su carga final.",
            caption="Carga de un capacitor a través de una resistencia.", size=38)
        self.problem(
            ["Un capacitor de 100 µF se carga con una batería de 12 V a través de una resistencia de 2 kΩ.",
             "Halle q(t), la constante de tiempo y el balance de energía durante la carga."],
            caption="El capacitor está inicialmente descargado.")
        self.analytic(
            [(rf"\tau = RC = {tau:.3f}\ \mathrm{{s}},\qquad q_\infty = C\varepsilon = {C * eps * 1e3:.2f}\ \mathrm{{mC}}", ACCENT),
             rf"E_{{bateria}} = C\varepsilon^2 = {C * eps**2 * 1e3:.2f}\ \mathrm{{mJ}},\quad E_{{capacitor}} = E_{{disipada}} = \tfrac12C\varepsilon^2 = {0.5 * C * eps**2 * 1e3:.2f}\ \mathrm{{mJ}}"],
            size=34, caption="La mitad de la energía de la batería se disipa en la resistencia, sin importar R.")

        self.numeric_phase("RK4 de dq/dt = (ε − q/C)/R (2400 pasos); las energías se obtienen integrando I²R y εI con Simpson.")
        ax, deco = axes_block((0, 1.2), (0, 1.3), 6.4, 3.6, LEFT * 3.2 + DOWN * 0.15, (0, 0.3, 0.6, 0.9, 1.2), (0, 0.4, 0.8, 1.2), "t [s]", "q [mC]  ·  I [mA ÷ 5]")
        ts = np.linspace(0, 1.2, 120)
        cq = polyline(ax, ts, e7.q_analytic(ts) * 1000, ANALYTIC, 3.3)
        ci = polyline(ax, ts, e7.i_analytic(ts) * 1000 / 5, ERROR, 2.6)
        self.play(Create(ax), FadeIn(deco), Create(cq), Create(ci), run_time=2.0)
        dots = VGroup(*[Dot(ax.c2p(float(t[i]), float(y[i, 0]) * 1000), radius=0.05, color=NUMERIC) for i in range(0, len(t), 100)])
        leg = legend_block([(ANALYTIC, "q(t) analítico"), (ERROR, "I(t)/5 analítico"), (NUMERIC, "q(t) RK4")], RIGHT * 0.9 + UP * 2.0)
        self.play(LaggedStart(*[FadeIn(d_, scale=1.4) for d_ in dots], lag_ratio=0.05), FadeIn(leg), run_time=2.0)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("t(63.2 %) [s]", f"{e7.time_to_fraction_numeric():.5f}", f"{-tau * np.log(1 - 0.632):.5f}"),
                          ("E disipada [mJ]", f"{en['dissipated'] * 1e3:.4f}", f"{0.5 * C * eps**2 * 1e3:.4f}"),
                          ("E capacitor [mJ]", f"{en['capacitor'] * 1e3:.4f}", f"{0.5 * C * eps**2 * 1e3:.4f}"),
                          ("E batería [mJ]", f"{en['battery'] * 1e3:.4f}", f"{C * eps**2 * 1e3:.4f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.4 + DOWN * 0.3)
        self.say("RK4 reproduce q(t); las integrales numéricas confirman que la energía disipada iguala a la almacenada.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)
