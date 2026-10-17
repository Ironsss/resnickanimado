"""Capitulo 10. Colisiones."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, ValueTracker, VGroup, always_redraw, linear

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch10_collisions as c10


class Collision1D(PhysicsScene):
    chapter_number = 10
    chapter_title = "Colisiones"

    def construct(self):
        m1, m2, v1 = c10.M1, c10.M2, c10.V1
        self.foundations(
            [r"m_1v_1 + m_2v_2 = m_1v_1' + m_2v_2'",
             r"e = \dfrac{v_2' - v_1'}{v_1 - v_2}\in[0,1]",
             r"v_1' = \dfrac{m_1 - e\,m_2}{m_1+m_2}v_1,\qquad v_2' = \dfrac{(1+e)m_1}{m_1+m_2}v_1"],
            note="El momento se conserva siempre; la energía cinética solo si e = 1 (colisión elástica).",
            caption="Colisión unidimensional con el segundo cuerpo en reposo.")
        self.problem(
            ["Un cuerpo de 2 kg que se mueve a 3 m/s choca con otro de 1 kg en reposo.",
             "Halle las velocidades finales para e = 1, 0.5 y 0 (perfectamente inelástica)."],
            caption="Movimiento sobre una recta sin fricción.")
        r = {e: c10.outcome_analytic(e) for e in (1.0, 0.5, 0.0)}
        self.analytic(
            [rf"e = 1:\quad v_1' = {r[1.0][0]:.2f},\ v_2' = {r[1.0][1]:.2f}\ \mathrm{{m/s}}",
             rf"e = 0.5:\quad v_1' = {r[0.5][0]:.2f},\ v_2' = {r[0.5][1]:.2f}\ \mathrm{{m/s}}",
             (rf"e = 0:\quad v_1' = v_2' = {r[0.0][0]:.2f}\ \mathrm{{m/s}}", ACCENT)],
            caption="Con e = 0 ambos cuerpos quedan unidos y se mueven con la velocidad del centro de masa.")

        self.numeric_phase("Simulación del contacto: resorte-amortiguador lineal; el amortiguamiento ζ fija el coeficiente e.")
        ts, us = c10.contact_history(0.0)
        tms = ts * 1000
        ax, deco = axes_block((0, tms[-1]), (-0.5, 4.5), 7.2, 3.7, LEFT * 2.8 + DOWN * 0.15,
                              tuple(np.round(np.linspace(0, tms[-1], 4), 1)), (0, 1, 2, 3, 4), "t [ms]", "v [m/s]", "{:g}")
        c1 = polyline(ax, tms, us[:, 0], FG, 3.5)
        c2 = polyline(ax, tms, us[:, 1], NUMERIC, 3.5)
        leg = legend_block([(FG, "cuerpo 1 (2 kg)"), (NUMERIC, "cuerpo 2 (1 kg)")], RIGHT * 4.7 + UP * 2.0)
        self.play(Create(ax), FadeIn(deco), FadeIn(leg), run_time=1.0)
        self.play(Create(c1), Create(c2), run_time=3.0)
        self.say("Para ζ = 0 el contacto es elástico: se recuperan e = 1 y las velocidades de la tabla.", run_time=0.5)
        rows = []
        for z in (0.0, 0.2, 0.4, 0.6, 0.8):
            e_an = c10.restitution_from_damping(z)
            e_nu = c10.restitution_measured(z)
            v1p, v2p = c10.contact_simulation(z)
            rows.append((f"{z:.1f}", f"{e_an:.4f}", f"{e_nu:.4f}", f"{v1p:.3f}", f"{v2p:.3f}"))
        tab = mono_table(["ζ", "e teoría", "e medido", "v1'", "v2'"], rows, size=15, colors=[FG, ANALYTIC, NUMERIC, MUTED, MUTED], gap=0.28).move_to(RIGHT * 4.7 + DOWN * 0.4)
        self.say("El coeficiente medido en la simulación coincide con e = exp(−πζ/√(1−ζ²)).", FadeIn(tab), run_time=1.4)
        self.wait(2.5)


class BallisticPendulum(PhysicsScene):
    chapter_number = 10
    chapter_title = "Colisiones"

    def construct(self):
        an, nu = c10.bullet_block_analytic(), c10.bullet_block_numeric()
        self.foundations(
            [r"mv = (m+M)\,V\qquad\text{(colisión perfectamente inelástica)}",
             r"\tfrac{1}{2}(m+M)V^{2} = (m+M)\,g\,h\qquad\text{(el péndulo sube)}"],
            note="Durante el choque se conserva el momento, no la energía; después, sí se conserva la energía mecánica.",
            caption="Dos etapas: choque y ascenso.")
        self.problem(
            ["Una bala de 20 g a 250 m/s se incrusta en un bloque de 2 kg colgado de una cuerda de 1.2 m.",
             "Halle la velocidad del conjunto, la altura que alcanza y la fracción de energía cinética perdida."],
            caption="La cuerda es ligera e inextensible.")
        self.analytic(
            [rf"V = \dfrac{{mv}}{{m+M}} = {an['V']:.3f}\ \mathrm{{m/s}}",
             (rf"h = \dfrac{{V^{{2}}}}{{2g}} = {an['h']:.4f}\ \mathrm{{m}}", ACCENT),
             (rf"\dfrac{{\Delta K}}{{K_0}} = \dfrac{{M}}{{m+M}} = {an['loss_fraction'] * 100:.2f}\,\%", ACCENT),
             rf"v = \dfrac{{m+M}}{{m}}\sqrt{{2gh}} = {an['v_from_h']:.1f}\ \mathrm{{m/s}}"],
            size=34, caption="Medir h permite recuperar la velocidad de la bala.")

        self.numeric_phase("Tras el impulso, el péndulo se integra con RK4 hasta su primer máximo y se mide la altura.")
        ts, th = c10.swing_history(nu["V"])
        pivot = np.array([-3.4, 2.0, 0.0])
        rod = 2.6
        clock = ValueTracker(0.0)

        def pos():
            a = float(np.interp(clock.get_value(), ts, th))
            return pivot + rod * np.array([np.sin(a), -np.cos(a), 0.0])

        rope = always_redraw(lambda: Line(pivot, pos(), color=FG, stroke_width=3))
        bob = always_redraw(lambda: Rectangle(width=0.55, height=0.4, stroke_color=FG, stroke_width=2.5, fill_color=PANEL, fill_opacity=1).move_to(pos()))
        support = Line(pivot + LEFT * 0.7, pivot + RIGHT * 0.7, color=FAINT, stroke_width=4)
        low = pivot + np.array([0, -rod, 0])
        hmax_line = DashedLine(low + np.array([-1.2, nu["h"] * (rod / c10.ROPE), 0]), low + np.array([1.2, nu["h"] * (rod / c10.ROPE), 0]),
                               color=ACCENT, stroke_width=2, dash_length=0.1)
        base_line = DashedLine(low + np.array([-1.2, 0, 0]), low + np.array([1.2, 0, 0]), color=FAINT, stroke_width=1.5, dash_length=0.1)
        self.play(FadeIn(support), FadeIn(rope), FadeIn(bob), FadeIn(base_line), run_time=1.0)
        self.play(clock.animate.set_value(float(ts[-1])), run_time=3.0, rate_func=linear)
        self.play(FadeIn(hmax_line), run_time=0.6)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("V [m/s]", f"{nu['V']:.4f}", f"{an['V']:.4f}"), ("h [m]", f"{nu['h']:.4f}", f"{an['h']:.4f}"),
                          ("ΔK/K₀", f"{nu['loss_fraction']:.5f}", f"{an['loss_fraction']:.5f}")],
                         size=17, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 3.4 + UP * 0.6)
        note = T("la línea ámbar marca la altura máxima h", 15, MUTED).next_to(tab, DOWN, buff=0.4).align_to(tab, LEFT)
        self.say("La altura máxima integrada coincide con V²/2g; casi toda la energía de la bala se disipa en el choque.",
                 FadeIn(tab), FadeIn(note), run_time=1.4)
        self.wait(2.5)
