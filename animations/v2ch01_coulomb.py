"""Volumen 2, capitulo 1. Carga electrica y ley de Coulomb."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, Line, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table, polyline,
    text_lines,
)
from physics import v2ch01_coulomb as e1


class CoulombForces(PhysicsScene):
    volume = 2
    chapter_number = 1
    chapter_title = "Carga eléctrica y ley de Coulomb"

    def construct(self):
        fa, fn = e1.force_analytic(), e1.force_numeric()
        self.foundations(
            [r"\vec F_{12} = k\dfrac{q_1q_2}{r^{2}}\,\hat r_{12},\qquad k = \dfrac{1}{4\pi\varepsilon_0} = 8.988\times10^{9}\ \mathrm{N\,m^2/C^2}",
             r"\vec F_{i} = \sum_{j\ne i} \vec F_{ij}\qquad(\text{superposición}),\qquad \vec F = -\nabla U"],
            note="La fuerza es central, proporcional al producto de las cargas e inversa del cuadrado de la distancia.",
            caption="Ley de Coulomb y superposición.", size=36)
        self.problem(
            ["Tres cargas puntuales: q₁ = +3 µC en (0, 0), q₂ = −2 µC en (0.4, 0) m y q₃ = +1 µC en (0.2, 0.3) m.",
             "Halle la fuerza neta sobre q₃ y verifique que las fuerzas internas suman cero."],
            caption="Las coordenadas están en metros.")
        self.analytic(
            [r"\vec F_3 = k q_3\left(\dfrac{q_1\,\vec r_{13}}{r_{13}^3} + \dfrac{q_2\,\vec r_{23}}{r_{23}^3}\right),\quad r_{13} = r_{23} = 0.36\ \mathrm{m}",
             (rf"\vec F_3 = ({fa[0] * 1000:.3f},\ {fa[1] * 1000:.3f})\ \mathrm{{mN}},\quad |\vec F_3| = {np.linalg.norm(fa) * 1000:.3f}\ \mathrm{{mN}}", ACCENT)],
            size=34, caption="La repulsión de q₁ y la atracción de q₂ se suman vectorialmente.")

        self.numeric_phase("Se calcula la fuerza como −∇U con diferencias centradas de la energía potencial total respecto de la posición de q₃.")
        sc = 7.5
        org = np.array([-5.4, -2.0, 0.0])

        def P(p):
            return org + sc * np.array([p[0], p[1], 0.0])

        cols = [NUMERIC, ERROR, ACCENT]
        dots = VGroup(*[Dot(P(p), radius=0.12, color=c) for p, c in zip(e1.POS, cols)])
        labs = VGroup(*[T(n, 18, c).next_to(P(p), d_, buff=0.14) for n, c, p, d_ in zip(("q₁ = +3 µC", "q₂ = −2 µC", "q₃ = +1 µC"), cols, e1.POS, (DOWN, DOWN, UP))])
        self.play(LaggedStart(*[FadeIn(d, scale=1.5) for d in dots], lag_ratio=0.3), FadeIn(labs), run_time=1.5)
        base = P(e1.POS[2])
        parts = []
        for j in (0, 1):
            r = e1.POS[2] - e1.POS[j]
            fj = e1.K_E * e1.Q[2] * e1.Q[j] * r / np.linalg.norm(r) ** 3
            parts.append(Arrow(base, base + 8 * np.array([fj[0], fj[1], 0]), buff=0, color=cols[j], stroke_width=3, max_tip_length_to_length_ratio=0.2))
        net = Arrow(base, base + 8 * np.array([fa[0], fa[1], 0]), buff=0, color=ANALYTIC, stroke_width=5, max_tip_length_to_length_ratio=0.15)
        self.play(*[Create(p) for p in parts], run_time=1.2)
        self.play(Create(net), run_time=1.0)
        tab = mono_table(["componente", "−∇U [mN]", "analítico [mN]"],
                         [("Fx", f"{fn[0] * 1000:.5f}", f"{fa[0] * 1000:.5f}"), ("Fy", f"{fn[1] * 1000:.5f}", f"{fa[1] * 1000:.5f}"),
                          ("ΣF interna", f"{e1.newton_third_law_residual():.1e} N", "0")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 2.4 + UP * 0.8)
        note = text_lines([("flechas: color de la carga que ejerce cada fuerza", MUTED), ("blanco: fuerza neta (longitud ∝ F)", MUTED)], 14).move_to(RIGHT * 2.8 + DOWN * 1.0)
        self.say("El gradiente numérico de la energía potencial reproduce la suma vectorial de Coulomb; las fuerzas internas se anulan.", FadeIn(tab), FadeIn(note), run_time=1.3)
        self.wait(2.5)


class EquilibriumPoint(PhysicsScene):
    volume = 2
    chapter_number = 1
    chapter_title = "Carga eléctrica y ley de Coulomb"

    def construct(self):
        xa, xn = e1.zero_force_point_analytic(), e1.zero_force_point_numeric()
        oa, on = e1.opposite_charges_point_analytic(), e1.opposite_charges_point_numeric()
        self.foundations(
            [r"F = kq_t\left(\dfrac{q_1}{x^{2}} - \dfrac{q_2}{(d - x)^{2}}\right) = 0",
             r"\dfrac{x}{d - x} = \sqrt{\dfrac{q_1}{q_2}}"],
            note="Entre cargas del mismo signo el punto de fuerza nula está más cerca de la carga menor.",
            caption="Equilibrio de una carga de prueba entre dos cargas fijas.", size=40)
        self.problem(
            ["Dos cargas de +4 µC y +1 µC están a 0.60 m de distancia. ¿Dónde debe colocarse una carga de prueba para quedar en equilibrio?",
             "Repita el análisis si la segunda carga fuera −1 µC."],
            caption="Todas las cargas están sobre una recta.")
        self.analytic(
            [(rf"x = d\,\dfrac{{\sqrt{{q_1}}}}{{\sqrt{{q_1}} + \sqrt{{q_2}}}} = {xa:.3f}\ \mathrm{{m}}\ \text{{(desde }}q_1)", ACCENT),
             r"\text{Con }q_2 = -1\ \mu\mathrm{C}:\quad \dfrac{x}{x - d} = 2 \;\Rightarrow\; x = 2d",
             (rf"x = {oa:.3f}\ \mathrm{{m}}\ \text{{(fuera del segmento, tras la carga menor)}}", ACCENT)],
            size=32, caption="Con signos opuestos el equilibrio existe fuera del segmento que une las cargas.")

        self.numeric_phase("Se grafica la fuerza neta sobre la carga de prueba y se busca su cero por bisección.")
        ax, deco = axes_block((0, 0.6), (-4, 4), 7.0, 3.5, LEFT * 2.8 + DOWN * 0.15, (0, 0.2, 0.4, 0.6), (-4, -2, 0, 2, 4), "x [m]", "F [mN]")
        xs = np.linspace(0.03, 0.57, 150)
        f_mN = np.clip(e1.net_force_on_test(xs, qt=1e-9) * 1e3, -4.0, 4.0)
        curve = polyline(ax, xs, f_mN, ANALYTIC, 3.5)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(0.6, 0), color=FAINT, stroke_width=1.5, dash_length=0.1)
        mark = Dot(ax.c2p(xn, 0), radius=0.1, color=ACCENT)
        self.play(Create(ax), FadeIn(deco), Create(zero), run_time=1.0)
        self.play(Create(curve), run_time=2.0)
        self.play(FadeIn(mark, scale=2.0), run_time=0.8)
        tab = mono_table(["caso", "bisección [m]", "analítico [m]"],
                         [("+4 µC, +1 µC", f"{xn:.6f}", f"{xa:.6f}"), ("+4 µC, −1 µC", f"{on:.6f}", f"{oa:.6f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 4.3 + UP * 0.8)
        leg = text_lines([("carga de prueba: 1 nC", MUTED), ("F > 0: empuja hacia +x", MUTED)], 14).move_to(RIGHT * 4.3 + DOWN * 0.9)
        self.say("La fuerza cambia de signo en x = 0.40 m: es un equilibrio inestable, porque una pequeña desviación la aleja más.", FadeIn(tab), FadeIn(leg), run_time=1.3)
        self.wait(2.5)
