"""Capitulo 9. Sistemas de particulas."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, LaggedStart, Line, ValueTracker, VGroup, always_redraw, linear, Cross

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch09_particle_systems as c9


class CenterOfMass(PhysicsScene):
    chapter_number = 9
    chapter_title = "Sistemas de partículas"

    def construct(self):
        cm_a, cm_n = c9.cm_discrete_analytic(), c9.cm_discrete_numeric()
        rod_a, rod_n = c9.rod_cm_analytic(), c9.rod_cm_numeric()
        mc, samples = c9.rod_cm_monte_carlo()
        self.foundations(
            [r"\vec r_{cm} = \dfrac{1}{M}\sum_i m_i\,\vec r_i",
             r"x_{cm} = \dfrac{\int x\,dm}{\int dm},\qquad dm = \lambda(x)\,dx"],
            note="El centro de masa se mueve como si toda la masa y todas las fuerzas externas estuvieran en él.",
            caption="Promedio de las posiciones ponderado por la masa.")
        self.problem(
            ["a) Tres partículas de 2, 3 y 5 kg están en (0, 0), (4, 0) y (2, 3) m.",
             "b) Una barra de 2 m tiene densidad lineal λ(x) = λ₀x/L. Halle el centro de masa de ambos sistemas."],
            caption="Coordenadas en metros.")
        self.analytic(
            [rf"x_{{cm}} = \dfrac{{2(0)+3(4)+5(2)}}{{10}} = {cm_a[0]:.2f}\ \mathrm{{m}},\quad y_{{cm}} = \dfrac{{5(3)}}{{10}} = {cm_a[1]:.2f}\ \mathrm{{m}}",
             r"x_{cm} = \dfrac{\int_0^{L} x^{2}dx}{\int_0^{L} x\,dx} = \dfrac{2L}{3}",
             (rf"x_{{cm}} = {rod_a:.4f}\ \mathrm{{m}}", ACCENT)],
            size=34, caption="La densidad creciente desplaza el centro de masa hacia el extremo más pesado.")

        self.numeric_phase("Partículas: producto matricial m·r / M. Barra: cuadratura de Simpson y muestreo Monte Carlo con densidad ∝ x.")
        s = 0.62
        ax, deco = axes_block((0, 5), (0, 4), 5 * s * 1.6, 4 * s * 1.6, LEFT * 3.9 + DOWN * 0.1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4), "x [m]", "y [m]")
        parts = VGroup(*[Dot(ax.c2p(*p), radius=0.07 + 0.035 * mi, color=FG) for p, mi in zip(c9.POSITIONS, c9.MASSES)])
        cm = Cross(scale_factor=0.14, stroke_color=ACCENT, stroke_width=4).move_to(ax.c2p(*cm_n))
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(p, scale=1.4) for p in parts], lag_ratio=0.3), run_time=1.5)
        self.play(FadeIn(cm, scale=2.0), run_time=0.8)
        out = text_lines([(f"numérico   ({cm_n[0]:.3f}, {cm_n[1]:.3f}) m", NUMERIC), (f"analítico  ({cm_a[0]:.3f}, {cm_a[1]:.3f}) m", ANALYTIC),
                          "tamaño del punto ∝ masa"], 17).move_to(RIGHT * 4.2 + UP * 1.4)
        self.play(FadeIn(out), run_time=0.8)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (ax, deco, parts, cm, out)], run_time=0.6)

        # barra: media acumulada de Monte Carlo
        self.say("Monte Carlo: la media acumulada de x converge al valor analítico 2L/3.", run_time=0.5)
        ns = np.unique(np.logspace(1, np.log10(len(samples)), 60).astype(int))
        run = np.array([samples[:n].mean() for n in ns])
        ax2, deco2 = axes_block((1, 5.6), (1.1, 1.5), 7.2, 3.6, LEFT * 2.8 + DOWN * 0.1, (1, 2, 3, 4, 5), (1.1, 1.3, 1.5), "log₁₀ N", "x̄ [m]", "{:g}")
        ref = DashedLine(ax2.c2p(1, rod_a), ax2.c2p(5.6, rod_a), color=ANALYTIC, stroke_width=2, dash_length=0.1)
        curve = polyline(ax2, np.log10(ns), run, NUMERIC, 3.2)
        self.play(Create(ax2), FadeIn(deco2), Create(ref), run_time=1.0)
        self.play(Create(curve), run_time=3.0)
        tab = mono_table(["método", "x_cm [m]"],
                         [("Simpson", f"{rod_n:.6f}"), ("Monte Carlo", f"{mc:.6f}"), ("analítico", f"{rod_a:.6f}")],
                         size=17, colors=[MUTED, NUMERIC], gap=0.5).move_to(RIGHT * 4.4 + UP * 0.2)
        self.play(FadeIn(tab), run_time=1.0)
        self.wait(2.5)


class ExplodingProjectile(PhysicsScene):
    chapter_number = 9
    chapter_title = "Sistemas de partículas"

    def construct(self):
        v0, ang, g = c9.V0, c9.ANGLE_DEG, c9.G
        ap = c9.apex_state()
        an = c9.fragments_analytic()
        nu = c9.fragments_numeric()
        R = c9.range_analytic()
        self.foundations(
            [r"\vec F_{ext} = M\vec a_{cm},\qquad \vec P = M\vec v_{cm}",
             r"\text{En una explosión interna: } \vec P \text{ se conserva}"],
            note="Las fuerzas internas de la explosión no alteran el movimiento del centro de masa.",
            caption="El centro de masa de los fragmentos sigue la trayectoria del proyectil intacto.")
        self.problem(
            ["Un proyectil de 4 kg se lanza a 20 m/s y 60°. En el punto más alto explota en dos fragmentos:",
             "A (1 kg) cae verticalmente desde el reposo y B (3 kg) sale horizontalmente. ¿Dónde caen?"],
            caption="Terreno plano y sin resistencia del aire.")
        self.analytic(
            [rf"v_x = {ap['vx']:.1f}\ \mathrm{{m/s}},\quad P_x = 4(10) = 3\,v_B \Rightarrow v_B = {an['vb']:.2f}\ \mathrm{{m/s}}",
             rf"t_{{\text{{caída}}}} = \sqrt{{2h/g}} = {an['t_fall']:.3f}\ \mathrm{{s}}",
             rf"x_A = {an['xa']:.2f}\ \mathrm{{m}},\qquad x_B = {an['xb']:.2f}\ \mathrm{{m}}",
             (rf"x_{{cm}} = \dfrac{{1\,x_A + 3\,x_B}}{{4}} = {an['x_cm']:.2f}\ \mathrm{{m}} = R", ACCENT)],
            size=34, caption="El alcance del proyectil intacto, R = v₀² sen 2θ / g, coincide con la posición final del centro de masa.")

        self.numeric_phase("Se integran ambos fragmentos con RK4 desde el punto más alto y se reconstruye el centro de masa.")
        sc = 0.17
        ax, deco = axes_block((0, 45), (0, 18), 45 * sc, 18 * sc, LEFT * 2.2 + DOWN * 0.45, (0, 10, 20, 30, 40), (0, 5, 10, 15), "x [m]", "y [m]")
        th = np.deg2rad(ang)
        tt = np.linspace(0, ap["t"], 80)
        pre = polyline(ax, v0 * np.cos(th) * tt, v0 * np.sin(th) * tt - 0.5 * g * tt**2, FAINT, 2.5)
        ta, ya = nu["A"]
        tb, yb = nu["B"]
        pa = polyline(ax, ya[:, 0], ya[:, 1], FG, 3)
        pb = polyline(ax, yb[:, 0], yb[:, 1], NUMERIC, 3)
        self.play(Create(ax), FadeIn(deco), Create(pre), run_time=1.5)
        apex = Dot(ax.c2p(ap["x"], ap["y"]), radius=0.1, color=ACCENT)
        self.play(FadeIn(apex, scale=2.0), run_time=0.6)
        self.play(Create(pa), Create(pb), run_time=2.5)
        # centro de masa a lo largo del tiempo de caida
        clock = ValueTracker(0.0)

        def cm_pos():
            tau = clock.get_value()
            a = np.array([np.interp(tau, ta, ya[:, 0]), np.interp(tau, ta, ya[:, 1])])
            b = np.array([np.interp(tau, tb, yb[:, 0]), np.interp(tau, tb, yb[:, 1])])
            return (c9.M_A * a + c9.M_B * b) / (c9.M_A + c9.M_B)

        cmdot = always_redraw(lambda: Dot(ax.c2p(*cm_pos()), radius=0.09, color=ACCENT))
        self.add(cmdot)
        self.play(clock.animate.set_value(min(ta[-1], tb[-1]) * 0.999), run_time=2.5, rate_func=linear)
        tab = mono_table(["magnitud", "RK4", "analítico"],
                         [("x_A [m]", f"{nu['xa']:.4f}", f"{an['xa']:.4f}"), ("x_B [m]", f"{nu['xb']:.4f}", f"{an['xb']:.4f}"),
                          ("x_cm [m]", f"{nu['x_cm']:.4f}", f"{R:.4f}")],
                         size=17, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 4.6 + UP * 0.9)
        self.say("El punto ámbar sigue la parábola original hasta el suelo: la explosión no cambia el movimiento del centro de masa.",
                 FadeIn(tab), run_time=1.4)
        self.wait(2.5)
