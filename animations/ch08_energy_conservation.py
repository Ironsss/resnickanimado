"""Capitulo 8. Conservacion de la energia."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, ValueTracker, VGroup, always_redraw, linear, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch08_energy_conservation as c8


class PendulumEnergy(PhysicsScene):
    chapter_number = 8
    chapter_title = "Conservación de la energía"

    def construct(self):
        L, g, th0 = c8.LENGTH, c8.G, c8.THETA0_DEG
        v_an, v_nu = c8.bottom_speed_analytic(), c8.max_speed_numeric()
        self.foundations(
            [r"U = mgL(1 - \cos\theta),\qquad K = \tfrac{1}{2}mv^{2}",
             r"E = K + U = \text{constante}\quad(\text{solo fuerzas conservativas})"],
            note="La tensión de la cuerda es perpendicular al movimiento y no realiza trabajo.",
            caption="La energía mecánica se conserva cuando solo actúan fuerzas conservativas.")
        self.problem(
            ["Un péndulo de L = 1.5 m y m = 1 kg se suelta desde el reposo a 60° de la vertical.",
             "Calcule la rapidez en el punto más bajo y verifique numéricamente la conservación de E."],
            caption="Sin rozamiento ni resistencia del aire.")
        self.analytic(
            [rf"h = L(1 - \cos 60^\circ) = {L * (1 - np.cos(np.deg2rad(th0))):.3f}\ \mathrm{{m}}",
             r"mgh = \tfrac{1}{2}mv^{2}",
             (rf"v = \sqrt{{2gh}} = {v_an:.3f}\ \mathrm{{m/s}}", ACCENT)],
            caption="Toda la energía potencial inicial pasa a cinética en el punto más bajo.")

        self.numeric_phase("RK4 sobre θ'' = −(g/L) sen θ: se grafican K, U y E durante tres segundos.")
        t, y = c8.pendulum_run("rk4", t_end=3.0, n=600)
        k, u = c8.energy(y[:, 0], y[:, 1])
        ax, deco = axes_block((0, 3), (0, 8), 6.6, 3.6, LEFT * 3.4 + DOWN * 0.15, (0, 1, 2, 3), (0, 2, 4, 6, 8), "t [s]", "E [J]")
        ek = polyline(ax, t, k, NUMERIC, 3)
        eu = polyline(ax, t, u, ACCENT, 3)
        et = polyline(ax, t, k + u, ANALYTIC, 3)
        leg = legend_block([(NUMERIC, "K"), (ACCENT, "U"), (ANALYTIC, "K + U")], LEFT * 3.4 + UP * 2.4)
        leg.arrange(RIGHT, buff=0.6).move_to(LEFT * 3.4 + UP * 2.4)
        pivot = np.array([3.9, 1.7, 0.0])
        rod_len = 2.4
        clock = ValueTracker(0.0)

        def bob():
            th = float(np.interp(clock.get_value(), t, y[:, 0]))
            return pivot + rod_len * np.array([np.sin(th), -np.cos(th), 0.0])

        rod = always_redraw(lambda: Line(pivot, bob(), color=FG, stroke_width=3))
        ball = always_redraw(lambda: Dot(bob(), radius=0.15, color=ANALYTIC))
        cursor = always_redraw(lambda: DashedLine(ax.c2p(clock.get_value(), 0), ax.c2p(clock.get_value(), 8), color=FAINT, stroke_width=1.5, dash_length=0.08))
        self.play(Create(ax), FadeIn(deco), FadeIn(leg), FadeIn(rod), FadeIn(ball), FadeIn(cursor), run_time=1.0)
        self.add(ek, eu, et)
        self.play(clock.animate.set_value(3.0), run_time=6.0, rate_func=linear)
        self.play(FadeOut(cursor), FadeOut(rod), FadeOut(ball), run_time=0.5)
        drift = {m_: c8.energy_drift(m_, t_end=6.0, n=600) for m_ in ("euler", "verlet", "rk4")}
        tab = mono_table(["integrador", "deriva de E (6 s)"],
                         [("Euler", f"{drift['euler']:.1e}"), ("Verlet", f"{drift['verlet']:.1e}"), ("RK4", f"{drift['rk4']:.1e}")],
                         size=18, colors=[MUTED, NUMERIC], gap=0.6).move_to(RIGHT * 3.9 + UP * 0.3)
        res = text_lines([(f"v máxima RK4   {v_nu:.4f} m/s", NUMERIC), (f"v analítica   {v_an:.4f} m/s", ANALYTIC)], 17).move_to(RIGHT * 3.9 + DOWN * 1.5)
        self.say("RK4 conserva E con error mínimo; Euler la hace crecer y Verlet la mantiene acotada.",
                 FadeIn(tab), FadeIn(res), run_time=1.4)
        self.wait(2.5)


class SpringFriction(PhysicsScene):
    chapter_number = 8
    chapter_title = "Conservación de la energía"

    def construct(self):
        m, k, mu, x0, g = c8.M, c8.KS, c8.MU, c8.X0, c8.G
        d = c8.delta()
        an = c8.turning_points_analytic()
        t, x, v, turn = c8.spring_friction_numeric()
        self.foundations(
            [r"W_{nc} = \Delta E_{mec}",
             r"f_k = \mu_k mg,\qquad \delta = \dfrac{\mu_k mg}{k}"],
            note="Con fricción cinética la amplitud disminuye 2δ en cada semiciclo y el bloque se detiene si |x| ≤ δ.",
            caption="Una fuerza no conservativa disipa energía mecánica.")
        self.problem(
            ["Un bloque de 1 kg unido a un resorte de k = 100 N/m se comprime 0.20 m sobre una superficie con μk = 0.25.",
             "Halle los puntos de retorno sucesivos y la posición donde se detiene."],
            caption="Se suelta desde el reposo.")
        self.analytic(
            [rf"\delta = \dfrac{{\mu_k mg}}{{k}} = {d:.4f}\ \mathrm{{m}}",
             r"x_n = \pm\left(x_0 - 2n\delta\right),\quad n = 0, 1, 2, \ldots",
             (rf"x_n = {', '.join(f'{p:+.3f}' for p in an)}\ \mathrm{{m}}", ACCENT),
             (rf"\text{{se detiene en }}x = {an[-1]:+.3f}\ \mathrm{{m}}\ (|x| \le \delta)", ACCENT)],
            caption="La disminución de amplitud por semiciclo es constante, a diferencia del amortiguamiento viscoso.")

        self.numeric_phase("RK4 con fricción −μmg·signo(v) y detección de los puntos de retorno por interpolación del cambio de signo de v.")
        ax, deco = axes_block((0, 1.5), (-0.25, 0.25), 7.4, 3.6, LEFT * 2.7 + DOWN * 0.1, (0, 0.5, 1.0, 1.5), (-0.2, 0, 0.2), "t [s]", "x [m]")
        curve = polyline(ax, t[::20], x[::20], NUMERIC, 3.2)
        zero = DashedLine(ax.c2p(0, 0), ax.c2p(1.5, 0), color=FAINT, stroke_width=1.5, dash_length=0.1)
        self.play(Create(ax), FadeIn(deco), Create(zero), run_time=1.0)
        self.play(Create(curve), run_time=3.0)
        tt = [t[int(np.argmin(np.abs(x - p)))] for p in turn]
        dots = VGroup(*[Dot(ax.c2p(ti, p), radius=0.08, color=ACCENT) for ti, p in zip(tt, turn)])
        envelope = VGroup(
            polyline(ax, [0, 1.5], [d, d], FAINT, 1.5), polyline(ax, [0, 1.5], [-d, -d], FAINT, 1.5))
        self.play(FadeIn(dots), FadeIn(envelope), run_time=1.0)
        rows = [(i, f"{a:+.4f}", f"{b:+.4f}") for i, (a, b) in enumerate(zip(an, turn))]
        tab = mono_table(["n", "analítico [m]", "RK4 [m]"], rows, size=17, colors=[FG, ANALYTIC, NUMERIC], gap=0.5).move_to(RIGHT * 4.6 + UP * 0.2)
        note = T("las líneas grises marcan ±δ", 15, MUTED).next_to(tab, DOWN, buff=0.4).align_to(tab, LEFT)
        self.say("Los puntos de retorno numéricos coinciden con x_n; el bloque se detiene cuando el resorte no vence la fricción estática.",
                 FadeIn(tab), FadeIn(note), run_time=1.4)
        self.wait(2.5)
