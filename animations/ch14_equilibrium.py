"""Capitulo 14. Equilibrio de cuerpos rigidos."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Polygon, Rectangle, ValueTracker, VGroup, always_redraw, linear, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch14_equilibrium as c14


class LadderFriction(PhysicsScene):
    chapter_number = 14
    chapter_title = "Equilibrio de los cuerpos rígidos"

    def construct(self):
        L, th, mu, wl, wp = c14.L_LAD, c14.THETA_DEG, c14.MU, c14.W_LAD, c14.W_MAN
        dmax_a, dmax_n = c14.d_max_analytic(), c14.d_max_numeric()
        amin_a, amin_n = c14.min_angle_analytic(), c14.min_angle_numeric()
        self.foundations(
            [r"\sum F_x = 0,\qquad \sum F_y = 0,\qquad \sum \tau = 0",
             r"f \le \mu_s N_{suelo}"],
            note="Los torques se calculan respecto al pie de la escalera para eliminar las fuerzas del suelo.",
            caption="Equilibrio estático: fuerza neta y torque neto nulos.")
        self.problem(
            ["Una escalera de 5 m y 150 N se apoya a 60° en una pared lisa; el suelo tiene μs = 0.40.",
             "Un hombre de 700 N sube por ella. ¿Hasta qué distancia d puede subir sin que resbale?"],
            caption="La pared no ejerce fricción.")
        self.analytic(
            [r"N_{pared} = f = \dfrac{(W_e L/2 + W_h d)\cos\theta}{L\sin\theta},\qquad N_{suelo} = W_e + W_h",
             r"f \le \mu_s (W_e + W_h) \;\Rightarrow\; d_{max} = \dfrac{\mu_s(W_e + W_h)L\tan\theta - W_eL/2}{W_h}",
             (rf"d_{{max}} = {dmax_a:.3f}\ \mathrm{{m}}\quad(\text{{de los }}5\ \mathrm{{m}})", ACCENT)],
            size=32, caption="La fricción requerida crece linealmente con la altura del hombre.")

        self.numeric_phase("Para cada posición d se resuelve el sistema lineal de equilibrio 3×3 y se compara f con μs N.")
        sc = 0.78
        foot = np.array([-5.2, -2.7, 0.0])
        th_r = np.deg2rad(th)
        top = foot + L * sc * np.array([np.cos(th_r), np.sin(th_r), 0.0])
        wall = Line(top + np.array([0, 0.45, 0]), np.array([top[0], foot[1], 0.0]), color=FAINT, stroke_width=4)
        floor = Line(foot + LEFT * 0.4, np.array([top[0] + 0.4, foot[1], 0.0]), color=FAINT, stroke_width=3)
        ladder = Line(foot, top, color=FG, stroke_width=6)
        clock = ValueTracker(0.0)
        man = always_redraw(lambda: Dot(foot + clock.get_value() * sc * np.array([np.cos(th_r), np.sin(th_r), 0.0]) + np.array([0.0, 0.0, 0.0]), radius=0.14, color=ACCENT))
        self.play(FadeIn(wall), FadeIn(floor), FadeIn(ladder), FadeIn(man), run_time=0.9)
        ax, deco = axes_block((0, 5), (0, 0.8), 5.6, 3.5, RIGHT * 1.7 + DOWN * 0.15, (0, 1, 2, 3, 4, 5), (0, 0.2, 0.4, 0.6, 0.8), "d [m]", "f / N", "{:g}")
        ds = np.linspace(0, 5, 100)
        ratio = [c14.ladder_forces_analytic(d)["f"] / (wl + wp) for d in ds]
        curve = polyline(ax, ds, ratio, ANALYTIC, 3.2)
        limit = DashedLine(ax.c2p(0, mu), ax.c2p(5, mu), color=ERROR, stroke_width=2, dash_length=0.1)
        lim_lab = T("μs = 0.40", 15, ERROR, mono=True).next_to(ax.c2p(5, mu), UP, buff=0.1).shift(LEFT * 0.7)
        self.play(Create(ax), FadeIn(deco), Create(limit), FadeIn(lim_lab), run_time=1.0)
        self.play(Create(curve), clock.animate.set_value(5.0), run_time=4.0, rate_func=linear)
        dots = VGroup()
        for d in np.linspace(0, 5, 11):
            fr = c14.ladder_forces_numeric(float(d))
            dots.add(Dot(ax.c2p(float(d), fr["f"] / fr["N_floor"]), radius=0.06, color=NUMERIC))
        self.play(LaggedStart(*[FadeIn(x, scale=1.4) for x in dots], lag_ratio=0.1), run_time=1.5)
        mark = DashedLine(ax.c2p(dmax_n, 0), ax.c2p(dmax_n, mu), color=ACCENT, stroke_width=2, dash_length=0.08)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("d_max [m]", f"{dmax_n:.5f}", f"{dmax_a:.5f}"), ("θ_min [°]", f"{amin_n:.4f}", f"{amin_a:.4f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 4.9 + UP * 2.0)
        self.say(f"Los puntos azules (sistema lineal) caen sobre la curva; el cruce con μs da d_max. θ_min: ángulo para subir hasta el extremo.",
                 Create(mark), FadeIn(tab), run_time=1.4)
        self.wait(2.5)


class BeamReactions(PhysicsScene):
    chapter_number = 14
    chapter_title = "Equilibrio de los cuerpos rígidos"

    def construct(self):
        L, wb, wp = c14.BEAM_L, c14.BEAM_W, c14.PERSON_W
        self.foundations(
            [r"\sum F_y = 0:\quad R_A + R_B = W_b + W_p",
             r"\sum \tau_A = 0:\quad R_B L = W_b\,\dfrac{L}{2} + W_p\,x"],
            note="Se puede elegir cualquier punto como eje de torques: el resultado no cambia.",
            caption="Viga apoyada en sus extremos con una carga puntual.")
        self.problem(
            ["Una viga uniforme de 400 N y 6 m descansa sobre dos apoyos en sus extremos.",
             "Una persona de 700 N camina sobre ella. Halle las reacciones en función de su posición x."],
            caption="Los apoyos solo ejercen fuerzas verticales.")
        self.analytic(
            [r"R_B = \dfrac{W_b}{2} + W_p\,\dfrac{x}{L},\qquad R_A = \dfrac{W_b}{2} + W_p\left(1 - \dfrac{x}{L}\right)",
             r"x = 0:\ R_A = W_b/2 + W_p,\qquad x = L:\ R_B = W_b/2 + W_p",
             (rf"x = 2\ \mathrm{{m}}:\ R_A = {c14.beam_reactions_analytic(2.0)[0]:.1f}\ \mathrm{{N}},\ R_B = {c14.beam_reactions_analytic(2.0)[1]:.1f}\ \mathrm{{N}}", ACCENT)],
            size=34, caption="Cada reacción varía linealmente con la posición de la carga.")

        self.numeric_phase("Se resuelve el sistema lineal para cada x tomando torques respecto a tres puntos distintos.")
        ax, deco = axes_block((0, L), (0, 1000), 7.0, 3.6, LEFT * 2.9 + DOWN * 0.15, (0, 2, 4, 6), (0, 250, 500, 750, 1000), "x [m]", "R [N]")
        xs = np.linspace(0, L, 60)
        ra = polyline(ax, xs, [c14.beam_reactions_analytic(float(x))[0] for x in xs], ANALYTIC, 3.2)
        rb = polyline(ax, xs, [c14.beam_reactions_analytic(float(x))[1] for x in xs], ACCENT, 3.2)
        leg = legend_block([(ANALYTIC, "R_A analítico"), (ACCENT, "R_B analítico"), (NUMERIC, "numérico (3 pivotes)")], RIGHT * 4.3 + UP * 2.0)
        self.play(Create(ax), FadeIn(deco), FadeIn(leg), Create(ra), Create(rb), run_time=2.0)
        dots = VGroup()
        for i, x in enumerate(np.linspace(0, L, 13)):
            piv = (0.0, L / 2, L)[i % 3]
            a_, b_ = c14.beam_reactions_numeric(float(x), piv)
            dots.add(Dot(ax.c2p(float(x), a_), radius=0.06, color=NUMERIC), Dot(ax.c2p(float(x), b_), radius=0.06, color=NUMERIC))
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.05), run_time=2.0)
        rows = [(f"{p:.0f}", *[f"{v:.4f}" for v in c14.beam_reactions_numeric(2.0, p)]) for p in (0.0, 3.0, 6.0)]
        tab = mono_table(["pivote [m]", "R_A [N]", "R_B [N]"], rows, size=16, colors=[FG, NUMERIC, NUMERIC], gap=0.35).move_to(RIGHT * 4.4 + DOWN * 0.7)
        note = T("x = 2 m: mismo resultado con cualquier pivote", 14, MUTED).next_to(tab, DOWN, buff=0.3).align_to(tab, LEFT)
        self.say("Los puntos numéricos (con pivotes distintos) caen sobre las rectas: el equilibrio no depende del eje elegido.", FadeIn(tab), FadeIn(note), run_time=1.3)
        self.wait(2.5)
