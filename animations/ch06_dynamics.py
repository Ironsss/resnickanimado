"""Capitulo 6. Dinamica de la particula."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Arc, Create, DashedLine, Dot, FadeIn, FadeOut, LaggedStart, Line, Polygon, Square,
    ValueTracker, VGroup, always_redraw, linear,
)

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block,
    mono_table, polyline, text_lines,
)
from physics import ch06_dynamics as c6


class FrictionIncline(PhysicsScene):
    chapter_number = 6
    chapter_title = "Dinámica de la partícula"

    def construct(self):
        th, mus, muk = c6.THETA_DEG, c6.MU_S, c6.MU_K
        a_an = c6.incline_acceleration_analytic(th)
        thc_an, thc_nu = c6.critical_angle_analytic(), c6.critical_angle_numeric()
        t, s, v = c6.simulate_incline(th, 2.0, dt=2e-3)

        self.foundations(
            [r"f_s \le \mu_s N,\qquad f_k = \mu_k N,\qquad N = mg\cos\theta",
             r"\sum F_\parallel = mg\sin\theta - f_k = ma"],
            note="La fricción estática se opone al deslizamiento incipiente; la cinética actúa mientras hay movimiento.",
            caption="La fuerza normal fija el máximo de fricción disponible.")
        self.problem(
            ["Un bloque descansa sobre un plano inclinado 35° con μs = 0.45 y μk = 0.30.",
             "¿Desliza? Si lo hace, ¿con qué aceleración? ¿A partir de qué ángulo comienza a deslizar?"],
            caption="Se suelta desde el reposo.")
        self.analytic(
            [rf"\tan\theta = {np.tan(np.deg2rad(th)):.3f} > \mu_s = {mus} \;\Rightarrow\; \text{{desliza}}",
             (rf"a = g(\sin\theta - \mu_k\cos\theta) = {a_an:.3f}\ \mathrm{{m/s^2}}", ACCENT),
             (rf"\theta_c = \tan^{{-1}}\mu_s = {thc_an:.2f}^\circ", ACCENT)],
            caption="Si tan θ supera a μs, la fricción estática no alcanza para sostener el bloque.")

        # ---------------------------------------------------------- numerica
        self.numeric_phase("Se integra el bloque con RK4 y una máquina de estados: reposo con fricción estática, o deslizamiento con cinética.")
        th_r = np.deg2rad(th)
        n = np.array([-np.sin(th_r), np.cos(th_r), 0.0])
        u = np.array([np.cos(th_r), np.sin(th_r), 0.0])
        p0 = np.array([-6.3, -2.75, 0.0])
        wedge = Polygon(p0, p0 + 5.4 * u, np.array([p0[0] + 5.4 * np.cos(th_r), p0[1], 0.0]),
                        stroke_color=FAINT, stroke_width=2, fill_color=PANEL, fill_opacity=1)
        floor = Line(p0 + LEFT * 0.3, np.array([p0[0] + 5.4 * np.cos(th_r) + 0.4, p0[1], 0.0]), color=FAINT, stroke_width=2)
        ang = Arc(radius=0.9, start_angle=0, angle=th_r, arc_center=p0, color=ACCENT, stroke_width=2.5)
        ang_lab = M(r"35^\circ", 24, ACCENT).move_to(p0 + np.array([1.3, 0.22, 0.0]))
        clock = ValueTracker(0.0)

        def center():
            d = float(np.interp(clock.get_value(), t, s)) * 0.33
            return p0 + (1.0 + d) * u + 0.3 * n

        block = always_redraw(lambda: Square(0.6, stroke_color=FG, stroke_width=2.5, fill_color=PANEL, fill_opacity=1)
                              .rotate(th_r).move_to(center()))
        readout = always_redraw(lambda: text_lines([
            f"t = {clock.get_value():4.2f} s",
            (f"s = {float(np.interp(clock.get_value(), t, s)):5.2f} m", NUMERIC),
            (f"v = {float(np.interp(clock.get_value(), t, v)):5.2f} m/s", NUMERIC)], 19)
            .move_to(RIGHT * 3.4 + UP * 1.4))
        self.play(FadeIn(wedge), FadeIn(floor), FadeIn(ang), FadeIn(ang_lab), FadeIn(block), FadeIn(readout), run_time=1.0)
        self.play(clock.animate.set_value(2.0), run_time=3.5, rate_func=linear)
        tab = mono_table(["método", "a [m/s²]", "s(2 s) [m]"],
                         [("RK4 + estados", f"{v[-1] / 2.0:.4f}", f"{s[-1]:.4f}"),
                          ("analítico", f"{a_an:.4f}", f"{0.5 * a_an * 4:.4f}")],
                         size=18, colors=[MUTED, NUMERIC, NUMERIC], gap=0.45).move_to(RIGHT * 3.7 + DOWN * 0.7)
        self.play(FadeIn(tab), run_time=1.0)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (wedge, floor, ang, ang_lab, block, readout, tab)], run_time=0.7)

        # barrido del angulo
        self.say("Barrido del ángulo del plano: la aceleración es nula hasta θc y salta cuando el bloque empieza a deslizar.", run_time=0.5)
        ax, deco = axes_block((0, 60), (0, 8), 7.4, 3.7, LEFT * 2.7 + DOWN * 0.15, (0, 15, 30, 45, 60), (0, 2, 4, 6, 8), "θ [°]", "a [m/s²]")
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        grid = np.linspace(0, 60, 241)
        an_curve = polyline(ax, grid, c6.incline_sweep(grid), ANALYTIC, 3.0)
        self.play(Create(an_curve), run_time=1.5)
        pts = VGroup()
        for a_deg in range(5, 60, 5):
            _, ss, _ = c6.simulate_incline(a_deg, 1.0, dt=5e-3)
            pts.add(Dot(ax.c2p(a_deg, 2 * ss[-1] / 1.0), radius=0.07, color=NUMERIC))
        self.play(LaggedStart(*[FadeIn(p, scale=1.5) for p in pts], lag_ratio=0.12), run_time=2.0)
        mark = DashedLine(ax.c2p(thc_nu, 0), ax.c2p(thc_nu, 8), color=ACCENT, stroke_width=2, dash_length=0.08)
        info = text_lines([
            (f"θc numérico  {thc_nu:.2f}°", NUMERIC),
            (f"θc analítico {thc_an:.2f}°", ANALYTIC),
            ("a = 2s/t² con t = 1 s", MUTED)], 17).move_to(RIGHT * 4.5 + UP * 0.2)
        self.play(Create(mark), FadeIn(info), run_time=1.2)
        self.wait(2.5)


class DragFall(PhysicsScene):
    chapter_number = 6
    chapter_title = "Dinámica de la partícula"

    def construct(self):
        m, b, g = c6.MASS, c6.DRAG_B, c6.G
        vt, ta = c6.terminal_speed(), c6.tau()
        t95_an, t95_nu = c6.time_to_fraction_analytic(), c6.time_to_fraction_numeric()

        self.foundations(
            [r"\vec F_{res} = -b\,\vec v",
             r"m\dfrac{dv}{dt} = mg - bv",
             r"v_t = \dfrac{mg}{b},\qquad \tau = \dfrac{m}{b}"],
            note="La resistencia crece con la rapidez hasta equilibrar al peso: la aceleración se anula y v tiende a v_t.",
            caption="Resistencia lineal en la velocidad (régimen viscoso).")
        self.problem(
            ["Una esfera de 0.5 kg cae desde el reposo en un medio con b = 0.8 kg/s.",
             "Determine v(t), la rapidez terminal y el tiempo para alcanzar el 95 % de ella."],
            caption="La caída es vertical y g = 9.81 m/s².")
        self.analytic(
            [r"v(t) = v_t\left(1 - e^{-t/\tau}\right)",
             (rf"v_t = {vt:.3f}\ \mathrm{{m/s}},\qquad \tau = {ta:.3f}\ \mathrm{{s}}", ACCENT),
             (rf"t_{{95}} = \tau\ln 20 = {t95_an:.3f}\ \mathrm{{s}}", ACCENT)],
            caption="Se separan variables en la ecuación de movimiento y se integra desde v = 0.")

        self.numeric_phase("Con un paso grande (Δt = 0.5 s), Euler se aparta de la curva exacta y RK4 permanece sobre ella.")
        ax, deco = axes_block((0, 4), (0, 7), 7.2, 3.8, LEFT * 2.8 + DOWN * 0.1, (0, 1, 2, 3, 4), (0, 2, 4, 6), "t [s]", "v [m/s]")
        grid = np.linspace(0, 4, 200)
        asym = DashedLine(ax.c2p(0, vt), ax.c2p(4, vt), color=FAINT, stroke_width=1.5, dash_length=0.1)
        asym_lab = T("v_t", 15, MUTED, mono=True).next_to(ax.c2p(4, vt), RIGHT, buff=0.1)
        curve = polyline(ax, grid, c6.drag_velocity_analytic(grid), ANALYTIC, 3.5)
        self.play(Create(ax), FadeIn(deco), Create(asym), FadeIn(asym_lab), run_time=1.0)
        self.play(Create(curve), run_time=1.5)
        te, ye = c6.drag_numeric(8, "euler")
        tr, yr = c6.drag_numeric(8, "rk4")
        euler = VGroup(*[Dot(ax.c2p(x, y[1]), radius=0.07, color=ERROR) for x, y in zip(te, ye)])
        rk4 = VGroup(*[Dot(ax.c2p(x, y[1]), radius=0.07, color=NUMERIC) for x, y in zip(tr, yr)])
        leg = legend_block([(ANALYTIC, "analítica"), (ERROR, "Euler"), (NUMERIC, "RK4")], RIGHT * 4.9 + UP * 1.9)
        self.play(FadeIn(leg), LaggedStart(*[FadeIn(d) for d in euler], lag_ratio=0.2), run_time=1.8)
        self.play(LaggedStart(*[FadeIn(d) for d in rk4], lag_ratio=0.2), run_time=1.8)
        rows = c6.drag_convergence()
        tab = mono_table(["n", "err Euler", "err RK4"],
                         [(r["n"], f"{r['err_euler']:.1e}", f"{r['err_rk4']:.1e}") for r in rows],
                         size=17, colors=[FG, ERROR, NUMERIC], gap=0.4).move_to(RIGHT * 4.6 + DOWN * 0.35)
        note = T("error en v(1 s)", 16, MUTED).next_to(tab, UP, buff=0.25).align_to(tab, LEFT)
        t95 = text_lines([(f"t95 RK4       {t95_nu:.4f} s", NUMERIC), (f"t95 analítico {t95_an:.4f} s", ANALYTIC)], 17).move_to(RIGHT * 4.6 + DOWN * 2.3)
        self.say("Euler es de primer orden y RK4 de cuarto: el error de RK4 cae mucho más rápido al refinar el paso.",
                 FadeIn(note), FadeIn(tab), FadeIn(t95), run_time=1.4)
        self.wait(2.5)
