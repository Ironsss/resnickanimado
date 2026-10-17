"""Capitulo 2. Movimiento en una dimension."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Create, Dot, FadeIn, FadeOut, Line, ValueTracker, VGroup, Write,
    always_redraw, Transform, DashedLine, ReplacementTransform, LaggedStart,
)

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, make_axes, mono_table,
)
from physics import ch03_motion_1d as m1d


class InstantaneousVelocity(PhysicsScene):
    chapter_number = 3
    chapter_title = "Movimiento en una dimensión"

    def construct(self):
        t0 = 2.0
        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"v = \lim_{\Delta t \to 0} \dfrac{\Delta x}{\Delta t} = \dfrac{dx}{dt}", 46),
            M(r"a = \dfrac{dv}{dt} = \dfrac{d^{2}x}{dt^{2}}", 46),
            M(r"\Delta x = \int_{t_0}^{t_1} v\, dt", 46),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(ORIGIN_SHIFT)
        self.say("La velocidad es la pendiente de x(t); el desplazamiento es el área bajo v(t).",
                 *[Write(x) for x in f], run_time=3.0)
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        pos = M(r"x(t) = 2 + 3t - \tfrac{1}{2}t^{2} + \tfrac{1}{10}t^{3}\quad [\mathrm{m},\ \mathrm{s}]", 46).move_to(UP * 1.3)
        ask = T("Determine v y a en t = 2 s, y el desplazamiento entre t = 0 y t = 4 s.", 25, MUTED).move_to(DOWN * 0.5)
        self.say("Una partícula se mueve sobre una recta con la posición dada.", Write(pos), run_time=2.0)
        self.play(FadeIn(ask))
        self.wait(1.2)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        d = VGroup(
            M(r"v(t) = \dfrac{dx}{dt} = 3 - t + 0.3\,t^{2}", 44),
            M(rf"v(2) = {m1d.v_analytic(t0):.2f}\ \mathrm{{m/s}}", 44, ACCENT),
            M(r"a(t) = \dfrac{dv}{dt} = -1 + 0.6\,t", 44),
            M(rf"a(2) = {m1d.a_analytic(t0):.2f}\ \mathrm{{m/s^2}}", 44, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(LEFT * 1.0 + DOWN * 0.1)
        self.say("Se deriva x(t) término a término y se evalúa en t = 2 s.", *[Write(x) for x in d], run_time=4.0)
        self.wait(1.5)
        self.clear_stage()

        # ----------------------------------------- numerica 1: secantes -> tangente
        self.phase("Solución numérica")
        axes = make_axes([0, 6], [0, 26], 6.6, 4.0, 1, 5, numbers=False).move_to(LEFT * 3.2 + DOWN * 0.3)
        xlab, ylab = T("t [s]", 17, MUTED).next_to(axes.x_axis, RIGHT, buff=0.3).shift(DOWN * 0.2), T("x [m]", 17, MUTED).next_to(axes.y_axis, UP, buff=0.1)
        ticks = VGroup(*[T(str(k), 15, MUTED, mono=True).next_to(axes.c2p(k, 0), DOWN, buff=0.1) for k in range(0, 7, 2)])
        curve = axes.plot(m1d.x_of_t, x_range=[0, 6], color=ANALYTIC, stroke_width=3.5)
        self.say("Pendiente de la secante entre t y t + Δt, con Δt cada vez menor.",
                 Create(axes), FadeIn(xlab), FadeIn(ylab), FadeIn(ticks), run_time=1.0)
        self.play(Create(curve), run_time=1.5)

        dt = ValueTracker(1.0)
        p0 = Dot(axes.c2p(t0, m1d.x_of_t(t0)), radius=0.09, color=ACCENT)

        def secant():
            h = dt.get_value()
            a = axes.c2p(t0, m1d.x_of_t(t0))
            b = axes.c2p(t0 + h, m1d.x_of_t(t0 + h))
            direction = (b - a) / np.linalg.norm(b - a)
            return Line(a - direction * 2.0, a + direction * 2.4, color=NUMERIC, stroke_width=3)

        sec = always_redraw(secant)
        p1 = always_redraw(lambda: Dot(axes.c2p(t0 + dt.get_value(), m1d.x_of_t(t0 + dt.get_value())), radius=0.07, color=NUMERIC))
        readout = always_redraw(lambda: VGroup(
            T(f"Δt        = {dt.get_value():.3f} s", 19, NUMERIC, mono=True),
            T(f"pendiente = {m1d.secant_slope(t0, dt.get_value()):.4f} m/s", 19, NUMERIC, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 3.7 + UP * 1.6))
        self.play(FadeIn(p0), FadeIn(sec), FadeIn(p1), FadeIn(readout))
        for target in (0.5, 0.1, 0.01):
            self.play(dt.animate.set_value(target), run_time=1.6)
            self.wait(0.3)
        tab = mono_table(["Δt [s]", "pendiente", "error"],
                         [(f"{r['dt']:g}", f"{r['slope']:.4f}", f"{r['error']:.1e}") for r in m1d.secant_table(t0)],
                         size=19, colors=[FG, NUMERIC, MUTED], gap=0.55).move_to(RIGHT * 3.6 + DOWN * 0.3)
        exact = T(f"v(2) analítica = {m1d.v_analytic(t0):.4f} m/s", 19, ACCENT, mono=True).next_to(tab, DOWN, buff=0.35).align_to(tab, LEFT)
        a_fd = m1d.central_second_difference(t0, 1e-2)
        a_line = T(f"a(2) por diferencias = {a_fd:.4f} m/s²", 19, NUMERIC, mono=True).next_to(exact, DOWN, buff=0.2).align_to(tab, LEFT)
        self.say("La pendiente converge a v(2); la segunda diferencia centrada da a(2) = 0.2 m/s².",
                 FadeOut(readout), FadeIn(tab), run_time=1.4)
        self.play(FadeIn(exact), FadeIn(a_line))
        self.wait(2.0)
        self.clear_stage()

        # ------------------------------------------ numerica 2: area bajo v(t)
        self.phase("Solución numérica")
        axv = make_axes([0, 4], [0, 8], 6.6, 3.8, 1, 2, numbers=False).move_to(LEFT * 3.2 + DOWN * 0.3)
        vcurve = axv.plot(m1d.v_analytic, x_range=[0, 4], color=ANALYTIC, stroke_width=3.5)
        vticks = VGroup(*[T(str(k), 15, MUTED, mono=True).next_to(axv.c2p(k, 0), DOWN, buff=0.1) for k in range(0, 5)])
        vlab = T("v [m/s]", 17, MUTED).next_to(axv.y_axis, UP, buff=0.1)
        self.say("El desplazamiento es el área bajo v(t); se aproxima con rectángulos y con trapecios.",
                 Create(axv), Create(vcurve), FadeIn(vticks), FadeIn(vlab), run_time=1.6)
        exact_disp = m1d.x_of_t(4.0) - m1d.x_of_t(0.0)
        prev = None
        txt_prev = None
        for n in (4, 8, 32):
            rects = axv.get_riemann_rectangles(vcurve, x_range=[0, 4], dx=4.0 / n, input_sample_type="left",
                                               stroke_width=0.5, stroke_color=BG_COLOR, fill_opacity=0.55, color=NUMERIC)
            d = m1d.displacement_from_velocity(0.0, 4.0, n)
            txt = VGroup(
                T(f"n = {n}", 20, FG, mono=True),
                T(f"Riemann  {d['riemann']:.4f} m", 20, NUMERIC, mono=True),
                T(f"trapecio {d['trapezoid']:.4f} m", 20, NUMERIC, mono=True),
                T(f"exacto   {exact_disp:.4f} m", 20, ACCENT, mono=True),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 4.0 + UP * 0.2)
            if prev is None:
                self.play(Create(rects), FadeIn(txt), run_time=1.2)
            else:
                self.play(ReplacementTransform(prev, rects), ReplacementTransform(txt_prev, txt), run_time=1.2)
            prev, txt_prev = rects, txt
            self.wait(0.6)
        self.wait(1.5)


ORIGIN_SHIFT = np.array([-1.0, -0.1, 0.0])
BG_COLOR = "#09090b"


class ConstantAcceleration(PhysicsScene):
    chapter_number = 3
    chapter_title = "Movimiento en una dimensión"

    def construct(self):
        v0, g = m1d.V0, m1d.G
        s = m1d.free_fall_summary()
        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f = VGroup(
            M(r"v = v_0 + a\,t", 46),
            M(r"x = x_0 + v_0 t + \tfrac{1}{2}a\,t^{2}", 46),
            M(r"v^{2} = v_0^{2} + 2a\,(x - x_0)", 46),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(LEFT * 3.0 + DOWN * 0.1)
        side = T("Caída libre:\na = −g\ng = 9.81 m/s²", 26, MUTED).move_to(RIGHT * 3.8 + DOWN * 0.1)
        self.say("Con aceleración constante, las tres ecuaciones cinemáticas son exactas.",
                 *[Write(x) for x in f], run_time=3.0)
        self.play(FadeIn(side))
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = T("Una pelota se lanza hacia arriba desde el suelo con v₀ = 20 m/s.", 27, FG).move_to(UP * 1.3)
        ask = T("Calcule la altura máxima, el tiempo de vuelo y la altura en t = 3 s.", 25, MUTED).move_to(DOWN * 0.1)
        self.say("Se desprecia la resistencia del aire.", FadeIn(q), run_time=1.0)
        self.play(FadeIn(ask))
        self.wait(1.2)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        y3 = m1d.free_fall_analytic(3.0)[0]
        d = VGroup(
            M(rf"v = 0 \;\Rightarrow\; t_{{\max}} = \dfrac{{v_0}}{{g}} = {s['t_peak']:.3f}\ \mathrm{{s}}", 40),
            M(rf"y_{{\max}} = \dfrac{{v_0^{{2}}}}{{2g}} = {s['y_max']:.3f}\ \mathrm{{m}}", 40, ACCENT),
            M(rf"y = 0 \;\Rightarrow\; t = \dfrac{{2v_0}}{{g}} = {s['t_return']:.3f}\ \mathrm{{s}}", 40, ACCENT),
            M(rf"y(3) = 20(3) - \tfrac{{1}}{{2}}(9.81)(3)^{{2}} = {float(y3):.3f}\ \mathrm{{m}}", 40, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(LEFT * 0.4 + DOWN * 0.1)
        self.say("En el punto más alto v = 0; la subida y la bajada tardan lo mismo.", *[Write(x) for x in d], run_time=5.0)
        self.wait(1.5)
        self.clear_stage()

        # ----------------------------------------------------------- numerica
        self.phase("Solución numérica")
        axes = make_axes([0, 4.2], [0, 24], 6.4, 3.9, 1, 5, numbers=False).move_to(LEFT * 3.4 + DOWN * 0.3)
        ticks = VGroup(*[T(str(k), 15, MUTED, mono=True).next_to(axes.c2p(k, 0), DOWN, buff=0.1) for k in range(0, 5)])
        yticks = VGroup(*[T(str(k), 15, MUTED, mono=True).next_to(axes.c2p(0, k), LEFT, buff=0.1) for k in (10, 20)])
        xlab, ylab = T("t [s]", 17, MUTED).next_to(axes.x_axis, RIGHT, buff=0.3).shift(DOWN * 0.2), T("y [m]", 17, MUTED).next_to(axes.y_axis, UP, buff=0.1)
        curve = axes.plot(lambda t: m1d.free_fall_analytic(t)[0], x_range=[0, s["t_return"]], color=ANALYTIC, stroke_width=3.5)
        self.say("Paso de tiempo grande (Δt = 0.3 s): Euler acumula error, RK4 permanece sobre la curva.",
                 Create(axes), FadeIn(ticks), FadeIn(yticks), FadeIn(xlab), FadeIn(ylab), Create(curve), run_time=1.8)
        te, ye = m1d.free_fall_numeric(10, "euler")
        tr, yr = m1d.free_fall_numeric(10, "rk4")
        euler = VGroup(*[Dot(axes.c2p(t, y[0]), radius=0.07, color=ERROR) for t, y in zip(te, ye)])
        rk4 = VGroup(*[Dot(axes.c2p(t, y[0]), radius=0.07, color=NUMERIC) for t, y in zip(tr, yr)])
        leg = VGroup(
            VGroup(Dot(radius=0.07, color=ERROR), T("Euler", 18, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(radius=0.07, color=NUMERIC), T("RK4", 18, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
            VGroup(Line(LEFT * 0.12, RIGHT * 0.12, color=ANALYTIC), T("analítica", 18, MUTED, mono=True)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(RIGHT * 3.2 + UP * 1.9)
        self.play(FadeIn(leg), LaggedStart(*[FadeIn(d) for d in euler], lag_ratio=0.15), run_time=1.8)
        self.play(LaggedStart(*[FadeIn(d) for d in rk4], lag_ratio=0.15), run_time=1.8)
        rows = m1d.free_fall_convergence()
        tab = mono_table(["n", "error Euler [m]", "error RK4 [m]"],
                         [(r["n"], f"{r['err_euler']:.2e}", f"{r['err_rk4']:.1e}") for r in rows],
                         size=18, colors=[FG, ERROR, NUMERIC], gap=0.5).move_to(RIGHT * 3.8 + DOWN * 0.6)
        note = T("error en y(3 s) al refinar el paso", 17, MUTED).next_to(tab, UP, buff=0.3).align_to(tab, LEFT)
        self.say("Euler es de primer orden (el error se divide por 2 al duplicar n); RK4 reproduce el polinomio exacto.",
                 FadeIn(note), FadeIn(tab), run_time=1.4)
        self.wait(2.5)
