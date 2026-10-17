"""Capitulo 19. Movimiento ondulatorio."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, ValueTracker, VGroup, always_redraw, linear

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch19_waves as c19


class WavePulse(PhysicsScene):
    chapter_number = 19
    chapter_title = "Movimiento ondulatorio"

    def construct(self):
        v, mu, tension = c19.V, c19.MU, c19.TENSION
        snaps_t = tuple(np.linspace(0.0, 0.15, 31))
        x, dt, snaps = c19.fdtd(800, 0.9, snaps_t)
        keys = sorted(snaps)
        err9 = c19.fdtd_error(0.9)
        err1 = c19.fdtd_error(1.0)
        self.foundations(
            [r"\dfrac{\partial^{2}y}{\partial t^{2}} = v^{2}\dfrac{\partial^{2}y}{\partial x^{2}},\qquad v = \sqrt{T/\mu}",
             r"y(x,t) = f(x - vt) + g(x + vt)"],
            note="Cualquier perfil que se propague sin deformarse es solución de la ecuación de onda.",
            caption="Ecuación de onda en una cuerda tensa.")
        self.problem(
            ["Una cuerda con T = 80 N y μ = 0.05 kg/m recibe un pulso gaussiano (σ = 0.6 m) centrado en x = 10 m, inicialmente en reposo.",
             "Halle la velocidad de la onda y la forma del pulso a los 0.15 s."],
            caption="Se desprecia la rigidez de la cuerda.")
        self.analytic(
            [(rf"v = \sqrt{{T/\mu}} = {v:.1f}\ \mathrm{{m/s}}", ACCENT),
             r"y(x,t) = \tfrac{1}{2}\left[f(x - vt) + f(x + vt)\right]\qquad(\text{d'Alembert})",
             rf"t = 0.15\ \mathrm{{s}}:\quad \text{{dos pulsos de amplitud }}1/2\text{{ en }}x = {10 - v * 0.15:.0f}\ \mathrm{{m}}\ \text{{y}}\ x = {10 + v * 0.15:.0f}\ \mathrm{{m}}"],
            size=34, caption="Un pulso en reposo se divide en dos mitades que viajan en sentidos opuestos.")

        self.numeric_phase("Diferencias finitas (leapfrog) con 800 intervalos y número de Courant 0.9; la solución exacta se superpone en blanco.")
        ax, deco = axes_block((0, 20), (-0.2, 1.1), 11.4, 3.0, LEFT * 0.4 + UP * 0.5, (0, 5, 10, 15, 20), (0, 0.5, 1.0), "x [m]", "y [m]")
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        clock = ValueTracker(0.0)

        def idx():
            tt = clock.get_value()
            return min(range(len(keys)), key=lambda i: abs(keys[i] - tt))

        num_c = always_redraw(lambda: polyline(ax, x[::4], snaps[keys[idx()]][::4], NUMERIC, 3.5))
        an_c = always_redraw(lambda: polyline(ax, x[::4], c19.dalembert(x[::4], keys[idx()]), ANALYTIC, 1.8))
        tl = always_redraw(lambda: T(f"t = {keys[idx()]:.3f} s", 18, MUTED, mono=True).move_to(RIGHT * 5.0 + UP * 2.4))
        leg = legend_block([(NUMERIC, "diferencias finitas"), (ANALYTIC, "d'Alembert")], RIGHT * 4.9 + UP * 1.5)
        self.add(num_c, an_c, tl)
        self.play(FadeIn(leg), clock.animate.set_value(0.15), run_time=7.0, rate_func=linear)
        tab = mono_table(["Courant", "error máx. a 0.15 s"],
                         [("1.0", f"{err1:.1e}"), ("0.9", f"{err9:.1e}"), ("0.5 (n = 1600)", f"{c19.fdtd_error(0.5, 1600):.1e}")],
                         size=16, colors=[FG, NUMERIC], gap=0.5).move_to(LEFT * 0.4 + DOWN * 2.35)
        self.say("Con Courant = 1 el esquema es exacto para la ecuación de onda; con 0.9 aparece una pequeña dispersión numérica.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class StandingModes(PhysicsScene):
    chapter_number = 19
    chapter_title = "Movimiento ondulatorio"

    def construct(self):
        L, v = c19.STRING_L, c19.V
        freqs, xs, shapes = c19.modes_numeric()
        self.foundations(
            [r"y_n(x,t) = A_n\sin\!\left(\dfrac{n\pi x}{L}\right)\cos(\omega_n t),\qquad n = 1, 2, 3, \ldots",
             r"f_n = \dfrac{nv}{2L}"],
            note="Los extremos fijos imponen nodos en x = 0 y x = L: solo caben un número entero de semilongitudes de onda.",
            caption="Modos normales de una cuerda con extremos fijos.", size=38)
        self.problem(
            ["Una cuerda de 1.2 m con v = 40 m/s está fija en ambos extremos.",
             "Halle las frecuencias de los primeros modos y verifique que las formas son senoidales."],
            caption="Modos de vibración y armónicos.")
        self.analytic(
            [r"\lambda_n = \dfrac{2L}{n},\qquad f_n = \dfrac{nv}{2L}",
             (rf"f_1 = {c19.mode_frequency_analytic(1):.2f}\ \mathrm{{Hz}},\quad f_2 = {c19.mode_frequency_analytic(2):.2f}\ \mathrm{{Hz}},\quad f_3 = {c19.mode_frequency_analytic(3):.2f}\ \mathrm{{Hz}}", ACCENT)],
            caption="Las frecuencias son múltiplos enteros de la fundamental.")

        self.numeric_phase("Se diagonaliza la matriz tridiagonal de la segunda derivada (400 puntos): los autovalores dan ω_n y los autovectores, las formas.")
        clock = ValueTracker(0.0)
        rows_y = [1.9, 0.75, -0.4, -1.55]
        panels = VGroup()
        for n, ycen in enumerate(rows_y, start=1):
            base = Line(np.array([-6.2, ycen, 0]), np.array([-1.0, ycen, 0]), color=FAINT, stroke_width=1.5)
            panels.add(base, T(f"n = {n}", 15, MUTED, mono=True).move_to(np.array([-6.5, ycen + 0.45, 0])))
        self.play(FadeIn(panels), run_time=0.8)

        def make(n):
            ycen = rows_y[n - 1]

            def draw():
                ph = np.cos(2 * np.pi * freqs[n - 1] * clock.get_value() * 0.04)
                pts = [np.array([-6.2 + 5.2 * xx / L, ycen + 0.42 * float(shapes[n - 1][j]) * ph, 0.0])
                       for j, xx in zip(range(0, len(xs), 6), xs[::6])]
                from manim import VMobject
                m_ = VMobject(color=NUMERIC, stroke_width=3)
                m_.set_points_as_corners(pts)
                ref = [np.array([-6.2 + 5.2 * xx / L, ycen + 0.42 * float(np.sin(n * np.pi * xx / L)) * ph, 0.0]) for xx in xs[::6]]
                m2 = VMobject(color=ANALYTIC, stroke_width=1.5)
                m2.set_points_as_corners(ref)
                return VGroup(m2, m_)
            return draw

        waves = [always_redraw(make(n)) for n in range(1, 5)]
        for w_ in waves:
            self.add(w_)
        self.play(clock.animate.set_value(60.0), run_time=6.0, rate_func=linear)
        tab = mono_table(["n", "f numérico [Hz]", "f analítico [Hz]"],
                         [(n, f"{freqs[n - 1]:.4f}", f"{c19.mode_frequency_analytic(n):.4f}") for n in range(1, 6)],
                         size=16, colors=[FG, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 4.0 + UP * 0.2)
        err = text_lines([(f"|forma n=1 − sen(πx/L)| máx = {np.max(np.abs(shapes[0] - np.sin(np.pi * xs / L))):.1e}", MUTED)], 14).move_to(RIGHT * 4.0 + DOWN * 1.9)
        self.say("Los autovalores numéricos reproducen f_n = n v / 2L, y los autovectores (azul) coinciden con sen(nπx/L) (blanco).", FadeIn(tab), FadeIn(err), run_time=1.3)
        self.wait(2.5)
