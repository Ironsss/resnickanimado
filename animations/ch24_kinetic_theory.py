"""Capitulo 24. Teoria cinetica de los gases."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, ValueTracker, VGroup, always_redraw, linear, Transform, ReplacementTransform

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch24_kinetic_theory as c24


class GasPressure(PhysicsScene):
    chapter_number = 24
    chapter_title = "Teoría cinética de los gases"

    def construct(self):
        n = 4000
        p_sim, t_sim = c24.pressure_simulation(n=n)
        p_an = c24.pressure_analytic(n, t_sim)
        self.foundations(
            [r"p = \dfrac{1}{3}\dfrac{N}{V}\,m\langle v^{2}\rangle = \dfrac{NkT}{V}",
             r"\tfrac{1}{2}m\langle v^{2}\rangle = \tfrac{3}{2}kT"],
            note="La presión es el impulso medio por unidad de tiempo y de área que las moléculas transfieren a las paredes.",
            caption="Interpretación cinética de la presión y de la temperatura.", size=40)
        self.problem(
            ["4000 átomos de helio a 300 K se mueven libremente en un cubo de 1 µm de lado y rebotan elásticamente en las paredes.",
             "Calcule la presión por el impulso transferido a una pared y compárela con p = NkT/V."],
            caption="Las moléculas no interactúan entre sí.")
        self.analytic(
            [(rf"p = \dfrac{{NkT}}{{V}} = {c24.pressure_analytic(n):.3f}\ \mathrm{{Pa}}", ACCENT),
             rf"v_{{rms}} = \sqrt{{3kT/m}} = {np.sqrt(3 * c24.KB * 300 / c24.M_HE):.0f}\ \mathrm{{m/s}}"],
            size=38, caption="La rapidez cuadrática media del helio a 300 K es de más de un kilómetro por segundo.")

        self.numeric_phase("Se simulan 4000 átomos con pasos de 2 ps: cada rebote en la pared x = L aporta un impulso 2m|vx|.")
        rng = np.random.default_rng(4)
        box = Rectangle(width=4.2, height=4.2, stroke_color=FG, stroke_width=3).move_to(np.array([-3.4, -0.2, 0.0]))
        pos = rng.uniform(-2.0, 2.0, (40, 2))
        vel = rng.normal(0, 1.0, (40, 2)) * 1.6
        dots = VGroup(*[Dot(radius=0.05, color=NUMERIC) for _ in range(40)])
        wall = Line(np.array([-1.3, 1.9, 0.0]), np.array([-1.3, -2.3, 0.0]), color=ACCENT, stroke_width=5)

        def step(mob, dt):
            global_pos = pos
            global_pos += vel * dt
            for i in range(2):
                hi = global_pos[:, i] > 2.0
                lo = global_pos[:, i] < -2.0
                vel[hi, i] *= -1
                vel[lo, i] *= -1
                global_pos[hi, i] = 4.0 - global_pos[hi, i]
                global_pos[lo, i] = -4.0 - global_pos[lo, i]
            for d, p in zip(mob, global_pos):
                d.move_to(np.array([-3.4 + p[0], -0.2 + p[1], 0.0]))
        dots.add_updater(step)
        self.play(FadeIn(box), FadeIn(wall), FadeIn(dots), run_time=0.8)
        self.wait(5.0)
        dots.clear_updaters()
        tab = mono_table(["magnitud", "simulación", "NkT/V"],
                         [("p [Pa]", f"{p_sim:.3f}", f"{p_an:.3f}"), ("T [K]", f"{t_sim:.1f}", "300.0")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 3.6 + UP * 0.8)
        rel = text_lines([(f"error relativo de p: {abs(p_sim - p_an) / p_an * 100:.2f} %", MUTED),
                          ("(T medida de la energía cinética de la simulación)", MUTED)], 14).move_to(RIGHT * 3.6 + DOWN * 0.7)
        self.say("La presión obtenida del impulso en la pared coincide con NkT/V usando la temperatura medida de la propia simulación.", FadeIn(tab), FadeIn(rel), run_time=1.3)
        self.wait(2.5)


class MaxwellBoltzmann(PhysicsScene):
    chapter_number = 24
    chapter_title = "Teoría cinética de los gases"

    def construct(self):
        an = c24.speeds_analytic()
        sp = c24.speeds_sampled()
        q = c24.moments_by_quadrature()
        self.foundations(
            [r"f(v) = 4\pi\left(\dfrac{m}{2\pi kT}\right)^{3/2}v^{2}\,e^{-mv^{2}/2kT}",
             r"v_p = \sqrt{\dfrac{2kT}{m}},\qquad \bar v = \sqrt{\dfrac{8kT}{\pi m}},\qquad v_{rms} = \sqrt{\dfrac{3kT}{m}}"],
            note="Cada componente cartesiana de la velocidad sigue una distribución normal; la rapidez resulta de combinar las tres.",
            caption="Distribución de rapideces de Maxwell-Boltzmann.", size=36)
        self.problem(
            ["Nitrógeno (N₂) a 300 K. Halle las rapideces más probable, media y cuadrática media,",
             "y verifique la distribución con un muestreo de 400 000 moléculas y con cuadratura de la densidad."],
            caption="Gas ideal en equilibrio térmico.")
        self.analytic(
            [(rf"v_p = {an['v_mp']:.1f}\ \mathrm{{m/s}},\quad \bar v = {an['v_mean']:.1f}\ \mathrm{{m/s}},\quad v_{{rms}} = {an['v_rms']:.1f}\ \mathrm{{m/s}}", ACCENT),
             r"v_p < \bar v < v_{rms}"],
            size=36, caption="La distribución es asimétrica: la cola de rapideces altas desplaza los promedios.")

        self.numeric_phase("Se muestrean las tres componentes de la velocidad con distribución normal y se compara el histograma con f(v).")
        ax, deco = axes_block((0, 1500), (0, 0.0030), 7.4, 3.6, LEFT * 2.7 + DOWN * 0.15, (0, 300, 600, 900, 1200, 1500), (0, 0.001, 0.002, 0.003), "v [m/s]", "f(v)")
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        centers, hist = sp["centers"], sp["hist"]
        from manim import Rectangle as Rect
        bars = VGroup()
        w_bin = (centers[1] - centers[0])
        for cx, h in zip(centers, hist):
            p0, p1 = ax.c2p(cx - w_bin / 2, 0), ax.c2p(cx + w_bin / 2, h)
            r_ = Rect(width=(p1 - p0)[0] * 0.92, height=max((p1 - p0)[1], 1e-3), stroke_width=0, fill_color=NUMERIC, fill_opacity=0.65)
            r_.move_to([(p0[0] + p1[0]) / 2, p0[1] + r_.height / 2, 0])
            bars.add(r_)
        self.play(FadeIn(bars), run_time=1.5)
        vs = np.linspace(0, 1500, 150)
        curve = polyline(ax, vs, c24.mb_pdf(vs), ANALYTIC, 3.5)
        self.play(Create(curve), run_time=1.5)
        marks = VGroup()
        for key, col in (("v_mp", ACCENT), ("v_mean", FG), ("v_rms", ERROR)):
            marks.add(DashedLine(ax.c2p(an[key], 0), ax.c2p(an[key], 0.0030), color=col, stroke_width=1.8, dash_length=0.08))
        self.play(Create(marks), run_time=1.0)
        tab = mono_table(["rapidez [m/s]", "muestreo", "cuadratura", "analítico"],
                         [("v más prob.", f"{sp['v_mp']:.0f}", "", f"{an['v_mp']:.1f}"), ("v media", f"{sp['v_mean']:.1f}", f"{q['v_mean']:.1f}", f"{an['v_mean']:.1f}"),
                          ("v rms", f"{sp['v_rms']:.1f}", f"{q['v_rms']:.1f}", f"{an['v_rms']:.1f}")],
                         size=13, colors=[MUTED, NUMERIC, FG, ANALYTIC], gap=0.16).move_to(RIGHT * 4.45 + UP * 0.6)
        leg = legend_block([(ACCENT, "v_p"), (FG, "v media"), (ERROR, "v_rms")], RIGHT * 5.0 + UP * 2.0)
        norm = text_lines([(f"∫f dv = {q['norm']:.6f}", MUTED)], 14).move_to(RIGHT * 4.7 + DOWN * 1.1)
        self.say("El histograma sigue a f(v); la moda muestral tiene la resolución del ancho de bin (25 m/s).", FadeIn(tab), FadeIn(leg), FadeIn(norm), run_time=1.3)
        self.wait(2.5)
