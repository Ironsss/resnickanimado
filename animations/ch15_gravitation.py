"""Capitulo 15. Gravitacion."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Polygon, ValueTracker, VGroup, always_redraw, linear, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch15_gravitation as c15


class KeplerOrbit(PhysicsScene):
    chapter_number = 15
    chapter_title = "Gravitación"

    def construct(self):
        a, e = c15.A_ORBIT, c15.E_ORBIT
        T_an, T_nu = c15.period_analytic(), c15.period_numeric()
        t, y = c15.orbit_numeric(revolutions=1.0)
        self.foundations(
            [r"F = \dfrac{GMm}{r^{2}}",
             r"\text{(1) elipse con el Sol en un foco}\quad \text{(2) } \dfrac{dA}{dt} = \text{cte}\quad \text{(3) } T^{2} \propto a^{3}"],
            note="Las tres leyes de Kepler se deducen de la ley de gravitación universal de Newton.",
            caption="Gravitación universal y leyes de Kepler.", size=40)
        self.problem(
            ["Un planeta orbita el Sol con semieje a = 2 UA y excentricidad e = 0.5, partiendo del perihelio.",
             "Halle su periodo, y verifique numéricamente la segunda y la tercera ley."],
            caption="Unidades: UA y años, con GM = 4π² UA³/año².")
        self.analytic(
            [r"T^{2} = a^{3}\quad(\text{UA, años})",
             (rf"T = a^{{3/2}} = {T_an:.4f}\ \text{{años}}", ACCENT),
             rf"r_p = a(1 - e) = {a * (1 - e):.1f}\ \mathrm{{UA}},\qquad v_p = \sqrt{{\dfrac{{GM(1+e)}}{{a(1-e)}}}} = {c15.perihelion_speed():.3f}\ \text{{UA/año}}"],
            size=36, caption="La velocidad en el perihelio es la máxima de la órbita.")

        self.numeric_phase("RK4 sobre r'' = −GM r/|r|³: el planeta recorre la elipse y se sombrean áreas barridas en tiempos iguales.")
        sc = 1.45
        origin = np.array([-1.0, -0.25, 0.0])

        def pt(x, yv):
            return origin + sc * np.array([x, yv, 0.0])

        orbit = polyline_pts([pt(x, yv) for x, yv in zip(y[::10, 0], y[::10, 1])], FAINT)
        sun = Dot(origin, radius=0.16, color=ACCENT)
        self.play(Create(orbit), FadeIn(sun), run_time=1.5)
        n_sec = 8
        idx = np.linspace(0, len(t) - 1, n_sec + 1).astype(int)
        sectors = VGroup()
        for k in range(n_sec):
            seg = y[idx[k]: idx[k + 1] + 1: 8]
            pts = [origin] + [pt(p[0], p[1]) for p in seg] + [pt(y[idx[k + 1], 0], y[idx[k + 1], 1])]
            sectors.add(Polygon(*pts, stroke_width=0.8, stroke_color="#09090b", fill_color=NUMERIC if k % 2 == 0 else ACCENT, fill_opacity=0.35))
        clock = ValueTracker(0.0)
        planet = always_redraw(lambda: Dot(pt(float(np.interp(clock.get_value(), t, y[:, 0])), float(np.interp(clock.get_value(), t, y[:, 1]))), radius=0.11, color=FG))
        self.add(planet)
        for k in range(n_sec):
            self.play(clock.animate.set_value(float(t[idx[k + 1]])), FadeIn(sectors[k]), run_time=0.8, rate_func=linear)
        area = c15.areal_velocity(t, y)
        en = c15.energy(y)
        tab = mono_table(["magnitud", "numérico", "esperado"],
                         [("T [años]", f"{T_nu:.5f}", f"{T_an:.5f}"), ("dA/dt variación", f"{np.ptp(area) / abs(area[0]):.1e}", "0"),
                          ("E variación", f"{np.ptp(en) / abs(en[0]):.1e}", "0")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 4.6 + UP * 1.5)
        rows = c15.kepler_third_law_table((0.5, 1.0, 2.0, 4.0))
        tab3 = mono_table(["a [UA]", "T numérico", "a^1.5"], [(f"{r['a']:g}", f"{r['T_nu']:.4f}", f"{r['T_an']:.4f}") for r in rows],
                          size=16, colors=[FG, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 4.6 + DOWN * 1.1)
        self.say("Todas las áreas sombreadas son iguales (2.ª ley), y T² = a³ se cumple para otros semiejes (3.ª ley).", FadeIn(tab), FadeIn(tab3), run_time=1.4)
        self.wait(2.5)


def polyline_pts(points, color):
    from manim import VMobject
    m = VMobject(color=color, stroke_width=3)
    m.set_points_as_corners(points)
    return m


class EscapeVelocity(PhysicsScene):
    chapter_number = 15
    chapter_title = "Gravitación"

    def construct(self):
        G, M_, R = c15.G_N, c15.M_EARTH, c15.R_EARTH
        v_esc_a, v_esc_n = c15.escape_speed_analytic(), c15.escape_speed_numeric()
        self.foundations(
            [r"U(r) = -\dfrac{GMm}{r}",
             r"E = \tfrac{1}{2}mv^{2} - \dfrac{GMm}{r}"],
            note="Si E < 0 la partícula está ligada y alcanza una distancia máxima; si E ≥ 0 escapa.",
            caption="Energía potencial gravitacional y velocidad de escape.")
        self.problem(
            ["Un proyectil se lanza radialmente desde la superficie de la Tierra con rapidez v₀.",
             "Halle la distancia máxima para v₀ = 8 km/s y la velocidad de escape."],
            caption="Se desprecian la atmósfera y la rotación de la Tierra.")
        rm = c15.rmax_analytic(8000.0)
        self.analytic(
            [r"\tfrac{1}{2}v_0^2 - \dfrac{GM}{R} = -\dfrac{GM}{r_{max}} \;\Rightarrow\; r_{max} = \dfrac{R}{1 - v_0^2R/(2GM)}",
             (rf"v_0 = 8\ \mathrm{{km/s}}:\quad r_{{max}} = {rm / R:.3f}\,R = {rm / 1000:.0f}\ \mathrm{{km}}", ACCENT),
             (rf"E = 0 \;\Rightarrow\; v_{{esc}} = \sqrt{{\dfrac{{2GM}}{{R}}}} = {v_esc_a / 1000:.3f}\ \mathrm{{km/s}}", ACCENT)],
            size=32, caption="La velocidad de escape es √2 veces la velocidad circular rasante.")

        self.numeric_phase("RK4 de r'' = −GM/r² para varias rapideces de lanzamiento; la velocidad de escape se halla por bisección.")
        speeds = [6000.0, 8000.0, 10000.0, 11000.0]
        ax, deco = axes_block((0, 5000), (1, 12), 7.0, 3.6, LEFT * 3.0 + DOWN * 0.15, (0, 1000, 2000, 3000, 4000, 5000), (1, 4, 8, 12), "t [s]", "r / R")
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        cols = [FAINT, MUTED, NUMERIC, ACCENT]
        curves = VGroup()
        rows = []
        for v0, c in zip(speeds, cols):
            tt, yy = c15.radial_launch_numeric(v0, t_end=5000.0, n=10000)
            rr = yy[:, 0] / R
            below = np.where(rr[1:] < 1.0)[0]
            end = int(below[0]) + 1 if len(below) else len(rr)
            rr = np.minimum(rr[:end], 12.0)
            curves.add(polyline(ax, tt[:end:20], rr[::20], c, 3))
            rows.append((f"{v0 / 1000:.0f}", f"{c15.rmax_numeric(v0, t_end=40000.0, n=40000) / R:.3f}", f"{c15.rmax_analytic(v0) / R:.3f}"))
        self.play(*[Create(c_) for c_ in curves], run_time=3.0)
        leg = legend_block([(c, f"{v / 1000:.0f} km/s") for c, v in zip(cols, speeds)], RIGHT * 4.9 + UP * 2.0)
        tab = mono_table(["v0 [km/s]", "r_max/R RK4", "analítico"], rows, size=15, colors=[FG, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.7 + DOWN * 0.1)
        res = text_lines([(f"v_esc bisección  {v_esc_n / 1000:.4f} km/s", NUMERIC), (f"v_esc analítica  {v_esc_a / 1000:.4f} km/s", ANALYTIC)], 15).move_to(RIGHT * 4.6 + DOWN * 1.8)
        self.say("Cuanto más cerca de v_esc, más lejos llega el proyectil; a v_esc la distancia máxima diverge.", FadeIn(leg), FadeIn(tab), FadeIn(res), run_time=1.4)
        self.wait(2.5)
