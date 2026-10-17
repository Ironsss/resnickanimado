"""Capitulo 17. Dinamica de fluidos."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Polygon, Rectangle, ValueTracker, VGroup, always_redraw, linear, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch17_fluid_dynamics as c17


class VenturiTube(PhysicsScene):
    chapter_number = 17
    chapter_title = "Dinámica de fluidos"

    def construct(self):
        an = c17.venturi_analytic()
        dp = c17.P1 - an["p2"]
        q_a, q_n = c17.flow_from_pressure_drop_analytic(dp), c17.flow_from_pressure_drop_numeric(dp)
        a1, a2 = c17.areas()
        self.foundations(
            [r"A_1v_1 = A_2v_2\qquad(\text{continuidad})",
             r"p_1 + \tfrac{1}{2}\rho v_1^{2} = p_2 + \tfrac{1}{2}\rho v_2^{2}\qquad(\text{Bernoulli, tubo horizontal})"],
            note="Donde el tubo se estrecha el fluido acelera y su presión baja: la caída de presión mide el caudal.",
            caption="Fluido ideal: incompresible, no viscoso y en régimen estacionario.")
        self.problem(
            ["Por un tubo horizontal de 20 cm de diámetro que se estrecha a 10 cm circula agua con Q = 0.030 m³/s y p₁ = 200 kPa.",
             "Halle las velocidades y la presión en la garganta; luego, recupere Q a partir de la caída de presión."],
            caption="Tubo de Venturi.")
        self.analytic(
            [rf"v_1 = \dfrac{{Q}}{{A_1}} = {an['v1']:.3f}\ \mathrm{{m/s}},\qquad v_2 = \dfrac{{Q}}{{A_2}} = {an['v2']:.3f}\ \mathrm{{m/s}}",
             (rf"p_2 = p_1 + \tfrac{{1}}{{2}}\rho(v_1^2 - v_2^2) = {an['p2'] / 1000:.2f}\ \mathrm{{kPa}}", ACCENT),
             r"Q = A_2\sqrt{\dfrac{2\Delta p}{\rho\,(1 - A_2^2/A_1^2)}}"],
            size=34, caption="Con la estrangulación de área 4:1, la velocidad se cuadruplica.")

        self.numeric_phase("Se resuelve numéricamente Bernoulli + continuidad por bisección sobre Q, dada la caída de presión medida.")
        # tubo
        x0, xm, x1 = -6.2, -2.3, 1.6
        hw, hn = 0.9, 0.45
        cy = 0.9
        pipe = Polygon(np.array([x0, cy + hw, 0]), np.array([xm - 0.8, cy + hw, 0]), np.array([xm, cy + hn, 0]), np.array([x1, cy + hn, 0]),
                       np.array([x1, cy - hn, 0]), np.array([xm, cy - hn, 0]), np.array([xm - 0.8, cy - hw, 0]), np.array([x0, cy - hw, 0]),
                       stroke_color=FG, stroke_width=3, fill_opacity=0)
        self.play(Create(pipe), run_time=1.0)
        ratio = an["v2"] / an["v1"]
        base = 0.55
        parts = [Dot(np.array([x0 + 0.3 + i * 0.9, cy + (-0.35 + 0.35 * (i % 3)) * (1 if i < 3 else 1), 0]), radius=0.06, color=NUMERIC) for i in range(0)]
        pts = []
        for i in range(10):
            s = x0 + 0.3 + i * (x1 - x0 - 0.6) / 9
            yoff = (-0.28 + 0.28 * (i % 3))
            pts.append(Dot(np.array([s, cy + yoff, 0]), radius=0.06, color=NUMERIC))
        grp = VGroup(*pts)

        def adv(mob, dt):
            for d in mob:
                x, y, _ = d.get_center()
                if x < xm - 0.8:
                    v, scale = base, 1.0
                elif x < xm:
                    v, scale = base * (1 + (ratio - 1) * (x - (xm - 0.8)) / 0.8), 1.0
                else:
                    v, scale = base * ratio, 1.0
                x += v * dt
                yy = y - cy
                lim = hw if x < xm - 0.8 else (hn if x > xm else hw - (hw - hn) * (x - (xm - 0.8)) / 0.8)
                yy = np.clip(yy, -lim + 0.08, lim - 0.08)
                if x > x1 - 0.1:
                    x = x0 + 0.1
                d.move_to(np.array([x, cy + yy * 1.0, 0.0]))
        grp.add_updater(adv)
        self.add(grp)
        lab1 = text_lines([(f"v₁ = {an['v1']:.2f} m/s", FG), (f"p₁ = {c17.P1 / 1000:.0f} kPa", MUTED)], 15).move_to(np.array([x0 + 1.6, cy - 1.4, 0]))
        lab2 = text_lines([(f"v₂ = {an['v2']:.2f} m/s", NUMERIC), (f"p₂ = {an['p2'] / 1000:.1f} kPa", MUTED)], 15).move_to(np.array([x1 - 1.5, cy - 1.4, 0]))
        self.play(FadeIn(lab1), FadeIn(lab2), run_time=0.8)
        self.wait(4.0)
        grp.clear_updaters()
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("Δp [kPa]", f"{dp / 1000:.3f}", f"{dp / 1000:.3f}"), ("Q [m³/s]", f"{q_n:.6f}", f"{q_a:.6f}"),
                          ("A₁v₁ = A₂v₂", f"{a1 * an['v1']:.5f}", f"{a2 * an['v2']:.5f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 3.5 + DOWN * 1.9)
        self.say("Las partículas se aceleran en la garganta; la bisección recupera el caudal a partir de la caída de presión.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class TorricelliDrain(PhysicsScene):
    chapter_number = 17
    chapter_title = "Dinámica de fluidos"

    def construct(self):
        h0, a, A = c17.H0, c17.A_HOLE, c17.A_TANK
        te_a, te_n = c17.drain_time_analytic(), c17.drain_time_numeric()
        v0 = float(c17.torricelli_speed(h0))
        t, y = c17.height_numeric(2000)
        self.foundations(
            [r"v = \sqrt{2gh}\qquad(\text{Torricelli})",
             r"A\,\dfrac{dh}{dt} = -a\,v = -a\sqrt{2gh}"],
            note="El nivel baja cada vez más despacio porque la velocidad de salida depende de la altura.",
            caption="Bernoulli entre la superficie libre y el orificio de salida.")
        self.problem(
            ["Un depósito abierto de 0.50 m² de sección contiene agua hasta 1.5 m. Se abre un orificio de 2.0 cm² en el fondo.",
             "Halle la velocidad inicial de salida, h(t) y el tiempo de vaciado."],
            caption="La sección del orificio es mucho menor que la del depósito.")
        self.analytic(
            [(rf"v_0 = \sqrt{{2gh_0}} = {v0:.3f}\ \mathrm{{m/s}}", ACCENT),
             r"h(t) = \left(\sqrt{h_0} - \dfrac{a}{A}\sqrt{\dfrac{g}{2}}\,t\right)^{2}",
             (rf"t_{{vac}} = \dfrac{{A}}{{a}}\sqrt{{\dfrac{{2h_0}}{{g}}}} = {te_a:.1f}\ \mathrm{{s}}\ ({te_a / 60:.1f}\ \mathrm{{min}})", ACCENT)],
            caption="El nivel sigue una parábola en el tiempo.")

        self.numeric_phase("RK4 de dh/dt = −(a/A)√(2gh); el vaciado se estima con corrección local cerca de h = 0 (la derivada diverge).")
        ax, deco = axes_block((0, 1500), (0, 1.6), 6.2, 3.5, LEFT * 3.7 + DOWN * 0.1, (0, 500, 1000, 1500), (0, 0.5, 1.0, 1.5), "t [s]", "h [m]")
        ts = np.linspace(0, te_a, 120)
        an_c = polyline(ax, ts, c17.height_analytic(ts), ANALYTIC, 3.5)
        tank_x = 3.0
        sc = 2.0
        tank = Rectangle(width=2.0, height=h0 * sc + 0.2, stroke_color=FAINT, stroke_width=3).move_to(np.array([tank_x, -2.4 + (h0 * sc + 0.2) / 2, 0.0]))
        clock = ValueTracker(0.0)

        def level():
            hh = float(np.interp(clock.get_value(), t, y[:, 0]))
            hh = max(hh, 1e-3)
            return Rectangle(width=1.96, height=hh * sc, stroke_width=0, fill_color=NUMERIC, fill_opacity=0.35).move_to(np.array([tank_x, -2.4 + hh * sc / 2, 0.0]))

        def jet():
            hh = max(float(np.interp(clock.get_value(), t, y[:, 0])), 0.0)
            ln = 0.2 + 0.9 * np.sqrt(hh / h0)
            return Line(np.array([tank_x, -2.4, 0.0]), np.array([tank_x + ln, -2.7, 0.0]), color=NUMERIC, stroke_width=3) if hh > 1e-3 else VGroup()

        lv, jt = always_redraw(level), always_redraw(jet)
        cursor = always_redraw(lambda: Dot(ax.c2p(clock.get_value(), max(float(np.interp(clock.get_value(), t, y[:, 0])), 0.0)), radius=0.08, color=ACCENT))
        self.play(Create(ax), FadeIn(deco), Create(an_c), FadeIn(tank), FadeIn(lv), FadeIn(jt), FadeIn(cursor), run_time=1.5)
        self.play(clock.animate.set_value(float(te_a)), run_time=6.0, rate_func=linear)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("t_vaciado [s]", f"{te_n:.2f}", f"{te_a:.2f}"),
                          ("h(t_vac/2) [m]", f"{float(np.interp(te_a / 2, t, y[:, 0])):.5f}", f"{float(c17.height_analytic(te_a / 2)):.5f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 0.5 + UP * 2.0)
        self.say("El punto ámbar (RK4) recorre la parábola exacta; el depósito se vacía en t = (A/a)√(2h₀/g).", FadeIn(tab), run_time=1.3)
        self.wait(2.5)
