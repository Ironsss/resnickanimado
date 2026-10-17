"""Capitulo 16. Estatica de fluidos."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, ValueTracker, VGroup, always_redraw, linear, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch16_fluid_statics as c16


class PressureDepth(PhysicsScene):
    chapter_number = 16
    chapter_title = "Estática de fluidos"

    def construct(self):
        p_w = c16.water_pressure(100.0)
        h_n, p_n = c16.atmosphere_numeric()
        p5_an = float(c16.atmosphere_analytic(5000.0))
        p5_nu = float(np.interp(5000.0, h_n, p_n))
        p5_cd = float(c16.atmosphere_constant_density(5000.0))
        self.foundations(
            [r"\dfrac{dp}{dy} = -\rho g",
             r"\text{Líquido (}\rho\text{ constante): } p = p_0 + \rho g h",
             r"\text{Gas ideal isotermo: } \rho = \dfrac{pM}{RT} \Rightarrow p = p_0\,e^{-Mgh/RT}"],
            note="La presión depende solo de la profundidad, no de la forma del recipiente.",
            caption="Ecuación fundamental de la estática de fluidos.", size=38)
        self.problem(
            ["a) ¿Qué presión absoluta hay a 100 m de profundidad en agua?",
             "b) ¿Cuál es la presión atmosférica a 5000 m de altura si la temperatura es 15 °C?"],
            caption="Presión en la superficie: p₀ = 101 325 Pa.")
        self.analytic(
            [(rf"p = p_0 + \rho g h = {p_w / 1e6:.3f}\ \mathrm{{MPa}}\ ({p_w / c16.P0:.2f}\ \mathrm{{atm}})", ACCENT),
             r"p(h) = p_0\,\exp\!\left(-\dfrac{Mgh}{RT}\right)",
             (rf"p(5000) = {p5_an / 1000:.2f}\ \mathrm{{kPa}}\ ({p5_an / c16.P0:.3f}\ \mathrm{{atm}})", ACCENT)],
            caption="Un modelo de densidad constante sería incorrecto para la atmósfera.")

        self.numeric_phase("Se integra dp/dh = −ρ(p) g con RK4 y se compara con la exponencial y con un modelo de densidad constante.")
        ax, deco = axes_block((0, 10), (0, 110), 7.0, 3.6, LEFT * 3.0 + DOWN * 0.15, (0, 2, 4, 6, 8, 10), (0, 25, 50, 75, 100), "h [km]", "p [kPa]")
        hs = np.linspace(0, 10000, 120)
        an = polyline(ax, hs / 1000, c16.atmosphere_analytic(hs) / 1000, ANALYTIC, 3.5)
        lin = polyline(ax, hs / 1000, np.maximum(c16.atmosphere_constant_density(hs), 0) / 1000, ERROR, 2.5)
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        self.play(Create(an), Create(lin), run_time=2.0)
        dots = VGroup(*[Dot(ax.c2p(h / 1000, p / 1000), radius=0.06, color=NUMERIC) for h, p in zip(h_n[::10], p_n[::10])])
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.1), run_time=1.8)
        leg = legend_block([(ANALYTIC, "exponencial"), (NUMERIC, "RK4"), (ERROR, "ρ constante")], RIGHT * 4.9 + UP * 2.0)
        tab = mono_table(["modelo", "p(5 km) [kPa]"],
                         [("exponencial", f"{p5_an / 1000:.3f}"), ("RK4", f"{p5_nu / 1000:.3f}"), ("ρ constante", f"{p5_cd / 1000:.3f}")],
                         size=16, colors=[MUTED, FG], gap=0.4).move_to(RIGHT * 4.6 + DOWN * 0.2)
        self.say("RK4 coincide con la exponencial; con densidad constante la presión se anularía a ~8.4 km.", FadeIn(leg), FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class FloatingCylinder(PhysicsScene):
    chapter_number = 16
    chapter_title = "Estática de fluidos"

    def construct(self):
        h_a, h_b = c16.submerged_depth_analytic(), c16.equilibrium_bisection()
        w_a, T_n = c16.omega_analytic(), c16.period_numeric()
        t, s = c16.bobbing_numeric(damping=4.0, y0=0.08, t_end=4.0)
        self.foundations(
            [r"B = \rho_f\,g\,V_{sum}\qquad(\text{Arquímedes})",
             r"\text{Flotación: } B = mg \;\Rightarrow\; \dfrac{h}{L} = \dfrac{\rho}{\rho_f}"],
            note="Si se hunde un poco más, el empuje aumenta y aparece una fuerza restauradora: el cilindro oscila.",
            caption="Principio de Arquímedes.")
        self.problem(
            ["Un cilindro de madera (ρ = 600 kg/m³, L = 0.50 m, A = 0.010 m²) flota vertical en agua.",
             "Halle la profundidad sumergida de equilibrio y el periodo de pequeñas oscilaciones verticales."],
            caption="Se desprecia la masa de agua que acompaña al movimiento.")
        self.analytic(
            [(rf"h = \dfrac{{\rho}}{{\rho_f}}L = {h_a:.3f}\ \mathrm{{m}}", ACCENT),
             r"m\ddot y = -\rho_f g A\,y \;\Rightarrow\; \omega = \sqrt{\dfrac{\rho_f gA}{m}}",
             (rf"\omega = {w_a:.3f}\ \mathrm{{rad/s}},\qquad T = {2 * np.pi / w_a:.3f}\ \mathrm{{s}}", ACCENT)],
            caption="Es un movimiento armónico simple alrededor del equilibrio.")

        self.numeric_phase("Se integra m y'' = mg − ρ_f g A y − c y' con RK4: con amortiguamiento el cilindro se estabiliza en h.")
        ax, deco = axes_block((0, 4), (0.2, 0.4), 6.2, 3.4, LEFT * 3.7 + DOWN * 0.15, (0, 1, 2, 3, 4), (0.2, 0.3, 0.4), "t [s]", "h [m]")
        eq = DashedLine(ax.c2p(0, h_a), ax.c2p(4, h_a), color=ANALYTIC, stroke_width=2, dash_length=0.1)
        curve = polyline(ax, t, s[:, 0], NUMERIC, 3.2)
        tank_x = 3.4
        water_top = -0.4
        tank = Rectangle(width=2.4, height=3.2, stroke_color=FAINT, stroke_width=3).move_to(np.array([tank_x, -1.2, 0.0]))
        water = Rectangle(width=2.4, height=2.4, stroke_width=0, fill_color=NUMERIC, fill_opacity=0.25).move_to(np.array([tank_x, -1.6, 0.0]))
        clock = ValueTracker(0.0)
        sc = 4.0

        def cyl():
            hh = float(np.interp(clock.get_value(), t, s[:, 0]))
            top = water_top + (hh - 0.0) * sc * (-1) * 0 + 0.0  # la superficie del agua define el cero
            surface = water_top
            return Rectangle(width=0.8, height=0.5 * sc, stroke_color=FG, stroke_width=2.5, fill_color=PANEL, fill_opacity=1).move_to(
                np.array([tank_x, surface + (0.5 * sc) / 2 - hh * sc, 0.0]))

        cy = always_redraw(cyl)
        surf = DashedLine(np.array([tank_x - 1.2, water_top, 0.0]), np.array([tank_x + 1.2, water_top, 0.0]), color=NUMERIC, stroke_width=2, dash_length=0.1)
        self.play(Create(ax), FadeIn(deco), Create(eq), FadeIn(tank), FadeIn(water), FadeIn(surf), FadeIn(cy), run_time=1.2)
        self.add(curve)
        self.play(clock.animate.set_value(4.0), run_time=5.0, rate_func=linear)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("h [m]", f"{h_b:.6f}", f"{h_a:.6f}"), ("T [s]", f"{T_n:.5f}", f"{2 * np.pi / w_a:.5f}")],
                         size=16, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.35).move_to(RIGHT * 0.2 + UP * 2.0)
        self.say("La bisección sobre B(h) = mg da la profundidad de equilibrio y el periodo sin amortiguar coincide con 2π/ω.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)
