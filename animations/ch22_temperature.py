"""Capitulo 22. Temperatura y ley cero."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch22_temperature as c22


class ThermalExpansion(PhysicsScene):
    chapter_number = 22
    chapter_title = "Temperatura y ley cero"

    def construct(self):
        L0, a0 = c22.L0, c22.ALPHA0
        d_lin, d_ex, d_nu = c22.length_linear() - L0, c22.length_analytic() - L0, c22.length_numeric() - L0
        self.foundations(
            [r"\Delta L = \alpha L_0\,\Delta T,\qquad \Delta V = \beta V_0\,\Delta T,\ \ \beta \approx 3\alpha",
             r"\dfrac{dL}{dT} = \alpha(T)\,L"],
            note="Si α depende de la temperatura, la expresión lineal es una aproximación; la ecuación diferencial es exacta.",
            caption="Dilatación térmica lineal.", size=38)
        self.problem(
            ["Un riel de acero de 30 m (α₀ = 12×10⁻⁶ K⁻¹ a 20 °C) se calienta hasta 120 °C.",
             "Calcule ΔL con α constante y con α(T) = α₀[1 + 0.002 (T − 20 °C)], y compárelos."],
            caption="El coeficiente varía ligeramente con la temperatura.")
        self.analytic(
            [(rf"\Delta L_{{lineal}} = \alpha_0L_0\Delta T = {d_lin * 1000:.2f}\ \mathrm{{mm}}", ACCENT),
             r"L = L_0\exp\!\left[\alpha_0\left(\Delta T + \tfrac{\beta}{2}\Delta T^{2}\right)\right]",
             (rf"\Delta L_{{exacto}} = {d_ex * 1000:.2f}\ \mathrm{{mm}}", ACCENT)],
            size=36, caption="La variación de α añade unos 11 mm a los 100 K de calentamiento.")

        self.numeric_phase("RK4 sobre dL/dT = α(T) L, de 20 °C a 120 °C, frente a la aproximación lineal.")
        ax, deco = axes_block((20, 120), (0, 50), 7.2, 3.6, LEFT * 2.8 + DOWN * 0.15, (20, 45, 70, 95, 120), (0, 10, 20, 30, 40, 50), "T [°C]", "ΔL [mm]")
        temps = np.linspace(20, 120, 60)
        exact = polyline(ax, temps, [(c22.length_analytic(t_) - L0) * 1000 for t_ in temps], ANALYTIC, 3.5)
        lin = polyline(ax, temps, [(c22.length_linear(t_) - L0) * 1000 for t_ in temps], ERROR, 2.5)
        self.play(Create(ax), FadeIn(deco), Create(exact), Create(lin), run_time=2.0)
        pts = VGroup(*[Dot(ax.c2p(float(t_), (c22.length_numeric(float(t_)) - L0) * 1000), radius=0.07, color=NUMERIC) for t_ in np.linspace(30, 120, 9)])
        self.play(LaggedStart(*[FadeIn(p, scale=1.5) for p in pts], lag_ratio=0.12), run_time=1.8)
        rows = [(f"{r['T']:.0f}", f"{r['linear'] * 1000:.3f}", f"{r['exact'] * 1000:.3f}", f"{r['numeric'] * 1000:.3f}") for r in c22.expansion_table()]
        tab = mono_table(["T [°C]", "lineal [mm]", "exacto [mm]", "RK4 [mm]"], rows, size=15, colors=[FG, ERROR, ANALYTIC, NUMERIC], gap=0.3).move_to(RIGHT * 4.6 + DOWN * 0.3)
        leg = legend_block([(ANALYTIC, "exacto α(T)"), (ERROR, "lineal α₀"), (NUMERIC, "RK4")], RIGHT * 4.8 + UP * 1.8)
        self.say("RK4 reproduce la solución exacta; el modelo con α constante subestima la dilatación a temperaturas altas.", FadeIn(tab), FadeIn(leg), run_time=1.3)
        self.wait(2.5)


class ThermalEquilibrium(PhysicsScene):
    chapter_number = 22
    chapter_title = "Temperatura y ley cero"

    def construct(self):
        tf = c22.final_temperature_analytic()
        lam = c22.rate_analytic()
        t, y = c22.temperatures_numeric(100.0, 2000)
        self.foundations(
            [r"\text{Ley cero: si } A \sim C \text{ y } B \sim C \Rightarrow A \sim B",
             r"C_1\dfrac{dT_1}{dt} = -h(T_1 - T_2),\qquad C_2\dfrac{dT_2}{dt} = +h(T_1 - T_2)"],
            note="Dos cuerpos en contacto térmico alcanzan una temperatura común, siempre que no pierdan calor al exterior.",
            caption="Equilibrio térmico y ley cero de la termodinámica.", size=36)
        self.problem(
            ["Un bloque de aluminio de 0.30 kg (c = 900 J/kg K) a 150 °C se sumerge en 0.50 kg de agua (c = 4186 J/kg K) a 20 °C.",
             "La conductancia térmica entre ambos es h = 20 W/K. Halle la temperatura final y la constante de tiempo."],
            caption="El sistema está aislado del exterior.")
        self.analytic(
            [(rf"T_f = \dfrac{{C_1T_1 + C_2T_2}}{{C_1 + C_2}} = {tf:.2f}\ ^\circ\mathrm{{C}}", ACCENT),
             r"T_i(t) = T_f + (T_{i0} - T_f)\,e^{-\lambda t},\qquad \lambda = h\left(\dfrac{1}{C_1} + \dfrac{1}{C_2}\right)",
             (rf"\lambda = {lam:.4f}\ \mathrm{{s^{{-1}}}},\qquad \tau = 1/\lambda = {1 / lam:.1f}\ \mathrm{{s}}", ACCENT)],
            size=34, caption="La diferencia de temperaturas decae exponencialmente.")

        self.numeric_phase("Se integran las dos ecuaciones acopladas con RK4 y se comparan con las exponenciales exactas.")
        ax, deco = axes_block((0, 100), (0, 160), 7.4, 3.7, LEFT * 2.7 + DOWN * 0.15, (0, 25, 50, 75, 100), (0, 40, 80, 120, 160), "t [s]", "T [°C]")
        ta, tb = c22.temperatures_analytic(t)
        a1, a2 = polyline(ax, t[::10], ta[::10], ANALYTIC, 3.2), polyline(ax, t[::10], tb[::10], ANALYTIC, 3.2)
        eqline = DashedLine(ax.c2p(0, tf), ax.c2p(100, tf), color=ACCENT, stroke_width=1.8, dash_length=0.1)
        self.play(Create(ax), FadeIn(deco), Create(eqline), Create(a1), Create(a2), run_time=2.0)
        pts = VGroup()
        for i in range(0, len(t), 100):
            pts.add(Dot(ax.c2p(float(t[i]), float(y[i, 0])), radius=0.06, color=NUMERIC), Dot(ax.c2p(float(t[i]), float(y[i, 1])), radius=0.06, color=ERROR))
        self.play(LaggedStart(*[FadeIn(p, scale=1.4) for p in pts], lag_ratio=0.04), run_time=2.0)
        t2, y2 = c22.temperatures_numeric(300.0, 6000)
        tab = mono_table(["magnitud", "RK4", "analítico"],
                         [("T_f [°C]", f"{y2[-1, 0]:.4f}", f"{tf:.4f}"), ("T_1(20 s) [°C]", f"{float(np.interp(20, t, y[:, 0])):.4f}", f"{float(c22.temperatures_analytic(20.0)[0]):.4f}"),
                          ("T_2(20 s) [°C]", f"{float(np.interp(20, t, y[:, 1])):.4f}", f"{float(c22.temperatures_analytic(20.0)[1]):.4f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.6 + UP * 0.8)
        conv = text_lines([(f"{c22.to_kelvin(tf):.2f} K = {c22.to_fahrenheit(tf):.2f} °F", MUTED)], 15).move_to(RIGHT * 4.6 + DOWN * 1.0)
        self.say("Los puntos RK4 (aluminio azul, agua rojo) convergen a la misma temperatura final T_f marcada en ámbar.", FadeIn(tab), FadeIn(conv), run_time=1.3)
        self.wait(2.5)
