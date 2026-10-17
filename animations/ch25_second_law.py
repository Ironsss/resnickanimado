"""Capitulo 25. Segunda ley y entropia."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Polygon, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch25_second_law as c25
from physics.integrators import integrate, rk4_step
from physics import ch22_temperature as c22


class CarnotCycle(PhysicsScene):
    chapter_number = 25
    chapter_title = "Segunda ley y entropía"

    def construct(self):
        th, tc = c25.TH, c25.TC
        eta_an = c25.carnot_efficiency_analytic()
        o = c25.carnot_numeric()
        st = c25.carnot_states()
        self.foundations(
            [r"\eta = \dfrac{W}{Q_h} = 1 - \dfrac{Q_c}{Q_h},\qquad \eta_{Carnot} = 1 - \dfrac{T_c}{T_h}",
             r"\text{Isotermas: } pV = \text{cte},\qquad \text{adiabatas: } pV^{\gamma} = \text{cte}"],
            note="Ninguna máquina que opere entre dos reservorios es más eficiente que una de Carnot: segunda ley.",
            caption="Máquinas térmicas y ciclo de Carnot.", size=38)
        self.problem(
            ["Una máquina de Carnot con 1 mol de gas ideal monoatómico opera entre 500 K y 300 K; el volumen en la isoterma caliente va de 1 L a 2 L.",
             "Halle el trabajo neto y la eficiencia."],
            caption="Todos los procesos son reversibles.")
        self.analytic(
            [(rf"\eta = 1 - \dfrac{{T_c}}{{T_h}} = {eta_an:.3f}", ACCENT),
             rf"Q_h = nRT_h\ln\dfrac{{V_B}}{{V_A}} = {o['Q_h']:.1f}\ \mathrm{{J}}",
             (rf"W = \eta\,Q_h = {eta_an * o['Q_h']:.1f}\ \mathrm{{J}}", ACCENT)],
            size=38, caption="La eficiencia depende solo de las temperaturas de los reservorios.")

        self.numeric_phase("Isotermas por cuadratura de p = nRT/V y adiabatas con RK4 sobre dp/dV = −γp/V; se suma el trabajo de los cuatro tramos.")
        ax, deco = axes_block((0.8e-3, 4.0e-3), (0, 5.5e6), 6.8, 3.7, LEFT * 3.1 + DOWN * 0.1, (1e-3, 2e-3, 3e-3, 4e-3), (0, 2e6, 4e6), "V [m³]", "p [Pa]", "{:g}")
        cols = {"AB": ERROR, "BC": NUMERIC, "CD": NUMERIC, "DA": ERROR}
        cols = {"AB": ACCENT, "BC": NUMERIC, "CD": ERROR, "DA": ANALYTIC}
        curves = {k: polyline(ax, v_, p_, cols[k], 3.5) for k, (v_, p_) in o["paths"].items()}
        poly_pts = []
        for k in ("AB", "BC", "CD", "DA"):
            v_, p_ = o["paths"][k]
            poly_pts += [ax.c2p(float(a), float(b)) for a, b in zip(v_[::10], p_[::10])]
        fill = Polygon(*poly_pts, stroke_width=0, fill_color=NUMERIC, fill_opacity=0.12)
        labs = VGroup(*[T(n_, 18, FG).next_to(ax.c2p(st[n_][1], st[n_][0]), d_, buff=0.12) for n_, d_ in zip("ABCD", (UP, RIGHT, DOWN, LEFT))])
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        self.play(*[Create(c_) for c_ in curves.values()], run_time=3.0)
        self.play(FadeIn(fill), FadeIn(labs), run_time=1.0)
        leg = legend_block([(ACCENT, "A→B isoterma 500 K"), (NUMERIC, "B→C adiabática"), (ERROR, "C→D isoterma 300 K"), (ANALYTIC, "D→A adiabática")], RIGHT * 4.6 + UP * 1.9)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("η", f"{o['eta']:.6f}", f"{eta_an:.6f}"), ("W_neto [J]", f"{o['W_net']:.2f}", f"{eta_an * o['Q_h']:.2f}"),
                          ("p_A final [MPa]", f"{o['pa_end'] / 1e6:.6f}", f"{o['pa'] / 1e6:.6f}")],
                         size=13, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.2).move_to(RIGHT * 4.4 + DOWN * 0.7)
        self.say("Las cuatro integrales numéricas dan η = 1 − Tc/Th; la adiabática D→A cierra el ciclo exactamente en p_A.", FadeIn(leg), FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class EntropyProduction(PhysicsScene):
    chapter_number = 25
    chapter_title = "Segunda ley y entropía"

    def construct(self):
        s_heat_a, s_heat_n = c25.heating_entropy_analytic(), c25.heating_entropy_numeric()
        s_c_a, s_c_n = c25.contact_entropy_analytic(), c25.contact_entropy_numeric()
        self.foundations(
            [r"dS = \dfrac{dQ_{rev}}{T},\qquad \Delta S = \int \dfrac{dQ}{T}",
             r"\Delta S_{universo} \ge 0\qquad(\text{segunda ley})"],
            note="La entropía de un sistema aislado nunca disminuye; los procesos irreversibles la aumentan.",
            caption="Entropía y segunda ley.", size=40)
        self.problem(
            ["a) Un kilogramo de agua se calienta de 20 °C a 80 °C. Halle ΔS.",
             "b) Un bloque de aluminio caliente y agua fría (los del capítulo de temperatura) se igualan por contacto. Halle ΔS total."],
            caption="El calor específico del agua es 4186 J/kg K.")
        self.analytic(
            [(rf"\Delta S_a = mc\ln\dfrac{{T_2}}{{T_1}} = {s_heat_a:.1f}\ \mathrm{{J/K}}", ACCENT),
             r"\Delta S_b = C_1\ln\dfrac{T_f}{T_{1}} + C_2\ln\dfrac{T_f}{T_{2}}",
             (rf"\Delta S_b = {s_c_a:.2f}\ \mathrm{{J/K}} > 0", ACCENT)],
            size=36, caption="El resultado es positivo: el contacto térmico es irreversible.")

        self.numeric_phase("Se integra la producción de entropía dS/dt = h(T₁ − T₂)²/(T₁T₂) con RK4 junto con las temperaturas.")
        rhs = lambda _t, y: np.array([-c22.H * (y[0] - y[1]) / (c22.M1 * c22.C1), c22.H * (y[0] - y[1]) / (c22.M2 * c22.C2),
                                      c22.H * (y[0] - y[1]) ** 2 / ((y[0] + 273.15) * (y[1] + 273.15))])
        t, y = integrate(rhs, np.array([c22.T1_0, c22.T2_0, 0.0]), 0.0, 120.0, 2400, rk4_step)
        ax, deco = axes_block((0, 120), (0, 20), 7.2, 3.6, LEFT * 2.8 + DOWN * 0.15, (0, 30, 60, 90, 120), (0, 5, 10, 15, 20), "t [s]", "S producida [J/K]")
        ref = DashedLine(ax.c2p(0, s_c_a), ax.c2p(120, s_c_a), color=ANALYTIC, stroke_width=2, dash_length=0.1)
        curve = polyline(ax, t[::10], y[::10, 2], NUMERIC, 3.5)
        self.play(Create(ax), FadeIn(deco), Create(ref), run_time=1.2)
        self.play(Create(curve), run_time=3.0)
        tab = mono_table(["caso", "numérico [J/K]", "analítico [J/K]"],
                         [("calentamiento", f"{s_heat_n:.3f}", f"{s_heat_a:.3f}"), ("contacto", f"{s_c_n:.4f}", f"{s_c_a:.4f}")],
                         size=13, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.2).move_to(RIGHT * 4.5 + UP * 0.8)
        note = text_lines([("calentamiento: Simpson de mc dT/T", MUTED), ("contacto: RK4 de la producción de entropía", MUTED)], 13).move_to(RIGHT * 4.5 + DOWN * 0.8)
        self.say("La entropía producida crece monótonamente y se satura en el valor analítico (línea blanca): ΔS > 0.", FadeIn(tab), FadeIn(note), run_time=1.3)
        self.wait(2.5)
