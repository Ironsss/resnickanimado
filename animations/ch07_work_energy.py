"""Capitulo 7. Trabajo y energia cinetica."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, LaggedStart, VGroup, ReplacementTransform

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch07_work_energy as c7


class SpringWork(PhysicsScene):
    chapter_number = 7
    chapter_title = "Trabajo y energía"

    def construct(self):
        k, x1, x2 = c7.K, c7.X1, c7.X2
        w = c7.spring_work_analytic()
        self.foundations(
            [r"W = \int_{x_1}^{x_2} F(x)\,dx",
             r"F = kx \;\Rightarrow\; W = \tfrac{1}{2}k\,(x_2^{2} - x_1^{2})"],
            note="El trabajo es el área bajo la curva fuerza-posición; para un resorte esa área es un trapecio.",
            caption="Trabajo de una fuerza que depende de la posición.")
        self.problem(
            ["Un resorte de k = 200 N/m se estira desde x = 0.10 m hasta x = 0.30 m.",
             "Calcule el trabajo realizado por el agente externo."],
            caption="El resorte obedece la ley de Hooke en todo el intervalo.")
        self.analytic(
            [r"W = \tfrac{1}{2}(200)\,(0.30^{2} - 0.10^{2})",
             (rf"W = {w:.2f}\ \mathrm{{J}}", ACCENT)],
            caption="Se integra F = kx entre los dos extremos.")

        self.numeric_phase("Se aproxima el área con rectángulos (suma de Riemann), trapecios y la regla de Simpson.")
        ax, deco = axes_block((0, 0.35), (0, 70), 7.2, 3.8, LEFT * 2.8 + DOWN * 0.15, (0, 0.1, 0.2, 0.3), (0, 20, 40, 60), "x [m]", "F [N]", "{:g}")
        line = polyline(ax, [0, 0.35], [0, k * 0.35], ANALYTIC, 3.5)
        self.play(Create(ax), FadeIn(deco), Create(line), run_time=1.2)
        prev = txt_prev = None
        for n in (2, 4, 16):
            rects = ax.get_riemann_rectangles(ax.plot(lambda x: k * x, x_range=[0, 0.35], color=ANALYTIC), x_range=[x1, x2],
                                              dx=(x2 - x1) / n, input_sample_type="left", stroke_width=0.6,
                                              stroke_color="#09090b", fill_opacity=0.55, color=NUMERIC)
            txt = text_lines([f"n = {n}", (f"Riemann   {c7.spring_work_numeric(n, 'riemann'):.4f} J", NUMERIC),
                              (f"trapecio  {c7.spring_work_numeric(n, 'trapezoid'):.4f} J", NUMERIC),
                              (f"Simpson   {c7.spring_work_numeric(n, 'simpson'):.4f} J", NUMERIC),
                              (f"exacto    {w:.4f} J", ACCENT)], 18).move_to(RIGHT * 4.4 + UP * 1.55)
            if prev is None:
                self.play(Create(rects), FadeIn(txt), run_time=1.2)
            else:
                self.play(ReplacementTransform(prev, rects), ReplacementTransform(txt_prev, txt), run_time=1.1)
            prev, txt_prev = rects, txt
            self.wait(0.5)
        rows = c7.work_convergence((2, 4, 8, 16, 32))
        tab = mono_table(["n", "Riemann", "trapecio"],
                         [(r["n"], f"{r['riemann']:.1e}", f"{r['trapezoid']:.0e}") for r in rows], size=16,
                         colors=[FG, ERROR, NUMERIC], gap=0.4).move_to(RIGHT * 4.4 + DOWN * 1.75)
        note = T("error absoluto en W [J]", 15, MUTED).next_to(tab, UP, buff=0.2).align_to(tab, LEFT)
        self.say("Con una fuerza lineal el trapecio y Simpson son exactos; Riemann converge con error proporcional a 1/n.",
                 FadeIn(note), FadeIn(tab), run_time=1.4)
        self.wait(2.5)


class WorkEnergyTheorem(PhysicsScene):
    chapter_number = 7
    chapter_title = "Trabajo y energía"

    def construct(self):
        m, f0, L = c7.MASS, c7.F0, c7.LENGTH
        v_an, v_nu = c7.final_speed_analytic(), c7.final_speed_numeric()
        t, x, v = c7.newton_numeric()
        self.foundations(
            [r"W_{neto} = \Delta K = \tfrac{1}{2}mv^{2} - \tfrac{1}{2}mv_0^{2}",
             r"W = \int F(x)\,dx,\qquad P = Fv"],
            note="El teorema trabajo-energía evita resolver la trayectoria cuando solo importa la rapidez final.",
            caption="El trabajo neto sobre una partícula es igual a su cambio de energía cinética.")
        self.problem(
            ["Un bloque de 2 kg parte del reposo sobre una superficie sin fricción.",
             "Lo empuja una fuerza F(x) = 30(1 − x/2) N, con x en metros, de x = 0 a x = 2 m. ¿Con qué rapidez termina?"],
            caption="La fuerza decrece linealmente y se anula en x = 2 m.")
        self.analytic(
            [r"W = \int_0^{2} 30\left(1 - \dfrac{x}{2}\right)dx = 30\ \mathrm{J}",
             r"K(x) = 30\left(x - \dfrac{x^{2}}{4}\right)",
             (rf"v = \sqrt{{2W/m}} = {v_an:.3f}\ \mathrm{{m/s}}", ACCENT)],
            caption="Todo el trabajo se convierte en energía cinética.")

        self.numeric_phase("Se integra la segunda ley de Newton con RK4 y se compara ½mv² con el trabajo acumulado.")
        ax, deco = axes_block((0, 2), (0, 35), 7.2, 3.8, LEFT * 2.8 + DOWN * 0.15, (0, 0.5, 1, 1.5, 2), (0, 10, 20, 30), "x [m]", "K [J]")
        xs = np.linspace(0, L, 120)
        curve = polyline(ax, xs, c7.kinetic_energy_analytic(xs), ANALYTIC, 3.5)
        self.play(Create(ax), FadeIn(deco), Create(curve), run_time=1.6)
        idx = np.linspace(0, len(x) - 1, 14).astype(int)
        dots = VGroup(*[Dot(ax.c2p(x[i], 0.5 * m * v[i] ** 2), radius=0.07, color=NUMERIC) for i in idx])
        leg = legend_block([(ANALYTIC, "W acumulado"), (NUMERIC, "½mv² (RK4)")], RIGHT * 4.6 + UP * 1.8)
        self.play(FadeIn(leg), LaggedStart(*[FadeIn(d, scale=1.5) for d in dots], lag_ratio=0.2), run_time=2.5)
        pmax, xp = c7.max_power()
        tab = mono_table(["magnitud", "RK4", "analítico"],
                         [("v final [m/s]", f"{v_nu:.4f}", f"{v_an:.4f}"),
                          ("W [J]", f"{c7.work_integral_numeric():.4f}", "30.0000")],
                         size=17, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.4).move_to(RIGHT * 4.6 + DOWN * 0.1)
        pw = text_lines([(f"P máxima = {pmax:.1f} W en x = {xp:.2f} m", NUMERIC)], 16).move_to(RIGHT * 4.6 + DOWN * 1.5)
        self.say("La energía cinética numérica coincide con el trabajo acumulado en todo el recorrido.",
                 FadeIn(tab), FadeIn(pw), run_time=1.4)
        self.wait(2.5)
