"""Capitulo 21. Teoria especial de la relatividad."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch21_relativity as c21


class MuonDecay(PhysicsScene):
    chapter_number = 21
    chapter_title = "Teoría especial de la relatividad"

    def construct(self):
        tau, beta, d = c21.TAU_MUON, c21.BETA_MUON, c21.D_MUON
        g = c21.gamma(beta)
        p_rel, p_cl = c21.survival_analytic(), c21.survival_analytic(relativistic=False)
        p_mc, err = c21.survival_monte_carlo()
        self.foundations(
            [r"\Delta t = \gamma\,\Delta t_0,\qquad \gamma = \dfrac{1}{\sqrt{1 - v^{2}/c^{2}}}",
             r"P_{sobrevive}(d) = \exp\!\left(-\dfrac{d}{v\,\gamma\,\tau}\right)"],
            note="El tiempo propio de un reloj en movimiento transcurre más despacio que el tiempo del laboratorio.",
            caption="Dilatación del tiempo.", size=38)
        self.problem(
            ["Muones con v = 0.995c y vida media propia τ = 2.197 µs se producen a 10 km de altura.",
             "¿Qué fracción llega al suelo con y sin dilatación del tiempo?"],
            caption="Se desprecian las pérdidas de energía en la atmósfera.")
        self.analytic(
            [(rf"\gamma = {g:.2f},\qquad \gamma\tau = {g * tau * 1e6:.2f}\ \mu\mathrm{{s}}", ACCENT),
             rf"\text{{Sin dilatación: }}P = e^{{-d/(v\tau)}} = {p_cl:.2e}",
             (rf"\text{{Con dilatación: }}P = e^{{-d/(v\gamma\tau)}} = {p_rel:.4f}\ ({p_rel * 100:.1f}\,\%)", ACCENT)],
            size=36, caption="Sin relatividad casi ningún muón llegaría al suelo.")

        self.numeric_phase("Monte Carlo: 2 millones de vidas aleatorias exponenciales en el laboratorio; se cuenta la fracción que supera el tiempo de vuelo.")
        ax, deco = axes_block((0, 15), (0, 1.0), 7.2, 3.6, LEFT * 2.8 + DOWN * 0.15, (0, 5, 10, 15), (0, 0.25, 0.5, 0.75, 1.0), "d [km]", "fracción que sobrevive")
        ds = np.linspace(0, 15000, 120)
        rel = polyline(ax, ds / 1000, [c21.survival_analytic(x) for x in ds], ANALYTIC, 3.5)
        cl = polyline(ax, ds / 1000, [c21.survival_analytic(x, relativistic=False) for x in ds], ERROR, 3.0)
        self.play(Create(ax), FadeIn(deco), Create(rel), Create(cl), run_time=2.0)
        pts = VGroup()
        for dd in np.linspace(1000, 15000, 8):
            pm, _ = c21.survival_monte_carlo(d=float(dd), n=400_000)
            pts.add(Dot(ax.c2p(dd / 1000, pm), radius=0.08, color=NUMERIC))
        self.play(LaggedStart(*[FadeIn(p, scale=1.5) for p in pts], lag_ratio=0.2), run_time=2.0)
        leg = legend_block([(ANALYTIC, "con dilatación"), (ERROR, "sin dilatación"), (NUMERIC, "Monte Carlo")], RIGHT * 4.8 + UP * 1.9)
        tab = mono_table(["d = 10 km", "valor"],
                         [("analítico", f"{p_rel:.5f}"), ("Monte Carlo", f"{p_mc:.5f}"), ("± σ", f"{err:.5f}"), ("sin dilatación", f"{p_cl:.1e}")],
                         size=16, colors=[MUTED, FG], gap=0.5).move_to(RIGHT * 4.7 + DOWN * 0.3)
        self.say("Los puntos azules (Monte Carlo) siguen la curva relativista; la clásica cae a cero por debajo de los 3 km.", FadeIn(leg), FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class VelocityAndEnergy(PhysicsScene):
    chapter_number = 21
    chapter_title = "Teoría especial de la relatividad"

    def construct(self):
        u = v = 0.8
        w_an, w_nu, w_rap = c21.velocity_addition_analytic(u, v), c21.velocity_addition_numeric(u, v), c21.rapidity_addition(u, v)
        self.foundations(
            [r"u' = \dfrac{u + v}{1 + uv/c^{2}}",
             r"K = (\gamma - 1)mc^{2},\qquad K = \int_0^{v} v\,dp,\ \ p = \gamma mv"],
            note="Ningún cuerpo material alcanza c: la energía cinética diverge cuando v → c.",
            caption="Composición de velocidades y energía cinética relativista.", size=38)
        self.problem(
            ["Una nave viaja a 0.8c respecto de la Tierra y lanza un proyectil a 0.8c respecto de la nave en la misma dirección.",
             "¿Qué velocidad tiene el proyectil respecto de la Tierra, y cuánto más energía cinética que la clásica tiene un cuerpo a 0.9c?"],
            caption="Todas las velocidades en unidades de c.")
        self.analytic(
            [(rf"u' = \dfrac{{0.8 + 0.8}}{{1 + 0.64}} = {w_an:.5f}\,c", ACCENT),
             rf"K_{{rel}} = (\gamma - 1)mc^2 = {c21.kinetic_analytic(0.9) / c21.C**2:.4f}\,mc^2\quad(\beta = 0.9)",
             rf"K_{{clásica}} = \tfrac12mv^2 = {c21.kinetic_classical(0.9) / c21.C**2:.4f}\,mc^2"],
            size=36, caption="A 0.9c la energía cinética relativista es más del doble que la clásica.")

        self.numeric_phase("Composición de dos transformaciones de Lorentz (matrices 2×2) y cuadratura de Simpson de ∫v dp.")
        ax, deco = axes_block((0, 1), (0, 1.1), 3.4, 3.4, LEFT * 5.0 + DOWN * 0.1, (0, 0.5, 1.0), (0, 0.5, 1.0), "v [c]", "u' [c]")
        vs = np.linspace(0, 0.99, 100)
        rel = polyline(ax, vs, [c21.velocity_addition_analytic(0.8, x) for x in vs], ANALYTIC, 3.2)
        gal = polyline(ax, vs, [min(0.8 + x, 1.1) for x in vs], ERROR, 2.3)
        limit = DashedLine(ax.c2p(0, 1), ax.c2p(1, 1), color=FAINT, stroke_width=1.5, dash_length=0.08)
        self.play(Create(ax), FadeIn(deco), Create(limit), Create(rel), Create(gal), run_time=2.0)
        pts = VGroup(*[Dot(ax.c2p(float(x), c21.velocity_addition_numeric(0.8, float(x))), radius=0.06, color=NUMERIC) for x in np.linspace(0.05, 0.95, 10)])
        self.play(LaggedStart(*[FadeIn(p, scale=1.4) for p in pts], lag_ratio=0.1), run_time=1.5)
        ax2, deco2 = axes_block((0, 0.99), (0, 6), 3.6, 3.4, LEFT * 0.4 + DOWN * 0.1, (0, 0.5, 0.9), (0, 2, 4, 6), "β", "K / mc²")
        bs = np.linspace(0, 0.99, 100)
        k_rel = polyline(ax2, bs, [c21.kinetic_analytic(b_) / c21.C**2 for b_ in bs], ANALYTIC, 3.2)
        k_cl = polyline(ax2, bs, [c21.kinetic_classical(b_) / c21.C**2 for b_ in bs], ERROR, 2.3)
        self.play(Create(ax2), FadeIn(deco2), Create(k_rel), Create(k_cl), run_time=2.0)
        pts2 = VGroup(*[Dot(ax2.c2p(float(b_), c21.kinetic_numeric(float(b_)) / c21.C**2), radius=0.06, color=NUMERIC) for b_ in np.linspace(0.1, 0.97, 9)])
        self.play(LaggedStart(*[FadeIn(p, scale=1.4) for p in pts2], lag_ratio=0.1), run_time=1.5)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("u'(0.8, 0.8)", f"{w_nu:.6f}", f"{w_an:.6f}"), ("rapidez (tanh)", f"{w_rap:.6f}", f"{w_an:.6f}"),
                          ("K(0.9)/mc²", f"{c21.kinetic_numeric(0.9) / c21.C**2:.5f}", f"{c21.kinetic_analytic(0.9) / c21.C**2:.5f}")],
                         size=14, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.28).move_to(RIGHT * 4.9 + UP * 0.3)
        leg = legend_block([(ANALYTIC, "relativista"), (ERROR, "clásica"), (NUMERIC, "numérico")], RIGHT * 4.9 + UP * 2.0)
        self.say("Las matrices de Lorentz y la integral numérica coinciden con las fórmulas; la velocidad nunca supera c.", FadeIn(tab), FadeIn(leg), run_time=1.3)
        self.wait(2.5)
