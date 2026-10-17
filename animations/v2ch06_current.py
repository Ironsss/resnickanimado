"""Volumen 2, capitulo 6. Corriente y resistencia."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Polygon, Rectangle, ValueTracker, VGroup, ReplacementTransform, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table, polyline,
    text_lines,
)
from physics import v2ch06_current as e6


class DrudeElectrons(PhysicsScene):
    volume = 2
    chapter_number = 6
    chapter_title = "Corriente y resistencia"

    def construct(self):
        vd_a = e6.drift_speed_analytic()
        mu_n, sig, mean = e6.drift_mobility_monte_carlo()
        mu_a = e6.drift_mobility_analytic()
        sg = e6.conductivity_analytic()
        self.foundations(
            [r"I = \dfrac{dq}{dt},\qquad J = nqv_d,\qquad \vec J = \sigma\vec E",
             r"v_d = \dfrac{eE\tau}{m},\qquad \sigma = \dfrac{ne^{2}\tau}{m}"],
            note="Entre colisiones los electrones aceleran; las colisiones los frenan, y el resultado neto es una deriva lenta.",
            caption="Modelo de Drude de la conducción.", size=40)
        self.problem(
            ["En un conductor de cobre (n = 8.5×10²⁸ m⁻³, τ = 2.5×10⁻¹⁴ s) hay un campo de 100 V/m.",
             "Halle la velocidad de deriva y la conductividad, y verifique la movilidad v_d/E con una simulación de 20 000 electrones."],
            caption="Los electrones tienen velocidades térmicas del orden de 10⁵ m/s.")
        self.analytic(
            [(rf"v_d = \dfrac{{eE\tau}}{{m}} = {vd_a:.4f}\ \mathrm{{m/s}}", ACCENT),
             (rf"\sigma = \dfrac{{ne^2\tau}}{{m}} = {sg / 1e7:.3f}\times10^{{7}}\ \mathrm{{S/m}}", ACCENT)],
            size=40, caption="La velocidad de deriva es mucho menor que la térmica.")

        self.numeric_phase("Monte Carlo: cada electrón acelera y, con probabilidad dt/τ por paso, sufre una colisión que reinicia su velocidad térmica.")
        rng = np.random.default_rng(3)
        box = Rectangle(width=4.2, height=3.2, stroke_color=FG, stroke_width=3).move_to(np.array([-3.4, 0.2, 0.0]))
        pos = rng.uniform(-2.0, 2.0, (30, 2))
        pos[:, 1] = rng.uniform(-1.5, 1.5, 30)
        vel = rng.normal(0, 1.0, (30, 2)) * 1.2
        dots = VGroup(*[Dot(radius=0.05, color=NUMERIC) for _ in range(30)])
        arrow_e = VGroup(*[T("E →", 16, ACCENT).move_to(np.array([-3.4, 2.1, 0.0]))])

        def step(mob, dt):
            vel[:, 0] += 0.9 * dt  # arrastre exagerado (solo visual)
            hit = rng.uniform(size=30) < dt / 0.5
            vel[hit] = rng.normal(0, 1.0, (hit.sum(), 2)) * 1.2
            pos[:] += vel * dt
            for ax_, lim in ((0, 2.0), (1, 1.5)):
                over = pos[:, ax_] > lim
                under = pos[:, ax_] < -lim
                pos[over, ax_] = -lim
                pos[under, ax_] = lim
            for d_, p in zip(mob, pos):
                d_.move_to(np.array([-3.4 + p[0], 0.2 + p[1], 0.0]))
        dots.add_updater(step)
        self.play(FadeIn(box), FadeIn(dots), FadeIn(arrow_e), run_time=0.8)
        self.wait(4.0)
        dots.clear_updaters()
        self.play(FadeOut(dots), FadeOut(box), FadeOut(arrow_e), run_time=0.6)
        t_ps = len(mean) * e6.TAU / 50 * 1e12
        ax, deco = axes_block((0, t_ps), (0, 6e-3), 6.2, 3.3, LEFT * 3.4 + DOWN * 0.15, tuple(np.round(np.linspace(0, t_ps, 5), 1)), (0, 0.002, 0.004, 0.006), "t [ps]", "⟨v⟩/E [m²/V·s]")
        tt = np.arange(len(mean)) * (e6.TAU / 50) * 1e12
        curve = polyline(ax, tt[::20], mean[::20], NUMERIC, 2.5)
        ref = DashedLine(ax.c2p(0, mu_a), ax.c2p(float(tt[-1]), mu_a), color=ANALYTIC, stroke_width=2, dash_length=0.1)
        self.play(Create(ax), FadeIn(deco), Create(ref), run_time=1.0)
        self.play(Create(curve), run_time=2.5)
        tab = mono_table(["magnitud", "simulación", "analítico"],
                         [("μ = v_d/E [10⁻³ m²/Vs]", f"{mu_n * 1e3:.3f} ± {sig * 1e3:.3f}", f"{mu_a * 1e3:.3f}"), ("v_d a 100 V/m [m/s]", f"{mu_n * 100:.4f}", f"{vd_a:.4f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.2 + UP * 1.0)
        note = text_lines([("n_e = 20 000, 4000 pasos de τ/50", MUTED), ("campo de simulación 2×10⁶ V/m (el modelo es lineal)", MUTED)], 14).move_to(RIGHT * 4.2 + DOWN * 0.7)
        self.say("La movilidad medida converge a eτ/m (línea blanca): con 100 V/m la deriva es de solo 0.44 m/s frente a ~10⁵ m/s térmicos.", FadeIn(tab), FadeIn(note), run_time=1.3)
        self.wait(2.5)


class TaperedWire(PhysicsScene):
    volume = 2
    chapter_number = 6
    chapter_title = "Corriente y resistencia"

    def construct(self):
        rho, L, r1, r2 = e6.RHO, e6.LENGTH, e6.R1, e6.R2
        ra, rn = e6.tapered_resistance_analytic(), e6.tapered_resistance_numeric()
        ru = e6.uniform_resistance()
        self.foundations(
            [r"R = \dfrac{V}{I},\qquad R = \rho\dfrac{L}{A},\qquad \rho(T) = \rho_0\left[1 + \alpha\,\Delta T\right]",
             r"R = \int_0^{L}\dfrac{\rho\,dx}{A(x)}\qquad(\text{sección variable})"],
            note="Los tramos de un conductor no uniforme están en serie: sus resistencias diferenciales se suman.",
            caption="Resistencia y ley de Ohm.", size=40)
        self.problem(
            ["Un cable de cobre (ρ = 1.68×10⁻⁸ Ω·m) de 2 m tiene un radio que crece linealmente de 0.5 mm a 1.5 mm.",
             "Halle su resistencia por integración y compárela con la de un cable uniforme de 0.5 mm."],
            caption="La corriente es la misma en todas las secciones.")
        self.analytic(
            [r"r(x) = r_1 + (r_2 - r_1)\dfrac{x}{L},\qquad R = \dfrac{\rho}{\pi}\int_0^{L}\dfrac{dx}{r(x)^{2}} = \dfrac{\rho L}{\pi r_1 r_2}",
             (rf"R = {ra * 1000:.3f}\ \mathrm{{m\Omega}}", ACCENT),
             rf"R_{{uniforme}}(r_1) = {ru * 1000:.3f}\ \mathrm{{m\Omega}}"],
            size=34, caption="La resistencia depende del producto de los radios extremos.")

        self.numeric_phase("Simpson sobre dR/dx = ρ/(π r²) con 400 subintervalos; el área bajo la curva es la resistencia total.")
        wire_poly = Polygon(np.array([-6.2, 1.85 + 0.18, 0]), np.array([-3.0, 1.85 + 0.54, 0]), np.array([-3.0, 1.85 - 0.54, 0]), np.array([-6.2, 1.85 - 0.18, 0]),
                            stroke_color=FG, stroke_width=3, fill_color="#27272a", fill_opacity=1)
        self.play(Create(wire_poly), run_time=1.0)
        ax, deco = axes_block((0, L), (0, 40), 6.0, 2.5, LEFT * 2.7 + DOWN * 1.1, (0, 0.5, 1.0, 1.5, 2.0), (0, 10, 20, 30, 40), "x [m]", "dR/dx [mΩ/m]")
        xs = np.linspace(0, L, 120)
        curve = polyline(ax, xs, e6.RHO / (np.pi * (r1 + (r2 - r1) * xs / L) ** 2) * 1000, ANALYTIC, 3.5)
        self.play(Create(ax), FadeIn(deco), Create(curve), run_time=1.5)
        for n in (4, 16):
            xr = np.linspace(0, L, n + 1)
            from manim import Rectangle as Rect
            rects = VGroup()
            for k in range(n):
                y_left = e6.RHO / (np.pi * (r1 + (r2 - r1) * xr[k] / L) ** 2) * 1000
                p0, p1 = ax.c2p(xr[k], 0), ax.c2p(xr[k + 1], y_left)
                rr = Rect(width=(p1 - p0)[0], height=(p1 - p0)[1], stroke_width=0.6, stroke_color="#09090b", fill_color=NUMERIC, fill_opacity=0.5)
                rr.move_to([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0])
                rects.add(rr)
            if n == 4:
                self.play(FadeIn(rects), run_time=1.0)
                prev = rects
            else:
                self.play(ReplacementTransform(prev, rects), run_time=1.0)
        tab = mono_table(["método", "R [mΩ]"],
                         [("Simpson (n = 400)", f"{rn * 1000:.6f}"), ("analítico", f"{ra * 1000:.6f}"), ("cable uniforme", f"{ru * 1000:.3f}"),
                          ("a 80 °C más (α = 0.0039)", f"{e6.resistance_at_temperature(ra, 80.0) * 1000:.3f}")],
                         size=15, colors=[MUTED, NUMERIC], gap=0.4).move_to(RIGHT * 4.2 + UP * 1.0)
        self.say("La integral numérica reproduce ρL/(πr₁r₂); el tramo delgado aporta casi toda la resistencia.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)
