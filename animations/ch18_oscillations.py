"""Capitulo 18. Oscilaciones."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, ValueTracker, VGroup, always_redraw, linear, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, PANEL, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch18_oscillations as c18
from physics.integrators import integrate, rk4_step


class SimpleHarmonic(PhysicsScene):
    chapter_number = 18
    chapter_title = "Oscilaciones"

    def construct(self):
        w0, A = c18.omega0(), c18.AMP
        Tn = c18.period_numeric()
        t, y = c18.shm_numeric(1.0, 2000)
        self.foundations(
            [r"m\ddot x = -kx \;\Rightarrow\; x(t) = A\cos(\omega t + \varphi),\quad \omega = \sqrt{k/m}",
             r"E = \tfrac{1}{2}kA^{2},\qquad T = \dfrac{2\pi}{\omega},\qquad T_{pendulo} \approx 2\pi\sqrt{L/g}"],
            note="La fuerza restauradora proporcional al desplazamiento produce un movimiento sinusoidal.",
            caption="Movimiento armónico simple.", size=38)
        self.problem(
            ["Un bloque de 0.5 kg unido a un resorte de k = 200 N/m se suelta desde x = 8 cm. Halle ω, T, v máxima y la energía.",
             "Después compare el periodo exacto de un péndulo de 1 m con el de pequeñas oscilaciones."],
            caption="Sin fricción.")
        self.analytic(
            [(rf"\omega = \sqrt{{k/m}} = {w0:.1f}\ \mathrm{{rad/s}},\quad T = {2 * np.pi / w0:.4f}\ \mathrm{{s}}", ACCENT),
             rf"v_{{max}} = A\omega = {A * w0:.2f}\ \mathrm{{m/s}},\quad E = \tfrac12kA^2 = {c18.energy_analytic():.3f}\ \mathrm{{J}}",
             r"T_{exacto} = \dfrac{2\pi\sqrt{L/g}}{\mathrm{AGM}\,(1,\ \cos(\theta_0/2))}"],
            size=34, caption="El periodo del péndulo crece con la amplitud: la aproximación armónica es válida solo para θ₀ pequeño.")

        self.numeric_phase("RK4 sobre x'' = −(k/m)x: se compara con x(t) = A cos ωt y se mide el periodo por cruces de cero.")
        center = np.array([-2.7, 1.85, 0.0])
        sc = 20.0
        clock = ValueTracker(0.0)

        def blk():
            xx = float(np.interp(clock.get_value(), t, y[:, 0])) * sc
            pos = center + np.array([xx, 0, 0])
            wall = Line(center + np.array([-2.6, -0.5, 0]), center + np.array([-2.6, 0.5, 0]), color=FAINT, stroke_width=4)
            spring = VGroup(*[Line(center + np.array([-2.6 + (pos[0] - center[0] + 2.6 - 0.3) * k / 12, 0.18 * (-1) ** k, 0]),
                                   center + np.array([-2.6 + (pos[0] - center[0] + 2.6 - 0.3) * (k + 1) / 12, 0.18 * (-1) ** (k + 1), 0]), color=FG, stroke_width=2) for k in range(12)])
            box = Rectangle(width=0.6, height=0.6, stroke_color=NUMERIC, stroke_width=2.5, fill_color=PANEL, fill_opacity=1).move_to(pos)
            return VGroup(wall, spring, box)

        sys_ = always_redraw(blk)
        ax, deco = axes_block((0, 1.0), (-0.1, 0.1), 6.8, 2.5, LEFT * 2.9 + DOWN * 1.35, (0, 0.25, 0.5, 0.75, 1.0), (-0.08, 0, 0.08), "t [s]", "x [m]")
        an = polyline(ax, t, c18.x_analytic(t), ANALYTIC, 3)
        dots = VGroup(*[Dot(ax.c2p(float(t[i]), float(y[i, 0])), radius=0.04, color=NUMERIC) for i in range(0, len(t), 40)])
        self.play(FadeIn(sys_), Create(ax), FadeIn(deco), run_time=1.0)
        self.play(Create(an), clock.animate.set_value(1.0), run_time=6.0, rate_func=linear)
        self.play(FadeIn(dots), run_time=1.0)
        rows = [(f"{r['theta']}°", f"{r['small']:.4f}", f"{r['exact']:.4f}", f"{r['numeric']:.4f}") for r in c18.pendulum_table()]
        tab = mono_table(["θ₀", "T pequeño", "T exacto", "T RK4"], rows, size=15, colors=[FG, MUTED, ANALYTIC, NUMERIC], gap=0.3).move_to(RIGHT * 4.3 + DOWN * 0.5)
        res = text_lines([(f"T resorte RK4  {Tn:.5f} s", NUMERIC), (f"T resorte teoría {2 * np.pi / w0:.5f} s", ANALYTIC)], 15).move_to(RIGHT * 4.3 + UP * 1.5)
        self.say("RK4 coincide con el coseno exacto y con T = 2π/ω; el periodo del péndulo exacto (AGM) coincide con la integración.", FadeIn(tab), FadeIn(res), run_time=1.4)
        self.wait(2.5)


class DrivenResonance(PhysicsScene):
    chapter_number = 18
    chapter_title = "Oscilaciones"

    def construct(self):
        m, k, b, f0 = c18.M_D, c18.K_D, c18.B_D, c18.F0
        wr = c18.resonance_frequency_analytic()
        self.foundations(
            [r"m\ddot x + b\dot x + kx = F_0\cos\omega t",
             r"A(\omega) = \dfrac{F_0/m}{\sqrt{(\omega_0^{2} - \omega^{2})^{2} + (\gamma\omega)^{2}}},\qquad \gamma = \dfrac{b}{m}"],
            note="Tras un transitorio que decae como e^(−γt/2), el oscilador sigue la frecuencia del forzamiento.",
            caption="Oscilador amortiguado y forzado.", size=38)
        self.problem(
            ["Un oscilador de m = 1 kg, k = 100 N/m y b = 2 kg/s se fuerza con F₀ = 5 N.",
             "Halle la amplitud estacionaria en función de ω y la frecuencia de máxima respuesta."],
            caption="ω₀ = 10 rad/s y γ = 2 s⁻¹ (Q = 5).")
        self.analytic(
            [(rf"\omega_r = \sqrt{{\omega_0^2 - \gamma^2/2}} = {wr:.3f}\ \mathrm{{rad/s}}", ACCENT),
             rf"A(\omega_r) = {float(c18.driven_amplitude_analytic(wr)):.4f}\ \mathrm{{m}},\qquad A(0) = F_0/k = {f0 / k:.2f}\ \mathrm{{m}}",
             r"Q = \omega_0/\gamma = 5"],
            size=36, caption="La resonancia de amplitud ocurre ligeramente por debajo de la frecuencia natural.")

        self.numeric_phase("Se simula con RK4 hasta el régimen estacionario para cada ω y se mide la amplitud del último tramo.")
        t_end = 12.0
        w_drive = wr
        t, y = integrate(lambda tt, s: np.array([s[1], (f0 * np.cos(w_drive * tt) - b * s[1] - k * s[0]) / m]), np.zeros(2), 0.0, t_end, 12000, rk4_step)
        ax, deco = axes_block((0, t_end), (-0.7, 0.7), 6.8, 2.0, LEFT * 3.0 + UP * 1.4, (0, 3, 6, 9, 12), (-0.5, 0, 0.5), "t [s]", "x [m]")
        curve = polyline(ax, t[::6], y[::6, 0], NUMERIC, 2.2)
        env = polyline(ax, t, np.full_like(t, float(c18.driven_amplitude_analytic(wr))), FAINT, 1.5)
        self.play(Create(ax), FadeIn(deco), Create(env), run_time=0.9)
        self.play(Create(curve), run_time=3.0)
        ws = np.linspace(6, 14, 17)
        num = c18.resonance_curve_numeric(ws)
        grid = np.linspace(5.5, 14.5, 200)
        ax2, deco2 = axes_block((5, 15), (0, 0.8), 6.8, 2.0, LEFT * 3.0 + DOWN * 1.6, (6, 8, 10, 12, 14), (0, 0.25, 0.5, 0.75), "ω [rad/s]", "A [m]")
        c_an = polyline(ax2, grid, c18.driven_amplitude_analytic(grid), ANALYTIC, 3.2)
        pts = VGroup(*[Dot(ax2.c2p(float(w_), float(a_)), radius=0.07, color=NUMERIC) for w_, a_ in zip(ws, num)])
        self.play(Create(ax2), FadeIn(deco2), Create(c_an), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(p, scale=1.5) for p in pts], lag_ratio=0.12), run_time=2.0)
        rows = [(f"{w_:.1f}", f"{a:.4f}", f"{float(c18.driven_amplitude_analytic(w_)):.4f}") for w_, a in list(zip(ws, num))[::4]]
        tab = mono_table(["ω", "A numérico", "A analítico"], rows, size=15, colors=[FG, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.6 + UP * 0.8)
        peak = text_lines([(f"pico numérico  ω = {ws[int(np.argmax(num))]:.1f}", NUMERIC), (f"ω_r analítico   {wr:.3f}", ANALYTIC)], 15).move_to(RIGHT * 4.6 + DOWN * 1.2)
        self.say("Las amplitudes simuladas caen sobre la curva de resonancia; el máximo está cerca de ω_r.", FadeIn(tab), FadeIn(peak), run_time=1.3)
        self.wait(2.5)
