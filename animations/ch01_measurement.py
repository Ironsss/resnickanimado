"""Capitulo 1. Medicion."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, Create, Dot, FadeIn, FadeOut, Line, VGroup, Write, always_redraw,
    ValueTracker, Transform, linear, smooth, DashedLine, ORIGIN,
)

from physics import ch01_measurement as c1
from physics.integrators import integrate, rk4_step

from animations.common import (
    ACCENT, ANALYTIC, BORDER, ERROR, FAINT, FG, MUTED, NUMERIC, PhysicsScene, T, M, make_axes, panel,
)


class DimensionalAnalysis(PhysicsScene):
    chapter_number = 1
    chapter_title = "Medición"

    def construct(self):
        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        dims = VGroup(
            M(r"[m] = \mathsf{M}", 40), M(r"[L] = \mathsf{L}", 40),
            M(r"[g] = \mathsf{L}\,\mathsf{T}^{-2}", 40), M(r"[T] = \mathsf{T}", 40),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(LEFT * 3.4 + UP * 0.1)
        rule = T("Una ecuación física es válida\nsolo si ambos miembros tienen\nlas mismas dimensiones.", 24, MUTED).move_to(RIGHT * 3.4 + UP * 0.1)
        self.say("Las dimensiones base de la mecánica son masa M, longitud L y tiempo T.",
                 *[Write(d) for d in dims], run_time=2.0)
        self.play(FadeIn(rule), run_time=1.0)
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        q = T("¿De qué magnitudes depende el periodo de un péndulo simple?", 28, FG).move_to(UP * 1.9)
        ansatz = M(r"T = k\, L^{a}\, g^{b}\, m^{c}", 56).move_to(UP * 0.3)
        note = T("k es un número sin dimensiones; a, b y c son exponentes por determinar.", 22, MUTED).move_to(DOWN * 1.0)
        self.say("Se propone una ley de potencias y se exige coherencia dimensional.",
                 FadeIn(q), Write(ansatz), run_time=1.6)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        step0 = M(r"\mathsf{T} = (\mathsf{L})^{a}\,(\mathsf{L}\mathsf{T}^{-2})^{b}\,(\mathsf{M})^{c}", 44).move_to(UP * 1.9)
        eqs = VGroup(
            M(r"\mathsf{M}:\quad c = 0", 42),
            M(r"\mathsf{L}:\quad a + b = 0", 42),
            M(r"\mathsf{T}:\quad -2b = 1", 42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(LEFT * 3.6 + DOWN * 0.4)
        sol = VGroup(
            M(r"a = \tfrac{1}{2}", 46), M(r"b = -\tfrac{1}{2}", 46), M(r"c = 0", 46),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 0.9 + DOWN * 0.4)
        result = M(r"T = k\sqrt{L/g}", 54, ACCENT).move_to(RIGHT * 4.3 + DOWN * 0.4)
        self.say("Igualando exponentes de cada dimensión se obtiene un sistema lineal.", Write(step0), run_time=1.6)
        for e in eqs:
            self.play(Write(e), run_time=0.8)
        self.say("La masa no interviene: el periodo no depende de m.", *[FadeIn(s) for s in sol], run_time=1.4)
        arrow = Line(sol.get_right() + RIGHT * 0.25, result.get_left() + LEFT * 0.25, color=FAINT).add_tip(tip_length=0.18)
        self.play(Create(arrow), Write(result), run_time=1.2)
        self.wait(1.2)
        self.clear_stage()

        # ----------------------------------------------------------- numerica
        self.phase("Solución numérica")
        exps = c1.pendulum_exponents()
        mat = M(r"\begin{bmatrix}0&0&1\\1&1&0\\0&-2&0\end{bmatrix}"
                r"\begin{bmatrix}a\\b\\c\end{bmatrix}="
                r"\begin{bmatrix}0\\0\\1\end{bmatrix}", 40).move_to(UP * 1.9)
        lsq = T(f"numpy.linalg.lstsq   →   a = {exps['L']:+.6f}   b = {exps['g']:+.6f}   c = {exps['m']:+.6f}",
                22, NUMERIC, mono=True).move_to(UP * 0.55)
        self.say("Las filas son M, L, T; las columnas, las magnitudes L, g y m.", Write(mat), run_time=1.6)
        self.play(FadeIn(lsq))

        # El analisis dimensional no fija k: se obtiene integrando la EDO del pendulo.
        length, g, th0 = 1.0, 9.81, 0.35
        t_end = 3.0
        ts, ys = integrate(lambda t, y: np.array([y[1], -(g / length) * np.sin(y[0])]),
                           np.array([th0, 0.0]), 0.0, t_end, 900, rk4_step)
        axes = make_axes([0, t_end], [-0.5, 0.5], 5.4, 1.9, 1, 0.5, numbers=False).move_to(LEFT * 3.3 + DOWN * 1.55)
        curve = axes.plot_line_graph(ts, ys[:, 0], add_vertex_dots=False, line_color=NUMERIC, stroke_width=3)
        lab = T("θ(t),  RK4", 18, MUTED).next_to(axes, UP, buff=0.12).align_to(axes, LEFT)

        pivot = np.array([3.3, -0.25, 0.0])
        theta = ValueTracker(th0)
        rod_len = 2.0

        def bob_pos():
            th = theta.get_value()
            return pivot + rod_len * np.array([np.sin(th), -np.cos(th), 0.0])

        rod = always_redraw(lambda: Line(pivot, bob_pos(), color=FG, stroke_width=3))
        bob = always_redraw(lambda: Dot(bob_pos(), radius=0.13, color=ANALYTIC))
        support = Line(pivot + LEFT * 0.5, pivot + RIGHT * 0.5, color=FAINT, stroke_width=4)
        self.play(Create(axes), FadeIn(lab), FadeIn(support), FadeIn(rod), FadeIn(bob), run_time=1.0)
        self.say("La constante k no sigue del análisis dimensional: se mide integrando θ'' = −(g/L) sen θ.", run_time=0.6)

        def theta_at(t):
            return float(np.interp(t, ts, ys[:, 0]))

        clock = ValueTracker(0.0)
        theta.add_updater(lambda m: m.set_value(theta_at(clock.get_value())))
        self.add(theta)
        trace_dot = always_redraw(lambda: Dot(axes.c2p(clock.get_value(), theta_at(clock.get_value())), radius=0.07, color=ACCENT))
        self.add(trace_dot)
        self.play(Create(curve), clock.animate.set_value(t_end), run_time=t_end * 1.6, rate_func=linear)
        theta.clear_updaters()
        self.remove(trace_dot)

        # Convergencia de k cuando la amplitud tiende a cero.
        self.play(FadeOut(axes), FadeOut(curve), FadeOut(lab), FadeOut(rod), FadeOut(bob), FadeOut(support), FadeOut(mat), FadeOut(lsq), run_time=0.6)
        amps = [0.8, 0.4, 0.1, 0.01]
        header = T("amplitud θ₀ (rad)", 19, MUTED, mono=True)
        header2 = T("k = T·√(g/L)", 19, MUTED, mono=True)
        rows = [(header, header2)]
        for a in amps:
            per = c1.pendulum_period_numeric(a, length, g)
            k = per / np.sqrt(length / g)
            rows.append((T(f"{a:g}", 22, FG, mono=True), T(f"{k:.5f}", 22, NUMERIC, mono=True)))
        col1 = VGroup(*[r[0] for r in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        col2 = VGroup(*[r[1] for r in rows]).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        table = VGroup(col1, col2).arrange(RIGHT, buff=1.2, aligned_edge=UP).move_to(LEFT * 3.4 + DOWN * 0.1)
        target = VGroup(M(r"2\pi = 6.28319\ldots", 50, ACCENT), T("límite de amplitud pequeña", 20, MUTED)).arrange(DOWN, buff=0.25).move_to(RIGHT * 3.6 + DOWN * 0.1)
        self.say("Al reducir la amplitud, k converge a 2π: el resultado analítico y el numérico coinciden.",
                 FadeIn(table), run_time=1.4)
        self.play(Write(target), run_time=1.2)
        self.wait(2.0)


class ErrorPropagation(PhysicsScene):
    chapter_number = 1
    chapter_title = "Medición"

    def construct(self):
        an = c1.density_analytic()
        mc = c1.density_monte_carlo()
        m, r, h = c1.CYLINDER["m"], c1.CYLINDER["r"], c1.CYLINDER["h"]

        # ---------------------------------------------------------- fundamentos
        self.phase("Fundamentos")
        f1 = M(r"\rho = \dfrac{m}{\pi r^{2} h}", 56).move_to(LEFT * 3.5 + UP * 0.6)
        f2 = M(r"\left(\dfrac{\sigma_\rho}{\rho}\right)^{2} = \left(\dfrac{\sigma_m}{m}\right)^{2} + \left(2\dfrac{\sigma_r}{r}\right)^{2} + \left(\dfrac{\sigma_h}{h}\right)^{2}", 36).move_to(RIGHT * 2.6 + UP * 0.6)
        self.say("Para magnitudes independientes, los errores relativos se suman en cuadratura; el radio pesa el doble.",
                 Write(f1), run_time=1.6)
        self.play(Write(f2), run_time=2.0)
        self.wait(1.0)
        self.clear_stage()

        # -------------------------------------------------------------- problema
        self.phase("Problema")
        data = VGroup(
            T(f"masa       m = {m.value:.1f} ± {m.sigma} {m.unit}", 25, FG, mono=True),
            T(f"radio      r = {r.value:.3f} ± {r.sigma} {r.unit}", 25, FG, mono=True),
            T(f"altura     h = {h.value:.2f} ± {h.sigma} {h.unit}", 25, FG, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(UP * 0.9)
        ask = T("¿Cuál es la densidad del cilindro y con qué incertidumbre?", 26, MUTED).move_to(DOWN * 1.3)
        self.say("Se mide un cilindro de aluminio con tres instrumentos distintos.", FadeIn(data), run_time=1.4)
        self.play(FadeIn(ask))
        self.wait(1.2)
        self.clear_stage()

        # ---------------------------------------------------------- analitica
        self.phase("Solución analítica")
        rel = VGroup(
            M(rf"\dfrac{{\sigma_m}}{{m}} = {an['rel_m']*100:.2f}\,\%", 38),
            M(rf"2\dfrac{{\sigma_r}}{{r}} = {an['rel_r']*100:.2f}\,\%", 38),
            M(rf"\dfrac{{\sigma_h}}{{h}} = {an['rel_h']*100:.2f}\,\%", 38),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to(LEFT * 3.8 + DOWN * 0.1)
        out = VGroup(
            M(rf"\rho = {an['rho']:.3f}\ \mathrm{{g/cm^3}}", 44),
            M(rf"\sigma_\rho = {an['sigma_quadrature']:.3f}\ \mathrm{{g/cm^3}}", 44, ACCENT),
            T(f"cota máxima (suma lineal): {an['sigma_worst_case']:.3f}", 20, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.2 + DOWN * 0.1)
        self.say("Cada contribución relativa se calcula por separado; la del radio domina.", *[Write(x) for x in rel], run_time=2.2)
        self.play(FadeIn(out, shift=UP * 0.1), run_time=1.2)
        self.wait(1.5)
        self.clear_stage()

        # ----------------------------------------------------------- numerica
        self.phase("Solución numérica")
        samples = mc["samples"]
        lo, hi = an["rho"] - 4 * an["sigma_quadrature"], an["rho"] + 4 * an["sigma_quadrature"]
        bins = np.linspace(lo, hi, 41)
        counts, _ = np.histogram(samples, bins=bins, density=True)
        ymax = counts.max() * 1.35
        axes = make_axes([lo, hi], [0, ymax], 7.4, 3.5, (hi - lo) / 4, ymax, numbers=False).move_to(LEFT * 2.7 + DOWN * 0.1)
        title = T("distribución de ρ", 20, MUTED).next_to(axes, UP, buff=0.15).align_to(axes, LEFT)
        ticks = VGroup(*[
            T(f"{mu_:.3f}", 15, MUTED, mono=True).next_to(axes.c2p(mu_, 0), DOWN, buff=0.12)
            for mu_ in (an["rho"] - 3 * an["sigma_quadrature"], an["rho"], an["rho"] + 3 * an["sigma_quadrature"])
        ])
        unit_lab = T("ρ  [g/cm³]", 15, MUTED, mono=True).next_to(ticks[-1], RIGHT, buff=0.3)
        self.play(FadeIn(ticks), FadeIn(unit_lab), run_time=0.4)
        self.say("Monte Carlo: se muestrea cada medición con su incertidumbre y se propaga punto por punto.",
                 Create(axes), FadeIn(title), run_time=1.2)

        # La histograma crece con el numero de muestras acumuladas.
        for n in (400, 4000, 40000, 400000):
            c, _ = np.histogram(samples[:n], bins=bins, density=True)
            bars = VGroup()
            for i, v in enumerate(c):
                x0, x1 = axes.c2p(bins[i], 0), axes.c2p(bins[i + 1], 0)
                top = axes.c2p(bins[i], v)
                from manim import Rectangle
                rect = Rectangle(width=(x1 - x0)[0] * 0.92, height=max(top[1] - x0[1], 1e-3),
                                 stroke_width=0, fill_color=NUMERIC, fill_opacity=0.75)
                rect.move_to([(x0[0] + x1[0]) / 2, x0[1] + rect.height / 2, 0])
                bars.add(rect)
            counter = T(f"N = {n:,}".replace(",", " "), 20, FG, mono=True).next_to(axes, UP, buff=0.15).align_to(axes, RIGHT)
            if n == 400:
                self.play(FadeIn(bars), FadeIn(counter), run_time=0.8)
            else:
                self.play(Transform(old_bars, bars), Transform(old_counter, counter), run_time=0.9)
                bars, counter = old_bars, old_counter
            old_bars, old_counter = bars, counter
            self.wait(0.3)

        # curva normal analitica sobre el histograma
        sigma, mu = an["sigma_quadrature"], an["rho"]
        gauss = axes.plot(lambda x: np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                          x_range=[lo, hi], color=ANALYTIC, stroke_width=3)
        self.play(Create(gauss), run_time=1.6)
        cmp_ = VGroup(
            T("analítico  σ = " + f"{an['sigma_quadrature']:.4f}", 17, ANALYTIC, mono=True),
            T("Monte Carlo σ = " + f"{mc['std']:.4f}", 17, NUMERIC, mono=True),
            T("diferencia relativa " + f"{abs(mc['std']-an['sigma_quadrature'])/an['sigma_quadrature']*100:.2f} %", 17, MUTED, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(RIGHT * 4.75 + UP * 0.1)
        self.say("La propagación analítica y la simulación coinciden dentro de la precisión estadística.",
                 FadeIn(cmp_), run_time=1.2)
        self.wait(2.0)
