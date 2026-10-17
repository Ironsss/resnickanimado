"""Capitulo 23. Primera ley de la termodinamica."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, Polygon, VGroup, LaggedStart

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table,
    polyline, text_lines,
)
from physics import ch23_first_law as c23


class GasProcesses(PhysicsScene):
    chapter_number = 23
    chapter_title = "La primera ley de la termodinámica"

    def construct(self):
        R, P1, V1, V2 = c23.R, c23.P1, c23.V1, c23.V2
        t1 = c23.temperature(P1, V1)
        w_iso, w_isb, w_adi = (c23.work_analytic(k) for k in ("isothermal", "isobaric", "adiabatic"))
        rows = c23.first_law_table()
        self.foundations(
            [r"\Delta U = Q - W,\qquad W = \int_{V_1}^{V_2} p\,dV",
             r"\Delta U = nC_V\Delta T,\qquad C_V = \tfrac{3}{2}R\ \ (\text{gas monoatómico ideal})"],
            note="El trabajo es el área bajo la curva p–V; depende del camino, mientras que ΔU solo depende de los estados.",
            caption="Primera ley de la termodinámica para un gas ideal.", size=38)
        self.problem(
            ["Un mol de gas ideal monoatómico a 2.0×10⁵ Pa y 10 L se expande hasta 20 L por tres caminos:",
             "isotérmico, isobárico y adiabático. Halle W, ΔU y Q en cada uno."],
            caption="T₁ = p₁V₁/(nR) = 240.6 K.")
        self.analytic(
            [(rf"W_{{isot}} = nRT\ln\dfrac{{V_2}}{{V_1}} = {w_iso:.1f}\ \mathrm{{J}}", ACCENT),
             (rf"W_{{isob}} = p\,\Delta V = {w_isb:.1f}\ \mathrm{{J}}", ACCENT),
             (rf"W_{{adiab}} = \dfrac{{p_1V_1 - p_2V_2}}{{\gamma - 1}} = {w_adi:.1f}\ \mathrm{{J}},\qquad Q_{{adiab}} = 0", ACCENT)],
            size=34, caption="Con el mismo volumen final, el trabajo isobárico es mayor y el adiabático el menor.")

        self.numeric_phase("Se integra ∫p dV con Simpson (400 subintervalos); para la adiabática se integra dp/dV = −γp/V con RK4.")
        sc_x, sc_y = 7.0 / 0.02, 3.6 / 5.0e5
        ax, deco = axes_block((0.01, 0.02), (0.5e5, 2.5e5), 6.4, 3.5, LEFT * 3.3 + DOWN * 0.1, (0.01, 0.0125, 0.015, 0.0175, 0.02), (0.5e5, 1.0e5, 1.5e5, 2.0e5, 2.5e5), "V [m³]", "p [Pa]", "{:g}")
        vs = np.linspace(V1, V2, 80)
        iso = polyline(ax, vs, c23.pressure_path("isothermal", vs), ANALYTIC, 3.5)
        isb = polyline(ax, vs, c23.pressure_path("isobaric", vs), ACCENT, 3.5)
        adi = polyline(ax, vs, c23.pressure_path("adiabatic", vs), NUMERIC, 3.5)
        area = Polygon(*[ax.c2p(float(v_), float(p_)) for v_, p_ in zip(vs, c23.pressure_path("isothermal", vs))], ax.c2p(V2, 0.5e5), ax.c2p(V1, 0.5e5),
                       stroke_width=0, fill_color=ANALYTIC, fill_opacity=0.12)
        leg = legend_block([(ANALYTIC, "isotérmico"), (ACCENT, "isobárico"), (NUMERIC, "adiabático")], RIGHT * 0.5 + UP * 2.0)
        self.play(Create(ax), FadeIn(deco), FadeIn(leg), run_time=1.0)
        self.play(Create(iso), Create(isb), Create(adi), run_time=2.5)
        self.play(FadeIn(area), run_time=0.8)
        tab = mono_table(["camino", "W num [J]", "W teoría [J]", "ΔU [J]", "Q [J]"],
                         [(r_["kind"][:7], f"{r_['W']:.2f}", f"{c23.work_analytic(r_['kind']):.2f}", f"{r_['dU']:.1f}", f"{r_['Q']:.1f}") for r_ in rows],
                         size=13, colors=[MUTED, NUMERIC, ANALYTIC, FG, FG], gap=0.18).move_to(RIGHT * 4.0 + DOWN * 0.7)
        self.say("Q = ΔU + W se cumple en cada camino; en el adiabático Q ≈ 0 y el gas se enfría hasta unos 152 K.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class OttoCycle(PhysicsScene):
    chapter_number = 23
    chapter_title = "La primera ley de la termodinámica"

    def construct(self):
        r, g = c23.RATIO, c23.GAMMA_DIA
        eta_an = c23.otto_efficiency_analytic()
        o = c23.otto_numeric()
        st = o["states"]
        self.foundations(
            [r"\oint dU = 0 \;\Rightarrow\; W_{neto} = Q_{entra} - Q_{sale}",
             r"\eta = \dfrac{W_{neto}}{Q_{entra}} = 1 - r^{1-\gamma}\qquad(\text{ciclo de Otto})"],
            note="En un ciclo completo el gas vuelve a su estado inicial: el trabajo neto es el calor neto absorbido.",
            caption="Ciclos termodinámicos.", size=40)
        self.problem(
            ["Un motor de gasolina ideal (Otto) con gas diatómico (γ = 7/5) y relación de compresión 8 absorbe calor a volumen constante",
             "hasta 1800 K. Parte de 300 K y 25 L (1 mol, ≈1 bar). Halle el trabajo neto por ciclo y la eficiencia."],
            caption="Compresión y expansión adiabáticas reversibles.")
        self.analytic(
            [(rf"\eta = 1 - r^{{1-\gamma}} = 1 - 8^{{-0.4}} = {eta_an:.4f}", ACCENT),
             rf"T_B = T_A\,r^{{\gamma-1}} = {st['B'][2]:.1f}\ \mathrm{{K}},\qquad T_D = T_C\,r^{{1-\gamma}} = {st['D'][2]:.1f}\ \mathrm{{K}}",
             rf"Q_{{entra}} = nC_V(T_C - T_B) = {o['Q_in']:.0f}\ \mathrm{{J}}"],
            size=34, caption="La eficiencia solo depende de la relación de compresión y de γ.")

        self.numeric_phase("Las adiabáticas se integran con RK4 (dp/dV = −γp/V) y el trabajo neto con Simpson sobre cada tramo.")
        ax, deco = axes_block((0.0, 0.03), (0, 6e6), 6.4, 3.7, LEFT * 3.3 + DOWN * 0.1, (0.0, 0.01, 0.02, 0.03), (0, 2e6, 4e6, 6e6), "V [m³]", "p [Pa]", "{:g}")
        vab, pab = o["ab"]
        vcd, pcd = o["cd"]
        pa, va, _ = st["A"]; pb, vb, _ = st["B"]; pc, vc, _ = st["C"]; pd, vd, _ = st["D"]
        c_ab, c_cd = polyline(ax, vab, pab, NUMERIC, 3.5), polyline(ax, vcd, pcd, ERROR, 3.5)
        c_bc = polyline(ax, [vb, vc], [pb, pc], ACCENT, 3.5)
        c_da = polyline(ax, [vd, va], [pd, pa], ANALYTIC, 3.5)
        poly = Polygon(*[ax.c2p(float(v_), float(p_)) for v_, p_ in zip(vab, pab)], ax.c2p(vc, pc), *[ax.c2p(float(v_), float(p_)) for v_, p_ in zip(vcd, pcd)],
                       stroke_width=0, fill_color=NUMERIC, fill_opacity=0.12)
        pts = VGroup(*[Dot(ax.c2p(v_, p_), radius=0.08, color=FG) for (p_, v_, _), _ in zip((st["A"], st["B"], st["C"], st["D"]), "ABCD")])
        labs = VGroup(*[T(n_, 18, FG).next_to(ax.c2p(st[n_][1], st[n_][0]), dir_, buff=0.12) for n_, dir_ in zip("ABCD", (DOWN, LEFT, UP, RIGHT))])
        self.play(Create(ax), FadeIn(deco), run_time=1.0)
        self.play(Create(c_ab), Create(c_bc), Create(c_cd), Create(c_da), run_time=3.0)
        self.play(FadeIn(poly), FadeIn(pts), FadeIn(labs), run_time=1.0)
        legend = legend_block([(NUMERIC, "A→B compresión"), (ACCENT, "B→C calentamiento"), (ERROR, "C→D expansión"), (ANALYTIC, "D→A enfriamiento")], RIGHT * 4.6 + UP * 1.9)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("η", f"{o['eta']:.6f}", f"{eta_an:.6f}"), ("W_neto [J]", f"{o['W_net']:.2f}", f"{eta_an * o['Q_in']:.2f}"),
                          ("Q_entra [J]", f"{o['Q_in']:.1f}", f"{o['Q_in']:.1f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.6 + DOWN * 0.6)
        self.say("El área encerrada (azul tenue) es el trabajo neto; la integración numérica reproduce η = 1 − r^(1−γ).", FadeIn(legend), FadeIn(tab), run_time=1.3)
        self.wait(2.5)
