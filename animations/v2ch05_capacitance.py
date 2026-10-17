"""Volumen 2, capitulo 5. Capacitancia y dielectricos."""
from __future__ import annotations

import numpy as np
from manim import Circle, DOWN, LEFT, RIGHT, UP, Create, DashedLine, Dot, FadeIn, FadeOut, VGroup, LaggedStart, VMobject

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, battery_symbol, capacitor_symbol, legend_block,
    mono_table, polyline, text_lines, wire,
)
from physics import v2ch05_capacitance as e5


def contour_paths(v: np.ndarray, xs: np.ndarray, level: float, to_scene) -> VMobject:
    """Curva de nivel de la solucion numerica por marching squares (un unico VMobject con muchos segmentos)."""
    m = VMobject(stroke_width=2.0)
    n = len(xs)
    for i in range(n - 1):
        for j in range(n - 1):
            c = [v[i, j], v[i + 1, j], v[i + 1, j + 1], v[i, j + 1]]
            pts = []
            edges = [((i, j), (i + 1, j)), ((i + 1, j), (i + 1, j + 1)), ((i + 1, j + 1), (i, j + 1)), ((i, j + 1), (i, j))]
            for (p0, p1) in edges:
                f0, f1 = v[p0] - level, v[p1] - level
                if f0 * f1 < 0:
                    t = f0 / (f0 - f1)
                    pts.append((xs[p0[0]] + t * (xs[p1[0]] - xs[p0[0]]), xs[p0[1]] + t * (xs[p1[1]] - xs[p0[1]])))
            if len(pts) >= 2:
                m.start_new_path(to_scene(*pts[0]))
                m.add_line_to(to_scene(*pts[1]))
    return m


class CoaxialLaplace(PhysicsScene):
    volume = 2
    chapter_number = 5
    chapter_title = "Capacitancia y dieléctricos"

    def construct(self):
        a, b = e5.A_IN, e5.B_OUT
        c_an = e5.coax_capacitance_per_length_analytic()
        xs, V, h, fixed = e5.coax_laplace_numeric(n=161, iterations=4000)
        gx, gy = np.gradient(V, h, h)
        u = 0.5 * e5.EPS0 * np.sum((gx**2 + gy**2)[~fixed]) * h * h
        c_nu = 2 * u
        self.foundations(
            [r"C = \dfrac{Q}{\Delta V},\qquad U = \tfrac{1}{2}C\,(\Delta V)^{2} = \tfrac{1}{2}\varepsilon_0\int E^{2}\,dV",
             r"\nabla^{2}V = 0\qquad\text{(región sin cargas)},\qquad C' = \dfrac{2\pi\varepsilon_0}{\ln(b/a)}\ \text{(coaxial, por unidad de longitud)}"],
            note="La capacitancia depende solo de la geometría: se calcula resolviendo la ecuación de Laplace entre los conductores.",
            caption="Capacitancia y ecuación de Laplace.", size=34)
        self.problem(
            ["Un cable coaxial tiene un conductor interno de radio 0.5 cm y uno externo de radio 2 cm. Halle su capacitancia por unidad de longitud",
             "resolviendo la ecuación de Laplace con una malla y calculándola a partir de la energía almacenada."],
            caption="El conductor interno está a potencial V₀ y el externo a tierra.")
        self.analytic(
            [r"E(r) = \dfrac{\lambda}{2\pi\varepsilon_0 r},\qquad \Delta V = \dfrac{\lambda}{2\pi\varepsilon_0}\ln\dfrac{b}{a}",
             (rf"C' = \dfrac{{\lambda}}{{\Delta V}} = \dfrac{{2\pi\varepsilon_0}}{{\ln(b/a)}} = {c_an * 1e12:.2f}\ \mathrm{{pF/m}}", ACCENT)],
            size=36, caption="Solo depende de la razón b/a de los radios.")

        self.numeric_phase("Relajación sobremuestreada (SOR rojo-negro) en una malla de 161×161 hasta converger; C se calcula de ½ε₀∫E² dA.")
        center = np.array([-4.0, -0.15, 0.0])
        half = 2.1
        scale = half / (b * 1.02)
        to_scene = lambda x, y: center + scale * np.array([x, y, 0.0])  # noqa: E731
        sub = slice(None, None, 2)
        vs, xsub = V[sub, sub], xs[sub]
        levels = np.linspace(0.1, 0.9, 9)
        rings = VGroup()
        for k, lv in enumerate(levels):
            c = contour_paths(vs, xsub, float(lv), to_scene)
            c.set_stroke(color=NUMERIC if lv < 0.5 else ACCENT, width=2.0)
            rings.add(c)
        inner = Circle(radius=a * scale, stroke_color=ACCENT, stroke_width=3, fill_color=ACCENT, fill_opacity=0.25).move_to(center)
        outer = Circle(radius=b * scale, stroke_color=FG, stroke_width=3).move_to(center)
        self.play(Create(outer), FadeIn(inner), run_time=1.0)
        self.play(LaggedStart(*[Create(r_) for r_ in rings], lag_ratio=0.1), run_time=2.5)
        mid = len(xs) // 2
        r_prof = xs[mid:]
        v_prof = V[mid, mid:]
        sel = (r_prof >= a) & (r_prof <= b)
        ax, deco = axes_block((a * 100, b * 100), (0, 1.0), 4.2, 3.0, RIGHT * 0.9 + DOWN * 0.1, (0.5, 1.0, 1.5, 2.0), (0, 0.5, 1.0), "r [cm]", "V / V₀")
        rr = np.linspace(a, b, 100)
        c_an_curve = polyline(ax, rr * 100, np.log(b / rr) / np.log(b / a), ANALYTIC, 3.2)
        dots = VGroup(*[Dot(ax.c2p(float(r) * 100, float(v)), radius=0.04, color=NUMERIC) for r, v in zip(r_prof[sel][::2], v_prof[sel][::2])])
        self.play(Create(ax), FadeIn(deco), Create(c_an_curve), run_time=1.2)
        self.play(FadeIn(dots), run_time=1.0)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("C' [pF/m]", f"{c_nu * 1e12:.3f}", f"{c_an * 1e12:.3f}"), ("error relativo", f"{abs(c_nu - c_an) / c_an * 100:.2f} %", "")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.6 + UP * 1.2)
        note = text_lines([("curvas: V = 0.1 … 0.9 V₀ (numérico)", MUTED), ("puntos: V(r) numérico", MUTED)], 14).move_to(RIGHT * 4.6 + DOWN * 0.9)
        self.say(f"La malla reproduce el perfil logarítmico V(r); el error de C ({abs(c_nu - c_an) / c_an * 100:.1f} %) viene de discretizar las fronteras circulares.", FadeIn(tab), FadeIn(note), run_time=1.3)
        self.wait(2.5)


class CapacitorNetwork(PhysicsScene):
    volume = 2
    chapter_number = 5
    chapter_title = "Capacitancia y dieléctricos"

    def construct(self):
        v = e5.V_SRC
        ceq_a = e5.network_ceq_analytic()
        nd = e5.network_nodal_numeric()
        d = e5.dielectric_energies()
        self.foundations(
            [r"\text{Paralelo: } C_{eq} = \sum_i C_i\qquad\text{Serie: } \dfrac{1}{C_{eq}} = \sum_i \dfrac{1}{C_i}",
             r"C = \kappa C_0,\qquad U = \dfrac{Q^{2}}{2C} = \tfrac{1}{2}CV^{2}"],
            note="Un dieléctrico multiplica la capacitancia por κ; la energía almacenada cambia según se mantenga V o Q.",
            caption="Combinación de capacitores y dieléctricos.", size=38)
        self.problem(
            ["Un capacitor de 2 µF está en serie con dos capacitores en paralelo de 3 µF y 6 µF, conectados a 12 V. Halle C_eq, la carga y la energía.",
             "Luego inserte un dieléctrico κ = 4 en un capacitor de 2 µF: ¿cómo cambia su energía con y sin batería?"],
            caption="Capacitores inicialmente descargados.")
        self.analytic(
            [(rf"C_{{23}} = 3 + 6 = 9\ \mu\mathrm{{F}},\qquad C_{{eq}} = \dfrac{{2\cdot 9}}{{2 + 9}} = {ceq_a * 1e6:.4f}\ \mu\mathrm{{F}}", ACCENT),
             rf"Q = C_{{eq}}V = {ceq_a * v * 1e6:.2f}\ \mu\mathrm{{C}},\qquad U = \tfrac12 C_{{eq}}V^2 = {0.5 * ceq_a * v**2 * 1e6:.1f}\ \mu\mathrm{{J}}",
             r"\text{Dieléctrico: } U_V = \kappa U_0\ (V\text{ cte}),\qquad U_Q = U_0/\kappa\ (Q\text{ cte})"],
            size=32, caption="El capacitor en serie comparte la misma carga; los de la rama paralela, el mismo voltaje.")

        self.numeric_phase("Se resuelve la conservación de la carga en el nodo central con un sistema lineal y se comparan las energías con dieléctrico.")
        x0, y0 = -6.0, 0.4
        bat = battery_symbol((x0 + 0.2, y0 - 0.9), vertical=True)
        c1 = capacitor_symbol((x0 + 2.4, y0 + 0.8), vertical=False)
        c2 = capacitor_symbol((x0 + 5.0, y0 + 1.6), vertical=True)
        c3 = capacitor_symbol((x0 + 5.0, y0 - 0.1), vertical=True)
        wires = VGroup(
            wire((x0 + 0.2, y0 - 0.6), (x0 + 0.2, y0 + 0.8), (x0 + 2.0, y0 + 0.8)), wire((x0 + 2.8, y0 + 0.8), (x0 + 3.8, y0 + 0.8)),
            wire((x0 + 3.8, y0 + 0.8), (x0 + 3.8, y0 + 1.6), (x0 + 4.65, y0 + 1.6)), wire((x0 + 3.8, y0 + 0.8), (x0 + 3.8, y0 - 0.1), (x0 + 4.65, y0 - 0.1)),
            wire((x0 + 5.35, y0 + 1.6), (x0 + 6.0, y0 + 1.6), (x0 + 6.0, y0 - 0.1), (x0 + 5.35, y0 - 0.1)),
            wire((x0 + 6.0, y0 + 0.75), (x0 + 6.0, y0 - 1.5), (x0 + 0.2, y0 - 1.5), (x0 + 0.2, y0 - 1.2)))
        labs = VGroup(T("12 V", 15, MUTED).next_to(bat, LEFT, buff=0.15), T("C₁ = 2 µF", 15, MUTED).next_to(c1, UP, buff=0.3),
                      T("C₂ = 3 µF", 15, MUTED).next_to(c2, UP, buff=0.3), T("C₃ = 6 µF", 15, MUTED).next_to(c3, DOWN, buff=0.3))
        self.play(Create(wires), FadeIn(bat), FadeIn(c1), FadeIn(c2), FadeIn(c3), FadeIn(labs), run_time=2.0)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("C_eq [µF]", f"{nd['c_eq'] * 1e6:.5f}", f"{ceq_a * 1e6:.5f}"), ("V nodo [V]", f"{nd['vx']:.4f}", f"{v * 2e-6 / (2e-6 + 9e-6):.4f}"),
                          ("Q total [µC]", f"{nd['q'] * 1e6:.3f}", f"{ceq_a * v * 1e6:.3f}"), ("U [µJ]", f"{nd['energy'] * 1e6:.2f}", f"{0.5 * ceq_a * v**2 * 1e6:.2f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 3.4 + UP * 1.5)
        diel = mono_table(["dieléctrico κ = 4", "U [µJ]"],
                          [("sin dieléctrico", f"{d['u0'] * 1e6:.1f}"), ("V constante (con batería)", f"{d['u_const_v'] * 1e6:.1f}"), ("Q constante (aislado)", f"{d['u_const_q'] * 1e6:.1f}")],
                          size=15, colors=[MUTED, FG], gap=0.4).move_to(RIGHT * 3.4 + DOWN * 1.1)
        self.say("Con batería la energía se multiplica por κ (el trabajo lo aporta la fuente); aislado, se divide por κ.", FadeIn(tab), FadeIn(diel), run_time=1.4)
        self.wait(2.5)
