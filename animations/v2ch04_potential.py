"""Volumen 2, capitulo 4. Potencial electrico."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, Create, DashedLine, Dot, FadeIn, FadeOut, VGroup, LaggedStart, VMobject

from animations.common import (
    ACCENT, ANALYTIC, ERROR, FAINT, FG, MUTED, NUMERIC, M, PhysicsScene, T, axes_block, legend_block, mono_table, polyline,
    text_lines,
)
from physics import v2ch04_potential as e4


class DiskPotential(PhysicsScene):
    volume = 2
    chapter_number = 4
    chapter_title = "Potencial eléctrico"

    def construct(self):
        s, R = e4.SIGMA, e4.R_DISK
        z0 = 0.05
        v_an, v_nu = float(e4.disk_potential_analytic(z0)), e4.disk_potential_numeric(z0)
        e_an = float(e4.disk_field_analytic(z0))
        e_nu = e4.field_from_potential_numeric(lambda z: float(e4.disk_potential_analytic(z)), z0)
        self.foundations(
            [r"V = k\int\dfrac{dq}{r},\qquad \vec E = -\nabla V,\qquad U = qV",
             r"\text{Disco: } V(z) = \dfrac{\sigma}{2\varepsilon_0}\left(\sqrt{z^{2} + R^{2}} - z\right)"],
            note="El potencial es un escalar: se suma sin componentes; el campo se obtiene después como su gradiente.",
            caption="Potencial de distribuciones continuas.", size=38)
        self.problem(
            ["Un disco de radio 10 cm tiene densidad superficial de carga σ = 2 µC/m². Halle V y E sobre su eje a 5 cm del centro,",
             "sumando los potenciales de anillos concéntricos y derivando después."],
            caption="El potencial se toma cero en el infinito.")
        self.analytic(
            [(rf"V(0.05\ \mathrm{{m}}) = {v_an / 1000:.3f}\ \mathrm{{kV}}", ACCENT),
             r"E_z = -\dfrac{dV}{dz} = \dfrac{\sigma}{2\varepsilon_0}\left(1 - \dfrac{z}{\sqrt{z^{2} + R^{2}}}\right)",
             (rf"E_z(0.05\ \mathrm{{m}}) = {e_an / 1000:.3f}\ \mathrm{{kV/m}}", ACCENT)],
            size=36, caption="Lejos del disco el potencial decae como 1/z, como el de una carga puntual.")

        self.numeric_phase("V(z) se obtiene con Simpson sobre anillos (dq = σ2πs ds); E se calcula derivando V con diferencias centradas.")
        axv, dv = axes_block((0, 0.3), (0, 12000), 4.6, 3.3, LEFT * 4.5 + DOWN * 0.1, (0, 0.1, 0.2, 0.3), (0, 4000, 8000, 12000), "z [m]", "V [V]")
        axe, de = axes_block((0, 0.3), (0, 120000), 4.6, 3.3, RIGHT * 0.4 + DOWN * 0.1, (0, 0.1, 0.2, 0.3), (0, 40000, 80000, 120000), "z [m]", "E [V/m]")
        zs = np.linspace(0, 0.3, 120)
        cv = polyline(axv, zs, e4.disk_potential_analytic(zs), ANALYTIC, 3.3)
        ce = polyline(axe, zs, e4.disk_field_analytic(zs), ANALYTIC, 3.3)
        self.play(Create(axv), FadeIn(dv), Create(axe), FadeIn(de), run_time=1.0)
        self.play(Create(cv), Create(ce), run_time=2.0)
        pz = np.linspace(0.01, 0.29, 12)
        dots_v = VGroup(*[Dot(axv.c2p(float(z), e4.disk_potential_numeric(float(z))), radius=0.06, color=NUMERIC) for z in pz])
        dots_e = VGroup(*[Dot(axe.c2p(float(z), e4.field_from_potential_numeric(lambda x: e4.disk_potential_numeric(x), float(z), h=1e-5)), radius=0.06, color=NUMERIC) for z in pz])
        self.play(LaggedStart(*[FadeIn(d_, scale=1.4) for d_ in list(dots_v) + list(dots_e)], lag_ratio=0.05), run_time=2.0)
        tab = mono_table(["magnitud", "numérico", "analítico"],
                         [("V(5 cm) [V]", f"{v_nu:.4f}", f"{v_an:.4f}"), ("E(5 cm) [V/m]", f"{e_nu:.3f}", f"{e_an:.3f}")],
                         size=15, colors=[MUTED, NUMERIC, ANALYTIC], gap=0.3).move_to(RIGHT * 4.6 + UP * 1.0)
        self.say("La suma de anillos reproduce V; derivar V numéricamente da el mismo campo que la fórmula cerrada.", FadeIn(tab), run_time=1.3)
        self.wait(2.5)


class EquipotentialPair(PhysicsScene):
    volume = 2
    chapter_number = 4
    chapter_title = "Potencial eléctrico"

    def construct(self):
        qa, qb, d = e4.QA, e4.QB, e4.D
        err = e4.field_grid_error()
        u = e4.interaction_energy()
        self.foundations(
            [r"V(\vec r) = k\sum_i \dfrac{q_i}{r_i},\qquad U = k\dfrac{q_aq_b}{d}",
             r"\vec E \perp \text{equipotenciales},\qquad \vec E = -\nabla V"],
            note="Las superficies equipotenciales son perpendiculares a las líneas de campo y el campo apunta hacia el potencial decreciente.",
            caption="Equipotenciales y relación entre E y V.", size=40)
        self.problem(
            ["Dos cargas de +4 nC y −4 nC están a 20 cm de distancia. Trace las equipotenciales,",
             "calcule E = −∇V sobre una malla y compárelo con el campo coulombiano. ¿Cuánta energía almacena el par?"],
            caption="Distribución en el plano de las cargas.")
        self.analytic(
            [(rf"U = k\dfrac{{q_aq_b}}{{d}} = {u * 1e6:.3f}\ \mu\mathrm{{J}}", ACCENT),
             r"V = 0 \text{ en el plano mediatriz (por simetría)}",
             r"\vec E = k\sum_i \dfrac{q_i\,\vec r_i}{r_i^{3}}"],
            size=36, caption="La energía es negativa: hay que realizar trabajo externo para separar las cargas.")

        self.numeric_phase("V se evalúa en una malla de 801×801 y E se obtiene con np.gradient (diferencias centradas).")
        sc = 5.0
        org = np.array([-3.4, -0.1, 0.0])
        axes_pos = lambda x, y: org + sc * np.array([x, y, 0.0])  # noqa: E731
        levels = [-400, -250, -150, -80, -30, 0, 30, 80, 150, 250, 400]
        curves = VGroup()
        from manim import Axes
        ax = Axes(x_range=[-0.5, 0.5, 0.25], y_range=[-0.5, 0.5, 0.25], x_length=5.0, y_length=5.0, axis_config={"stroke_width": 0, "include_tip": False}).move_to(org)
        for lv in levels:
            col = ERROR if lv > 0 else (NUMERIC if lv < 0 else FG)
            try:
                c = ax.plot_implicit_curve(lambda x, y, lv=lv: float(e4.potential_pair(x, y)) - lv, color=col, stroke_width=1.8, min_depth=7, max_quads=1500)
                curves.add(c)
            except Exception:
                pass
        qdots = VGroup(Dot(ax.c2p(-d / 2, 0), radius=0.1, color=ERROR), Dot(ax.c2p(d / 2, 0), radius=0.1, color=NUMERIC))
        self.play(FadeIn(qdots), run_time=0.6)
        self.play(LaggedStart(*[Create(c_) for c_ in curves], lag_ratio=0.08), run_time=4.0)
        gxs, ex, ey = e4.field_grid_numeric()
        arrows = VGroup()
        for xv, yv in [(-0.3, 0.1), (-0.3, -0.1), (0.3, 0.1), (0.3, -0.1), (0.0, 0.15), (0.0, -0.15), (0.0, 0.0), (-0.12, 0.2), (0.12, 0.2), (-0.12, -0.2), (0.12, -0.2)]:
            i = int(np.argmin(np.abs(gxs - xv)))
            j = int(np.argmin(np.abs(gxs - yv)))
            v_ = np.array([ex[i, j], ey[i, j], 0.0])
            v_ = v_ / np.linalg.norm(v_) * 0.45
            p0 = ax.c2p(xv, yv)
            arrows.add(Arrow(p0 - v_ / 2, p0 + v_ / 2, buff=0, color=ACCENT, stroke_width=3, max_tip_length_to_length_ratio=0.35))
        self.play(FadeIn(arrows), run_time=1.0)
        tab = mono_table(["magnitud", "valor"],
                         [("error máx. de −∇V", f"{err:.1e}"), ("U del par [µJ]", f"{u * 1e6:.4f}"), ("V en x = 0 [V]", f"{float(e4.potential_pair(0.0, 0.1)):.2e}")],
                         size=15, colors=[MUTED, FG], gap=0.4).move_to(RIGHT * 3.8 + UP * 0.8)
        note = text_lines([("rojo: V > 0, azul: V < 0, blanco: V = 0", MUTED), ("ámbar: E = −∇V (malla numérica)", MUTED), ("error relativo lejos de las cargas (r > 12 cm)", MUTED)], 13).move_to(RIGHT * 3.8 + DOWN * 1.1)
        self.say("Las flechas (gradiente numérico) cruzan las equipotenciales en ángulo recto y su error relativo es inferior al 0.5 %.", FadeIn(tab), FadeIn(note), run_time=1.4)
        self.wait(2.5)
