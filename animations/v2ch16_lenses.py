"""Volumen 2, capitulo 16. Optica geometrica: espejos y lentes."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch16_lenses as m16

CT = "Óptica geométrica: espejos y lentes"


class ThinLensImage(PhysicsScene):
    volume = 2
    chapter_number = 16
    chapter_title = CT

    def construct(self):
        f = m16.focal_lensmaker()
        s = np.linspace(0.12, 0.5, 100)
        sn = np.array([0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        curve_scene(
            self,
            found=[r"\dfrac{1}{f} = (n-1)\left(\dfrac{1}{R_1} - \dfrac{1}{R_2}\right)\qquad(\text{ecuación del fabricante})", r"\dfrac{1}{s} + \dfrac{1}{s'} = \dfrac{1}{f},\qquad m = -\dfrac{s'}{s}"],
            note="Los espejos esféricos cumplen la misma ecuación con f = R/2.",
            fcap="Lentes delgadas y espejos esféricos.",
            problem=["Una lente biconvexa de vidrio (n = 1.5, R₁ = +10 cm, R₂ = −10 cm) forma la imagen de un objeto axial.",
                     "Halle f y s′ para s = 30 cm, y la imagen en un espejo cóncavo de R = 40 cm con el objeto a 60 cm."],
            pcap="Se comparan con el trazado exacto de rayos.",
            analytic=[(rf"f = {f * 100:.3f}\ \mathrm{{cm}},\qquad s'(30\ \mathrm{{cm}}) = {m16.image_distance_analytic(0.3, f) * 100:.4f}\ \mathrm{{cm}}", ACCENT),
                      rf"\text{{espejo}}:\ s' = {m16.mirror_image_analytic(0.6, 0.4) * 100:.4f}\ \mathrm{{cm}}"],
            acap="Imagen real invertida en ambos casos.",
            numeric="Se traza un rayo paraxial (h = 1 mm) con la ley de Snell vectorial exacta en cada superficie esférica, y se halla dónde corta el eje.",
            xr=(0.1, 0.5), yr=(0, 0.7), xticks=(0.1, 0.3, 0.5), yticks=(0, 0.2, 0.4, 0.6), xlabel="s [m]", ylabel="s′ [m]",
            curves=[(s, m16.image_distance_analytic(s, f), ANALYTIC)], dots=(sn, [m16.lens_image_numeric(float(x)) for x in sn]),
            legend=[(ANALYTIC, "1/s + 1/s′ = 1/f"), (NUMERIC, "trazado de rayos")],
            table=(["magnitud", "numérico", "analítico"],
                   [("s′ lente [cm]", f"{m16.lens_image_numeric(0.3) * 100:.5f}", f"{m16.image_distance_analytic(0.3, f) * 100:.5f}"),
                    ("s′ espejo [cm]", f"{m16.mirror_image_numeric(0.6, 0.4) * 100:.5f}", f"{m16.mirror_image_analytic(0.6, 0.4) * 100:.5f}")]),
            final="La diferencia se debe a la aberración esférica residual y al espesor finito.", fsize=32)


class TwoLensSystem(PhysicsScene):
    volume = 2
    chapter_number = 16
    chapter_title = CT

    def construct(self):
        d = np.linspace(0.0, 0.4, 100)
        dn = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        za, ma = m16.two_lens_analytic()
        zn, mn = m16.two_lens_matrix()
        curve_scene(
            self,
            found=[r"\begin{pmatrix} y' \\ \theta' \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}\begin{pmatrix} y \\ \theta \end{pmatrix}",
                   r"\text{imagen: } B = 0\ \text{en la matriz objeto}\to\text{imagen}"],
            note="Las matrices ABCD permiten componer cualquier número de elementos ópticos paraxiales.",
            fcap="Sistema de dos lentes y matrices de rayos.",
            problem=["Dos lentes delgadas, f₁ = 20 cm y f₂ = 10 cm, separadas 15 cm. Un objeto está a 50 cm de la primera.",
                     "Halle la posición y el aumento de la imagen final, y cómo cambia la posición con la separación."],
            pcap="Se aplica la ecuación de las lentes dos veces y se compara con el método matricial.",
            analytic=[(rf"s_1' = {m16.image_distance_analytic(0.5, 0.2) * 100:.3f}\ \mathrm{{cm}},\quad s_2 = d - s_1' = {(0.15 - m16.image_distance_analytic(0.5, 0.2)) * 100:.3f}\ \mathrm{{cm}}", ACCENT),
                      rf"s_2' = {za * 100:.4f}\ \mathrm{{cm}},\qquad m = {ma:.5f}"],
            acap="El objeto virtual para la segunda lente se trata con s₂ < 0.",
            numeric="Producto de matrices (propagación libre y lente delgada) y búsqueda del plano conjugado con B = 0.",
            xr=(0, 0.4), yr=(-0.3, 0.3), xticks=(0, 0.1, 0.2, 0.3, 0.4), yticks=(-0.2, 0, 0.2), xlabel="d [m]", ylabel="s₂′ [m]",
            curves=[(d, [m16.two_lens_analytic(0.5, m16.F1, m16.F2, float(x))[0] for x in d], ANALYTIC)], dots=(dn, [m16.two_lens_matrix(0.5, m16.F1, m16.F2, float(x))[0] for x in dn]),
            legend=[(ANALYTIC, "dos ecuaciones de lente"), (NUMERIC, "matrices ABCD")],
            table=(["magnitud", "numérico", "analítico"], [("s₂′ [cm]", f"{zn * 100:.6f}", f"{za * 100:.6f}"), ("aumento", f"{mn:.6f}", f"{ma:.6f}")]),
            final="Ambos métodos coinciden a precisión de máquina.", fsize=30)
