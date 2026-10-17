"""Volumen 2, capitulo 22. Estructura atomica."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch22_atom as m22

CT = "Estructura atómica"


class HydrogenSpectrum(PhysicsScene):
    volume = 2
    chapter_number = 22
    chapter_title = CT

    def construct(self):
        e, _, _ = m22.radial_numeric(8)
        nn = np.arange(3, 9)
        nc = np.linspace(3, 8.3, 100)
        la = np.array([m22.balmer_wavelength_analytic(int(n)) for n in nn])
        ln = np.array([m22.balmer_wavelength_numeric(int(n), e) for n in nn])
        curve_scene(
            self,
            found=[r"E_n = -\dfrac{13.6\ \mathrm{eV}}{n^2},\qquad \dfrac{1}{\lambda} = R_\infty\left(\dfrac{1}{2^2} - \dfrac{1}{n^2}\right)\ (\text{Balmer})",
                   r"-\dfrac{1}{2}u'' - \dfrac{u}{r} = E\,u\qquad(\text{ecuación radial, } l=0,\ \text{unidades atómicas})"],
            note="El modelo de Bohr y la ecuación de Schrödinger dan los mismos niveles de energía del hidrógeno.",
            fcap="Niveles del hidrógeno y serie de Balmer.",
            problem=["Para el átomo de hidrógeno, halle los niveles de energía y las longitudes de onda de la serie de Balmer.",
                     "Verifique los niveles resolviendo numéricamente la ecuación radial."],
            pcap="Transiciones n → 2.",
            analytic=[(rf"E_1 = {m22.level_analytic(1):.4f},\ E_2 = {m22.level_analytic(2):.4f},\ E_3 = {m22.level_analytic(3):.4f}\ \mathrm{{eV}}", ACCENT),
                      rf"\lambda_{{H\alpha}} = {la[0]:.3f}\ \mathrm{{nm}},\quad \lambda_{{H\beta}} = {la[1]:.3f}\ \mathrm{{nm}},\quad \lambda_{{H\gamma}} = {la[2]:.3f}\ \mathrm{{nm}}"],
            acap="Las longitudes de onda convergen al límite de la serie en 364.6 nm.",
            numeric="Diagonalización tridiagonal de la ecuación radial con 32 000 nodos hasta r = 160 a₀; λ = hc/(Eₙ − E₂).",
            xr=(3, 8.3), yr=(380, 680), xticks=(3, 4, 5, 6, 7, 8), yticks=(400, 500, 600), xlabel="n (nivel superior)", ylabel="λ [nm]",
            curves=[(nc, 1e9 / (m22.RINF_M * (0.25 - 1 / nc**2)), ANALYTIC)], dots=(nn, ln), legend=[(ANALYTIC, "fórmula de Rydberg"), (NUMERIC, "autovalores radiales")],
            table=(["magnitud", "numérico", "analítico"],
                   [("E₁ [eV]", f"{e[0]:.5f}", f"{m22.level_analytic(1):.5f}"), ("E₂ [eV]", f"{e[1]:.5f}", f"{m22.level_analytic(2):.5f}"), ("E₃ [eV]", f"{e[2]:.5f}", f"{m22.level_analytic(3):.5f}"),
                    ("Hα [nm]", f"{ln[0]:.4f}", f"{la[0]:.4f}"), ("Hβ [nm]", f"{ln[1]:.4f}", f"{la[1]:.4f}")]),
            final="Los autovalores numéricos reproducen los niveles de Bohr con error relativo ~10⁻⁴.", fsize=30, asize=30)


class RadialProbability(PhysicsScene):
    volume = 2
    chapter_number = 22
    chapter_title = CT

    def construct(self):
        _, r, u = m22.radial_numeric(1)
        rr = np.linspace(0, 6, 200)
        sel = np.arange(100, 1200, 50)  # r < 6 a0 (paso de malla 0.005 a0)
        curve_scene(
            self,
            found=[r"\psi_{1s} = \dfrac{1}{\sqrt{\pi a_0^3}}\,e^{-r/a_0}", r"P(r) = 4\pi r^2|\psi|^2 = \dfrac{4r^2}{a_0^3}\,e^{-2r/a_0},\qquad \langle r\rangle = \dfrac{3}{2}a_0"],
            note="La densidad de probabilidad es máxima en el núcleo, pero la probabilidad radial es máxima en r = a₀.",
            fcap="Densidad de probabilidad radial.",
            problem=["Para el estado fundamental del hidrógeno, halle el radio más probable y el radio medio.",
                     "Compare con la solución numérica de la ecuación radial."],
            pcap="Se usan unidades de a₀.",
            analytic=[(r"r_{mp} = a_0,\qquad \langle r\rangle = \tfrac32 a_0,\qquad \int_0^\infty P\,dr = 1", ACCENT)],
            acap="El radio medio supera al más probable por la cola exponencial.",
            numeric="Se usa u(r) del autovector fundamental: P = u²; ⟨r⟩ por Simpson y r_mp por ajuste parabólico del máximo.",
            xr=(0, 6), yr=(0, 0.6), xticks=(0, 2, 4, 6), yticks=(0, 0.2, 0.4, 0.6), xlabel="r / a₀", ylabel="P(r) a₀",
            curves=[(rr, m22.radial_probability_analytic_1s(rr), ANALYTIC)], dots=(r[sel], u[sel, 0] ** 2), legend=[(ANALYTIC, "4r² e^(−2r)"), (NUMERIC, "u(r)² numérico")],
            table=(["magnitud", "numérico", "analítico"], [("⟨r⟩ [a₀]", f"{m22.mean_radius_1s_numeric():.6f}", "1.500000"), ("r_mp [a₀]", f"{m22.most_probable_radius_numeric():.6f}", "1.000000")]),
            final="Radio medio y radio más probable coinciden con los valores analíticos.", fsize=32)
