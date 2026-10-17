"""Volumen 2, capitulo 17. Interferencia."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch17_interference as m17


class YoungSlits(PhysicsScene):
    volume = 2
    chapter_number = 17
    chapter_title = "Interferencia"

    def construct(self):
        y = np.linspace(-0.009, 0.009, 400)
        yn = np.linspace(-0.009, 0.009, 61)
        sp = m17.LAM * m17.L_SCR / m17.D_SLIT
        curve_scene(
            self,
            found=[r"\delta = d\sin\theta,\qquad \text{máximos: } \delta = m\lambda,\quad \text{mínimos: } \delta = (m+\tfrac12)\lambda", r"I(\theta) = 4I_0\cos^2\!\left(\dfrac{\pi d\sin\theta}{\lambda}\right)"],
            note="La diferencia de camino entre las dos rendijas determina si las ondas se refuerzan o se anulan.",
            fcap="Experimento de Young.",
            problem=["Dos rendijas separadas 0.25 mm se iluminan con luz de 600 nm y se observan en una pantalla a 1.5 m.",
                     "Halle la posición de las franjas brillantes y el espaciado entre ellas."],
            pcap="La amplitud numérica suma las ondas esféricas exactas.",
            analytic=[(rf"y_m = L\tan\!\left(\arcsin\dfrac{{m\lambda}}{{d}}\right),\qquad \Delta y \approx \dfrac{{\lambda L}}{{d}} = {sp * 1e3:.4f}\ \mathrm{{mm}}", ACCENT),
                      rf"y_3 = {m17.fringe_position_analytic(3) * 1e3:.5f}\ \mathrm{{mm}}"],
            acap="Aproximación de ángulos pequeños: franjas equiespaciadas.",
            numeric="Suma compleja de dos ondas esféricas con las distancias exactas a cada rendija; las posiciones de las franjas por bisección sobre r₂ − r₁ = mλ.",
            xr=(-9, 9), yr=(0, 1.1), xticks=(-8, -4, 0, 4, 8), yticks=(0, 0.5, 1.0), xlabel="y [mm]", ylabel="I / 4I₀",
            curves=[(y * 1e3, m17.intensity_analytic(y), ANALYTIC)], dots=(yn * 1e3, m17.intensity_numeric(yn)), legend=[(ANALYTIC, "cos²(πd sen θ/λ)"), (NUMERIC, "suma de ondas esféricas")],
            table=(["magnitud", "numérico", "analítico"],
                   [(f"y_{m} [mm]", f"{m17.bright_fringe_numeric(m) * 1e3:.6f}", f"{m17.fringe_position_analytic(m) * 1e3:.6f}") for m in (1, 3, 5)]),
            final="Las franjas brillantes numéricas coinciden con las de la fórmula exacta.", fsize=34)


class ThinFilmCoating(PhysicsScene):
    volume = 2
    chapter_number = 17
    chapter_title = "Interferencia"

    def construct(self):
        lam = np.linspace(400e-9, 700e-9, 200)
        ln = np.linspace(400e-9, 700e-9, 25)
        r0 = m17.bare_reflectance()
        curve_scene(
            self,
            found=[r"2nt = (m+\tfrac12)\lambda\ \ (\text{refuerzo})\qquad\text{con dos inversiones de fase o ninguna: } 2nt = m\lambda\ (\text{refuerzo})",
                   r"R = \left|\dfrac{r_{01} + r_{12}e^{2i\delta}}{1 + r_{01}r_{12}e^{2i\delta}}\right|^2,\qquad \delta = \dfrac{2\pi nt}{\lambda}"],
            note="Una capa de espesor λ/4n anula la reflexión cuando n² = n₀ n_s; con MgF₂ la reducción es parcial.",
            fcap="Películas delgadas y recubrimientos antirreflejo.",
            problem=["Una película de MgF₂ (n = 1.38) de espesor λ₀/4n con λ₀ = 550 nm recubre vidrio (n = 1.52).",
                     "Halle la reflectancia en función de la longitud de onda y compárela con la del vidrio desnudo."],
            pcap="Incidencia normal desde el aire.",
            analytic=[(rf"R_{{vidrio}} = \left(\dfrac{{n_s-1}}{{n_s+1}}\right)^2 = {r0 * 100:.4f}\,\%", ACCENT), rf"R(550\ \mathrm{{nm}}) = {float(m17.reflectance_analytic(550e-9)) * 100:.4f}\,\%"],
            acap="Mínimo de reflexión en λ₀ y aumento hacia los extremos del visible.",
            numeric="Método de la matriz de transferencia: producto de la matriz característica de la capa y evaluación del coeficiente de reflexión.",
            xr=(400, 700), yr=(0, 6), xticks=(400, 500, 600, 700), yticks=(0, 2, 4, 6), xlabel="λ [nm]", ylabel="R [%]",
            curves=[(lam * 1e9, m17.reflectance_analytic(lam) * 100, ANALYTIC)], dots=(ln * 1e9, m17.reflectance_matrix(ln) * 100), legend=[(ANALYTIC, "fórmula de Airy"), (NUMERIC, "matriz de transferencia")],
            table=(["magnitud", "numérico", "analítico"],
                   [("R(450) [%]", f"{float(m17.reflectance_matrix(450e-9)[0]) * 100:.6f}", f"{float(m17.reflectance_analytic(450e-9)) * 100:.6f}"),
                    ("R(550) [%]", f"{float(m17.reflectance_matrix(550e-9)[0]) * 100:.6f}", f"{float(m17.reflectance_analytic(550e-9)) * 100:.6f}"),
                    ("R(650) [%]", f"{float(m17.reflectance_matrix(650e-9)[0]) * 100:.6f}", f"{float(m17.reflectance_analytic(650e-9)) * 100:.6f}")]),
            final="Las dos formulaciones coinciden; la película reduce R de 4.3 % a 1.3 % en 550 nm.", fsize=30)
