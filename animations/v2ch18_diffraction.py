"""Volumen 2, capitulo 18. Difraccion y polarizacion."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, ERROR, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch18_diffraction as m18

CT = "Difracción y polarización"


class SlitAndGrating(PhysicsScene):
    volume = 2
    chapter_number = 18
    chapter_title = CT

    def construct(self):
        s = np.linspace(0, 0.3, 600)
        sn = np.linspace(0, 0.3, 90)
        ms, ang = m18.principal_maxima_analytic()
        num = m18.principal_maxima_numeric()
        curve_scene(
            self,
            found=[r"I(\theta) = I_0\,\mathrm{sinc}^2\!\left(\dfrac{\pi a\sin\theta}{\lambda}\right)\ (\text{rendija}),\qquad a\sin\theta = m\lambda\ (\text{mínimos})",
                   r"I = I_0\,\mathrm{sinc}^2\beta\left(\dfrac{\sin N\gamma}{N\sin\gamma}\right)^2,\qquad d\sin\theta = m\lambda\ (\text{máximos principales})"],
            note="La envolvente de difracción de cada rendija modula el patrón de interferencia de las N rendijas.",
            fcap="Difracción de Fraunhofer y rejillas.",
            problem=["Una rejilla de 5 rendijas de ancho a = 2.0 µm y paso d = 6.0 µm se ilumina con luz de 550 nm.",
                     "Halle los ángulos de los máximos principales y el patrón de intensidad."],
            pcap="Se compara la fórmula con la suma coherente de fuentes puntuales.",
            analytic=[(rf"\sin\theta_m = \dfrac{{m\lambda}}{{d}}:\quad \theta_1 = {ang[1]:.4f}^\circ,\ \theta_2 = {ang[2]:.4f}^\circ,\ \theta_3 = {ang[3]:.4f}^\circ", ACCENT),
                      r"a\sin\theta = \lambda\ \Rightarrow\ \text{el orden } m = 3 \text{ queda suprimido por la envolvente}"],
            acap="d/a = 3: los órdenes múltiplos de 3 desaparecen.",
            numeric="Suma coherente de 5 rendijas, cada una con 200 fuentes puntuales; los máximos se localizan buscando extremos locales de la intensidad.",
            xr=(0, 0.3), yr=(0, 1.05), xticks=(0, 0.1, 0.2, 0.3), yticks=(0, 0.5, 1.0), xlabel="sen θ", ylabel="I / I₀",
            curves=[(s, m18.grating_analytic(s), ANALYTIC)], dots=(sn, m18.grating_numeric(sn)), legend=[(ANALYTIC, "fórmula analítica"), (NUMERIC, "suma coherente")],
            table=(["magnitud", "numérico", "analítico"], [(f"θ_{m} [°]", f"{num[m]:.4f}", f"{ang[m]:.4f}") for m in (1, 2, 4)]),
            final="Las posiciones de los máximos coinciden con d sen θ = mλ.", fsize=30)


class PolarizersMalus(PhysicsScene):
    volume = 2
    chapter_number = 18
    chapter_title = CT

    def construct(self):
        a = np.linspace(0, np.pi / 2, 200)
        an = np.linspace(0, np.pi / 2, 19)
        curve_scene(
            self,
            found=[r"I = I_0\cos^2\theta\qquad(\text{ley de Malus})", r"\vec E' = P(\theta)\vec E,\qquad P(\theta) = \begin{pmatrix}\cos^2\theta & \cos\theta\sin\theta\\ \cos\theta\sin\theta & \sin^2\theta\end{pmatrix}"],
            note="El cálculo de Jones describe la polarización completa; cada polarizador proyecta el campo sobre su eje.",
            fcap="Polarización y ley de Malus.",
            problem=["Luz linealmente polarizada pasa por dos polarizadores cruzados (0° y 90°), y entre ellos se intercala un tercero a un ángulo θ.",
                     "Halle la transmisión en función de θ y el ángulo de máxima transmisión."],
            pcap="La luz incidente ya está polarizada a 0°.",
            analytic=[(r"\dfrac{I}{I_0} = \cos^2\theta\,\sin^2\theta = \tfrac14\sin^2 2\theta", ACCENT), rf"\theta = 45^\circ:\ I/I_0 = {m18.three_polarizers_analytic(np.pi / 4):.4f}"],
            acap="Sin el polarizador intermedio, la transmisión es nula.",
            numeric="Producto de matrices de Jones de los tres polarizadores aplicado al vector de campo incidente.",
            xr=(0, 90), yr=(0, 0.3), xticks=(0, 30, 60, 90), yticks=(0, 0.1, 0.2, 0.3), xlabel="θ [°]", ylabel="I / I₀",
            curves=[(np.degrees(a), [m18.three_polarizers_analytic(float(x)) for x in a], ANALYTIC)], dots=(np.degrees(an), [m18.jones_intensity([0.0, float(x), np.pi / 2]) for x in an]),
            legend=[(ANALYTIC, "¼ sen² 2θ"), (NUMERIC, "matrices de Jones")],
            table=(["magnitud", "numérico", "analítico"],
                   [("I(30°)", f"{m18.jones_intensity([0, np.pi / 6, np.pi / 2]):.8f}", f"{m18.three_polarizers_analytic(np.pi / 6):.8f}"),
                    ("I(45°)", f"{m18.jones_intensity([0, np.pi / 4, np.pi / 2]):.8f}", f"{m18.three_polarizers_analytic(np.pi / 4):.8f}"),
                    ("Malus a 60°", f"{m18.jones_intensity([0, np.pi / 3]):.8f}", f"{float(m18.malus_analytic(np.pi / 3)):.8f}")]),
            final="La máxima transmisión es 25 % a 45°.", fsize=32)
