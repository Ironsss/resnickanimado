"""Volumen 2, capitulo 9. Ley de Ampere y de Biot-Savart."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch09_biot_savart as m9


class BiotSavartLoop(PhysicsScene):
    volume = 2
    chapter_number = 9
    chapter_title = "Campo magnético de una corriente"

    def construct(self):
        z = np.linspace(-0.15, 0.15, 150)
        zs = np.linspace(-0.15, 0.15, 13)
        b0 = float(m9.loop_axis_analytic(0.0))
        curve_scene(
            self,
            found=[r"d\vec B = \dfrac{\mu_0 I}{4\pi}\dfrac{d\vec\ell\times\hat r}{r^2}", r"B_z(z) = \dfrac{\mu_0 I a^2}{2\,(a^2+z^2)^{3/2}}"],
            note="La ley de Biot-Savart permite construir el campo de cualquier distribución de corriente por superposición.",
            fcap="Ley de Biot-Savart.",
            problem=["Una espira circular de radio 5.0 cm conduce 3.0 A. Halle B sobre su eje.",
                     "Verifique además el campo de un hilo recto finito de 30 cm a 4.0 cm de su punto medio."],
            pcap="Se compara con la integral numérica sobre 720 segmentos.",
            analytic=[(rf"B(0) = \dfrac{{\mu_0 I}}{{2a}} = {b0 * 1e6:.4f}\ \mu\mathrm{{T}}", ACCENT),
                      rf"B_{{hilo}} = \dfrac{{\mu_0 I}}{{4\pi d}}\,\dfrac{{2\sin\theta}}{{1}} = {m9.wire_field_analytic() * 1e6:.4f}\ \mu\mathrm{{T}}"],
            acap="El hilo finito es menor que el infinito.",
            numeric="Se suma la ley de Biot-Savart sobre segmentos rectos (720 por espira, 2000 por hilo) con productos vectoriales.",
            xr=(-0.15, 0.15), yr=(0, 45), xticks=(-0.1, 0, 0.1), yticks=(0, 20, 40), xlabel="z [m]", ylabel="B [µT]",
            curves=[(z, m9.loop_axis_analytic(z) * 1e6, ANALYTIC)], dots=(zs, [m9.loop_axis_numeric(float(v)) * 1e6 for v in zs]),
            legend=[(ANALYTIC, "analítico"), (NUMERIC, "Biot-Savart")],
            table=(["magnitud", "numérico", "analítico"],
                   [("B(0) [µT]", f"{m9.loop_axis_numeric(0.0) * 1e6:.5f}", f"{b0 * 1e6:.5f}"),
                    ("B(0.05) [µT]", f"{m9.loop_axis_numeric(0.05) * 1e6:.5f}", f"{float(m9.loop_axis_analytic(0.05)) * 1e6:.5f}"),
                    ("hilo finito [µT]", f"{m9.wire_field_numeric() * 1e6:.5f}", f"{m9.wire_field_analytic() * 1e6:.5f}"),
                    ("hilo infinito [µT]", "n.a.", f"{m9.infinite_wire() * 1e6:.5f}")]),
            final="La suma numérica coincide con la forma cerrada en espira e hilo.", fsize=36)


class SolenoidAxis(PhysicsScene):
    volume = 2
    chapter_number = 9
    chapter_title = "Campo magnético de una corriente"

    def construct(self):
        z = np.linspace(-0.5, 0.5, 200)
        zs = np.linspace(-0.5, 0.5, 11)
        bi = m9.solenoid_center_infinite()
        be = float(m9.solenoid_axis_exact(0.0))
        curve_scene(
            self,
            found=[r"\oint \vec B\cdot d\vec\ell = \mu_0 I_{enc}\qquad\Rightarrow\qquad B = \mu_0 n I\ \ (\text{solenoide largo})",
                   r"B(z) = \dfrac{\mu_0 n I}{2}\left(\cos\alpha_2 - \cos\alpha_1\right)\ \ (\text{solenoide finito})"],
            note="Dentro de un solenoide largo el campo es uniforme; en los extremos cae a la mitad.",
            fcap="Ley de Ampère y solenoide.",
            problem=["Un solenoide de 500 vueltas, 50 cm de largo y 2.0 cm de radio conduce 2.0 A.",
                     "Halle B en el centro y su perfil sobre el eje; compare con el solenoide infinito."],
            pcap="El perfil se obtiene sumando las 500 espiras.",
            analytic=[(rf"B_\infty = \mu_0 nI = {bi * 1e3:.4f}\ \mathrm{{mT}}", ACCENT), rf"B(0) = {be * 1e3:.4f}\ \mathrm{{mT}},\qquad B(L/2) \approx {float(m9.solenoid_axis_exact(0.25)) * 1e3:.4f}\ \mathrm{{mT}}"],
            acap="La forma cerrada integra las espiras en el límite continuo.",
            numeric="Biot-Savart sobre 500 espiras de 60 segmentos cada una, evaluado en puntos del eje.",
            xr=(-0.5, 0.5), yr=(0, 3.0), xticks=(-0.5, 0, 0.5), yticks=(0, 1, 2, 3), xlabel="z [m]", ylabel="B [mT]",
            curves=[(z, m9.solenoid_axis_exact(z) * 1e3, ANALYTIC)], dots=(zs, [m9.solenoid_axis_numeric(float(v)) * 1e3 for v in zs]),
            legend=[(ANALYTIC, "forma cerrada"), (NUMERIC, "Biot-Savart")],
            table=(["magnitud", "numérico", "analítico"],
                   [("B(0) [mT]", f"{m9.solenoid_axis_numeric(0.0) * 1e3:.5f}", f"{be * 1e3:.5f}"),
                    ("B(L/2) [mT]", f"{m9.solenoid_axis_numeric(0.25) * 1e3:.5f}", f"{float(m9.solenoid_axis_exact(0.25)) * 1e3:.5f}"),
                    ("μ₀nI [mT]", "n.a.", f"{bi * 1e3:.5f}")]),
            final="El campo finito es 0.3 % menor que μ₀nI; en el extremo vale casi la mitad.", fsize=34)
