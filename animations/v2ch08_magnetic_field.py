"""Volumen 2, capitulo 8. El campo magnetico."""
from __future__ import annotations

import numpy as np

from animations.common import ACCENT as ACCENT_, ANALYTIC, NUMERIC, PhysicsScene, curve_scene
from physics import v2ch08_magnetic_field as m8


class HelicalMotion(PhysicsScene):
    volume = 2
    chapter_number = 8
    chapter_title = "El campo magnético"

    def construct(self):
        r, T0 = m8.radius_analytic(), m8.period_analytic()
        tb, yb = m8.helix_boris()
        tr, yr = m8.helix_rk4()
        th = np.linspace(0, 2 * np.pi, 200)
        curve_scene(
            self,
            found=[r"\vec F = q\,\vec v\times\vec B,\qquad m\dfrac{v_\perp^2}{r} = q v_\perp B",
                   r"r = \dfrac{m v_\perp}{qB},\qquad T = \dfrac{2\pi m}{qB},\qquad p = v_\parallel T"],
            note="La fuerza magnética no realiza trabajo: la rapidez se conserva y la trayectoria es una hélice.",
            fcap="Fuerza de Lorentz magnética.",
            problem=["Un protón entra en B = 0.10 T con v⊥ = 3.0×10⁵ m/s y v∥ = 1.0×10⁵ m/s.",
                     "Halle el radio, el periodo y el paso de la hélice, y compruebe la conservación de la rapidez."],
            pcap="El campo apunta en +z.",
            analytic=[(rf"r = {r * 100:.4f}\ \mathrm{{cm}},\qquad T = {T0 * 1e6:.4f}\ \mu\mathrm{{s}}", ACCENT_),
                      rf"p = v_\parallel T = {m8.pitch_analytic() * 100:.4f}\ \mathrm{{cm}}"],
            acap="El periodo no depende de la rapidez.",
            numeric="Se integra m dv/dt = q v×B con RK4 y con el integrador de Boris (50 pasos por vuelta, 4 vueltas).",
            xr=(-0.07, 0.07), yr=(-0.075, 0.005), xticks=(-0.06, 0, 0.06), yticks=(-0.06, -0.03, 0), xlabel="x [m]", ylabel="y [m]",
            curves=[(r * np.sin(th), -r + r * np.cos(th), ANALYTIC)], dots=(yb[::2, 0], yb[::2, 1]), legend=[(ANALYTIC, "círculo analítico"), (NUMERIC, "Boris")],
            table=(["magnitud", "numérico", "analítico"],
                   [("r [cm]", f"{m8.measured_radius(yb) * 100:.5f}", f"{r * 100:.5f}"), ("T [µs]", f"{m8.measured_period(tb, yb) * 1e6:.5f}", f"{T0 * 1e6:.5f}"),
                    ("deriva |v| RK4", f"{m8.speed_drift(yr):.1e}", "0"), ("deriva |v| Boris", f"{m8.speed_drift(yb):.1e}", "0")]),
            final="Boris conserva la rapidez al redondeo; RK4 la degrada lentamente.", fsize=34)


class HallEffect(PhysicsScene):
    volume = 2
    chapter_number = 8
    chapter_title = "El campo magnético"

    def construct(self):
        va, vn = m8.hall_voltage_analytic(), m8.hall_voltage_numeric()
        bs = np.linspace(0, 1.0, 50)
        curve_scene(
            self,
            found=[r"q E_H = q v_d B,\qquad v_d = \dfrac{I}{n q w t}", r"V_H = E_H w = \dfrac{I B}{n q t},\qquad n = \dfrac{I B}{q t V_H}"],
            note="El campo de Hall equilibra la fuerza magnética sobre los portadores en régimen estacionario.",
            fcap="Efecto Hall.",
            problem=["Una cinta de cobre (n = 8.5×10²⁸ m⁻³, espesor 0.10 mm, ancho 10 mm) conduce 5.0 A en B = 0.5 T.",
                     "Halle la tensión de Hall y recupere la densidad de portadores."],
            pcap="El campo es perpendicular a la cinta.",
            analytic=[(rf"V_H = \dfrac{{IB}}{{nqt}} = {va * 1e6:.4f}\ \mu\mathrm{{V}}", ACCENT_), rf"n = \dfrac{{IB}}{{qtV_H}} = {m8.carrier_density_from_hall(va):.4e}\ \mathrm{{m^{{-3}}}}"],
            acap="La tensión de Hall es lineal en B.",
            numeric="Se integra el modelo de Drude (RK4) y se busca por bisección el campo transversal que anula la velocidad transversal.",
            xr=(0, 1.0), yr=(0, 12), xticks=(0, 0.5, 1.0), yticks=(0, 4, 8, 12), xlabel="B [T]", ylabel="V_H [µV]",
            curves=[(bs, m8.hall_voltage_analytic(b=bs) * 1e6, ANALYTIC)],
            dots=(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([m8.hall_voltage_numeric(b=b) * 1e6 for b in (0.1, 0.3, 0.5, 0.7, 0.9)])),
            legend=[(ANALYTIC, "IB/(nqt)"), (NUMERIC, "Drude + bisección")],
            table=(["magnitud", "numérico", "analítico"],
                   [("V_H [µV]", f"{vn * 1e6:.5f}", f"{va * 1e6:.5f}"), ("n [10²⁸ m⁻³]", f"{m8.carrier_density_from_hall(vn) / 1e28:.5f}", f"{m8.N_CU / 1e28:.5f}")]),
            final="El modelo microscópico reproduce V_H = IB/(nqt).", fsize=34)

