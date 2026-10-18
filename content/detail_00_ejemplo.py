"""Explicacion y pasos a mano. Ejemplo de referencia del formato: HelicalMotion.

Formato de cada entrada de DETAIL (clave = id de escena):
  explanation:    lista de parrafos que explican el tema (con matematica en linea $...$ si hace falta)
  analytic_steps: lista de (titulo, prosa, [ecuaciones TeX]) con TODAS las cifras sustituidas, como a mano
  numeric_steps:  igual, mostrando el algoritmo numerico con su primera iteracion explicita y la convergencia
Todos los numeros se calculan con las funciones de `physics/` (nunca se escriben a mano).
"""
from __future__ import annotations

import numpy as np

from physics import v2ch08_magnetic_field as m8
from content.texfmt import fx, sci
from physics.constants import E_CH, M_P


def _helical() -> dict:
    b, vp, vz = m8.B, m8.V_PERP, m8.V_PAR
    r, t0, p = m8.radius_analytic(), m8.period_analytic(), m8.pitch_analytic()
    w = E_CH * b / M_P
    n = 50
    dt = t0 / n
    th = w * dt / 2
    s = 2 * th / (1 + th**2)
    v0 = np.array([vp, 0.0, vz])
    vprime = v0 + np.cross(v0, [0, 0, th])
    v1 = v0 + np.cross(vprime, [0, 0, s])
    _, yb = m8.helix_boris()
    _, yr = m8.helix_rk4()
    return dict(
        explanation=[
            "Una carga en movimiento dentro de un campo magnético siente una fuerza perpendicular tanto a su velocidad como al campo. Como esa fuerza es siempre perpendicular al desplazamiento, no realiza trabajo: la rapidez no cambia, solo cambia la dirección de la velocidad.",
            "Si la velocidad es perpendicular a B, la trayectoria es una circunferencia. Si además hay una componente de velocidad paralela a B, esa componente no siente fuerza y se mantiene constante: la suma de ambos movimientos es una hélice.",
            "Lo importante es que el periodo del giro no depende de la rapidez: partículas más rápidas dan círculos más grandes pero tardan lo mismo en cada vuelta. Esa propiedad es la base del ciclotrón.",
        ],
        analytic_steps=[
            ("Descomponer la velocidad respecto del campo", "Se separa $\\vec v$ en la parte perpendicular y la paralela a $\\vec B = B\\hat z$. Solo la perpendicular produce fuerza.",
             [rf"v_\perp = {sci(vp,4)}\ \mathrm{{m/s}},\qquad v_\parallel = {sci(vz,4)}\ \mathrm{{m/s}},\qquad B = {b}\ \mathrm{{T}}"]),
            ("Igualar la fuerza magnética con la fuerza centrípeta", "En el plano perpendicular la única fuerza es la magnética, y debe producir la aceleración centrípeta del movimiento circular.",
             [r"q v_\perp B = \dfrac{m v_\perp^2}{r}"]),
            ("Despejar el radio", "Se cancela una $v_\\perp$ y se despeja $r$. Se sustituyen $m = 1.6726\\times10^{-27}$ kg y $q = 1.6022\\times10^{-19}$ C.",
             [r"r = \dfrac{m v_\perp}{qB}", rf"r = \dfrac{{({sci(M_P,5)})({sci(vp,4)})}}{{({sci(E_CH,5)})({b})}} = {r:.5f}\ \mathrm{{m}} = {r * 100:.4f}\ \mathrm{{cm}}"]),
            ("Obtener el periodo", "El periodo es la longitud de la circunferencia entre la rapidez. Al simplificar, $v_\\perp$ desaparece.",
             [r"T = \dfrac{2\pi r}{v_\perp} = \dfrac{2\pi m}{qB}", rf"T = \dfrac{{2\pi ({sci(M_P,5)})}}{{({sci(E_CH,5)})({b})}} = {t0 * 1e6:.5f}\ \mu\mathrm{{s}}"]),
            ("Calcular el paso de la hélice", "En un periodo la partícula avanza a lo largo de $z$ con velocidad constante $v_\\parallel$.",
             [rf"p = v_\parallel T = ({sci(vz,4)})({sci(t0,5)}) = {p:.5f}\ \mathrm{{m}} = {p * 100:.4f}\ \mathrm{{cm}}"]),
        ],
        numeric_steps=[
            ("Elegir el método y el paso", "Se integra $m\\,d\\vec v/dt = q\\vec v\\times\\vec B$ con el método de Boris, que gira la velocidad en lugar de aproximar la derivada y por eso conserva la rapidez. Se usan 50 pasos por vuelta.",
             [rf"\Delta t = T/50 = {sci(dt,4)}\ \mathrm{{s}},\qquad \omega = qB/m = {sci(w,5)}\ \mathrm{{rad/s}}"]),
            ("Preparar los vectores de giro", "Boris usa un vector $\\vec t$ (medio ángulo girado en un paso) y otro $\\vec s$ (corrección para que el giro sea exacto en módulo).",
             [rf"t = \omega\,\dfrac{{\Delta t}}{{2}} = {th:.6f},\qquad s = \dfrac{{2t}}{{1+t^2}} = {s:.6f}"]),
            ("Primer paso a mano", "Con $\\vec v_0 = (3\\times10^5,\\,0,\\,10^5)$ m/s se calcula primero $\\vec v' = \\vec v_0 + \\vec v_0\\times\\vec t$ y luego $\\vec v_1 = \\vec v_0 + \\vec v'\\times\\vec s$ (con $\\vec t = t\\hat z$, $\\vec s = s\\hat z$).",
             [rf"\vec v' = ({v0[0]:.1f},\,0,\,{v0[2]:.1f}) + ({sci(vp,2)}\hat x\times t\hat z) = ({vprime[0]:.1f},\,{vprime[1]:.1f},\,{vprime[2]:.1f})",
              rf"\vec v_1 = ({v1[0]:.2f},\,{v1[1]:.2f},\,{v1[2]:.1f})\ \mathrm{{m/s}},\qquad |\vec v_1| = {np.linalg.norm(v1):.4f}\ \mathrm{{m/s}} = |\vec v_0|"]),
            ("Avanzar la posición y repetir", "Se actualiza $\\vec x_{k+1} = \\vec x_k + \\vec v_{k+1}\\Delta t$ y se repite el mismo giro 200 veces (4 vueltas). Cada paso conserva el módulo de la velocidad.",
             [rf"\vec x_1 = \vec v_1\Delta t = ({sci(v1[0]*dt,4)},\,{sci(v1[1]*dt,4)},\,{sci(v1[2]*dt,4)})\ \mathrm{{m}}"]),
            ("Medir radio y periodo en la trayectoria", "El radio es la mitad del rango en $y$ y el periodo es el tiempo entre dos cruces ascendentes por el valor medio, con interpolación lineal.",
             [rf"r_{{num}} = {m8.measured_radius(yb) * 100:.5f}\ \mathrm{{cm}}\ \ (\text{{analítico }}{r * 100:.5f})"]),
            ("Comparar con Runge-Kutta", "RK4 aproxima la derivada y la rapidez se degrada lentamente; Boris la conserva a precisión de máquina.",
             [rf"\dfrac{{\Delta|v|}}{{|v|}}_{{RK4}} = {sci(m8.speed_drift(yr),2)},\qquad \dfrac{{\Delta|v|}}{{|v|}}_{{Boris}} = {sci(m8.speed_drift(yb),2)}"]),
        ],
    )


DETAIL = {"HelicalMotion": _helical()}
