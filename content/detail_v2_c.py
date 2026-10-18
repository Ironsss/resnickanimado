"""Explicacion y pasos a mano: capitulos 9 a 13 del volumen 2.

Escenas: BiotSavartLoop, SolenoidAxis, RotatingLoop, SlidingRod, RLCircuit,
RLCOscillator, DipoleTorque, ParamagnetCurie, SeriesRLC, PowerTransformer.
Todas las cifras se calculan con las funciones de `physics/` o replicando su algoritmo.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch09_biot_savart as m9
from physics import v2ch10_faraday as m10
from physics import v2ch11_inductance as m11
from physics import v2ch12_magnetic_matter as m12
from physics import v2ch13_ac as m13
from physics.ch07_work_energy import simpson
from physics.constants import K_B, MU0
from physics.integrators import rk4_step


def _tab(header: list[str], rows: list[list[str]]) -> str:
    """Tabla TeX alineada a la derecha con encabezado."""
    cols = "r" * len(header)
    out = [r"\begin{array}{" + cols + "}", " & ".join(header) + r" \\ \hline"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out.append(r"\end{array}")
    return "\n".join(out)


def _rk4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return k1, k2, k3, k4, y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# =============================================================================== capitulo 9
def _loop() -> dict:
    a, cur, mu0 = m9.A_LOOP, m9.I, MU0
    k = mu0 * cur / (4 * np.pi)
    b0 = float(m9.loop_axis_analytic(0.0))
    b5 = float(m9.loop_axis_analytic(0.05))
    n0 = m9.loop_axis_numeric(0.0)
    n5 = m9.loop_axis_numeric(0.05)
    wl, wd = m9.LEN_WIRE, m9.DIST
    bw, bi, bwn = m9.wire_field_analytic(), m9.infinite_wire(), m9.wire_field_numeric()
    s = (wl / 2) / np.hypot(wl / 2, wd)
    # espira con 4 segmentos
    seg4 = m9.loop_segments(a, 4)
    p0, p1 = seg4[0]
    mid = 0.5 * (p0 + p1)
    dl = p1 - p0
    r = np.array([0.0, 0.0, 0.0]) - mid
    cr = np.cross(dl, r)
    d = np.linalg.norm(r)
    dbz = k * cr[2] / d**3
    b4 = m9.loop_axis_numeric(0.0, n=4)
    rows_n = []
    for n in (4, 8, 16, 720):
        v = m9.loop_axis_numeric(0.0, n=n)
        rows_n.append([str(n), sci(v * 1e6, 6), sci(abs(v - b0) / b0, 2)])
    # primer segmento del hilo
    n_w = 2000
    h = wl / n_w
    z0 = -wl / 2
    zm = z0 + h / 2
    rw = np.array([wd, 0.0, -zm])
    dlw = np.array([0.0, 0.0, h])
    crw = np.cross(dlw, rw)
    dw = np.linalg.norm(rw)
    dbw = k * np.linalg.norm(crw) / dw**3
    return dict(
        explanation=[
            "La ley de Biot-Savart es la receta para construir el campo magnético de cualquier corriente: se divide el conductor en pequeños elementos $d\\vec\\ell$, cada uno aporta un $d\\vec B$ proporcional a $I\\,d\\ell$ y a $1/r^2$, y se suman vectorialmente todas las contribuciones. Es el análogo magnético de sumar campos de cargas puntuales con la ley de Coulomb, con la diferencia de que aparece un producto vectorial.",
            "El producto vectorial $d\\vec\\ell\\times\\hat r$ fija la dirección: $d\\vec B$ es perpendicular tanto al elemento de corriente como al vector que va de él al punto de observación. Por eso el campo rodea al conductor (regla de la mano derecha) y un elemento no produce campo sobre su propia línea. Cuando el problema tiene simetría, muchas componentes se cancelan y solo sobrevive la del eje.",
            "En una espira circular, todos los elementos están a la misma distancia $\\sqrt{a^2+z^2}$ de un punto del eje y sus componentes radiales se anulan por pares; solo queda la componente axial, y la integral se reduce a multiplicar por la longitud $2\\pi a$. En el centro el campo vale $\\mu_0 I/2a$ y decae como $1/z^3$ lejos de la espira, que es el comportamiento de un dipolo magnético.",
            "Un hilo recto de longitud finita muestra la otra cara del método: el campo es el del hilo infinito $\\mu_0I/2\\pi d$ multiplicado por el factor geométrico $\\sin\\alpha$, que mide qué fracción del ángulo completo cubre el hilo visto desde el punto. Con los datos del problema el hilo de 30 cm cubre casi todo el ángulo, pero la pequeña diferencia con el hilo infinito es medible. El resultado tiene sentido físico solo para una corriente estacionaria; un segmento aislado de corriente no existe por sí solo, sino como parte de un circuito cerrado.",
        ],
        analytic_steps=[
            ("Plantear Biot-Savart para la espira", "Se toma un elemento $d\\vec\\ell$ de la espira de radio $a$ y se escribe su aporte en un punto del eje a altura $z$. Como $d\\vec\\ell\\perp\\hat r$, el producto vectorial tiene módulo $d\\ell$.",
             [r"dB = \dfrac{\mu_0 I}{4\pi}\dfrac{d\ell}{a^2+z^2}"]),
            ("Quedarse con la componente axial e integrar", "Por simetría las componentes perpendiculares al eje se cancelan; la axial lleva el factor $\\cos\\theta = a/\\sqrt{a^2+z^2}$. Integrar $d\\ell$ sobre toda la espira da $2\\pi a$.",
             [r"B_z = \dfrac{\mu_0 I}{4\pi}\dfrac{a}{(a^2+z^2)^{1/2}}\dfrac{2\pi a}{a^2+z^2} = \dfrac{\mu_0 I a^2}{2(a^2+z^2)^{3/2}}"]),
            ("Sustituir en el centro de la espira", "Con $z=0$ la expresión se simplifica a $\\mu_0 I/2a$. Se usan $\\mu_0 = 4\\pi\\times10^{-7}$ T m/A, $I = 3.0$ A y $a = 5.0$ cm.",
             [rf"B(0) = \dfrac{{\mu_0 I}}{{2a}} = \dfrac{{({sci(mu0, 5)})(3.0)}}{{2(0.050)}} = {sci(b0, 5)}\ \mathrm{{T}} = {b0 * 1e6:.4f}\ \mu\mathrm{{T}}"]),
            ("Evaluar a una distancia igual al radio", "A $z = 5.0$ cm el denominador crece como $(2a^2)^{3/2}$, de modo que el campo cae a $1/2^{3/2}$ del valor central.",
             [rf"B(5\,\mathrm{{cm}}) = \dfrac{{({sci(mu0, 5)})(3.0)(0.050)^2}}{{2\,(0.050^2 + 0.050^2)^{{3/2}}}} = {b5 * 1e6:.4f}\ \mu\mathrm{{T}}"]),
            ("Hilo finito: integrar sobre su longitud", "El punto está a $d = 4.0$ cm del hilo, frente a su punto medio. En cada elemento $|d\\vec\\ell\\times\\hat r| = dz\\,\\sin\\theta = dz\\,d/r$ con $r^2 = d^2+z^2$, y se integra de $-L/2$ a $L/2$.",
             [r"B = \dfrac{\mu_0 I}{4\pi}\int_{-L/2}^{L/2}\dfrac{d\,dz}{(d^2+z^2)^{3/2}} = \dfrac{\mu_0 I}{4\pi d}\left[\dfrac{z}{\sqrt{d^2+z^2}}\right]_{-L/2}^{L/2} = \dfrac{\mu_0 I}{2\pi d}\,\dfrac{L/2}{\sqrt{(L/2)^2+d^2}}"]),
            ("Sustituir los datos del hilo", "Primero el factor geométrico $\\sin\\alpha = (L/2)/\\sqrt{(L/2)^2+d^2}$ y después el campo.",
             [rf"\sin\alpha = \dfrac{{0.150}}{{\sqrt{{0.150^2 + 0.040^2}}}} = {s:.5f}",
              rf"B = \dfrac{{({sci(mu0, 5)})(3.0)}}{{2\pi(0.040)}}({s:.5f}) = {bw * 1e6:.4f}\ \mu\mathrm{{T}}"]),
            ("Comparar con el hilo infinito", "Si el hilo fuera infinito, $\\sin\\alpha = 1$. La diferencia relativa mide cuánto pesa truncar el hilo.",
             [rf"B_\infty = \dfrac{{\mu_0 I}}{{2\pi d}} = {bi * 1e6:.4f}\ \mu\mathrm{{T}},\qquad 1 - \dfrac{{B}}{{B_\infty}} = {(1 - bw / bi) * 100:.2f}\,\%"]),
        ],
        numeric_steps=[
            ("Discretizar la espira en segmentos rectos", "El código reemplaza la integral por una suma sobre $N$ cuerdas. Cada segmento aporta $\\Delta\\vec B = \\frac{\\mu_0 I}{4\\pi}\\frac{\\Delta\\vec\\ell\\times\\vec r}{r^3}$, con $\\vec r$ medido desde el punto medio de la cuerda hasta el punto de observación. El código real usa $N=720$; aquí se muestra primero con $N=4$.",
             [rf"\dfrac{{\mu_0 I}}{{4\pi}} = {sci(k, 5)}\ \mathrm{{T\,m/A\cdot A}},\qquad \Delta\varphi = 2\pi/N = 90^\circ"]),
            ("Primer segmento con N = 4", "Va de $(a,0,0)$ a $(0,a,0)$. Se calculan su vector $\\Delta\\vec\\ell$, su punto medio, el vector $\\vec r$ hacia el origen y el producto vectorial.",
             [rf"\Delta\vec\ell = ({dl[0]:.3f},\,{dl[1]:.3f},\,0)\ \mathrm{{m}},\qquad \vec r_{{mid}} = ({mid[0]:.3f},\,{mid[1]:.3f},\,0)\ \mathrm{{m}}",
              rf"\vec r = \vec p - \vec r_{{mid}} = ({r[0]:.3f},\,{r[1]:.3f},\,0)\ \mathrm{{m}},\qquad |\vec r| = {d:.5f}\ \mathrm{{m}}",
              rf"\Delta\vec\ell\times\vec r = (0,\,0,\,{cr[2]:.5f})\ \mathrm{{m^2}}"]),
            ("Contribución del primer segmento y suma", "Se sustituye en la fórmula; por simetría los otros tres segmentos aportan lo mismo en $z$ y sus componentes horizontales se anulan en la suma.",
             [rf"\Delta B_z = ({sci(k, 5)})\dfrac{{{cr[2]:.5f}}}{{({d:.5f})^3}} = {sci(dbz, 5)}\ \mathrm{{T}}",
              rf"B_z(N=4) = 4\,\Delta B_z = {sci(b4, 5)}\ \mathrm{{T}}\quad(\text{{analítico: }}{sci(b0, 5)}\ \mathrm{{T}})"]),
            ("Aumentar el número de segmentos", "Con pocas cuerdas el punto medio de cada una queda más cerca del punto que el arco real, y el campo se sobreestima. Al duplicar $N$ el error cae aproximadamente cuatro veces.",
             [_tab([r"N", r"B(0)\ [\mu\mathrm{T}]", r"\text{error relativo}"], rows_n)]),
            ("Resultado con N = 720 en dos puntos del eje", "Es el cálculo real del código: 720 productos vectoriales sumados en cada punto.",
             [rf"B_{{num}}(0) = {n0 * 1e6:.5f}\ \mu\mathrm{{T}}\ \ (\text{{analítico }}{b0 * 1e6:.5f}),\qquad B_{{num}}(5\,\mathrm{{cm}}) = {n5 * 1e6:.5f}\ \mu\mathrm{{T}}\ \ (\text{{analítico }}{b5 * 1e6:.5f})"]),
            ("Hilo finito: primer segmento de 2000", "El hilo se divide en 2000 tramos de longitud $h = L/2000$. El primero va de $z=-15$ cm a $z=-15$ cm $+h$; el producto vectorial apunta en $\\hat y$ con módulo $h\\,d$.",
             [rf"h = {sci(h, 4)}\ \mathrm{{m}},\quad z_{{mid}} = {zm:.6f}\ \mathrm{{m}},\quad r = \sqrt{{d^2+z_{{mid}}^2}} = {dw:.6f}\ \mathrm{{m}}",
              rf"\Delta B_1 = ({sci(k, 5)})\dfrac{{(h)(d)}}{{r^3}} = {sci(dbw, 5)}\ \mathrm{{T}}"]),
            ("Sumar los 2000 tramos y comparar", "Sumando los 2000 aportes (módulo del vector resultante) se obtiene el campo numérico del hilo.",
             [rf"B_{{num}} = \sum_{{j=1}}^{{2000}}\Delta B_j = {bwn * 1e6:.5f}\ \mu\mathrm{{T}},\qquad B_{{analit}} = {bw * 1e6:.5f}\ \mu\mathrm{{T}},\qquad \dfrac{{|B_{{num}}-B_{{analit}}|}}{{B_{{analit}}}} = {sci(abs(bwn - bw) / bw, 2)}"]),
        ],
    )


def _solenoid() -> dict:
    n, ln, r, cur = m9.N_TURNS, m9.L_SOL, m9.R_SOL, m9.I_SOL
    mu0 = MU0
    nn = n / ln
    binf = m9.solenoid_center_infinite()
    z1, z2 = ln / 2, -ln / 2
    c1, c2 = z1 / np.hypot(z1, r), z2 / np.hypot(z2, r)
    bc = float(m9.solenoid_axis_exact(0.0))
    be = float(m9.solenoid_axis_exact(0.25))
    zs = np.linspace(-ln / 2 + ln / (2 * n), ln / 2 - ln / (2 * n), n)
    dz = zs[1] - zs[0]
    nc, ne = m9.solenoid_axis_numeric(0.0), m9.solenoid_axis_numeric(0.25)
    # primer anillo, 60 segmentos
    ring = m9.loop_segments(r, 60, zs[0])
    b_ring = float(m9.biot_savart(np.array([[0.0, 0.0, 0.0]]), ring, cur)[0, 2])
    b_ring_an = float(m9.loop_axis_analytic(zs[0], r, cur))
    sg = ring[0]
    mid = 0.5 * (sg[0] + sg[1])
    dl = sg[1] - sg[0]
    rr = -mid
    cr = np.cross(dl, rr)
    dist = np.linalg.norm(rr)
    k = mu0 * cur / (4 * np.pi)
    dbz = k * cr[2] / dist**3
    # suma analitica de anillos (ideal) para separar errores
    sum_rings = float(np.sum(m9.loop_axis_analytic(zs, r, cur))) * 1.0
    return dict(
        explanation=[
            "Un solenoide es un alambre enrollado en hélice apretada. Dentro de él los campos de las espiras vecinas se refuerzan y fuera se cancelan casi por completo, así que el campo interior es casi uniforme y paralelo al eje. La ley de Ampère permite obtener ese valor sin integrar: con un circuito rectangular que cruza la pared del solenoide se encuentra $B=\\mu_0 n I$, donde $n$ es el número de vueltas por unidad de longitud.",
            "Esa fórmula supone un solenoide infinitamente largo. En uno real el campo se debilita hacia los extremos, porque faltan espiras a un lado del punto: justo en la boca del solenoide el campo cae aproximadamente a la mitad del valor central. Esto es útil para estimar cuánto se equivocaría una aproximación de solenoide largo.",
            "Para el solenoide finito sobre su eje se suman los campos de las espiras individuales: cada una aporta lo que da la fórmula de la espira, y al convertir la suma en una integral queda $B=\\tfrac12\\mu_0nI(\\cos\\alpha_2-\\cos\\alpha_1)$, donde $\\alpha_{1,2}$ son los ángulos bajo los que se ven los extremos desde el punto. Si el solenoide es largo comparado con su radio, ambos cosenos se acercan a $\\pm1$ y se recupera el caso infinito.",
            "El orden de magnitud es el de los electroimanes de laboratorio: unos pocos mT para corrientes de amperios y mil vueltas por metro. Para campos grandes hace falta mucha corriente o un núcleo ferromagnético, cuyo efecto se estudia en el capítulo de materia magnética.",
        ],
        analytic_steps=[
            ("Densidad de vueltas", "Se necesita el número de vueltas por metro, que es la única cantidad geométrica del solenoide largo.",
             [rf"n = \dfrac{{N}}{{\ell}} = \dfrac{{{n}}}{{{ln}\ \mathrm{{m}}}} = {nn:.1f}\ \mathrm{{m^{{-1}}}}"]),
            ("Ley de Ampère para el solenoide infinito", "Un lazo rectangular con un lado dentro (campo $B$) y el otro fuera (campo nulo) encierra $nL$ vueltas por cada longitud $L$ del lado interior; solo el lado interior contribuye a la circulación.",
             [r"BL = \mu_0 (nL) I \ \Rightarrow\ B_\infty = \mu_0 n I",
              rf"B_\infty = ({sci(mu0, 5)})({nn:.1f})(2.0) = {binf * 1e3:.4f}\ \mathrm{{mT}}"]),
            ("Campo de una espira del solenoide finito", "Cada vuelta aporta la fórmula del eje de una espira. Un trozo $dz'$ contiene $n\\,dz'$ vueltas, y se integra a lo largo de todo el solenoide.",
             [r"dB = \dfrac{\mu_0 (n\,dz') I R^2}{2\,(R^2+(z-z')^2)^{3/2}}"]),
            ("Resolver la integral", "Con el cambio $u = z'-z$ la integral es elemental y da la diferencia de cosenos de los ángulos de los extremos.",
             [r"B(z) = \dfrac{\mu_0 n I}{2}\left[\dfrac{z+\ell/2}{\sqrt{R^2+(z+\ell/2)^2}} - \dfrac{z-\ell/2}{\sqrt{R^2+(z-\ell/2)^2}}\right]"]),
            ("Evaluar en el centro", "En $z=0$ los dos extremos están a $\\pm 25$ cm y el radio es $R = 2.0$ cm, de modo que los cosenos valen casi 1.",
             [rf"\cos\alpha_2-\cos\alpha_1 = 2\,\dfrac{{0.250}}{{\sqrt{{0.020^2+0.250^2}}}} = {c1 - c2:.6f}",
              rf"B(0) = \dfrac{{({sci(mu0, 5)})({nn:.1f})(2.0)}}{{2}}({c1 - c2:.6f}) = {bc * 1e3:.4f}\ \mathrm{{mT}}"]),
            ("Evaluar en el extremo", "En $z = \\ell/2$ solo queda el término del extremo lejano (el cercano vale $z-\\ell/2=0$), y el campo es casi la mitad del central.",
             [rf"B(\ell/2) = \dfrac{{({sci(mu0, 5)})({nn:.1f})(2.0)}}{{2}}\dfrac{{0.500}}{{\sqrt{{0.020^2+0.500^2}}}} = {be * 1e3:.4f}\ \mathrm{{mT}}"]),
            ("Comparar con el límite infinito", "El cociente indica qué tan buena es la aproximación del solenoide largo en el centro.",
             [rf"\dfrac{{B(0)}}{{B_\infty}} = {bc / binf:.5f},\qquad \dfrac{{B(\ell/2)}}{{B_\infty}} = {be / binf:.5f}"]),
        ],
        numeric_steps=[
            ("Modelo discreto del solenoide", "El código no usa la integral cerrada: coloca $N = 500$ espiras circulares separadas $\\Delta z = \\ell/N$, cada una dividida en 60 segmentos rectos, y suma Biot-Savart sobre los $30\\,000$ segmentos. Las espiras se ubican en el centro de cada celda.",
             [rf"\Delta z = \dfrac{{\ell}}{{N}} = {dz * 1e3:.3f}\ \mathrm{{mm}},\qquad z_1 = -\dfrac{{\ell}}{{2}}+\dfrac{{\Delta z}}{{2}} = {zs[0]:.4f}\ \mathrm{{m}},\ \ldots,\ z_{{500}} = {zs[-1]:.4f}\ \mathrm{{m}}"]),
            ("Primer segmento de la primera espira", "Se calcula con el mismo procedimiento que la espira simple. Este segmento va del ángulo $0^\\circ$ al $6^\\circ$ sobre la circunferencia de radio $R$, ubicada en $z_1$; el punto de observación es el origen (centro del solenoide).",
             [rf"\Delta\vec\ell = ({sci(dl[0], 4)},\ {sci(dl[1], 4)},\ 0)\ \mathrm{{m}},\qquad \vec r = ({rr[0]:.5f},\ {rr[1]:.5f},\ {rr[2]:.5f})\ \mathrm{{m}},\quad |\vec r| = {dist:.6f}\ \mathrm{{m}}",
              rf"(\Delta\vec\ell\times\vec r)_z = {sci(cr[2], 5)}\ \mathrm{{m^2}}"]),
            ("Aporte axial de ese segmento", "Con $\\mu_0 I/4\\pi$ para $I = 2.0$ A, la componente $z$ del primer segmento es pequeña porque la espira está lejos del origen; las 60 contribuciones de la espira se suman.",
             [rf"\Delta B_z = \dfrac{{\mu_0 I}}{{4\pi}}\dfrac{{(\Delta\vec\ell\times\vec r)_z}}{{r^3}} = {sci(dbz, 5)}\ \mathrm{{T}}"]),
            ("Campo de la espira completa con 60 segmentos", "Se suman los 60 aportes y se compara con la fórmula exacta de una espira a la distancia $z_1$. La diferencia es el error de discretizar con cuerdas.",
             [rf"B_{{espira}}^{{num}} = {sci(b_ring, 6)}\ \mathrm{{T}},\qquad B_{{espira}}^{{exacto}} = \dfrac{{\mu_0 I R^2}}{{2(R^2+z_1^2)^{{3/2}}}} = {sci(b_ring_an, 6)}\ \mathrm{{T}}"]),
            ("Sumar las 500 espiras en el centro", "Se repite lo anterior para cada $z_j$ y se suman los 500 campos de espira.",
             [rf"B_{{num}}(0) = \sum_{{j=1}}^{{500}} B_j = {nc * 1e3:.5f}\ \mathrm{{mT}}"]),
            ("Contrastar con el resultado analítico", "La suma de las 500 espiras ideales (sin discretizar el círculo) da el valor exacto del modelo de espiras; la parte restante de la diferencia viene de las 60 cuerdas por espira.",
             [rf"\sum_j B^{{exacto}}_j = {sum_rings * 1e3:.5f}\ \mathrm{{mT}},\qquad B_{{analit}}(0) = {bc * 1e3:.5f}\ \mathrm{{mT}}",
              rf"\text{{error relativo}} = {sci(abs(nc - bc) / bc, 2)}"]),
            ("Evaluar en el extremo y concluir", "En $z = 0.25$ m el mismo algoritmo reproduce el valor exacto con un error del mismo orden, dominado por las cuerdas.",
             [rf"B_{{num}}(\ell/2) = {ne * 1e3:.5f}\ \mathrm{{mT}},\qquad B_{{analit}}(\ell/2) = {be * 1e3:.5f}\ \mathrm{{mT}},\qquad \text{{error}} = {sci(abs(ne - be) / be, 2)}"]),
        ],
    )


# =============================================================================== capitulo 10
def _rotating() -> dict:
    nc, a, b0, f = m10.N_COIL, m10.AREA, m10.B0, m10.FREQ
    w = m10.OMEGA
    phi0 = nc * b0 * a
    e0 = m10.emf_peak_analytic()
    erms = e0 / np.sqrt(2)
    t = 0.005
    h = 1e-6
    fp, fm = float(m10.flux(t + h)), float(m10.flux(t - h))
    en = float(m10.emf_numeric(np.array([t]))[0])
    ea = float(m10.emf_analytic(t))
    tt = np.linspace(0, 1 / f, 9)
    y8 = m10.emf_analytic(tt) ** 2
    s8 = simpson(y8, tt)
    ym = m10.emf_analytic(np.linspace(0, 1 / f, 4001)) ** 2
    rms_num = m10.emf_rms_numeric()
    hh = (1 / f) / 8
    return dict(
        explanation=[
            "Cuando cambia el flujo magnético que atraviesa un circuito aparece en él una fem inducida igual a menos la rapidez de cambio del flujo. El signo negativo (ley de Lenz) dice que la corriente inducida se opone a la variación que la produce. En un generador no se mueve el campo ni el circuito se deforma: lo que cambia es el ángulo entre el campo y la normal de la bobina.",
            "Si la bobina de $N$ vueltas y área $A$ gira con velocidad angular $\\omega$ en un campo uniforme $B$, el flujo es $\\Phi_B = NBA\\cos\\omega t$ y la fem es su derivada cambiada de signo, una senoide de amplitud $NBA\\omega$. La fem es máxima cuando el plano de la bobina es paralelo al campo, justo cuando el flujo pasa por cero pero cambia más rápido.",
            "Una fem alterna se caracteriza por su valor eficaz: el de una tensión continua que disiparía la misma potencia en una resistencia. Para una senoide es el máximo dividido por $\\sqrt 2$. Es el valor que indican los voltímetros de corriente alterna y el que aparece en las especificaciones de la red eléctrica.",
            "Con los datos del problema (100 vueltas, $0.020\\ \\mathrm{m^2}$, 0.50 T, 50 Hz) la fem máxima es de unos 300 V. Aumentarla exige más vueltas, más área, más campo o más frecuencia; esas cuatro cantidades son los parámetros de diseño de cualquier alternador.",
        ],
        analytic_steps=[
            ("Flujo a través de la bobina", "El flujo por vuelta es $BA\\cos\\theta$ con $\\theta = \\omega t$ el ángulo entre el campo y la normal; con $N$ vueltas en serie el flujo se multiplica por $N$.",
             [rf"\Phi_B(t) = NBA\cos\omega t,\qquad NBA = (100)(0.50)(0.020) = {phi0:.3f}\ \mathrm{{Wb}}"]),
            ("Velocidad angular", "Se convierte la frecuencia en velocidad angular.",
             [rf"\omega = 2\pi f = 2\pi(50\ \mathrm{{Hz}}) = {w:.4f}\ \mathrm{{rad/s}}"]),
            ("Aplicar la ley de Faraday", "Se deriva el flujo respecto del tiempo y se cambia el signo.",
             [r"\varepsilon = -\dfrac{d\Phi_B}{dt} = NBA\,\omega\,\sin\omega t"]),
            ("Fem máxima", "El seno vale 1 cuando $\\omega t = \\pi/2$, es decir a los 5 ms.",
             [rf"\varepsilon_0 = NBA\omega = ({phi0:.3f}\ \mathrm{{Wb}})({w:.4f}\ \mathrm{{rad/s}}) = {e0:.4f}\ \mathrm{{V}}"]),
            ("Valor eficaz por definición", "Se promedia el cuadrado de la fem en un periodo y se extrae la raíz. El promedio de $\\sin^2$ es $1/2$.",
             [r"\varepsilon_{rms} = \sqrt{\dfrac1T\int_0^T\varepsilon_0^2\sin^2\omega t\,dt} = \dfrac{\varepsilon_0}{\sqrt2}"]),
            ("Sustituir", "Se divide la fem máxima entre $\\sqrt2$.",
             [rf"\varepsilon_{{rms}} = \dfrac{{{e0:.4f}\ \mathrm{{V}}}}{{\sqrt2}} = {erms:.4f}\ \mathrm{{V}}"]),
            ("Valor instantáneo a 5 ms", "Es el punto que el código compara con la derivada numérica del flujo.",
             [rf"\varepsilon(5\,\mathrm{{ms}}) = {e0:.4f}\,\sin\!\left({w:.4f}\cdot0.005\right) = {ea:.4f}\ \mathrm{{V}}"]),
        ],
        numeric_steps=[
            ("Idea del método", "La fem se obtiene derivando numéricamente el flujo con diferencias centrales, sin usar la fórmula de $\\varepsilon$. El valor eficaz se obtiene integrando $\\varepsilon^2$ con la regla de Simpson. El paso de derivación es $h = 1\\ \\mu\\mathrm{s}$.",
             [r"\varepsilon(t)\approx -\dfrac{\Phi(t+h)-\Phi(t-h)}{2h}"]),
            ("Evaluar el flujo en t ± h", "A $t = 5$ ms el flujo pasa por cero, así que los dos valores son pequeños y de signo opuesto.",
             [rf"\Phi(t+h) = {phi0:.3f}\cos\!\left({w:.4f}\cdot0.005001\right) = {sci(fp, 6)}\ \mathrm{{Wb}}",
              rf"\Phi(t-h) = {phi0:.3f}\cos\!\left({w:.4f}\cdot0.004999\right) = {sci(fm, 6)}\ \mathrm{{Wb}}"]),
            ("Diferencia central", "Se restan, se divide entre $2h = 2\\times10^{-6}$ s y se cambia el signo.",
             [rf"\varepsilon \approx -\dfrac{{({sci(fp, 6)}) - ({sci(fm, 6)})}}{{2\times10^{{-6}}}} = {en:.5f}\ \mathrm{{V}}\qquad(\text{{analítico }}{ea:.5f}\ \mathrm{{V}})"]),
            ("Regla de Simpson: ejemplo reducido con 8 subintervalos", "Para $\\int_0^T\\varepsilon^2dt$ se toman muestras equiespaciadas y la regla $\\frac h3[y_0+y_8+4(y_1+y_3+y_5+y_7)+2(y_2+y_4+y_6)]$ con $h = T/8$. Aquí $y_j = \\varepsilon(t_j)^2$.",
             [rf"h = \dfrac{{T}}{{8}} = {sci(hh, 4)}\ \mathrm{{s}}",
              rf"y_j = ({', '.join(f'{v:.1f}' for v in y8)})\ \mathrm{{V^2}}"]),
            ("Resultado con 8 subintervalos", "Aplicada a un periodo completo de una función senoidal, Simpson ya es exacta con 8 puntos: la integral vale $\\varepsilon_0^2T/2$.",
             [rf"\int_0^T\varepsilon^2dt \approx {s8:.5f}\ \mathrm{{V^2 s}}\quad\Rightarrow\quad \varepsilon_{{rms}}\approx\sqrt{{{s8:.5f}/T}} = {np.sqrt(s8 * f):.5f}\ \mathrm{{V}}"]),
            ("Resultado con el número de intervalos real", "El código usa 4000 subintervalos de longitud $T/4000 = 5\\ \\mu\\mathrm{s}$ y obtiene el valor eficaz.",
             [rf"\varepsilon_{{rms}}^{{num}} = {rms_num:.5f}\ \mathrm{{V}},\qquad \varepsilon_{{rms}}^{{analit}} = {erms:.5f}\ \mathrm{{V}}",
              rf"\text{{error relativo}} = {sci(abs(rms_num - erms) / erms, 2)}"]),
        ],
    )


def _sliding() -> dict:
    b, l_, m, rc, v0 = m10.B_ROD, m10.L_ROD, m10.M_ROD, m10.R_CIRC, m10.V0_ROD
    tau = m10.tau_rod()
    kk = b**2 * l_**2 / (m * rc)
    xinf = m10.rod_distance_analytic()
    e0j = 0.5 * m * v0**2
    emf0 = b * l_ * v0
    i0 = emf0 / rc
    f0 = b * l_ * i0
    p0 = i0**2 * rc
    v4 = float(m10.rod_velocity_analytic(4.0))
    x12 = xinf * (1 - np.exp(-12 / tau))
    t, y = m10.rod_numeric()
    v4n = float(np.interp(4.0, t, y[:, 1]))
    h = 12.0 / 2400
    f = lambda _t, yy: np.array([yy[1], -kk * yy[1]])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, 0.0, np.array([0.0, v0]), h)
    en = m10.rod_energy_dissipated_numeric()
    hE = 25.0 / 10000
    tE, yE = m10.rod_numeric(25.0, 10000)
    pw = (b * l_ * yE[:, 1] / rc) ** 2 * rc
    s2 = hE / 3 * (pw[0] + 4 * pw[1] + pw[2])
    return dict(
        explanation=[
            "Una barra conductora que se desplaza sobre rieles dentro de un campo magnético uniforme sufre un flujo creciente a través del circuito que forma con la resistencia. Equivale a una fem de movimiento $\\varepsilon = B\\ell v$: las cargas libres de la barra sienten la fuerza $q\\vec v\\times\\vec B$ y se acumulan en un extremo hasta que aparece la corriente $I = \\varepsilon/R$.",
            "Esa corriente, a su vez, está inmersa en el campo, de modo que sobre la barra actúa una fuerza $BI\\ell$ opuesta al movimiento (ley de Lenz). La fuerza es proporcional a la velocidad, igual que una fricción viscosa, de modo que la velocidad decae exponencialmente con constante de tiempo $\\tau = mR/B^2\\ell^2$: a mayor campo o longitud, el frenado es más rápido.",
            "La energía cinética inicial no desaparece sino que se transforma en calor en la resistencia: la potencia $I^2R$ integrada en todo el tiempo coincide exactamente con $\\tfrac12mv_0^2$. Este balance energético es la comprobación más fuerte de que el modelo es coherente, y es el principio del freno magnético y de los frenos de corrientes de Foucault.",
            "Además, la barra se detiene a una distancia finita $x_\\infty = v_0\\tau$, aunque la exponencial nunca llega a cero. Con los datos del problema, $\\tau$ es de 2.5 s y la barra recorre 10 m antes de pararse.",
        ],
        analytic_steps=[
            ("Fem de movimiento", "El flujo a través del circuito es $B\\ell x$, con $x$ la posición de la barra; su derivada es $B\\ell v$.",
             [rf"\varepsilon = B\ell v,\qquad \varepsilon(0) = (0.80)(0.50)(4.0) = {emf0:.3f}\ \mathrm{{V}}"]),
            ("Corriente y fuerza magnética", "La ley de Ohm da la corriente, y la fuerza sobre la barra es $BI\\ell$ con sentido opuesto a $\\vec v$.",
             [rf"I = \dfrac{{\varepsilon}}{{R}} = \dfrac{{B\ell v}}{{R}},\qquad F = -B I \ell = -\dfrac{{B^2\ell^2}}{{R}}v"]),
            ("Segunda ley de Newton", "Con esa fuerza como única fuerza horizontal se obtiene una ecuación diferencial lineal de primer orden.",
             [r"m\dfrac{dv}{dt} = -\dfrac{B^2\ell^2}{R}\,v\ \Rightarrow\ \dfrac{dv}{dt} = -\dfrac{v}{\tau},\quad \tau = \dfrac{mR}{B^2\ell^2}"]),
            ("Constante de tiempo", "Se sustituyen los datos con unidades coherentes: $B^2\\ell^2/R$ tiene unidades de kg/s.",
             [rf"\tau = \dfrac{{(0.20)(2.0)}}{{(0.80)^2(0.50)^2}} = {tau:.3f}\ \mathrm{{s}}"]),
            ("Velocidad y posición", "Se integra: la velocidad es exponencial y la posición se obtiene integrando $v$.",
             [r"v(t) = v_0 e^{-t/\tau},\qquad x(t) = v_0\tau\left(1-e^{-t/\tau}\right)",
              rf"v(4\,\mathrm{{s}}) = 4.0\,e^{{-4/{tau:.1f}}} = {v4:.5f}\ \mathrm{{m/s}}"]),
            ("Alcance total y posición a 12 s", "Para $t\\to\\infty$ la posición tiende a $v_0\\tau$.",
             [rf"x_\infty = v_0\tau = (4.0)({tau:.3f}) = {xinf:.4f}\ \mathrm{{m}},\qquad x(12\,\mathrm{{s}}) = {xinf:.3f}\left(1-e^{{-12/{tau:.1f}}}\right) = {x12:.5f}\ \mathrm{{m}}"]),
            ("Energía disipada", "Se integra la potencia $I^2R$; el resultado debe ser la energía cinética inicial, ya que la barra termina en reposo.",
             [r"E = \int_0^\infty I^2R\,dt = \int_0^\infty \dfrac{B^2\ell^2v_0^2}{R}e^{-2t/\tau}dt = \dfrac{B^2\ell^2v_0^2}{R}\cdot\dfrac{\tau}{2} = \tfrac12mv_0^2",
              rf"E = \tfrac12(0.20)(4.0)^2 = {e0j:.4f}\ \mathrm{{J}}"]),
        ],
        numeric_steps=[
            ("Sistema de primer orden y paso", "Con $y = (x, v)$ se integra $y' = (v,\\,-kv)$ con $k = B^2\\ell^2/mR = 1/\\tau$ mediante Runge-Kutta de cuarto orden, 2400 pasos en 12 s.",
             [rf"k = \dfrac{{B^2\ell^2}}{{mR}} = {kk:.4f}\ \mathrm{{s^{{-1}}}},\qquad h = \dfrac{{12}}{{2400}} = {h:.4f}\ \mathrm{{s}},\qquad y_0 = (0,\ 4.0)"]),
            ("Pendientes del primer paso", "RK4 evalúa la derivada cuatro veces: al inicio, dos veces en el punto medio y al final.",
             [rf"k_1 = ({k1[0]:.5f},\ {k1[1]:.5f})",
              rf"k_2 = f(y_0+\tfrac h2k_1) = ({k2[0]:.5f},\ {k2[1]:.5f})",
              rf"k_3 = f(y_0+\tfrac h2k_2) = ({k3[0]:.5f},\ {k3[1]:.5f}),\qquad k_4 = f(y_0+hk_3) = ({k4[0]:.5f},\ {k4[1]:.5f})"]),
            ("Actualizar la solución", "Se combina $y_1 = y_0 + \\frac h6(k_1+2k_2+2k_3+k_4)$.",
             [rf"y_1 = (0,\ 4.0) + \dfrac{{{h:.4f}}}{{6}}(\ldots) = (x_1,\ v_1) = ({y1[0]:.7f}\ \mathrm{{m}},\ {y1[1]:.7f}\ \mathrm{{m/s}})",
              rf"v_0e^{{-h/\tau}} = {v0 * np.exp(-h / tau):.7f}\ \mathrm{{m/s}}\ \ (\text{{exacto}})"]),
            ("Repetir hasta t = 12 s", "El mismo paso se repite 2400 veces. La velocidad a 4 s se interpola linealmente entre las dos muestras vecinas del malla.",
             [rf"v_{{num}}(4\,\mathrm{{s}}) = {v4n:.5f}\ \mathrm{{m/s}}\ \ (\text{{analítico }}{v4:.5f}),\qquad x_{{num}}(12\,\mathrm{{s}}) = {y[-1, 0]:.5f}\ \mathrm{{m}}\ \ (\text{{analítico }}{x12:.5f})"]),
            ("Energía: integrando I²R con Simpson", "Se integra de nuevo el sistema hasta 25 s con 10000 pasos y se evalúa en cada nodo la potencia $P = (B\\ell v/R)^2R$. Los tres primeros valores permiten ver la mecánica de Simpson.",
             [rf"P_0 = \left(\dfrac{{(0.80)(0.50)({v0:.1f})}}{{2.0}}\right)^2(2.0) = {p0:.5f}\ \mathrm{{W}},\qquad P_1 = {pw[1]:.6f}\ \mathrm{{W}},\qquad P_2 = {pw[2]:.6f}\ \mathrm{{W}}"]),
            ("Primer par de intervalos", "Cada par de subintervalos de ancho $h_E = 2.5\\ \\mathrm{ms}$ aporta $\\frac{h_E}{3}(P_0+4P_1+P_2)$ a la integral.",
             [rf"\Delta E_{{1}} = \dfrac{{{hE}}}{{3}}\left({p0:.5f}+4({pw[1]:.5f})+{pw[2]:.5f}\right) = {s2:.7f}\ \mathrm{{J}}"]),
            ("Suma completa y error", "Se suman los 5000 pares. El resultado debe coincidir con la energía cinética inicial.",
             [rf"E_{{num}} = {en:.5f}\ \mathrm{{J}},\qquad E_{{analit}} = {e0j:.5f}\ \mathrm{{J}},\qquad \text{{error relativo}} = {sci(abs(en - e0j) / e0j, 2)}"]),
        ],
    )


# =============================================================================== capitulo 11
def _rl() -> dict:
    eps, r, l_ = m11.EPS, m11.R, m11.L
    tau = m11.tau_rl()
    iinf = eps / r
    u = m11.energy_magnetic()
    i_tau = float(m11.i_growth_analytic(tau))
    th = tau * np.log(2)
    # decaimiento
    hd = 0.3 / 3000
    fd = lambda _t, i: np.array([-r * i[0] / l_])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(fd, 0.0, np.array([iinf]), hd)
    t, y = m11.integrate(fd, np.array([iinf]), 0.0, 0.3, 3000, rk4_step) if hasattr(m11, "integrate") else (None, None)
    target = 0.5 * iinf
    kx = int(np.argmax(y[:, 0] <= target))
    t0_, t1_ = t[kx - 1], t[kx]
    y0_, y1_ = y[kx - 1, 0], y[kx, 0]
    thn = m11.time_to_half_numeric()
    # energia
    he = 0.5 / 5000
    fg = lambda _t, i: np.array([(eps - r * i[0]) / l_])  # noqa: E731
    g1, g2, g3, g4, i1 = _rk4(fg, 0.0, np.array([0.0]), he)
    te, ye = m11.rl_numeric(0.5, 5000)
    ie = ye[:, 0]
    pl = ie * (eps - r * ie)
    s1 = he / 3 * (pl[0] + 4 * pl[1] + pl[2])
    un = m11.energy_integral_numeric()
    return dict(
        explanation=[
            "Un inductor se opone a los cambios de corriente: cuando $i$ varía aparece en él una fem autoinducida $\\varepsilon_L = -L\\,di/dt$. Por eso en un circuito RL la corriente no salta de golpe al valor $\\varepsilon/R$ que le daría la ley de Ohm, sino que crece de forma gradual, igual que la velocidad de un cuerpo que cae con rozamiento viscoso. La inductancia $L$ hace el papel de la masa.",
            "La ecuación del circuito, $\\varepsilon = iR + L\\,di/dt$, tiene solución exponencial con constante de tiempo $\\tau_L = L/R$. Tras un tiempo $\\tau_L$ la corriente ha alcanzado el 63 % de su valor final y tras cinco constantes de tiempo está a menos del 1 %. Si se retira la fuente y se deja el circuito cerrado, la corriente decae con la misma constante de tiempo.",
            "La energía que la fuente entrega al inductor, $\\int i\\,\\varepsilon_L\\,dt$ con signo, se almacena en su campo magnético, $U_B = \\tfrac12Li^2$. Esa energía es independiente de cómo se llegó a la corriente final: solo depende de $L$ y de $i_\\infty$. Es análoga a la energía cinética $\\tfrac12mv^2$ y a la de un capacitor, $\\tfrac12CV^2$.",
            "Con $0.20\\ \\mathrm{H}$, $4.0\\ \\Omega$ y 12 V, la constante de tiempo es de 50 ms: el transitorio es perceptible en un osciloscopio pero no a simple vista. Las bobinas de los relés y las de los motores pueden dar chispas al abrir el interruptor precisamente porque $di/dt$ es enorme y con él la fem $-L\\,di/dt$.",
        ],
        analytic_steps=[
            ("Ecuación del circuito", "Por la regla de las mallas, la fem de la fuente se reparte entre la caída óhmica y la fem del inductor.",
             [r"\varepsilon - iR - L\dfrac{di}{dt} = 0 \ \Rightarrow\ L\dfrac{di}{dt} = \varepsilon - iR"]),
            ("Separar variables", "Se reúnen las $i$ de un lado y los $t$ del otro y se integra desde $i = 0$ en $t = 0$.",
             [r"\int_0^i\dfrac{di'}{\varepsilon - Ri'} = \int_0^t\dfrac{dt'}{L} \ \Rightarrow\ -\dfrac1R\ln\dfrac{\varepsilon - Ri}{\varepsilon} = \dfrac tL"]),
            ("Despejar la corriente", "Se exponencia y se despeja $i$. Aparece la constante de tiempo $\\tau_L = L/R$.",
             [r"i(t) = \dfrac{\varepsilon}{R}\left(1 - e^{-t/\tau_L}\right),\qquad \tau_L = \dfrac LR"]),
            ("Valores numéricos", "Se sustituyen $L = 0.20$ H, $R = 4.0\\ \\Omega$ y $\\varepsilon = 12$ V.",
             [rf"\tau_L = \dfrac{{0.20}}{{4.0}} = {tau * 1e3:.2f}\ \mathrm{{ms}},\qquad i_\infty = \dfrac{{12}}{{4.0}} = {iinf:.2f}\ \mathrm{{A}}",
              rf"i(\tau_L) = {iinf:.2f}(1-e^{{-1}}) = {i_tau:.4f}\ \mathrm{{A}}"]),
            ("Energía almacenada", "A régimen permanente $di/dt = 0$ y toda la corriente final fluye por la resistencia; la energía del campo es $\\tfrac12Li_\\infty^2$.",
             [rf"U_B = \tfrac12Li_\infty^2 = \tfrac12(0.20)({iinf:.2f})^2 = {u:.4f}\ \mathrm{{J}}"]),
            ("Decaimiento tras retirar la fuente", "Si en $t=0$ la corriente vale $i_\\infty$ y $\\varepsilon$ se anula, la ecuación es $L\\,di/dt = -Ri$ y la corriente decae exponencialmente.",
             [r"i(t) = i_\infty e^{-t/\tau_L}"]),
            ("Tiempo para llegar al 50 %", "Se iguala $i = i_\\infty/2$ y se despeja el tiempo.",
             [rf"\tfrac12 = e^{{-t_{{1/2}}/\tau_L}}\ \Rightarrow\ t_{{1/2}} = \tau_L\ln2 = ({tau * 1e3:.2f}\ \mathrm{{ms}})(0.693147) = {th * 1e3:.6f}\ \mathrm{{ms}}"]),
        ],
        numeric_steps=[
            ("Plantear la ecuación como sistema de primer orden", "Para el decaimiento se integra $di/dt = -Ri/L$ con RK4, 3000 pasos en 0.3 s. Es una sola variable, de modo que RK4 se reduce a cuatro pendientes escalares.",
             [rf"h = \dfrac{{0.3}}{{3000}} = {sci(hd, 3)}\ \mathrm{{s}},\qquad f(i) = -\dfrac{{R}}{{L}}i = -{r / l_:.1f}\,i,\qquad i_0 = {iinf:.1f}\ \mathrm{{A}}"]),
            ("Primer paso de RK4", "Se evalúan las cuatro pendientes y se combina.",
             [rf"k_1 = {k1[0]:.5f},\quad k_2 = {k2[0]:.5f},\quad k_3 = {k3[0]:.5f},\quad k_4 = {k4[0]:.5f}\ \mathrm{{A/s}}",
              rf"i_1 = i_0 + \dfrac h6(k_1+2k_2+2k_3+k_4) = {y1[0]:.8f}\ \mathrm{{A}}\ \ (\text{{exacto }}{iinf * np.exp(-hd / tau):.8f})"]),
            ("Buscar el cruce con el 50 %", "Se repite hasta que la corriente cae por debajo de $1.5$ A; se halla el primer índice $k$ con $i_k \\le i_\\infty/2$ y se interpolan linealmente las dos muestras vecinas.",
             [rf"t_{{k-1}} = {t0_ * 1e3:.5f}\ \mathrm{{ms}},\ i_{{k-1}} = {y0_:.7f}\ \mathrm{{A}};\qquad t_k = {t1_ * 1e3:.5f}\ \mathrm{{ms}},\ i_k = {y1_:.7f}\ \mathrm{{A}}"]),
            ("Interpolación lineal", "Se despeja el instante en que la recta que une los dos puntos pasa por el objetivo $i = 1.5$ A.",
             [rf"t_{{1/2}} = t_{{k-1}} + \dfrac{{(i_{{k-1}} - 1.5)(t_k - t_{{k-1}})}}{{i_{{k-1}} - i_k}} = {thn * 1e3:.6f}\ \mathrm{{ms}}",
              rf"\text{{analítico: }}{th * 1e3:.6f}\ \mathrm{{ms}},\qquad \text{{error relativo}} = {sci(abs(thn - th) / th, 2)}"]),
            ("Energía: crecimiento de i con RK4", "Para la energía se integra ahora el crecimiento, $di/dt = (\\varepsilon - Ri)/L$, con 5000 pasos hasta 0.5 s. En el primer paso la corriente parte de cero.",
             [rf"h = {sci(he, 3)}\ \mathrm{{s}},\qquad k_1 = {g1[0]:.4f},\ k_2 = {g2[0]:.5f},\ k_3 = {g3[0]:.5f},\ k_4 = {g4[0]:.5f}\ \mathrm{{A/s}},\qquad i_1 = {i1[0]:.7f}\ \mathrm{{A}}"]),
            ("Integrando la potencia del inductor con Simpson", "La potencia entregada al inductor es $p = i\\,v_L$ con $v_L = \\varepsilon - Ri$. Se evalúa en los nodos y se integra con la regla de Simpson; los dos primeros intervalos dan:",
             [rf"p_0 = {pl[0]:.4f}\ \mathrm{{W}},\quad p_1 = {pl[1]:.6f}\ \mathrm{{W}},\quad p_2 = {pl[2]:.6f}\ \mathrm{{W}}",
              rf"\Delta U_1 = \dfrac{{h}}{{3}}(p_0+4p_1+p_2) = {s1:.4e}\ \mathrm{{J}}".replace("e-", r"\times10^{-").replace("e+", r"\times10^{") + "}" if False else rf"\Delta U_1 = \dfrac{{h}}{{3}}(p_0+4p_1+p_2) = {sci(s1, 4)}\ \mathrm{{J}}"]),
            ("Resultado y error", "Se suman los 2500 pares de intervalos y se compara con $\\tfrac12Li_\\infty^2$.",
             [rf"U_{{num}} = {un:.6f}\ \mathrm{{J}},\qquad U_{{analit}} = {u:.6f}\ \mathrm{{J}},\qquad \text{{error relativo}} = {sci(abs(un - u) / u, 2)}"]),
        ],
    )


def _rlc() -> dict:
    l_, c, r, q0 = m11.L2, m11.C2, m11.R2, m11.Q0
    w0, g, wd = m11.omega0(), m11.gamma_(), m11.omega_d()
    th = 2 * np.pi / wd
    qf = m11.quality_factor()
    ratio_a = np.exp(-2 * g * th)
    ratio_n = m11.energy_ratio_after_period()
    tn = m11.period_numeric()
    u0 = q0**2 / (2 * c)
    # RK4 primer paso (period_numeric: 20000 pasos en 0.2 s)
    h = 0.2 / 20000
    f = lambda _t, y: np.array([y[1], -(r / l_) * y[1] - y[0] / (l_ * c)])  # noqa: E731
    y0 = np.array([q0, 0.0])
    k1, k2, k3, k4, y1 = _rk4(f, 0.0, y0, h)
    t, y = m11.rlc_numeric(0.2, 20000)
    q = y[:, 0]
    idx = [i for i in range(1, len(q)) if q[i - 1] > 0 >= q[i]]

    def cross(i):
        return t[i] - q[i] * (t[i] - t[i - 1]) / (q[i] - q[i - 1])

    c0_, c1_ = cross(idx[0]), cross(idx[1])
    i0 = idx[0]
    # energia
    tT, yT = m11.rlc_numeric(th + 1e-9, 8000, r)
    qT, iT = yT[-1]
    uT = qT**2 / (2 * c) + 0.5 * l_ * iT**2
    # cantidades del primer paso en energia
    return dict(
        explanation=[
            "Un circuito RLC en serie es el análogo eléctrico de un oscilador mecánico con rozamiento: la inductancia hace de masa, el inverso de la capacitancia de constante del resorte y la resistencia de coeficiente de amortiguamiento. La carga del capacitor oscila porque la energía pasa una y otra vez del campo eléctrico del capacitor al campo magnético del inductor.",
            "La resistencia disipa energía en forma de calor, de modo que la amplitud de las oscilaciones decae exponencialmente con el factor $e^{-\\gamma t}$, con $\\gamma = R/2L$. Mientras $\\gamma < \\omega_0 = 1/\\sqrt{LC}$ el circuito oscila (régimen subamortiguado) con la frecuencia reducida $\\omega_d = \\sqrt{\\omega_0^2-\\gamma^2}$; si $\\gamma \\ge \\omega_0$ la carga decae sin oscilar.",
            "El factor de calidad $Q = \\omega_0L/R$ cuenta aproximadamente cuántos radianes avanza la fase mientras la energía cae a $1/e$: un $Q$ grande significa oscilaciones que duran muchos ciclos. Con los datos del problema $Q\\approx6.3$, de modo que la energía se reduce en torno a una cuarta parte de su valor tras un solo periodo.",
            "La energía almacenada es proporcional al cuadrado de la envolvente, por eso decae como $e^{-2\\gamma t}$. Es una buena aproximación analítica, pero no exacta, porque la energía instantánea oscila dentro de la envolvente y no vale exactamente lo mismo en dos instantes separados por un periodo; la integración numérica calcula la energía exacta.",
        ],
        analytic_steps=[
            ("Ecuación de la malla", "La suma de caídas de tensión alrededor del circuito sin fuente es cero. Con $i = \\dot q$ queda una ecuación idéntica a la del oscilador amortiguado.",
             [r"L\ddot q + R\dot q + \dfrac qC = 0\ \Rightarrow\ \ddot q + 2\gamma\dot q + \omega_0^2q = 0"]),
            ("Frecuencia natural y coeficiente de amortiguamiento", "Se identifican los dos parámetros con los datos $L=0.10$ H, $C = 100\\ \\mu$F y $R = 5.0\\ \\Omega$.",
             [rf"\omega_0 = \dfrac1{{\sqrt{{LC}}}} = \dfrac{{1}}{{\sqrt{{(0.10)({sci(c, 3)})}}}} = {w0:.3f}\ \mathrm{{rad/s}},\qquad \gamma = \dfrac R{{2L}} = \dfrac{{5.0}}{{0.20}} = {g:.2f}\ \mathrm{{s^{{-1}}}}"]),
            ("Frecuencia amortiguada y periodo", "Como $\\gamma\\ll\\omega_0$ el sistema es subamortiguado y oscila con $\\omega_d$.",
             [rf"\omega_d = \sqrt{{{w0:.3f}^2 - {g:.2f}^2}} = {wd:.3f}\ \mathrm{{rad/s}},\qquad T = \dfrac{{2\pi}}{{\omega_d}} = {th * 1e3:.5f}\ \mathrm{{ms}}"]),
            ("Solución con las condiciones iniciales", "Con $q(0)=Q_0$ y $i(0) = 0$ (capacitor cargado y sin corriente) la solución es una oscilación con amplitud decreciente.",
             [r"q(t) = Q_0e^{-\gamma t}\left[\cos\omega_dt + \dfrac{\gamma}{\omega_d}\sin\omega_dt\right]",
              rf"\dfrac{{\gamma}}{{\omega_d}} = \dfrac{{{g:.2f}}}{{{wd:.3f}}} = {g / wd:.5f}"]),
            ("Factor de calidad", "Se evalúa con la definición $Q = \\omega_0L/R$.",
             [rf"Q = \dfrac{{\omega_0L}}{{R}} = \dfrac{{({w0:.3f})(0.10)}}{{5.0}} = {qf:.3f}"]),
            ("Energía tras un periodo (envolvente)", "La energía es proporcional a $e^{-2\\gamma t}$, de modo que tras un periodo cae por el factor:",
             [rf"\dfrac{{U(T)}}{{U(0)}} \approx e^{{-2\gamma T}} = e^{{-2({g:.2f})({th:.6f})}} = {ratio_a:.5f}"]),
            ("Energía inicial", "Antes de soltar, toda la energía está en el capacitor.",
             [rf"U(0) = \dfrac{{Q_0^2}}{{2C}} = \dfrac{{(10^{{-3}})^2}}{{2({sci(c, 3)})}} = {u0:.4f}\ \mathrm{{J}},\qquad U(T)\approx {ratio_a * u0:.5f}\ \mathrm{{J}}"]),
        ],
        numeric_steps=[
            ("Sistema de primer orden", "Con $y = (q, i)$ se integra $y' = (i,\\ -(R/L)i - q/LC)$ con RK4. Se usan 20000 pasos para el periodo (0.2 s) y 8000 para la energía; el paso de la medición es el siguiente.",
             [rf"h = \dfrac{{0.2}}{{20000}} = {sci(h, 3)}\ \mathrm{{s}},\qquad \dfrac{{R}}{{L}} = {r / l_:.1f}\ \mathrm{{s^{{-1}}}},\qquad \dfrac1{{LC}} = {sci(1 / (l_ * c), 4)}\ \mathrm{{s^{{-2}}}}"]),
            ("Pendientes del primer paso", "Partiendo de $y_0 = (10^{-3}\\ \\mathrm{C},\\ 0)$ se calculan las cuatro pendientes de RK4.",
             [rf"k_1 = ({sci(k1[0], 4)},\ {sci(k1[1], 4)})",
              rf"k_2 = ({sci(k2[0], 4)},\ {sci(k2[1], 4)}),\qquad k_3 = ({sci(k3[0], 4)},\ {sci(k3[1], 4)}),\qquad k_4 = ({sci(k4[0], 4)},\ {sci(k4[1], 4)})"]),
            ("Primera actualización", "Se combina $y_1 = y_0 + \\frac h6(k_1+2k_2+2k_3+k_4)$. La carga baja un poco y aparece una corriente negativa.",
             [rf"q_1 = {sci(y1[0], 8)}\ \mathrm{{C}},\qquad i_1 = {sci(y1[1], 6)}\ \mathrm{{A}}"]),
            ("Detectar los cruces por cero", "Tras los 20000 pasos se buscan los índices con $q_{k-1}>0\\ge q_k$ (cruces descendentes). Se toman los dos primeros y se interpola linealmente cada uno.",
             [rf"t_{{k-1}} = {t[i0 - 1] * 1e3:.5f}\ \mathrm{{ms}},\ q_{{k-1}} = {sci(q[i0 - 1], 5)}\ \mathrm{{C}};\qquad t_k = {t[i0] * 1e3:.5f}\ \mathrm{{ms}},\ q_k = {sci(q[i0], 5)}\ \mathrm{{C}}",
              rf"t_c = t_k - q_k\dfrac{{t_k-t_{{k-1}}}}{{q_k - q_{{k-1}}}} = {c0_ * 1e3:.6f}\ \mathrm{{ms}}"]),
            ("Periodo medido", "El segundo cruce descendente se interpola igual y la diferencia es el periodo amortiguado.",
             [rf"t_c' = {c1_ * 1e3:.6f}\ \mathrm{{ms}}\ \Rightarrow\ T_{{num}} = t_c'-t_c = {tn * 1e3:.6f}\ \mathrm{{ms}}\ \ (\text{{analítico }}{th * 1e3:.6f}),\ \ \text{{error}} = {sci(abs(tn - th) / th, 2)}"]),
            ("Energía tras un periodo", "Se integra con 8000 pasos exactamente hasta $t=T$, y se calcula $U = q^2/2C + \\tfrac12Li^2$ con los valores finales de $q$ e $i$.",
             [rf"q(T) = {sci(qT, 5)}\ \mathrm{{C}},\qquad i(T) = {sci(iT, 5)}\ \mathrm{{A}},\qquad U(T) = {uT:.5f}\ \mathrm{{J}}",
              rf"\dfrac{{U(T)}}{{U(0)}} = \dfrac{{{uT:.5f}}}{{{u0:.4f}}} = {ratio_n:.5f}"]),
            ("Comparación y aclaración", "La integración numérica da el valor exacto de la energía; el valor $e^{-2\\gamma T}$ de la envolvente difiere un poco porque, tras un periodo, la corriente no es cero.",
             [rf"\text{{numérico }}{ratio_n:.5f}\ \ \text{{vs}}\ \ \text{{envolvente }}{ratio_a:.5f},\qquad \text{{diferencia relativa}} = {sci(abs(ratio_n - ratio_a) / ratio_a, 2)}"]),
        ],
    )


# =============================================================================== capitulo 12
def _agm_rows(theta0_deg: float):
    a, b = 1.0, float(np.cos(np.deg2rad(theta0_deg) / 2))
    rows = [["0", f"{a:.8f}", f"{b:.8f}"]]
    for k in range(1, 6):
        a, b = 0.5 * (a + b), float(np.sqrt(a * b))
        rows.append([str(k), f"{a:.8f}", f"{b:.8f}"])
        if abs(a - b) < 1e-9:
            break
    return rows, a


def _dipole() -> dict:
    mu, bf, i_ = m12.MU, m12.B_FIELD, m12.INERTIA
    w2 = m12.omega_small() ** 2
    w = m12.omega_small()
    t0 = 2 * np.pi / w
    rows30, a30 = _agm_rows(30.0)
    rows90, a90 = _agm_rows(90.0)
    t30, t90 = m12.period_exact(30.0), m12.period_exact(90.0)
    n30, n90 = m12.period_numeric(30.0), m12.period_numeric(90.0)
    drift = m12.energy_conservation_drift()
    th0 = np.deg2rad(30.0)
    h = 6.0 / 40000
    f = lambda _t, y: np.array([y[1], -w2 * np.sin(y[0])])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, 0.0, np.array([th0, 0.0]), h)
    t, y = m12.integrate(f, np.array([th0, 0.0]), 0.0, 6.0, 40000, rk4_step)
    th = y[:, 0]
    idx = [i for i in range(1, len(th)) if th[i - 1] > 0 >= th[i]]

    def cross(i):
        return t[i] - th[i] * (t[i] - t[i - 1]) / (th[i] - th[i - 1])

    c0_, c1_ = cross(idx[0]), cross(idx[1])
    i0 = idx[0]
    e0 = m12.potential_energy(np.deg2rad(60.0))
    return dict(
        explanation=[
            "Un dipolo magnético de momento $\\vec\\mu$ colocado en un campo uniforme $\\vec B$ sufre un torque $\\vec\\tau = \\vec\\mu\\times\\vec B$ que tiende a alinearlo con el campo. No hay fuerza neta, solo torque; la energía potencial asociada es $U = -\\vec\\mu\\cdot\\vec B$, mínima cuando están paralelos. Una brújula es exactamente eso.",
            "Si el dipolo (aguja imanada) puede girar libremente alrededor de un eje perpendicular al campo, la ecuación de movimiento es $I\\ddot\\theta = -\\mu B\\sin\\theta$, idéntica a la del péndulo simple con $g/L$ sustituido por $\\mu B/I$. Para ángulos pequeños, $\\sin\\theta\\approx\\theta$ y el movimiento es armónico simple con $\\omega=\\sqrt{\\mu B/I}$, cuyo periodo no depende de la amplitud.",
            "Para amplitudes grandes ese resultado deja de valer: la fuerza restauradora es menor que la lineal, la aguja tarda más y el periodo crece con la amplitud. El periodo exacto se expresa con la media aritmético-geométrica (AGM), un algoritmo que converge muy rápido y que evita evaluar integrales elípticas.",
            "Con $\\mu = 0.10\\ \\mathrm{A\\,m^2}$, $B = 50\\ \\mu\\mathrm{T}$ e $I = 10^{-6}\\ \\mathrm{kg\\,m^2}$, el periodo de pequeñas oscilaciones ronda los 2.8 s. Medir ese periodo es una forma práctica de determinar la componente horizontal del campo magnético terrestre.",
        ],
        analytic_steps=[
            ("Torque y ecuación de movimiento", "El torque tiene módulo $\\mu B\\sin\\theta$ y tiende a reducir $\\theta$, por eso lleva signo negativo en la segunda ley de Newton para la rotación.",
             [r"I\ddot\theta = -\mu B\sin\theta"]),
            ("Aproximación de pequeñas oscilaciones", "Para $\\theta\\ll1$ se reemplaza $\\sin\\theta$ por $\\theta$ y queda la ecuación del oscilador armónico.",
             [r"\ddot\theta = -\omega^2\theta,\qquad \omega^2 = \dfrac{\mu B}{I}"]),
            ("Frecuencia angular", "Se sustituyen los datos con las unidades del SI: $\\mu B = 0.10\\times5\\times10^{-5} = 5\\times10^{-6}$ N m.",
             [rf"\omega^2 = \dfrac{{(0.10)({sci(bf, 3)})}}{{{sci(i_, 3)}}} = {w2:.4f}\ \mathrm{{s^{{-2}}}},\qquad \omega = {w:.5f}\ \mathrm{{rad/s}}"]),
            ("Periodo de pequeñas oscilaciones", "Es independiente de la amplitud, como en cualquier oscilador armónico.",
             [rf"T_0 = \dfrac{{2\pi}}{{\omega}} = 2\pi\sqrt{{\dfrac{{I}}{{\mu B}}}} = \dfrac{{2\pi}}{{{w:.5f}}} = {t0:.4f}\ \mathrm{{s}}"]),
            ("Corrección por amplitud finita", "El periodo exacto de la ecuación con $\\sin\\theta$ completo es $T_0$ dividido entre la media aritmético-geométrica de $1$ y $\\cos(\\theta_0/2)$.",
             [r"T(\theta_0) = \dfrac{T_0}{\mathrm{AGM}\!\left(1,\ \cos\dfrac{\theta_0}{2}\right)}"]),
            ("Amplitud de 30°", "La AGM de $1$ y $\\cos15^\\circ$ se calcula iterando $a_{n+1} = (a_n+b_n)/2$, $b_{n+1} = \\sqrt{a_nb_n}$ hasta que coinciden (ver tabla en el apartado numérico).",
             [rf"\mathrm{{AGM}}(1,\cos15^\circ) = {a30:.8f},\qquad T(30^\circ) = \dfrac{{{t0:.4f}}}{{{a30:.8f}}} = {t30:.5f}\ \mathrm{{s}}"]),
            ("Amplitud de 90°", "Se repite con $\\cos45^\\circ$; el periodo aumenta casi 18 % sobre $T_0$.",
             [rf"\mathrm{{AGM}}(1,\cos45^\circ) = {a90:.8f},\qquad T(90^\circ) = \dfrac{{{t0:.4f}}}{{{a90:.8f}}} = {t90:.5f}\ \mathrm{{s}}\quad({(t90 / t0 - 1) * 100:.2f}\,\%\ \text{{mayor que }}T_0)"]),
        ],
        numeric_steps=[
            ("Reducir a un sistema de primer orden", "Con $y = (\\theta, \\omega_\\theta)$ se integra $y' = (\\omega_\\theta,\\ -\\omega^2\\sin\\theta)$ con RK4, 40000 pasos en 6 s, partiendo del reposo en $\\theta_0$. Se muestra el caso de 30°.",
             [rf"h = \dfrac{{6}}{{40000}} = {sci(h, 3)}\ \mathrm{{s}},\qquad \omega^2 = {w2:.3f}\ \mathrm{{s^{{-2}}}},\qquad y_0 = ({th0:.6f}\ \mathrm{{rad}},\ 0)"]),
            ("Primer paso de RK4", "Se evalúan las cuatro pendientes (la segunda coordenada de $k_1$ es la aceleración angular inicial $-\\omega^2\\sin\\theta_0$).",
             [rf"k_1 = ({k1[0]:.6f},\ {k1[1]:.6f}),\quad k_2 = ({k2[0]:.6f},\ {k2[1]:.6f})",
              rf"k_3 = ({k3[0]:.6f},\ {k3[1]:.6f}),\quad k_4 = ({k4[0]:.6f},\ {k4[1]:.6f})",
              rf"y_1 = y_0 + \dfrac h6(k_1+2k_2+2k_3+k_4) = ({y1[0]:.8f}\ \mathrm{{rad}},\ {y1[1]:.8f}\ \mathrm{{rad/s}})"]),
            ("Medir el periodo", "El primer cruce descendente de $\\theta$ por cero ocurre en $T/4$, el siguiente a $5T/4$; su diferencia es un periodo completo. Cada cruce se interpola linealmente entre las muestras $k-1$ y $k$.",
             [rf"t_{{k-1}} = {t[i0 - 1]:.5f}\ \mathrm{{s}},\ \theta_{{k-1}} = {th[i0 - 1]:.6f};\qquad t_k = {t[i0]:.5f}\ \mathrm{{s}},\ \theta_k = {th[i0]:.6f}",
              rf"t_c = t_k - \theta_k\dfrac{{t_k - t_{{k-1}}}}{{\theta_k - \theta_{{k-1}}}} = {c0_:.6f}\ \mathrm{{s}}"]),
            ("Segundo cruce y periodo", "Se obtiene el segundo cruce descendente y se resta.",
             [rf"t_c' = {c1_:.6f}\ \mathrm{{s}},\qquad T_{{num}}(30^\circ) = t_c'-t_c = {n30:.6f}\ \mathrm{{s}}"]),
            ("Tabla de la AGM para comparar (30°)", "El valor de referencia se obtiene iterando la media aritmética y la geométrica; en pocas iteraciones ambas coinciden.",
             [_tab([r"n", r"a_n", r"b_n"], rows30)]),
            ("Comparación para 30° y 90°", "El mismo procedimiento con 90° (donde la corrección es mayor) da:",
             [rf"T_{{num}}(30^\circ) = {n30:.6f}\ \mathrm{{s}}\ \ (\text{{AGM }}{t30:.6f}),\qquad T_{{num}}(90^\circ) = {n90:.6f}\ \mathrm{{s}}\ \ (\text{{AGM }}{t90:.6f})",
              rf"\text{{errores relativos: }}{sci(abs(n30 - t30) / t30, 2)},\ \ {sci(abs(n90 - t90) / t90, 2)}"]),
            ("Conservación de la energía", "Como comprobación, a 60° se calcula $E = \\tfrac12I\\dot\\theta^2 - \\mu B\\cos\\theta$ en cada paso (6000 pasos en 3 s) y se mide la mayor desviación respecto de $E_0 = -\\mu B\\cos60^\\circ$.",
             [rf"E_0 = -(0.10)({sci(bf, 3)})\cos60^\circ = {sci(e0, 5)}\ \mathrm{{J}},\qquad \max_k\dfrac{{|E_k - E_0|}}{{|E_0|}} = {sci(drift, 2)}"]),
        ],
    )


def _curie() -> dict:
    xs = (0.1, 0.8, 1.6, 3.2)
    xp = m12.x_param(1.0, 300.0)
    mm = {x: m12.magnetization_metropolis(x) for x in xs}
    # Metropolis reducido: 10 espines, x = 0.8
    xr = 0.8
    nr = 10
    rng = np.random.default_rng(4)
    s0 = rng.choice([-1, 1], size=nr)
    de = 2 * xr * s0
    u = rng.uniform(size=nr)
    pacc = np.exp(-np.clip(de, 0, 50))
    acc = (de <= 0) | (u < pacc)
    s1 = np.where(acc, -s0, s0)

    def sg(v):
        return "+" if v > 0 else "-"

    rows = []
    for j in range(nr):
        rows.append([str(j + 1), sg(s0[j]), f"{de[j]:+.1f}", f"{pacc[j]:.4f}", f"{u[j]:.4f}", "si" if acc[j] else "no", sg(s1[j])])
    tab = _tab([r"j", r"s", r"\Delta E/kT", r"p_{acc}", r"u", r"\text{acepta}", r"s'"], [[c if k != 5 else r"\text{" + c + "}" for k, c in enumerate(rw)] for rw in rows])
    w = np.exp(-2 * xr)
    p_plus = 1 / (1 + w)
    n, sweeps = 100_000, 80
    nav = sweeps - sweeps // 2
    return dict(
        explanation=[
            "En un paramagneto ideal cada átomo tiene un momento magnético permanente que no interactúa con los vecinos. Sin campo, la agitación térmica los orienta al azar y la magnetización neta es cero; el campo aplicado favorece la orientación paralela porque baja la energía $-\\vec\\mu\\cdot\\vec B$, pero la temperatura mantiene parte de los momentos en sentido contrario. Lo que decide el resultado es la razón entre la energía magnética $\\mu_BB$ y la energía térmica $kT$.",
            "Para espines $1/2$ solo hay dos estados, con energías $\\mp\\mu_BB$. Según la distribución de Boltzmann, las probabilidades son proporcionales a $e^{\\pm x}$, con $x = \\mu_BB/kT$, y la magnetización relativa a la de saturación es $M/M_s = \\tanh x$. A campos fuertes o temperaturas bajas $\\tanh x\\to1$ (saturación); a campos débiles, $\\tanh x\\approx x$.",
            "El límite $x\\ll1$ es la ley de Curie: la magnetización es proporcional a $B/T$, y la susceptibilidad decrece como $1/T$. Con $B=1$ T y $T=300$ K, $x\\approx2\\times10^{-3}$: a temperatura ambiente la magnetización es solo unas milésimas de la de saturación, y por eso los paramagnetos corrientes son tan débilmente magnéticos.",
            "El método de Monte Carlo de Metropolis permite comprobar la teoría sin usar la distribución de Boltzmann en el cálculo: se proponen inversiones al azar de los espines y se aceptan con la probabilidad dictada por el cambio de energía. Después de varios barridos la población se estabiliza en el valor de equilibrio, con un error estadístico del orden de $1/\\sqrt{N}$.",
        ],
        analytic_steps=[
            ("Niveles de energía", "Un espín $1/2$ con momento $\\mu_B$ en el campo $B$ tiene dos estados, con el momento paralelo (energía baja) o antiparalelo (energía alta) al campo.",
             [r"U_\pm = \mp\mu_BB"]),
            ("Probabilidades de Boltzmann", "La probabilidad de cada estado es proporcional a $e^{-U/kT}$. La función de partición es la suma de los dos pesos.",
             [r"x \equiv \dfrac{\mu_BB}{kT},\qquad p_\pm = \dfrac{e^{\pm x}}{e^{x}+e^{-x}}"]),
            ("Magnetización media", "Cada espín paralelo aporta $+\\mu_B$ y cada antiparalelo $-\\mu_B$, de modo que el promedio normalizado por la saturación es $p_+-p_-$.",
             [r"\dfrac{M}{M_s} = p_+ - p_- = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = \tanh x"]),
            ("Límite de campo débil", "Para $x\\ll1$, $\\tanh x = x - x^3/3 + \\cdots \\approx x$, y se recupera la ley de Curie $M\\propto B/T$.",
             [r"\dfrac{M}{M_s}\approx x = \dfrac{\mu_BB}{kT}"]),
            ("Valor típico de x", "Con $\\mu_B = 9.274\\times10^{-24}$ J/T, $B = 1$ T, $T = 300$ K y $k = 1.3806\\times10^{-23}$ J/K:",
             [rf"x = \dfrac{{({sci(m12.MU_B, 5)})(1.0)}}{{({sci(K_B, 5)})(300)}} = {sci(xp, 4)}"]),
            ("Valores de tanh x", "Para las cuatro condiciones del problema se obtiene la magnetización teórica; se aprecia que el límite lineal ya falla a $x\\sim1$.",
             [_tab([r"x", r"\tanh x", r"x\ \text{(Curie)}"], [[f"{x}", f"{np.tanh(x):.5f}", f"{x}"] for x in xs])]),
            ("Error de la aproximación de Curie", "La ley lineal sobrestima $M/M_s$ y la sobrestimación crece con $x$.",
             [rf"\dfrac{{x - \tanh x}}{{\tanh x}}\Big|_{{x=0.1}} = {(0.1 - np.tanh(0.1)) / np.tanh(0.1) * 100:.2f}\,\%,\qquad \Big|_{{x=0.8}} = {(0.8 - np.tanh(0.8)) / np.tanh(0.8) * 100:.1f}\,\%"]),
        ],
        numeric_steps=[
            ("Regla de Metropolis", "Se mide la energía en unidades de $kT$: un espín $s=\\pm1$ tiene $E = -xs$. Invertirlo cuesta $\\Delta E = 2xs$ (en unidades de $kT$). Si $\\Delta E\\le0$ la inversión se acepta siempre; si no, con probabilidad $e^{-\\Delta E}$.",
             [r"p_{acc} = \min\!\left(1,\ e^{-\Delta E}\right),\qquad \Delta E = 2xs"]),
            ("Un barrido a mano con 10 espines (x = 0.8)", "Se parte de 10 espines al azar (semilla 4). Para cada uno se calcula $\\Delta E$ y $p_{acc}$ y se compara con un número uniforme $u\\in[0,1)$: se acepta si $u<p_{acc}$ (o si $\\Delta E\\le0$).",
             [tab]),
            ("Resultado del barrido reducido", "Se cuentan los espines en cada sentido antes y después del barrido.",
             [rf"\langle s\rangle_{{antes}} = {s0.mean():+.1f},\qquad \langle s\rangle_{{despues}} = {s1.mean():+.1f}\qquad(\tanh0.8 = {np.tanh(0.8):.4f})"]),
            ("Por qué converge", "Cada barrido hace que todos los espines hacia abajo giren (energía baja) y que los de arriba giren con probabilidad $w = e^{-2x}$. La fracción de espines arriba cumple $p' = 1 - w\\,p$, con punto fijo $p_+ = 1/(1+w)$, que da exactamente $\\langle s\\rangle = \\tanh x$.",
             [rf"w = e^{{-2x}} = e^{{-1.6}} = {w:.5f},\qquad p_+ = \dfrac{{1}}{{1+w}} = {p_plus:.5f},\qquad \langle s\rangle = 2p_+-1 = {2 * p_plus - 1:.5f}"]),
            ("Escala del cálculo real", "El código usa $N = 100\\,000$ espines, 80 barridos y promedia la media de espines $\\langle s\\rangle$ de los 40 últimos barridos. El error estadístico es $1/\\sqrt{N\\times40}$.",
             [rf"\sigma = \dfrac{{1}}{{\sqrt{{{n}\cdot{nav}}}}} = {sci(1 / np.sqrt(n * nav), 3)}"]),
            ("Resultados para los cuatro valores de x", "El mismo algoritmo, con las mismas reglas, para cada $x$:",
             [_tab([r"x", r"\tanh x", r"M/M_s\ (\mathrm{MC})", r"\sigma", r"\text{error}"],
                   [[f"{x}", f"{np.tanh(x):.5f}", f"{mm[x][0]:.5f}", f"{mm[x][1]:.5f}", f"{abs(mm[x][0] - np.tanh(x)):.5f}"] for x in xs])]),
            ("Interpretación del error", "La diferencia con el valor teórico es del orden de la incertidumbre estadística $\\sigma$ (algo menor o algo mayor, según la semilla), lo cual confirma que el modelo reproduce $\\tanh x$ sin usar esa fórmula en el algoritmo.",
             [rf"\dfrac{{|M_{{MC}} - \tanh x|}}{{\sigma}}\ \text{{para cada }}x:\quad " + ",\ ".join(f"{abs(mm[x][0] - np.tanh(x)) / mm[x][1]:.2f}" for x in xs)]),
        ],
    )


# =============================================================================== capitulo 13
def _series() -> dict:
    r, l_, c, v0 = m13.R, m13.L, m13.C, m13.V0
    w0 = m13.omega0()
    q = m13.q_factor()
    imax = float(m13.current_amplitude_analytic(w0))
    w1 = 1250.0
    xl, xc = w1 * l_, 1 / (w1 * c)
    z1 = float(m13.impedance(w1))
    i1 = float(m13.current_amplitude_analytic(w1))
    ph1 = float(m13.phase_analytic(w1))
    a0 = m13.steady_state_numeric(w0)
    a1 = m13.steady_state_numeric(w1)
    # primer paso RK4 en w0
    per = 2 * np.pi / w0
    h = per / 400
    f = lambda t, y: np.array([y[1], (v0 * np.sin(w0 * t) - r * y[1] - y[0] / c) / l_])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, 0.0, np.zeros(2), h)
    # deteccion sincrona: ejemplo reducido con 8 muestras por ciclo
    phi = ph1
    th8 = 2 * np.pi * np.arange(8) / 8
    i8 = i1 * np.sin(th8 - phi)
    a8 = 2 / 8 * np.sum(i8 * np.sin(th8))
    b8 = 2 / 8 * np.sum(i8 * np.cos(th8))
    return dict(
        explanation=[
            "Cuando un circuito RLC en serie se alimenta con una fem senoidal $V_0\\sin\\omega t$, tras un transitorio breve la corriente también es senoidal con la misma frecuencia, pero con otra amplitud y desfasada. El inductor y el capacitor se oponen a la corriente alterna de formas distintas: la reactancia inductiva $\\omega L$ crece con la frecuencia y la capacitiva $1/\\omega C$ decrece, y además sus tensiones están en oposición de fase.",
            "La impedancia $Z=\\sqrt{R^2+(\\omega L-1/\\omega C)^2}$ combina las tres oposiciones como si fueran las componentes de un triángulo rectángulo; la amplitud de la corriente es $V_0/Z$ y la fase $\\phi$ cumple $\\tan\\phi=(\\omega L-1/\\omega C)/R$. Si $\\phi>0$ el circuito es inductivo y la corriente atrasa a la tensión; si $\\phi<0$ es capacitivo y la adelanta.",
            "A la frecuencia de resonancia $\\omega_0=1/\\sqrt{LC}$ las dos reactancias se igualan y se cancelan: la impedancia es mínima, igual a $R$, la corriente es máxima y está en fase con la fem. El factor de calidad $Q=\\omega_0L/R$ mide lo aguda que es la resonancia, y también cuánto se amplifican las tensiones en $L$ y en $C$ respecto de la fuente.",
            "Con $R=20\\ \\Omega$, $L=0.10$ H y $C=10\\ \\mu$F la resonancia está en $1000$ rad/s con $Q=5$. Este mismo principio es el que usa un receptor de radio para sintonizar una emisora entre las demás.",
        ],
        analytic_steps=[
            ("Ecuación de la malla con fuente", "La fem se reparte entre el inductor, la resistencia y el capacitor. En régimen estacionario se busca una solución de la forma $i = I_0\\sin(\\omega t-\\phi)$.",
             [r"L\dfrac{di}{dt} + Ri + \dfrac qC = V_0\sin\omega t"]),
            ("Reactancias e impedancia", "Cada elemento reacciona de manera distinta; las amplitudes de las caídas son $I_0R$, $I_0\\omega L$ e $I_0/\\omega C$ y se suman como vectores (fasores) perpendiculares.",
             [r"Z = \sqrt{R^2 + (X_L - X_C)^2},\qquad X_L = \omega L,\quad X_C = \dfrac{1}{\omega C}"]),
            ("Resonancia", "La impedancia es mínima cuando $X_L=X_C$, lo que ocurre a $\\omega_0=1/\\sqrt{LC}$.",
             [rf"\omega_0 = \dfrac{{1}}{{\sqrt{{(0.10)({sci(c, 3)})}}}} = {w0:.3f}\ \mathrm{{rad/s}}\ \ (f_0 = {w0 / (2 * np.pi):.3f}\ \mathrm{{Hz}})"]),
            ("Factor de calidad y corriente máxima", "En la resonancia $Z=R$, por lo que la corriente máxima solo depende de la resistencia.",
             [rf"Q = \dfrac{{\omega_0L}}{{R}} = \dfrac{{({w0:.3f})(0.10)}}{{20}} = {q:.3f},\qquad I_{{max}} = \dfrac{{V_0}}{{R}} = \dfrac{{50}}{{20}} = {imax:.4f}\ \mathrm{{A}}"]),
            ("Fuera de la resonancia: 1250 rad/s", "Se calculan las dos reactancias y la impedancia.",
             [rf"X_L = (1250)(0.10) = {xl:.2f}\ \Omega,\qquad X_C = \dfrac{{1}}{{(1250)({sci(c, 3)})}} = {xc:.2f}\ \Omega",
              rf"Z = \sqrt{{20^2 + ({xl:.2f} - {xc:.2f})^2}} = {z1:.4f}\ \Omega"]),
            ("Amplitud de la corriente a 1250 rad/s", "Se aplica la ley de Ohm para amplitudes.",
             [rf"I_0 = \dfrac{{V_0}}{{Z}} = \dfrac{{50}}{{{z1:.4f}}} = {i1:.5f}\ \mathrm{{A}}"]),
            ("Fase a 1250 rad/s", "Como $X_L>X_C$ la fase es positiva y la corriente atrasa a la tensión.",
             [rf"\phi = \arctan\dfrac{{X_L - X_C}}{{R}} = \arctan\dfrac{{{xl - xc:.2f}}}{{20}} = {ph1:.5f}\ \mathrm{{rad}} = {np.degrees(ph1):.3f}^\circ"]),
        ],
        numeric_steps=[
            ("Formular el problema en el dominio del tiempo", "El código no usa fasores: integra $y=(q,i)$ con $y' = (i,\\ [V_0\\sin\\omega t - Ri - q/C]/L)$ por RK4, con 400 pasos por ciclo durante 25 ciclos de transitorio más 6 de medición (12400 pasos), partiendo de $y=(0,0)$. Se muestra $\\omega = \\omega_0$.",
             [rf"T = \dfrac{{2\pi}}{{\omega_0}} = {per * 1e3:.5f}\ \mathrm{{ms}},\qquad h = \dfrac{{T}}{{400}} = {sci(h, 4)}\ \mathrm{{s}}"]),
            ("Primer paso de RK4", "Con $y_0=(0,0)$ y $\\sin(0)=0$, $k_1=(0,0)$. Los términos siguientes incorporan la fuente en $t = h/2$ y $t = h$.",
             [rf"k_1 = ({sci(k1[0], 3)},\ {sci(k1[1], 3)}),\qquad k_2 = ({sci(k2[0], 4)},\ {sci(k2[1], 4)})",
              rf"k_3 = ({sci(k3[0], 4)},\ {sci(k3[1], 4)}),\qquad k_4 = ({sci(k4[0], 4)},\ {sci(k4[1], 4)})",
              rf"y_1 = \dfrac h6(k_1+2k_2+2k_3+k_4) = ({sci(y1[0], 5)}\ \mathrm{{C}},\ {sci(y1[1], 5)}\ \mathrm{{A}})"]),
            ("Esperar el régimen estacionario", "Durante los primeros 25 ciclos el transitorio decae como $e^{-Rt/2L}$ (aquí $R/2L = 100\\ \\mathrm{s^{-1}}$, y 25 ciclos duran 157 ms), de modo que al terminar la fracción que queda es despreciable.",
             [rf"e^{{-(R/2L)\,t_{{espera}}}} = e^{{-100\cdot{25 * per:.4f}}} = {sci(float(np.exp(-100 * 25 * per)), 2)}"]),
            ("Detección síncrona", "En los últimos 6 ciclos se proyecta $i$ sobre $\\sin\\omega t$ y $\\cos\\omega t$ con Simpson. Si $i=I_0\\sin(\\omega t-\\phi)$, las proyecciones son $a = I_0\\cos\\phi$ y $b=-I_0\\sin\\phi$.",
             [r"a = \dfrac{2}{\Delta t}\int i\sin\omega t\,dt,\qquad b = \dfrac{2}{\Delta t}\int i\cos\omega t\,dt,\qquad I_0 = \sqrt{a^2+b^2},\quad \phi = \mathrm{atan2}(-b,\ a)"]),
            ("Ejemplo reducido de la proyección (8 muestras por ciclo, 1250 rad/s)", "Para ver la mecánica se toma $i_k = I_0\\sin(\\theta_k-\\phi)$ con los valores analíticos en 8 puntos $\\theta_k = 2\\pi k/8$ y se aplican las sumas $a=\\frac28\\sum i_k\\sin\\theta_k$, $b=\\frac28\\sum i_k\\cos\\theta_k$.",
             [rf"i_k = ({', '.join(f'{v:.4f}' for v in i8)})\ \mathrm{{A}}",
              rf"a = {a8:.5f},\quad b = {b8:.5f},\quad I_0 = \sqrt{{a^2+b^2}} = {np.hypot(a8, b8):.5f}\ \mathrm{{A}},\quad \phi = {np.arctan2(-b8, a8):.5f}\ \mathrm{{rad}}"]),
            ("Resultado en resonancia", "Con el cálculo completo del código en $\\omega=\\omega_0$:",
             [rf"I_0^{{num}} = {a0[0]:.5f}\ \mathrm{{A}}\ \ (\text{{analítico }}{imax:.5f}),\qquad \phi^{{num}} = {a0[1]:.3e}\ \mathrm{{rad}}".replace("e-0", r"\times10^{-").replace("e+0", r"\times10^{") + "}" if False else rf"I_0^{{num}} = {a0[0]:.5f}\ \mathrm{{A}}\ \ (\text{{analítico }}{imax:.5f}),\qquad \phi^{{num}} = {sci(a0[1], 3)}\ \mathrm{{rad}}\ \ (\text{{esperado }}0)"]),
            ("Resultado fuera de resonancia", "Repitiendo todo el cálculo con $\\omega = 1250$ rad/s:",
             [rf"I_0^{{num}} = {a1[0]:.5f}\ \mathrm{{A}}\ \ (\text{{analítico }}{i1:.5f}),\qquad \phi^{{num}} = {a1[1]:.5f}\ \mathrm{{rad}}\ \ (\text{{analítico }}{ph1:.5f})",
              rf"\text{{errores relativos: }}{sci(abs(a0[0] - imax) / imax, 2)},\ \ {sci(abs(a1[1] - ph1) / ph1, 2)}"]),
        ],
    )


def _power() -> dict:
    vp, ip, phd = m13.V_PEAK, m13.I_PEAK, m13.PHI_DEG
    phi = np.deg2rad(phd)
    pa = m13.power_average_analytic()
    pn = m13.power_average_numeric()
    vr, ir = vp / np.sqrt(2), ip / np.sqrt(2)
    vrn = m13.rms_numeric(vp)
    ta, tn = m13.transformer_analytic(), m13.transformer_numeric()
    # Simpson reducido con 8 subintervalos del producto v*i en un periodo normalizado
    t8 = np.linspace(0, 1.0, 9)
    v8 = vp * np.sin(2 * np.pi * t8)
    i8 = ip * np.sin(2 * np.pi * t8 - phi)
    p8 = v8 * i8
    s8 = simpson(p8, t8)
    # transformador: primera muestra
    tt = np.linspace(0, 1.0, 4001)
    vpp = np.sqrt(2) * 120.0 * np.sin(2 * np.pi * tt)
    ipp = np.sqrt(2) * 2.0 * np.sin(2 * np.pi * tt)
    vs = vpp * m13.N_S / m13.N_P
    is_ = ipp * m13.N_P / m13.N_S
    return dict(
        explanation=[
            "En corriente alterna la tensión y la corriente son senoides de la misma frecuencia, pero no necesariamente en fase. La potencia instantánea $p=vi$ oscila: es positiva cuando el circuito toma energía de la fuente y negativa cuando la devuelve. Lo que interesa en la práctica es su promedio en un periodo.",
            "Si $v=V_0\\sin\\omega t$ e $i=I_0\\sin(\\omega t-\\phi)$, el producto contiene una parte constante $\\tfrac12V_0I_0\\cos\\phi$ y una parte que oscila al doble de frecuencia y se promedia a cero. La potencia media es entonces $\\tfrac12V_0I_0\\cos\\phi = V_{rms}I_{rms}\\cos\\phi$, donde $\\cos\\phi$ es el factor de potencia: solo la parte de la corriente que está en fase con la tensión realiza trabajo neto.",
            "El valor eficaz de una magnitud senoidal es su máximo dividido entre $\\sqrt2$. Con él, las fórmulas de potencia tienen la misma apariencia que en corriente continua, y es por eso por lo que los voltajes de las redes domésticas (120 V, 230 V) se dan como valores eficaces.",
            "Un transformador ideal usa el flujo variable en un núcleo común a dos bobinas para cambiar la amplitud de la tensión sin pérdidas. La razón de tensiones es la razón de vueltas, $V_s/V_p = N_s/N_p$, y como no disipa energía, $V_pI_p=V_sI_s$: lo que se gana en tensión se pierde en corriente, y viceversa.",
        ],
        analytic_steps=[
            ("Potencia instantánea", "Se multiplican las dos senoides y se usa la identidad $\\sin A\\sin B = \\tfrac12[\\cos(A-B)-\\cos(A+B)]$.",
             [r"p = V_0I_0\sin\omega t\,\sin(\omega t-\phi) = \tfrac12V_0I_0\left[\cos\phi - \cos(2\omega t-\phi)\right]"]),
            ("Promedio en un periodo", "El segundo término tiene promedio nulo, así que solo sobrevive el primero.",
             [r"\bar P = \tfrac12V_0I_0\cos\phi"]),
            ("Sustituir los datos", "Con $V_0 = 170$ V, $I_0 = 8.0$ A y $\\phi = 40^\\circ$:",
             [rf"\bar P = \tfrac12(170)(8.0)\cos40^\circ = {0.5 * vp * ip:.1f}\,({np.cos(phi):.5f}) = {pa:.4f}\ \mathrm{{W}}"]),
            ("Valores eficaces", "Se dividen los máximos entre $\\sqrt2$.",
             [rf"V_{{rms}} = \dfrac{{170}}{{\sqrt2}} = {vr:.4f}\ \mathrm{{V}},\qquad I_{{rms}} = \dfrac{{8.0}}{{\sqrt2}} = {ir:.4f}\ \mathrm{{A}}",
              rf"\bar P = V_{{rms}}I_{{rms}}\cos\phi = ({vr:.4f})({ir:.4f})({np.cos(phi):.5f}) = {vr * ir * np.cos(phi):.4f}\ \mathrm{{W}}"]),
            ("Transformador ideal: tensión del secundario", "La razón de tensiones es la razón de vueltas.",
             [rf"V_s = V_p\dfrac{{N_s}}{{N_p}} = (120\ \mathrm{{V}})\dfrac{{{m13.N_S}}}{{{m13.N_P}}} = {ta['vs']:.2f}\ \mathrm{{V}}"]),
            ("Corriente del secundario", "En el transformador ideal se conserva la potencia, $V_pI_p = V_sI_s$, de donde la corriente cambia en la razón inversa.",
             [rf"I_s = I_p\dfrac{{N_p}}{{N_s}} = (2.0\ \mathrm{{A}})\dfrac{{{m13.N_P}}}{{{m13.N_S}}} = {ta['is']:.2f}\ \mathrm{{A}}"]),
            ("Verificar el balance de potencia", "Las potencias aparentes de primario y secundario coinciden.",
             [rf"V_pI_p = (120)(2.0) = {120 * 2.0:.1f}\ \mathrm{{W}},\qquad V_sI_s = ({ta['vs']:.2f})({ta['is']:.2f}) = {ta['vs'] * ta['is']:.1f}\ \mathrm{{W}}"]),
        ],
        numeric_steps=[
            ("Regla de Simpson para el promedio", "El promedio de $p(t)$ en un periodo se calcula como $\\frac1T\\int_0^Tp\\,dt$ con Simpson, en tiempo normalizado $t\\in[0,1]$ (un periodo). Se muestra primero con 8 subintervalos, $h = 1/8$.",
             [r"\int_0^1p\,dt\approx\dfrac h3\left[p_0+p_8+4(p_1+p_3+p_5+p_7)+2(p_2+p_4+p_6)\right]"]),
            ("Muestras de la potencia instantánea", "Se evalúan $v_j=170\\sin2\\pi t_j$ e $i_j = 8.0\\sin(2\\pi t_j-40^\\circ)$ y su producto.",
             [_tab([r"j", r"t_j", r"v_j\ [\mathrm{V}]", r"i_j\ [\mathrm{A}]", r"p_j\ [\mathrm{W}]"],
                   [[str(j), f"{t8[j]:.3f}", f"{v8[j]:.3f}", f"{i8[j]:.4f}", f"{p8[j]:.3f}"] for j in range(9)])]),
            ("Suma con 8 subintervalos", "Se aplica la regla a esos nueve valores.",
             [rf"\bar P \approx \dfrac{{1/8}}{{3}}\left[{p8[0]:.3f}+{p8[8]:.3f}+4({p8[1] + p8[3] + p8[5] + p8[7]:.3f})+2({p8[2] + p8[4] + p8[6]:.3f})\right] = {s8:.4f}\ \mathrm{{W}}",
              rf"\text{{analítico: }}{pa:.4f}\ \mathrm{{W}}"]),
            ("Con el número de intervalos real", "El código usa 4000 subintervalos; el resultado coincide con el analítico casi hasta el redondeo.",
             [rf"\bar P_{{num}} = {pn:.5f}\ \mathrm{{W}},\qquad \text{{error relativo}} = {sci(abs(pn - pa) / pa, 2)}"]),
            ("Valor eficaz numérico", "Se integra $v^2$ en un periodo con la misma regla y se toma la raíz: $V_{rms} = \\sqrt{\\int_0^1v^2dt}$.",
             [rf"V_{{rms}}^{{num}} = {vrn:.5f}\ \mathrm{{V}},\qquad V_{{rms}}^{{analit}} = {vr:.5f}\ \mathrm{{V}},\qquad \text{{error}} = {sci(abs(vrn - vr) / vr, 2)}"]),
            ("Transformador: muestrear las ondas", "Se genera un periodo del primario, $v_p = \\sqrt2(120)\\sin2\\pi t$ e $i_p = \\sqrt2(2.0)\\sin2\\pi t$ en 4001 puntos, y las del secundario por la razón de vueltas: $v_s = v_pN_s/N_p$, $i_s = i_pN_p/N_s$.",
             [rf"\text{{ejemplo en }}t=0.25:\ \ v_p = {vpp[1000]:.4f}\ \mathrm{{V}},\ \ v_s = {vs[1000]:.4f}\ \mathrm{{V}},\ \ i_p = {ipp[1000]:.4f}\ \mathrm{{A}},\ \ i_s = {is_[1000]:.4f}\ \mathrm{{A}}"]),
            ("Eficaces del secundario por Simpson", "Se integra el cuadrado de cada onda del secundario y se extrae la raíz.",
             [rf"V_s^{{num}} = \sqrt{{\int_0^1 v_s^2\,dt}} = {tn['vs']:.5f}\ \mathrm{{V}},\qquad I_s^{{num}} = \sqrt{{\int_0^1 i_s^2\,dt}} = {tn['is']:.5f}\ \mathrm{{A}}"]),
            ("Comprobación de potencia", "Se integra $v_pi_p$ y $v_si_s$ y se comparan: el transformador ideal debe conservarlas.",
             [rf"P_p = {tn['p_p']:.5f}\ \mathrm{{W}},\qquad P_s = {tn['p_s']:.5f}\ \mathrm{{W}},\qquad P_p-P_s = {sci(tn['p_p'] - tn['p_s'], 2)}\ \mathrm{{W}}"]),
        ],
    )


DETAIL = {
    "BiotSavartLoop": _loop(),
    "SolenoidAxis": _solenoid(),
    "RotatingLoop": _rotating(),
    "SlidingRod": _sliding(),
    "RLCircuit": _rl(),
    "RLCOscillator": _rlc(),
    "DipoleTorque": _dipole(),
    "ParamagnetCurie": _curie(),
    "SeriesRLC": _series(),
    "PowerTransformer": _power(),
}
