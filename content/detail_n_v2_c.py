"""Explicacion del metodo numerico (intro, codigo, convergencia): capitulos 9 a 13 del volumen 2.

Escenas: BiotSavartLoop, SolenoidAxis, RotatingLoop, SlidingRod, RLCircuit,
RLCOscillator, DipoleTorque, ParamagnetCurie, SeriesRLC, PowerTransformer.
Todas las cifras de los estudios de error se calculan ejecutando las funciones de `physics/`
(o replicando su algoritmo con otro parametro de discretizacion).
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
from physics.constants import MU0
from physics.integrators import euler_step, integrate, rk4_step

EPS_MACH = float(np.finfo(float).eps)


# ------------------------------------------------------------------------------ utilidades
def _tab(header: list[str], rows: list[list[str]]) -> str:
    """Tabla TeX alineada a la derecha con encabezado."""
    out = [r"\begin{array}{" + "r" * len(header) + "}", " & ".join(header) + r" \\ \hline"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out.append(r"\end{array}")
    return "\n".join(out)


def _m(x: float, d: int = 2) -> str:
    """Numero en notacion cientifica listo para prosa (entre signos $)."""
    return f"${sci(x, d)}$"


def _slopes(xs, errs) -> list[str]:
    """Pendiente observada p = ln(e_prev/e)/ln(x/x_prev) entre filas consecutivas (x crece al refinar)."""
    out = ["--"]
    for i in range(1, len(xs)):
        e0, e1 = errs[i - 1], errs[i]
        out.append(f"{np.log(e0 / e1) / np.log(xs[i] / xs[i - 1]):.2f}" if e0 > 0 and e1 > 0 else "--")
    return out


# =============================================================================== capitulo 9
def _loop() -> dict:
    a, cur = m9.A_LOOP, m9.I
    b0 = float(m9.loop_axis_analytic(0.0))
    b5 = float(m9.loop_axis_analytic(0.05))
    ns = [8, 16, 32, 64, 128]
    errs = [abs(m9.loop_axis_numeric(0.0, n=n) - b0) / b0 for n in ns]
    pred = [5 / 6 * (np.pi / n) ** 2 for n in ns]
    rows = [[str(n), sci(e, 3), sci(p, 3), s] for n, e, p, s in zip(ns, errs, pred, _slopes(ns, errs))]
    wl = m9.wire_field_analytic()
    nw = [20, 40, 80, 160, 320]
    ew = [abs(m9.wire_field_numeric(n=n) - wl) / wl for n in nw]
    rows_w = [[str(n), sci(e, 3), s] for n, e, s in zip(nw, ew, _slopes(nw, ew))]
    n720 = m9.loop_axis_numeric(0.0)
    e720 = abs(n720 - b0) / b0
    e720_5 = abs(m9.loop_axis_numeric(0.05) - b5) / b5
    e2000 = abs(m9.wire_field_numeric() - wl) / wl
    n_need = int(np.ceil(np.pi * np.sqrt(5 / 6 / 1e-5)))
    return dict(
        numeric_intro=[
            r"La ley de Biot-Savart expresa el campo como una integral de línea sobre la corriente. Para la espira y el hilo existe una primitiva (por eso hay solución analítica), pero para una curva arbitraria casi nunca la hay: el método numérico sustituye la curva por $N$ segmentos rectos y la integral por una suma finita. La ventaja es que el mismo código sirve para cualquier forma de conductor, y por eso aquí sirve también como comprobación de la fórmula cerrada.",
            r"La regla es $\vec B(\vec p)\approx \dfrac{\mu_0 I}{4\pi}\sum_{j=1}^{N}\dfrac{\Delta\vec\ell_j\times\vec r_j}{|\vec r_j|^3}$, donde $\Delta\vec\ell_j$ es la cuerda $j$ y $\vec r_j$ va de su punto medio al punto de observación $\vec p$. Es la regla del punto medio aplicada a la integral vectorial: cada tramo se aproxima por el valor del integrando en su centro multiplicado por su longitud. El producto vectorial se calcula componente a componente con `np.cross`, sin usar ninguna simetría.",
            r"El error de la regla del punto medio es de orden $O(h^2)$, con $h$ la longitud del tramo; para la espira $h=2\pi a/N$, de modo que al duplicar $N$ el error cae cuatro veces. De hecho, en el centro de la espira la suma tiene forma cerrada y da $B_{num}/B = (N/\pi)\sin(\pi/N)/\cos^2(\pi/N)\approx 1+\tfrac56(\pi/N)^2$, que se compara abajo con lo medido. El método falla cuando el punto de observación está muy cerca del conductor (el integrando varía en una escala menor que $h$) y si el circuito no es cerrado el resultado deja de tener sentido físico.",
            r"Los parámetros se eligieron para que la discretización quede muy por debajo de cualquier otra incertidumbre: $N=720$ (un segmento cada $0.5^\circ$) en la espira y 2000 tramos de $0.15$ mm en el hilo, con un costo de apenas miles de productos vectoriales. El resultado significa que el campo de las 720 cuerdas coincide con $\mu_0I/2a$ en unas cinco cifras. El método es independiente del analítico porque nunca usa $\cos\theta$, la simetría axial ni la primitiva $z/\sqrt{d^2+z^2}$: solo suma vectores.",
        ],
        numeric_code="""# Biot-Savart por suma de segmentos rectos (physics/v2ch09_biot_savart.py)
mu0, I, a = 4e-7*pi, 3.0, 0.05
def loop_segments(a, n):                       # n cuerdas inscritas en la circunferencia
    phi = linspace(0, 2*pi, n+1)               # n+1 angulos igualmente espaciados
    P = [a*cos(phi), a*sin(phi), 0*phi]        # vertices de la espira (plano z = 0)
    return pares(P[:-1], P[1:])                # segmento j va del vertice j al j+1
def biot_savart(p, segs, I):
    mid = (segs[:,0] + segs[:,1]) / 2          # punto medio de cada cuerda
    dl  = segs[:,1] - segs[:,0]                # vector dl_j de cada cuerda
    r   = p - mid                              # r_j del tramo al punto de observacion
    d   = norma(r)                             # |r_j|
    return mu0*I/(4*pi) * suma(cruz(dl, r)/d**3)   # suma vectorial de dl x r / r^3
N = 720                                        # un tramo cada 0.5 grados
Bz = biot_savart([0, 0, z], loop_segments(a, N), I)[2]   # componente axial en (0,0,z)
# hilo: igual, con 2000 tramos entre z = -L/2 y +L/2 y p = (d, 0, 0)
err = abs(Bz - B_analitico) / B_analitico      # solo para comparar; no entra al calculo""",
        numeric_converge=[
            ("Error relativo contra el número de segmentos", "Se recalcula $B_z(0)$ para $N$ creciente y se compara con $\\mu_0I/2a$. La columna de la derecha es la predicción $\\tfrac56(\\pi/N)^2$ del desarrollo de la suma. La pendiente $p=\\ln(e_{N/2}/e_N)/\\ln 2$ se acerca a 2, el orden del punto medio.",
             [_tab([r"N", r"\text{error medido}", r"\tfrac56(\pi/N)^2", r"p"], rows),
              rf"N=720:\quad \dfrac{{|B_{{num}}-B|}}{{B}} = {sci(e720, 3)}\ \text{{ (centro)}},\qquad {sci(e720_5, 3)}\ \text{{ (}}z=5\ \mathrm{{cm}}\text{{)}}"]),
            ("Error del hilo finito con el número de tramos", "Mismo estudio para el hilo de 30 cm. El tramo del hilo es $h=L/n$, y de nuevo el error baja como $h^2$ una vez que $h$ es menor que la distancia $d=4$ cm; con pocos tramos todavía no es asintótico.",
             [_tab([r"n", r"\text{error relativo}", r"p"], rows_w),
              rf"n=2000:\quad \dfrac{{|B_{{num}}-B|}}{{B}} = {sci(e2000, 3)}"]),
            ("Fuentes de error y criterio de confiabilidad", f"El error es de truncamiento (discretización por cuerdas): para $N=720$ vale {_m(e720, 2)}. El de redondeo es del orden de $\\varepsilon_{{mach}}\\sqrt{{N}}\\approx {sci(EPS_MACH * np.sqrt(720), 2)}$, muchísimo menor, de modo que aumentar $N$ sigue mejorando el resultado hasta bajar a ese nivel. Como criterio práctico, el resultado es confiable cuando al duplicar $N$ el cambio relativo es menor que la tolerancia; para $10^{{-5}}$ en el centro bastan $N\\ge {n_need}$ segmentos.",
             [rf"N_{{min}}\approx \pi\sqrt{{\tfrac56/\mathrm{{tol}}}} = \pi\sqrt{{0.8333/10^{{-5}}}} \approx {n_need}"]),
        ],
    )


def _solenoid() -> dict:
    n_t, ln, r, cur = m9.N_TURNS, m9.L_SOL, m9.R_SOL, m9.I_SOL
    be = float(m9.solenoid_axis_exact(0.0))
    sg = [8, 16, 32, 60, 120]
    es = [abs(m9.solenoid_axis_numeric(0.0, seg=s) - be) / be for s in sg]
    rows_s = [[str(s), sci(e, 3), p] for s, e, p in zip(sg, es, _slopes(sg, es))]
    nr = [5, 10, 20, 50, 100]
    er = [abs(m9.solenoid_axis_numeric(0.0, n=k, current=cur * n_t / k) - be) / be for k in nr]
    rows_r = [[str(k), sci(ln / k * 1e3, 3), sci(e, 3)] for k, e in zip(nr, er)]
    ez = abs(m9.solenoid_axis_numeric(0.25) - float(m9.solenoid_axis_exact(0.25))) / float(m9.solenoid_axis_exact(0.25))
    e60 = es[3]
    return dict(
        numeric_intro=[
            r"Un solenoide de 500 vueltas con radio 2 cm y largo 50 cm no tiene una fórmula simple para todo punto, pero sobre el eje sí se obtiene una expresión cerrada integrando espiras. El método numérico evita esa integración: modela cada vuelta como un polígono de 60 lados y suma con Biot-Savart los $500\times60=30\,000$ segmentos. Es la misma técnica que en la espira, solo que con muchas espiras apiladas, y funcionaría igual con vueltas desiguales o un devanado inclinado.",
            r"La regla es $\vec B\approx \dfrac{\mu_0 I}{4\pi}\sum_{k=1}^{500}\sum_{j=1}^{60}\dfrac{\Delta\vec\ell_{kj}\times\vec r_{kj}}{|\vec r_{kj}|^3}$, con las vueltas centradas en $z_k=-L/2+(k-\tfrac12)L/500$. Aquí hay dos discretizaciones distintas: la de cada vuelta en cuerdas (regla del punto medio, error $O(h^2)$ con $h=2\pi r/\mathrm{seg}$) y la del devanado en vueltas separadas $L/500=1$ mm, que es real, no una aproximación. La fórmula analítica, en cambio, supone una lámina continua de corriente.",
            r"Por eso el error total tiene dos fuentes que se estudian por separado más abajo. La del polígono decrece como $1/\mathrm{seg}^2$ y es la dominante con 60 lados. La de la separación entre vueltas es la regla del punto medio en $z$: es despreciable mientras la separación sea menor que el radio, y por eso la lámina continua es una buena idealización. El método falla si se pide el campo a una distancia de un hilo menor que el lado del polígono.",
            r"La elección $\mathrm{seg}=60$ equilibra costo y precisión: cada punto del eje cuesta 30 000 productos vectoriales (decenas de milisegundos en NumPy) y el error queda en el orden de $10^{-3}$, más que suficiente para comparar con el solenoide infinito, que difiere en un 0.3 %. El método es independiente del analítico porque este último usa la integral de $\cos\alpha$ sobre una lámina continua, y aquel jamás calcula ángulos: solo suma productos vectoriales vuelta por vuelta.",
        ],
        numeric_code="""# Solenoide finito sobre el eje (physics/v2ch09_biot_savart.py)
N, L, r, I, seg = 500, 0.5, 0.02, 2.0, 60
zs = linspace(-L/2 + L/(2*N), L/2 - L/(2*N), N)      # centro de cada vuelta (punto medio en z)
phi = linspace(0, 2*pi, seg+1)                       # vertices del poligono de cada vuelta
anillo = [r*cos(phi), r*sin(phi), 0*phi]             # una vuelta de radio r en z = 0
vertices = [anillo + (0, 0, z_k) for z_k in zs]      # N poligonos desplazados a lo largo del eje
segs = pares_consecutivos(vertices)                  # N*seg = 30000 segmentos rectos
def B_eje(z):
    p = (0, 0, z)                                    # punto de observacion sobre el eje
    mid = (segs[:,0] + segs[:,1]) / 2                # punto medio de cada segmento
    dl = segs[:,1] - segs[:,0]                       # vector del segmento
    rr = p - mid; d = norma(rr)                      # vector al punto y su modulo
    return mu0*I/(4*pi) * suma(cruz(dl, rr)/d**3)[2] # solo se conserva la componente axial
Bc, Be = B_eje(0.0), B_eje(0.25)                     # centro y extremo del solenoide
err = abs(Bc - B_exacto) / B_exacto                  # comparacion final con la formula con cos(alfa)""",
        numeric_converge=[
            ("Error contra el número de lados por vuelta", "Se fija el devanado (500 vueltas) y se aumenta el número de lados. El error relativo en el centro cae como $1/\\mathrm{seg}^2$ (pendiente cercana a 2), el orden de la regla del punto medio aplicada a cada polígono.",
             [_tab([r"\mathrm{seg}", r"\text{error relativo}", r"p"], rows_s),
              rf"\mathrm{{seg}}=60:\quad \dfrac{{|B_{{num}}-B|}}{{B}} = {sci(e60, 3)}\ \text{{ en el centro}},\qquad {sci(ez, 3)}\ \text{{ en el extremo}}"]),
            ("Error por representar la corriente con vueltas discretas", f"Para aislar el efecto de la separación entre vueltas se usan $k$ espiras con corriente $500I/k$ (mismo $nI$ por unidad de largo) y 60 lados. Mientras el paso $L/k$ sea comparable al radio ($r={fx(r * 1e3, 0)}$ mm $={fx(r * 100, 0)}$ cm) el error es grande, y cae muy rápido al hacer el paso menor que $r$; para $k\\ge50$ ya queda tapado por el error del polígono (cerca de $9\\times10^{{-4}}$), que no depende de $k$.",
             [_tab([r"k", r"L/k\ [\mathrm{mm}]", r"\text{error relativo}"], rows_r)]),
            ("Fuentes de error y criterio", f"Con 500 vueltas reales (paso 1 mm) el error por separación es despreciable frente al del polígono, que domina con {_m(e60, 2)}. El redondeo acumulado en 30 000 sumas es del orden de $10^{{-13}}$. El resultado es confiable cuando se cumplen dos condiciones: el paso entre vueltas es menor que el radio y, al duplicar los lados, el valor cambia menos que la tolerancia deseada; con la ley $p\\approx2$ basta extrapolar: el error de 120 lados es cuatro veces menor que el de 60.",
             [rf"e_{{120}}/e_{{60}} = {fx(es[4] / es[3], 3)}\ \approx\ \tfrac14 = 0.250"]),
        ],
    )


# =============================================================================== capitulo 10
def _rotating() -> dict:
    t0 = 0.002
    e_an = float(m10.emf_analytic(t0))
    phi0 = m10.N_COIL * m10.B0 * m10.AREA
    om = m10.OMEGA
    hs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    rows = []
    for h in hs:
        e = abs(float(m10.emf_numeric(np.array([t0]), h)[0]) - e_an)
        tr = h**2 / 6 * phi0 * om**3 * abs(np.sin(om * t0))
        rd = EPS_MACH * phi0 / h
        rows.append([sci(h, 1), sci(e, 3), sci(tr, 3), sci(rd, 3)])
    errs = [abs(float(m10.emf_numeric(np.array([t0]), h)[0]) - e_an) for h in hs]
    h_opt = (3 * EPS_MACH / om**3) ** (1 / 3)
    e_def = abs(float(m10.emf_numeric(np.array([t0]))[0]) - e_an)
    ns = [2, 4, 6, 8, 16, 4000]
    rms_ex = m10.emf_peak_analytic() / np.sqrt(2)
    er = [abs(m10.emf_rms_numeric(n) - rms_ex) for n in ns]
    rows_r = [[str(n), sci(e, 3) if e > 0 else "0"] for n, e in zip(ns, er)]
    pend = np.log(errs[0] / errs[3]) / np.log(hs[0] / hs[3])
    return dict(
        numeric_intro=[
            r"La fem es la derivada temporal del flujo, $\varepsilon=-d\Phi/dt$. Aquí la derivada tiene forma cerrada, pero en muchos generadores reales el flujo solo se conoce como una tabla de valores o por simulación, y entonces hay que derivar numéricamente. El método reemplaza la derivada por un cociente de diferencias del flujo calculado en dos instantes cercanos, y el valor eficaz por una integral numérica sobre un periodo.",
            r"La derivada se aproxima con la diferencia central $\varepsilon(t)\approx -\dfrac{\Phi(t+h)-\Phi(t-h)}{2h}$. Al desarrollar $\Phi$ en serie de Taylor, los términos pares se cancelan y queda $\varepsilon_{num}=\varepsilon-\tfrac{h^2}{6}\,\Phi'''(t)+O(h^4)$: el error de truncamiento es $O(h^2)$. El valor eficaz usa la regla de Simpson, $\int f\,dt\approx\tfrac h3[f_0+4f_1+2f_2+\dots+f_n]$, de orden $O(h^4)$ en general.",
            r"Derivar numéricamente es una operación mal condicionada: al achicar $h$ el truncamiento baja como $h^2$, pero el redondeo sube como $\varepsilon_{mach}\,\Phi/h$ porque se restan dos números casi iguales. Existe un paso óptimo $h_{opt}\sim(3\varepsilon_{mach}/\omega^3)^{1/3}$ a partir del cual refinar empeora el resultado. Con $h=1\ \mu\mathrm{s}$ (el valor del código) se está bien por encima de ese mínimo, en la zona donde domina el truncamiento.",
            r"El resultado de $h=1\ \mu\mathrm{s}$ equivale a muestrear el flujo 20 000 veces por periodo, y da la fem con un error relativo cercano a $10^{-8}$ frente a los 314 V de amplitud. El método es independiente del analítico porque solo usa valores de $\Phi(t\pm h)$ y no deriva el coseno a mano; la comprobación del eficaz con Simpson es independiente de la identidad $\varepsilon_0/\sqrt2$, que viene de promediar $\sin^2$.",
        ],
        numeric_code="""# fem por diferencia central y valor eficaz por Simpson (physics/v2ch10_faraday.py)
N, B, A, f = 100, 0.5, 0.02, 50.0
w = 2*pi*f
def flujo(t):  return N*B*A*cos(w*t)           # solo el flujo; la derivada NO se escribe a mano
def fem(t, h=1e-6):
    return -(flujo(t+h) - flujo(t-h)) / (2*h)  # diferencia central: error O(h^2)
def fem_rms(n=4000):
    t = linspace(0, 1/f, n+1)                  # exactamente un periodo
    e = fem_analitica_muestreada(t)            # onda sinusoidal de la fem
    return sqrt(simpson(e**2, t) * f)          # raiz de (1/T) * integral de e^2 dt
def simpson(y, x):
    n = len(x) - 1; h = (x[-1]-x[0]) / n       # n debe ser par
    return h/3 * (y[0] + y[-1] + 4*suma(y[1:-1:2]) + 2*suma(y[2:-1:2]))  # pesos 1,4,2,4,...,1
e5 = fem(0.005)                                # fem a 5 ms
Vrms = fem_rms()                               # valor eficaz sobre un periodo""",
        numeric_converge=[
            ("Error de la derivada contra el paso h", f"Se evalúa $\\varepsilon$ en $t={fx(t0 * 1e3, 0)}$ ms variando $h$. Para $h\\ge10^{{-6}}$ s el error medido coincide con el truncamiento teórico $\\tfrac{{h^2}}{{6}}\\Phi_0\\omega^3|\\sin\\omega t|$ y la pendiente entre $10^{{-3}}$ y $10^{{-6}}$ s es {fx(pend, 2)}. Para $h$ pequeño el redondeo $\\varepsilon_{{mach}}\\Phi_0/h$ toma el control.",
             [_tab([r"h\ [\mathrm{s}]", r"\text{error medido}\ [\mathrm{V}]", r"\text{truncamiento}", r"\text{redondeo}"], rows)]),
            ("Paso óptimo y valor del código", f"El mínimo del error total ocurre donde truncamiento y redondeo se igualan, en $h_{{opt}}\\approx{sci(h_opt, 2)}$ s. El código usa $h=10^{{-6}}$ s, con error absoluto {_m(e_def, 3)} V sobre un máximo de {fx(m10.emf_peak_analytic(), 2)} V, es decir un error relativo de {_m(e_def / m10.emf_peak_analytic(), 2)}.",
             [rf"h_{{opt}}\approx\left(\dfrac{{3\varepsilon_{{mach}}}}{{\omega^3}}\right)^{{1/3}} = \left(\dfrac{{3({sci(EPS_MACH, 3)})}}{{({sci(om**3, 3)})}}\right)^{{1/3}} = {sci(h_opt, 3)}\ \mathrm{{s}}"]),
            ("Valor eficaz: Simpson converge casi de inmediato", f"El integrando $\\varepsilon^2$ es un polinomio trigonométrico y se integra sobre exactamente un periodo; con pocos puntos Simpson ya es exacto salvo redondeo. Con 4 o 2 subintervalos no basta, pero desde $n=6$ el error es del orden de $10^{{-14}}$ V: el valor eficaz es confiable siempre que el intervalo sea un número entero de periodos.",
             [_tab([r"n", r"|V_{rms}^{num}-\varepsilon_0/\sqrt2|\ [\mathrm{V}]"], rows_r)]),
        ],
    )


def _sliding() -> dict:
    tau = m10.tau_rod()
    k = m10.B_ROD**2 * m10.L_ROD**2 / (m10.M_ROD * m10.R_CIRC)
    ex = float(m10.rod_velocity_analytic(12.0))
    ns = [12, 24, 48, 96, 192]
    rhs = lambda _t, y: np.array([y[1], -k * y[1]])  # noqa: E731
    e_rk, e_eu = [], []
    for n in ns:
        _, y = integrate(rhs, np.array([0.0, m10.V0_ROD]), 0.0, 12.0, n, rk4_step)
        e_rk.append(abs(y[-1, 1] - ex))
        _, y = integrate(rhs, np.array([0.0, m10.V0_ROD]), 0.0, 12.0, n, euler_step)
        e_eu.append(abs(y[-1, 1] - ex))
    rows = [[str(n), sci(h, 2), sci(a, 3), pa, sci(b, 3), pb] for n, h, a, pa, b, pb in
            zip(ns, [12.0 / n for n in ns], e_rk, _slopes(ns, e_rk), e_eu, _slopes(ns, e_eu))]
    e_ex = 0.5 * m10.M_ROD * m10.V0_ROD**2
    nq = [20, 40, 80, 160, 320]
    eq = [abs(m10.rod_energy_dissipated_numeric(25.0, n) - e_ex) for n in nq]
    rows_q = [[str(n), sci(e, 3), p] for n, e, p in zip(nq, eq, _slopes(nq, eq))]
    te = [5.0, 10.0, 15.0, 25.0]
    rows_t = [[fx(t, 0), sci(abs(m10.rod_energy_dissipated_numeric(t, 10000) - e_ex), 3), sci(e_ex * np.exp(-2 * t / tau), 3)] for t in te]
    n_def = 2400
    _, yd = m10.rod_numeric(12.0, n_def)
    e_def = abs(yd[-1, 1] - ex)
    return dict(
        numeric_intro=[
            r"El frenado magnético de la barra obedece $m\,dv/dt=-(B^2\ell^2/R)\,v$, que sí se resuelve a mano, pero el integrador numérico no distingue entre ésta y una versión más difícil con fricción, cable inductivo o campo no uniforme. Por eso se usa Runge-Kutta de cuarto orden (RK4) sobre el sistema de primer orden $y=(x,v)$, y Simpson para integrar la potencia disipada $I^2R$ y comprobar la conservación de la energía.",
            r"RK4 avanza con $y_{n+1}=y_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$, donde $k_1=f(t_n,y_n)$, $k_2=f(t_n+\tfrac h2,\,y_n+\tfrac h2k_1)$, $k_3=f(t_n+\tfrac h2,\,y_n+\tfrac h2k_2)$ y $k_4=f(t_n+h,\,y_n+hk_3)$. Es una media ponderada de cuatro pendientes que reproduce la serie de Taylor de la solución hasta $h^4$, de modo que el error local es $O(h^5)$ y el global $O(h^4)$. Simpson integra con el mismo orden, $O(h^4)$.",
            r"Para esta ecuación lineal $y'=-ky$ RK4 equivale a multiplicar en cada paso por el polinomio $1-kh+\tfrac{(kh)^2}{2}-\tfrac{(kh)^3}{6}+\tfrac{(kh)^4}{24}$, que aproxima a $e^{-kh}$; con $k=1/\tau=0.4\ \mathrm{s^{-1}}$ y $h=5$ ms el producto $kh=0.002$ es diminuto y el error de truncamiento es mínimo. El método se vuelve inestable si $kh>2.78$; por debajo de eso, el error se controla con $h$.",
            r"Los parámetros del código (2400 pasos en 12 s, $h=5$ ms; 10 000 pasos en 25 s para la energía) hacen que $h\ll\tau=2.5$ s. El tiempo final de 25 s equivale a 10 constantes de tiempo, suficiente para que solo quede sin disipar una fracción $e^{-20}\approx2\times10^{-9}$ de la energía. El método es independiente del analítico: nunca usa $e^{-t/\tau}$; integra la ecuación diferencial y, para la energía, suma $I^2R\,\Delta t$ con la velocidad numérica.",
        ],
        numeric_code="""# Barra deslizante: RK4 para (x, v) y Simpson para la energia (physics/v2ch10_faraday.py)
B, l, m, R, v0 = 0.8, 0.5, 0.2, 2.0, 4.0
k = B**2 * l**2 / (m * R)                      # k = 1/tau = 0.4 1/s
f = lambda t, y: [y[1], -k*y[1]]               # y = (x, v):  x' = v,  v' = -k v
def rk4(f, t, y, h):
    k1 = f(t, y)                               # pendiente al inicio del paso
    k2 = f(t + h/2, y + h/2*k1)                # pendiente en el punto medio (con k1)
    k3 = f(t + h/2, y + h/2*k2)                # pendiente en el punto medio (con k2)
    k4 = f(t + h,   y + h*k3)                  # pendiente al final del paso
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)     # promedio ponderado 1-2-2-1
t = linspace(0, 12, 2401);  h = 12/2400        # 2400 pasos de 5 ms
y = [[0, v0]]
for i in range(2400):
    y.append(rk4(f, t[i], y[-1], h))           # se avanza el estado completo
# energia: integral de P = I^2 R con I = B*l*v/R, hasta 25 s con 10000 pasos
P = (B*l*v/R)**2 * R                           # potencia disipada en cada instante
E = simpson(P, t25)                            # debe dar (1/2) m v0^2""",
        numeric_converge=[
            ("RK4 frente a Euler: error de v(12 s) contra el número de pasos", "Se integra el mismo sistema con RK4 y con Euler explícito. RK4 muestra pendiente cercana a 4 y Euler cercana a 1, los órdenes teóricos; con 192 pasos RK4 llega a $5\times10^{-10}$ m/s mientras Euler sigue en el orden de $10^{-3}$ m/s.",
             [_tab([r"n", r"h\ [\mathrm{s}]", r"e_{RK4}\ [\mathrm{m/s}]", r"p", r"e_{Euler}", r"p"], rows)]),
            ("Energía disipada: Simpson y ventana de tiempo", f"La energía total es $\\tfrac12mv_0^2={fx(e_ex, 4)}$ J. Con 25 s fijos y $n$ variable el error baja como $h^4$ (Simpson). Con $n=10\\,000$ y $t_{{end}}$ variable el error es el de truncar la integral: $E\\,e^{{-2t_{{end}}/\\tau}}$, columna de la derecha.",
             [_tab([r"n", r"|E_{num}-E|\ [\mathrm{J}]", r"p"], rows_q),
              _tab([r"t_{end}\ [\mathrm{s}]", r"|E_{num}-E|\ [\mathrm{J}]", r"E\,e^{-2t_{end}/\tau}"], rows_t)]),
            ("Fuentes de error y criterio", f"Con 2400 pasos el error de truncamiento de $v(12\\ \\mathrm{{s}})$ es {_m(e_def, 2)} m/s, tan pequeño que ya compite con el redondeo ($\\varepsilon_{{mach}}\\cdot v_0\\approx{sci(EPS_MACH * m10.V0_ROD, 1)}$ m/s). El error de la energía lo domina la ventana: hay que integrar al menos hasta $t_{{end}}\\approx\\tau\\ln(E/\\mathrm{{tol}})/2$. Un resultado de RK4 es confiable si al duplicar los pasos el valor cambia menos que la tolerancia y se cumple $kh\\ll1$ (aquí $kh={sci(k * 12.0 / n_def, 2)}$).",
             [rf"kh = \dfrac{{h}}{{\tau}} = \dfrac{{{sci(12.0 / n_def, 3)}}}{{{fx(tau, 1)}}} = {sci(12.0 / n_def / tau, 3)}\ \ll 2.78"]),
        ],
    )


# =============================================================================== capitulo 11
def _rl() -> dict:
    tau = m11.tau_rl()
    t_end = 0.4
    ex = float(m11.i_growth_analytic(t_end))
    ns = [8, 16, 32, 64, 128]
    errs = []
    for n in ns:
        _, y = m11.rl_numeric(t_end, n)
        errs.append(abs(y[-1, 0] - ex))
    rows = [[str(n), sci(t_end / n * 1e3, 3), sci(e, 3), p] for n, e, p in zip(ns, errs, _slopes(ns, errs))]
    u = m11.energy_magnetic()
    te = [0.1, 0.2, 0.3, 0.5]
    rows_t = [[fx(t * 1e3, 0), sci(abs(m11.energy_integral_numeric(t, 5000) - u), 3), sci(2 * u * np.exp(-t / tau), 3)] for t in te]
    en = [50, 100, 400, 5000]
    ev = [abs(m11.energy_integral_numeric(0.5, n) - u) for n in en]
    e_def = abs(m11.energy_integral_numeric() - u)
    _, yd = m11.rl_numeric(0.4, 2000)
    e2000 = abs(yd[-1, 0] - ex)
    t_half = m11.time_to_half_numeric()
    t_half_ex = tau * np.log(2)
    return dict(
        numeric_intro=[
            r"En el circuito RL la corriente satisface $L\,di/dt=\varepsilon-Ri$. Esta ecuación lineal se resuelve a mano, pero se integra numéricamente para tener una comprobación independiente que también funcione cuando $R$ o $L$ dependan de la corriente (núcleos saturables, diodos). Además de i(t), se calculan dos cantidades derivadas: el tiempo de caída al 50 % (por interpolación) y la energía almacenada (por integración de $i\,v_L$).",
            r"El integrador es RK4: $i_{n+1}=i_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$, con $k_1=f(t_n,i_n)$, $k_2=f(t_n+\tfrac h2,\,i_n+\tfrac h2k_1)$, $k_3=f(t_n+\tfrac h2,\,i_n+\tfrac h2k_2)$ y $k_4=f(t_n+h,\,i_n+hk_3)$, para $f=(\varepsilon-Ri)/L$. Su error global es $O(h^4)$ porque la combinación de pendientes coincide con el desarrollo de Taylor hasta el término de cuarto orden. La energía se integra con Simpson, $O(h^4)$, evaluando $\int_0^{t_{end}} i\,(\varepsilon-Ri)\,dt$.",
            r"Hay una limitación importante: la estabilidad. Para la ecuación $i'=-i/\tau$, RK4 solo es estable si $h<2.78\,\tau$, y con pasos de ese orden la solución es inútil aunque no explote. Además, la integral de la energía se trunca en $t_{end}$: como la corriente se acerca a $i_\infty$ solo exponencialmente, a $t_{end}$ queda una fracción $\sim2e^{-t_{end}/\tau}$ de la energía fuera de la integral, error que ningún refinamiento de $h$ elimina.",
            r"Los parámetros ($\tau=50$ ms; 2000 pasos de 0.2 ms en 0.4 s para i(t); 5000 pasos en 0.5 s para la energía) mantienen $h/\tau\approx4\times10^{-3}$ y $t_{end}=10\tau$. El resultado de la energía es $\tfrac12Li_\infty^2=0.9$ J con un error de unos $10^{-4}$ J por la ventana. El método es independiente del analítico: no usa $1-e^{-t/\tau}$ ni la fórmula $\tfrac12Li^2$; integra la ecuación del circuito y suma la potencia entregada al inductor.",
        ],
        numeric_code="""# Circuito RL: RK4 para i(t) y Simpson para la energia (physics/v2ch11_inductance.py)
eps, R, L = 12.0, 4.0, 0.2
f = lambda t, i: (eps - R*i) / L               # di/dt de L di/dt = eps - R i
def rk4(f, t, i, h):
    k1 = f(t, i)                               # pendiente inicial
    k2 = f(t + h/2, i + h/2*k1)                # pendiente en el punto medio
    k3 = f(t + h/2, i + h/2*k2)                # corrige con k2
    k4 = f(t + h,   i + h*k3)                  # pendiente al final
    return i + h/6*(k1 + 2*k2 + 2*k3 + k4)     # promedio ponderado
n, t_end = 2000, 0.4;  h = t_end / n           # h = 0.2 ms  (h/tau = 0.004)
i = [0.0]
for j in range(n):
    i.append(rk4(f, j*h, i[-1], h))            # i_{n+1} a partir de i_n
# energia entregada al inductor: potencia i * v_L, con v_L = eps - R i
vL = eps - R*i                                 # tension en el inductor en cada instante
U = simpson(i * vL, t)                         # integral de i*vL dt hasta t_end = 0.5 s
# tiempo al 50 %: se integra di/dt = -R i / L y se interpola linealmente entre muestras""",
        numeric_converge=[
            ("Error de i(0.4 s) contra el número de pasos", f"Se integra con RK4 hasta $t={fx(t_end, 1)}$ s y se compara con $\\varepsilon/R\\,(1-e^{{-t/\\tau}})$. Cada vez que se duplica $n$ el error se reduce unas 16 veces (pendiente cercana a 4), el orden teórico. Con los 2000 pasos del código el error es {_m(e2000, 2)} A, es decir ya en el piso de redondeo.",
             [_tab([r"n", r"h\ [\mathrm{ms}]", r"|i_{num}-i|\ [\mathrm{A}]", r"p"], rows)]),
            ("Energía almacenada: el error lo domina la ventana de tiempo", f"Con $n=5000$ y distinta $t_{{end}}$ el error es la energía que queda sin integrar, $\\approx 2Ue^{{-t_{{end}}/\\tau}}$ con $U={fx(u, 1)}$ J. La tabla lo confirma (la aproximación es buena cuando $t_{{end}}\\gtrsim3\\tau$). Aumentar $n$ mejora poco: con $t_{{end}}=0.5$ s y $n=50,\\,100,\\,400,\\,5000$ el error es {_m(ev[0], 2)}, {_m(ev[1], 2)}, {_m(ev[2], 2)} y {_m(ev[3], 2)} J, estabilizado en el valor de truncamiento.",
             [_tab([r"t_{end}\ [\mathrm{ms}]", r"|U_{num}-U|\ [\mathrm{J}]", r"2U\,e^{-t_{end}/\tau}"], rows_t)]),
            ("Fuentes de error y criterio", f"Para $t_{{end}}=0.5$ s ($10\\tau$) el error de energía es {_m(e_def, 2)} J, un {_m(e_def / u * 100, 2)} % de $U$; el de $i(t)$ es de truncamiento $O(h^4)$ y el de redondeo está en $10^{{-14}}$ A. El tiempo al 50 % con interpolación lineal da {fx(t_half * 1e3, 6)} ms contra {fx(t_half_ex * 1e3, 6)} ms, pues el cruce se interpola con una recta (error $O(h^2)$, con $h=0.1$ ms). Criterio: el resultado es confiable si $h<\\tau/10$ (estabilidad y precisión) y $t_{{end}}\\ge10\\tau$ (ventana).",
             [rf"t_{{1/2}}^{{num}} = {fx(t_half * 1e3, 6)}\ \mathrm{{ms}},\qquad t_{{1/2}} = \tau\ln2 = {fx(t_half_ex * 1e3, 6)}\ \mathrm{{ms}},\qquad \dfrac{{|\Delta|}}{{t_{{1/2}}}} = {sci(abs(t_half - t_half_ex) / t_half_ex, 2)}"]),
        ],
    )


def _period_rlc(n: int) -> float:
    t, y = m11.rlc_numeric(0.2, n)
    q = y[:, 0]
    idx = [i for i in range(1, len(q)) if q[i - 1] > 0 >= q[i]]
    cross = lambda i: t[i] - q[i] * (t[i] - t[i - 1]) / (q[i] - q[i - 1])  # noqa: E731
    return float(cross(idx[1]) - cross(idx[0]))


def _energy_ratio_rlc(n: int) -> float:
    t, y = m11.rlc_numeric(2 * np.pi / m11.omega_d() + 1e-9, n)
    q, i = y[-1]
    return float((q**2 / (2 * m11.C2) + 0.5 * m11.L2 * i**2) / (m11.Q0**2 / (2 * m11.C2)))


def _rlc() -> dict:
    th = 2 * np.pi / m11.omega_d()
    ns = [150, 300, 600, 1200]
    eq = [abs(m11.rlc_numeric(0.15, n)[1][-1, 0] - float(m11.q_analytic(0.15))) for n in ns]
    rows = [[str(n), sci(0.15 / n * 1e6, 3), sci(e, 3), p] for n, e, p in zip(ns, eq, _slopes(ns, eq))]
    nu = [500, 1000, 2000, 4000]
    ex_e = float(np.exp(-2 * m11.gamma_() * th))
    ee = [abs(_energy_ratio_rlc(n) - ex_e) for n in nu]
    rows_e = [[str(n), sci(e, 3), p] for n, e, p in zip(nu, ee, _slopes(nu, ee))]
    npd = [500, 1000, 2000, 4000, 20000]
    ep = [abs(_period_rlc(n) - th) for n in npd]
    rows_p = [[str(n), fx(0.2 / n * 1e6, 1), sci(e, 3), sci(e / th, 2)] for n, e in zip(npd, ep)]
    return dict(
        numeric_intro=[
            r"La ecuación del RLC, $L\ddot q+R\dot q+q/C=0$, tiene solución analítica solo mientras los coeficientes son constantes y la amortiguación no es crítica; el método numérico trata el caso general y sirve para validar las fórmulas de $\omega_d$, $Q$ y de la energía tras un periodo. Se convierte en un sistema de primer orden con $y=(q,i)$, $y'=(i,\,-(R/L)i-q/LC)$, y se integra con RK4.",
            r"RK4 usa cuatro evaluaciones de la derivada por paso: $y_{n+1}=y_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$, con $k_2$ y $k_3$ en el punto medio y $k_4$ al final. Su error local es $O(h^5)$ y el global $O(h^4)$: si $h$ se divide entre dos, el error cae 16 veces. Para un oscilador el efecto principal del error es un pequeño error de amplitud y de fase, que se acumula con el número de oscilaciones.",
            r"Con las magnitudes de la escena ($\omega_0=316$ rad/s, $\gamma=R/2L=25\ \mathrm{s^{-1}}$, periodo $T\approx19.9$ ms) los 20 000 pasos de 10 µs dan unos 2000 puntos por periodo (la energía usa 8000 pasos en un solo periodo, $h\approx2.5\ \mu\mathrm{s}$). El periodo se mide entre dos cruces descendentes por cero con interpolación lineal; como ambos cruces se interpolan igual, la mayor parte de ese error se cancela en la resta. El método falla si $h\omega_0$ es de orden 1.",
            r"El resultado numérico no usa $\sqrt{\omega_0^2-\gamma^2}$ ni $e^{-\gamma t}$: la frecuencia amortiguada sale de medir cuánto tarda la carga en repetir su cruce por cero, y la razón de energías de evaluar $q^2/2C+\tfrac12Li^2$ al final de un periodo en el estado integrado. Por eso es una verificación genuina de la teoría de oscilaciones amortiguadas.",
        ],
        numeric_code="""# RLC serie: RK4 sobre (q, i) y medicion del periodo (physics/v2ch11_inductance.py)
L, C, R, Q0 = 0.1, 100e-6, 5.0, 1e-3
f = lambda t, y: [y[1], -(R/L)*y[1] - y[0]/(L*C)]   # q' = i ;  i' = -(R/L) i - q/(LC)
def rk4(f, t, y, h):
    k1 = f(t, y)                                     # cuatro pendientes por paso
    k2 = f(t + h/2, y + h/2*k1)
    k3 = f(t + h/2, y + h/2*k2)
    k4 = f(t + h,   y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)           # promedio 1-2-2-1
n, t_end = 20000, 0.2;  h = t_end / n                # 10 microsegundos por paso
y = [[Q0, 0.0]]                                      # condiciones iniciales: q = Q0, i = 0
for j in range(n): y.append(rk4(f, j*h, y[-1], h))
q = [yy[0] for yy in y]
idx = [j for j in range(1, n+1) if q[j-1] > 0 >= q[j]]   # cruces descendentes por cero
cruce = lambda j: t[j] - q[j]*(t[j]-t[j-1])/(q[j]-q[j-1])   # interpolacion lineal del cruce
T = cruce(idx[1]) - cruce(idx[0])                    # periodo amortiguado entre dos cruces
U = (q_f**2/(2*C) + 0.5*L*i_f**2) / (Q0**2/(2*C))    # U(T)/U(0) al final de un periodo""",
        numeric_converge=[
            ("Error de q(0.15 s) contra el número de pasos", "Se compara la carga integrada con la fórmula analítica de la oscilación amortiguada en $t=0.15$ s (unos 7.5 periodos). La pendiente medida cerca de 4 confirma el orden de RK4; cada duplicación de $n$ mejora la carga unas 16 veces.",
             [_tab([r"n", r"h\ [\mu\mathrm{s}]", r"|q_{num}-q|\ [\mathrm{C}]", r"p"], rows)]),
            ("Razón de energías tras un periodo", f"La cantidad $U(T)/U(0)$ debe ser $e^{{-2\\gamma T}}={fx(ex_e, 8)}$. El error del integrador baja con pendiente cercana a 4 y llega a $10^{{-12}}$ con 2000 pasos.",
             [_tab([r"n", r"|U_{num}/U_0-e^{-2\gamma T}|", r"p"], rows_e)]),
            ("Error en la medición del periodo", f"El periodo analítico es $T={fx(th * 1e3, 6)}$ ms. Aquí el error combina la fase de RK4 y la interpolación lineal del cruce; como son dos cruces consecutivos casi idénticos, el error baja más rápido que $h^2$ y llega a unos {_m(ep[-1] / th, 1)} relativos (piso de redondeo e interpolación) con los 20 000 pasos del código. Criterio de confiabilidad: más de 100 puntos por periodo y que al duplicar $n$ no cambie el periodo en la cifra deseada.",
             [_tab([r"n", r"h\ [\mu\mathrm{s}]", r"|T_{num}-T|\ [\mathrm{s}]", r"\text{relativo}"], rows_p),
              rf"T/h\Big|_{{n=20000}} = {fx(th / (0.2 / 20000), 0)}\ \text{{puntos por periodo}}"]),
        ],
    )


# =============================================================================== capitulo 12
def _dipole_rhs():
    w2 = m12.omega_small() ** 2
    return lambda _t, y: np.array([y[1], -w2 * np.sin(y[0])])  # noqa: E731


def _dipole_period(a_deg: float, n: int) -> float:
    t, y = integrate(_dipole_rhs(), np.array([np.deg2rad(a_deg), 0.0]), 0.0, 6.0, n, rk4_step)
    th = y[:, 0]
    idx = [i for i in range(1, len(th)) if th[i - 1] > 0 >= th[i]]
    cross = lambda i: t[i] - th[i] * (t[i] - t[i - 1]) / (th[i] - th[i - 1])  # noqa: E731
    return float(cross(idx[1]) - cross(idx[0]))


def _dipole_drift(n: int, a_deg: float = 60.0) -> float:
    t, y = integrate(_dipole_rhs(), np.array([np.deg2rad(a_deg), 0.0]), 0.0, 3.0, n, rk4_step)
    e = 0.5 * m12.INERTIA * y[:, 1] ** 2 + m12.potential_energy(y[:, 0])
    return float(np.max(np.abs(e - e[0])) / abs(e[0]))


def _dipole() -> dict:
    t0 = 2 * np.pi / m12.omega_small()
    nd = [500, 1000, 2000, 4000]
    dr = [_dipole_drift(n) for n in nd]
    rows_d = [[str(n), sci(3.0 / n * 1e3, 3), sci(e, 3), p] for n, e, p in zip(nd, dr, _slopes(nd, dr))]
    npd = [250, 500, 1000, 2000, 4000, 40000]
    rows_p = [[str(n), sci(abs(_dipole_period(30.0, n) - m12.period_exact(30.0)), 2), sci(abs(_dipole_period(90.0, n) - m12.period_exact(90.0)), 2)] for n in npd]
    t30, t90 = m12.period_exact(30.0), m12.period_exact(90.0)
    e90 = abs(m12.period_numeric(90.0) - t90) / t90
    h_def = 6.0 / 40000
    return dict(
        numeric_intro=[
            r"Para ángulos grandes la ecuación del dipolo, $I\ddot\theta=-\mu B\sin\theta$, no es la de un oscilador armónico y su periodo depende de la amplitud mediante una integral elíptica. Con el método numérico no se aproxima $\sin\theta\approx\theta$: se integra la ecuación completa en el tiempo con RK4 y el periodo se mide directamente en la trayectoria. La conservación de la energía sirve de prueba interna, porque el integrador no la impone.",
            r"Con $y=(\theta,\dot\theta)$ el sistema es $y'=(\dot\theta,\,-\omega^2\sin\theta)$ con $\omega^2=\mu B/I=5\ \mathrm{s^{-2}}$. RK4 combina cuatro pendientes, $y_{n+1}=y_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$, con error global $O(h^4)$. El periodo es la diferencia entre dos cruces descendentes por cero, localizados por interpolación lineal entre las dos muestras que lo encierran; la interpolación introduce un error $O(h^2)$ en cada cruce, pero se cancela casi por completo porque ambos cruces ocurren en la misma fase.",
            r"Una limitación inherente a RK4 es que no conserva la energía exactamente: sufre una deriva lenta que crece con el tiempo y con $h$. Con 6000 pasos en 3 s la deriva relativa es del orden de $10^{-14}$ o menor, por lo que no afecta el resultado; para integraciones largas haría falta un integrador simpléctico. Con $h=150\ \mu\mathrm{s}$ se tienen unos 19 000 puntos por periodo ($T_0\approx2.81$ s).",
            r"El resultado es independiente del analítico porque éste usa la media aritmético-geométrica, $T=T_0/\mathrm{AGM}(1,\cos(\theta_0/2))$, y el numérico solo evoluciona la ecuación diferencial y mide el tiempo entre cruces. Que coincidan a 12 cifras valida a la vez el programa y la fórmula elíptica, y muestra que el periodo a 90° es 18 % mayor que $T_0$.",
        ],
        numeric_code="""# Dipolo (pendulo) con RK4 y medicion del periodo (physics/v2ch12_magnetic_matter.py)
mu, B, I = 0.1, 5e-5, 1e-6
w2 = mu*B/I                                      # omega^2 = 5 1/s^2
f = lambda t, y: [y[1], -w2*sin(y[0])]           # y = (theta, omega_ang): sin(theta) completo
def rk4(f, t, y, h):
    k1 = f(t, y);  k2 = f(t + h/2, y + h/2*k1)   # cuatro pendientes por paso
    k3 = f(t + h/2, y + h/2*k2);  k4 = f(t + h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)
n, t_end = 40000, 6.0;  h = t_end / n            # h = 150 microsegundos
y = [[theta0, 0.0]]                              # parte del reposo en theta0
for j in range(n): y.append(rk4(f, j*h, y[-1], h))
th = [yy[0] for yy in y]
cr = [j for j in range(1, n+1) if th[j-1] > 0 >= th[j]]   # cruces descendentes por cero
t_c = lambda j: t[j] - th[j]*(t[j]-t[j-1])/(th[j]-th[j-1])   # cruce interpolado
T = t_c(cr[1]) - t_c(cr[0])                      # periodo = tiempo entre dos cruces descendentes
E = 0.5*I*omega**2 - mu*B*cos(theta)             # energia en cada instante (prueba de calidad)
deriva = max(abs(E - E[0])) / abs(E[0])          # debe ser ~ 1e-15""",
        numeric_converge=[
            ("Deriva de la energía contra el número de pasos", "A $60^\\circ$ se integra 3 s y se mide $\\max|E-E_0|/|E_0|$. La pendiente cercana a 4 confirma que RK4 es de cuarto orden; con 2000 pasos la deriva ya es de $10^{-13}$ y con los 6000 del código llega al piso de redondeo.",
             [_tab([r"n", r"h\ [\mathrm{ms}]", r"\Delta E/E", r"p"], rows_d)]),
            ("Error del periodo medido a 30° y a 90°", f"Los valores de referencia son $T(30^\\circ)={fx(t30, 6)}$ s y $T(90^\\circ)={fx(t90, 6)}$ s. El error del periodo baja rápido, aunque de forma irregular porque la interpolación lineal de los cruces no es monótona; con los 40 000 pasos del código el error absoluto es de $10^{{-13}}$ s, relativo {_m(e90, 1)} a 90°.",
             [_tab([r"n", r"|T_{num}-T|_{30^\circ}\ [\mathrm{s}]", r"|T_{num}-T|_{90^\circ}\ [\mathrm{s}]"], rows_p)]),
            ("Fuentes de error y criterio", f"Hay tres: truncamiento de RK4 ($O(h^4)$), interpolación del cruce (parcialmente cancelada) y redondeo ($\\sim10^{{-16}}$ por paso). Con $h={sci(h_def, 2)}$ s, el primero es despreciable. Un resultado es confiable cuando la deriva de energía es mucho menor que la tolerancia pedida y al duplicar $n$ el periodo no cambia. Además se observa la física: a $90^\\circ$ el periodo es {fx(t90 / t0, 4)} veces el de pequeñas oscilaciones $T_0={fx(t0, 4)}$ s.",
             [rf"\dfrac{{T(90^\circ)}}{{T_0}} = \dfrac{{{fx(t90, 4)}}}{{{fx(t0, 4)}}} = {fx(t90 / t0, 4)}"]),
        ],
    )


def _mc_error(x: float, n: int, seeds: range) -> float:
    errs = [m12.magnetization_metropolis(x, n=n, seed=s)[0] - np.tanh(x) for s in seeds]
    return float(np.sqrt(np.mean(np.square(errs))))


def _mc_sweeps(x: float, n: int, k: int, seed: int = 4) -> list[float]:
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=n)
    out = []
    for _ in range(k):
        de = 2 * x * s
        accept = (de <= 0) | (rng.uniform(size=n) < np.exp(-np.clip(de, 0, 50)))
        s = np.where(accept, -s, s)
        out.append(float(s.mean()))
    return out


def _curie() -> dict:
    x = 0.8
    ns = [1000, 3000, 10000, 30000, 100000]
    seeds = range(16)
    errs = [_mc_error(x, n, seeds) for n in ns]
    slope = float(np.polyfit(np.log(ns), np.log(errs), 1)[0])
    rows = [[sci(n, 1), sci(e, 3), sci(1 / np.sqrt(n * 40), 3), sci(np.sqrt((1 - np.tanh(x) ** 2) / (n * 40)), 3)] for n, e in zip(ns, errs)]
    sw = _mc_sweeps(x, 100000, 6)
    rows_s = [[str(i + 1), fx(v, 5), fx(abs(v - np.tanh(x)), 5)] for i, v in enumerate(sw)]
    xs = (0.1, 0.8, 1.6, 3.2)
    n3 = 20000
    rows_x = []
    for xv in xs:
        e = _mc_error(xv, n3, range(12))
        rows_x.append([fx(xv, 1), fx(np.tanh(xv), 4), sci(e, 2), sci(np.sqrt((1 - np.tanh(xv) ** 2) / (n3 * 40)), 2)])
    return dict(
        numeric_intro=[
            r"En un paramagneto ideal los espines son independientes, de modo que la magnetización se obtiene analíticamente de la distribución de Boltzmann: $M/M_s=\tanh x$. El método de Monte Carlo no usa esa distribución; la genera: se simula un conjunto de $N$ espines que cambian de orientación con probabilidades que reproducen el equilibrio a temperatura $T$, y se promedia. Se usa porque es el único camino cuando los espines interactúan (Ising, ferromagnetismo), donde no hay solución exacta.",
            r"La regla de Metropolis: se propone invertir un espín, se calcula el cambio de energía $\Delta E=2x\,s$ (en unidades de $kT$, con $E=-xs$) y se acepta si $\Delta E\le0$, o con probabilidad $e^{-\Delta E}$ si $\Delta E>0$. Esto cumple el balance detallado, de modo que la distribución estacionaria es la de Boltzmann. La magnetización es el estimador $\langle s\rangle=\dfrac1{N}\sum_i s_i$, promediado sobre los 40 últimos barridos.",
            r"El error es estadístico, no de truncamiento: los $N$ espines casi independientes dan una incertidumbre $\sigma\approx\sqrt{(1-m^2)/(N\,n_{barr})}\sim1/\sqrt{N}$ (teorema central del límite), de modo que para ganar una cifra hay que multiplicar el esfuerzo por 100. Hay además un sesgo de equilibración: los primeros barridos no están en equilibrio, por eso se descarta la mitad. El método falla cerca de transiciones de fase o con interacciones fuertes, donde las muestras se correlacionan.",
            r"Los parámetros ($N=100\,000$, 80 barridos con 40 descartados, semilla 4 para reproducibilidad) dan una incertidumbre de unos $5\times10^{-4}$ (cota del código) y una real menor. El resultado es independiente del analítico porque nunca evalúa $\tanh$: genera una lista de espines $\pm1$ mediante números aleatorios y los promedia. Que el promedio coincida con $\tanh x$ dentro de $\sigma$ es una verificación real de la mecánica estadística.",
        ],
        numeric_code="""# Metropolis para espines independientes (physics/v2ch12_magnetic_matter.py)
def magnetizacion(x, N=100_000, barridos=80, semilla=4):
    rng = generador(semilla)                      # semilla fija: resultado reproducible
    s = elegir([-1, +1], N)                       # estado inicial aleatorio (M ~ 0)
    medias = []
    for k in range(barridos):
        dE = 2*x*s                                # cambio de energia (en kT) al invertir cada espin
        u = rng.uniforme(N)                       # un numero aleatorio por espin
        acepta = (dE <= 0) | (u < exp(-dE))       # Metropolis: baja siempre, sube con prob. exp(-dE)
        s = donde(acepta, -s, s)                  # se invierten los espines aceptados
        if k >= barridos // 2:                    # se descartan los 40 primeros barridos
            medias.append(media(s))               # <s> de este barrido
    M = media(medias)                             # estimador de M/Ms
    sigma = sqrt(1 / (N * len(medias)))           # cota del error estadistico ~ 1/sqrt(N*40)
    return M, sigma
M, sigma = magnetizacion(x=0.8)                   # comparar con tanh(0.8) = 0.664""",
        numeric_converge=[
            ("Error contra el número de espines: ley 1/√N", f"Para $x={x}$ se repite la simulación con 16 semillas distintas para cada $N$ y se reporta el error cuadrático medio contra $\\tanh x={fx(np.tanh(x), 6)}$. El ajuste en escala logarítmica da pendiente {fx(slope, 2)}, cercana a $-1/2$. La cota del código $1/\\sqrt{{40N}}$ es conservadora (supone varianza 1); la teórica $\\sqrt{{(1-m^2)/40N}}$ (barridos independientes) queda del mismo orden, algo por encima del error medido.",
             [_tab([r"N", r"\text{RMS (16 semillas)}", r"1/\sqrt{40N}", r"\sqrt{(1-m^2)/40N}"], rows)]),
            ("Equilibración: cuántos barridos hacen falta", f"Cada fila es la media de espines tras el barrido indicado (un solo barrido, $N=10^5$, $x=0.8$). Partiendo de $M\\approx0$ la regla de Metropolis salta a valores cercanos al equilibrio en 2 o 3 barridos, con una oscilación inicial; descartar 40 de 80 es muy conservador y deja el sesgo muy por debajo del ruido.",
             [_tab([r"\text{barrido}", r"\langle s\rangle", r"|\langle s\rangle-\tanh x|"], rows_s)]),
            ("Dependencia con x y criterio de confiabilidad", f"La varianza de un espín es $1-\\tanh^2x$, que se anula cuando el sistema se satura; por eso el error disminuye al saturarse ($N={n3}$, 12 semillas). A $x$ pequeño el error medido queda bien por debajo de $\\sigma_{{teo}}$ porque casi todos los espines se invierten en cada barrido y barridos consecutivos quedan anticorrelacionados, lo que reduce la varianza del promedio; $\\sigma_{{teo}}$ es una cota. El resultado es confiable cuando $|M-M_{{teo}}|\\lesssim2\\sigma$, la pendiente $-1/2$ se mantiene al variar $N$ y el descarte de barridos no cambia el valor.",
             [_tab([r"x", r"\tanh x", r"\text{RMS}", r"\sigma_{teo}"], rows_x)]),
        ],
    )


# =============================================================================== capitulo 13
def _steady(w: float, settle: int = 25, meas: int = 6, ppc: int = 400) -> tuple[float, float]:
    """Replica de m13.steady_state_numeric con puntos por ciclo (ppc) y ciclos de transitorio parametrizables."""
    period = 2 * np.pi / w
    t_end = (settle + meas) * period
    n = (settle + meas) * ppc
    rhs = lambda t, y: np.array([y[1], (m13.V0 * np.sin(w * t) - m13.R * y[1] - y[0] / m13.C) / m13.L])  # noqa: E731
    t, y = integrate(rhs, np.zeros(2), 0.0, t_end, n, rk4_step)
    i = y[:, 1]
    k0 = settle * ppc
    tm, im = t[k0:], i[k0:]
    a = 2 * simpson(im * np.sin(w * tm), tm) / (tm[-1] - tm[0])
    b = 2 * simpson(im * np.cos(w * tm), tm) / (tm[-1] - tm[0])
    return float(np.hypot(a, b)), float(np.arctan2(-b, a))


def _series() -> dict:
    w0 = m13.omega0()
    ex = float(m13.current_amplitude_analytic(w0))
    gam = m13.R / (2 * m13.L)
    ppcs = [25, 50, 100, 200]
    e1 = [abs(_steady(w0, ppc=p)[0] - ex) for p in ppcs]
    rows1 = [[str(p), sci(2 * np.pi / w0 / p * 1e6, 3), sci(e, 3), s] for p, e, s in zip(ppcs, e1, _slopes(ppcs, e1))]
    sts = [1, 2, 5, 10, 25]
    e2 = [abs(_steady(w0, settle=s)[0] - ex) for s in sts]
    rows2 = [[str(s), fx(s * 2 * np.pi / w0 * 1e3, 1), sci(e, 3), sci(ex * np.exp(-gam * s * 2 * np.pi / w0), 3)] for s, e in zip(sts, e2)]
    ef, ph = abs(_steady(w0)[0] - ex), abs(_steady(w0)[1])
    w1 = 1250.0
    ex1 = float(m13.current_amplitude_analytic(w1))
    e_w1 = abs(m13.steady_state_numeric(w1)[0] - ex1)
    return dict(
        numeric_intro=[
            r"En régimen alterno la solución analítica sale de pasar a fasores (impedancia compleja); el método numérico, en cambio, trabaja en el dominio del tiempo: integra la ecuación diferencial completa $L\ddot q+R\dot q+q/C=V_0\sin\omega t$ desde el reposo, espera a que el transitorio se apague y luego extrae amplitud y fase de la corriente estacionaria. Así se comprueba que el modelo de fasores describe la dinámica real, y se puede aplicar a circuitos no lineales.",
            r"Son dos algoritmos encadenados. El primero es RK4 sobre $y=(q,i)$, $y_{n+1}=y_n+\tfrac h6(k_1+2k_2+2k_3+k_4)$, con error global $O(h^4)$. El segundo es una detección síncrona: si $i=I_0\sin(\omega t-\phi)$, entonces $a=\tfrac2T\int i\sin\omega t\,dt=I_0\cos\phi$ y $b=\tfrac2T\int i\cos\omega t\,dt=-I_0\sin\phi$, de donde $I_0=\sqrt{a^2+b^2}$ y $\phi=\mathrm{atan2}(-b,a)$. Las integrales se hacen con Simpson sobre un número entero de ciclos, lo que garantiza ortogonalidad exacta de senos y cosenos.",
            r"Hay dos fuentes de error independientes. La discretización: con $h=T/400$ el error de RK4 es de $10^{-7}$ A o menos. El transitorio residual: la corriente homogénea decae como $e^{-\gamma t}$ con $\gamma=R/2L=100\ \mathrm{s^{-1}}$, y después de 25 ciclos queda una fracción de $10^{-7}$. Cerca de la resonancia ($Q=5$) y con $R$ pequeña el transitorio dura más, y el número de ciclos de espera debe crecer como $1/\gamma$.",
            r"Los parámetros (25 ciclos de transitorio, 6 de medición, 400 puntos por ciclo) son suficientes para $Q=5$. El resultado es independiente del analítico: no se usa $Z=\sqrt{R^2+(\omega L-1/\omega C)^2}$ ni $\tan\phi$, sino solo el comportamiento temporal de $i(t)$. Es una comprobación de que, en resonancia, $I_0=V_0/R$ y $\phi=0$.",
        ],
        numeric_code="""# RLC en serie forzado: RK4 + deteccion sincrona (physics/v2ch13_ac.py)
R, L, C, V0 = 20.0, 0.1, 10e-6, 50.0
def amplitud_y_fase(w, ciclos_espera=25, ciclos_medida=6, ppc=400):
    T = 2*pi/w;  n = (ciclos_espera + ciclos_medida) * ppc   # 400 puntos por ciclo
    f = lambda t, y: [y[1], (V0*sin(w*t) - R*y[1] - y[0]/C) / L]   # (q, i); fuente senoidal
    t, y = integrar_rk4(f, [0, 0], 0, (ciclos_espera + ciclos_medida)*T, n)   # parte del reposo
    k0 = ciclos_espera * ppc                          # se descartan los primeros 25 ciclos
    tm, im = t[k0:], y[k0:, 1]                        # solo el regimen estacionario
    a = 2*simpson(im*sin(w*tm), tm) / (tm[-1]-tm[0])  # proyeccion sobre sen: I0 cos(phi)
    b = 2*simpson(im*cos(w*tm), tm) / (tm[-1]-tm[0])  # proyeccion sobre cos: -I0 sen(phi)
    return hypot(a, b), atan2(-b, a)                  # amplitud I0 y fase phi
w0 = 1/sqrt(L*C)                                      # 1000 rad/s (solo para elegir la frecuencia)
I0, phi = amplitud_y_fase(w0)                         # debe dar I0 = V0/R = 2.5 A y phi = 0
curva = [amplitud_y_fase(w)[0] for w in barrido_w]    # curva de resonancia punto por punto""",
        numeric_converge=[
            ("Error de la amplitud contra los puntos por ciclo", f"Se fija el transitorio (25 ciclos) y se varía el paso $h=T/\\mathrm{{ppc}}$. En resonancia ($\\omega_0={fx(w0, 0)}$ rad/s) la amplitud analítica es $V_0/R={fx(ex, 4)}$ A. La pendiente observada entre 25 y 100 puntos por ciclo es cercana a 4; con 200 puntos el error sale menor que el esperado (cancelación casual con el transitorio residual) y con 400 ya se está en el piso fijado por ese transitorio, del orden de $10^{{-7}}$ A.",
             [_tab([r"\mathrm{ppc}", r"h\ [\mu\mathrm{s}]", r"|I_{0,num}-I_0|\ [\mathrm{A}]", r"p"], rows1)]),
            ("Error contra los ciclos de espera del transitorio", f"Con $400$ puntos por ciclo, el error depende de cuánto decae el transitorio $e^{{-\\gamma t}}$, $\\gamma={fx(gam, 0)}\\ \\mathrm{{s^{{-1}}}}$: cada ciclo de espera lo reduce un factor $e^{{-\\gamma T}}={fx(np.exp(-gam * 2 * np.pi / w0), 3)}$. La última columna es la estimación $I_0e^{{-\\gamma t}}$, una cota superior: el error medido es unas cuatro veces menor, pero cae con el mismo ritmo.",
             [_tab([r"\text{ciclos}", r"t\ [\mathrm{ms}]", r"\text{error medido}\ [\mathrm{A}]", r"I_0e^{-\gamma t}\ [\mathrm{A}]"], rows2)]),
            ("Fuentes de error y criterio", f"Con los parámetros del código el error de amplitud es {_m(ef, 2)} A y el de fase {_m(ph, 2)} rad en resonancia; fuera de ella, a $1250$ rad/s, la amplitud difiere {_m(e_w1, 2)} A de $V_0/Z={fx(ex1, 5)}$ A. El resultado es confiable cuando el error de ambos barridos (ppc y ciclos de espera) es menor que la tolerancia y la amplitud no cambia al ampliar la ventana de medición. Para $Q$ más alto hay que esperar $\\gtrsim 10/\\gamma$ de transitorio.",
             [rf"e^{{-\gamma\, t_{{espera}}}} = e^{{-({fx(gam, 0)})({fx(25 * 2 * np.pi / w0, 4)})}} = {sci(float(np.exp(-gam * 25 * 2 * np.pi / w0)), 2)}"]),
        ],
    )


def _power() -> dict:
    pa = float(m13.power_average_analytic())
    vp = m13.V_PEAK
    ns = [2, 4, 6, 8, 16, 4000]
    ep = [abs(m13.power_average_numeric(n) - pa) for n in ns]
    ev = [abs(m13.rms_numeric(vp, n) - vp / np.sqrt(2)) for n in ns]
    fmt = lambda e: sci(e, 2) if e > 0 else "0"  # noqa: E731
    rows = [[str(n), fmt(a), fmt(b)] for n, a, b in zip(ns, ep, ev)]
    ph = np.deg2rad(m13.PHI_DEG)
    rows_w = []
    for frac in (1.0, 1.1, 1.25, 1.5, 2.0):
        t = np.linspace(0, frac, 4001)
        v = vp * np.sin(2 * np.pi * t)
        i = m13.I_PEAK * np.sin(2 * np.pi * t - ph)
        rows_w.append([fx(frac, 2), fx(simpson(v * i, t) / frac, 4), sci(abs(simpson(v * i, t) / frac - pa), 2) if abs(simpson(v * i, t) / frac - pa) > 1e-9 else "0"])
    tn = m13.transformer_numeric()
    ta = m13.transformer_analytic()
    return dict(
        numeric_intro=[
            r"La potencia media es $\bar P=\tfrac1T\int_0^Tv(t)\,i(t)\,dt$ y los valores eficaces son $\sqrt{\tfrac1T\int_0^Tv^2dt}$: ambos son promedios temporales. La fórmula analítica se obtiene con identidades trigonométricas, $\bar P=\tfrac12V_0I_0\cos\phi$, pero con formas de onda reales (rectificadores, cargas no lineales, ruido) la señal no es una senoide y la integral numérica es el procedimiento general; aquí se verifica en el caso senoidal donde se conoce la respuesta.",
            r"El método muestrea $v$ e $i$ en una malla uniforme de $n$ subintervalos sobre un periodo y aplica la regla de Simpson, $\int f\,dt\approx\tfrac h3[f_0+4f_1+2f_2+4f_3+\dots+4f_{n-1}+f_n]$, que sustituye $f$ por parábolas de tres puntos. El error general es $O(h^4)$ (con derivada cuarta acotada), y requiere $n$ par. Para el transformador se escalan las muestras por la razón de vueltas y se comprueba que $v\,i$ se conserva.",
            r"Aquí ocurre algo especial: el integrando es un polinomio trigonométrico de bajo grado (frecuencia máxima doble) integrado sobre un periodo completo, y las reglas de cuadratura uniforme son exactas para esas funciones en cuanto la malla distingue las frecuencias presentes. Por eso el error numérico no sigue $h^4$ sino que cae a cero (redondeo) en cuanto $n\ge6$. El método falla, y de modo espectacular, si el intervalo no es un número entero de periodos.",
            r"Con $n=4000$ (4000 muestras por periodo) se está muy por encima de lo necesario, lo que da un error de redondeo de $10^{-13}$ W. El método es independiente del analítico porque no usa $\cos\phi$ ni $1/\sqrt2$: obtiene la potencia integrando el producto $v\cdot i$ tal como se ve en el osciloscopio. El mismo código verifica el transformador: la potencia primaria y secundaria se integran por separado y se compara.",
        ],
        numeric_code="""# Potencia media, valor eficaz y transformador por Simpson (physics/v2ch13_ac.py)
V0, I0, phi = 170.0, 8.0, radianes(40.0)
def simpson(y, x):
    n = len(x) - 1; h = (x[-1]-x[0]) / n              # n debe ser par
    return h/3 * (y[0] + y[-1] + 4*suma(y[1:-1:2]) + 2*suma(y[2:-1:2]))   # pesos 1,4,2,4,...,1
def potencia_media(n=4000):
    t = linspace(0, 1, n+1)                            # un periodo normalizado
    v = V0 * sin(2*pi*t)                               # tension
    i = I0 * sin(2*pi*t - phi)                         # corriente atrasada 40 grados
    return simpson(v*i, t)                             # promedio de v*i (T = 1)
def rms(A, n=4000):
    t = linspace(0, 1, n+1)
    return sqrt(simpson((A*sin(2*pi*t))**2, t))        # raiz del promedio de v^2
# transformador 200:50, primario 120 V y 2 A eficaces
t = linspace(0, 1, 4001)
vp = sqrt(2)*120*sin(2*pi*t);  ip = sqrt(2)*2*sin(2*pi*t)   # ondas del primario
vs = vp*50/200;  i_s = ip*200/50                       # secundario: v escala con Ns/Np, i con Np/Ns
Pp, Ps = simpson(vp*ip, t), simpson(vs*i_s, t)         # potencias integradas por separado
Vs, Is = rms_de(vs), rms_de(i_s)                       # valores eficaces del secundario""",
        numeric_converge=[
            ("Error de Simpson contra el número de subintervalos", f"La potencia media exacta es $\\bar P={fx(pa, 4)}$ W y el eficaz exacto $V_0/\\sqrt2={fx(vp / np.sqrt(2), 4)}$ V. Con $n=2$ y $n=4$ la malla no resuelve la señal y el error es enorme; desde $n=6$ el error es de orden $10^{{-14}}$ y con 4000 sigue en el piso de redondeo, así que refinar más no sirve.",
             [_tab([r"n", r"|\bar P_{num}-\bar P|\ [\mathrm{W}]", r"|V_{rms}^{num}-V_{rms}|\ [\mathrm{V}]"], rows)]),
            ("La ventana debe ser un número entero de periodos", "Si se integra sobre una ventana de $\\tau$ periodos y se divide entre $\\tau$, el resultado solo es correcto para $\\tau$ entero (o semientero, por la simetría de esta onda). Este es el error dominante en la práctica y no tiene nada que ver con $n$, que aquí es 4000.",
             [_tab([r"\tau\ [\text{periodos}]", r"\bar P_{num}\ [\mathrm{W}]", r"|\bar P_{num}-\bar P|\ [\mathrm{W}]"], rows_w)]),
            ("Transformador: comprobación cruzada", f"Con los 4001 puntos el secundario sale en $V_s={fx(tn['vs'], 6)}$ V e $I_s={fx(tn['is'], 6)}$ A frente a {fx(ta['vs'], 2)} V y {fx(ta['is'], 2)} A, y las potencias integradas coinciden en $P_p-P_s={sci(tn['p_p'] - tn['p_s'], 2)}$ W, es decir cero salvo redondeo. Criterio de confiabilidad: ventana entera de periodos, $n\\ge6$ por periodo para señales senoidales, y balance $P_p=P_s$ como prueba de consistencia.",
             [rf"P_p = {fx(tn['p_p'], 6)}\ \mathrm{{W}},\qquad P_s = {fx(tn['p_s'], 6)}\ \mathrm{{W}},\qquad V_pI_p = (120)(2.0) = 240\ \mathrm{{W}}"]),
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
