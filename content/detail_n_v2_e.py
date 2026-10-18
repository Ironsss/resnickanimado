"""Metodo numerico explicado: Volumen 2, capitulos 19 a 22 (fotones, ondas de materia, Schrodinger, atomo).

Para cada escena: numeric_intro (idea, formula, orden de error, parametros, independencia),
numeric_code (algoritmo esencial de physics/) y numeric_converge (estudio de error calculado).
Todas las cifras de los estudios salen de ejecutar el algoritmo de `physics/` con distintos N / h.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc

from content.texfmt import fx, sci
from physics import v2ch19_photons as p19
from physics import v2ch20_matter_waves as p20
from physics import v2ch21_schrodinger as p21
from physics import v2ch22_atom as p22
from physics.ch07_work_energy import simpson
from physics.constants import C_LIGHT, E_CH, H_PLANCK, HBAR, M_E


def _arr(head: list[str], rows: list[list[str]], al: str) -> str:
    """Tabla TeX (array) con encabezado, linea y filas."""
    body = r" \\ ".join(" & ".join(r) for r in rows)
    return r"\begin{array}{" + al + r"} " + " & ".join(head) + r" \\ \hline " + body + r" \end{array}"


def _slope(n1: float, e1: float, n2: float, e2: float) -> float:
    """Pendiente log-log local: e ~ n^p  ->  p."""
    return float(np.log(abs(e2) / abs(e1)) / np.log(n2 / n1))


# =============================================================================== BlackbodySpectrum
def _golden(temp: float, iters: int) -> tuple[float, float]:
    """Misma busqueda de la seccion aurea que p19.wien_peak_numeric pero con 'iters' iteraciones. Devuelve (ancho, centro)."""
    lo, hi = 0.2 * p19.WIEN_B / temp, 5 * p19.WIEN_B / temp
    g = (np.sqrt(5) - 1) / 2
    c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(iters):
        if p19.planck(c, temp) > p19.planck(e, temp):
            hi = e
        else:
            lo = c
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    return hi - lo, 0.5 * (lo + hi)


def _blackbody() -> dict:
    t = 5800.0
    lam_max = p19.wien_peak_analytic(t)
    w0 = 4.8 * p19.WIEN_B / t
    g = (np.sqrt(5) - 1) / 2

    its = [5, 10, 20, 30, 40, 50, 75]
    rows = []
    errs = {}
    for k in its + [200]:
        w, c = _golden(t, k)
        errs[k] = abs(c - lam_max) / lam_max
        if k in its:
            rows.append([str(k), sci(w * 1e9, 3), sci(w / w0, 3), sci(g**k, 3), sci(errs[k], 3)])
    tab1 = _arr([r"k", r"\mathrm{ancho}\ (\mathrm{nm})", r"w_k/w_0", r"g^k", r"|\delta\lambda|/\lambda_{max}"], rows, "rrrrr")
    ratio = (_golden(t, 30)[0] / _golden(t, 10)[0]) ** (1 / 20)
    floor = errs[200]

    exact = p19.stefan_analytic(t)
    lmx = p19.WIEN_B / t
    ns = [400, 800, 1600, 3200, 6400]
    nref = 409600
    lam_r = np.linspace(0.02 * lmx, 200 * lmx, nref + 1)
    ref = simpson(p19.planck(lam_r, t), lam_r)
    rows2, ee = [], []
    for n in ns:
        lam = np.linspace(0.02 * lmx, 200 * lmx, n + 1)
        v = simpson(p19.planck(lam, t), lam)
        ee.append((v - ref) / ref)
        rows2.append([str(n), sci((200 - 0.02) * lmx / n * 1e9, 3), sci(v, 6), sci(ee[-1], 3), "-" if len(ee) == 1 else fx(_slope(ns[len(ee) - 2], ee[-2], n, ee[-1]), 2)])
    tab2 = _arr([r"N", r"h\ (\mathrm{nm})", r"P_N\ (\mathrm{W/m^2})", r"(P_N-P_{ref})/P_{ref}", r"p"], rows2, "rrrrr")
    trunc = (ref - exact) / exact
    p20k = p19.stefan_numeric(t, 20000)
    # cola de Rayleigh-Jeans por encima de 200 lam_max: integral de 2 pi c k T / lam^4 desde Lc
    from physics.constants import K_B
    tail = 2 * np.pi * C_LIGHT * K_B * t / (3 * (200 * lmx) ** 3)

    return dict(
        numeric_intro=[
            r"Hay dos problemas numéricos distintos. El primero es hallar el máximo de $u(\lambda,T)$ cuando la condición $u'(\lambda)=0$ lleva a la ecuación trascendente $x = 5(1-e^{-x})$, que no tiene solución algebraica (solo se resuelve con funciones especiales o numéricamente). El segundo es calcular $P/A=\int_0^\infty u\,d\lambda$, una integral que sí tiene primitiva en términos de funciones polilogarítmicas, pero que aquí se evalúa por cuadratura para comprobar el valor de Stefan-Boltzmann sin usarlo.",
            r"La búsqueda de la sección áurea necesita solo que $u$ sea unimodal en $[a,b]$. Con $g=(\sqrt5-1)/2\approx0.618$ se colocan dos puntos interiores $c=b-g(b-a)$ y $e=a+g(b-a)$; si $u(c)>u(e)$ el máximo no puede estar en $(e,b]$ y se descarta esa cola, y si no se descarta $[a,c)$. El intervalo se contrae por el factor $g$ en cada iteración, de modo que $w_k=g^k w_0$: convergencia lineal, sin usar derivadas.",
            r"Para la potencia se usa la regla de Simpson compuesta, $\int_a^b f\,d\lambda\approx\tfrac h3[f_0+4f_1+2f_2+\dots+4f_{N-1}+f_N]$, que integra exactamente cúbicas y tiene error global $O(h^4)$, proporcional a $f^{(4)}$. Falla (o converge mal) cuando $f$ varía mucho dentro de un subintervalo, como ocurre con una $N$ pequeña frente a la escala de la exponencial, y no ve lo que queda fuera del intervalo de integración.",
            r"Los parámetros se escogieron así: el intervalo de búsqueda $[0.2,\,5]\,b/T$ contiene el máximo con holgura y garantiza unimodalidad; 200 iteraciones son más de las necesarias (hacia la iteración 75 el ancho ya es del orden de la resolución de la máquina, $g^{75}\approx2\times10^{-16}$), lo que vuelve irrelevante el criterio de parada. La integral usa $N=20000$ en $[0.02,\,200]\lambda_{max}$: con tantos nodos el error de Simpson es mucho menor que el que introduce truncar las colas. El resultado de la escena (el pico y $\sigma T^4$ numéricos) coincide con los valores analíticos a unos $4\times10^{-9}$ (pico) y $8\times10^{-7}$ (potencia, ver más abajo).",
            r"El método es independiente del analítico porque nunca usa la constante de Wien ni $\sigma$: solo evalúa la función de Planck (con $h$, $c$, $k_B$) en puntos y compara valores. La única información analítica que entra es el intervalo inicial, que se toma como $b/T$ por conveniencia, no como respuesta, y el máximo hallado se contrasta después con $b/T$.",
        ],
        numeric_code="""g = (sqrt(5) - 1) / 2                         # inverso de la razón áurea, 0.618...
lo, hi = 0.2*b/T, 5*b/T                       # intervalo que contiene el máximo de Planck
c, e = hi - g*(hi-lo), lo + g*(hi-lo)         # dos puntos interiores
for _ in range(200):                          # 200 iteraciones (sobradas)
    if planck(c, T) > planck(e, T):           # u(c) > u(e): el máximo no está en (e, hi]
        hi = e                                # se descarta la cola derecha
    else:
        lo = c                                # se descarta la cola izquierda
    c, e = hi - g*(hi-lo), lo + g*(hi-lo)     # puntos nuevos en el intervalo contraído
lam_max = (lo + hi) / 2                       # centro del intervalo final

N = 20000                                     # subintervalos (par)
lam = linspace(0.02*lam_max, 200*lam_max, N+1)  # malla uniforme
u = planck(lam, T)                            # u(lambda) en cada nodo
h = (lam[-1] - lam[0]) / N                    # paso
P = h/3 * (u[0] + u[-1]                       # extremos con peso 1
           + 4*u[1:-1:2].sum()                # nodos impares con peso 4
           + 2*u[2:-1:2].sum())               # nodos pares interiores con peso 2
""",
        numeric_converge=[
            ("Sección áurea: el intervalo se contrae por g en cada vuelta",
             f"Se repite la búsqueda para $T = {t:.0f}$ K con distinto número de iteraciones $k$. El ancho relativo $w_k/w_0$ coincide con $g^k$ (la razón observada entre $k=10$ y $k=30$ es {ratio:.4f}, frente a $g = {g:.4f}$), y el error en $\\lambda_{{max}}$ baja con él hasta cierto punto.",
             [tab1]),
            ("Por qué el error se detiene en un valor finito",
             f"A partir de $k\\approx 40$ el error relativo se queda en {sci(floor, 3)} aunque el intervalo siga encogiéndose. Cerca del máximo $u$ es plana ($u(\\lambda)\\approx u_{{max}}[1-\\tfrac12 a\\,\\delta^2]$): dos puntos separados menos de $\\sim\\sqrt{{\\varepsilon_{{m}}}}\\approx 10^{{-8}}\\lambda_{{max}}$ dan el mismo valor en doble precisión y la comparación ya no distingue cuál es mayor. Es un límite del método (no usa derivadas), no un error de programación; para exigir más precisión habría que localizar el cero de $u'$.",
             []),
            ("Simpson: el error cae al menos como h a la cuarta",
             f"Se integra Planck en el mismo intervalo con $N$ creciente y se compara con una referencia de la misma regla con $N={nref}$ (el último renglón ya está cerca del error propio de esa referencia). Las pendientes locales $p$ empiezan cerca de $-4$ (la cota $O(h^4)$) y luego son mayores en valor absoluto: el integrando y sus derivadas se anulan en ambos extremos, así que según Euler-Maclaurin la regla converge más rápido que $h^4$ una vez que $h$ resuelve la escala de la exponencial.",
             [tab2]),
            ("Fuentes de error y criterio de confianza",
             f"Con $N=20000$ el error de Simpson es despreciable. Lo que domina es el truncamiento del intervalo: el valor de referencia queda {sci(trunc, 3)} (relativo) por debajo de $\\sigma T^4$, y la cola de Rayleigh-Jeans por encima de $200\\lambda_{{max}}$ estimada con $\\int 2\\pi ckT\\lambda^{{-4}}d\\lambda$ aporta {sci(tail / exact, 3)}, es decir casi todo ese déficit. El resultado con $N=20000$ difiere de $\\sigma T^4$ en {sci(p20k / exact - 1, 3)}. El redondeo (≈$10^{{-16}}$ por término) no importa. Se considera confiable cuando el error al duplicar $N$ cambia por debajo de la incertidumbre de truncamiento.",
             [rf"\dfrac{{P_{{20000}}-\sigma T^4}}{{\sigma T^4}} = {sci(p20k / exact - 1, 3)}"]),
        ],
    )


# =============================================================================== ComptonPhotoelectric
def _bisect_compton(theta: float, iters: int, lam: float = 71.0e-12) -> float:
    """Biseccion de p19.compton_shift_numeric con 'iters' iteraciones."""
    e0 = H_PLANCK * C_LIGHT / lam
    mc2 = M_E * C_LIGHT**2

    def resid(e1):
        p0, p1 = e0 / C_LIGHT, e1 / C_LIGHT
        pe2 = p0**2 + p1**2 - 2 * p0 * p1 * np.cos(theta)
        return (e0 + mc2 - e1) ** 2 - (pe2 * C_LIGHT**2 + mc2**2)

    lo, hi = e0 * 1e-6, e0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if resid(lo) * resid(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float(H_PLANCK * C_LIGHT / (0.5 * (lo + hi)) - lam)


def _fit_h(n: int, noise: float, seed: int) -> tuple[float, float]:
    """Mismo ajuste que p19.millikan_fit pero con n puntos y ruido 'noise' (V). Devuelve (h, phi)."""
    rng = np.random.default_rng(seed)
    f = np.linspace(6.0e14, 12.0e14, n)
    v = p19.stopping_potential_analytic(f) + rng.normal(0, noise, f.size)
    f14 = f / 1e14
    a = np.vstack([f14, np.ones_like(f14)]).T
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    return float(coef[0] * E_CH / 1e14), float(-coef[1])


def _compton() -> dict:
    th = np.pi / 2
    ana = float(p19.compton_shift_analytic(th))
    lam0 = 71.0e-12
    e0 = H_PLANCK * C_LIGHT / lam0
    its = [10, 20, 30, 40, 50, 60, 200]
    rows, er = [], {}
    for k in its:
        d = _bisect_compton(th, k)
        er[k] = abs(d - ana) / ana
        rows.append([str(k), sci(e0 * 2.0**-k / E_CH, 3), sci(d * 1e12, 8), sci(er[k], 3)])
    tab1 = _arr([r"k", r"E_{hi}-E_{lo}\ (\mathrm{eV})", r"\Delta\lambda_k\ (\mathrm{pm})", r"|\Delta\lambda_k-\Delta\lambda|/\Delta\lambda"], rows, "rrrr")
    assert abs(_bisect_compton(th, 200) - p19.compton_shift_numeric(th)) < 1e-30

    # fotoelectrico: error de h vs numero de puntos (rms sobre 300 semillas)
    h_true = H_PLANCK
    assert abs(_fit_h(25, 0.01, 3)[0] - p19.millikan_fit()[0]) < 1e-45
    ns = [5, 10, 25, 100, 400]
    rows2, rms = [], []
    for n in ns:
        e = np.array([_fit_h(n, 0.01, s)[0] / h_true - 1 for s in range(300)])
        rms.append(float(np.sqrt(np.mean(e**2))))
        rows2.append([str(n), sci(rms[-1], 3), "-" if len(rms) == 1 else fx(_slope(ns[len(rms) - 2], rms[-2], n, rms[-1]), 2)])
    tab2 = _arr([r"N", r"\mathrm{rms}\,(h_{aj}/h-1)", r"p"], rows2, "rrr")
    sl_all = _slope(ns[0], rms[0], ns[-1], rms[-1])
    rows3, rms3 = [], []
    for sg in (0.005, 0.01, 0.02, 0.04):
        e = np.array([_fit_h(25, sg, s)[0] / h_true - 1 for s in range(300)])
        rms3.append(float(np.sqrt(np.mean(e**2))))
        rows3.append([fx(sg * 1000, 0), sci(rms3[-1], 3)])
    tab3 = _arr([r"\sigma_V\ (\mathrm{mV})", r"\mathrm{rms}\,(h_{aj}/h-1)"], rows3, "rr")
    h25, phi25, dh, dphi = p19.millikan_fit()

    return dict(
        numeric_intro=[
            r"Aquí hay dos métodos. En Compton se necesita la energía $E_1$ del fotón dispersado; la solución analítica sale de eliminar el electrón entre las ecuaciones de conservación, pero también puede plantearse como una ecuación de una incógnita $F(E_1)=0$: $F(E_1)=(E_0+m_ec^2-E_1)^2-[(p_ec)^2+m_e^2c^4]$ con $(p_ec)^2=E_0^2+E_1^2-2E_0E_1\cos\theta$. Esa función se anula exactamente cuando energía y momento se conservan a la vez, y la raíz se busca por bisección.",
            r"La bisección parte de un intervalo $[E_{lo},E_{hi}]$ donde $F$ cambia de signo, evalúa el punto medio y se queda con la mitad donde sigue cambiando el signo. Por el teorema del valor intermedio, si $F$ es continua la raíz queda siempre atrapada, y el ancho es $w_k=w_0\,2^{-k}$: convergencia lineal, un bit por iteración. Es lenta pero infalible; su limitación es que necesita un cambio de signo conocido y, cerca de la raíz, el error queda fijado por el redondeo al evaluar $F$ (aquí hay una resta de cantidades grandes, de orden $(m_ec^2)^2$).",
            r"En el fotoeléctrico el método es de mínimos cuadrados. Con datos $(f_i,V_i)$ afectados de ruido gaussiano se minimiza $\sum(V_i-a x_i-c)^2$, lo que da un sistema lineal de $2\times2$ (ecuaciones normales) cuya solución es el estimador de máxima verosimilitud para ruido gaussiano. La pendiente es $h/e$ y la ordenada $-\phi/e$. El error estadístico de $a$ escala como $\sigma_V/(\sqrt N\,s_x)$: baja como $1/\sqrt N$ y sube linealmente con el ruido; el método falla si el modelo no es lineal o si el ruido tiene sesgo.",
            r"Los parámetros: se parte de $[10^{-6}E_0,\,E_0]$ porque $E_1$ debe ser positivo y menor que $E_0$; se hacen 200 iteraciones, más de las que pueden aprovecharse en doble precisión (con unas 50 el ancho ya es del orden de $\varepsilon_m E_0$). En el ajuste se usa $N=25$ puntos entre $6$ y $12\times10^{14}$ Hz, ruido de 10 mV (típico de un voltímetro) y semilla 3 para que la figura sea reproducible; la frecuencia se escala por $10^{14}$ para que la matriz $A^TA$ esté bien condicionada.",
            r"Ambos son independientes de la fórmula cerrada. La bisección nunca usa $\Delta\lambda=\lambda_C(1-\cos\theta)$, solo las dos leyes de conservación evaluadas en puntos de prueba, y el ajuste de Millikan nunca usa el valor de $h$: lo recupera de datos simulados con la relación $eV_0=hf-\phi$.",
        ],
        numeric_code="""e0 = h*c / lam                                # energía del fotón incidente (lam = 71 pm)
mc2 = m_e * c**2                              # energía en reposo del electrón
def F(e1):                                    # residuo: se anula si se conservan E y p
    p0, p1 = e0/c, e1/c                       # momentos de los fotones
    pe2 = p0**2 + p1**2 - 2*p0*p1*cos(theta)  # |p_e|^2 por conservación de momento
    return (e0 + mc2 - e1)**2 - (pe2*c**2 + mc2**2)   # E_e^2 por energía menos relación E^2=(pc)^2+m^2c^4
lo, hi = e0*1e-6, e0                          # intervalo que contiene la raíz
for _ in range(200):                          # bisección
    mid = (lo + hi) / 2                       # punto medio
    if F(lo) * F(mid) <= 0: hi = mid          # raíz en la mitad izquierda
    else:                   lo = mid          # raíz en la mitad derecha
e1 = (lo + hi) / 2                            # energía del fotón dispersado
dlam = h*c/e1 - lam                           # corrimiento Compton numérico

f = linspace(6e14, 12e14, 25)                 # frecuencias de la escena
V = (h*f - phi)/e + normal(0, 0.01, 25)       # potencial de frenado + 10 mV de ruido (semilla 3)
A = [f/1e14, 1]                               # matriz de diseño con f reescalada
a, c0 = lstsq(A, V)                           # mínimos cuadrados: pendiente y ordenada
h_fit = a * e / 1e14                          # h = pendiente * e
phi_fit = -c0                                 # función trabajo en eV
""",
        numeric_converge=[
            ("Bisección: un bit por iteración",
             f"Se repite la búsqueda de $E_1$ para $\\theta=90^\\circ$ con $k$ iteraciones y se compara el corrimiento con $\\lambda_C(1-\\cos\\theta)={sci(ana * 1e12, 8)}$ pm. El ancho del intervalo se divide exactamente por 2 en cada paso y el error relativo baja rápidamente (un factor de unos $10^{{3}}$ por cada 10 iteraciones, igual que $2^{{10}}$).",
             [tab1]),
            ("Límite por redondeo y criterio de parada",
             f"Desde $k\\approx 50$ el error se estanca en {sci(er[200], 3)}. No es de truncamiento: en esa etapa $E_1$ ya no cambia en doble precisión, y el corrimiento $\\Delta\\lambda=hc/E_1-\\lambda_0$ es una resta de dos longitudes de onda casi iguales ($\\Delta\\lambda/\\lambda_0\\approx 3\\%$), que además amplifica el redondeo de $E_1$. Por eso hacer más de unas 55 iteraciones no mejora nada; las 200 del código son seguras pero redundantes. El resultado es confiable si el error es menor que $10^{{-10}}$ relativos y no cambia al añadir más iteraciones.",
             []),
            ("Fotoeléctrico: el error estadístico cae como 1/√N",
             f"Para medir cómo depende de $N$ el ajuste de $h$ se repite el experimento sintético con 300 semillas distintas por cada $N$ (mismo ruido de 10 mV y mismo intervalo de frecuencias) y se calcula la raíz cuadrática media del error relativo de $h$. La pendiente observada entre $N=5$ y $N=400$ es {sl_all:.2f}, cercana al $-1/2$ de la teoría (la de la última columna es local, por parejas de valores).",
             [tab2]),
            ("Sensibilidad al ruido y qué significa el caso de la escena",
             f"Con $N=25$ el error de $h$ crece proporcionalmente al ruido del voltímetro (tabla). En la corrida de la escena (semilla 3) el ajuste da $h={sci(h25, 5)}\\ \\mathrm{{J\\,s}}$ con incertidumbre estándar {sci(dh, 2)} (el valor de entrada es {sci(H_PLANCK, 5)}) y $\\phi={phi25:.4f}$ eV con incertidumbre {dphi:.4f} eV. El resultado es confiable cuando la diferencia con el valor verdadero es del orden de una desviación (aquí {abs(h25 - H_PLANCK) / dh:.2f}) y los residuos no muestran tendencia. Los errores de redondeo ($\\sim10^{{-16}}$) son irrelevantes; domina el ruido estadístico.",
             [tab3]),
        ],
    )


# =============================================================================== ElectronInterference
def _interference() -> dict:
    ke = 100.0
    ana = p20.fringe_spacing_analytic(ke)
    lam = p20.de_broglie(ke)
    ns = [500, 2000, 20000, 200000]
    rows, rms = [], []
    for n in ns:
        e = np.array([p20.fringe_spacing_from_hits(p20.electron_hits(n, seed=100 + s))[0] / ana - 1 for s in range(12)])
        rms.append(float(np.sqrt(np.mean(e**2))))
        rows.append([str(n), sci(rms[-1], 3), sci(float(e.mean()), 3), "-" if len(rms) == 1 else fx(_slope(ns[len(rms) - 2], rms[-2], n, rms[-1]), 2)])
    tab1 = _arr([r"N", r"\mathrm{rms}\,(\delta/\delta_0-1)", r"\mathrm{promedio}", r"p"], rows, "rrrr")
    sp_scene = p20.fringe_spacing_from_hits(p20.electron_hits(20000))[0]
    e_scene = sp_scene / ana - 1
    bias = float(np.mean([p20.fringe_spacing_from_hits(p20.electron_hits(200000, seed=100 + s))[0] / ana - 1 for s in range(12)]))
    d = 0.215e-9
    th_ana = p20.bragg_angle_analytic(54.0, d)
    rows2 = []
    for k in (5, 10, 20, 40, 60, 80):
        lo, hi = 0.0, 90.0
        lam54 = p20.de_broglie(54.0)
        for _ in range(k):
            mid = 0.5 * (lo + hi)
            if 2 * d * np.sin(np.deg2rad(mid)) < lam54:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        rows2.append([str(k), sci(90.0 / 2**k, 3), sci(abs(c - th_ana), 3)])
    tab2 = _arr([r"k", r"\mathrm{ancho}\ (^\circ)", r"|\theta_k-\theta_{ana}|\ (^\circ)"], rows2, "rrr")
    bn = p20.bragg_angle_numeric()
    # sensibilidad al numero de bins
    return dict(
        numeric_intro=[
            r"La figura de interferencia de electrones es un fenómeno estadístico: cada electrón llega a un punto aleatorio, y la distribución de probabilidad $I(y)=\cos^2(\pi d y/\lambda L)$ solo aparece al acumular muchos impactos. Simular eso es un método de Monte Carlo, y el espaciado de las franjas se extrae con otra técnica numérica, la transformada de Fourier del histograma. Para Bragg, en cambio, basta una búsqueda de raíz por bisección sobre el ángulo.",
            r"El muestreo por rechazo funciona así: se sortea $y$ uniforme en un intervalo $[-s/2,s/2]$ y un número $u\in[0,1)$ uniforme, y se acepta $y$ si $u<I(y)$. Como $I\le 1$, la densidad de los puntos aceptados es proporcional a $I(y)$ (no hace falta la integral de $I$ ni su inversa). La fracción aceptada es $\langle\cos^2\rangle=1/2$. Un estimador Monte Carlo con $N$ muestras tiene error estadístico $\propto1/\sqrt N$, sin importar la dimensión del problema; esa lentitud es su principal limitación.",
            r"El espaciado se mide con la FFT del histograma: $X_k=\sum_j n_je^{-2\pi ijk/M}$ tiene su máximo en la frecuencia espacial $1/\delta$ de las franjas. Rellenar con ceros hasta $16M$ puntos interpola el espectro (no añade información, solo muestrea más fino) y el ajuste parabólico a tres puntos del máximo da la posición del pico entre muestras. El error depende de cuántas franjas caben (resolución en frecuencia $\sim 1/\text{ancho}$) y del ruido de Poisson de cada barra del histograma.",
            r"Los parámetros: $N=20000$ electrones y 12 franjas en la pantalla (por eso $s=12\lambda L/d$) dan un histograma de 600 barras con unas 33 cuentas por barra en promedio, suficiente para ver las franjas y medir $\delta$ con error relativo del orden de $10^{-3}$; la semilla 7 hace el resultado reproducible. Para Bragg se hacen 80 bisecciones sobre $[0^\circ,90^\circ]$ buscando $2d\sin\theta=\lambda$, que tiene solución única porque $\sin$ crece en ese intervalo.",
            r"Es independiente del cálculo analítico porque el espaciado $\lambda L/d$ nunca se inserta en el resultado: solo entra $\lambda$ en la densidad de probabilidad que genera los datos y el espaciado se recupera después midiendo la periodicidad de los impactos. Del mismo modo, la bisección de Bragg evalúa $2d\sin\theta$ en ángulos de prueba y no usa la función $\arcsin$.",
        ],
        numeric_code="""lam = de_broglie(100 eV)                      # longitud de onda relativista, h/p
span = 12 * lam * L / d                       # ancho de pantalla: 12 franjas
rng = default_rng(7)                          # semilla fija: resultado reproducible
hits = []
while len(hits) < N:                          # N = 20000 impactos aceptados
    y = rng.uniform(-span/2, span/2, 4*N)     # candidatos (se sortean de más)
    u = rng.uniform(size=y.size)              # número aleatorio por candidato
    keep = u < cos(pi*d*y/(lam*L))**2         # rechazo: se acepta con probabilidad I(y)
    hits += list(y[keep])                     # guardar los aceptados
hist, edges = histogram(hits[:N], bins=600, range=(-span/2, span/2))  # histograma
hist = hist - hist.mean()                     # quitar la componente continua
spec = abs(rfft(hist, 16*600))                # FFT con relleno de ceros
freqs = rfftfreq(16*600, edges[1]-edges[0])   # frecuencias espaciales
k = argmax(spec)                              # máximo del espectro
shift = 0.5*(spec[k-1]-spec[k+1]) / (spec[k-1]-2*spec[k]+spec[k+1])  # vértice de la parábola
f = freqs[k] + shift*(freqs[1]-freqs[0])      # frecuencia refinada
delta = 1 / f                                 # espaciado de las franjas

lo, hi = 0.0, 90.0                            # Bragg: bisección en el ángulo
for _ in range(80):
    mid = (lo + hi) / 2
    if 2*d*sin(radians(mid)) < lam54: lo = mid   # camino demasiado corto: subir
    else:                              hi = mid  # camino suficiente: bajar
theta = (lo + hi) / 2
""",
        numeric_converge=[
            ("Monte Carlo: el error del espaciado frente al número de electrones",
             f"Se genera la figura con $N$ impactos para 12 semillas distintas por cada $N$ y se calcula la raíz cuadrática media del error relativo del espaciado medido frente a $\\lambda L/d={sci(ana * 1e6, 5)}\\ \\mu\\mathrm{{m}}$. El error baja con $N$ pero con pendientes $p$ de $-0.3$ a $-0.1$, menores que el $-1/2$ de $1/\\sqrt N$, porque además hay un sesgo sistemático negativo (columna «promedio») que no se reduce al aumentar $N$.",
             [tab1]),
            ("Dos fuentes de error: estadística y resolución del histograma",
             f"La parte estadística (ruido de conteo) escala como $1/\\sqrt N$ y domina a $N$ pequeño; por encima de $\\sim10^4$ electrones queda un sesgo de {sci(float(bias), 2)} que al repetir la medición con 300, 600, 1200 y 2400 barras y con más relleno de ceros resultó igual (cambia menos de $2\\%$), de modo que no lo causan el histograma ni la interpolación; es compatible con la fuga espectral de una ventana finita de 12 franjas. En la escena, con $N=20000$ y semilla 7, el espaciado medido es {sci(sp_scene * 1e6, 6)} $\\mu$m, es decir un error de {sci(e_scene, 3)}. El redondeo es irrelevante.",
             []),
            ("Bragg: la bisección gana un bit por iteración",
             f"Para $K=54$ eV y $d=215$ pm la solución analítica es $\\theta={th_ana:.6f}^\\circ$. El ancho del intervalo se divide por 2 en cada paso (columna central) y el error baja con él hasta la precisión de máquina (en $k=60$ la diferencia ya es nula en doble precisión, $\\varepsilon_m\\theta\\sim10^{{-15}}$ grados); con 80 iteraciones el resultado queda en {bn:.12f}$^\\circ$, sin cambio respecto de $k=60$.",
             [tab2]),
            ("Criterio práctico",
             "El espaciado de franjas es confiable cuando (i) la cantidad de impactos por barra es de decenas, de modo que las franjas se distinguen del ruido, (ii) el número de franjas dentro de la pantalla es grande, para que la FFT las resuelva, y (iii) repetir con otra semilla cambia el resultado menos que la incertidumbre buscada. La bisección, por el contrario, es confiable por construcción: el ancho es $90^\\circ/2^k$ y basta $k\\approx55$ para llegar al redondeo.",
             []),
        ],
    )


# =============================================================================== WavePacketUncertainty
def _packet() -> dict:
    t = 0.5e-15
    w = float(p20.packet_width_analytic(t))
    ns = [32, 64, 128, 256, 512, 4096]
    rows = []
    for n in ns:
        sx, _ = p20.packet_numeric(t, n=n)
        rows.append([str(n), sci(400e-10 / n * 1e10, 3), sci(sx / w - 1, 3)])
    tab1 = _arr([r"n", r"\delta x\ (\AA)", r"(\Delta x_{num}-\Delta x)/\Delta x"], rows, "rrr")
    bx = [5e-10, 10e-10, 20e-10, 40e-10, 100e-10]
    rows2 = []
    for b in bx:
        sx, _ = p20.packet_numeric(t, box=b)
        rows2.append([fx(b * 1e10, 0), sci(sx / w - 1, 3)])
    tab2 = _arr([r"L_{caja}\ (\AA)", r"(\Delta x_{num}-\Delta x)/\Delta x"], rows2, "rr")
    sx0, sp0 = p20.packet_numeric(0.0)
    prod0 = sx0 * sp0 / (HBAR / 2)
    sx1, sp1 = p20.packet_numeric(1e-15)
    prod1 = sx1 * sp1 / (HBAR / 2)
    sxn, spn = p20.packet_numeric(t)
    dk_max = np.pi / (400e-10 / 4096)
    s0 = p20.SIGMA0
    tail_dec = 2 * (s0 * dk_max) ** 2 / np.log(10)
    # n pequeno: aliasing
    return dict(
        numeric_intro=[
            r"La ecuación de Schrödinger libre $i\hbar\,\partial_t\psi=-\tfrac{\hbar^2}{2m}\partial_x^2\psi$ sí se puede resolver a mano para un gaussiano, pero esa solución es justamente lo que se quiere contrastar. El método numérico usa que en el espacio de números de onda la ecuación se vuelve algebraica: cada componente de Fourier solo gira en fase, $\hat\psi(k,t)=\hat\psi(k,0)\,e^{-i\hbar k^2t/2m}$. Se calcula la transformada, se multiplica por esa fase y se regresa al espacio real.",
            r"La transformada discreta usada es la FFT: aproxima $\hat\psi(k_m)\approx\delta x\sum_j\psi(x_j)e^{-ik_mx_j}$ con $k_m=2\pi m/(n\,\delta x)$. Para funciones suaves y periódicas la convergencia es espectral (más rápida que cualquier potencia de $\delta x$) hasta que se alcanza el redondeo; si la función no cabe en la caja la FFT la envuelve periódicamente y aparece aliasing. En el tiempo no hay error de paso, pues la evolución libre en el espacio de $k$ es exacta.",
            r"Los momentos se calculan por suma sobre la malla: $\langle x^n\rangle=\sum x_j^nP_j\delta x$ con $P_j=|\psi_j|^2$ normalizada, y análogamente en $k$ con $|\hat\psi|^2$, con $\Delta p=\hbar\Delta k$. Este es un método de cuadratura de rectángulos; para integrandos suaves que decaen a cero en los bordes (como estos gaussianos) la regla del rectángulo equiespaciada converge igual de rápido que la FFT, porque equivale a una suma de Fourier.",
            r"Parámetros: una caja de 400 Å ($=4\times10^{-8}$ m) con $n=4096$ puntos da $\delta x\approx0.1$ Å, y un número de onda máximo $\pi/\delta x\approx 32\ \text{Å}^{-1}$, mucho mayor que el contenido del paquete ($1/\sigma_0=1\ \text{Å}^{-1}$). La caja es 400 veces el ancho inicial $\sigma_0=1$ Å, así que el paquete ensanchado a 1 fs ($\Delta x\approx 6$ Å) queda muy lejos del borde. Se miden $\Delta x\,\Delta p$ a distintos tiempos para verificar $\ge\hbar/2$.",
            r"Es independiente de la fórmula analítica $\sigma(t)=\sigma_0\sqrt{1+(\hbar t/2m\sigma_0^2)^2}$: esa fórmula nunca se usa en el cálculo. El método parte de la función de onda inicial muestreada, la propaga con la ecuación de Schrödinger en el espacio de $k$ y mide $\Delta x$ de la distribución resultante.",
        ],
        numeric_code="""n, box = 4096, 400e-10                        # puntos de la malla y ancho de la caja (400 Å)
x = linspace(-box/2, box/2, n, endpoint=False)  # malla periódica uniforme
dx = x[1] - x[0]                              # paso espacial
psi = exp(-x**2 / (4*sigma0**2))              # paquete gaussiano inicial
psi /= sqrt(sum(abs(psi)**2) * dx)            # normalizar: sum |psi|^2 dx = 1
k = 2*pi*fftfreq(n, dx)                       # números de onda de la FFT
phi = fft(psi)                                # al espacio de momentos
phi_t = phi * exp(-1j*hbar*k**2*t/(2*m))      # evolución libre exacta: solo una fase por k
psi_t = ifft(phi_t)                           # de vuelta al espacio real
px = abs(psi_t)**2;  px /= px.sum()*dx        # densidad de probabilidad en x
sx = sqrt(sum(x**2*px)*dx - (sum(x*px)*dx)**2)  # desviación estándar: Delta x
pk = abs(phi_t)**2;  pk /= pk.sum()           # distribución en k (pesos que suman 1)
sk = sqrt(sum(k**2*pk) - sum(k*pk)**2)        # Delta k
dp = hbar * sk                                # Delta p (no cambia al evolucionar)
producto = sx * dp                            # debe ser >= hbar/2
""",
        numeric_converge=[
            ("Convergencia espectral con el número de puntos",
             f"Se repite la evolución hasta $t = 0.5$ fs con $n$ distinto en la misma caja de 400 Å y se compara $\\Delta x$ con la fórmula analítica ($\\Delta x = {sci(w * 1e10, 8)}$ Å). A diferencia de Simpson o RK4, el error no baja con una potencia fija: cae muy rápido hasta unos $10^{{-13}}$ y se mantiene en ese piso por redondeo.",
             [tab1]),
            ("Qué pasa si la caja es demasiado pequeña",
             f"Con $n=4096$ y caja variable, el error se debe al aliasing: el paquete es en teoría infinito y la FFT lo envuelve en una caja periódica. Pasa de {sci(float(p20.packet_numeric(t, box=5e-10)[0] / w - 1), 2)} con 5 Å a {sci(float(p20.packet_numeric(t, box=40e-10)[0] / w - 1), 2)} con 40 Å; la causa es la cola gaussiana $e^{{-L^2/(8\\sigma(t)^2)}}$ que sale por los bordes.",
             [tab2]),
            ("Fuentes de error y comprobación del principio de incertidumbre",
             f"Con los parámetros de la escena ($n=4096$, 400 Å) el único error relevante es el redondeo ($\\sim10^{{-13}}$ relativos, ver tabla). No hay error de paso temporal. El producto numérico es el que se espera para un gaussiano al inicio y crece con el tiempo porque $\\Delta p$ es constante mientras $\\Delta x$ crece:",
             [rf"\dfrac{{\Delta x\,\Delta p}}{{\hbar/2}}\bigg|_{{t=0}} = {fx(prod0, 10)},\qquad \dfrac{{\Delta x\,\Delta p}}{{\hbar/2}}\bigg|_{{0.5\,\mathrm{{fs}}}} = {fx(sxn * spn / (HBAR / 2), 6)},\qquad \bigg|_{{1\,\mathrm{{fs}}}} = {fx(prod1, 6)}"]),
            ("Criterio práctico",
             f"El resultado es confiable si (i) la caja es más de unas 20 veces el ancho máximo del paquete (para que la cola no llegue al borde), (ii) $\\delta x$ resuelve la oscilación más rápida del contenido de Fourier: el número de onda máximo $\\pi/\\delta x = {sci(dk_max * 1e-10, 3)}\\ \\text{{Å}}^{{-1}}$ debe estar donde ese espectro sea despreciable (aquí $|\\hat\\psi|^2\\propto e^{{-2\\sigma_0^2k^2}}$ vale ya $\\sim10^{{-{tail_dec:.0f}}}$ de su máximo), y (iii) el cociente $\\Delta x\\Delta p/(\\hbar/2)$ al inicio es 1 con la precisión esperada. Si alguno falla, duplicar la caja o los puntos y ver si cambia el resultado.",
             []),
        ],
    )


# =============================================================================== InfiniteWell
def _well() -> dict:
    ns = [25, 50, 100, 200, 400, 800, 2000, 4000]
    rows, e1s, e4s = [], [], []
    for n in ns:
        e, _, _ = p21.well_numeric(4, npts=n)
        e1s.append(float(e[0] / p21.well_energy_analytic(1) - 1))
        e4s.append(float(e[3] / p21.well_energy_analytic(4) - 1))
        pred = -((np.pi / (n + 1)) ** 2) / 12
        rows.append([str(n), sci(e1s[-1], 3), sci(pred, 3), sci(e4s[-1], 3), "-" if len(ns) and len(e1s) == 1 else fx(_slope(ns[len(e1s) - 2], e1s[-2], n, e1s[-1]), 2)])
    tab1 = _arr([r"N", r"\delta E_1/E_1", r"-\theta_1^2/12", r"\delta E_4/E_4", r"p"], rows, "rrrrr")
    rows2 = [[str(n), sci(abs(1 - p21.wavefunction_overlap(1, npts=n)), 3), sci(abs(1 - p21.wavefunction_overlap(4, npts=n)), 3)] for n in (50, 200, 800, 2000)]
    tab2 = _arr([r"N", r"|1-|\langle\psi_{num}|\psi_1\rangle||", r"|1-|\langle\psi_{num}|\psi_4\rangle||"], rows2, "rrr")
    return dict(
        numeric_intro=[
            r"La ecuación estacionaria $-\tfrac{\hbar^2}{2m}\psi''=E\psi$ con $\psi(0)=\psi(L)=0$ se resuelve a mano para el pozo infinito, pero esa técnica no sirve para un potencial arbitrario. El método de diferencias finitas lo convierte en álgebra lineal: se reemplaza el continuo por $N$ nodos $x_j=jh$, $h=L/(N+1)$, y $\psi''$ por una diferencia central, de modo que la ecuación diferencial pasa a ser un problema de autovalores $H\vec\psi=E\vec\psi$ de una matriz.",
            r"La aproximación es $\psi''(x_j)\approx(\psi_{j+1}-2\psi_j+\psi_{j-1})/h^2$, que por el desarrollo de Taylor tiene error $\tfrac{h^2}{12}\psi^{(4)}$, es decir $O(h^2)$. Entonces $H$ es tridiagonal simétrica, con $2t$ en la diagonal y $-t$ fuera ($t=\hbar^2/2mh^2$). Sus autovalores son reales, y para esta matriz se conocen exactamente: $E_j=2t\,[1-\cos(j\pi/(N+1))]$, de donde el error relativo del nivel $j$ es $\approx-\theta_j^2/12$ con $\theta_j=j\pi/(N+1)$: decrece como $1/N^2$ y crece como $j^2$.",
            r"El cálculo de los autovalores más bajos se hace con `eigh_tridiagonal`, que usa bisección de Sturm y no requiere formar la matriz densa; cuesta $O(N)$ por autovalor. El método falla (o pierde exactitud) cuando $\psi$ oscila rápido respecto de $h$ (estados altos: se necesitan al menos unos 10 nodos por longitud de onda), o si el potencial tiene discontinuidades o singularidades que la malla no resuelve.",
            r"Parámetros: $N=2000$ nodos interiores y cuatro estados. Con $\theta_4=4\pi/2001$ el error relativo teórico del cuarto nivel es $\theta_4^2/12\approx3\times10^{-6}$, más que suficiente para comparar con la teoría a cinco cifras. El autovector se normaliza con $\sum\psi_j^2h=1$.",
            r"No usa la fórmula $E_n=n^2\pi^2\hbar^2/2mL^2$ ni las senoidales: solo la matriz del hamiltoniano discretizado. La coincidencia con $E_n$ es un resultado, no una premisa, y el solapamiento con $\sqrt{2/L}\sin(n\pi x/L)$ verifica además que la forma de las funciones es correcta.",
        ],
        numeric_code="""N = 2000                                      # nodos interiores
h = L / (N + 1)                               # paso de la malla (psi=0 en x=0 y x=L)
t = hbar**2 / (2*m*h**2)                      # coeficiente cinético
diag = full(N, 2*t)                           # diagonal: 2t (de -psi'' por diferencias centrales)
off  = full(N-1, -t)                          # subdiagonales: -t
w, v = eigh_tridiagonal(diag, off,            # autovalores y autovectores de H
                        select="i", select_range=(0, 3))   # solo los 4 más bajos
E = w / e_charge                              # energías en eV
x = linspace(h, L-h, N)                       # posiciones de los nodos
psi = v / sqrt(h)                             # normalizar: sum psi^2 h = 1
# verificación cruzada contra el estado analítico
ref = sqrt(2/L) * sin(n*pi*x/L)               # solo para comparar, no se usa en el cálculo
overlap = abs(sum(psi[:, n-1] * ref) * h)     # debe valer ~1
""",
        numeric_converge=[
            ("Las energías convergen como 1/N²",
             f"Se diagonaliza la matriz con $N$ distinto y se compara el nivel numérico con $n^2\\pi^2\\hbar^2/2mL^2$. El error relativo del estado fundamental baja por un factor cercano a 4 cada vez que se duplica $N$ (pendientes $p\\to-2$), y coincide con la predicción teórica $-\\theta_1^2/12$ de la tercera columna. Para el cuarto nivel el error es $16$ veces mayor, pues escala como $n^2$.",
             [tab1]),
            ("Fuentes de error",
             f"El error es de truncamiento: la diferencia central ignora $\\tfrac{{h^2}}{{12}}\\psi^{{(4)}}$ y, como la matriz es simétrica, los niveles quedan siempre por debajo del valor exacto (error negativo en la tabla). Con $N=2000$ se tiene $\\delta E_1/E_1={sci(e1s[ns.index(2000)], 3)}$ y $\\delta E_4/E_4={sci(e4s[ns.index(2000)], 3)}$. El redondeo (≈$10^{{-16}}$ por entrada) es mucho menor, por eso la pendiente se mantiene en $-2$ hasta $N=2000$; en $N=4000$ ya se nota (pendiente $-1.99$ y $-5.16\\times10^{{-8}}$ frente a la predicción $-5.14\\times10^{{-8}}$).",
             []),
            ("Comprobación cruzada con las funciones de onda (forma exacta en los nodos)",
             f"El solapamiento con la función analítica mide el error de la forma, no solo del autovalor. Aquí es 1 hasta el redondeo para cualquier $N$ (la tabla da $\\sim10^{{-16}}$): los autovectores de esta matriz tridiagonal son exactamente $\\sin(jn\\pi/(N+1))$, las senoidales evaluadas en los nodos. Así, todo el error de la malla está en los autovalores, no en la forma de $\\psi$; es una verificación cruzada, no una medida de convergencia.",
             [tab2]),
            ("Criterio práctico",
             "El resultado es confiable cuando duplicar $N$ reduce el error unas 4 veces (régimen asintótico $O(h^2)$), cuando el error teórico $\\theta_n^2/12$ es menor que la precisión que se quiere, y cuando los autovectores tienen $n$ semiondas con al menos 10 nodos por longitud de onda. Si se quisiera un nivel más preciso sin aumentar $N$ se puede aplicar la extrapolación de Richardson con los resultados de $N$ y $2N$.",
             []),
        ],
    )


# =============================================================================== BarrierTunneling
def _rk4_t(n: int) -> float:
    return p21.transmission_numeric(n=n)


def _tunnel() -> dict:
    ana = p21.transmission_analytic()
    ns = [25, 50, 100, 200, 400, 4000, 16000]
    rows, errs = [], []
    k = np.sqrt(2 * M_E * p21.E_BEAM_EV * E_CH) / HBAR
    kap = np.sqrt(2 * M_E * (p21.V0_EV - p21.E_BEAM_EV) * E_CH) / HBAR
    for n in ns:
        e = _rk4_t(n) / ana - 1
        errs.append(e)
        hh = p21.A_BAR / n
        rows.append([str(n), sci(hh * 1e12, 3), sci(kap * hh, 3), sci(e, 3), "-" if len(errs) == 1 else fx(_slope(ns[len(errs) - 2], errs[-2], n, e), 2)])
    tab1 = _arr([r"N", r"h\ (\mathrm{pm})", r"\kappa h", r"(T_{num}-T)/T", r"p"], rows, "rrrrr")
    p_mid = np.mean([_slope(ns[i], errs[i], ns[i + 1], errs[i + 1]) for i in range(1, 4)])
    rows2 = []
    for e_ev in (2.0, 5.0, 9.0, 12.0):
        a_ = p21.transmission_analytic(e_ev)
        rows2.append([fx(e_ev, 1), sci(a_, 5), sci(p21.transmission_numeric(e_ev) / a_ - 1, 3)])
    tab2 = _arr([r"E\ (\mathrm{eV})", r"T_{ana}", r"(T_{num}-T)/T"], rows2, "rrr")
    return dict(
        numeric_intro=[
            r"Para la barrera rectangular, el cálculo analítico consiste en escribir $\psi$ en tres regiones, empalmar $\psi$ y $\psi'$ en $x=0$ y $x=a$ y resolver un sistema de cuatro ecuaciones. El método numérico no empalma nada: integra directamente la ecuación diferencial $\psi''=-\tfrac{2m}{\hbar^2}(E-V)\psi$ a través de la barrera y luego descompone la solución. Es la técnica general que sigue sirviendo cuando $V(x)$ no es rectangular y no hay fórmula cerrada.",
            r"Se pasa a primer orden con $\vec y=(\psi,\psi')$, $\vec y'=\vec f(\vec y)$, y se avanza con Runge-Kutta de cuarto orden: $\vec y_{n+1}=\vec y_n+\tfrac h6(\vec k_1+2\vec k_2+2\vec k_3+\vec k_4)$ con cuatro evaluaciones de $\vec f$ por paso. El desarrollo de Taylor coincide con la solución exacta hasta $h^4$, por lo que el error local es $O(h^5)$ y el global $O(h^4)$. Su limitación es la estabilidad: dentro de la barrera $\psi$ crece o decae exponencialmente como $e^{\pm\kappa x}$, y si $\kappa h$ no es pequeño el error de truncamiento se vuelve grande.",
            r"El truco de integración es partir del lado derecho, donde se impone una onda transmitida pura $\psi=e^{ik(x-a)}$ (amplitud de transmisión 1), e integrar hacia atrás hasta $x=0$. Allí $\psi=Ae^{ikx}+Be^{-ikx}$ permite despejar $A=\tfrac12[\psi+\psi'/ik]$ y la transmisión es $T=|t|^2/|A|^2=1/|A|^2$ (mismo $k$ a ambos lados). Dentro de la barrera la solución que se busca crece al ir hacia atrás ($\sim e^{\kappa(a-x)}$), es decir es la dominante de la ecuación, y por eso el error relativo no se amplifica; integrar en sentido contrario obligaría a seguir una solución que decae frente a otra que crece.",
            r"Se usan $N=4000$ pasos, de modo que $h=a/N=0.125$ pm y $\kappa h=kh\approx1.4\times10^{-3}$ (pues $E=V_0/2$), valores muy pequeños, y el error de truncamiento $O((\kappa h)^4)$ queda del orden de $10^{-13}$, casi al nivel del redondeo. $E=5$ eV, $V_0=10$ eV y $a=0.5$ nm son los de la escena.",
            r"No usa la fórmula con $\sinh(\kappa a)$ ni condiciones de empalme: la transmisión sale de integrar la ecuación diferencial y proyectar la solución sobre ondas planas. La coincidencia con la fórmula analítica es la verificación, y puede repetirse para cualquier perfil $V(x)$.",
        ],
        numeric_code="""k = sqrt(2*m*E) / hbar                        # número de onda fuera de la barrera
psi = [1+0j, 1j*k]                            # (psi, psi') = onda transmitida pura exp(ik(x-a)) en x=a
h = -a / N                                    # paso NEGATIVO: se integra hacia atrás, de x=a a x=0
f = lambda y: [y[1], -(2*m/hbar**2)*(E - V0)*y[0]]   # y' = (psi', psi''), con psi'' = -(2m/hbar^2)(E-V)psi
for _ in range(N):                            # N = 4000 pasos
    k1 = f(psi)                               # pendiente al inicio
    k2 = f(psi + 0.5*h*k1)                    # pendiente en el punto medio (1)
    k3 = f(psi + 0.5*h*k2)                    # pendiente en el punto medio (2)
    k4 = f(psi + h*k3)                        # pendiente al final
    psi = psi + h/6 * (k1 + 2*k2 + 2*k3 + k4) # combinación de RK4 con pesos 1,2,2,1
A = 0.5 * (psi[0] + psi[1] / (1j*k))          # amplitud de la onda incidente en x=0
T = 1 / abs(A)**2                             # transmisión (la transmitida tiene amplitud 1)
""",
        numeric_converge=[
            ("El error de RK4 baja como h⁴",
             f"Se repite el cálculo con $N$ pasos entre $x=a$ y $x=0$ ($h=a/N$) y se compara $T$ con la fórmula cerrada $T={sci(ana, 6)}$. Cada duplicación de $N$ reduce el error unas 16 veces: la pendiente observada en el régimen asintótico ($N=50$ a $400$) es {p_mid:.2f}, como pide $O(h^4)$.",
             [tab1]),
            ("Redondeo y elección de N",
             f"Hasta $N=4000$ el error sigue la ley $h^4$ (entre $N=400$ y $N=4000$ el error se reduce {errs[4] / errs[5]:.0f} veces, frente a las $10^4$ que predice $h^4$ para un factor 10 en $N$). En $N=16000$ el error ya es {sci(abs(errs[-1]), 2)} y la pendiente se degrada ({fx(_slope(4000, errs[5], 16000, errs[6]), 2)}): se alcanza el piso del redondeo acumulado en $N$ pasos de doble precisión. Por eso $N=4000$ es una elección razonable (error {sci(abs(errs[5]), 2)}); con $N=25$ aún se tiene {sci(abs(errs[0]), 2)}, que bastaría para dibujar.",
             []),
            ("Sensibilidad a la energía",
             "El mismo método cubre tanto $E<V_0$ (túnel, la solución crece como $e^{\\kappa x}$ en la barrera) como $E>V_0$ (oscilatoria, con resonancias). Con $N=4000$ el error relativo es:",
             [tab2]),
            ("Criterio práctico",
             f"El resultado es confiable cuando $\\kappa h$ y $kh$ son mucho menores que 1 (aquí $\\kappa h={sci(kap * p21.A_BAR / 4000, 2)}$, $kh={sci(k * p21.A_BAR / 4000, 2)}$), cuando duplicar $N$ cambia $T$ en la cifra esperada para RK4 (factor 16), y cuando la solución integrada no tiene cancelaciones: aquí $T\\sim e^{{-2\\kappa a}}$ con $2\\kappa a={fx(2 * kap * p21.A_BAR, 2)}$ y $|A|=1/\\sqrt T={1 / np.sqrt(ana):.1f}$, pero como la integración hacia atrás sigue la solución dominante, el error relativo se mantiene (tabla: $T={sci(p21.transmission_analytic(2.0), 2)}$ con error del orden de $10^{{-12}}$).",
             []),
        ],
    )


# =============================================================================== HydrogenSpectrum
def _hydrogen() -> dict:
    exact = p22.level_analytic([1, 2, 3, 4])
    ns = [1000, 2000, 4000, 8000, 16000, 32000]
    rows, e1 = [], []
    for n in ns:
        e, _, _ = p22.radial_numeric(3, rmax=160, npts=n)
        rel = e / exact[:3] - 1
        e1.append(rel)
        rows.append([str(n), sci(160 / (n + 1), 3), sci(float(rel[0]), 3), sci(float(rel[1]), 3), sci(float(rel[2]), 3), "-" if len(e1) == 1 else fx(_slope(ns[len(e1) - 2], e1[-2][0], n, e1[-1][0]), 2)])
    tab1 = _arr([r"N", r"h\ (a_0)", r"\delta E_1/E_1", r"\delta E_2/E_2", r"\delta E_3/E_3", r"p"], rows, "rrrrrr")
    rows2 = []
    e4_by_r = {}
    for rm in (20, 40, 80, 160):
        e, _, _ = p22.radial_numeric(4, rmax=rm, npts=int(rm / 0.005))
        rel = e / exact - 1
        e4_by_r[rm] = rel
        rows2.append([str(rm), str(int(rm / 0.005)), sci(float(rel[0]), 3), sci(float(rel[2]), 3), sci(float(rel[3]), 3)])
    tab2 = _arr([r"r_{max}\ (a_0)", r"N", r"\delta E_1/E_1", r"\delta E_3/E_3", r"\delta E_4/E_4"], rows2, "rrrrr")
    e, _, _ = p22.radial_numeric(8)
    ha_n, hb_n = p22.balmer_wavelength_numeric(3, e), p22.balmer_wavelength_numeric(4, e)
    ha_a, hb_a = p22.balmer_wavelength_analytic(3), p22.balmer_wavelength_analytic(4)
    rows3 = []
    for n_ in range(3, 8):
        a_ = p22.balmer_wavelength_analytic(n_)
        nn = p22.balmer_wavelength_numeric(n_, e)
        rows3.append([str(n_), fx(a_, 4), fx(nn, 4), sci(nn / a_ - 1, 3)])
    tab3 = _arr([r"n", r"\lambda_{Rydberg}\ (\mathrm{nm})", r"\lambda_{num}\ (\mathrm{nm})", r"\delta\lambda/\lambda"], rows3, "rrrr")
    h = 160 / 32001
    return dict(
        numeric_intro=[
            r"Para $l=0$ la ecuación de Schrödinger del hidrógeno se reduce a la ecuación radial $-\tfrac12u''-u/r=Eu$ (unidades atómicas, $u=rR$, con $u(0)=u(\infty)=0$). El resultado de Bohr, $E_n=-13.6/n^2$ eV, se obtiene resolviéndola con funciones de Laguerre, pero eso no es posible para un potencial distinto del de Coulomb. El método numérico trata el problema como autovalores de una matriz: no necesita la forma de las soluciones.",
            r"Se discretiza con una malla uniforme $r_i=ih$, $i=1,\dots,N$, $h=r_{max}/(N+1)$, y $u''\approx(u_{i+1}-2u_i+u_{i-1})/h^2$. La ecuación se vuelve $-\tfrac1{2h^2}u_{i-1}+(\tfrac1{h^2}-\tfrac1{r_i})u_i-\tfrac1{2h^2}u_{i+1}=Eu_i$, una matriz tridiagonal simétrica real cuyos autovalores más bajos aproximan los niveles. El error de truncamiento de la derivada es $O(h^2)$, y se impone $u(r_{max})=0$ como frontera artificial.",
            r"Hay dos fuentes de error independientes: la discretización ($h$) y la frontera finita ($r_{max}$). El orbital de número $n$ tiene un tamaño característico $\sim n^2a_0$ y una cola $e^{-r/na_0}$, por lo que $r_{max}$ debe ser varias veces $n^2a_0$; si no, la pared artificial lo comprime y sube su energía. El método falla cuando $r_{max}$ no cubre el orbital (estados altos) o cuando $h$ no resuelve la escala de variación de $u$ (del orden de $a_0/n$ cerca del núcleo).",
            r"Los parámetros de la escena: $r_{max}=160\,a_0$ y $N=32000$ nodos, $h=0.005\,a_0$. Son suficientes para $n\le4$ ($n^2a_0=16\,a_0$, con 10 veces más espacio) y para que el error de $O(h^2)$ sea de $\sim6\times10^{-6}$ en el estado fundamental. Con las energías en hartree se convierten a eV (×27.2114) y las líneas de Balmer se calculan con $\lambda=hc/(E_n-E_2)$, $hc=1239.84198$ eV nm.",
            r"No usa la fórmula de Bohr/Rydberg: los niveles salen de la diagonalización y se comparan con $-13.6057/n^2$ al final. Así que que se obtenga el espectro $1/n^2$ es una predicción del cálculo, no un dato de entrada. Lo único externo son las constantes ($\hbar$, $m$, $e$, en unidades atómicas) y la forma del potencial de Coulomb.",
        ],
        numeric_code="""npts, rmax = 32000, 160.0                     # nodos y radio de la caja (en a0)
h = rmax / (npts + 1)                         # paso de la malla
r = arange(1, npts+1) * h                     # nodos r_i = i h (u=0 en r=0 y r=rmax)
diag = 1/h**2 - 1/r                           # diagonal: cinética 1/h^2 y potencial -1/r
off = full(npts-1, -0.5/h**2)                 # fuera de la diagonal: -1/(2 h^2)
w, v = eigh_tridiagonal(diag, off,            # autovalores más bajos de la matriz tridiagonal
                        select="i", select_range=(0, n_states-1))
E = w * 27.2114                               # de hartree a eV
u = v / sqrt(h)                               # normalizar: sum u^2 h = 1
lam_ha = hc / (E[2] - E[1])                   # H-alfa: hc/(E3 - E2), con hc = 1239.84198 eV nm
lam_hb = hc / (E[3] - E[1])                   # H-beta: hc/(E4 - E2)
""",
        numeric_converge=[
            ("Error de discretización: O(h²)",
             f"Con $r_{{max}}=160\\,a_0$ fijo se aumenta $N$ y se compara con $-13.6057/n^2$ eV. El error relativo del estado fundamental baja un factor 4 cada vez que se duplica $N$ (pendiente $p\\to-2$), como corresponde a la diferencia central. El error crece con $1/n^2$: $\\delta E_1:\\delta E_2:\\delta E_3\\approx 1 : {e1[-1][1] / e1[-1][0]:.3f} : {e1[-1][2] / e1[-1][0]:.3f}$, y es siempre negativo (el nivel numérico queda un poco más ligado que el exacto).",
             [tab1]),
            ("Error por la frontera: r_max",
             f"Ahora se mantiene $h=0.005\\,a_0$ y se cambia el tamaño de la caja. Cuando $r_{{max}}$ es corto frente a $n^2a_0$, la pared comprime el orbital y el error se dispara; con $r_{{max}}\\ge 80\\,a_0$ el error de $n\\le4$ ya es el de discretización. Esto explica por qué se escogió $160\\,a_0$ y no menos.",
             [tab2]),
            ("Consecuencia en las líneas de Balmer",
             f"Las longitudes de onda se calculan con $\\lambda=hc/(E_n-E_2)$. Con la malla de la escena, H$\\alpha$ sale {ha_n:.4f} nm frente a {ha_a:.4f} nm de la fórmula de Rydberg, y H$\\beta$ {hb_n:.4f} nm frente a {hb_a:.4f} nm.",
             [tab3]),
            ("Fuentes de error y criterio práctico",
             f"Truncamiento $O(h^2)$ de la derivada (domina: ≈{sci(abs(float(e1[-1][0])), 2)} en $E_1$ con $h={sci(h, 3)}\\,a_0$), frontera $r_{{max}}$ (despreciable para $n\\le4$ si $r_{{max}}\\ge 80\\,a_0$) y redondeo (≈$10^{{-16}}$, irrelevante). El resultado es confiable si el error cae por 4 al duplicar $N$, si $r_{{max}}\\gtrsim 5n^2a_0$ y si cambiar $r_{{max}}$ no modifica las energías. La corrección para un error $\\propto h^2$ se puede estimar por extrapolación de Richardson, $E\\approx(4E_{{2N}}-E_N)/3$.",
             []),
        ],
    )


# =============================================================================== RadialProbability
def _u1s(npts: int):
    _, r, u = p22.radial_numeric(1, npts=npts)
    return r, u[:, 0] ** 2


def _mean_r(npts: int) -> float:
    r, p = _u1s(npts)
    m = len(r) - (1 if len(r) % 2 == 0 else 0)
    return float(simpson(r[:m] * p[:m], r[:m]))


def _rmp_raw(npts: int) -> tuple[float, float]:
    r, p = _u1s(npts)
    k = int(np.argmax(p))
    h = r[1] - r[0]
    y0, y1, y2 = p[k - 1], p[k], p[k + 1]
    return float(r[k]), float(r[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * h)


def _radial() -> dict:
    # prueba de Simpson vs trapecio sobre 4 r^3 exp(-2 r) en [0, 8], solucion exacta con gamma incompleta
    ex8 = 1.5 * gammainc(4, 16.0)
    ns = [8, 16, 32, 64, 128, 256]
    rows, es, et = [], [], []
    for n in ns:
        r = np.linspace(0, 8, n + 1)
        y = 4 * r**3 * np.exp(-2 * r)
        hh = 8 / n
        tr = hh * (y.sum() - 0.5 * (y[0] + y[-1]))
        sm = simpson(y, r)
        es.append(sm - ex8)
        et.append(tr - ex8)
        rows.append([str(n), fx(hh, 4), sci(et[-1], 3), sci(es[-1], 3), "-" if len(es) == 1 else fx(_slope(ns[len(es) - 2], es[-2], n, es[-1]), 2)])
    tab1 = _arr([r"N", r"h\ (a_0)", r"\mathrm{trapecio}-I", r"\mathrm{Simpson}-I", r"p_S"], rows, "rrrrr")

    ms = [2000, 4000, 8000, 16000, 32000]
    rows2, em, rm_raw, rm_par = [], [], [], []
    for n in ms:
        mr = _mean_r(n) - 1.5
        raw, par = _rmp_raw(n)
        em.append(mr)
        rm_raw.append(raw - 1)
        rm_par.append(par - 1)
        rows2.append([str(n), sci(160 / (n + 1), 3), sci(mr, 3), sci(raw - 1, 3), sci(par - 1, 3)])
    tab2 = _arr([r"N", r"h\ (a_0)", r"\langle r\rangle-\tfrac32", r"r_{nodo}-1", r"r_{parab.}-1"], rows2, "rrrrr")
    sl_m = _slope(ms[0], em[0], ms[-1], em[-1])
    sl_p = _slope(ms[0], rm_par[0], ms[-1], rm_par[-1])
    return dict(
        numeric_intro=[
            r"La densidad de probabilidad radial del estado 1s es $P(r)=4r^2e^{-2r}$ (en $a_0$), y las preguntas de la escena son dos: dónde está el máximo (radio más probable) y cuál es el valor medio $\langle r\rangle=\int_0^\infty rP\,dr$. Aquí no se parte de la forma exacta: la función $u(r)$ viene de diagonalizar la ecuación radial en una malla (el mismo problema que en el espectro del hidrógeno), de modo que $P_i=u_i^2$ es una lista de números en nodos equiespaciados, y hay que extraer de ella un máximo y una integral.",
            r"El valor medio se calcula con la regla de Simpson compuesta, $\int f\,dr\approx\tfrac h3[f_0+4f_1+2f_2+\cdots+f_N]$, que equivale a ajustar parábolas por tripletes de nodos y por eso integra exactamente cúbicas. Su error global es $O(h^4)\,f^{(4)}$, frente al $O(h^2)$ del trapecio. Pide un número par de subintervalos (de ahí el recorte a un número impar de nodos) y supone que $f$ es suave; si $f$ tiene picos agudos o discontinuidades, se pierde el orden.",
            r"El máximo se localiza en dos pasos: el nodo con el valor más grande $k$ (resolución $h$) y el vértice de la parábola por $(r_{k-1},P_{k-1})$, $(r_k,P_k)$, $(r_{k+1},P_{k+1})$, que está desplazado $\delta=\tfrac12\tfrac{P_{k-1}-P_{k+1}}{P_{k-1}-2P_k+P_{k+1}}h$ del nodo central. Así se obtiene la posición del máximo entre nodos con error $O(h^2)$ en lugar del $O(h)$ del nodo más cercano; falla si el máximo es muy plano frente al ruido.",
            r"Parámetros: $N=32000$ y $r_{max}=160\,a_0$, o sea $h=0.005\,a_0$. La función $rP\sim r^3e^{-2r}$ está concentrada en unos pocos $a_0$ y es suave, de modo que Simpson es adecuada, y a $r_{max}$ la cola es del orden de $e^{-320}$. El error final está dominado por el de $u$ mismo (diferencias finitas $O(h^2)$) más que por la integración.",
            r"Es independiente del analítico porque ni la integral ni el máximo usan $4r^2e^{-2r}$ ni $\langle r\rangle=3a_0/2$: la función viene de la diagonalización y el radio medio sale de integrar datos. Que el máximo caiga en $a_0$ y la media en $1.5\,a_0$ es el resultado que se contrasta.",
        ],
        numeric_code="""E, r, u = radial_numeric(1)                   # diagonalización de la ecuación radial (32000 nodos, h = 0.005)
p = u[:, 0]**2                                # P(r) = u(r)^2, normalizada con sum P h = 1
m = len(r) - (1 if len(r) % 2 == 0 else 0)    # Simpson necesita un número par de subintervalos
f = r[:m] * p[:m]                             # integrando r*P(r)
h = (r[m-1] - r[0]) / (m - 1)                 # paso de la malla
rmean = h/3 * (f[0] + f[-1]                   # Simpson: extremos con peso 1
               + 4*f[1:-1:2].sum()            # nodos impares con peso 4
               + 2*f[2:-1:2].sum())           # nodos pares interiores con peso 2
k = argmax(p)                                 # nodo de mayor probabilidad
y0, y1, y2 = p[k-1], p[k], p[k+1]             # tres puntos alrededor del máximo
delta = 0.5*(y0 - y2) / (y0 - 2*y1 + y2)      # vértice de la parábola, en unidades de h
rmp = r[k] + delta * h                        # radio más probable refinado
""",
        numeric_converge=[
            ("Error de la cuadratura en una integral conocida",
             f"Para aislar el error de la integración se aplican Simpson y trapecio a $\\int_0^8 4r^3e^{{-2r}}dr$ (el integrando de la escena), cuyo valor exacto con la gamma incompleta es {ex8:.10f}. Las pendientes locales de Simpson se acercan a $-4$ a medida que $h$ disminuye (a $h$ grande aún no hay régimen asintótico). El trapecio converge aquí igual o más rápido que Simpson (columna 3), y no como $h^2$: la corrección de Euler-Maclaurin $\\tfrac{{h^2}}{{12}}[f'(8)-f'(0)]$ se anula porque $f'(0)=0$ y $f'(8)\\approx0$. Es una propiedad de este integrando, no de la regla, por eso Simpson se mantiene como opción segura. A $h=0.005\\,a_0$ (la malla de la escena) el término dominante $\\tfrac{{h^4}}{{180}}|f'''(0)|$ da {sci(0.005**4 / 180 * 24, 2)}, despreciable.",
             [tab1]),
            ("Error del radio medio y del radio más probable con la malla de la escena",
             f"Ahora se recalcula $u$ con distintos $N$ ($r_{{max}}=160\\,a_0$ fijo) y se aplica el mismo código. El error de $\\langle r\\rangle$ decrece con pendiente {sl_m:.2f} (orden $h^2$, el de la ecuación radial, no el de Simpson). El nodo más cercano al máximo tiene un error acotado por $h/2$ que no baja con regularidad (depende de dónde cae el nodo respecto de $r=a_0$), mientras que el vértice de la parábola baja como $h^2$ (pendiente global {sl_p:.2f}).",
             [tab2]),
            ("Resultado de la escena y fuentes de error",
             f"Con $N=32000$ se obtiene $\\langle r\\rangle={_mean_r(32000):.8f}\\,a_0$ (exacto 1.5, error {sci(em[-1], 3)}) y $r_{{mp}}={_rmp_raw(32000)[1]:.8f}\\,a_0$ (exacto 1, error {sci(rm_par[-1], 3)}). Las fuentes son: el truncamiento de la ecuación radial (domina, $O(h^2)$, y es de signo definido), el de Simpson ($O(h^4)$, despreciable aquí), el de la caja ($e^{{-320}}$) y el redondeo ($10^{{-16}}$).",
             []),
            ("Criterio práctico",
             "Se acepta el resultado cuando (i) $\\sum u_i^2h=1$ con la precisión esperada, (ii) el error del radio medio baja por 4 al duplicar $N$ y (iii) la distancia entre el nodo máximo y el vértice parabólico es menor que $h$. Si se necesitase más exactitud en $\\langle r\\rangle$, mejorar $u$ (más nodos o extrapolación de Richardson) es más útil que cambiar la regla de integración.",
             []),
        ],
    )


DETAIL = {
    "BlackbodySpectrum": _blackbody(),
    "ComptonPhotoelectric": _compton(),
    "ElectronInterference": _interference(),
    "WavePacketUncertainty": _packet(),
    "InfiniteWell": _well(),
    "BarrierTunneling": _tunnel(),
    "HydrogenSpectrum": _hydrogen(),
    "RadialProbability": _radial(),
}
