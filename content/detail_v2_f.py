"""Explicacion y pasos a mano. Volumen 2, capitulos 23 a 25: BandStructure, DiodeCircuit, DecayChain,
BindingEnergy, TwoBodyDecay, FriedmannAge. Todas las cifras se calculan con physics/ o replicando su algoritmo."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from content.texfmt import fx, sci
from physics import v2ch23_solids as m23
from physics import v2ch24_nuclear as m24
from physics import v2ch25_particles as m25
from physics.integrators import rk4_step


# ====================================================================== capitulo 23
def _band() -> dict:
    g = m23.G_STRENGTH
    lim = m23.band_limits_analytic(3)
    edges = [e for b in lim for e in b]
    qs = [float(np.sqrt(2 * e)) for e in edges]
    f = lambda q: np.cos(q) + g / q * np.sin(q)  # noqa: E731
    num = m23.band_edges_numeric(3)
    ana = m23.band_edges_analytic(3)
    # reduccion a 3 ondas planas (m = -1, 0, 1) en k = 0 y k = pi
    def h3(k):
        m = np.array([-1, 0, 1])
        h = np.full((3, 3), g)
        h[np.diag_indices(3)] += 0.5 * (k + 2 * np.pi * m) ** 2
        return h
    h0, hp = h3(0.0), h3(np.pi)
    e0, ep = np.linalg.eigvalsh(h0), np.linalg.eigvalsh(hp)
    d0 = 0.5 * (2 * np.pi) ** 2
    dp = [0.5 * (np.pi + 2 * np.pi * m) ** 2 for m in (-1, 0, 1)]
    n201_lo = m23.bands_plane_waves(0.0, 3, 201)
    n201_hi = m23.bands_plane_waves(np.pi, 3, 201)
    n21 = np.sort(np.concatenate([m23.bands_plane_waves(0.0, 3, 21), m23.bands_plane_waves(np.pi, 3, 21)]))
    n401 = np.sort(np.concatenate([m23.bands_plane_waves(0.0, 3, 401), m23.bands_plane_waves(np.pi, 3, 401)]))
    gap = lim[1][0] - lim[0][1]
    k_mid = np.pi / 2
    e_mid = m23.band_energy_analytic(k_mid, 0)
    return dict(
        explanation=[
            "Un electrón en un cristal no ve un potencial constante sino uno periódico, producido por la hilera de iones. Según el teorema de Bloch, sus estados estacionarios son ondas planas moduladas por una función con el periodo de la red, y quedan etiquetados por un número de onda cristalino $k$ en el intervalo $-\\pi/a < k \\le \\pi/a$.",
            "La periodicidad hace que no todas las energías sean posibles. Para ciertos valores de $E$ la onda se propaga a través del cristal (energías permitidas, bandas) y para otros se refleja de forma coherente en cada ion y se atenúa (energías prohibidas, brechas). El modelo de Kronig-Penney, aquí con un peine de deltas de intensidad $g$, es el caso más simple donde esto se puede resolver exactamente.",
            "La condición de banda sale de exigir que $\\cos ka$ sea una cantidad real entre $-1$ y $1$: solo si el miembro derecho de la relación de dispersión cae en ese intervalo existe una $k$ real. Las brechas aparecen justo en los bordes de zona ($k=0$ y $k=\\pi/a$), donde dos ondas planas degeneradas se mezclan y se separan en energía.",
            "Intuitivamente, cuanto más intensas son las deltas, más ligados quedan los electrones a cada sitio: las bandas se estrechan hacia los niveles de la delta aislada y las brechas se ensanchan. En el límite $g\\to0$ se recupera la partícula libre y las brechas desaparecen. Que el sólido sea conductor, semiconductor o aislante depende de cuántas bandas llenan los electrones disponibles.",
        ],
        analytic_steps=[
            ("Plantear la relación de dispersión", "Con $\\hbar=m=a=1$ y un potencial $V=g\\sum_n\\delta(x-n)$, la condición de Bloch sobre la solución de la ecuación de Schrödinger conduce a la relación de Kronig-Penney, con $q=\\sqrt{2E}$.",
             [r"\cos k = \cos q + \dfrac{g}{q}\sin q \equiv F(q),\qquad q=\sqrt{2E}", rf"g = {g:g}"]),
            ("Condición de energía permitida", "Una $k$ real existe solo si $|\\cos k|\\le 1$. Los bordes de banda son entonces las raíces de $F(q)=\\pm1$; $F=+1$ corresponde a $k=0$ y $F=-1$ a $k=\\pi$.",
             [r"|F(q)|\le 1 \quad\Longleftrightarrow\quad \text{banda permitida}"]),
            ("Localizar la primera banda", "Se evalúa $F$ en puntos de prueba y se refina el cambio de signo de $|F|-1$ por el método de Brent. El primer borde inferior es la raíz de $F=+1$ más pequeña con $q>0$.",
             [rf"F({fx(qs[0],5)}) = \cos({fx(qs[0],5)}) + \dfrac{{{g:g}}}{{{fx(qs[0],5)}}}\sin({fx(qs[0],5)}) = {fx(f(qs[0]),6)}",
              rf"E_1 = \dfrac{{q_1^2}}{{2}} = \dfrac{{({fx(qs[0],5)})^2}}{{2}} = {fx(edges[0],4)}\ \mathrm{{Ha}}"]),
            ("Techo de la primera banda", "El siguiente cambio de signo (raíz de $F=-1$) cierra la banda, que termina en $k=\\pi$.",
             [rf"F({fx(qs[1],5)}) = {fx(f(qs[1]),6)},\qquad E_2 = \dfrac{{({fx(qs[1],5)})^2}}{{2}} = {fx(edges[1],4)}\ \mathrm{{Ha}}"]),
            ("Bordes de la segunda banda", "Se repite el procedimiento para $q$ mayores: $F=-1$ abre la segunda banda en $k=\\pi$ y $F=+1$ la cierra en $k=0$.",
             [rf"q_3={fx(qs[2],5)},\ q_4={fx(qs[3],5)}\ \Rightarrow\ E_3={fx(edges[2],4)}\ \mathrm{{Ha}},\ E_4={fx(edges[3],4)}\ \mathrm{{Ha}}"]),
            ("Intervalos de banda y brecha", "La brecha es la distancia entre el techo de la banda 1 y el piso de la banda 2.",
             [rf"\text{{banda 1}}:[{fx(lim[0][0],4)},{fx(lim[0][1],4)}]\ \mathrm{{Ha}},\qquad \text{{banda 2}}:[{fx(lim[1][0],4)},{fx(lim[1][1],4)}]\ \mathrm{{Ha}}",
              rf"E_{{gap}} = E_3-E_2 = {fx(lim[1][0],4)}-{fx(lim[0][1],4)} = {fx(gap,4)}\ \mathrm{{Ha}}"]),
            ("Obtener E(k) dentro de una banda", "Para una $k$ dada se resuelve $F(\\sqrt{2E})=\\cos k$ con una raíz acotada entre los bordes de la banda. Por ejemplo, a mitad de zona $k=\\pi/2$ se tiene $\\cos k=0$.",
             [rf"E_1(k=\pi/2) = {fx(e_mid,5)}\ \mathrm{{Ha}}\quad\left(\text{{entre }}{fx(lim[0][0],4)}\text{{ y }}{fx(lim[0][1],4)}\right)"]),
        ],
        numeric_steps=[
            ("Idea del método", "En lugar de resolver la ecuación trascendente se escribe el hamiltoniano en la base de ondas planas $e^{i(k+2\\pi m)x}$, $m\\in\\mathbb Z$. El potencial de deltas acopla todas las ondas con el mismo elemento $V_{mm'}=g$, y la energía cinética es diagonal.",
             [r"H_{mm'} = \tfrac12 (k+2\pi m)^2\,\delta_{mm'} + g"]),
            ("Versión reducida 3x3 en k = 0", "A mano se toman solo $m=-1,0,1$. Las energías cinéticas son $0.5(2\\pi m)^2$, es decir $0$ para $m=0$ y $" + fx(d0, 4) + "$ Ha para $m=\\pm1$.",
             [rf"H(0)=\begin{{pmatrix}} {fx(d0+g,3)} & {g:g} & {g:g}\\ {g:g} & {g:g} & {g:g}\\ {g:g} & {g:g} & {fx(d0+g,3)} \end{{pmatrix}}\ \mathrm{{Ha}}",
              rf"\text{{autovalores}}:\ {fx(e0[0],4)},\ {fx(e0[1],4)},\ {fx(e0[2],4)}\ \mathrm{{Ha}}"]),
            ("Versión reducida 3x3 en k = pi", "En el borde de zona las energías cinéticas son $0.5(\\pi+2\\pi m)^2$ para $m=-1,0,1$.",
             [rf"T_m = ({fx(dp[0],3)},\ {fx(dp[1],3)},\ {fx(dp[2],3)})\ \mathrm{{Ha}}",
              rf"\text{{autovalores}}:\ {fx(ep[0],4)},\ {fx(ep[1],4)},\ {fx(ep[2],4)}\ \mathrm{{Ha}}"]),
            ("Qué dice la versión reducida", "Con solo tres ondas se obtiene la estructura cualitativa (el nivel más bajo en $k=0$ queda en " + fx(e0[0], 3) + " Ha, el más bajo en $k=\\pi$ en " + fx(ep[0], 3) + " Ha) pero las cifras todavía están lejos: el tamaño de la base controla la precisión.",
             [rf"E_1^{{(3)}} = {fx(e0[0],4)}\ \mathrm{{Ha}}\ \text{{frente a}}\ E_1^{{(201)}} = {fx(n201_lo[0],4)}\ \mathrm{{Ha}}"]),
            ("Escalar a 201 ondas planas", "El código hace exactamente lo mismo con $m=-100,\\dots,100$ (matriz $201\\times201$) y la diagonaliza con `eigvalsh` en $k=0$ y $k=\\pi$. Los tres autovalores más bajos de cada $k$ dan los bordes de las tres primeras bandas.",
             [rf"k=0:\ ({fx(n201_lo[0],4)},\ {fx(n201_lo[1],4)},\ {fx(n201_lo[2],4)})\ \mathrm{{Ha}}",
              rf"k=\pi:\ ({fx(n201_hi[0],4)},\ {fx(n201_hi[1],4)},\ {fx(n201_hi[2],4)})\ \mathrm{{Ha}}"]),
            ("Ordenar los bordes y comparar", "Los seis valores se ordenan de menor a mayor y los cuatro primeros se comparan con los bordes analíticos.",
             [r"\begin{array}{c|cccc} \text{borde} & E_1 & E_2 & E_3 & E_4\\ \hline "
              + rf"\text{{analítico}} & {' & '.join(fx(x,4) for x in ana[:4])}\\ \text{{numérico}} & {' & '.join(fx(x,4) for x in num[:4])}\\ \text{{error rel.}} & {' & '.join(sci(abs(num[i]-ana[i])/ana[i],2) for i in range(4))}"
              + r"\end{array}"]),
            ("Convergencia con el tamaño de la base", "El potencial de deltas es singular, por lo que los coeficientes en ondas planas decaen lentamente y el error baja aproximadamente como $1/N$. Se muestra el primer borde numérico para varios tamaños.",
             [rf"N=21:\ E_1={fx(n21[0],4)},\quad N=201:\ E_1={fx(num[0],4)},\quad N=401:\ E_1={fx(n401[0],4)}\ \mathrm{{Ha}}\quad(\text{{analítico }}{fx(ana[0],4)})"]),
        ],
    )


def _diode() -> dict:
    from scipy.special import lambertw
    nvt = m23.N_IDEAL * m23.V_T
    isat, r = m23.I_S, m23.R_SERIES
    e = 5.0
    arg = isat * r / nvt * np.exp((e + isat * r) / nvt)
    w = float(lambertw(arg).real)
    vd = m23.diode_voltage_analytic(e)
    cur = m23.diode_current(e, vd)
    va = {x: m23.diode_voltage_analytic(x) for x in (1.0, 5.0, 10.0)}
    vn = {x: m23.diode_voltage_numeric(x) for x in (1.0, 5.0, 10.0)}
    f = lambda v: isat * np.expm1(v / nvt) - (e - v) / r  # noqa: E731
    lo, hi = 0.0, e
    its = []
    for k in range(1, 9):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        its.append((k, lo, hi, mid, fm))
        if fm > 0:
            hi = mid
        else:
            lo = mid
    # iteraciones hasta que el intervalo deja de reducirse
    lo, hi = 0.0, e
    stall = None
    for k in range(1, 201):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if stall is None and hi - lo <= np.spacing(hi) * 2:
            stall = k
    rows = " \\\\ ".join(rf"{k} & {fx(a,5)} & {fx(b,5)} & {fx(mid,5)} & {sci(fm,3)} & {'+' if fm > 0 else '-'}" for k, a, b, mid, fm in its[:6])
    return dict(
        explanation=[
            "Un diodo de unión p-n conduce muy bien en un sentido y casi nada en el otro. Su curva corriente-tensión sigue la ley de Shockley: la corriente crece exponencialmente con la tensión directa $V_d$ y, en inversa, se satura en $-I_s$ (la corriente de saturación, aquí $1$ pA).",
            "En un circuito real el diodo va con una resistencia que limita la corriente. La corriente que atraviesa el diodo es la misma que atraviesa la resistencia, y la fuente reparte su tensión entre ambos: $E=V_d+IR$. Se obtiene así una ecuación que mezcla una exponencial con una recta (la recta de carga) y que no tiene solución algebraica elemental.",
            "Su solución exacta se expresa con la función $W$ de Lambert, definida por $W(x)e^{W(x)}=x$. Como $V_d$ solo crece logarítmicamente con la corriente, la caída en el diodo se queda cerca de 0.5 a 0.7 V aunque la fuente suba mucho: casi todo el exceso de tensión cae en la resistencia.",
            "Esta es la razón por la que un diodo de silicio se aproxima con una tensión umbral casi fija. El modelo falla a corrientes muy altas (resistencia interna) o muy bajas (recombinación en la región de carga espacial), y no incluye la temperatura salvo a través de $V_T=kT/q$.",
        ],
        analytic_steps=[
            ("Ley de Kirchhoff de la malla", "La fuente $E$ alimenta al diodo y a la resistencia en serie, de modo que comparten la corriente $I$.",
             [r"E = V_d + I R,\qquad I = I_s\left(e^{V_d/nV_T}-1\right)"]),
            ("Eliminar la corriente", "Se iguala la corriente del diodo con la de la resistencia y se aísla la exponencial.",
             [r"I_s\left(e^{V_d/nV_T}-1\right)=\dfrac{E-V_d}{R}\ \Rightarrow\ e^{V_d/nV_T}=\dfrac{E+I_sR-V_d}{I_sR}"]),
            ("Cambio de variable", "Se define $w=(E+I_sR-V_d)/nV_T$, así $V_d=E+I_sR-nV_Tw$. Al sustituir en la exponencial queda la forma $w\\,e^{w}=\\text{constante}$.",
             [r"w\,e^{w} = \dfrac{I_sR}{nV_T}\,e^{(E+I_sR)/nV_T}\ \Rightarrow\ w=W\!\left(\dfrac{I_sR}{nV_T}e^{(E+I_sR)/nV_T}\right)"]),
            ("Evaluar el argumento de W", "Datos: $I_s=10^{-12}$ A, $nV_T=" + fx(nvt, 6) + "$ V, $R=100\\ \\Omega$ y $E=5$ V. El argumento es enorme porque la exponencial lo es.",
             [rf"\dfrac{{I_sR}}{{nV_T}} = \dfrac{{({sci(isat,2)})(100)}}{{{fx(nvt,6)}}} = {sci(isat*r/nvt,4)},\qquad \dfrac{{E+I_sR}}{{nV_T}} = {fx((e+isat*r)/nvt,4)}",
              rf"x = {sci(arg,4)}"]),
            ("Calcular W", "Para argumentos grandes $W(x)\\approx\\ln x-\\ln\\ln x$; el valor exacto se obtiene numéricamente.",
             [rf"W(x) = {fx(w,6)}"]),
            ("Tensión en el diodo", "Se vuelve a $V_d=E+I_sR-nV_T\\,W$.",
             [rf"V_d = 5 + ({sci(isat*r,3)}) - ({fx(nvt,6)})({fx(w,6)}) = {fx(vd,6)}\ \mathrm{{V}}"]),
            ("Corriente del circuito", "Con $V_d$ conocida, la corriente sale de la resistencia (la caída en la resistencia es lo que queda de la fuente).",
             [rf"I = \dfrac{{E-V_d}}{{R}} = \dfrac{{5-{fx(vd,6)}}}{{100}} = {sci(cur,5)}\ \mathrm{{A}} = {fx(cur*1e3,4)}\ \mathrm{{mA}}"]),
            ("Comprobar con otras fuentes", "Tres veces más tensión apenas cambia $V_d$.",
             [rf"V_d(1\ \mathrm{{V}})={fx(va[1.0],6)},\quad V_d(5\ \mathrm{{V}})={fx(va[5.0],6)},\quad V_d(10\ \mathrm{{V}})={fx(va[10.0],6)}\ \mathrm{{V}}"]),
        ],
        numeric_steps=[
            ("Plantear la función cuyo cero se busca", "El código no usa $W$: busca la raíz de la diferencia entre la corriente del diodo y la de la resistencia, que cambia de signo en el intervalo $[0,E]$.",
             [r"f(V) = I_s\left(e^{V/nV_T}-1\right)-\dfrac{E-V}{R}"]),
            ("Comprobar el cambio de signo", "En $V=0$ el diodo no conduce y la resistencia sí, así que $f<0$; en $V=E$ la resistencia no conduce y el diodo sí, así que $f>0$.",
             [rf"f(0) = 0-\dfrac{{5}}{{100}} = {fx(f(0.0),4)},\qquad f(5) = {sci(f(5.0),4)} > 0"]),
            ("Regla de bisección", "Se toma el punto medio $m$. Si $f(m)>0$ la raíz está a la izquierda ($hi\\leftarrow m$); si no, a la derecha ($lo\\leftarrow m$). Cada paso divide el intervalo a la mitad.",
             [r"m_k=\dfrac{lo+hi}{2},\qquad \text{si } f(m_k)>0:\ hi=m_k;\ \text{si no}:\ lo=m_k"]),
            ("Primera iteración a mano", "Con $lo=0$ y $hi=5$ el punto medio es $2.5$ V, donde el diodo (con $V/nV_T\\approx96.7$) conduce una corriente gigantesca, así que $f>0$ y la raíz está por debajo.",
             [rf"m_1 = 2.5\ \mathrm{{V}},\qquad f(m_1) = {sci(its[0][4],4)} > 0\ \Rightarrow\ hi=2.5"]),
            ("Primeras iteraciones", "El intervalo se reduce a la mitad en cada paso, pero hace falta bajar hasta cerca de 0.5 V.",
             [r"\begin{array}{c|ccccc} k & lo & hi & m_k & f(m_k) & \text{signo}\\ \hline " + rows + r"\end{array}"]),
            ("Criterio de parada", "El código hace 200 iteraciones fijas. El ancho del intervalo es $E/2^k$ y llega a la precisión de la máquina (diferencia de dos flotantes consecutivos) alrededor de la iteración " + str(stall) + "; las demás no cambian nada.",
             [rf"\text{{ancho tras }}k\text{{ pasos}} = \dfrac{{5}}{{2^k}},\qquad k={stall}:\ {sci(5/2**stall,3)}\ \mathrm{{V}}"]),
            ("Resultado y error", "La tensión es el punto medio del último intervalo. Para las tres fuentes el acuerdo con la solución de Lambert es total hasta la precisión de coma flotante.",
             [rf"V_d^{{num}}(5\ \mathrm{{V}}) = {fx(vn[5.0],9)}\ \mathrm{{V}}\quad(\text{{analítico }}{fx(va[5.0],9)})",
              rf"\left|\Delta V_d\right| = {sci(abs(vn[5.0]-va[5.0]),2)}\ \mathrm{{V}}\ (E=1\ \mathrm{{V}}:\ {sci(abs(vn[1.0]-va[1.0]),2)};\ E=10\ \mathrm{{V}}:\ {sci(abs(vn[10.0]-va[10.0]),2)})"]),
        ],
    )


# ====================================================================== capitulo 24
def _chain() -> dict:
    la, lb = m24.LAM_A, m24.LAM_B
    tpk = m24.peak_time_analytic()
    npk = float(m24.bateman_b(tpk))
    n20 = float(m24.bateman_b(20.0))
    t, y = m24.chain_numeric(20.0, 4000)
    n = 4000
    h = 20.0 / n
    rhs = lambda _t, v: np.array([-la * v[0], la * v[0] - lb * v[1], lb * v[1]])  # noqa: E731
    y0 = np.array([1.0, 0.0, 0.0])
    k1 = rhs(0, y0)
    k2 = rhs(0, y0 + h / 2 * k1)
    k3 = rhs(0, y0 + h / 2 * k2)
    k4 = rhs(0, y0 + h * k3)
    y1 = rk4_step(rhs, 0.0, y0, h)
    tpn = m24.peak_time_numeric()
    t4, yy4 = m24.chain_numeric(20.0, 40000)
    k = int(np.argmax(yy4[:, 1]))
    q0, q1, q2 = yy4[k - 1, 1], yy4[k, 1], yy4[k + 1, 1]
    hh = t4[1] - t4[0]
    corr = 0.5 * (q0 - q2) / (q0 - 2 * q1 + q2) * hh
    mean, err = m24.mean_life_monte_carlo()
    rng = np.random.default_rng(11)
    u = rng.uniform(size=50_000)
    ts = -np.log(1 - u) / lb
    s5 = ts[:5]
    ex = [float(np.exp(-lb * x)) for x in (1, 2)]
    idx = [800, 1600, 2400, 3200]
    tab = " \\\\ ".join(rf"{fx(t[i],2)} & {fx(y[i,0],6)} & {fx(y[i,1],6)} & {fx(float(m24.bateman_b(t[i])),6)}" for i in idx)
    return dict(
        explanation=[
            "La radiactividad es un proceso aleatorio: no se puede decir cuándo decaerá un núcleo concreto, pero cada núcleo tiene la misma probabilidad por unidad de tiempo $\\lambda$ de hacerlo, sin importar cuánto lleve existiendo. Para una población grande eso da la ley exponencial $dN/dt=-\\lambda N$, con vida media $\\tau=1/\\lambda$ y semivida $T_{1/2}=\\ln2/\\lambda$.",
            "Muchos núcleos no decaen directamente a un estado estable sino a otro núcleo radiactivo. En una cadena $A\\to B\\to C$, el hijo $B$ se produce a razón $\\lambda_AN_A$ y se destruye a razón $\\lambda_BN_B$: al principio $N_B$ crece, alcanza un máximo cuando producción y destrucción se igualan y luego decae.",
            "La solución de Bateman da $N_B(t)$ en forma cerrada como diferencia de dos exponenciales. El máximo ocurre cuando $dN_B/dt=0$, es decir cuando la actividad del hijo iguala a la del padre. Aquí $\\lambda_A=1$ s$^{-1}$ y $\\lambda_B=0.25$ s$^{-1}$, de modo que el hijo vive cuatro veces más y se acumula antes de decaer.",
            "El Monte Carlo del código ilustra el origen estadístico de la ley: cada núcleo recibe un tiempo de vida aleatorio con distribución exponencial, y el promedio de muchos tiempos converge a $\\tau$ con un error que decrece como $1/\\sqrt N$. Esto sirve para entender las fluctuaciones de conteo en muestras pequeñas, donde la ley exponencial es solo una expectativa.",
        ],
        analytic_steps=[
            ("Ecuaciones de la cadena", "Cada especie pierde núcleos a su propia tasa y gana los que pierde la anterior. Condiciones iniciales: solo hay $A$.",
             [r"\dot N_A=-\lambda_AN_A,\qquad \dot N_B=\lambda_AN_A-\lambda_BN_B,\qquad \dot N_C=\lambda_BN_B", r"N_A(0)=N_0=1,\quad N_B(0)=N_C(0)=0"]),
            ("Resolver el padre", "La primera ecuación está desacoplada y se integra directamente.",
             [r"N_A(t)=N_0e^{-\lambda_At}"]),
            ("Resolver el hijo", "Se sustituye $N_A$ en la segunda ecuación y se resuelve con el factor integrante $e^{\\lambda_Bt}$: $d(N_Be^{\\lambda_Bt})/dt=\\lambda_AN_0e^{(\\lambda_B-\\lambda_A)t}$. Al integrar con $N_B(0)=0$ se obtiene la solución de Bateman.",
             [r"N_B(t)=N_0\dfrac{\lambda_A}{\lambda_B-\lambda_A}\left(e^{-\lambda_At}-e^{-\lambda_Bt}\right)",
              rf"N_B(t)=\dfrac{{{la:g}}}{{{lb:g}-{la:g}}}\left(e^{{-{la:g}\,t}}-e^{{-{lb:g}\,t}}\right) = -\dfrac{{4}}{{3}}\left(e^{{-t}}-e^{{-t/4}}\right)"]),
            ("Condición de máximo", "Se deriva $N_B$ y se iguala a cero: $-\\lambda_Ae^{-\\lambda_At}+\\lambda_Be^{-\\lambda_Bt}=0$, o sea $e^{(\\lambda_A-\\lambda_B)t}=\\lambda_A/\\lambda_B$.",
             [r"t_{max}=\dfrac{\ln(\lambda_A/\lambda_B)}{\lambda_A-\lambda_B}"]),
            ("Instante del máximo", "Sustitución numérica.",
             [rf"t_{{max}}=\dfrac{{\ln({la:g}/{lb:g})}}{{{la:g}-{lb:g}}}=\dfrac{{{fx(np.log(la/lb),6)}}}{{{la-lb:g}}}={fx(tpk,6)}\ \mathrm{{s}}"]),
            ("Valor del máximo", "Se evalúa $N_B$ en $t_{max}$: es la fracción máxima de núcleos hijos presentes a la vez.",
             [rf"N_B(t_{{max}}) = -\dfrac{{4}}{{3}}\left(e^{{-{fx(tpk,4)}}}-e^{{-{fx(tpk/4,4)}}}\right) = {fx(npk,6)}"]),
            ("Valor a 20 s", "A tiempos largos casi todo ha pasado a $C$ y queda una pequeña cola de $B$, dominada por el decaimiento lento $e^{-\\lambda_Bt}$.",
             [rf"N_B(20\ \mathrm{{s}}) = -\dfrac{{4}}{{3}}\left(e^{{-20}}-e^{{-5}}\right) = {sci(n20,5)}"]),
            ("Vida media de B", "La vida media (esperanza del tiempo de vida) es el inverso de la constante de decaimiento.",
             [rf"\tau_B=\dfrac{{1}}{{\lambda_B}}=\dfrac{{1}}{{{lb:g}\ \mathrm{{s}}^{{-1}}}}={1/lb:g}\ \mathrm{{s}}\qquad (T_{{1/2}}={fx(np.log(2)/lb,4)}\ \mathrm{{s}})"]),
        ],
        numeric_steps=[
            ("Parámetros de RK4", "Se integra el sistema de tres ecuaciones con Runge-Kutta de orden 4 en $[0,20]$ s con $n=4000$ pasos.",
             [rf"h=\dfrac{{20}}{{4000}}={h:g}\ \mathrm{{s}},\qquad \vec y=(N_A,N_B,N_C),\ \vec y_0=(1,0,0)"]),
            ("Función derecha", "El campo es lineal; en el estado $\\vec y=(y_A,y_B,y_C)$:",
             [r"\vec f(\vec y)=\left(-\lambda_Ay_A,\ \lambda_Ay_A-\lambda_By_B,\ \lambda_By_B\right)"]),
            ("Pendientes del primer paso", "Se calculan las cuatro pendientes con $\\vec y_0=(1,0,0)$: $k_1$ en el inicio, $k_2$ y $k_3$ en el punto medio y $k_4$ al final.",
             [rf"\vec k_1=({fx(k1[0],6)},\ {fx(k1[1],6)},\ {fx(k1[2],6)})",
              rf"\vec k_2=\vec f(\vec y_0+\tfrac h2\vec k_1)=({fx(k2[0],6)},\ {fx(k2[1],6)},\ {fx(k2[2],8)})",
              rf"\vec k_3=\vec f(\vec y_0+\tfrac h2\vec k_2)=({fx(k3[0],6)},\ {fx(k3[1],6)},\ {fx(k3[2],8)})",
              rf"\vec k_4=\vec f(\vec y_0+h\vec k_3)=({fx(k4[0],6)},\ {fx(k4[1],6)},\ {fx(k4[2],8)})"]),
            ("Combinar y avanzar", "El nuevo estado usa el promedio ponderado $(k_1+2k_2+2k_3+k_4)/6$.",
             [r"\vec y_1=\vec y_0+\dfrac h6\left(\vec k_1+2\vec k_2+2\vec k_3+\vec k_4\right)",
              rf"\vec y_1=({fx(y1[0],8)},\ {fx(y1[1],8)},\ {fx(y1[2],10)}),\qquad N_B^{{exacto}}(h)={fx(float(m24.bateman_b(h)),8)}"]),
            ("Repetir y comparar", "Se repite el mismo paso 4000 veces. Algunos puntos de la solución frente a Bateman (la suma $N_A+N_B+N_C=1$ se conserva):",
             [r"\begin{array}{c|ccc} t\ (\mathrm{s}) & N_A & N_B^{RK4} & N_B^{Bateman}\\ \hline " + tab + r"\end{array}"]),
            ("Instante del máximo numérico", "Se integra con $n=40000$ ($h=" + fx(hh, 6) + "$ s), se toma el índice del mayor $N_B$ y se afina con la parábola que pasa por los tres puntos vecinos: el vértice está desplazado $\\delta=\\tfrac12\\frac{y_{-1}-y_{+1}}{y_{-1}-2y_0+y_{+1}}\\,h$.",
             [rf"t_k={fx(t4[k],6)}\ \mathrm{{s}},\quad \delta={sci(corr,3)}\ \mathrm{{s}},\quad t_{{max}}^{{num}}={fx(tpn,6)}\ \mathrm{{s}}",
              rf"\left|t^{{num}}_{{max}}-t^{{an}}_{{max}}\right|={sci(abs(tpn-tpk),2)}\ \mathrm{{s}},\qquad \left|N_B^{{RK4}}(20)-N_B^{{Bat}}(20)\right|={sci(abs(y[-1,1]-n20),2)}"]),
            ("Monte Carlo de la vida media: muestreo por inversión", "Si $u$ es uniforme en $[0,1)$, el tiempo $t=-\\ln(1-u)/\\lambda_B$ tiene densidad $\\lambda_Be^{-\\lambda_Bt}$ (se invierte la función de distribución $F(t)=1-e^{-\\lambda t}$). Con la semilla 11, las cinco primeras muestras son:",
             [r"\begin{array}{c|ccccc} i & 1 & 2 & 3 & 4 & 5\\ \hline u_i & " + " & ".join(fx(x, 5) for x in u[:5]) + r"\\ t_i=-4\ln(1-u_i) & " + " & ".join(fx(x, 4) for x in s5) + r"\end{array}"]),
            ("Promedio y error estadístico", "Con $N=50000$ muestras la vida media es la media y su incertidumbre es $\\sigma/\\sqrt N$.",
             [rf"\bar t=\dfrac{{1}}{{N}}\sum t_i={fx(mean,4)}\ \mathrm{{s}},\qquad \sigma_{{\bar t}}=\dfrac{{s}}{{\sqrt N}}={fx(err,4)}\ \mathrm{{s}}",
              rf"\bar t-\tau_B={fx(mean-1/lb,4)}\ \mathrm{{s}}\ \left({fx((mean-1/lb)/err,2)}\,\sigma\right)"]),
        ],
    )


def _binding() -> dict:
    av, as_, ac, aa = m24.A_V, m24.A_S, m24.A_C, m24.A_A
    a = 56
    z0 = m24.z_stable_analytic(a)
    zn = m24.z_stable_numeric(a, False)
    bpa_an = float(m24.b_per_a_analytic(a))
    bpa_num = m24.b_per_a(a, False)
    aa_arr = np.arange(20, 240)
    va = m24.b_per_a_analytic(aa_arr)
    amax = int(np.argmax(va)) + 20
    anum = m24.peak_a_numeric()
    bmax = float(np.max(va))
    terms = [av * a, as_ * a ** (2 / 3), ac * z0 * (z0 - 1) / a ** (1 / 3), aa * (a - 2 * z0) ** 2 / a]
    bz = {z: m24.binding_energy(a, z, False) for z in range(23, 30)}
    zrows = " \\\\ ".join(rf"{z} & {a-z} & {fx(bz[z],3)} & {fx(bz[z]/a,4)}" for z in range(24, 29))
    pk = [(x, m24.b_per_a(x, False)) for x in (20, 40, 56, 62, 100, 150, 200, 238)]
    prow = " \\\\ ".join(rf"{x} & {z_} & {fx(b_,4)}" for (x, b_), z_ in zip(pk, [m24.z_stable_numeric(x, False) for x, _ in pk]))
    z1 = [1, 2, 3]
    b3 = [m24.binding_energy(a, z, False) for z in z1]
    return dict(
        explanation=[
            "La masa de un núcleo es menor que la suma de las masas de sus nucleones libres; la diferencia, multiplicada por $c^2$, es la energía de enlace $B$. La cantidad que mide qué tan fuertemente está ligado un núcleo es $B/A$, la energía de enlace por nucleón, y es casi constante (cerca de 8 MeV) para la mayoría de los núcleos.",
            "La fórmula semiempírica de masa (modelo de la gota líquida) explica $B(A,Z)$ con términos de significado físico: el volumen $a_VA$ (la fuerza fuerte es de corto alcance y saturada), la superficie $-a_SA^{2/3}$ (los nucleones de la superficie tienen menos vecinos), la repulsión coulombiana $-a_CZ(Z-1)/A^{1/3}$ entre protones, y la asimetría $-a_A(A-2Z)^2/A$ (el principio de Pauli favorece $N\\approx Z$). Aquí se omite el término de apareamiento.",
            "Para un $A$ fijo, el núcleo más estable maximiza $B$ respecto de $Z$: la asimetría empuja hacia $Z=A/2$ y la repulsión coulombiana hacia menos protones. El resultado es que los núcleos pesados tienen exceso de neutrones.",
            "La curva $B/A$ sube rápido a bajos $A$ (la superficie pesa mucho), alcanza un máximo en la zona del hierro y el níquel y baja lentamente por la repulsión coulombiana. Esa forma explica por qué la fusión de núcleos ligeros y la fisión de núcleos pesados liberan energía: ambas llevan hacia el máximo.",
        ],
        analytic_steps=[
            ("Fórmula de masa sin apareamiento", "Se toman los coeficientes del modelo de la gota (en MeV).",
             [r"B(A,Z)=a_VA-a_SA^{2/3}-a_C\dfrac{Z(Z-1)}{A^{1/3}}-a_A\dfrac{(A-2Z)^2}{A}",
              rf"a_V={av},\ a_S={as_},\ a_C={ac},\ a_A={aa}\ \mathrm{{MeV}}"]),
            ("Derivar respecto de Z", "Para $A$ fijo, el máximo de $B$ cumple $\\partial B/\\partial Z=0$. Se aproxima $Z(Z-1)\\approx Z^2$ en la derivada (la diferencia es un uno frente a $2Z$).",
             [r"\dfrac{\partial B}{\partial Z}=-\dfrac{2a_CZ}{A^{1/3}}+\dfrac{4a_A(A-2Z)}{A}=0"]),
            ("Despejar Z", "Se agrupan los términos en $Z$ y se divide.",
             [r"Z_0=\dfrac{A/2}{1+\dfrac{a_CA^{2/3}}{4a_A}}"]),
            ("Evaluar para A = 56", "Se sustituyen los números.",
             [rf"A^{{2/3}}={fx(a**(2/3),4)},\qquad \dfrac{{a_CA^{{2/3}}}}{{4a_A}}=\dfrac{{({ac})({fx(a**(2/3),4)})}}{{4({aa})}}={fx(ac*a**(2/3)/(4*aa),5)}",
              rf"Z_0=\dfrac{{28}}{{1+{fx(ac*a**(2/3)/(4*aa),5)}}}={fx(z0,3)}\ \Rightarrow\ Z=27\ \text{{(entero más próximo)}}" if round(z0) == 27 else rf"Z_0={fx(z0,3)}"]),
            ("Energía de enlace de A = 56", "Se evalúa cada término en $Z_0$ continuo.",
             [rf"B={fx(terms[0],2)}-{fx(terms[1],2)}-{fx(terms[2],2)}-{fx(terms[3],2)}={fx(sum([terms[0],-terms[1],-terms[2],-terms[3]]),2)}\ \mathrm{{MeV}}",
              rf"\dfrac{{B}}{{A}}={fx(bpa_an,4)}\ \mathrm{{MeV}}"]),
            ("B/A en función de A", "Sustituyendo $Z_0(A)$ en $B$ y dividiendo por $A$ queda una función de una sola variable.",
             [r"\dfrac{B}{A}\Big|_{Z_0}=a_V-a_SA^{-1/3}-a_C\dfrac{Z_0(Z_0-1)}{A^{4/3}}-a_A\dfrac{(A-2Z_0)^2}{A^2}"]),
            ("Localizar el máximo", "La condición $d(B/A)/dA=0$ no tiene forma cerrada sencilla (aparece $Z_0(A)$), pero se evalúa sobre los enteros $20\\le A\\le239$ y se toma el mayor valor.",
             [rf"A_{{max}}={amax},\qquad (B/A)_{{max}}={fx(bmax,4)}\ \mathrm{{MeV}}"]),
        ],
        numeric_steps=[
            ("Estrategia", "Para cada $A$ se calcula $B$ para todos los números enteros de protones $Z=1,\\dots,A-1$ y se elige el que da el mayor $B$. No hace falta derivar: es una búsqueda exhaustiva en un conjunto finito.",
             [r"Z_{num}(A)=\arg\max_{Z\in\{1,\dots,A-1\}}B(A,Z)"]),
            ("Primeros valores de la búsqueda (A = 56)", "Se calculan los tres primeros candidatos, ya muy lejos del máximo por la asimetría enorme.",
             [rf"B(56,1)={fx(b3[0],2)},\quad B(56,2)={fx(b3[1],2)},\quad B(56,3)={fx(b3[2],2)}\ \mathrm{{MeV}}"]),
            ("Cerca del máximo", "Tabla de los valores alrededor del óptimo continuo $Z_0=" + fx(z0, 2) + "$:",
             [r"\begin{array}{cccc} Z & N & B\ (\mathrm{MeV}) & B/A\ (\mathrm{MeV})\\ \hline " + zrows + r"\end{array}"]),
            ("Elegir el máximo", "El mayor $B$ se da en un entero; la diferencia con $Z_0$ es el error de redondeo, al que se limita la comparación.",
             [rf"Z_{{num}}={zn},\qquad Z_0={fx(z0,3)},\qquad |Z_{{num}}-Z_0|={fx(abs(zn-z0),3)}"]),
            ("B/A numérico frente al analítico", "El valor numérico usa el $Z$ entero; el analítico, el $Z_0$ continuo. Como $Z_0$ maximiza la función continua, el valor continuo es ligeramente mayor o igual.",
             [rf"(B/A)_{{num}}={fx(bpa_num,4)}\ \mathrm{{MeV}},\qquad (B/A)_{{an}}={fx(bpa_an,4)}\ \mathrm{{MeV}},\qquad \Delta={sci(bpa_an-bpa_num,2)}\ \mathrm{{MeV}}"]),
            ("Repetir para todo A", "El código repite la búsqueda de $Z$ para cada $A$ entre 20 y 239. Muestra de los resultados:",
             [r"\begin{array}{ccc} A & Z_{num} & B/A\ (\mathrm{MeV})\\ \hline " + prow + r"\end{array}"]),
            ("Máximo de B/A", "Se toma el $A$ con mayor $B/A$ numérico y se compara con el analítico.",
             [rf"A_{{max}}^{{num}}={anum},\qquad A_{{max}}^{{an}}={amax},\qquad (B/A)_{{max}}={fx(max(m24.b_per_a(int(x),False) for x in aa_arr),4)}\ \mathrm{{MeV}}"]),
        ],
    )


# ====================================================================== capitulo 25
def _twobody() -> dict:
    m, m1, m2 = m25.M_PARENT, m25.M_1, m25.M_2
    ps = m25.p_star_analytic()
    e1 = float(np.hypot(ps, m1))
    e2 = float(np.hypot(ps, m2))
    pl = 2000.0
    el = float(np.hypot(pl, m))
    gamma, beta = el / m, pl / el
    lo, hi = m25.lab_energy_range_analytic()
    ph, pm = m25.lab_momentum_limits_analytic()
    pn = m25.p_star_numeric()
    mean, std, pmax, pmin = m25.decay_monte_carlo()
    f = lambda p: np.hypot(p, m1) + np.hypot(p, m2) - m  # noqa: E731
    a_, b_ = 0.0, m
    its = []
    for k in range(1, 9):
        mid = 0.5 * (a_ + b_)
        fm = f(mid)
        its.append((k, a_, b_, mid, fm))
        if fm < 0:
            a_ = mid
        else:
            b_ = mid
    a_, b_ = 0.0, m
    stall = None
    for k in range(1, 201):
        mid = 0.5 * (a_ + b_)
        if f(mid) < 0:
            a_ = mid
        else:
            b_ = mid
        if stall is None and b_ - a_ <= 2 * np.spacing(b_):
            stall = k
    rows = " \\\\ ".join(rf"{k} & {fx(a,3)} & {fx(b,3)} & {fx(mid,3)} & {fx(fm,3)}" for k, a, b, mid, fm in its[:6])
    rng = np.random.default_rng(5)
    n = 100_000
    cost = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    ev = []
    for i in range(3):
        c, ph_ = cost[i], phi[i]
        s = np.sqrt(1 - c**2)
        px, py, pz = ps * s * np.cos(ph_), ps * s * np.sin(ph_), ps * c
        e1l = gamma * (e1 + beta * pz)
        p1z = gamma * (pz + beta * e1)
        e2l = gamma * (e2 - beta * pz)
        p2z = gamma * (-pz + beta * e2)
        inv = float(np.sqrt((e1l + e2l) ** 2 - (p1z + p2z) ** 2))
        ev.append((c, ph_, px, py, pz, e1l, p1z, e2l, p2z, inv))
    c, ph_, px, py, pz, e1l, p1z, e2l, p2z, inv = ev[0]
    erng = m25.lab_energy_samples()
    return dict(
        explanation=[
            "Cuando una partícula inestable de masa $M$ decae en dos, la conservación de energía y de impulso fija casi todo en el sistema en que el padre está en reposo: las dos hijas salen con impulsos iguales y opuestos de módulo $p^*$ y con energías determinadas únicamente por las masas. Solo la dirección es libre, y para un padre sin polarización es isótropa.",
            "En el laboratorio el padre se mueve, y las magnitudes de las hijas se obtienen con un impulso de Lorentz (boost) a lo largo del movimiento del padre. La energía y la componente longitudinal del impulso mezclan $E^*$, $p^*\\cos\\theta^*$ y los factores $\\gamma$ y $\\beta$; la componente transversal no cambia. Por eso el protón puede salir hacia atrás o hacia adelante, pero su energía en el laboratorio queda acotada.",
            "Como $\\cos\\theta^*$ es uniforme en $[-1,1]$ y $E_{lab}$ es lineal en él, la energía del protón en el laboratorio está distribuida uniformemente entre dos valores extremos. Esa propiedad (el espectro plano) se usa experimentalmente para medir masas y spins de partículas.",
            "Una cantidad que no depende del sistema de referencia es la masa invariante $M^2=(E_1+E_2)^2-|\\vec p_1+\\vec p_2|^2$. Al reconstruirla evento por evento a partir de las hijas medidas se debe recuperar la masa del padre (con dispersión por resolución del detector; en la simulación, solo por redondeo). Así se identifican resonancias en un experimento.",
        ],
        analytic_steps=[
            ("Datos y unidades", "Se usa $c=1$, masas en MeV. El padre es un $\\Lambda$ que decae a $p+\\pi^-$.",
             [rf"M={m}\ \mathrm{{MeV}},\quad m_1=m_p={m1}\ \mathrm{{MeV}},\quad m_2=m_{{\pi}}={m2}\ \mathrm{{MeV}},\quad p_{{lab}}={pl:.0f}\ \mathrm{{MeV}}/c"]),
            ("Impulso en el sistema en reposo", "De $E_1^*+E_2^*=M$ con $E_i^*=\\sqrt{p^{*2}+m_i^2}$ se despeja $p^*$.",
             [r"p^*=\dfrac{\sqrt{\left[M^2-(m_1+m_2)^2\right]\left[M^2-(m_1-m_2)^2\right]}}{2M}",
              rf"M^2-(m_1+m_2)^2={fx(m**2-(m1+m2)**2,2)},\qquad M^2-(m_1-m_2)^2={fx(m**2-(m1-m2)**2,2)}",
              rf"p^*=\dfrac{{\sqrt{{({fx(m**2-(m1+m2)**2,2)})({fx(m**2-(m1-m2)**2,2)})}}}}{{2({m})}}={fx(ps,4)}\ \mathrm{{MeV}}/c"]),
            ("Energías en reposo", "Con $p^*$ se obtienen las energías de las hijas; su suma debe dar $M$.",
             [rf"E_1^*=\sqrt{{{fx(ps,3)}^2+{m1}^2}}={fx(e1,4)}\ \mathrm{{MeV}},\qquad E_2^*={fx(e2,4)}\ \mathrm{{MeV}}",
              rf"E_1^*+E_2^*={fx(e1+e2,4)}\ \mathrm{{MeV}}\approx M"]),
            ("Parámetros del boost", "El padre tiene impulso $p_{lab}$: energía $E=\\sqrt{p^2+M^2}$, $\\gamma=E/M$, $\\beta=p/E$.",
             [rf"E_\Lambda=\sqrt{{{pl:.0f}^2+{m}^2}}={fx(el,3)}\ \mathrm{{MeV}},\quad \gamma={fx(gamma,5)},\quad \beta={fx(beta,5)}"]),
            ("Energía del protón en el laboratorio", "Se aplica $E_{lab}=\\gamma(E^*+\\beta p^*\\cos\\theta^*)$. Los extremos corresponden a $\\cos\\theta^*=\\mp1$.",
             [rf"E_{{min}}=\gamma(E_1^*-\beta p^*)={fx(gamma,5)}({fx(e1,3)}-{fx(beta,5)}\cdot{fx(ps,3)})={fx(lo,2)}\ \mathrm{{MeV}}",
              rf"E_{{max}}=\gamma(E_1^*+\beta p^*)={fx(hi,2)}\ \mathrm{{MeV}}"]),
            ("Impulso extremo del protón", "La componente longitudinal es $p_z=\\gamma(p^*\\cos\\theta^*+\\beta E^*)$; el módulo máximo ocurre con $\\cos\\theta^*=+1$ y el mínimo con $\\cos\\theta^*=-1$.",
             [rf"p_{{max}}=\gamma(p^*+\beta E_1^*)={fx(gamma,5)}({fx(ps,3)}+{fx(beta,5)}\cdot{fx(e1,3)})={fx(ph,3)}\ \mathrm{{MeV}}/c",
              rf"p_{{min}}=\left|\gamma(-p^*+\beta E_1^*)\right|={fx(pm,3)}\ \mathrm{{MeV}}/c"]),
            ("Masa invariante", "La suma de cuatro-vectores de las hijas es el cuatro-vector del padre, de modo que la masa invariante es la del padre en cualquier sistema.",
             [rf"M^2=(E_1+E_2)^2-|\vec p_1+\vec p_2|^2\ \Rightarrow\ M={m}\ \mathrm{{MeV}}"]),
        ],
        numeric_steps=[
            ("Parte 1: bisección para p*", "Se busca el cero de $f(p)=\\sqrt{p^2+m_1^2}+\\sqrt{p^2+m_2^2}-M$ en $[0,M]$. Como $f(0)=m_1+m_2-M<0$ y $f(M)>0$ hay una raíz entre ambos.",
             [rf"f(0)={fx(f(0.0),3)}\ \mathrm{{MeV}},\qquad f({m})={fx(f(m),3)}\ \mathrm{{MeV}}"]),
            ("Primeras iteraciones", "En cada paso se toma el punto medio; si $f<0$ la raíz está a la derecha ($lo\\leftarrow m$), si no a la izquierda ($hi\\leftarrow m$).",
             [r"\begin{array}{c|cccc} k & lo & hi & m_k & f(m_k)\\ \hline " + rows + r"\end{array}"]),
            ("Convergencia de la bisección", "Tras 200 iteraciones fijas (el ancho se hace del orden del redondeo hacia la iteración " + str(stall) + ") se obtiene:",
             [rf"p^*_{{num}}={fx(pn,9)}\ \mathrm{{MeV}}/c,\qquad p^*_{{an}}={fx(ps,9)},\qquad |\Delta|={sci(abs(pn-ps),2)}"]),
            ("Parte 2: Monte Carlo isótropo", "Para cada evento se sortea $\\cos\\theta^*\\in[-1,1]$ y $\\varphi\\in[0,2\\pi)$ (semilla 5), se arma $\\vec p_1=p^*(\\sin\\theta^*\\cos\\varphi,\\sin\\theta^*\\sin\\varphi,\\cos\\theta^*)$ y $\\vec p_2=-\\vec p_1$. Primer evento:",
             [rf"\cos\theta^*={fx(c,5)},\quad \varphi={fx(ph_,5)}\ \mathrm{{rad}}",
              rf"\vec p_1^{{\,*}}=({fx(px,3)},\ {fx(py,3)},\ {fx(pz,3)})\ \mathrm{{MeV}}/c"]),
            ("Boost de cada hija", "Se transforman $E$ y $p_z$ (el transversal no cambia): $E'=\\gamma(E+\\beta p_z)$, $p_z'=\\gamma(p_z+\\beta E)$.",
             [rf"p:\ E_1={fx(e1l,3)},\ p_{{1z}}={fx(p1z,3)}\ \mathrm{{MeV}}",
              rf"\pi^-:\ E_2={fx(e2l,3)},\ p_{{2z}}={fx(p2z,3)}\ \mathrm{{MeV}}"]),
            ("Masa invariante del evento", "Como los momentos transversales se cancelan, $M^2=(E_1+E_2)^2-(p_{1z}+p_{2z})^2$.",
             [rf"E_1+E_2={fx(e1l+e2l,3)},\quad p_{{1z}}+p_{{2z}}={fx(p1z+p2z,3)}\ \Rightarrow\ M_1={fx(inv,6)}\ \mathrm{{MeV}}"]),
            ("Estadística del conjunto", "El código repite lo mismo con 100000 eventos para la masa invariante (con 200000 muestras de $\\cos\\theta^*$ para el espectro de energía). La masa reconstruida es la del padre con dispersión de redondeo, y los extremos del impulso se acercan a los analíticos desde dentro, ya que el muestreo nunca llega exactamente a $\\cos\\theta^*=\\pm1$.",
             [rf"\bar M={fx(mean,6)}\ \mathrm{{MeV}},\quad \sigma_M={sci(std,2)}\ \mathrm{{MeV}}",
              rf"p_{{max}}^{{MC}}={fx(pmax,3)}\ (\text{{analítico }}{fx(ph,3)}),\qquad p_{{min}}^{{MC}}={fx(pmin,3)}\ (\text{{analítico }}{fx(pm,3)})",
              rf"E_p^{{MC}}\in[{fx(erng.min(),2)},\ {fx(erng.max(),2)}]\ \mathrm{{MeV}}\quad(\text{{analítico }}[{fx(lo,2)},\ {fx(hi,2)}])"]),
        ],
    )


def _friedmann() -> dict:
    h0, om, ol = m25.H0_PER_GYR, m25.OMEGA_M, m25.OMEGA_L
    t0 = m25.age_analytic()
    t0n = m25.age_numeric()
    s = np.sqrt(ol / om)
    arc = float(np.arcsinh(s))
    pref = 2 / (3 * h0 * np.sqrt(ol))

    def integrand(u):
        a = u**2
        return 0.0 if u == 0 else 2 * u / (a * h0 * np.sqrt(om / a**3 + ol))

    us = np.linspace(0, 1, 5)
    ys = [integrand(u) for u in us]
    h4 = 0.25
    s4 = h4 / 3 * (ys[0] + ys[4] + 4 * (ys[1] + ys[3]) + 2 * ys[2])
    # Simpson N=20 para mostrar convergencia
    def simp(n):
        u = np.linspace(0.0, 1.0, n + 1)
        y = np.array([integrand(x) for x in u])
        return float(1.0 / n / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))
    s20, s200 = simp(20), simp(200)
    # RK4
    ta, tb, n = 1e-3, 25.0, 5000
    hh = (tb - ta) / n
    a0 = float(m25.scale_factor_analytic(ta))
    rhs = lambda aa: h0 * np.sqrt(om / aa + ol * aa**2)  # noqa: E731
    k1 = rhs(a0)
    k2 = rhs(a0 + hh / 2 * k1)
    k3 = rhs(a0 + hh / 2 * k2)
    k4 = rhs(a0 + hh * k3)
    a1 = a0 + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t_, y_ = m25.scale_factor_numeric()
    a25n = float(y_[-1, 0])
    a25 = float(m25.scale_factor_analytic(25.0))
    idx = [1000, 2000, 3000, 4000]
    tab = " \\\\ ".join(rf"{fx(t_[i],3)} & {fx(y_[i,0],6)} & {fx(float(m25.scale_factor_analytic(t_[i])),6)}" for i in idx)
    return dict(
        explanation=[
            "La expansión del universo se describe con un único número, el factor de escala $a(t)$, que multiplica todas las distancias comoviles. La ecuación de Friedmann relaciona la tasa de expansión $H=\\dot a/a$ con el contenido de energía: aquí hay materia, cuya densidad diluye como $a^{-3}$, y energía oscura (constante cosmológica), cuya densidad no cambia.",
            "En un universo plano, $\\Omega_m+\\Omega_\\Lambda=1$ y $H^2=H_0^2(\\Omega_m a^{-3}+\\Omega_\\Lambda)$ con $a=1$ hoy. Al principio domina la materia y la expansión se frena; cuando $a$ crece la constante cosmológica pasa a dominar y la expansión se acelera. Para $a\\ll1$ se recupera el universo dominado por materia $a\\propto t^{2/3}$.",
            "La edad del universo es el tiempo que tarda $a$ en pasar de 0 a 1, o sea $t_0=\\int_0^1\\mathrm da/(aH)$. Con $H_0^{-1}\\approx14.2$ Gyr, la edad es cercana a $H_0^{-1}$ pero algo menor, porque la expansión pasada fue más lenta (efecto de la materia) y más rápida después (efecto de $\\Lambda$).",
            "La integral tiene una singularidad en apariencia: $1/(aH)\\to a^{1/2}/(H_0\\sqrt{\\Omega_m})$ cerca de $a=0$, que es finita pero con derivada infinita. Por eso el código cambia a $a=u^2$, con lo que el integrando se vuelve suave y la regla de Simpson converge rápido.",
        ],
        analytic_steps=[
            ("Ecuación de Friedmann", "Con $x=a$ y universo plano:",
             [r"\left(\dfrac{\dot a}{a}\right)^2=H_0^2\left(\dfrac{\Omega_m}{a^3}+\Omega_\Lambda\right),\qquad \Omega_m+\Omega_\Lambda=1",
              rf"H_0={h0}\ \mathrm{{Gyr^{{-1}}}},\quad \Omega_m={om},\quad \Omega_\Lambda={ol}"]),
            ("Separar variables", "Se despeja $dt=da/(aH)$ y se integra de $a=0$ (inicio) a $a=1$ (hoy).",
             [r"t_0=\int_0^1\dfrac{\mathrm da}{aH(a)}=\dfrac1{H_0}\int_0^1\dfrac{\mathrm da}{\sqrt{\Omega_m/a+\Omega_\Lambda a^2}}"]),
            ("Cambio de variable", "Se hace $x=a^{3/2}$, con lo que $\\mathrm dx=\\tfrac32a^{1/2}\\mathrm da$ y el radicando se vuelve $\\Omega_m+\\Omega_\\Lambda x^2$.",
             [r"t_0=\dfrac{2}{3H_0}\int_0^1\dfrac{\mathrm dx}{\sqrt{\Omega_m+\Omega_\Lambda x^2}}"]),
            ("Integral elemental", "La integral es de la forma $\\int\\mathrm dx/\\sqrt{c+bx^2}=\\mathrm{arcsinh}(x\\sqrt{b/c})/\\sqrt b$.",
             [r"t_0=\dfrac{2}{3H_0\sqrt{\Omega_\Lambda}}\,\mathrm{arcsinh}\sqrt{\dfrac{\Omega_\Lambda}{\Omega_m}}"]),
            ("Sustituir los datos", "Primero el prefactor y el argumento del arcoseno hiperbólico.",
             [rf"\dfrac{{2}}{{3H_0\sqrt{{\Omega_\Lambda}}}}=\dfrac{{2}}{{3({h0})\sqrt{{{ol}}}}}={fx(pref,4)}\ \mathrm{{Gyr}},\qquad \sqrt{{\dfrac{{{ol}}}{{{om}}}}}={fx(s,5)}",
              rf"\mathrm{{arcsinh}}({fx(s,5)})={fx(arc,5)}"]),
            ("Edad", "Se multiplica.",
             [rf"t_0=({fx(pref,4)})({fx(arc,5)})={fx(t0,4)}\ \mathrm{{Gyr}}"]),
            ("Factor de escala a(t)", "La misma sustitución, integrada hasta un $a$ cualquiera, se invierte para dar $a(t)$.",
             [r"a(t)=\left(\dfrac{\Omega_m}{\Omega_\Lambda}\right)^{1/3}\sinh^{2/3}\!\left(\tfrac32\sqrt{\Omega_\Lambda}H_0t\right)",
              rf"a(25\ \mathrm{{Gyr}})=({fx((om/ol)**(1/3),5)})\sinh^{{2/3}}\!\left({fx(1.5*np.sqrt(ol)*h0*25,5)}\right)={fx(a25,5)}"]),
            ("Comprobación", "En $t=t_0$ debe valer $a=1$.",
             [rf"a({fx(t0,4)})={fx(float(m25.scale_factor_analytic(t0)),6)}"]),
        ],
        numeric_steps=[
            ("Parte 1: integral de la edad", "Se calcula $t_0=\\int_0^1\\mathrm da/(aH)$ con la sustitución $a=u^2$, $\\mathrm da=2u\\,\\mathrm du$: el integrando en $u$ queda finito (vale 0 en $u=0$).",
             [r"g(u)=\dfrac{2u}{u^2H_0\sqrt{\Omega_m/u^6+\Omega_\Lambda}},\qquad t_0=\int_0^1g(u)\,\mathrm du"]),
            ("Simpson reducido, N = 4", "Para ver la mecánica se usa $h=0.25$ con cinco nodos. Simpson pesa $1,4,2,4,1$.",
             [r"\begin{array}{c|ccccc} u & 0 & 0.25 & 0.5 & 0.75 & 1\\ \hline g(u)\ (\mathrm{Gyr}) & " + " & ".join(fx(v, 3) for v in ys) + r"\end{array}",
              rf"S_4=\dfrac{{h}}{{3}}\left[g_0+4(g_1+g_3)+2g_2+g_4\right]=\dfrac{{0.25}}{{3}}\left[{fx(ys[0],3)}+4({fx(ys[1],3)}+{fx(ys[3],3)})+2({fx(ys[2],3)})+{fx(ys[4],3)}\right]={fx(s4,4)}\ \mathrm{{Gyr}}"]),
            ("Refinar la malla", "Al aumentar $N$ la suma converge rápidamente al valor analítico.",
             [rf"S_4={fx(s4,5)},\quad S_{{20}}={fx(s20,5)},\quad S_{{200}}={fx(s200,5)}\ \mathrm{{Gyr}}"]),
            ("Resultado con N = 20000", "El código usa $N=20000$ subintervalos. La diferencia con la fórmula cerrada es del orden de la precisión de máquina.",
             [rf"t_0^{{num}}={fx(t0n,9)}\ \mathrm{{Gyr}}\quad(\text{{analítico }}{fx(t0,9)}),\qquad |\Delta|={sci(abs(t0n-t0),2)}\ \mathrm{{Gyr}}"]),
            ("Parte 2: RK4 para a(t)", "Se integra $\\mathrm da/\\mathrm dt=H_0\\sqrt{\\Omega_m/a+\\Omega_\\Lambda a^2}$ de $t=10^{-3}$ Gyr a 25 Gyr con $n=5000$ pasos, empezando con la solución analítica en ese instante (a $a=0$ la derivada es infinita).",
             [rf"h=\dfrac{{25-0.001}}{{5000}}={fx(hh,7)}\ \mathrm{{Gyr}},\qquad a_0=a(10^{{-3}})={sci(a0,5)}"]),
            ("Primer paso de RK4", "Las pendientes dependen solo de $a$. Como $a_0$ es muy pequeño, domina la materia y la derivada es grande.",
             [rf"k_1={fx(k1,6)},\quad k_2={fx(k2,6)},\quad k_3={fx(k3,6)},\quad k_4={fx(k4,6)}\ \mathrm{{Gyr^{{-1}}}}",
              rf"a_1=a_0+\dfrac h6(k_1+2k_2+2k_3+k_4)={sci(a1,6)},\qquad a(t_0+h)_{{an}}={sci(float(m25.scale_factor_analytic(ta+hh)),6)}"]),
            ("Resultados a lo largo del intervalo", "Se repite hasta $t=25$ Gyr; algunos puntos frente a la solución analítica:",
             [r"\begin{array}{c|cc} t\ (\mathrm{Gyr}) & a_{RK4} & a_{an}\\ \hline " + tab + r"\end{array}"]),
            ("Valor final", "El factor de escala a 25 Gyr coincide con la solución analítica.",
             [rf"a_{{num}}(25)={fx(a25n,7)},\quad a_{{an}}(25)={fx(a25,7)},\quad |\Delta a|={sci(abs(a25n-a25),2)}"]),
        ],
    )


DETAIL = {
    "BandStructure": _band(),
    "DiodeCircuit": _diode(),
    "DecayChain": _chain(),
    "BindingEnergy": _binding(),
    "TwoBodyDecay": _twobody(),
    "FriedmannAge": _friedmann(),
}
