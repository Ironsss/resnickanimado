"""Metodo numerico explicado: Volumen 1, capitulos 21 a 25.

Escenas: MuonDecay, VelocityAndEnergy, ThermalExpansion, ThermalEquilibrium, GasProcesses, OttoCycle, GasPressure,
MaxwellBoltzmann, CarnotCycle, EntropyProduction. Campos: numeric_intro, numeric_code, numeric_converge.
Todas las cifras de los estudios de error salen de ejecutar las funciones de `physics/` con distintos N o h.
"""
from __future__ import annotations

import math
import re

import numpy as np

from content.texfmt import fx, sci
from physics import ch21_relativity as c21
from physics import ch22_temperature as c22
from physics import ch23_first_law as c23
from physics import ch24_kinetic_theory as c24
from physics import ch25_second_law as c25
from physics.integrators import euler_step

NL = r" \\ "


def _arr(cols: list[str], rows: list[list[str]]) -> str:
    """Tabla TeX de columnas centradas con encabezado."""
    spec = "c" * len(cols)
    head = " & ".join(cols)
    body = NL.join(" & ".join(r) for r in rows)
    return rf"\begin{{array}}{{{spec}}} {head} \\ \hline {body} \end{{array}}"


def _orders(errs: list[float], ratio: float = 2.0) -> list[str]:
    """Orden observado p = log(e_k-1 / e_k) / log(ratio); el primero no tiene predecesor."""
    out = ["-"]
    for a, b in zip(errs[:-1], errs[1:]):
        out.append(fx(math.log(a / b) / math.log(ratio), 2) if a > 0 and b > 0 else "-")
    return out


_SCI = re.compile(r"-?\d+(?:\.\d+)?\\times10\^\{-?\d+\}")


def _prose(t: str) -> str:
    """Envuelve en $...$ las cifras cientificas TeX que quedaron sueltas en la prosa."""
    parts = t.split("$")
    for i in range(0, len(parts), 2):
        parts[i] = _SCI.sub(lambda m: f"${m.group(0)}$", parts[i])
    return "$".join(parts)


def _clean_detail(d: dict) -> dict:
    d["numeric_intro"] = [_prose(t) for t in d["numeric_intro"]]
    d["numeric_converge"] = [(a, _prose(b), e) for a, b, e in d["numeric_converge"]]
    return d


def _fit(xs, errs) -> float:
    """Pendiente de log(error) contra log(x) por minimos cuadrados."""
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(errs, float)), 1)[0])


# =============================================================== MuonDecay
def _muon() -> dict:
    tau, beta, d, C = c21.TAU_MUON, c21.BETA_MUON, c21.D_MUON, c21.C
    g = c21.gamma(beta)
    life = g * tau
    tf = d / (beta * C)
    p_exact = c21.survival_analytic()
    n0 = 2_000_000
    p_mc, sig = c21.survival_monte_carlo()
    ns = [1_000, 10_000, 100_000, 1_000_000]
    seeds = range(1, 41)
    rows, rms = [], []
    for n in ns:
        errs = np.array([c21.survival_monte_carlo(n=n, seed=s)[0] - p_exact for s in seeds])
        r = float(np.sqrt(np.mean(errs**2)))
        rms.append(r)
        s_th = math.sqrt(p_exact * (1 - p_exact) / n)
        rows.append([f"{n}", fx(100 * c21.survival_monte_carlo(n=n, seed=3)[0], 3), sci(r, 3), sci(s_th, 3), fx(r / s_th, 2)])
    slope = _fit(ns, rms)
    p_cl = c21.survival_analytic(relativistic=False)
    need = 10.0 / p_cl
    n_sig = (1 - p_exact) / (p_exact * 1e-6)
    return dict(
        numeric_intro=[
            "El decaimiento de un muón es un proceso aleatorio: no hay una trayectoria que integrar, solo una probabilidad. Por eso aquí se usa un método de Monte Carlo, que en vez de evaluar una fórmula reproduce el experimento muchas veces en la computadora. Cada muón simulado recibe un tiempo de vida al azar; se cuenta cuántos de ellos viven más que el tiempo de vuelo y esa fracción estima la probabilidad de llegar al suelo.",
            f"El estimador es $\\hat P = \\dfrac{{1}}{{N}}\\sum_{{i=1}}^{{N}} \\mathbf 1\\!\\left[t_i > t_{{vuelo}}\\right]$, con $t_i \\sim \\mathrm{{Exp}}(\\gamma\\tau)$ obtenidos del generador pseudoaleatorio de NumPy (algoritmo PCG64) mediante la transformada inversa $t = -\\gamma\\tau \\ln u$ con $u$ uniforme. Cada muestra es un ensayo de Bernoulli con probabilidad de éxito $P$, de modo que $N\\hat P$ es binomial y la ley de los grandes números garantiza que $\\hat P\\to P$.",
            f"El error del método no es de truncamiento sino estadístico: la desviación típica de $\\hat P$ es $\\sigma = \\sqrt{{P(1-P)/N}}$, es decir, decrece como $1/\\sqrt N$ (orden $O(N^{{-1/2}})$). Es un orden lento: para ganar una cifra decimal hay que multiplicar $N$ por cien. Además, el método falla cuando el suceso es raro: si $P$ es muy pequeña casi ninguna muestra cumple la condición y la estimación es 0 o ruido puro.",
            f"Se eligió $N = 2\\times10^6$ y la semilla 3 para que el resultado sea reproducible. Con $P \\approx {fx(p_exact, 3)}$ eso da $\\sigma = {sci(sig, 3)}$, o sea un error relativo de {fx(100 * sig / p_exact, 2)} %, suficiente para distinguir la fracción relativista de la clásica. Con $N = 10^4$ la incertidumbre sería de unos {fx(100 * math.sqrt(p_exact * (1 - p_exact) / 1e4), 1)} puntos porcentuales y no habría confianza en la segunda cifra.",
            "El método es independiente del cálculo analítico porque nunca evalúa $e^{-t/\\gamma\\tau}$: solo sortea tiempos de vida y compara cada uno con el tiempo de vuelo. Lo único que comparte con la solución cerrada son los datos de entrada ($\\gamma$, $\\tau$, $d$, $\\beta$). Por eso el acuerdo dentro de $\\sigma$ es una verificación real de la fórmula y del cálculo del factor de Lorentz.",
        ],
        numeric_code=f"""import numpy as np
tau, beta, d, c = 2.197e-6, 0.995, 10_000.0, 299_792_458.0
gamma = 1 / np.sqrt(1 - beta**2)             # factor de Lorentz = {g:.4f}
vida = gamma * tau                           # vida media en el sistema de la Tierra
t_vuelo = d / (beta * c)                     # tiempo de vuelo = {tf*1e6:.3f} microsegundos
N = 2_000_000                                # numero de muones simulados
rng = np.random.default_rng(3)               # semilla fija: resultado reproducible
t_decay = rng.exponential(vida, N)           # tiempos de vida al azar, Exp(vida)
vivos = t_decay > t_vuelo                    # True si el muon llega al suelo
P = vivos.mean()                             # estimador: fraccion de exitos
sigma = np.sqrt(P * (1 - P) / N)             # error estadistico ~ 1/sqrt(N)
print(P, sigma)                              # {p_mc:.5f} +- {sig:.5f}
# control: la formula cerrada solo se usa para comparar, no para calcular P
P_exacta = np.exp(-t_vuelo / vida)           # {p_exact:.5f}
print(abs(P - P_exacta) / sigma)             # desviacion en unidades de sigma
""",
        numeric_converge=[
            ("Error contra el número de muestras", f"Se repitió la simulación con 40 semillas distintas para cada $N$ y se calculó la raíz del error cuadrático medio $\\mathrm{{RMS}}=\\sqrt{{\\langle(\\hat P-P)^2\\rangle}}$. Debe coincidir con $\\sigma=\\sqrt{{P(1-P)/N}}$ (columna 4) y, por tanto, el cociente de la última columna debe ser cercano a 1. La pendiente ajustada en el plano log-log es {fx(slope, 3)}, cercana al $-1/2$ teórico.",
             [_arr([r"N", r"\hat P\ (\%)\ \text{semilla 3}", r"\mathrm{RMS}", r"\sigma_{teo}", r"\mathrm{RMS}/\sigma_{teo}"], rows),
              rf"\text{{pendiente}} = \dfrac{{d\ln \mathrm{{RMS}}}}{{d\ln N}} = {fx(slope, 3)}"]),
            ("Fuentes de error y su tamaño aquí", f"El único error relevante es el estadístico, de {sci(sig, 3)} con $N=2\\times10^6$. No hay error de truncamiento (no se discretiza nada) y el redondeo en coma flotante, del orden de $10^{{-16}}$ por comparación, es despreciable frente a $\\sigma$. Hay un error de sesgo cero: el estimador es insesgado, así que más muestras reducen la dispersión pero no hay corrección sistemática que aplicar.",
             [rf"\sigma = \sqrt{{\dfrac{{({fx(p_exact, 5)})(1-{fx(p_exact, 5)})}}{{{n0}}}}} = {sci(sig, 3)},\qquad \dfrac{{\sigma}}{{P}} = {fx(100 * sig / p_exact, 3)}\,\%"]),
            ("Cuándo es confiable el resultado", f"Una regla práctica es exigir al menos unas 10 ocurrencias del suceso esperado, $N P \\gtrsim 10$, y luego comparar con la fórmula dentro de 2 o 3 $\\sigma$. Para la fracción relativista se cumple de sobra ($NP = {int(n0 * p_exact)}$). Para el caso clásico $P_{{cl}} = {sci(p_cl, 3)}$ harían falta {sci(need, 2)} muestras solo para ver 10 eventos, y para un error relativo de 0.1 % en $P_{{rel}}$ se necesitarían {sci(n_sig, 2)} muestras (el error relativo de la simulación es $\\sqrt{{(1-P)/(NP)}}$).",
             [rf"N_{{min}}\approx\dfrac{{10}}{{P_{{cl}}}} = {sci(need, 2)},\qquad N P_{{cl}} = {fx(n0 * p_cl, 2)}\ \text{{con }}N=2\times10^6"]),
        ],
    )


# =============================================================== VelocityAndEnergy
def _velocity() -> dict:
    b = 0.9
    ex = c21.kinetic_analytic(b)
    ns = [2, 4, 8, 16, 32, 64]
    errs = [abs(c21.kinetic_numeric(b, n=n) - ex) / ex for n in ns]
    h = [b / n for n in ns]
    ords = _orders(errs)
    rows = [[f"{n}", fx(hh, 4), sci(e, 3), o] for n, hh, e, o in zip(ns, h, errs, ords)]
    n0 = 4000
    e0 = abs(c21.kinetic_numeric(b, n=n0) - ex) / ex
    va = c21.velocity_addition_analytic(0.8, 0.8)
    vn = c21.velocity_addition_numeric(0.8, 0.8)
    vr = c21.rapidity_addition(0.8, 0.8)
    cond = float(np.linalg.cond(np.array([[c21.gamma(0.8), c21.gamma(0.8) * 0.8], [c21.gamma(0.8) * 0.8, c21.gamma(0.8)]])))
    f_top = (1 / math.sqrt(1 - b * b)) ** 3 * b
    return dict(
        numeric_intro=[
            "Esta escena usa dos métodos numéricos distintos. Para la suma de velocidades se compone la transformación de Lorentz como producto de matrices $2\\times2$ en las coordenadas $(ct, x)$ y se lee la velocidad del resultado; es álgebra lineal exacta, sin aproximación. Para la energía cinética se evalúa el trabajo relativista $K=\\int v\\,dp$ con la regla de Simpson, porque aquí se quiere comprobar $K=(\\gamma-1)mc^2$ sin suponerlo.",
            "Con $p=\\gamma m v$ y $\\beta=v/c$, un cambio de $p$ vale $dp = mc\\,\\gamma^3\\,d\\beta$, de modo que $K = mc^2\\int_0^{\\beta}\\beta'\\,\\gamma(\\beta')^3\\,d\\beta'$. Simpson reemplaza el integrando por parábolas sobre pares de subintervalos: $\\int_a^b f \\approx \\dfrac{h}{3}\\left[f_0 + 4f_1 + 2f_2 + \\cdots + 4f_{n-1} + f_n\\right]$ con $h=(b-a)/n$ y $n$ par.",
            "Simpson es exacto para polinomios de grado hasta 3 y su error global es $O(h^4)$, proporcional a $f^{(4)}$; el trapecio, en cambio, es solo $O(h^2)$. La limitación es la suavidad: el integrando $\\beta\\gamma^3$ diverge cuando $\\beta\\to1$ y su cuarta derivada crece muy rápido, por lo que cerca de $\\beta=0.9$ hacen falta más puntos que para un integrando suave y el orden 4 solo se observa cuando $h$ es lo bastante pequeño.",
            f"Se eligió $n={n0}$ subintervalos para $\\beta=0.9$: el error relativo resultante es {sci(e0, 2)}, que es el truncamiento $\\propto h^4$ ya muy pequeño, así que el número está muy por debajo de cualquier incertidumbre física. El resultado confirma que el trabajo acumulado es $(\\gamma-1)mc^2={fx(ex / c21.C**2, 5)}\\,mc^2$ y que la fórmula clásica $\\tfrac12 mv^2={fx(c21.kinetic_classical(b) / c21.C**2, 5)}\\,mc^2$ ya no sirve a esta rapidez.",
            f"Ambos cálculos son independientes de las fórmulas cerradas: la matriz producto no usa $(u+v)/(1+uv)$, solo multiplica dos matrices de impulso, y la integral no usa $\\gamma-1$, solo $\\gamma(\\beta)$ evaluada en una malla. Coinciden con la solución analítica a {sci(abs(vn - va), 1) if abs(vn - va) > 0 else '0'} en la velocidad compuesta, lo que valida tanto el álgebra como la implementación.",
        ],
        numeric_code=f"""import numpy as np
def boost(b):                                # matriz de Lorentz en (ct, x)
    g = 1 / np.sqrt(1 - b**2)                # factor gamma
    return np.array([[g, g*b], [g*b, g]])    # boost de rapidez b
M = boost(0.8) @ boost(0.8)                  # composicion de dos boosts
v_suma = M[1, 0] / M[0, 0]                   # velocidad compuesta = {vn:.6f} c
def simpson(y, x):                           # regla de Simpson, n par
    n = len(x) - 1; h = (x[-1] - x[0]) / n   # ancho de subintervalo
    return h/3 * (y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())
beta = np.linspace(0, 0.9, 4000 + 1)         # malla en beta (n = 4000)
g = 1 / np.sqrt(1 - beta**2)                 # gamma en cada nodo
dp_db = g**3                                 # dp = m c gamma^3 dbeta (m = c = 1)
K = simpson(beta * dp_db, beta)              # K/(m c^2) = integral de beta dp/dbeta
print(K)                                     # {c21.kinetic_numeric(b, n=n0)/c21.C**2:.8f}
print(g[-1] - 1)                             # control: gamma - 1 = {c21.kinetic_analytic(b)/c21.C**2:.8f}
""",
        numeric_converge=[
            ("Error de Simpson contra el número de subintervalos", f"Se calculó $K(\\beta=0.9)$ con $n=2,4,\\ldots,64$ y se comparó con $(\\gamma-1)mc^2$. La última columna es el orden observado $p=\\log_2(e_{{k-1}}/e_k)$ al duplicar $n$. Se acerca a 4, el orden teórico de Simpson, a medida que $h$ resuelve el crecimiento de $\\beta\\gamma^3$ (que llega a {fx(f_top, 2)} en el extremo).",
             [_arr([r"n", r"h", r"|K_n-K|/K", r"p"], rows)]),
            ("Suma de velocidades: condicionamiento y residuo", f"El producto de matrices es exacto salvo redondeo. Las matrices de boost tienen determinante 1 y número de condición $\\kappa={fx(cond, 2)}$ para $\\beta=0.8$, así que el redondeo no se amplifica de forma apreciable. Se comparan tres rutas: la fórmula cerrada, el producto de matrices y la suma de rapideces $\\tanh(\\operatorname{{arctanh}}u+\\operatorname{{arctanh}}v)$.",
             [rf"\dfrac{{u+v}}{{1+uv}} = {fx(va, 12)},\qquad \text{{matrices}} = {fx(vn, 12)},\qquad \text{{rapidez}} = {fx(vr, 12)}",
              rf"|v_{{mat}}-v_{{cerr}}| = {sci(abs(vn - va), 2) if abs(vn - va) > 0 else '0'}"]),
            ("Fuentes de error y cuándo confiar", f"Para la integral, el error es de truncamiento $\\propto h^4 f^{{(4)}}$ y se vuelve despreciable ({sci(e0, 2)}) con $n={n0}$. Para $\\beta\\to1$ ambos métodos se degradan: el integrando $\\gamma^3$ crece sin cota y el redondeo en $1-\\beta^2$ se amplifica. Una regla práctica es duplicar $n$ y exigir que $K$ cambie menos que la tolerancia deseada; aquí el cambio entre $n=32$ y $n=64$ ya es de {sci(abs(c21.kinetic_numeric(b, n=32) - c21.kinetic_numeric(b, n=64)) / ex, 2)} en valor relativo.",
             [rf"\left|\dfrac{{K_{{64}}-K_{{32}}}}{{K}}\right| = {sci(abs(c21.kinetic_numeric(b, n=32) - c21.kinetic_numeric(b, n=64)) / ex, 2)}"]),
        ],
    )


# =============================================================== ThermalExpansion
def _expansion() -> dict:
    n0 = 400
    ex = c22.length_analytic()
    dl = ex - c22.L0
    lin = c22.length_linear()
    f = lambda t, y: np.array([c22.alpha(t) * y[0]])  # noqa: E731
    ns = [1, 2, 4, 8, 16]
    e_eu, e_rk = [], []
    for n in ns:
        h = (c22.T1 - c22.T0) / n
        ye = np.array([c22.L0])
        t = c22.T0
        for _ in range(n):
            ye = euler_step(f, t, ye, h)
            t += h
        e_eu.append(abs(float(ye[0]) - ex))
        e_rk.append(abs(c22.length_numeric(n=n) - ex))
    ord_eu = _orders(e_eu)
    rows = [[f"{n}", fx((c22.T1 - c22.T0) / n, 2), sci(a, 3), o, sci(b, 2)] for n, a, o, b in zip(ns, e_eu, ord_eu, e_rk)]
    e_n = abs(c22.length_numeric() - ex)
    return dict(
        numeric_intro=[
            "La dilatación con coeficiente variable obedece la ecuación diferencial $dL/dT=\\alpha(T)\\,L$ con $\\alpha(T)=\\alpha_0\\left[1+\\beta(T-T_0)\\right]$. Para este $\\alpha$ lineal existe solución cerrada, pero el método numérico no la necesita: avanza la longitud paso a paso en temperatura y sirve igual para un $\\alpha(T)$ tabulado o medido, donde no hay fórmula.",
            "Se usa el método de Runge-Kutta clásico de cuarto orden (RK4). Con paso $h$ en temperatura, $L_{k+1}=L_k+\\dfrac{h}{6}(k_1+2k_2+2k_3+k_4)$ con $k_1=f(T_k,L_k)$, $k_2=f(T_k+\\tfrac h2,L_k+\\tfrac h2k_1)$, $k_3=f(T_k+\\tfrac h2,L_k+\\tfrac h2k_2)$ y $k_4=f(T_k+h,L_k+hk_3)$, siendo $f(T,L)=\\alpha(T)L$. Es una media ponderada de cuatro pendientes que coincide con el desarrollo de Taylor hasta $h^4$.",
            "El error local es $O(h^5)$ y el global $O(h^4)$: si se divide $h$ a la mitad el error baja unas 16 veces. Esto vale mientras la solución sea suave; falla o exige pasos mínimos si la ecuación es rígida o tiene saltos. Aquí $\\alpha\\Delta T\\sim10^{-3}$ es tan pequeño que la solución es casi lineal y RK4 ya es exacto a precisión de máquina con muy pocos pasos.",
            f"Se escogieron $n={n0}$ pasos, es decir $h={fx((c22.T1 - c22.T0) / n0, 2)}\\ ^\\circ\\mathrm{{C}}$, un valor muy conservador. Con él el error es de {sci(e_n, 1) if e_n > 0 else 'cero'} m sobre un alargamiento de {fx(dl * 1000, 3)} mm, mucho menor que cualquier medición real. El resultado significa que el efecto del término $\\beta$ (frente a la aproximación lineal) es genuino y no un artefacto del algoritmo.",
            "El método es independiente del analítico porque nunca usa la expresión $L_0\\exp[\\alpha_0(\\Delta T+\\beta\\Delta T^2/2)]$: integra la ecuación diferencial directamente evaluando $\\alpha(T)$ en cuatro temperaturas por paso. La coincidencia con la exponencial comprueba que la integral de $\\alpha$ se hizo bien.",
        ],
        numeric_code=f"""import numpy as np
L0, alpha0, beta, T0, T1 = 30.0, 12e-6, 2.0e-3, 20.0, 120.0
def alpha(T):                                # coeficiente variable
    return alpha0 * (1 + beta * (T - T0))    # alpha(T) lineal en T
def f(T, L):                                 # derivada dL/dT
    return alpha(T) * L                      # ecuacion diferencial
n = {n0}; h = (T1 - T0) / n                  # n pasos, h = {(c22.T1-c22.T0)/n0:.2f} grados
L, T = L0, T0
for _ in range(n):                           # avance paso a paso en temperatura
    k1 = f(T, L)                             # pendiente al inicio
    k2 = f(T + h/2, L + h/2 * k1)            # pendiente en el punto medio (1)
    k3 = f(T + h/2, L + h/2 * k2)            # pendiente en el punto medio (2)
    k4 = f(T + h, L + h * k3)                # pendiente al final
    L += h/6 * (k1 + 2*k2 + 2*k3 + k4)       # promedio ponderado RK4
    T += h                                   # siguiente temperatura
print(L - L0)                                # dilatacion = {c22.length_numeric()-c22.L0:.8f} m
# control (solo comparar): L0*exp(alpha0*(dT + beta*dT**2/2)) - L0 = {dl:.8f} m
""",
        numeric_converge=[
            ("Euler contra RK4 al refinar el paso", f"Se integró de 20 a 120 °C con $n=1,2,4,8,16$ pasos usando Euler (orden 1) y RK4 y se comparó el error absoluto en $L$ con la solución cerrada. Euler muestra el orden 1 esperado (la columna $p$ vale cerca de 1) mientras que RK4 ya está en el piso de redondeo de $L\\approx30$ m (unos $10^{{-14}}$ m) desde el primer paso, así que su pendiente no es medible: el error de truncamiento quedó por debajo del redondeo.",
             [_arr([r"n", r"h\ (^\circ\mathrm{C})", r"|e|_{Euler}\ (\mathrm{m})", r"p_{Euler}", r"|e|_{RK4}\ (\mathrm{m})"], rows)]),
            ("Magnitud de las fuentes de error", f"El alargamiento total es {fx(dl * 1000, 4)} mm. Con RK4 y $n={n0}$ el error es {sci(e_n, 1) if e_n > 0 else '0'} m, es decir, un error relativo de {sci(e_n / dl, 1) if e_n > 0 else '0'} en el alargamiento y gobernado por el redondeo (acumulado en $\\sim n$ sumas). El truncamiento es despreciable. Conviene no sobredimensionar $n$: más pasos solo acumulan más redondeo sin mejorar el resultado.",
             [rf"\Delta L_{{RK4}} = {fx(c22.length_numeric() - c22.L0, 8)}\ \mathrm{{m}},\qquad \Delta L_{{cerrado}} = {fx(dl, 8)}\ \mathrm{{m}}"]),
            ("Sensibilidad al parámetro β", f"El resultado es sensible al coeficiente cuadrático: si se ignorara $\\beta$ (modelo lineal con $\\alpha_0$ constante) el alargamiento sería {fx((lin - c22.L0) * 1000, 4)} mm, un {fx(100 * (dl - (lin - c22.L0)) / dl, 1)} % menor que el real de {fx(dl * 1000, 4)} mm. Esa diferencia es mucho mayor que el error del método, por lo que el efecto de $\\beta$ se resuelve con claridad. Criterio de confianza: el resultado es fiable si al duplicar $n$ el alargamiento cambia menos que la precisión con que se conoce $\\alpha_0$ (típicamente 1 %).",
             [rf"\Delta L_{{lineal}} = L_0\alpha_0\Delta T = {fx((lin - c22.L0) * 1000, 4)}\ \mathrm{{mm}},\qquad \Delta L_{{\beta}} = {fx(dl * 1000, 4)}\ \mathrm{{mm}}"]),
        ],
    )


# =============================================================== ThermalEquilibrium
def _equilibrium() -> dict:
    t_end, n0 = 60.0, 1200
    ta = c22.temperatures_analytic(t_end)[0]
    tf = c22.final_temperature_analytic()
    rate = c22.rate_analytic()
    ns = [6, 12, 24, 48, 96, 192]
    errs = [abs(c22.temperatures_numeric(t_end, n)[1][-1, 0] - ta) for n in ns]
    ords = _orders(errs)
    rows = [[f"{n}", fx(t_end / n, 3), sci(e, 3), o] for n, e, o in zip(ns, errs, ords)]
    y = c22.temperatures_numeric(t_end, n0)[1]
    e0 = abs(y[-1, 0] - ta)
    c1, c2 = c22.M1 * c22.C1, c22.M2 * c22.C2
    en0 = c1 * c22.T1_0 + c2 * c22.T2_0
    en = c1 * y[:, 0] + c2 * y[:, 1]
    drift = float(np.max(np.abs(en - en0)) / en0)
    h_max = 2.785 / rate
    return dict(
        numeric_intro=[
            "Dos cuerpos en contacto intercambian calor según la ley de enfriamiento de Newton, lo que da un sistema de ecuaciones diferenciales acopladas para $T_1(t)$ y $T_2(t)$. Existe solución analítica porque el sistema es lineal, pero el método numérico trata el problema como una ecuación general $\\dot{\\vec y}=\\vec f(t,\\vec y)$ y vale igual si la conductancia depende de la temperatura o si hay más de dos cuerpos.",
            "Se integra con RK4 de forma vectorial: $\\vec y_{k+1}=\\vec y_k+\\dfrac h6(\\vec k_1+2\\vec k_2+2\\vec k_3+\\vec k_4)$, con $\\dot T_1=-\\dfrac{hA}{m_1c_1}(T_1-T_2)$ y $\\dot T_2=+\\dfrac{hA}{m_2c_2}(T_1-T_2)$ (aquí la conductancia es $20\\ \\mathrm{W/K}$). Las cuatro pendientes se evalúan sobre el vector completo, de modo que el acoplamiento entre los cuerpos queda consistente dentro de cada paso.",
            f"RK4 tiene error global $O(h^4)$. Además debe ser estable: para un modo que decae como $e^{{-\\lambda t}}$ el método es estable si $\\lambda h \\lesssim 2.78$. Aquí $\\lambda={fx(rate, 5)}\\ \\mathrm{{s^{{-1}}}}$, por lo que el paso máximo estable sería {fx(h_max, 0)} s; usar pasos cercanos a ese valor produciría oscilaciones o resultados absurdos. Un problema rígido (conductancia enorme frente a las capacidades) exigiría un método implícito.",
            f"Se escogió $T_{{fin}}=60\\ \\mathrm{{s}}$ con $n={n0}$ pasos, es decir $h={fx(t_end / n0, 3)}\\ \\mathrm{{s}}$, unas {fx(h_max / (t_end / n0), 0)} veces menor que el límite de estabilidad y unas {fx(1 / rate / (t_end / n0), 0)} veces menor que la constante de tiempo $1/\\lambda={fx(1 / rate, 1)}\\ \\mathrm{{s}}$. El resultado numérico es la curva completa de ambas temperaturas y su valor límite $T_f={fx(tf, 3)}\\ ^\\circ\\mathrm{{C}}$.",
            "Es independiente del método analítico porque no usa ni la temperatura final por balance de energía ni la exponencial: solo aplica la ley de transferencia de calor en cada instante. Que las temperaturas simuladas converjan a $T_f$ y que la energía total se conserve a lo largo de la evolución son comprobaciones de consistencia que el integrador no tiene incorporadas.",
        ],
        numeric_code=f"""import numpy as np
m1, c1, T1 = 0.30, 900.0, 150.0              # aluminio: masa, calor especifico, T inicial
m2, c2, T2 = 0.50, 4186.0, 20.0              # agua
hA = 20.0                                    # conductancia termica [W/K]
def f(t, y):                                 # derivadas de las dos temperaturas
    q = hA * (y[0] - y[1])                   # potencia de calor de 1 a 2
    return np.array([-q/(m1*c1), q/(m2*c2)]) # cada cuerpo cambia segun C = m c
n, t_end = {n0}, 60.0                        # pasos y tiempo final
h = t_end / n                                # paso = {t_end/n0:.3f} s
y = np.array([T1, T2])                       # estado inicial
for _ in range(n):                           # RK4 vectorial
    k1 = f(0, y)                             # cuatro pendientes sobre el vector
    k2 = f(0, y + h/2 * k1)                  # (el sistema es autonomo: t no importa)
    k3 = f(0, y + h/2 * k2)
    k4 = f(0, y + h * k3)
    y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)    # promedio ponderado
print(y)                                     # T1, T2 en t = 60 s: {y[-1,0]:.4f}, {y[-1,1]:.4f}
E = m1*c1*y[0] + m2*c2*y[1]                  # energia: debe quedar constante
""",
        numeric_converge=[
            ("Error de RK4 contra el paso", f"Se integró hasta $t=60$ s con distintos $n$ y se midió el error absoluto de $T_1$ frente a la exponencial. El orden observado, $p=\\log_2(e_{{k-1}}/e_k)$, tiende a 4 al reducir $h$, confirmando el $O(h^4)$ teórico. Para $n$ pequeño ($h=10$ s) $\\lambda h={fx(rate * 10, 2)}$ y el método sigue estable pero aún inexacto.",
             [_arr([r"n", r"h\ (\mathrm{s})", r"|T_1^{num}-T_1|\ (^\circ\mathrm{C})", r"p"], rows)]),
            ("Magnitud del error con el paso elegido", f"Con $n={n0}$ el error en $T_1(60\\,\\mathrm{{s}})$ es {sci(e0, 2) if e0 > 0 else '0'} $^\\circ$C, es decir, está en el piso del redondeo; el truncamiento ya no domina. La energía total $E=m_1c_1T_1+m_2c_2T_2$ se conserva a lo largo de toda la evolución con una deriva relativa máxima de {sci(drift, 2) if drift > 0 else '0'}, y ese residuo no usa la solución analítica.",
             [rf"\max_t\left|\dfrac{{E(t)-E_0}}{{E_0}}\right| = {sci(drift, 2) if drift > 0 else '0'}"]),
            ("Estabilidad y criterio de confianza", f"El límite de estabilidad de RK4 para este sistema es $h<2.78/\\lambda={fx(h_max, 1)}$ s. El resultado es confiable si (i) $h\\lambda\\ll1$, (ii) al duplicar $n$ las temperaturas cambian menos que la tolerancia (aquí por debajo de $10^{{-6}}\\ ^\\circ$C ya con $n=100$) y (iii) la energía total se mantiene constante. Las tres condiciones se cumplen con holgura.",
             [rf"\lambda = h_{{A}}\left(\dfrac{{1}}{{m_1c_1}}+\dfrac{{1}}{{m_2c_2}}\right) = {fx(rate, 5)}\ \mathrm{{s^{{-1}}}},\qquad \lambda h = {sci(rate * t_end / n0, 3)}"]),
        ],
    )


# =============================================================== GasProcesses
def _gas_processes() -> dict:
    n0 = 400
    ns = [4, 8, 16, 32, 64]
    ei = [abs(c23.work_numeric("isothermal", n=n) - c23.work_analytic("isothermal")) for n in ns]
    ea = [abs(c23.work_numeric("adiabatic", n=n) - c23.work_analytic("adiabatic")) for n in ns]
    eb = [abs(c23.work_numeric("isobaric", n=n) - c23.work_analytic("isobaric")) for n in ns]
    oi, oa = _orders(ei), _orders(ea)
    rows = [[f"{n}", sci(a, 3), o1, sci(b, 3), o2] for n, a, o1, b, o2 in zip(ns, ei, oi, ea, oa)]
    wi, wa, wb = (c23.work_analytic(k) for k in ("isothermal", "adiabatic", "isobaric"))
    wn = {k: c23.work_numeric(k, n=n0) for k in ("isothermal", "isobaric", "adiabatic")}
    return dict(
        numeric_intro=[
            "El trabajo de un gas es el área bajo la curva $p(V)$: $W=\\int_{V_1}^{V_2}p\\,dV$. Para un gas ideal las tres trayectorias (isoterma, isóbara, adiabática) tienen primitivas conocidas, pero el método numérico muestra cómo se obtendría el trabajo de cualquier proceso, incluso uno medido punto a punto. Para la adiabática, además, ni siquiera se usa la relación $pV^\\gamma=\\text{cte}$: se integra la ecuación diferencial que la origina.",
            "El cálculo se hace en dos etapas. Primero se obtiene $p(V)$: en la isoterma y la isóbara se evalúa directamente en una malla uniforme de $V$, y en la adiabática se resuelve $\\dfrac{dp}{dV}=-\\gamma\\dfrac{p}{V}$ con Runge-Kutta de cuarto orden. Después se integra $p(V)$ con la regla de Simpson, $W\\approx\\dfrac h3\\left[p_0+4p_1+2p_2+\\cdots+p_n\\right]$ con $h=(V_2-V_1)/n$.",
            f"Tanto RK4 como Simpson tienen error global $O(h^4)$, de modo que el trabajo hereda el orden 4: al duplicar $n$ el error baja unas 16 veces. Simpson es exacto para polinomios cúbicos, por eso la isóbara (constante) sale exacta salvo redondeo. Las limitaciones aparecen si $p(V)$ tiene esquinas (cambios de proceso) o singularidades; entonces conviene integrar tramo por tramo, como se hace en los ciclos.",
            f"Se usó $n={n0}$ subintervalos entre {fx(c23.V1, 3)} y {fx(c23.V2, 3)} m³. El error resultante es de {sci(abs(wn['isothermal'] - wi), 1)} J en la isoterma y {sci(abs(wn['adiabatic'] - wa), 1)} J en la adiabática, frente a trabajos de {fx(wi, 1)} J y {fx(wa, 1)} J: errores relativos por debajo de $10^{{-9}}$. La primera ley $Q=\\Delta U+W$ se aplica después con $\\Delta U=nC_V\\Delta T$.",
            "Es independiente del cálculo analítico: no usa $nRT\\ln(V_2/V_1)$ ni $(p_1V_1-p_2V_2)/(\\gamma-1)$. Solo evalúa $p(V)$ en una malla (o integra su ecuación diferencial) y suma áreas. La concordancia de las tres trayectorias con sus fórmulas cerradas comprueba el álgebra de cada proceso.",
        ],
        numeric_code=f"""import numpy as np
R, n_mol, gamma = 8.314, 1.0, 5/3            # gas monoatomico ideal
p1, V1, V2 = 2.0e5, 0.010, 0.020             # estado inicial y volumen final
n = {n0}                                     # subintervalos (par, para Simpson)
V = np.linspace(V1, V2, n + 1)               # malla uniforme en el volumen
def simpson(y, x):                           # regla de Simpson
    h = (x[-1] - x[0]) / (len(x) - 1)        # ancho de cada subintervalo
    return h/3 * (y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())
p_iso = p1 * V1 / V                          # isoterma: p V = cte
p_isob = np.full_like(V, p1)                 # isobara: p constante
p = np.empty(n + 1); p[0] = p1               # adiabata: se integra la EDO con RK4
h = V[1] - V[0]
for i in range(n):                           # dp/dV = -gamma p / V
    f = lambda v, y: -gamma * y / v          # lado derecho de la ecuacion
    k1 = f(V[i], p[i]); k2 = f(V[i]+h/2, p[i]+h/2*k1)
    k3 = f(V[i]+h/2, p[i]+h/2*k2); k4 = f(V[i]+h, p[i]+h*k3)
    p[i+1] = p[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)   # paso RK4
W = [simpson(q, V) for q in (p_iso, p_isob, p)]     # trabajo = area bajo p(V)
print(W)                                     # {wn['isothermal']:.3f}, {wn['isobaric']:.1f}, {wn['adiabatic']:.3f} J
""",
        numeric_converge=[
            ("Error del trabajo contra el número de subintervalos", f"Se calculó $W$ con $n=4,\\ldots,64$ y se comparó con la solución cerrada. Para la isoterma el error es solo el de Simpson y para la adiabática se suman RK4 y Simpson; ambos muestran el orden observado cercano a 4 (columnas $p$). La isóbara, con $p$ constante, tiene error de redondeo en todos los casos (por debajo de {sci(max(eb), 1) if max(eb) > 0 else '10^{-9}'} J).",
             [_arr([r"n", r"|e_W|_{iso}\ (\mathrm{J})", r"p", r"|e_W|_{adia}\ (\mathrm{J})", r"p"], rows)]),
            ("Fuentes de error en esta escena", f"El error de truncamiento domina para $n\\lesssim100$ y decrece como $h^4$; el redondeo es del orden de $10^{{-16}}$ relativo por operación y no se alcanza con $n={n0}$. Con ese $n$ los errores relativos son {sci(abs(wn['isothermal'] - wi) / wi, 1)} (isoterma) y {sci(abs(wn['adiabatic'] - wa) / wa, 1)} (adiabática). Interpretación: para ver una cifra más basta duplicar $n$, el costo es lineal en $n$.",
             [rf"\dfrac{{|W_{{iso}}^{{num}}-W_{{iso}}|}}{{W_{{iso}}}} = {sci(abs(wn['isothermal'] - wi) / wi, 2)},\qquad \dfrac{{|W_{{adi}}^{{num}}-W_{{adi}}|}}{{W_{{adi}}}} = {sci(abs(wn['adiabatic'] - wa) / wa, 2)}"]),
            ("Criterio práctico", "El resultado es confiable si al duplicar $n$ el trabajo cambia menos que la tolerancia requerida (la diferencia entre $n$ y $2n$ estima el error de $n$, con factor $1/15$ para un método de orden 4). Además, se verifica la coherencia termodinámica: en la adiabática el calor $Q=\\Delta U+W$ debe valer cero salvo el error numérico, y eso usa la temperatura final calculada de forma independiente.",
             [rf"Q_{{adi}} = \Delta U + W = {fx(c23.first_law_table()[2]['dU'], 4)} + {fx(c23.first_law_table()[2]['W'], 4)} = {sci(c23.first_law_table()[2]['Q'], 2) if abs(c23.first_law_table()[2]['Q']) > 1e-9 else '0'}\ \mathrm{{J}}"]),
        ],
    )


# =============================================================== OttoCycle
def _otto() -> dict:
    n0 = 800
    ex = c23.otto_efficiency_analytic()
    ns = [2, 4, 8, 16, 32, 64]
    errs = [abs(c23.otto_numeric(n=n)["eta"] - ex) for n in ns]
    ords = _orders(errs)
    rows = [[f"{n}", fx(c23.otto_numeric(n=n)["eta"], 8), sci(e, 3), o] for n, e, o in zip(ns, errs, ords)]
    r0 = c23.otto_numeric()
    e0 = abs(r0["eta"] - ex)
    ws = [c23.otto_numeric(r=r)["eta"] for r in (6.0, 8.0, 10.0)]
    ex_r = [c23.otto_efficiency_analytic(r) for r in (6.0, 8.0, 10.0)]
    return dict(
        numeric_intro=[
            "El rendimiento de un ciclo es $\\eta=W_{net}/Q_{in}$. En el ciclo de Otto el trabajo neto sale de dos adiabáticas (compresión y expansión), porque las dos isocoras no hacen trabajo. El método numérico calcula esas áreas bajo $p(V)$ integrando la ecuación diferencial adiabática con RK4 y sumando con Simpson, en lugar de usar la fórmula $\\eta=1-r^{1-\\gamma}$.",
            "En cada adiabática se resuelve $\\dfrac{dp}{dV}=-\\gamma\\dfrac pV$ con RK4 desde el estado inicial (A para la compresión, C para la expansión), obteniendo $p(V)$ en $n+1$ puntos; después $W=\\int p\\,dV\\approx\\dfrac h3\\left[p_0+4p_1+2p_2+\\cdots+p_n\\right]$. El calor de entrada se obtiene de la isocora de calentamiento, $Q_{in}=nC_V(T_C-T_B)$, con $C_V=R/(\\gamma-1)$.",
            f"Los dos métodos tienen error global $O(h^4)$, así que $\\eta$ hereda el orden 4 respecto al número de pasos $n$. La principal limitación es que el modelo supone gas ideal con $\\gamma$ constante y procesos reversibles; el método numérico verifica la aritmética del ciclo, no la física del motor real. Por eso el valor $\\eta={fx(ex, 4)}$ es un techo teórico, no el rendimiento de un motor de gasolina.",
            f"Se usó $n={n0}$ pasos por adiabática con razón de compresión $r={c23.RATIO:g}$ y $\\gamma=7/5$ (gas diatómico). El error resultante en $\\eta$ es {sci(e0, 1) if e0 > 0 else '0'}, mucho menor que cualquier incertidumbre experimental. Los estados A, B, C y D se obtienen de las relaciones de estado, y el calor $Q_{{in}}={fx(r0['Q_in'], 1)}$ J y el trabajo $W_{{net}}={fx(r0['W_net'], 1)}$ J son los que se interpretan.",
            "El método es independiente del analítico: el cociente $W_{net}/Q_{in}$ se forma con integrales numéricas y no usa $r^{1-\\gamma}$. Que ambos coincidan confirma que las aproximaciones de la solución cerrada (cancelaciones entre $T_A,T_B,T_C,T_D$) están bien aplicadas.",
        ],
        numeric_code=f"""import numpy as np
R, n_mol, gamma, r = 8.314, 1.0, 7/5, 8.0     # gas diatomico, razon de compresion
Ta, Tc, Va = 300.0, 1800.0, 0.025            # T de admision, T tras la combustion, V inicial
Vb = Va / r                                  # volumen minimo (fin de la compresion)
Tb = Ta * r**(gamma - 1)                     # T al final de la compresion = {c23.otto_states()['B'][2]:.2f} K
pa = n_mol*R*Ta/Va; pc = n_mol*R*Tc/Vb      # presiones en A y en C
def adiabata(p0, v0, v1, n):                 # integra dp/dV = -gamma p/V con RK4
    v = np.linspace(v0, v1, n + 1); p = np.empty(n + 1); p[0] = p0
    h = v[1] - v[0]; f = lambda x, y: -gamma * y / x   # h < 0 en la compresion
    for i in range(n):
        k1 = f(v[i], p[i]); k2 = f(v[i]+h/2, p[i]+h/2*k1)
        k3 = f(v[i]+h/2, p[i]+h/2*k2); k4 = f(v[i]+h, p[i]+h*k3)
        p[i+1] = p[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)    # un paso RK4
    return v, p
n = {n0}                                     # pasos por adiabata
W_ab = simpson(*adiabata(pa, Va, Vb, n)[::-1])   # compresion: negativo (sobre el gas)
W_cd = simpson(*adiabata(pc, Vb, Va, n)[::-1])   # expansion: positivo
Qin = n_mol * R/(gamma-1) * (Tc - Tb)        # calor en la isocora B -> C
eta = (W_ab + W_cd) / Qin                    # rendimiento = {r0['eta']:.8f}
""",
        numeric_converge=[
            ("Rendimiento contra el número de pasos", f"Se calculó $\\eta$ con $n=2,\\ldots,64$ pasos por adiabática y se comparó con $1-r^{{1-\\gamma}}={fx(ex, 8)}$. El error cae con el orden observado cercano a 4 (última columna), el esperado para RK4 combinado con Simpson. Con muy pocos pasos ($n=2$) el error es grande porque el volumen cambia en un factor 8 y las parábolas no representan bien $p\\propto V^{{-1.4}}$.",
             [_arr([r"n", r"\eta_n", r"|\eta_n-\eta|", r"p"], rows)]),
            ("Fuentes de error y su magnitud", f"Con $n={n0}$ el error es de {sci(e0, 1) if e0 > 0 else '0'} y proviene del truncamiento de RK4 y de Simpson, que decrece como $h^4$; el redondeo (cerca de $10^{{-16}}$) se alcanza solo con pasos mucho más finos. No hay errores estadísticos ni de discretización temporal: el ciclo es cuasiestático. Una verificación extra que no usa la fórmula cerrada es el balance de calor: el trabajo neto debe igualar el calor absorbido menos el expulsado.",
             [rf"W_{{net}}={fx(r0['W_net'], 3)}\ \mathrm{{J}},\qquad Q_{{in}}={fx(r0['Q_in'], 3)}\ \mathrm{{J}},\qquad \eta = {fx(r0['eta'], 8)}"]),
            ("Sensibilidad a la razón de compresión", f"Si se cambia $r$ el método debe seguir a la fórmula con el mismo error. Con $n={n0}$ se obtiene lo siguiente, que también permite decidir cuándo el cálculo es confiable: si la columna de diferencia permanece del orden de $10^{{-9}}$ o menos, el método reproduce la tendencia de $\\eta$ creciente con $r$.",
             [_arr([r"r", r"\eta_{num}", r"\eta_{teo}", r"|\Delta|"], [[f"{r:g}", fx(a, 8), fx(b, 8), sci(abs(a - b), 2) if abs(a - b) > 0 else "0"] for r, a, b in zip((6.0, 8.0, 10.0), ws, ex_r)])]),
        ],
    )


# =============================================================== GasPressure
def _pressure() -> dict:
    n0 = 4000
    dur, dt0 = 2.0e-8, 2.0e-12
    p_an = c24.pressure_analytic(n0)
    p_sim, t_sim = c24.pressure_simulation(n=n0)
    ns = [125, 500, 2000, 8000]
    seeds = range(21, 41)
    rows, rms = [], []
    for n in ns:
        rel = []
        for s in seeds:
            p, _ = c24.pressure_simulation(n=n, dt=2e-11, seed=s)
            rel.append(p / c24.pressure_analytic(n) - 1)
        r = float(np.sqrt(np.mean(np.square(rel))))
        rms.append(r)
        rows.append([f"{n}", sci(r, 3), fx(r * math.sqrt(n), 2)])
    slope = _fit(ns, rms)
    dts = [1e-10, 2e-11, 2e-12]
    rowsd = []
    for dt in dts:
        p, t = c24.pressure_simulation(n=2000, dt=dt, seed=21)
        rowsd.append([sci(dt, 1), f"{int(round(dur / dt))}", fx(100 * (p / c24.pressure_analytic(2000) - 1), 2), fx(t, 2)])
    return dict(
        numeric_intro=[
            "Esta escena es una simulación de dinámica molecular simplificada: se siguen $N$ moléculas independientes que vuelan en línea recta y rebotan elásticamente en las paredes de un cubo. No se calcula una presión a partir de $pV=NkT$; se mide como lo define la teoría cinética, es decir, impulso transferido a la pared por unidad de tiempo y de área. Ese es el sentido en que la simulación comprueba la ley de los gases.",
            "El algoritmo avanza con pasos de tiempo $\\Delta t$: $\\vec x\\leftarrow\\vec x+\\vec v\\,\\Delta t$; si una molécula atraviesa la pared $x=L$ se la refleja ($x\\leftarrow2L-x$, $v_x\\leftarrow-v_x$) y se acumula el impulso $2mv_x$. La presión es $p=\\dfrac{\\sum 2mv_x}{\\Delta t_{tot}\\,L^2}$. Las velocidades iniciales se sortean de una gaussiana de desviación $\\sqrt{kT/m}$ por componente y las posiciones de forma uniforme; la temperatura se estima luego de $\\tfrac32kT=\\langle K\\rangle$.",
            f"El movimiento libre y la reflexión especular son exactos, así que $\\Delta t$ no introduce error de integración (solo decide con qué resolución se detecta cada choque) y el error es esencialmente estadístico: $p$ se compone de $N$ velocidades aleatorias, y su desviación relativa crece como $1/\\sqrt N$ (orden $O(N^{{-1/2}})$). La limitación es el tamaño de la muestra y el tiempo simulado: con pocas moléculas la presión fluctúa mucho, y si el tiempo es menor que unos pocos cruces de la caja, el promedio de impulsos no converge.",
            f"Se eligieron $N={n0}$ moléculas de helio a 300 K en una caja de 1 µm, una duración de {sci(dur, 1)} s y $\\Delta t={sci(dt0, 1)}$ s. A la rapidez típica de 790 m/s una molécula cruza la caja en un tiempo de unos 1.3 ns, así que cada una la recorre unas {fx(dur * 790 / 1e-6, 0)} veces y {int(round(dur / dt0))} pasos resuelven bien cada cruce. La semilla 21 hace el resultado reproducible. El resultado es una presión de {fx(p_sim, 3)} Pa frente a {fx(p_an, 3)} Pa de la ley de los gases.",
            f"Es independiente del método analítico porque no usa $p=NkT/V$ en la medición: usa solo la mecánica de choques elásticos. La temperatura de entrada interviene únicamente para sortear las velocidades. La comparación posterior con $NkT/V$ evalúa si la teoría cinética reproduce la ecuación de estado, y la que se hace con la temperatura efectiva de la muestra ({fx(t_sim, 1)} K) separa el error estadístico de las velocidades del error propio de la medición.",
        ],
        numeric_code=f"""import numpy as np
kB, m, T, L = 1.380649e-23, 6.646e-27, 300.0, 1e-6   # constante de Boltzmann, masa de He, T, lado de la caja
N, dur, dt = {n0}, 2.0e-8, 2.0e-12                   # moleculas, tiempo total, paso
rng = np.random.default_rng(21)              # semilla fija
sigma = np.sqrt(kB * T / m)                  # desviacion de cada componente de v
v = rng.normal(0, sigma, (N, 3))             # velocidades de Maxwell-Boltzmann
x = rng.uniform(0, L, (N, 3))                # posiciones uniformes en la caja
impulso = 0.0                                # impulso acumulado en la pared x = L
for _ in range(int(round(dur / dt))):        # {int(round(dur/dt0))} pasos de tiempo
    x += v * dt                              # movimiento libre
    hi = x[:, 0] > L                         # moleculas que cruzaron la pared derecha
    impulso += np.sum(2 * m * v[hi, 0])      # cada choque entrega 2 m vx
    x[hi, 0] = 2*L - x[hi, 0]; v[hi, 0] *= -1   # reflexion elastica
    lo = x[:, 0] < 0                         # pared izquierda: solo refleja
    x[lo, 0] *= -1; v[lo, 0] *= -1
    # (las caras y, z se reflejan igual para mantener las moleculas dentro)
p_sim = impulso / (dur * L**2)               # presion medida = {p_sim:.1f} Pa
T_sim = m * np.mean(np.sum(v**2, axis=1)) / (3 * kB)   # temperatura de la muestra = {t_sim:.2f} K
print(p_sim, N * kB * T / L**3)              # comparar con N k T / V = {p_an:.1f} Pa
""",
        numeric_converge=[
            ("Error contra el número de moléculas", f"Para cada $N$ se corrieron 20 semillas distintas y se calculó el error relativo cuadrático medio de $p_{{sim}}$ frente a $NkT/V$. La tabla muestra que el error baja como $N^{{-1/2}}$: la pendiente ajustada es {fx(slope, 2)} y el producto $\\mathrm{{RMS}}\\sqrt N$ es aproximadamente constante (última columna). Eso indica que el error es estadístico, no sistemático. (Se usó $\\Delta t=2\\times10^{{-11}}$ s para acelerar el barrido.)",
             [_arr([r"N", r"\mathrm{RMS}\left(p/p_{teo}-1\right)", r"\mathrm{RMS}\sqrt N"], rows),
              rf"\text{{pendiente}} = {fx(slope, 3)}\ \ (\text{{teórico }}-0.5)"]),
            ("Sensibilidad al paso de tiempo", "Al ser el movimiento libre y la reflexión exactos, cambiar $\\Delta t$ debería afectar poco el resultado; lo que cambia es solo cuántos choques se registran en el paso. Con $N=2000$ y la misma semilla, la presión relativa respecto de $NkT/V$ prácticamente no varía y la temperatura efectiva de la muestra queda fija porque las velocidades sorteadas son las mismas.",
             [_arr([r"\Delta t\ (\mathrm{s})", r"\text{pasos}", r"p/p_{teo}-1\ (\%)", r"T_{muestra}\ (\mathrm{K})"], rowsd)]),
            ("Fuentes de error y criterio de confianza", f"Con $N={n0}$ el error observado es de {fx(100 * (p_sim / p_an - 1), 2)} % en $p$ y de {fx(100 * (t_sim / c24.T_GAS - 1), 2)} % en la temperatura efectiva, coherente con la escala $\\sqrt{{2/N}}\\approx{fx(100 * math.sqrt(2 / n0), 1)}\\,\\%$ de las fluctuaciones del promedio de $v_x^2$. Si se compara con la ley de los gases usando la temperatura efectiva de la muestra, la presión coincide al {fx(100 * (p_sim / c24.pressure_analytic(n0, t_sim) - 1), 2)} %, lo que muestra que lo medido es consistente con $p=NkT/V$ y que la diferencia restante proviene del número finito de choques contados en un tiempo finito (cada molécula da un número entero de golpes). Criterio: el resultado es confiable si $\\mathrm{{RMS}}\\sim1/\\sqrt N$ y la duración abarca muchos cruces de la caja.",
             [rf"p_{{sim}} = {fx(p_sim, 1)}\ \mathrm{{Pa}},\qquad \dfrac{{N k T_{{sim}}}}{{V}} = {fx(c24.pressure_analytic(n0, t_sim), 1)}\ \mathrm{{Pa}},\qquad \dfrac{{NkT}}{{V}} = {fx(p_an, 1)}\ \mathrm{{Pa}}"]),
        ],
    )


# =============================================================== MaxwellBoltzmann
def _maxwell() -> dict:
    an = c24.speeds_analytic()
    n0 = 4000
    ns = [6, 12, 24, 48, 96, 192]
    errs = [abs(c24.moments_by_quadrature(n=n)["v_mean"] - an["v_mean"]) / an["v_mean"] for n in ns]
    ords = _orders(errs)
    rows = [[f"{n}", fx(6000 / n, 1), sci(e, 3), o] for n, e, o in zip(ns, errs, ords)]
    q0 = c24.moments_by_quadrature()
    nss = [1_000, 10_000, 100_000, 400_000]
    seeds = range(1, 11)
    rows2, rms = [], []
    for n in nss:
        er = np.array([c24.speeds_sampled(n=n, seed=s)["v_mean"] / an["v_mean"] - 1 for s in seeds])
        r = float(np.sqrt(np.mean(er**2)))
        rms.append(r)
        rows2.append([f"{n}", sci(r, 3), fx(r * math.sqrt(n), 3)])
    slope = _fit(nss, rms)
    sp = c24.speeds_sampled()
    return dict(
        numeric_intro=[
            "La distribución de Maxwell-Boltzmann da la fracción de moléculas con rapidez entre $v$ y $v+dv$. El método numérico la estudia de dos maneras que no usan las fórmulas cerradas de $v_{mp}$, $\\bar v$ y $v_{rms}$: (a) un muestreo de Monte Carlo, que simula las moléculas, y (b) una cuadratura de Simpson de los momentos $\\int v^kf(v)\\,dv$ para comprobar la normalización y los promedios.",
            "En el muestreo cada componente cartesiana de la velocidad se sortea de una gaussiana $N(0,\\sqrt{kT/m})$ y la rapidez es $v=\\sqrt{v_x^2+v_y^2+v_z^2}$; esa construcción da la distribución $f(v)=4\\pi\\left(\\dfrac{m}{2\\pi kT}\\right)^{3/2}v^2e^{-mv^2/2kT}$. Los promedios son medias muestrales $\\bar v=\\dfrac1N\\sum v_i$, $v_{rms}=\\sqrt{\\overline{v^2}}$, y la moda se localiza en el bin más poblado del histograma.",
            "El error del muestreo es estadístico, $O(N^{-1/2})$: la desviación típica de la media de $v$ es $\\sigma_v/\\sqrt N$. El error de la moda es peor, porque está limitado por el ancho del bin (25 m/s aquí). Simpson, en cambio, tiene error $O(h^4)$ para integrandos suaves como $v^kf(v)$ sobre un intervalo largo donde $f\\approx0$ en el extremo; falla si el intervalo $[0,v_{max}]$ no cubre la cola o si la malla es más gruesa que el ancho de la campana.",
            f"Se usó $N=400\\,000$ moléculas con semilla 8, 60 bins de 25 m/s en $[0,1500]$ m/s y, para la cuadratura, $n={n0}$ subintervalos en $[0,6000]$ m/s ($h={fx(6000 / n0, 1)}$ m/s, mucho menor que el ancho de la campana, cerca de 300 m/s). El resultado de la cuadratura reproduce $\\int f\\,dv={fx(q0['norm'], 8)}$, $\\bar v={fx(q0['v_mean'], 2)}$ m/s y $v_{{rms}}={fx(q0['v_rms'], 2)}$ m/s; el muestreo da $\\bar v={fx(sp['v_mean'], 1)}$ m/s.",
            "Ambos son independientes de los resultados analíticos de las rapideces características: el muestreo parte de gaussianas de las componentes y la cuadratura solo evalúa $f(v)$ en puntos; en ningún caso se usan $\\sqrt{8kT/\\pi m}$ ni $\\sqrt{3kT/m}$. Comprobar que ambas rutas coinciden entre sí y con esas fórmulas valida tanto el modelo como el código.",
        ],
        numeric_code=f"""import numpy as np
kB, m, T = 1.380649e-23, 4.652e-26, 300.0    # constante de Boltzmann, masa de N2, temperatura
# (a) muestreo Monte Carlo
N = 400_000
rng = np.random.default_rng(8)               # semilla fija
comp = rng.normal(0, np.sqrt(kB*T/m), (N, 3))   # tres componentes gaussianas
v = np.linalg.norm(comp, axis=1)             # rapidez de cada molecula
v_mean, v_rms = v.mean(), np.sqrt((v**2).mean())   # promedios muestrales
hist, e = np.histogram(v, bins=60, range=(0, 1500), density=True)   # bins de 25 m/s
v_mp = (0.5*(e[1:] + e[:-1]))[hist.argmax()]  # moda = centro del bin mas poblado
# (b) cuadratura de Simpson de los momentos de f(v)
def f(v):                                    # densidad de Maxwell-Boltzmann
    a = m / (2*np.pi*kB*T)
    return 4*np.pi * a**1.5 * v**2 * np.exp(-m*v**2/(2*kB*T))
vv = np.linspace(0, 6000, {n0} + 1)          # malla fina hasta 6000 m/s (cola ~ 0)
h = vv[1] - vv[0]
S = lambda y: h/3*(y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())   # Simpson
norm, mean = S(f(vv)), S(vv*f(vv))           # integral de f y de v f = {q0['norm']:.6f}, {q0['v_mean']:.2f}
rms = np.sqrt(S(vv**2 * f(vv)))              # sqrt de la integral de v^2 f
print(v_mean, v_rms, v_mp)                   # {sp['v_mean']:.1f}, {sp['v_rms']:.1f}, {sp['v_mp']:.1f} m/s
""",
        numeric_converge=[
            ("Cuadratura de Simpson contra el número de subintervalos", f"Se calculó $\\bar v=\\int vf\\,dv$ en $[0,6000]$ m/s con $n=6,\\ldots,192$ y se comparó con $\\sqrt{{8kT/\\pi m}}={fx(an['v_mean'], 3)}$ m/s. El orden observado es irregular mientras la malla no resuelve la campana (con $h=1000$ m/s ni siquiera la ve) y se estabiliza cerca de 4 cuando $h\\lesssim125$ m/s, como predice la teoría de Simpson.",
             [_arr([r"n", r"h\ (\mathrm{m/s})", r"|\bar v_n-\bar v|/\bar v", r"p"], rows)]),
            ("Error del muestreo contra el número de moléculas", f"Para cada $N$ se corrieron 10 semillas y se calculó el error relativo cuadrático medio de $\\bar v$. La pendiente ajustada en log-log es {fx(slope, 2)}, cerca del $-1/2$ teórico, y $\\mathrm{{RMS}}\\sqrt N$ queda casi constante (en la teoría vale $\\sigma_v/\\bar v\\approx0.42$).",
             [_arr([r"N", r"\mathrm{RMS}(\bar v/\bar v_{teo}-1)", r"\mathrm{RMS}\sqrt N"], rows2)]),
            ("Fuentes de error y criterio de confianza", f"Con $N=4\\times10^5$ el error relativo típico de $\\bar v$ es de unos {sci(rms[-1], 2)}, y para la cuadratura con $n={n0}$ es {sci(abs(q0['v_mean'] - an['v_mean']) / an['v_mean'], 1) if q0['v_mean'] != an['v_mean'] else '0'}. La moda tiene un error por discretización de hasta medio bin: se obtuvo {fx(sp['v_mp'], 1)} m/s frente a {fx(an['v_mp'], 1)} m/s (diferencia {fx(abs(sp['v_mp'] - an['v_mp']), 1)} m/s). Es confiable si la normalización de la cuadratura es 1 con la tolerancia deseada ({fx(q0['norm'], 8)} aquí) y el error de la media muestral es menor que la resolución que se exige.",
             [rf"\int_0^{{6000}} f\,dv = {fx(q0['norm'], 8)},\qquad v_{{mp}}^{{mues}} = {fx(sp['v_mp'], 1)}\ \mathrm{{m/s}},\qquad v_{{mp}} = {fx(an['v_mp'], 1)}\ \mathrm{{m/s}}"]),
        ],
    )


# =============================================================== CarnotCycle
def _carnot() -> dict:
    n0 = 2000
    ex = c25.carnot_efficiency_analytic()
    ns = [2, 4, 8, 16, 32, 64]
    res = [c25.carnot_numeric(n=n) for n in ns]
    errs = [abs(r["eta"] - ex) for r in res]
    ords = _orders(errs)
    closure = [abs(r["pa_end"] - r["pa"]) / r["pa"] for r in res]
    rows = [[f"{n}", fx(r["eta"], 8), sci(e, 3), o, sci(c, 2)] for n, r, e, o, c in zip(ns, res, errs, ords, closure)]
    r0 = c25.carnot_numeric()
    e0 = abs(r0["eta"] - ex)
    st = c25.carnot_states()
    return dict(
        numeric_intro=[
            "El ciclo de Carnot tiene cuatro ramas: dos isotermas y dos adiabáticas. Su rendimiento numérico se obtiene calculando el trabajo neto como la suma de las cuatro áreas bajo $p(V)$ y dividiéndolo entre el calor absorbido $Q_h$ en la isoterma caliente. Se hace así para comprobar que $\\eta=1-T_c/T_h$ se cumple sin suponerlo.",
            "Cada rama se trata con su método. En las isotermas $p=nRT/V$ se evalúa directamente en una malla y se integra con Simpson, $\\int_a^bf\\,dV\\approx\\dfrac h3\\left[f_0+4f_1+2f_2+\\cdots+f_n\\right]$. En las adiabáticas se integra $dp/dV=-\\gamma p/V$ con RK4 y luego se aplica Simpson al resultado. Como en la isoterma $\\Delta U=0$, el calor es igual al trabajo, $Q_h=W_{AB}$.",
            f"Los dos métodos son de orden 4 en el paso, por lo que $\\eta$ converge como $O(h^4)$. Una virtud adicional es la prueba de cierre: al integrar la adiabática $D\\to A$ con RK4 desde $p_D$, la presión final debe volver a $p_A$ si el ciclo está bien cerrado. Una limitación es que los volúmenes $V_C$ y $V_D$ se calculan con la relación adiabática analítica; el método numérico no sustituye esa geometría, solo integra sobre ella.",
            f"Se usó $n={n0}$ subintervalos por rama, $T_h={c25.TH:g}$ K, $T_c={c25.TC:g}$ K y $\\gamma=5/3$. El resultado es $\\eta={fx(r0['eta'], 8)}$ frente a $1-T_c/T_h={fx(ex, 8)}$, con diferencia de {sci(e0, 1) if e0 > 0 else '0'}. El cierre del ciclo es de {sci(abs(r0['pa_end'] - r0['pa']) / r0['pa'], 1) if r0['pa_end'] != r0['pa'] else '0'} en presión relativa, lo que significa que cada rama está bien integrada.",
            "El método es independiente del analítico: no usa $1-T_c/T_h$ ni el cociente de calores; el rendimiento se forma como el cociente entre dos áreas numéricas. Que coincidan confirma la termodinámica del ciclo y la aritmética de $V_C,V_D$.",
        ],
        numeric_code=f"""import numpy as np
R, n_mol, gamma = 8.314, 1.0, 5/3            # gas ideal monoatomico
Th, Tc, Va, Vb = 500.0, 300.0, 1.0e-3, 2.0e-3   # temperaturas y volumenes de A y B
k = (Th / Tc)**(1/(gamma - 1))               # relacion adiabatica T V^(g-1) = cte
Vc, Vd = Vb * k, Va * k                      # volumenes de C y D = {st['v_c']*1e3:.3f} y {st['v_d']*1e3:.3f} litros
n = {n0}                                     # subintervalos por rama
def isoterma(T, v0, v1):                     # trabajo = int n R T / V dV (simpson como arriba)
    v = np.linspace(v0, v1, n + 1)
    return simpson(n_mol*R*T/v, v)           # Simpson sobre p(V)
def adiabata(p0, v0, v1):                    # RK4 para dp/dV = -gamma p / V
    v = np.linspace(v0, v1, n + 1); p = np.empty(n + 1); p[0] = p0
    h = v[1] - v[0]; f = lambda x, y: -gamma*y/x
    for i in range(n):
        k1 = f(v[i], p[i]); k2 = f(v[i]+h/2, p[i]+h/2*k1)
        k3 = f(v[i]+h/2, p[i]+h/2*k2); k4 = f(v[i]+h, p[i]+h*k3)
        p[i+1] = p[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)    # paso RK4
    return simpson(p, v), p[-1]              # trabajo y presion final
W_ab = isoterma(Th, Va, Vb)                  # A->B, expansion isoterma (= Qh)
W_bc, _ = adiabata(n_mol*R*Th/Vb, Vb, Vc)    # B->C, expansion adiabatica
W_cd = isoterma(Tc, Vc, Vd)                  # C->D, compresion isoterma (negativo)
W_da, p_final = adiabata(n_mol*R*Tc/Vd, Vd, Va)   # D->A, compresion adiabatica
eta = (W_ab + W_bc + W_cd + W_da) / W_ab     # rendimiento = {r0['eta']:.8f}
print(eta, p_final / (n_mol*R*Th/Va))        # control de cierre: p_final/p_A = {r0['pa_end']/r0['pa']:.10f}
""",
        numeric_converge=[
            ("Rendimiento, error y cierre contra el número de pasos", f"Se calculó $\\eta$ con $n=2,\\ldots,64$ subintervalos por rama y se comparó con $1-T_c/T_h={fx(ex, 8)}$. El orden observado se acerca a 4 y el cierre $|p_A^{{fin}}-p_A|/p_A$ (la presión que se recupera al terminar la última adiabática) cae de manera similar, lo que da una verificación que no requiere conocer $\\eta$ de antemano.",
             [_arr([r"n", r"\eta_n", r"|\eta_n-\eta|", r"p", r"\text{cierre}"], rows)]),
            ("Fuentes de error en esta escena", f"El error de truncamiento domina para $n$ pequeño y decrece como $h^4$. Con $n={n0}$ el error en $\\eta$ es de {sci(e0, 1) if e0 > 0 else '0'}, comparable al redondeo acumulado, y el trabajo neto vale {fx(r0['W_net'], 4)} J con $Q_h={fx(r0['Q_h'], 4)}$ J. No hay error estadístico ni temporal: el ciclo es cuasiestático. Los volúmenes $V_C,V_D$ vienen de la geometría adiabática del ciclo, no se integran.",
             [rf"W_{{net}}={fx(r0['W_net'], 4)}\ \mathrm{{J}},\qquad Q_h={fx(r0['Q_h'], 4)}\ \mathrm{{J}},\qquad \eta={fx(r0['eta'], 8)}"]),
            ("Criterio de confianza", f"Una regla práctica es exigir dos cosas: que al duplicar $n$ el rendimiento cambie menos que la tolerancia (con $n=32$ el cambio a $n=64$ es {sci(abs(res[4]['eta'] - res[5]['eta']), 1)}) y que el ciclo cierre en presión con precisión similar. Con $n={n0}$ ambas condiciones se satisfacen con varios órdenes de magnitud de holgura.",
             [rf"|\eta_{{64}}-\eta_{{32}}| = {sci(abs(res[4]['eta'] - res[5]['eta']), 2)},\qquad \dfrac{{|p_A^{{fin}}-p_A|}}{{p_A}}\Big|_{{n={n0}}} = {sci(abs(r0['pa_end'] - r0['pa']) / r0['pa'], 2) if r0['pa_end'] != r0['pa'] else '0'}"]),
        ],
    )


# =============================================================== EntropyProduction
def _entropy() -> dict:
    sa = c25.heating_entropy_analytic()
    ns = [2, 4, 8, 16, 32, 64]
    errs = [abs(c25.heating_entropy_numeric(n=n) - sa) for n in ns]
    ords = _orders(errs)
    rows = [[f"{n}", sci(e, 3), o] for n, e, o in zip(ns, errs, ords)]
    sb = c25.contact_entropy_analytic()
    nc = [30, 60, 120, 240, 480]
    ec = [abs(c25.contact_entropy_numeric(n=n) - sb) for n in nc]
    oc = _orders(ec)
    rowsc = [[f"{n}", fx(120 / n, 2), sci(e, 3), o] for n, e, o in zip(nc, ec, oc)]
    ts = [30.0, 60.0, 120.0, 240.0]
    et = [abs(c25.contact_entropy_numeric(t_end=t, n=int(t * 10)) - sb) for t in ts]
    rowst = [[f"{t:g}", sci(math.exp(-c22.rate_analytic() * t), 2), sci(e, 3)] for t, e in zip(ts, et)]
    n_h, n_c = 600, 6000
    eh = abs(c25.heating_entropy_numeric() - sa)
    ecd = abs(c25.contact_entropy_numeric() - sb)
    return dict(
        numeric_intro=[
            "La entropía es una función de estado, pero se define con una integral sobre un camino reversible, $\\Delta S=\\int dQ_{rev}/T$. Este método la calcula con integración numérica de dos formas: para el calentamiento de agua con Simpson sobre $\\dfrac{mc}{T}\\,dT$, y para el contacto entre dos cuerpos integrando la tasa de producción de entropía $\\dot S=h\\,\\dfrac{(T_1-T_2)^2}{T_1T_2}$ junto con la evolución de las temperaturas. La segunda requiere resolver una EDO porque $T_1,T_2$ dependen del tiempo.",
            "En el calentamiento se aplica la regla de Simpson a $f(T)=mc/T$ con $h=(T_2-T_1)/n$: $\\Delta S\\approx\\dfrac h3\\left[f_0+4f_1+2f_2+\\cdots+f_n\\right]$. En el contacto se añade una tercera variable $S(t)$ al sistema de EDO de las temperaturas y se integra el conjunto con RK4, $\\vec y_{k+1}=\\vec y_k+\\dfrac h6(\\vec k_1+2\\vec k_2+2\\vec k_3+\\vec k_4)$, de modo que $S$ se acumula con el mismo paso y el mismo orden de exactitud que las temperaturas.",
            f"Simpson y RK4 son de orden 4 en el paso: $O(h^4)$. Además hay un error de truncamiento temporal propio de la escena: la integración se detiene en $t_{{fin}}=120$ s, y la entropía que se produciría después se pierde. Como las temperaturas se acercan como $e^{{-\\lambda t}}$ y $\\dot S\\propto(T_1-T_2)^2\\propto e^{{-2\\lambda t}}$, la cola omitida decae como $e^{{-2\\lambda t_{{fin}}}}$. Un método así falla si $t_{{fin}}$ es corto frente a $1/\\lambda$ o si $h\\lambda$ no es pequeño.",
            f"Se usaron $n={n_h}$ subintervalos para el calentamiento (error {sci(eh, 1)} J/K) y $n={n_c}$ pasos con $h={fx(120 / n_c, 3)}$ s hasta $t_{{fin}}=120$ s para el contacto (error {sci(ecd, 1)} J/K). La constante de tiempo es $1/\\lambda={fx(1 / c22.rate_analytic(), 1)}$ s, por lo que 120 s son unas {fx(120 * c22.rate_analytic(), 1)} constantes de tiempo y la cola es mínima. El resultado, $\\Delta S_{{total}}={fx(sb, 4)}$ J/K, es positivo, como exige la segunda ley.",
            "Es independiente del analítico: no usa $mc\\ln(T_2/T_1)$ ni la temperatura final por balance de energía. El primer caso integra $1/T$ en una malla y el segundo integra la tasa de producción de entropía, una cantidad que no aparece en el cálculo analítico (que suma $C_1\\ln(T_f/T_{1})+C_2\\ln(T_f/T_{2})$). Su coincidencia comprueba que la entropía generada por el flujo de calor irreversible es igual al cambio de entropía de los dos cuerpos.",
        ],
        numeric_code=f"""import numpy as np
# (a) calentamiento reversible: dS = m c dT / T, integrado con Simpson
m, c, T1, T2, n = 1.0, 4186.0, 293.15, 353.15, {n_h}     # kg, J/(kg K), Kelvin, subintervalos
T = np.linspace(T1, T2, n + 1)               # malla de temperaturas
h = T[1] - T[0]
f = m * c / T                                # integrando mc/T
dS = h/3 * (f[0] + f[-1] + 4*f[1:-1:2].sum() + 2*f[2:-1:2].sum())   # Simpson = {c25.heating_entropy_numeric():.6f} J/K
# (b) contacto entre dos cuerpos: y = [T1, T2, S], S acumula la produccion de entropia
C1, C2, hA = 0.30*900.0, 0.50*4186.0, 20.0   # capacidades termicas y conductancia
def rhs(t, y):
    q = hA * (y[0] - y[1])                   # potencia de calor 1 -> 2
    return np.array([-q/C1, q/C2,            # evolucion de las temperaturas
                     hA*(y[0]-y[1])**2 / ((y[0]+273.15)*(y[1]+273.15))])   # dS/dt = hA (T1-T2)^2/(T1 T2)
y, N, t_end = np.array([150.0, 20.0, 0.0]), {n_c}, 120.0
dt = t_end / N                               # paso = {120/n_c:.3f} s
for _ in range(N):                           # RK4 sobre las tres variables
    k1 = rhs(0, y); k2 = rhs(0, y + dt/2*k1); k3 = rhs(0, y + dt/2*k2); k4 = rhs(0, y + dt*k3)
    y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)     # paso RK4
print(y[2])                                  # entropia producida = {c25.contact_entropy_numeric():.6f} J/K
""",
        numeric_converge=[
            ("Simpson en el calentamiento", f"Se calculó $\\Delta S$ del calentamiento de 1 kg de agua de 20 a 80 °C con $n=2,\\ldots,64$ y se comparó con $mc\\ln(T_2/T_1)={fx(sa, 6)}$ J/K. El orden observado se acerca a 4 y el error con $n={n_h}$ queda por debajo de {sci(max(eh, 1e-15), 1)} J/K.",
             [_arr([r"n", r"|\Delta S_n-\Delta S|\ (\mathrm{J/K})", r"p"], rows)]),
            ("RK4 en el contacto entre los cuerpos", f"Se integró la producción de entropía hasta $t=120$ s con distintos números de pasos. La tabla muestra el orden observado del conjunto RK4 + acumulación de $S$, con la solución cerrada $\\Delta S={fx(sb, 6)}$ J/K como referencia; el orden observado ronda 4 (el último valor ya toca el piso de redondeo, por eso se dispara) y con el paso usado ($h=0.02$ s) el error es {sci(ecd, 1)} J/K.",
             [_arr([r"n", r"h\ (\mathrm{s})", r"|\Delta S_n-\Delta S|\ (\mathrm{J/K})", r"p"], rowsc)]),
            ("Error por tiempo final finito y criterio de confianza", f"Con $h=0.1$ s fijo, el error depende de cuánto tiempo se integra, porque $S$ sigue creciendo mientras haya diferencia de temperatura. La cola restante se amortigua como $e^{{-2\\lambda t}}$ y se vuelve despreciable cuando $t_{{fin}}\\gg1/(2\\lambda)$. Criterios de confianza: (i) $h\\lambda\\ll1$, (ii) $e^{{-\\lambda t_{{fin}}}}$ pequeño y (iii) $\\Delta S_{{total}}>0$ (la segunda ley actúa como control). Con $t_{{fin}}=120$ s la diferencia de temperatura restante es $e^{{-\\lambda t}}={sci(math.exp(-c22.rate_analytic() * 120), 2)}$ de la inicial.",
             [_arr([r"t_{fin}\ (\mathrm{s})", r"e^{-\lambda t_{fin}}", r"|\Delta S-\Delta S_{cerr}|\ (\mathrm{J/K})"], rowst)]),
        ],
    )


DETAIL = {k: _clean_detail(v) for k, v in {
    "MuonDecay": _muon(),
    "VelocityAndEnergy": _velocity(),
    "ThermalExpansion": _expansion(),
    "ThermalEquilibrium": _equilibrium(),
    "GasProcesses": _gas_processes(),
    "OttoCycle": _otto(),
    "GasPressure": _pressure(),
    "MaxwellBoltzmann": _maxwell(),
    "CarnotCycle": _carnot(),
    "EntropyProduction": _entropy(),
}.items()}
