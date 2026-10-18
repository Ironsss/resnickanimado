"""Explicacion del metodo numerico (intro, codigo, convergencia): Volumen 1, capitulos 13 a 16.

Escenas: SkaterSpin, GyroscopePrecession, LadderFriction, BeamReactions, KeplerOrbit,
EscapeVelocity, PressureDepth, FloatingCylinder. Todas las cifras de los estudios de error
salen de ejecutar (o replicar con otro N / h) las funciones de `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch13_angular_momentum as c13
from physics import ch14_equilibrium as c14
from physics import ch15_gravitation as c15
from physics import ch16_fluid_statics as c16
from physics.integrators import euler_step, integrate, rk4_step


def _slope(hs, errs) -> float:
    """Pendiente de log(err) contra log(h) por minimos cuadrados (orden observado)."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def _array(head: str, rows: list[str], cols: str) -> str:
    """Tabla TeX: head y cada fila ya vienen con '&' entre columnas."""
    return r"\begin{array}{" + cols + r"} " + head + r"\\ \hline " + r"\\ ".join(rows) + r"\end{array}"


# ============================================================================ SkaterSpin
def _skater() -> dict:
    wf = c13.omega_final_analytic()
    f = lambda t, y: np.array([-(c13.inertia_rate(t) / c13.inertia_profile(t)) * y[0]])  # noqa: E731

    def omega_end(n, step=rk4_step):
        _, y = integrate(f, np.array([c13.W_I]), 0.0, 4.0, n, step)
        return float(y[-1, 0])

    ns = [20, 40, 80, 160, 320]
    hs = [4.0 / n for n in ns]
    es = [abs(omega_end(n) - wf) for n in ns]
    rows = []
    for i, n in enumerate(ns):
        ratio = "-" if i == 0 else fx(es[i - 1] / es[i], 1)
        rows.append(f"{n} & {sci(hs[i], 3)} & {sci(es[i], 3)} & {ratio}")
    tab = _array(r"N & h\,(\mathrm{s}) & |\omega_f^{num}-\omega_f^{an}|\,(\mathrm{rad/s}) & \text{cociente}", rows, "r|rrr")
    p = _slope(hs, es)
    e4000 = abs(omega_end(4000) - wf)
    ee = [abs(omega_end(n, euler_step) - wf) for n in (1000, 4000)]
    pe = _slope([4.0 / 1000, 4.0 / 4000], ee)
    _, w, inertia, ang, _k = c13.skater_numeric()
    dl = float(np.ptp(ang))
    return dict(
        numeric_intro=[
            "Cuando no hay torque externo, el momento angular $L = I\\omega$ se conserva, y si $I$ cambia con el tiempo se puede despejar $\\omega = L/I$ de inmediato. El método numérico parte de otra idea: no usa la conservación, sino la ecuación diferencial $\\dot\\omega = -(\\dot I/I)\\,\\omega$ que resulta de derivar $I\\omega$ y pedir $\\tau = 0$. Se integra paso a paso y después se verifica a posteriori que $L$ quedó constante, de modo que la conservación es un resultado del cálculo y no un dato de entrada.",
            "El integrador es Runge-Kutta de cuarto orden. Para $\\dot y = f(t,y)$ con paso $h$ calcula cuatro pendientes, $k_1 = f(t_n,y_n)$, $k_2 = f(t_n + h/2,\\ y_n + \\tfrac h2 k_1)$, $k_3 = f(t_n + h/2,\\ y_n + \\tfrac h2 k_2)$, $k_4 = f(t_n + h,\\ y_n + h k_3)$, y avanza con el promedio ponderado $y_{n+1} = y_n + \\tfrac h6(k_1 + 2k_2 + 2k_3 + k_4)$. Los pesos $1,2,2,1$ son los de la regla de Simpson aplicada a $f$ dentro del paso, y por eso el desarrollo de Taylor del método coincide con el exacto hasta $h^4$.",
            "El error local de truncamiento es $O(h^5)$ y, acumulado sobre $N = T/h$ pasos, el error global es $O(h^4)$: reducir $h$ a la mitad divide el error por $2^4 = 16$. El método supone que $f$ es suficientemente suave; aquí $I(t)$ es un cúbico con derivada nula en los extremos, de modo que $\\dot I$ es continua, aunque su derivada salta en $t = 0$ y $t = 2$ s, y eso es lo que degrada ligeramente el orden observado. Si $\\dot I$ tuviera un salto, RK4 caería a orden 1 o 2 en el paso que contiene la discontinuidad.",
            "Se eligieron 4000 pasos en $[0, 4]$ s, es decir $h = 10^{-3}$ s, que es mucho más fino que la escala de variación de $I$ (2 s); con ese $h$ el error es del orden del redondeo y la gráfica de $L(t)$ es plana a simple vista. El resultado numérico $\\omega_f \\approx 5$ rad/s significa que la patinadora gira 2.5 veces más rápido al recoger los brazos. El método es independiente del analítico porque nunca usa $\\omega_f = I_i\\omega_i/I_f$: solo conoce el perfil $I(t)$ y su derivada.",
        ],
        numeric_code="""# RK4 para  dw/dt = -(I'(t)/I(t)) * w   (sin torque externo)
I_i, I_f, T_pull = 4.0, 1.6, 2.0                  # inercia inicial, final y duracion del tiron
def I(t):                                         # perfil cubico suave entre 0 y T_pull
    s = clip(t / T_pull, 0, 1)                    # s = fraccion del tiron realizado
    return I_i + (I_f - I_i) * (3*s**2 - 2*s**3)  # derivada nula en s=0 y s=1
def dI(t):                                        # derivada analitica del perfil
    s = clip(t / T_pull, 0, 1)
    return (I_f - I_i) * (6*s - 6*s**2) / T_pull if 0 < t < T_pull else 0.0
f = lambda t, w: -(dI(t) / I(t)) * w              # lado derecho de la EDO
N, t_end = 4000, 4.0
h = t_end / N                                     # paso fijo h = 1e-3 s
w = 2.0                                           # condicion inicial omega(0) = 2 rad/s
for n in range(N):
    t = n * h
    k1 = f(t, w)                                  # pendiente al inicio
    k2 = f(t + h/2, w + h/2 * k1)                 # pendiente a mitad de paso (1)
    k3 = f(t + h/2, w + h/2 * k2)                 # pendiente a mitad de paso (2)
    k4 = f(t + h, w + h * k3)                     # pendiente al final
    w += h/6 * (k1 + 2*k2 + 2*k3 + k4)           # promedio ponderado 1-2-2-1
L = I(t_end) * w                                  # verificacion posterior: L debe ser 8""",
        numeric_converge=[
            ("Error contra el paso: orden observado",
             "Se repite la integración con $N = 20, 40, \\dots, 320$ pasos y se compara $\\omega_f$ con el valor analítico. Cada duplicación de $N$ divide el error por un factor cercano a 16, que es lo esperado de un método de orden 4.",
             [tab, rf"p_{{obs}} = \dfrac{{\Delta\log|e|}}{{\Delta\log h}} = {p:.2f}\qquad (\text{{teórico: }}4)"]),
            ("Fuentes de error y comparación con Euler",
             f"El error de truncamiento domina mientras $h$ es grande; con $N = 4000$ ya llega a la zona de redondeo de doble precisión. Para dar escala, el método de Euler ($y_{{n+1}} = y_n + hf$, orden 1) con el mismo problema converge con pendiente {fx(pe,2)} y necesita muchos más pasos para igualar a RK4.",
             [rf"|\omega_f^{{num}} - \omega_f^{{an}}|_{{N=4000}} = {sci(e4000, 2)}\ \mathrm{{rad/s}},\qquad \Delta L = L_{{max}} - L_{{min}} = {sci(dl, 2)}\ \mathrm{{kg\,m^2/s}}",
              rf"\text{{Euler}}:\ N=1000 \to {sci(ee[0], 3)},\quad N=4000 \to {sci(ee[1], 3)}\ \mathrm{{rad/s}}"]),
            ("Criterio práctico de confiabilidad",
             "No se necesita conocer la respuesta exacta: basta repetir con $h/2$ y comprobar que $\\omega_f$ cambia menos que la tolerancia deseada, y que $L = I\\omega$ permanece constante. Esta invariante es un control independiente porque el integrador no la impone.",
             [rf"\text{{Se acepta si }}\ |\omega_f(h) - \omega_f(h/2)| < 10^{{-8}}\ \mathrm{{rad/s}}\ \text{{ y }}\ \Delta L/L < 10^{{-10}}\ \Rightarrow\ \text{{aquí }}\ {sci(abs(omega_end(2000) - omega_end(4000)), 2)}\ \mathrm{{rad/s}}"]),
        ],
    )


# ============================================================================ GyroscopePrecession
def _gyro() -> dict:
    an = c13.precession_analytic()
    ns = [30, 60, 120, 240, 480]
    hs = [6.0 / n for n in ns]
    rel, drift = [], []
    for n in ns:
        _, y, _phi, om = c13.precession_numeric(n=n)
        rel.append(abs(om - an) / an)
        drift.append(abs(np.linalg.norm(y[-1]) - y[0, 0]) / y[0, 0])
    rows = [f"{n} & {sci(hs[i], 3)} & {sci(rel[i], 3)} & {sci(drift[i], 3)}" for i, n in enumerate(ns)]
    tab = _array(r"N & h\,(\mathrm{s}) & |\Omega^{num}-\Omega^{an}|/\Omega^{an} & ||\vec L|-L_0|/L_0", rows, "r|rrr")
    p1, p2 = _slope(hs, rel), _slope(hs, drift)
    # Euler: |L| crece porque el torque es perpendicular a L
    big_l = c13.I_GYRO * c13.W_SPIN

    def rhs(_t, s):
        return np.cross(c13.R_GYRO * s / np.linalg.norm(s), np.array([0.0, 0.0, -c13.M_GYRO * c13.G]))

    _, ye = integrate(rhs, np.array([big_l, 0.0, 0.0]), 0.0, 6.0, 6000, euler_step)
    drift_e = abs(np.linalg.norm(ye[-1]) - big_l) / big_l
    _, y0_, _p, om0 = c13.precession_numeric()
    om400 = c13.precession_numeric(w_spin=400.0)[3]
    om_an400 = c13.precession_analytic(400.0)
    phi_total = an * 6.0
    return dict(
        numeric_intro=[
            "La precesión del giroscopio es una consecuencia de la ecuación vectorial $d\\vec L/dt = \\vec\\tau$, y la fórmula $\\Omega = mgr/(I\\omega)$ solo vale en la aproximación de rotación rápida. El método numérico integra la ecuación vectorial completa en tres dimensiones y obtiene $\\Omega$ midiendo cuánto gira el ángulo azimutal de $\\vec L$ por unidad de tiempo. Se necesita porque, en general, el sistema tiene un torque que depende de la orientación del propio $\\vec L$ y no hay una solución cerrada para cualquier condición inicial.",
            "Se resuelve con RK4 aplicado a un vector: cada pendiente $k_i$ es un vector de tres componentes, $\\vec k_1 = \\vec\\tau(\\vec L_n)$, $\\vec k_2 = \\vec\\tau(\\vec L_n + \\tfrac h2\\vec k_1)$, etc., y $\\vec L_{n+1} = \\vec L_n + \\tfrac h6(\\vec k_1 + 2\\vec k_2 + 2\\vec k_3 + \\vec k_4)$. Aquí $\\vec\\tau = r\\,\\hat L\\times m\\vec g$ con $\\hat L = \\vec L/|\\vec L|$, es decir, el centro de masa se supone sobre el eje de giro. Después se calcula $\\phi = \\operatorname{atan2}(L_y, L_x)$ (con desenvolvimiento de saltos de $2\\pi$) y se ajusta una recta $\\phi = \\Omega t + \\phi_0$ por mínimos cuadrados.",
            "Como el torque es siempre perpendicular a $\\vec L$, el módulo $|\\vec L|$ debe conservarse y el movimiento es una rotación rígida de $\\vec L$ alrededor de la vertical. RK4 tiene error global $O(h^4)$ en la fase, pero no conserva $|\\vec L|$ exactamente: lo desvía en $O(h^4)$ también, a diferencia de un esquema de rotación exacta. El método de Euler, en cambio, aumenta $|\\vec L|$ en cada paso por un factor $\\sqrt{1 + (\\Omega h)^2}$ y la órbita se abre en espiral. La limitación del modelo es físico y no numérico: ignora la nutación y supone el eje horizontal.",
            "Se tomaron 6000 pasos en 6 s ($h = 10^{-3}$ s), lo que da unos 5.9 rad de ángulo recorrido en total (casi una vuelta) y, por tanto, menos de $10^{-3}$ rad de giro por paso: muy lejos de cualquier inestabilidad. El ajuste lineal usa todos los puntos, de modo que el redondeo se promedia. Es un método independiente del analítico porque ni siquiera conoce $I$ ni $\\omega$ en la fórmula de $\\Omega$: solo recibe $\\vec L_0 = I\\omega\\hat x$, $m$, $g$ y $r$, y la frecuencia de precesión aparece como resultado de la geometría de $\\vec L\\times\\vec g$.",
        ],
        numeric_code=f"""# RK4 vectorial para  dL/dt = r_cm x (m g)   con  r_cm = r * L/|L|
m, r, g, I, w_spin = {c13.M_GYRO}, {c13.R_GYRO}, {c13.G}, {c13.I_GYRO}, {c13.W_SPIN}   # datos del problema
weight = array([0, 0, -m * g])                    # peso (hacia -z)
def torque(L):
    axis = L / norm(L)                            # direccion del eje = direccion de L
    return cross(r * axis, weight)                # tau = r x (m g), siempre perpendicular a L
N, t_end = 6000, 6.0
h = t_end / N                                     # h = 1e-3 s
L = array([I * w_spin, 0, 0])                     # L inicial horizontal: {big_l:.2f} kg m^2/s
phi = []
for n in range(N):
    k1 = torque(L)                                # pendiente vectorial (inicio)
    k2 = torque(L + h/2 * k1)                     # pendiente a mitad de paso
    k3 = torque(L + h/2 * k2)                     # otra a mitad de paso
    k4 = torque(L + h * k3)                       # pendiente al final
    L = L + h/6 * (k1 + 2*k2 + 2*k3 + k4)         # RK4 aplicado componente a componente
    phi.append(atan2(L[1], L[0]))                 # angulo azimutal del eje
phi = unwrap(phi)                                 # evita saltos de 2*pi
Omega = polyfit(t, phi, 1)[0]                     # pendiente de la recta = precesion""",
        numeric_converge=[
            ("Error contra el paso",
             "Se repite el cálculo con $N = 30, \\dots, 480$ pasos en 6 s. El error relativo de $\\Omega$ y la deriva del módulo de $\\vec L$ caen con pendientes de unas 3.5 y 5 en escala log-log, alrededor del 4 teórico de RK4 (el ajuste lineal de $\\phi$ añade algo de ruido a la primera).",
             [tab, rf"p_\Omega = {p1:.2f},\qquad p_{{|L|}} = {p2:.2f}\qquad (\text{{teórico: }}4)"]),
            ("Comparación con Euler y sensibilidad a $\\omega$",
             f"Con Euler y los mismos 6000 pasos el módulo de $\\vec L$ crece ${sci(drift_e, 2)}$ en términos relativos, unas nueve órdenes de magnitud peor que RK4 con $N=480$, que usa doce veces menos pasos. Además se comprueba la dependencia física: al duplicar el giro, la precesión se reduce a la mitad.",
             [rf"\dfrac{{\Delta|L|}}{{L}}_{{Euler}} = {sci(drift_e, 2)}\qquad\text{{frente a}}\qquad \dfrac{{\Delta|L|}}{{L}}_{{RK4}} = {sci(drift[-1], 2)}\ (N=480)",
              rf"\Omega_{{num}}(200) = {om0:.5f},\ \Omega_{{num}}(400) = {om400:.5f}\ \mathrm{{rad/s}};\qquad \Omega_{{an}}(400) = {om_an400:.5f}\ \mathrm{{rad/s}}"]),
            ("Fuentes de error y criterio práctico",
             f"El error de truncamiento de RK4 es el dominante y se puede reducir a voluntad; el redondeo aparece con $N = 6000$, donde el error relativo ya está cerca de $10^{{-15}}$. Un resultado es confiable cuando (i) $|\\vec L|$ se conserva a mejor que $10^{{-8}}$ y (ii) los residuos del ajuste lineal de $\\phi(t)$ son del orden del redondeo: aquí el ángulo total recorrido es {phi_total:.3f} rad y la pendiente se obtiene sin ambigüedad.",
             [rf"\dfrac{{|\Omega_{{num}} - \Omega_{{an}}|}}{{\Omega_{{an}}}}\Big|_{{N=6000}} = {sci(abs(om0 - an) / an, 2)}"]),
        ],
    )


# ============================================================================ LadderFriction
def _bisect_d(k: int, theta=c14.THETA_DEG) -> float:
    lo, hi = 0.0, c14.L_LAD
    for _ in range(k):
        mid = 0.5 * (lo + hi)
        fr = c14.ladder_forces_numeric(mid, theta)
        if fr["f"] <= c14.MU * fr["N_floor"]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ladder() -> dict:
    dmax = c14.d_max_analytic()
    ks = [5, 10, 20, 30, 40, 52]
    errs = [abs(_bisect_d(k) - dmax) for k in ks]
    rows = [f"{k} & {sci(c14.L_LAD / 2**k, 3)} & {sci(errs[i], 3)}" for i, k in enumerate(ks)]
    tab = _array(r"k & L/2^k\,(\mathrm{m}) & |d_k-d_{max}|\,(\mathrm{m})", rows, "r|rr")
    th = np.deg2rad(c14.THETA_DEG)
    a = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, c14.L_LAD * np.sin(th)]])
    conds = {}
    for ang in (20.0, 60.0, 89.0):
        t_ = np.deg2rad(ang)
        a_ = a.copy()
        a_[2, 2] = c14.L_LAD * np.sin(t_)
        conds[ang] = float(np.linalg.cond(a_))
    fr = c14.ladder_forces_numeric(2.5)
    b = np.array([0.0, c14.W_LAD + c14.W_MAN, (c14.W_LAD * c14.L_LAD / 2 + c14.W_MAN * 2.5) * np.cos(th)])
    x = np.array([fr["N_floor"], fr["f"], fr["N_wall"]])
    res = float(np.linalg.norm(a @ x - b))
    dn = c14.d_max_numeric()
    # medida de la sensibilidad: error en mu -> error en d_max
    mu2 = c14.MU * 1.01
    dd = c14.d_max_analytic(mu2) - dmax
    return dict(
        numeric_intro=[
            "La pregunta de esta escena es hasta dónde puede subir una persona antes de que la escalera resbale. La condición de deslizamiento, $f \\le \\mu_s N_{suelo}$, depende de $d$ a través de un sistema lineal de equilibrio, y el método numérico combina dos herramientas: se resuelve el sistema de tres ecuaciones con eliminación gaussiana, y la raíz de $g(d) = f(d) - \\mu_s N_{suelo}$ se busca con el método de bisección. Se usa bisección porque solo exige poder evaluar la condición de equilibrio, no tener su fórmula.",
            "La bisección parte de un intervalo $[d_{lo}, d_{hi}]$ donde $g$ cambia de signo, evalúa el punto medio $d_m = (d_{lo} + d_{hi})/2$ y descarta la mitad que no contiene la raíz. Tras $k$ iteraciones la raíz está atrapada en un intervalo de longitud $L/2^k$, de modo que el error absoluto cumple $|d_k - d^*| \\le L/2^{k+1}$. El método converge linealmente (orden 1, factor $1/2$) pero es robusto: siempre converge si $g$ es continua y el signo cambia, y no necesita derivadas.",
            "Sus limitaciones son que solo encuentra una raíz por intervalo, que requiere un cambio de signo bien definido y que no puede mejorar más allá de la precisión con que se evalúa $g$: cuando $L/2^k$ es comparable al redondeo de $d$ (unos $10^{-15}$ m aquí) más iteraciones no ayudan. El sistema lineal $A\\vec x = \\vec b$ se resuelve con pivoteo parcial; su error depende del número de condición $\\kappa(A)$ y está acotado por $\\kappa\\,\\varepsilon_{mac}$, con $\\varepsilon_{mac} \\approx 2\\times10^{-16}$.",
            "Se escogieron 60 iteraciones para que $L/2^{60} \\approx 4\\times10^{-18}$ m quede por debajo del redondeo, sin riesgo, aunque en la práctica bastan unas 50. El resultado $d_{max} \\approx 3.67$ m dice que, con $\\mu_s = 0.40$ y $60^\\circ$, la persona puede subir hasta cerca del 73% de la escalera. Es independiente del análisis porque la fórmula cerrada $d_{max} = (\\mu_s(W_e + W_h)L\\tan\\theta - W_eL/2)/W_h$ nunca se usa: el código solo evalúa si la fricción necesaria cabe bajo el límite.",
        ],
        numeric_code=f"""# Biseccion sobre d con equilibrio resuelto por un sistema lineal
L, We, Wh, theta, mu = {c14.L_LAD}, {c14.W_LAD}, {c14.W_MAN}, {c14.THETA_DEG}, {c14.MU}   # datos
def fuerzas(d):                                   # resuelve A x = b con x = (N_suelo, f, N_pared)
    A = [[0, 1, -1],                              # sum Fx:  f - N_pared = 0
         [1, 0,  0],                              # sum Fy:  N_suelo = We + Wh
         [0, 0, L * sin(theta)]]                  # torque respecto al pie
    b = [0, We + Wh, (We * L/2 + Wh * d) * cos(theta)]   # pesos con brazo de palanca
    return solve(A, b)                            # eliminacion gaussiana con pivoteo
lo, hi = 0.0, L                                   # intervalo inicial [0, L]
for k in range(60):                               # 60 iteraciones: L/2**60 ~ 4e-18 m
    mid = (lo + hi) / 2                           # punto medio
    N_suelo, f, N_pared = fuerzas(mid)
    if f <= mu * N_suelo:                         # aguanta: la raiz esta mas arriba
        lo = mid                                  # se descarta la mitad inferior
    else:                                         # resbala: la raiz esta mas abajo
        hi = mid                                  # se descarta la mitad superior
d_max = (lo + hi) / 2                             # estimacion final de la raiz""",
        numeric_converge=[
            ("Error contra iteraciones",
             "Con una bisección truncada en $k$ iteraciones se mide el error absoluto contra la solución analítica. Cada iteración añade un bit de precisión: el error queda siempre por debajo de la cota $L/2^{k+1}$ (a veces mucho menor, si el punto medio cae casualmente cerca de la raíz, como en $k=5$ y $k=10$) hasta que llega al redondeo, cerca de $k \\approx 50$.",
             [tab, rf"d_{{max}}^{{num}} (60\ \mathrm{{it}}) = {dn:.12f}\ \mathrm{{m}},\qquad d_{{max}}^{{an}} = {dmax:.12f}\ \mathrm{{m}}"]),
            ("Condicionamiento del sistema lineal",
             "El número de condición $\\kappa(A)$ mide cuánto amplifica el sistema los errores de los datos. Su variación viene del elemento $L\\sin\\theta$ de la tercera fila y es modesta: de $\\theta=20^\\circ$ a $89^\\circ$ solo se duplica, así que el sistema está bien condicionado en todo el rango útil y se pierde menos de una cifra decimal. El residuo $\\|A\\vec x - \\vec b\\|$ es cero en doble precisión, es decir, la solución cumple las ecuaciones hasta el último bit.",
             [rf"\kappa(A)\big|_{{20^\circ}} = {conds[20.0]:.3f},\qquad \kappa(A)\big|_{{60^\circ}} = {conds[60.0]:.3f},\qquad \kappa(A)\big|_{{89^\circ}} = {conds[89.0]:.3f}",
              rf"\|A\vec x - \vec b\|\big|_{{d=2.5\ \mathrm{{m}}}} = {sci(res, 2)}\ \mathrm{{N}}"]),
            ("Fuentes de error y criterio de confianza",
             f"No hay error de truncamiento en el sistema lineal (es exacto salvo redondeo); el único error de método es el del intervalo de la bisección. La sensibilidad física es mayor que el error numérico: un 1% de error en $\\mu_s$ mueve $d_{{max}}$ en ${sci(dd, 2)}$ m, muchísimo más que cualquier error del algoritmo. El resultado es confiable si $g$ cambia de signo en los extremos, el residuo es pequeño y el intervalo final es menor que la incertidumbre de los datos.",
             [rf"\Delta d_{{max}}\big|_{{\mu_s \to 1.01\mu_s}} = {sci(dd, 3)}\ \mathrm{{m}}\ \gg\ |d^{{num}}_{{max}} - d^{{an}}_{{max}}| = {sci(abs(dn - dmax), 2)}\ \mathrm{{m}}"]),
        ],
    )


# ============================================================================ BeamReactions
def _beam() -> dict:
    x = 2.0
    ra, rb = c14.beam_reactions_analytic(x)
    pivots = [0.0, 1.5, 3.0, 4.5, 6.0]
    rows = []
    maxerr = 0.0
    for xp in pivots:
        a = np.array([[1.0, 1.0], [-xp, c14.BEAM_L - xp]])
        b = np.array([c14.BEAM_W + c14.PERSON_W, c14.BEAM_W * (c14.BEAM_L / 2 - xp) + c14.PERSON_W * (x - xp)])
        sol = np.array(c14.beam_reactions_numeric(x, xp))
        res = float(np.linalg.norm(a @ sol - b))
        err = max(abs(sol[0] - ra), abs(sol[1] - rb))
        maxerr = max(maxerr, err)
        rows.append(f"{xp:.1f} & {np.linalg.cond(a):.3f} & {sci(res, 2)} & {sci(err, 2)}")
    tab = _array(r"x_p\,(\mathrm{m}) & \kappa(A) & \|A\vec x - \vec b\|\,(\mathrm{N}) & \max|R^{num}-R^{an}|\,(\mathrm{N})", rows, "r|rrr")
    # perturbacion de la posicion de la persona
    r0 = np.array(c14.beam_reactions_numeric(x, 0.0))
    r1 = np.array(c14.beam_reactions_numeric(x * (1 + 1e-3), 0.0))
    rel_in = 1e-3
    rel_out = float(np.linalg.norm(r1 - r0) / np.linalg.norm(r0))
    amp = rel_out / rel_in
    return dict(
        numeric_intro=[
            "Una viga apoyada en dos puntos tiene dos reacciones desconocidas, y el equilibrio da dos ecuaciones lineales: suma de fuerzas verticales nula y suma de torques nula respecto de un punto. Aquí el 'método numérico' es la resolución de un sistema lineal $A\\vec x = \\vec b$ con eliminación gaussiana (numpy.linalg.solve) y no una iteración: el resultado, en aritmética exacta, es la solución buscada. La parte interesante es estudiar cuánto se equivoca en aritmética finita, y comprobar que el pivote elegido para los torques no cambia nada.",
            "El sistema es $\\begin{pmatrix}1&1\\\\-x_p&L-x_p\\end{pmatrix}\\begin{pmatrix}R_A\\\\R_B\\end{pmatrix} = \\begin{pmatrix}W_b + W_p\\\\ W_b(L/2 - x_p) + W_p(x - x_p)\\end{pmatrix}$, donde $x_p$ es la posición del pivote de los torques. La eliminación gaussiana con pivoteo parcial factoriza $A = PLU$ y resuelve dos sistemas triangulares; para una matriz de $2\\times2$ son unas pocas operaciones.",
            "No existe un orden de convergencia porque no hay iteración, pero sí un error: el redondeo satisface $\\|\\delta\\vec x\\|/\\|\\vec x\\| \\lesssim \\kappa(A)\\,\\varepsilon_{mac}$, donde $\\kappa(A) = \\|A\\|\\,\\|A^{-1}\\|$ es el número de condición y $\\varepsilon_{mac} \\approx 2.2\\times10^{-16}$. El método falla cuando $A$ es casi singular, y aquí eso ocurriría si los dos apoyos coincidieran ($L \\to 0$), situación en que las reacciones no se pueden separar. Con pivotes dentro de la viga $\\kappa$ es de orden 1 a 10, así que se pierde menos de una cifra decimal.",
            "Se eligieron cinco pivotes (cada apoyo, el centro y dos puntos intermedios) para poner a prueba una propiedad física profunda: el equilibrio implica torque nulo respecto de cualquier punto, así que las cinco soluciones deben coincidir hasta el redondeo. Es un método independiente del analítico porque no usa la fórmula de palanca $R_A = W_b/2 + W_p(L-x)/L$ ni la simetría; el equilibrio se plantea desde las leyes generales y la solución sale del álgebra.",
        ],
        numeric_code=f"""# Equilibrio de la viga como sistema lineal 2x2, con pivote arbitrario x_p
L, Wb, Wp, x = {c14.BEAM_L}, {c14.BEAM_W}, {c14.PERSON_W}, 2.0   # largo, peso viga, peso persona, posicion
def reacciones(x_p):
    A = [[1, 1],                                  # sum Fy:  R_A + R_B = Wb + Wp
         [-x_p, L - x_p]]                         # sum torques respecto a x_p (brazos de palanca)
    b = [Wb + Wp,                                 # peso total
         Wb * (L/2 - x_p) + Wp * (x - x_p)]       # torque de los pesos respecto a x_p
    return solve(A, b)                            # LU con pivoteo parcial
for x_p in [0.0, 3.0, 6.0]:                       # tres pivotes distintos
    R_A, R_B = reacciones(x_p)
    residuo = norm(A @ [R_A, R_B] - b)            # verificacion: A x - b ~ 0
    cond = cond_number(A)                         # numero de condicion de la matriz
    print(x_p, R_A, R_B, residuo, cond)           # las reacciones deben ser identicas
# criterio: max(R_A) - min(R_A) sobre los pivotes < 1e-9 N""",
        numeric_converge=[
            ("Pivote, condicionamiento y residuo",
             f"Para cinco pivotes se calcula el número de condición, el residuo y el error contra la solución analítica con la persona en $x = {x}$ m. El condicionamiento va de 3 a 6 según el pivote; con $\\kappa\\,\\varepsilon_{{mac}} \\approx 10^{{-15}}$ el error teórico es mucho mayor que el observado, que queda en el nivel del último bit del flotante.",
             [tab, rf"R_A^{{an}} = {ra:.6f}\ \mathrm{{N}},\qquad R_B^{{an}} = {rb:.6f}\ \mathrm{{N}},\qquad \max_{{x_p}}\text{{error}} = {sci(maxerr, 2)}\ \mathrm{{N}}"]),
            ("Sensibilidad a los datos",
             "Una cosa es el error del algoritmo y otra la sensibilidad del problema. Si la posición de la persona cambia en un $0.1\\%$ (medida imprecisa con la cinta), las reacciones cambian en una fracción mayor o menor según el cociente $\\Delta R/R$ sobre $\\Delta x/x$; este amplificador es el que importa en la práctica porque es mucho mayor que el redondeo.",
             [rf"\dfrac{{\Delta x}}{{x}} = 10^{{-3}} \Rightarrow \dfrac{{\|\Delta\vec R\|}}{{\|\vec R\|}} = {sci(rel_out, 2)},\qquad \text{{amplificación}} = {amp:.3f}"]),
            ("Fuentes de error y criterio práctico",
             "No hay truncamiento ni estadística: la única fuente es el redondeo, que aquí es $\\lesssim 10^{-12}$ N sobre reacciones del orden de $10^3$ N. El resultado es confiable cuando el residuo es del orden de $\\varepsilon_{mac}\\|\\vec b\\|$, el número de condición es moderado y los distintos pivotes coinciden: esta última prueba es una verificación cruzada que detecta errores de planteamiento (un signo mal puesto rompería la independencia del pivote).",
             [rf"\max_{{x_p}}|R_A(x_p) - R_A(0)| \le {sci(max(abs(c14.beam_reactions_numeric(x, xp)[0] - c14.beam_reactions_numeric(x, 0.0)[0]) for xp in pivots), 2)}\ \mathrm{{N}}"]),
        ],
    )


# ============================================================================ KeplerOrbit
def _kepler_period(n_per: int) -> float:
    """Replica de c15.period_numeric con n_per pasos por orbita."""
    t, y = c15.orbit_numeric(c15.A_ORBIT, c15.E_ORBIT, 1.3, n_per)
    r = np.hypot(y[:, 0], y[:, 1])
    i = [j for j in range(1, len(r) - 1) if r[j] < r[j - 1] and r[j] <= r[j + 1]][0]
    y0, y1, y2 = r[i - 1], r[i], r[i + 1]
    return float(t[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (t[1] - t[0]))


def _kepler() -> dict:
    T = c15.period_analytic()
    KERRV = abs(c15.period_numeric() - T)
    ns = [250, 500, 1000, 2000, 4000]
    hs = [T / n for n in ns]
    de, close, dT = [], [], []
    for n in ns:
        t, y = c15.orbit_numeric(revolutions=1.0, n_per_orbit=n)
        e = c15.energy(y)
        de.append(float(np.ptp(e) / abs(e[0])))
        close.append(float(np.hypot(*(y[-1, :2] - y[0, :2]))))
        dT.append(abs(_kepler_period(n) - T))
    rows = [f"{n} & {sci(hs[i], 3)} & {sci(de[i], 3)} & {sci(close[i], 3)} & {sci(dT[i], 3)}" for i, n in enumerate(ns)]
    tab = _array(r"N/\mathrm{orb} & h\,(\mathrm{a\tilde{n}o}) & \Delta E/|E| & |\vec r_N - \vec r_0|\,(\mathrm{UA}) & |T^{num} - T^{an}|\,(\mathrm{a\tilde{n}o})", rows, "r|rrrr")
    pe, pc, pt = _slope(hs, de), _slope(hs, close), _slope(hs, dT)
    # Euler con 8000 pasos por orbita
    rp = c15.A_ORBIT * (1 - c15.E_ORBIT)
    y0 = np.array([rp, 0.0, 0.0, c15.perihelion_speed()])
    rhs = lambda _t, s: np.array([s[2], s[3], -c15.GM * s[0] / np.hypot(s[0], s[1]) ** 3, -c15.GM * s[1] / np.hypot(s[0], s[1]) ** 3])  # noqa: E731
    _, ye = integrate(rhs, y0, 0.0, T, 8000, euler_step)
    ee = c15.energy(ye)
    de_euler = float(np.ptp(ee) / abs(ee[0]))
    _, y8 = c15.orbit_numeric(revolutions=1.0)
    e8 = c15.energy(y8)
    a8 = c15.areal_velocity(None, y8)
    return dict(
        numeric_intro=[
            "El problema de dos cuerpos con fuerza $-GM\\vec r/r^3$ tiene solución cerrada (las cónicas de Kepler), pero el periodo y las leyes de Kepler se pueden comprobar sin usarla: basta integrar las ecuaciones de movimiento. Se escribe el sistema de primer orden $\\dot{\\vec r} = \\vec v$, $\\dot{\\vec v} = -GM\\vec r/r^3$ para el estado $(x,y,v_x,v_y)$ y se avanza con RK4. Con el estado en mano se miden la velocidad areolar, la energía y el periodo, que son lo que las leyes de Kepler predicen.",
            "RK4 usa $\\vec s_{n+1} = \\vec s_n + \\tfrac h6(\\vec k_1 + 2\\vec k_2 + 2\\vec k_3 + \\vec k_4)$, con $\\vec k_i = \\vec f(\\vec s)$ evaluada en cuatro estados intermedios, y tiene error global $O(h^4)$. El periodo se obtiene detectando el segundo mínimo de $r(t)$ (paso por el perihelio) y refinando su posición con una parábola por tres puntos, $\\delta = \\tfrac12\\,(r_{-}-r_{+})/(r_{-} - 2r_0 + r_{+})$, lo que reduce el error del instante de $O(h)$ a $O(h^2)$.",
            "La limitación importante es que RK4 no es simpléctico: no conserva la energía ni el momento angular, solo los mantiene con una deriva lenta del orden $h^4$ por órbita, que se acumula con muchas vueltas. En una órbita con $e = 0.5$ el paso cerca del perihelio es crítico, porque allí la aceleración cambia rápido; por eso un paso fijo calibrado para el afelio resulta insuficiente. Para integraciones de miles de órbitas se prefieren métodos simplécticos (leapfrog, Yoshida) o con paso adaptativo.",
            f"Se escogieron 8000 pasos por órbita para las gráficas de energía y áreas, y 20000 para medir el periodo, porque con 20000 pasos el error del periodo cae a unos ${sci(KERRV, 1)}$ años. El resultado $T \\approx 2.828$ años confirma la tercera ley, $T = a^{{3/2}}$. El método es independiente de la solución de Kepler porque no usa la elipse, ni el semieje en la ecuación de movimiento: $a$ y $e$ solo fijan la condición inicial en el perihelio, y todo lo demás (forma elíptica, periodo, constancia de $dA/dt$) debe emerger de la integración.",
        ],
        numeric_code=f"""# RK4 para la orbita de Kepler en unidades UA, anio (GM = 4 pi^2)
GM, a, e = 4*pi**2, {c15.A_ORBIT}, {c15.E_ORBIT}                      # parametro gravitacional y orbita
r_p = a * (1 - e)                                 # distancia en el perihelio
v_p = sqrt(GM * (1 + e) / (a * (1 - e)))          # velocidad en el perihelio (vis-viva)
def f(s):                                         # s = (x, y, vx, vy)
    r = hypot(s[0], s[1])
    return array([s[2], s[3], -GM*s[0]/r**3, -GM*s[1]/r**3])   # (v, a)
s = array([r_p, 0, 0, v_p])                       # partida en el perihelio
N_orb = 8000                                      # pasos por orbita
h = a**1.5 / N_orb                                # h = T/N con T = a^1.5 anios
for n in range(int(1.2 * N_orb)):                 # 1.2 orbitas
    k1 = f(s)                                     # pendiente en el inicio
    k2 = f(s + h/2 * k1)                          # pendiente a mitad de paso
    k3 = f(s + h/2 * k2)                          # segunda a mitad de paso
    k4 = f(s + h * k3)                            # pendiente al final
    s = s + h/6 * (k1 + 2*k2 + 2*k3 + k4)         # estado siguiente
# periodo: segundo minimo de r(t), refinado con parabola por tres puntos
delta = 0.5 * (r[i-1] - r[i+1]) / (r[i-1] - 2*r[i] + r[i+1])
T_num = t[i] + delta * h                          # instante del perihelio""",
        numeric_converge=[
            ("Error contra pasos por órbita",
             "Se integra una órbita con $N = 250, \\dots, 4000$ pasos y se mide la deriva relativa de la energía, el cierre de la órbita (distancia entre la posición final y la inicial) y el error del periodo medido. Cada duplicación de $N$ baja los dos primeros errores en un factor cercano a 16; el error del periodo, medido con la parábola por tres puntos, sigue la misma ley.",
             [tab, rf"p_E = {pe:.2f},\quad p_{{cierre}} = {pc:.2f},\quad p_T = {pt:.2f}\qquad (\text{{teórico RK4: }}4)"]),
            ("Fuentes de error y comparación con Euler",
             "El error de truncamiento domina siempre que $h$ no sea muy pequeño; el redondeo solo aparece cerca de $10^{-13}$ y no es relevante aquí. Un integrador de orden bajo falla de modo cualitativo: con Euler, para el mismo número de pasos, la energía no se conserva ni de lejos y la órbita se ensancha en cada vuelta.",
             [rf"\dfrac{{\Delta E}}{{|E|}}\Big|_{{Euler,\,8000}} = {sci(de_euler, 2)}\qquad \text{{frente a}}\qquad \dfrac{{\Delta E}}{{|E|}}\Big|_{{RK4,\,8000}} = {sci(float(np.ptp(e8) / abs(e8[0])), 2)}"]),
            ("Criterio práctico de confiabilidad",
             f"Se acepta la integración cuando las constantes de movimiento permanecen constantes dentro de la tolerancia y el resultado deja de cambiar al duplicar $N$. Con 8000 pasos la velocidad areolar y la energía varían ${sci(float(np.ptp(a8) / abs(a8[0])), 2)}$ y ${sci(float(np.ptp(e8) / abs(e8[0])), 2)}$ respectivamente, y el periodo medido con 20000 pasos coincide con $a^{{3/2}}$ en ${sci(abs(c15.period_numeric() - T), 2)}$ años.",
             [rf"T^{{num}} = {c15.period_numeric():.9f}\ \text{{años}},\qquad T^{{an}} = a^{{3/2}} = {T:.9f}\ \text{{años}}"]),
        ],
    )


# ============================================================================ EscapeVelocity
def _escape_hist(cut: float, iters: int):
    """Biseccion de c15.escape_speed_numeric con radio de corte configurable; devuelve la historia de puntos medios."""
    mu, radius = c15.G_N * c15.M_EARTH, c15.R_EARTH
    lo, hi = 0.5 * np.sqrt(mu / radius), 2.0 * np.sqrt(mu / radius)
    hist = []
    f = lambda _t, y: np.array([y[1], -mu / y[0] ** 2])  # noqa: E731
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = np.array([radius, mid])
        dt = 2.0
        escaped = True
        for _ in range(400000):
            s = rk4_step(f, 0.0, s, dt)
            if s[1] <= 0:
                escaped = False
                break
            if s[0] > cut * radius:
                break
            dt = min(1e5, max(2.0, 0.002 * s[0] / max(abs(s[1]), 1.0)))
        if escaped:
            hi = mid
        else:
            lo = mid
        hist.append(mid)
    return hist


def _escape() -> dict:
    va = c15.escape_speed_analytic()
    hist = _escape_hist(1e4, 45)
    ks = [5, 10, 15, 20, 30, 45]
    rows = []
    for k in ks:
        rows.append(f"{k} & {sci(1.5 * np.sqrt(c15.G_N * c15.M_EARTH / c15.R_EARTH) / 2**k, 3)} & {hist[k - 1]:.4f} & {sci(abs(hist[k - 1] - va), 3)}")
    tab = _array(r"k & \text{ancho}\,(\mathrm{m/s}) & v_k\,(\mathrm{m/s}) & |v_k - v_{esc}^{an}|\,(\mathrm{m/s})", rows, "r|rrr")
    cuts = [1e2, 1e3, 1e4]
    cv = []
    for c in cuts:
        cv.append(_escape_hist(c, 32)[-1] if c != 1e4 else hist[31])
    crow = [f"{int(c):d}\\,R & {cv[i]:.3f} & {sci(cv[i] - va, 3)} & {sci(-va * 0.5 / c, 3)}" for i, c in enumerate(cuts)]
    ctab = _array(r"r_{corte} & v^{num}\,(\mathrm{m/s}) & v^{num}-v^{an}\,(\mathrm{m/s}) & -v^{an}/(2\,r_{corte}/R)\,(\mathrm{m/s})", crow, "r|rrr")
    # r_max contra N
    v0 = 8000.0
    rm = c15.rmax_analytic(v0)
    nn = [2500, 5000, 10000, 20000, 40000]
    erm = [abs(c15.rmax_numeric(v0, n=n) - rm) / rm for n in nn]
    hh = [4000.0 / n for n in nn]
    prm = _slope(hh, erm)
    rrow = [f"{n} & {hh[i]:.2f} & {sci(erm[i], 3)}" for i, n in enumerate(nn)]
    rtab = _array(r"N & h\,(\mathrm{s}) & |r_{max}^{num} - r_{max}^{an}|/r_{max}^{an}", rrow, "r|rr")
    return dict(
        numeric_intro=[
            "La velocidad de escape se define como la velocidad inicial mínima para que un cuerpo alcance el infinito, y analíticamente sale de igualar la energía a cero. Numéricamente no se puede esperar al infinito, de modo que se lanza el cuerpo con una velocidad de prueba $v_0$ y se integra $\\ddot r = -GM/r^2$ hasta que ocurra una de dos cosas: la velocidad se anula (el cuerpo cae, no escapa) o el cuerpo llega a un radio de corte enorme ($10^4R$, escapa). Esa respuesta binaria es una función de $v_0$ que cambia en el umbral buscado.",
            "El umbral se encuentra con el método de bisección: se parte de un intervalo $[v_{lo}, v_{hi}] = [\\tfrac12\\sqrt{GM/R},\\ 2\\sqrt{GM/R}]$ con el cambio de comportamiento dentro, se prueba el punto medio y se descarta una mitad. La integración interna usa RK4, $y_{n+1} = y_n + \\tfrac h6(k_1 + 2k_2 + 2k_3 + k_4)$, con paso adaptativo $h = \\max(2\\ \\mathrm{s},\\ 0.002\\,r/|v|)$, que mantiene unos 500 pasos por 'tiempo característico' $r/|v|$ y deja que el paso crezca al alejarse. Después de $k$ iteraciones el ancho del intervalo es $\\approx 1.5\\sqrt{GM/R}/2^k$.",
            "Hay dos fuentes de error independientes. La bisección converge linealmente (error $\\sim 2^{-k}$), mientras que RK4 tiene error $O(h^4)$ en cada integración. Pero el criterio de corte introduce un sesgo sistemático: un cuerpo que llegue a $r_c$ con velocidad aún positiva ya escapó 'por muy poco', y con energía cero a $r_c$ le basta $v^2 = 2GM(1/R - 1/r_c)$, por lo que la velocidad de escape numérica sale subestimada en el factor $1 - R/(2r_c)$. El método falla si el criterio de corte es demasiado cercano (sesgo grande) o si el paso es excesivo cerca del apogeo.",
            "Se tomaron 45 iteraciones, de modo que el ancho final ($\\sim 10^{-10}$ m/s) está muy por debajo del sesgo del corte ($\\sim 0.5$ m/s) y de la incertidumbre de $G$ y $M$ (unas cuatro cifras). El resultado, unos 11.19 km/s, es la velocidad para dejar la superficie terrestre sin considerar el aire ni la rotación. El método es independiente del analítico porque nunca usa la conservación de la energía en forma de $v_{esc} = \\sqrt{2GM/R}$: solo integra la fuerza y clasifica la trayectoria final.",
        ],
        numeric_code=f"""# Biseccion sobre v0 + RK4 con paso adaptativo (movimiento radial)
mu, R = G * M, {c15.R_EARTH:.4g}                              # GM de la Tierra y su radio
f = lambda s: array([s[1], -mu / s[0]**2])        # s = (r, v):  r' = v,  v' = -mu/r^2
def escapa(v0):
    s, dt = array([R, v0]), 2.0                   # parte de la superficie, paso inicial 2 s
    while True:
        k1 = f(s); k2 = f(s + dt/2*k1)            # pendientes RK4 (1 y 2)
        k3 = f(s + dt/2*k2); k4 = f(s + dt*k3)    # pendientes RK4 (3 y 4)
        s = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)    # avance del paso
        if s[1] <= 0: return False                # se detuvo: vuelve a caer
        if s[0] > 1e4 * R: return True            # llego muy lejos: escapa
        dt = min(1e5, max(2.0, 0.002 * s[0] / max(abs(s[1]), 1.0)))   # paso crece con r/|v|
lo, hi = 0.5 * sqrt(mu / R), 2.0 * sqrt(mu / R)  # intervalo inicial de velocidades
for k in range(45):                               # 45 iteraciones
    mid = (lo + hi) / 2
    if escapa(mid): hi = mid                      # si escapa, bajar el extremo superior
    else:           lo = mid                      # si cae, subir el inferior
v_esc = (lo + hi) / 2                             # estimacion final""",
        numeric_converge=[
            ("Error contra iteraciones de la bisección",
             f"La tabla sigue el punto medio $v_k$ de la bisección completa (cada fila requiere una integración RK4). El ancho del intervalo baja a la mitad en cada paso, pero el error contra $v_{{esc}}^{{an}} = {va:.3f}$ m/s deja de bajar cuando se alcanza el sesgo del criterio de corte, cerca de $k \\approx 15$.",
             [tab]),
            ("Sesgo del radio de corte",
             "El error residual no es de la bisección sino del criterio 'llegó a $r_c$'. Se repite el cálculo con tres radios de corte (32 iteraciones cada uno): el error tiene el signo y el tamaño esperados, $-v_{esc}/(2\\,r_c/R)$, y baja con el corte. Esta es una fuente de error de discretización del problema, no del integrador.",
             [ctab]),
            ("Error de RK4 en $r_{max}$ y criterio práctico",
             f"Para $v_0 = 8$ km/s se integra con $N$ pasos fijos hasta 4000 s y se compara el máximo de la trayectoria con la fórmula analítica; el error relativo queda en el orden de $10^{{-9}}$ y baja hasta $10^{{-11}}$ con el paso más fino, pero no sigue una ley de potencia limpia: el máximo se toma entre puntos de la malla, y el error de muestreo (cuánto cae el apogeo entre dos nodos) se superpone al del integrador. El resultado es confiable cuando el valor no cambia al duplicar las iteraciones y mover el radio de corte un factor 10 solo lo modifica en el nivel esperado.",
             [rtab]),
        ],
    )


# ============================================================================ PressureDepth
def _pressure() -> dict:
    kk = c16.M_AIR * c16.G / (c16.R_GAS * c16.T_AIR)
    pa = float(c16.atmosphere_analytic(10000.0))
    f = lambda _h, y: np.array([-y[0] * kk])  # noqa: E731

    def p_end(n, step=rk4_step):
        _, y = integrate(f, np.array([c16.P0]), 0.0, 10000.0, n, step)
        return float(y[-1, 0])

    ns = [5, 10, 20, 40, 80]
    hs = [10000.0 / n for n in ns]
    er = [abs(p_end(n) - pa) / pa for n in ns]
    ee = [abs(p_end(n, euler_step) - pa) / pa for n in ns]
    rows = [f"{n} & {hs[i]:.0f} & {sci(kk * hs[i], 3)} & {sci(ee[i], 3)} & {sci(er[i], 3)}" for i, n in enumerate(ns)]
    tab = _array(r"N & \Delta h\,(\mathrm{m}) & \kappa\Delta h & \text{Euler} & \text{RK4}", rows, "r|rrrr")
    pr, pe = _slope(hs, er), _slope(hs, ee)
    e200 = abs(float(c16.atmosphere_numeric()[1][-1]) - pa) / pa
    p5 = float(c16.atmosphere_analytic(5000.0))
    pc5 = float(c16.atmosphere_constant_density(5000.0))
    return dict(
        numeric_intro=[
            "En el agua la densidad es casi constante y la presión crece linealmente con la profundidad, pero en la atmósfera la densidad cae con la altura y la ecuación $dp/dh = -\\rho(p)\\,g$ deja de ser una cuadratura sencilla. Con la ley del gas ideal isotermo, $\\rho = pM/(RT)$, la ecuación es $dp/dh = -\\kappa p$, donde $\\kappa = Mg/(RT)$. Se integra numéricamente porque el mismo código serviría con un perfil de temperatura $T(h)$ más realista, para el cual no hay fórmula cerrada.",
            "El método es RK4: para $y' = f(h,y)$ con paso $\\Delta h$, $y_{n+1} = y_n + \\tfrac{\\Delta h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$. Como $f$ es lineal, cada paso equivale a multiplicar por el polinomio de Taylor de $e^{-z}$ truncado en cuarto orden, $1 - z + z^2/2 - z^3/6 + z^4/24$ con $z = \\kappa\\Delta h$. El error local por paso es entonces el primer término omitido, $z^5/120$, y el error global es $O(\\Delta h^4)$.",
            "Para que RK4 sea preciso solo se necesita que $z = \\kappa\\,\\Delta h$ sea pequeño: la ecuación no es rígida y no hay inestabilidad (la estabilidad exige $|z| \\lesssim 2.78$, y aquí $z \\approx 0.006$). La limitación es otra: el resultado solo es tan realista como el modelo $\\rho(p)$, y ningún refinamiento de $\\Delta h$ corrige un modelo que ignora el gradiente de temperatura (que en la atmósfera real baja unos $6.5$ K/km).",
            "Se eligieron 200 pasos de 50 m: el error relativo de truncamiento resulta de orden $10^{-11}$, muy por debajo de cualquier incertidumbre física, así que el cálculo cuesta casi nada. El resultado dice que a 10 km la presión baja a unos 30.9 kPa. El método es independiente del analítico porque nunca evalúa $p_0 e^{-\\kappa h}$: solo calcula la derivada local en puntos intermedios y la acumula.",
        ],
        numeric_code=f"""# RK4 para la presion atmosferica  dp/dh = -rho(p) g,  rho = p M / (R T)
p0, M, R, T, g = {c16.P0:.0f}, {c16.M_AIR}, {c16.R_GAS}, {c16.T_AIR}, {c16.G}   # datos (SI)
kappa = M * g / (R * T)                           # constante: dp/dh = -kappa p
f = lambda h, p: -kappa * p                       # lado derecho (gas ideal isotermo)
N, h_end = 200, 10000.0
dh = h_end / N                                    # paso de 50 m
p = [p0]                                          # presion a nivel del mar
for n in range(N):
    h = n * dh
    k1 = f(h, p[n])                               # pendiente en el inicio
    k2 = f(h + dh/2, p[n] + dh/2 * k1)            # pendiente a mitad de paso
    k3 = f(h + dh/2, p[n] + dh/2 * k2)            # segunda estimacion a mitad
    k4 = f(h + dh, p[n] + dh * k3)                # pendiente al final
    p.append(p[n] + dh/6 * (k1 + 2*k2 + 2*k3 + k4))   # promedio 1-2-2-1
p_5km = interp(5000.0, alturas, p)                # lectura en un punto de la malla""",
        numeric_converge=[
            ("Error contra el paso, RK4 frente a Euler",
             "Se integra de 0 a 10 km con $N = 5, \\dots, 80$ y se compara con la exponencial. Para Euler el error baja a la mitad cuando $N$ se duplica (orden 1); para RK4 baja por 16 (orden 4). Con $N = 5$ ($z \\approx 0.24$) RK4 ya es mejor que Euler con 80.",
             [tab, rf"p_{{Euler}} = {pe:.2f},\qquad p_{{RK4}} = {pr:.2f}\qquad (\text{{teóricos: }}1\ \text{{y}}\ 4)"]),
            ("Truncamiento, redondeo y modelo",
             f"Con $N = 200$ (el valor usado por el código) el error relativo es ${sci(e200, 2)}$, todavía de truncamiento (el redondeo acumulado sería del orden de $10^{{-15}}$). Pero el error numérico es irrelevante frente al error del modelo: si se supusiera densidad constante, la presión a 5 km saldría {pc5 / 1000:.3f} kPa en vez de {p5 / 1000:.3f} kPa.",
             [rf"\dfrac{{p_{{\rho=cte}}(5\ \mathrm{{km}}) - p_{{an}}(5\ \mathrm{{km}})}}{{p_{{an}}}} = {sci((pc5 - p5) / p5, 2)}\ \ \gg\ \ \dfrac{{|p_{{num}} - p_{{an}}|}}{{p_{{an}}}} = {sci(e200, 2)}"]),
            ("Criterio práctico de confiabilidad",
             "Basta verificar que $\\kappa\\Delta h \\ll 1$ y que el resultado a 10 km no cambia al duplicar $N$. Una prueba adicional es que la presión disminuye de forma monótona y positiva. Si se cambiara a un perfil $T(h)$ sin solución cerrada, se compararía $N$ con $2N$ y se aceptaría cuando la diferencia relativa caiga por debajo de la tolerancia.",
             [rf"\dfrac{{|p_{{N=100}} - p_{{N=200}}|}}{{p_{{N=200}}}} = {sci(abs(p_end(100) - p_end(200)) / p_end(200), 2)}"]),
        ],
    )


# ============================================================================ FloatingCylinder
def _period_n(n: int, y0: float = 0.02) -> float:
    """Replica de c16.period_numeric con n pasos en 2 s."""
    t, s = c16.bobbing_numeric(0.0, y0, 2.0, n)
    d = s[:, 0] - c16.submerged_depth_analytic()
    cr = [i for i in range(1, len(d)) if d[i - 1] > 0 >= d[i]]
    ts = [t[i] - d[i] * (t[i] - t[i - 1]) / (d[i] - d[i - 1]) for i in cr[:2]]
    return float(ts[1] - ts[0])


def _bis_h(k: int) -> float:
    lo, hi = 0.0, c16.LEN
    for _ in range(k):
        mid = 0.5 * (lo + hi)
        if c16.buoyancy(mid) < c16.MASS * c16.G:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cylinder() -> dict:
    ta = 2 * np.pi / c16.omega_analytic()
    h_a = c16.submerged_depth_analytic()
    ns = [100, 200, 400, 800, 1600]
    hs = [2.0 / n for n in ns]
    ep = [abs(_period_n(n) - ta) for n in ns]
    rows = [f"{n} & {sci(hs[i], 3)} & {sci(ep[i], 3)}" for i, n in enumerate(ns)]
    tab = _array(r"N & h\,(\mathrm{s}) & |T^{num}-T^{an}|\,(\mathrm{s})", rows, "r|rr")
    pp = _slope(hs, ep)
    e40000 = abs(c16.period_numeric() - ta)
    # error del propio RK4 en y(t) sin interpolacion
    def y_err(n):
        t, s = c16.bobbing_numeric(0.0, 0.02, 2.0, n)
        ya = h_a + 0.02 * np.cos(c16.omega_analytic() * t)
        return float(np.max(np.abs(s[:, 0] - ya)))
    ey = [y_err(n) for n in ns]
    py = _slope(hs, ey)
    ks = [5, 10, 20, 40, 52]
    kr = [f"{k} & {sci(c16.LEN / 2**k, 3)} & {sci(abs(_bis_h(k) - h_a), 3)}" for k in ks]
    ktab = _array(r"k & L/2^k\,(\mathrm{m}) & |h_k - h_{eq}|\,(\mathrm{m})", kr, "r|rr")
    return dict(
        numeric_intro=[
            "Esta escena usa dos métodos numéricos, cada uno con un propósito. Para la profundidad de equilibrio se busca la raíz de $B(h) - mg = 0$ con el método de bisección, y para las oscilaciones verticales se integra $m\\ddot y = mg - \\rho_f g A\\,y$ con RK4 y se mide el periodo de la señal resultante. Ambos se usan aquí aunque la solución analítica es sencilla, porque son los métodos que sirven cuando la fuerza de flotación no es lineal (cuerpos de sección variable, fluidos estratificados).",
            "La bisección mantiene un intervalo $[h_{lo}, h_{hi}]$ donde $B - mg$ cambia de signo y lo divide a la mitad en cada iteración: tras $k$ pasos el error es $\\le L/2^{k+1}$, convergencia lineal con factor $1/2$. RK4 avanza el estado $(y, v)$ con $y_{n+1} = y_n + \\tfrac h6(k_1 + 2k_2 + 2k_3 + k_4)$ y tiene error global $O(h^4)$. Para medir el periodo se detectan dos cruces del mismo sentido por la posición de equilibrio, interpolando linealmente $t^* = t_i - \\delta_i(t_i - t_{i-1})/(\\delta_i - \\delta_{i-1})$ entre las muestras que rodean el cambio de signo.",
            "Lo normal es que la interpolación lineal del cruce comete un error $O(h^2)$ que domina sobre el $O(h^4)$ del integrador, pero aquí hay un efecto favorable: el cruce por el equilibrio ocurre en un punto de inflexión de $\\cos\\omega t$, donde la curvatura es nula, y el error de la interpolación es de orden superior; el periodo medido conserva el orden 4. Esto ilustra una regla general: el error de una cantidad derivada lo fija el eslabón más débil de la cadena, y a veces ese eslabón es mejor de lo que parece. Las limitaciones son que la bisección necesita un cambio de signo y una función monótona, y que RK4 con paso fijo no conserva la energía de forma exacta (el oscilador pierde o gana una fracción diminuta de amplitud por ciclo).",
            "Se usaron 60 iteraciones de bisección (intervalo final bastante menor que el redondeo) y 40000 pasos en 2 s para el periodo, con $h = 5\\times10^{-5}$ s, ya que el periodo de 1.1 s necesita buena resolución del cruce. El resultado, $T \\approx 1.10$ s, dice que el cilindro de $600\\ \\mathrm{kg/m^3}$ oscila con una frecuencia que depende de $\\rho_f gA/m$ y no de la amplitud. Los dos cálculos son independientes del analítico: ni $h_{eq} = (\\rho/\\rho_f)L$ ni $T = 2\\pi/\\omega$ se usan en el código.",
        ],
        numeric_code=f"""# (1) Biseccion para el equilibrio  B(h) = mg   (2) RK4 para el periodo de oscilacion
rho_f, g, A, L, m = {c16.RHO_WATER:.0f}, {c16.G}, {c16.AREA}, {c16.LEN}, {c16.MASS}      # datos del problema
B = lambda h: rho_f * g * A * max(h, 0)           # empuje con h metros sumergidos
lo, hi = 0.0, L
for k in range(60):                               # 60 iteraciones: L/2**60 ~ 4e-19 m
    mid = (lo + hi) / 2
    if B(mid) < m * g: lo = mid                   # empuje insuficiente: hundir mas
    else:              hi = mid                   # empuje excesivo: subir
h_eq = (lo + hi) / 2                              # profundidad de equilibrio
f = lambda s: array([s[1], g - B(s[0]) / m])      # s = (y, v):  y' = v,  v' = g - B(y)/m
s, N, h = array([h_eq + 0.02, 0.0]), 40000, 2.0 / 40000   # desplaza 2 cm, 40000 pasos de 2 s
cruces = []
for n in range(N):
    s_old = s
    k1 = f(s); k2 = f(s + h/2*k1)                 # pendientes RK4 (1 y 2)
    k3 = f(s + h/2*k2); k4 = f(s + h*k3)          # pendientes RK4 (3 y 4)
    s = s + h/6 * (k1 + 2*k2 + 2*k3 + k4)         # avance del estado
    if s_old[0] - h_eq > 0 >= s[0] - h_eq:        # cruce descendente por el equilibrio
        cruces.append(interp_lineal(n, s_old, s)) # tiempo refinado por interpolacion lineal
T = cruces[1] - cruces[0]                         # periodo = diferencia entre cruces""",
        numeric_converge=[
            ("Error del periodo contra el paso",
             f"Se mide el periodo con distintos números de pasos en 2 s. La pendiente observada es {pp:.2f}, cercana a 4: la interpolación lineal del cruce cae en un punto de inflexión de la señal, donde su error es de orden superior a $h^2$, aunque la tabla es irregular porque la cancelación depende de dónde caiga el cruce dentro del paso. Con los 40000 pasos del código el error es ${sci(e40000, 2)}$ s.",
             [tab, rf"p_T = {pp:.2f}\qquad (\text{{teórico RK4: }}4)"]),
            ("Error del propio integrador",
             f"Para aislar a RK4 se compara la trayectoria $y(t)$ calculada con la solución $y = h_{{eq}} + 0.02\\cos\\omega t$ en todos los pasos de tiempo, sin interpolar. El error máximo sigue limpiamente la ley de orden 4 y es del mismo tamaño que el error del periodo, de modo que el integrador es el factor limitante.",
             [rf"\max_t|y^{{num}}(t) - y^{{an}}(t)| = " + r",\ ".join(sci(v, 2) for v in ey) + r"\ \mathrm{m},\qquad p_y = " + f"{py:.2f}"]),
            ("Bisección y criterio práctico",
             "La bisección elimina un bit por iteración hasta alcanzar el redondeo de $h_{eq}$ (cerca de $10^{-16}$ m), de modo que 60 iteraciones sobran. Un resultado de periodo es confiable cuando se obtiene el mismo valor con $h$ y $h/2$ dentro de la tolerancia, el cruce no cae en una zona de pendiente casi nula y los dos periodos consecutivos coinciden.",
             [ktab, rf"T_{{num}} = {c16.period_numeric():.9f}\ \mathrm{{s}},\qquad T_{{an}} = {ta:.9f}\ \mathrm{{s}}"]),
        ],
    )


DETAIL = {
    "SkaterSpin": _skater(),
    "GyroscopePrecession": _gyro(),
    "LadderFriction": _ladder(),
    "BeamReactions": _beam(),
    "KeplerOrbit": _kepler(),
    "EscapeVelocity": _escape(),
    "PressureDepth": _pressure(),
    "FloatingCylinder": _cylinder(),
}
