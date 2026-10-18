"""Explicacion del metodo numerico y estudio de error: Volumen 1, capitulos 5 a 8.

Escenas: StaticEquilibrium, InclinePulley, FrictionIncline, DragFall,
SpringWork, WorkEnergyTheorem, PendulumEnergy, SpringFriction.
Campos: numeric_intro, numeric_code, numeric_converge. Todas las cifras salen de ejecutar `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch05_newton as c5
from physics import ch06_dynamics as c6
from physics import ch07_work_energy as c7
from physics import ch08_energy_conservation as c8
from physics.integrators import integrate, rk4_step, riemann_left, trapezoid

EPS = float(np.finfo(float).eps)


def _tab(cols: str, header: str, rows: list[str]) -> str:
    """Tabla TeX (array) a partir de filas ya formateadas."""
    return r"\begin{array}{" + cols + "}" + header + r"\\ \hline " + r"\\ ".join(rows) + r"\end{array}"


def _slopes(errs) -> list[float | None]:
    """Pendiente observada log2(e_i / e_{i+1}) al duplicar n (o dividir h entre 2)."""
    out: list[float | None] = []
    for a, b in zip(errs[:-1], errs[1:]):
        out.append(float(np.log2(a / b)) if a > 0 and b > 0 else None)
    return out


def _sl(p) -> str:
    return "--" if p is None else f"{p:.2f}"


# =================================================================== StaticEquilibrium
def _static() -> dict:
    w, al, be = c5.WEIGHT, c5.ALPHA_DEG, c5.BETA_DEG
    t1, t2 = c5.tensions_analytic()
    n1, n2 = c5.tensions_numeric()
    a, b = np.deg2rad(al), np.deg2rad(be)
    m = np.array([[np.cos(a), -np.cos(b)], [np.sin(a), np.sin(b)]])
    cond0 = float(np.linalg.cond(m))
    det0 = float(np.linalg.det(m))
    cases = [(30.0, 45.0), (60.0, 60.0), (80.0, 85.0), (87.0, 88.0), (89.0, 89.5)]
    rows = []
    for ca, cb in cases:
        ra, rb = np.deg2rad(ca), np.deg2rad(cb)
        mm = np.array([[np.cos(ra), -np.cos(rb)], [np.sin(ra), np.sin(rb)]])
        cd = float(np.linalg.cond(mm))
        an1, an2 = c5.tensions_analytic(w, ca, cb)
        nu1, nu2 = c5.tensions_numeric(w, ca, cb)
        rel = abs(nu1 - an1) / an1
        res = float(np.linalg.norm(c5.net_force(nu1, nu2, w, ca, cb)))
        rows.append(rf"{ca:.0f}^\circ,\,{cb:.1f}^\circ & {fx(cd,2)} & {fx(nu1,1)} & {sci(rel,2)} & {sci(res,2)}")
    tab = _tab("lrrrr", r"(\alpha,\beta) & \kappa(M) & T_1\ (\mathrm{N}) & |\Delta T_1|/T_1 & \|\sum\vec F\|\ (\mathrm{N})", rows)
    # sensibilidad a un error de 0.1 grado en alpha
    d = 0.1
    p1, p2 = c5.tensions_numeric(w, al + d, be)
    q1, q2 = c5.tensions_numeric(w, al - d, be)
    sens = [(c5.tensions_numeric(w, ca + d, cb)[0] - c5.tensions_numeric(w, ca - d, cb)[0]) / (2 * c5.tensions_numeric(w, ca, cb)[0]) for ca, cb in cases]
    srow = [rf"{ca:.0f}^\circ,\,{cb:.1f}^\circ & {sci(s_,2)}" for (ca, cb), s_ in zip(cases, sens)]
    stab = _tab("lr", r"(\alpha,\beta) & \delta T_1/T_1\ \text{ para }\ \delta\alpha=\pm0.1^\circ", srow)
    return dict(
        numeric_intro=[
            "En equilibrio la suma de fuerzas sobre el nudo es cero, y eso da un sistema de dos ecuaciones lineales con dos incógnitas, las tensiones. Aquí el método numérico no es una aproximación por pasos sino un solucionador de álgebra lineal: la pregunta es cómo resolver $M\\vec T=\\vec b$ de forma mecánica, sin despejar a mano. Se necesita porque, con tres o más cables, o con ángulos dados como datos, la fórmula cerrada con $\\sin(\\alpha+\\beta)$ deja de existir o se vuelve engorrosa, mientras que el sistema matricial se escribe igual para cualquier configuración.",
            "La regla es la eliminación gaussiana con pivoteo parcial, que es lo que hace `numpy.linalg.solve` mediante una factorización $PM=LU$. Se resuelve primero $L\\vec y=P\\vec b$ (sustitución hacia adelante) y luego $U\\vec T=\\vec y$ (hacia atrás). Para una matriz de $2\\times2$ el costo es unas pocas operaciones; para $n\\times n$ crece como $O(n^3)$. No hay paso $h$ ni orden de convergencia: el algoritmo es directo y termina en un número finito de operaciones, y el único error es el de redondeo.",
            f"Ese error de redondeo no se mide con un orden sino con el número de condición $\\kappa(M)=\\|M\\|\\,\\|M^{{-1}}\\|$: el error relativo del resultado es del orden de $\\kappa\\,\\varepsilon_{{mq}}$, con $\\varepsilon_{{mq}}={sci(EPS,3)}$ en doble precisión. Aquí $\\det M=\\sin(\\alpha+\\beta)={fx(det0,4)}$ y $\\kappa={fx(cond0,3)}$, de modo que el sistema está muy bien condicionado y se esperan unos 15 dígitos correctos. El método falla cuando $\\alpha+\\beta\\to180^\\circ$: los cables se vuelven colineales, $\\det M\\to0$, $\\kappa\\to\\infty$ y cualquier pequeño error en los ángulos se amplifica enormemente (físicamente, tensiones gigantes).",
            "Los parámetros no son de ajuste: lo único que se elige es la matriz, cuyas filas son las condiciones $\\sum F_x=0$ y $\\sum F_y=0$, y el vector independiente $(0,W)$. La solución numérica es independiente de la analítica porque no usa la fórmula $T_1=W\\cos\\beta/\\sin(\\alpha+\\beta)$ (que sale de eliminar a mano con la regla del seno): solo recibe los números de la matriz. Por eso comparar ambas es una verificación real, y la fuerza neta residual con las tensiones calculadas es una tercera prueba, que no requiere conocer la respuesta.",
        ],
        numeric_code="""import numpy as np

W, alpha, beta = 500.0, np.deg2rad(30), np.deg2rad(45)   # peso y angulos de los cables

M = np.array([[np.cos(alpha), -np.cos(beta)],             # fila 1: suma Fx = 0
              [np.sin(alpha),  np.sin(beta)]])            # fila 2: suma Fy = 0
b = np.array([0.0, W])                                    # lado derecho (0, W)

T = np.linalg.solve(M, b)                                 # LU con pivoteo: M T = b
T1, T2 = T                                                # tensiones en N

# verificacion independiente: la fuerza neta debe ser ~0
Fx = -T1*np.cos(alpha) + T2*np.cos(beta)                  # residuo horizontal
Fy =  T1*np.sin(alpha) + T2*np.sin(beta) - W              # residuo vertical
residuo = np.hypot(Fx, Fy)                                # debe ser ~1e-13 N

kappa = np.linalg.cond(M)                                 # numero de condicion
# error relativo esperado ~ kappa * eps (eps = 2.2e-16)
""",
        numeric_converge=[
            ("Qué se estudia en un método directo", f"No hay paso ni iteraciones que refinar, así que el estudio de error es de condicionamiento y de residuo. Se resuelve el sistema para varias geometrías, se compara con la fórmula cerrada y se evalúa la fuerza neta con las tensiones numéricas. La tabla se calculó ejecutando `tensions_numeric` y `net_force`.",
             [tab]),
            ("Cota de redondeo y su magnitud", f"Para la geometría de la escena $\\kappa={fx(cond0,3)}$, así que la cota $\\kappa\\varepsilon_{{mq}}={sci(cond0*EPS,2)}$ es la cota teórica, mientras que el observado en la tabla va de $0$ a unos $10^{{-14}}$ (dentro de la cota para los valores de $\\kappa$ de cada fila). A medida que $\\alpha+\\beta\\to180^\\circ$ crece $\\kappa$, y el error crece con él, pero sigue siendo diminuto: el error que importa en la práctica no es el del algoritmo sino el de los datos.",
             [rf"\dfrac{{\|\delta\vec T\|}}{{\|\vec T\|}}\ \lesssim\ \kappa(M)\,\varepsilon_{{mq}} = ({fx(cond0,3)})({sci(EPS,3)}) = {sci(cond0*EPS,2)}",
              rf"|T_1^{{num}}-T_1| = {sci(abs(n1-t1),2)}\ \mathrm{{N}},\qquad |T_2^{{num}}-T_2| = {sci(abs(n2-t2),2)}\ \mathrm{{N}}"]),
            ("Sensibilidad a los datos y criterio de confianza", f"Si los ángulos medidos tienen una incertidumbre de $0.1^\\circ$, la tensión cambia en la proporción que da la tabla, mucho mayor que el redondeo cuando los cables son casi verticales. Criterio: el resultado es confiable si el residuo $\\|\\sum\\vec F\\|$ es del orden de $\\varepsilon_{{mq}}W={sci(EPS*w,2)}$ N y $\\kappa$ es moderado ($\\kappa\\ll10^{{15}}$); lo que limita entonces es la precisión de los ángulos.",
             [stab]),
        ],
    )


# =================================================================== InclinePulley
def _bis_pulley(n_it: int) -> tuple[float, float]:
    lo, hi = 0.0, 89.9
    for _ in range(n_it):
        mid = 0.5 * (lo + hi)
        if c5.pulley_numeric(c5.M1, c5.M2, lo)[0] * c5.pulley_numeric(c5.M1, c5.M2, mid)[0] <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), hi - lo


def _pulley() -> dict:
    a_an, t_an = c5.pulley_analytic()
    a_n, t_n = c5.pulley_numeric()
    th_an = c5.balance_angle_analytic()
    rows, errs = [], []
    its = (5, 10, 20, 30, 40, 60)
    for k in its:
        th, wd = _bis_pulley(k)
        errs.append(abs(th - th_an))
        rows.append(rf"{k} & {sci(wd,3)} & {th:.10f} & {sci(abs(th-th_an),2)}")
    tab = _tab("rrrr", r"k & \text{ancho}\ (^\circ) & \theta_k\ (^\circ) & |\theta_k-\theta_{eq}|\ (^\circ)", rows)
    t, y = c5.pulley_trajectory()
    s_ex = 0.5 * a_an * 3.0**2
    v_ex = a_an * 3.0
    # sensibilidad: cond del sistema en (a,T)
    mm = np.array([[c5.M1, -1.0], [c5.M2, 1.0]])
    cnd = float(np.linalg.cond(mm))
    # pendiente de a(theta) en la raiz
    d = 1e-4
    sl = (c5.pulley_numeric(c5.M1, c5.M2, th_an + d)[0] - c5.pulley_numeric(c5.M1, c5.M2, th_an - d)[0]) / (2 * d)
    return dict(
        numeric_intro=[
            "Aquí hay tres piezas numéricas encadenadas. Primero, las dos leyes de Newton (una por cuerpo) forman un sistema lineal en las incógnitas $(a,T)$, que se resuelve por eliminación gaussiana en lugar de despejar a mano. Segundo, con $a$ conocida se integra el movimiento $(s,v)$ con Runge-Kutta de orden 4. Tercero, el ángulo de equilibrio se obtiene como raíz de $a(\\theta)=0$ con el método de bisección, que se usa porque solo necesita evaluar la función y un cambio de signo.",
            "En RK4 cada paso combina cuatro pendientes: $y_{n+1}=y_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$, con $k_1=f(t_n,y_n)$, $k_2=f(t_n+\\tfrac h2,y_n+\\tfrac h2k_1)$, $k_3=f(t_n+\\tfrac h2,y_n+\\tfrac h2k_2)$ y $k_4=f(t_n+h,y_n+hk_3)$. Su error local es $O(h^5)$ y el global $O(h^4)$. En la bisección, si $f(a)f(b)<0$ y $f$ es continua, el intervalo $[a,b]$ contiene una raíz; se evalúa el punto medio y se conserva la mitad con cambio de signo, de modo que tras $k$ iteraciones el ancho es $(b-a)/2^k$: convergencia lineal, un bit por iteración, sin depender de la derivada.",
            f"En esta escena la aceleración es constante, por lo que RK4 es exacto salvo redondeo (su error depende de derivadas de orden alto de $y(t)$, que aquí son nulas): el estudio del orden se hace mejor con la bisección. Con $60$ iteraciones el intervalo inicial de $89.9^\\circ$ baja a ${sci(89.9/2**60,2)}$ grados, por debajo del redondeo; con $40$ ya está en ${sci(89.9/2**40,2)}^\\circ$. La bisección puede fallar si no hay cambio de signo en el intervalo inicial o si hay raíces múltiples; aquí $a(0)>0$ y $a(89.9^\\circ)<0$ garantizan una única raíz.",
            "Los parámetros se eligen así: $n=300$ pasos en $3$ s para la trayectoria ($h=0.01$ s, más que suficiente porque el integrador es exacto en este caso) y $60$ iteraciones de bisección para llegar al límite de precisión de la máquina. Nada de esto reutiliza la fórmula $a=(m_2-m_1\\sin\\theta)g/(m_1+m_2)$: el sistema lineal se arma desde las ecuaciones de cada cuerpo y la raíz se busca sin saber que es $\\arcsin(m_2/m_1)$. Por eso coincidir con la solución analítica a $10^{-14}$ es una verificación genuina.",
        ],
        numeric_code="""m1, m2, g = 4.0, 3.0, 9.81                        # masas (kg) y gravedad
def solve_aT(theta):                              # sistema lineal en (a, T)
    s = np.sin(np.deg2rad(theta))
    M = np.array([[m1, -1.0], [m2, 1.0]])         # m1 a - T = -m1 g s ; m2 a + T = m2 g
    rhs = np.array([-m1*g*s, m2*g])
    return np.linalg.solve(M, rhs)                # (a, T) por eliminacion gaussiana

a, T = solve_aT(30.0)                             # a en m/s2 ; T en N
def f(t, y): return np.array([y[1], a])           # estado y=(s, v): s'=v, v'=a
y, h = np.array([0.0, 0.0]), 3.0/300              # 300 pasos de h = 0.01 s
for i in range(300):
    k1 = f(0, y); k2 = f(0, y + h/2*k1)           # cuatro pendientes de RK4
    k3 = f(0, y + h/2*k2); k4 = f(0, y + h*k3)
    y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)           # promedio ponderado 1,2,2,1
lo, hi = 0.0, 89.9                                # biseccion de a(theta) = 0
for _ in range(60):
    mid = 0.5*(lo + hi)
    if solve_aT(lo)[0]*solve_aT(mid)[0] <= 0:     # cambio de signo en [lo, mid]
        hi = mid
    else:
        lo = mid                                  # la raiz esta en [mid, hi]
theta_eq = 0.5*(lo + hi)                          # ancho final 89.9/2**60
""",
        numeric_converge=[
            ("Bisección: error contra iteraciones", f"Se ejecutó la misma bisección del código con $k=5,\\dots,60$ iteraciones y se comparó con $\\theta_{{eq}}=\\arcsin(m_2/m_1)={th_an:.10f}^\\circ$. El ancho del intervalo se divide por 2 en cada iteración (error lineal en los bits) hasta que se alcanza el piso de redondeo.",
             [tab]),
            ("Pendiente observada y piso de redondeo", f"Entre $k=5$ y $k=10$ el error baja de ${sci(errs[0],2)}^\\circ$ a ${sci(errs[1],2)}^\\circ$, mientras el ancho se divide por $2^5=32$: el error observado es siempre menor o igual que el ancho y baja, en promedio, a la mitad por iteración. Después de unas $50$ iteraciones el error ya no baja, porque lo domina el redondeo de $a(\\theta)$ cerca de la raíz: con pendiente $|a'(\\theta_{{eq}})|={abs(sl):.4f}\\ \\mathrm{{m/s^2}}$ por grado y $a\\sim\\varepsilon_{{mq}}g$, el piso es del orden de $\\varepsilon_{{mq}}g/|a'|$.",
             [rf"\text{{ancho}}_k = \dfrac{{89.9}}{{2^k}},\qquad \text{{piso}}\approx\dfrac{{\varepsilon_{{mq}}\,g}}{{|a'(\theta_{{eq}})|}} = {sci(EPS*c5.G/abs(sl),2)}\ ^\circ"]),
            ("Sistema lineal y RK4: error de redondeo", f"El sistema $2\\times2$ tiene $\\kappa={fx(cnd,3)}$ (bien condicionado) y RK4 con $a$ constante es exacto salvo redondeo; la trayectoria se compara con $s=\\tfrac12at^2$ y $v=at$ en $t=3$ s. Criterio de confianza: error de la raíz menor que la tolerancia pedida y residuos de $a$, $T$ y $s$ del orden de $10^{{-14}}$ o menores.",
             [rf"|a^{{num}}-a| = {sci(abs(a_n-a_an),2)}\ \mathrm{{m/s^2}},\qquad |T^{{num}}-T| = {sci(abs(t_n-t_an),2)}\ \mathrm{{N}}",
              rf"|s^{{num}}(3)-\tfrac12at^2| = {sci(abs(float(y[-1,0])-s_ex),2)}\ \mathrm{{m}},\qquad |v^{{num}}(3)-at| = {sci(abs(float(y[-1,1])-v_ex),2)}\ \mathrm{{m/s}}"]),
        ],
    )


# =================================================================== FrictionIncline
def _bis_fric(n_it: int) -> tuple[float, float]:
    lo, hi = 1.0, 80.0
    for _ in range(n_it):
        mid = 0.5 * (lo + hi)
        _, s, _ = c6.simulate_incline(mid, 1.0, c6.MU_S, c6.MU_K, c6.G, dt=0.01)
        if s[-1] > 1e-9:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), hi - lo


def _friction() -> dict:
    th0, mus, muk, g = c6.THETA_DEG, c6.MU_S, c6.MU_K, c6.G
    thc = c6.critical_angle_analytic()
    a_an = c6.incline_acceleration_analytic(th0)
    rows, errs = [], []
    its = (5, 10, 15, 20, 30, 40)
    for k in its:
        th, wd = _bis_fric(k)
        errs.append(abs(th - thc))
        rows.append(rf"{k} & {sci(wd,3)} & {th:.9f} & {sci(abs(th-thc),2)}")
    tab = _tab("rrrr", r"k & \text{ancho}\ (^\circ) & \theta_k\ (^\circ) & |\theta_k-\theta_c|\ (^\circ)", rows)
    drows = []
    for dt in (0.1, 0.01, 1e-3):
        _, s, v = c6.simulate_incline(th0, 2.0, dt=dt)
        drows.append(rf"{sci(dt,1)} & {int(round(2.0/dt))} & {sci(abs(float(v[-1])/2-a_an),2)} & {sci(abs(float(s[-1])-0.5*a_an*4),2)}")
    dtab = _tab("rrrr", r"\Delta t\ (\mathrm{s}) & N & |a^{num}-a|\ (\mathrm{m/s^2}) & |s^{num}-\tfrac12at^2|\ (\mathrm{m})", drows)
    thn = c6.critical_angle_numeric()
    # sensibilidad al umbral de desplazamiento 1e-9
    return dict(
        numeric_intro=[
            "La fricción estática y la cinética hacen que el movimiento sea una máquina de estados: mientras el bloque está en reposo y $\\tan\\theta\\le\\mu_s$ no se mueve, y al superar el umbral desliza con aceleración constante. No hay una sola ecuación diferencial suave que integrar de punta a punta, y por eso se simula paso a paso con una variable de estado (reposo o deslizamiento) en lugar de usar una fórmula cerrada. El ángulo crítico, además, no es la solución de una ecuación derivable sino de una condición lógica (“¿se movió?”), lo que pide un método que solo use evaluaciones.",
            "Dentro del estado deslizante se integra $\\dot s=v$, $\\dot v=g(\\sin\\theta-\\mu_k\\cos\\theta)$ con RK4, $y_{n+1}=y_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$, de error global $O(h^4)$. Para encontrar el ángulo crítico se aplica bisección sobre la función-indicador $\\chi(\\theta)=[s(1\\ \\mathrm{s})>10^{-9}]$: se simula en el punto medio $\\theta_m$; si el bloque se movió, el umbral está abajo y $\\theta_{hi}\\leftarrow\\theta_m$; si no, $\\theta_{lo}\\leftarrow\\theta_m$. Tras $k$ iteraciones el ancho es $79^\\circ/2^k$, sin importar que $\\chi$ sea discontinua.",
            f"Aquí RK4 es exacto salvo redondeo porque la aceleración es constante, de modo que el error de la simulación no depende de $\\Delta t$ (se verá en la tabla) y toda la incertidumbre del ángulo crítico viene de la bisección: con $40$ iteraciones el ancho es ${sci(79/2**40,2)}^\\circ$. El método falla si el indicador no es monótono (si hubiera dos cambios de estado en el intervalo, la bisección convergería a uno cualquiera) o si el umbral $10^{{-9}}$ m es tan pequeño que el ruido lo confunde con movimiento; con pasos y desplazamientos exactamente nulos en reposo no hay ambigüedad.",
            "Los parámetros del código son $\\Delta t=10^{-3}$ s y $2000$ pasos para la simulación principal, y $\\Delta t=0.01$ s, $1$ s de simulación y $40$ iteraciones para la bisección (cada evaluación cuesta solo $100$ pasos). Esto es independiente de la solución analítica: no se usa $\\arctan\\mu_s$ ni $g(\\sin\\theta-\\mu_k\\cos\\theta)$ para el ángulo, sino que se descubre simulando; la coincidencia con $\\arctan\\mu_s$ confirma que la lógica de estados implementa la misma física.",
        ],
        numeric_code="""g, mu_s, mu_k = 9.81, 0.45, 0.30                  # gravedad y coeficientes de friccion
def simulate(theta_deg, t_end, dt):               # maquina de estados reposo/deslizamiento
    th = np.deg2rad(theta_deg)
    n = round(t_end/dt); s = np.zeros(n+1); v = np.zeros(n+1)
    moving = False
    for i in range(n):
        if not moving:                            # en reposo: se comprueba el umbral estatico
            if g*np.sin(th) > mu_s*g*np.cos(th):  # componente del peso > friccion maxima
                moving = True
            else:
                continue                          # sigue quieto: s, v no cambian
        a = g*(np.sin(th) - mu_k*np.cos(th))      # aceleracion con friccion cinetica
        f = lambda t, y: np.array([y[1], a])
        y = np.array([s[i], v[i]])
        k1 = f(0, y); k2 = f(0, y + dt/2*k1)      # cuatro pendientes de RK4
        k3 = f(0, y + dt/2*k2); k4 = f(0, y + dt*k3)
        s[i+1], v[i+1] = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return s, v
lo, hi = 1.0, 80.0                                # biseccion del angulo critico
for _ in range(40):
    mid = 0.5*(lo + hi)
    s, _ = simulate(mid, 1.0, 0.01)               # 100 pasos por evaluacion
    if s[-1] > 1e-9: hi = mid                     # se movio: el umbral esta abajo
    else:            lo = mid                     # no se movio: esta arriba
theta_c = 0.5*(lo + hi)                           # ancho final 79/2**40 grados
""",
        numeric_converge=[
            ("Bisección del ángulo crítico", f"Se repitió la bisección del código con distintas iteraciones $k$ y se comparó con $\\theta_c=\\arctan\\mu_s={thc:.9f}^\\circ$. Por ser un indicador discontinuo, el error no decae de forma suave sino en escalón, acotado por el ancho del intervalo.",
             [tab]),
            ("Orden y piso del método", f"El ancho cae exactamente por $2$ en cada iteración, y el error observado es menor o igual al ancho (de $k=5$ a $k=40$ cae de ${sci(errs[0],2)}^\\circ$ a ${sci(errs[-1],2)}^\\circ$). Con el valor del código ($40$ iteraciones) se obtiene $\\theta_c^{{num}}={thn:.9f}^\\circ$. El piso no es el redondeo sino la resolución del indicador: el test $g\\sin\\theta>\\mu_sg\\cos\\theta$ se evalúa con error $\\sim\\varepsilon_{{mq}}$, es decir del orden de ${sci(EPS,1)}$ en términos relativos, muy por debajo del ancho.",
             [rf"|\theta_c^{{num}}-\theta_c| = {sci(abs(thn-thc),2)}^\circ,\qquad \text{{ancho}}_{{40}} = \dfrac{{79}}{{2^{{40}}}} = {sci(79/2**40,2)}^\circ"]),
            ("Convergencia de RK4 en el paso de tiempo", f"Para el estado deslizante se repitió la simulación con $\\Delta t=0.1,\\,10^{{-2}},\\,10^{{-3}}$ s a $\\theta={th0:.0f}^\\circ$ y se comparó con $a=g(\\sin\\theta-\\mu_k\\cos\\theta)={a_an:.5f}\\ \\mathrm{{m/s^2}}$. El error no depende de $\\Delta t$ (sube ligeramente con $N$ por el acumulado de redondeo) porque con $f$ constante RK4 reproduce exactamente $s=\\tfrac12at^2$. Criterio de confianza: error del ángulo menor que el ancho de la bisección, y errores de $a$ y $s$ del orden de $N\\varepsilon_{{mq}}$.",
             [dtab]),
        ],
    )


# =================================================================== DragFall
def _t95(n: int) -> float:
    t, y = c6.drag_numeric(n, "rk4", 6.0)
    target = 0.95 * c6.terminal_speed()
    i = int(np.argmax(y[:, 1] >= target))
    t0, t1, v0, v1 = t[i - 1], t[i], y[i - 1, 1], y[i, 1]
    return float(t0 + (target - v0) * (t1 - t0) / (v1 - v0))


def _drag() -> dict:
    ns = (8, 16, 32, 64, 128)
    ex = float(c6.drag_velocity_analytic(1.0))
    ee, er = [], []
    for n in ns:
        ee.append(abs(c6.drag_numeric(n, "euler", 1.0)[1][-1, 1] - ex))
        er.append(abs(c6.drag_numeric(n, "rk4", 1.0)[1][-1, 1] - ex))
    se, sr = _slopes(ee), _slopes(er)
    rows = []
    for i, n in enumerate(ns):
        rows.append(rf"{n} & {fx(1.0/n,4)} & {sci(ee[i],3)} & {'--' if i==0 else _sl(se[i-1])} & {sci(er[i],3)} & {'--' if i==0 else _sl(sr[i-1])}")
    tab = _tab("rrrrrr", r"n & h\ (\mathrm{s}) & \varepsilon_{Euler} & p_E & \varepsilon_{RK4} & p_{RK4}", rows)
    tau, vt = c6.tau(), c6.terminal_speed()
    t95 = c6.time_to_fraction_analytic()
    nn = (100, 400, 1600, 4000)
    r2e = [abs(_t95(n) - t95) for n in nn]
    r2 = [rf"{n} & {fx(6.0/n,5)} & {sci(e,2)}" for n, e in zip(nn, r2e)]
    tab2 = _tab("rrr", r"n & h\ (\mathrm{s}) & |t_{95}^{num}-t_{95}|\ (\mathrm{s})", r2)
    pasos_crit = 2 * tau
    return dict(
        numeric_intro=[
            f"La ecuación $m\\dot v=mg-bv$ tiene solución analítica, pero se integra numéricamente para mostrar cómo se comporta un método de paso fijo cuando se conoce la respuesta exacta, y porque en cuanto la resistencia deja de ser lineal (por ejemplo $\\propto v^2$) ya no hay fórmula cerrada y el único camino es este. Se escribe el estado $y=(y,v)$ y se avanza en pasos $h$ con dos métodos: Euler y Runge-Kutta de orden 4, para comparar sus órdenes.",
            "Euler avanza con la pendiente al inicio del paso, $y_{n+1}=y_n+h\\,f(t_n,y_n)$. Es el desarrollo de Taylor cortado en el primer término, con error local $O(h^2)$ y global $O(h)$: duplicar $n$ reduce el error a la mitad. RK4 usa cuatro pendientes, $y_{n+1}=y_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$, coincide con Taylor hasta $h^4$, tiene error local $O(h^5)$ y global $O(h^4)$: duplicar $n$ lo divide por $16$. Ambos son consistentes con la ecuación y estables para $h$ pequeño.",
            f"Las limitaciones: Euler es estable solo si $h<2\\tau=2m/b={pasos_crit:.3f}$ s (el factor de amplificación por paso es $1-h/\\tau$), y RK4 pierde su ventaja si $f$ no es suave. En esta escena $\\tau=m/b={tau:.4f}$ s y la rapidez terminal es $v_t=mg/b={vt:.4f}$ m/s; el estudio se hace en $t\\in[0,1]$ s con $n=8,\\dots,128$, es decir $h$ entre ${fx(1/8,3)}$ y ${fx(1/128,4)}$ s, siempre muy por debajo del límite de estabilidad. Para el tiempo al $95\\,\\%$ se usan $n=4000$ pasos en $[0,6]$ s con interpolación lineal del cruce, para que el resultado no quede limitado por la rejilla.",
            "El método es independiente del analítico porque nunca usa $v_t(1-e^{-t/\\tau})$: solo aplica la regla de recurrencia a la ecuación diferencial. La solución exacta aparece únicamente para medir el error, no para producir el resultado. Esto permite el estudio clásico de convergencia: la pendiente $p=\\log_2(\\varepsilon_n/\\varepsilon_{2n})$ debe aproximarse a $1$ para Euler y a $4$ para RK4.",
        ],
        numeric_code="""m, b, g = 0.5, 0.8, 9.81                          # masa (kg), coef. de arrastre (kg/s), gravedad
def f(t, y):                                      # estado y = (posicion, velocidad)
    return np.array([y[1], g - (b/m)*y[1]])       # y' = v ; v' = g - (b/m) v

def euler(f, t, y, h):
    return y + h*f(t, y)                          # una sola pendiente: error global O(h)

def rk4(f, t, y, h):
    k1 = f(t, y)                                  # pendiente al inicio
    k2 = f(t + h/2, y + h/2*k1)                   # pendiente en el punto medio (1)
    k3 = f(t + h/2, y + h/2*k2)                   # pendiente en el punto medio (2)
    k4 = f(t + h, y + h*k3)                       # pendiente al final
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)        # promedio 1,2,2,1: error global O(h^4)

def integrar(step, n, t_end=1.0):
    h, y = t_end/n, np.array([0.0, 0.0])          # parte del reposo
    for i in range(n):
        y = step(f, i*h, y, h)
    return y[1]                                   # velocidad en t_end

v_ex = (m*g/b)*(1 - np.exp(-1.0/(m/b)))           # solo para medir el error
for n in (8, 16, 32, 64, 128):
    eE = abs(integrar(euler, n) - v_ex)           # error de Euler
    eR = abs(integrar(rk4, n) - v_ex)             # error de RK4
    # pendiente observada: p = log2(e_n / e_2n)
""",
        numeric_converge=[
            ("Error contra el número de pasos", f"Se integró hasta $t=1$ s con $n=8,16,32,64,128$ y se midió el error de la velocidad contra $v(1)={ex:.8f}\\ \\mathrm{{m/s}}$. La última columna de cada método es la pendiente observada $p=\\log_2(\\varepsilon_n/\\varepsilon_{{2n}})$.",
             [tab]),
            ("Lectura del estudio", f"Euler tiende a $p=1$ (de {_sl(se[0])} a {_sl(se[-1])}) y RK4 a $p=4$ (de {_sl(sr[0])} a {_sl(sr[-1])}), como predice la teoría. Con el mismo $h=1/128$ s el error de RK4 (${sci(er[-1],2)}$ m/s) es ${sci(ee[-1]/er[-1],2)}$ veces menor que el de Euler (${sci(ee[-1],2)}$ m/s). La fuente dominante es el error de truncamiento; el de redondeo (del orden de ${sci(EPS*vt,1)}$ m/s) solo se alcanzaría con $n$ enorme.",
             [rf"\varepsilon_{{Euler}}(h)\approx C_E\,h,\qquad \varepsilon_{{RK4}}(h)\approx C_4\,h^4,\qquad \dfrac{{\varepsilon_{{Euler}}(1/128)}}{{\varepsilon_{{RK4}}(1/128)}} = {sci(ee[-1]/er[-1],2)}"]),
            ("Tiempo al 95 % y criterio de confianza", f"El tiempo en que $v=0.95v_t$ se obtiene interpolando linealmente entre los dos pasos que encierran el valor objetivo; el error de esa interpolación es $O(h^2)$ y la tabla muestra una pendiente de unos {np.log2(r2e[1]/r2e[2])/2:.1f} entre $n=400$ y $n=1600$ al comparar con $t_{{95}}=-\\tau\\ln0.05={t95:.6f}$ s. Criterio práctico: el resultado es confiable cuando al duplicar $n$ el resultado cambia menos que la precisión deseada (con $n=4000$ el error es de apenas ${sci(r2e[3],1)}$ s) y la pendiente observada coincide con el orden teórico.",
             [tab2]),
        ],
    )


# =================================================================== SpringWork
def _spring_work() -> dict:
    k, x1, x2 = c7.K, c7.X1, c7.X2
    ns = (2, 4, 8, 16, 32)
    wex = c7.spring_work_analytic()
    conv = c7.work_convergence(ns)
    er = [r["riemann"] for r in conv]
    sr = _slopes(er)
    rows = []
    for i, r in enumerate(conv):
        rows.append(rf"{r['n']} & {fx((x2-x1)/r['n'],4)} & {sci(r['riemann'],3)} & {'--' if i==0 else _sl(sr[i-1])} & {sci(r['trapezoid'],2)} & {sci(r['simpson'],2)}")
    tab = _tab("rrrrrr", r"n & h\ (\mathrm{m}) & \varepsilon_{R} & p_R & \varepsilon_{T} & \varepsilon_{S}", rows)
    # prueba con fuerza no lineal F = kx + c x^4
    c = 2.0e4
    xs_ = lambda n: np.linspace(x1, x2, n + 1)  # noqa: E731
    fnl = lambda x: k * x + c * x**4  # noqa: E731
    wnl = 0.5 * k * (x2**2 - x1**2) + c * (x2**5 - x1**5) / 5
    et, es = [], []
    for n in ns:
        x = xs_(n)
        et.append(abs(trapezoid(fnl(x), x) - wnl))
        es.append(abs(c7.simpson(fnl(x), x) - wnl))
    st, ss = _slopes(et), _slopes(es)
    rows2 = []
    for i, n in enumerate(ns):
        rows2.append(rf"{n} & {sci(et[i],3)} & {'--' if i==0 else _sl(st[i-1])} & {sci(es[i],3)} & {'--' if i==0 else _sl(ss[i-1])}")
    tab2 = _tab("rrrrr", r"n & \varepsilon_T & p_T & \varepsilon_S & p_S", rows2)
    # estimador de Richardson para simpson
    x8, x16 = xs_(8), xs_(16)
    s8, s16 = c7.simpson(fnl(x8), x8), c7.simpson(fnl(x16), x16)
    est = abs(s16 - s8) / 15
    real = abs(s16 - wnl)
    return dict(
        numeric_intro=[
            "El trabajo de una fuerza que depende de la posición es $W=\\int_{x_1}^{x_2}F(x)\\,dx$. Aquí la integral se puede hacer a mano, pero en muchos problemas reales $F(x)$ solo se conoce en puntos medidos o su primitiva no existe en forma cerrada; la integración numérica sustituye la integral por una suma ponderada de valores de $F$ en una malla de nodos $x_i=x_1+ih$, $h=(x_2-x_1)/n$.",
            "Se comparan tres reglas. La suma de Riemann por la izquierda usa rectángulos, $W\\approx h\\sum_{i=0}^{n-1}F_i$, con error $O(h)$. El trapecio reemplaza la curva por segmentos, $W\\approx h\\left[\\tfrac12F_0+F_1+\\dots+F_{n-1}+\\tfrac12F_n\\right]$, con error $-\\tfrac{(b-a)}{12}h^2F''(\\xi)=O(h^2)$. Simpson ajusta una parábola a cada par de subintervalos, $W\\approx\\tfrac h3\\left[F_0+4F_1+2F_2+\\dots+4F_{n-1}+F_n\\right]$ con $n$ par, y su error es $-\\tfrac{(b-a)}{180}h^4F^{(4)}(\\xi)=O(h^4)$. Es exacta para polinomios de grado $\\le3$.",
            f"Como aquí $F=kx$ es lineal, $F''=0$ y $F^{{(4)}}=0$: trapecio y Simpson son exactos salvo redondeo (${sci(EPS,1)}$ relativo), mientras Riemann conserva su error $O(h)$ porque el rectángulo no reproduce la pendiente. Eso hace que la escena no ilustre los órdenes $2$ y $4$; por eso en la verificación se repite el estudio con una fuerza no lineal de prueba $F=kx+cx^4$ ($c={sci(c,2)}\\ \\mathrm{{N/m^4}}$), donde sí aparecen. Las reglas fallan cuando $F$ tiene saltos o singularidades dentro del intervalo, pues las derivadas del error dejan de estar acotadas.",
            f"Se eligió $n=4$ para mostrar la mecánica a mano ($h=0.05$ m) y $n=2,4,8,16,32$ para estudiar la convergencia; Simpson exige $n$ par. El método es independiente de la solución analítica $W=\\tfrac12k(x_2^2-x_1^2)$: solo suma valores de $F$ en nodos, y la fórmula cerrada aparece únicamente como referencia ($W={wex:.4f}$ J) para calcular el error.",
        ],
        numeric_code="""k, x1, x2 = 200.0, 0.10, 0.30                     # constante (N/m) y limites (m)
def trabajo(n, metodo):
    x = np.linspace(x1, x2, n + 1)                # nodos x_i = x1 + i h
    F = k*x                                       # fuerza en cada nodo
    h = (x2 - x1)/n                               # ancho del subintervalo
    if metodo == "riemann":
        return h*F[:-1].sum()                     # rectangulos a la izquierda: O(h)
    if metodo == "trapecio":
        return h*(0.5*F[0] + F[1:-1].sum() + 0.5*F[-1])   # segmentos rectos: O(h^2)
    # Simpson (n par): pesos 1,4,2,4,...,4,1
    return h/3*(F[0] + F[-1] + 4*F[1:-1:2].sum() + 2*F[2:-1:2].sum())   # O(h^4)

W_ex = 0.5*k*(x2**2 - x1**2)                      # solo para medir el error
for n in (2, 4, 8, 16, 32):
    for m in ("riemann", "trapecio", "simpson"):
        err = abs(trabajo(n, m) - W_ex)           # error absoluto
    # p = log2(err_n / err_2n) da el orden observado
""",
        numeric_converge=[
            ("Error contra n para F = kx", f"Se calculó el error absoluto de las tres reglas con las funciones de `physics/ch07_work_energy` frente a $W={wex:.4f}$ J. Riemann baja a la mitad al duplicar $n$ ($p\\to1$); trapecio y Simpson quedan en el redondeo porque $F$ es lineal.",
             [tab]),
            ("Confirmar los órdenes con una fuerza no lineal", f"Para ver los órdenes teóricos se repitió el estudio con $F=kx+cx^4$, cuyo trabajo exacto es $W={wnl:.4f}$ J. El trapecio muestra $p\\to2$ y Simpson $p\\to4$ (pendientes observadas de {_sl(ss[0])} a {_sl(ss[-1])}), con errores muy por encima del redondeo.",
             [tab2]),
            ("Estimación del error sin conocer la respuesta", f"En la práctica no se conoce $W$; se usa que, para una regla de orden $p$, $\\varepsilon_{{2n}}\\approx|S_{{2n}}-S_n|/(2^p-1)$. Con Simpson ($p=4$) y la fuerza no lineal, $|S_{{16}}-S_8|/15={sci(est,2)}$ J frente al error real ${sci(real,2)}$ J. Criterio de confianza: la diferencia entre $n$ y $2n$ es menor que la tolerancia, y la pendiente observada es la del orden teórico.",
             [rf"S_8 = {s8:.6f}\ \mathrm{{J}},\qquad S_{{16}} = {s16:.6f}\ \mathrm{{J}},\qquad \dfrac{{|S_{{16}}-S_8|}}{{15}} = {sci(est,2)}\ \mathrm{{J}}"]),
        ],
    )


# =================================================================== WorkEnergyTheorem
def _wet() -> dict:
    m, f0, L = c7.MASS, c7.F0, c7.LENGTH
    om = float(np.sqrt(f0 / (m * L)))
    va = c7.final_speed_analytic()
    w_an = float(c7.kinetic_energy_analytic(L))
    # RK4 a tiempo fijo contra la solucion armonica x = L(1 - cos wt)
    tf = 0.5
    rhs = lambda _t, y: np.array([y[1], c7.force_profile(y[0]) / m])  # noqa: E731
    ns = (5, 10, 20, 40, 80)
    ex_v = L * om * np.sin(om * tf)
    ex_x = L * (1 - np.cos(om * tf))
    ev, ex = [], []
    for n in ns:
        _, y = integrate(rhs, np.array([0.0, 0.0]), 0.0, tf, n, rk4_step)
        ev.append(abs(y[-1, 1] - ex_v))
        ex.append(abs(y[-1, 0] - ex_x))
    sv = _slopes(ev)
    rows = [rf"{n} & {fx(tf/n,4)} & {sci(ex[i],3)} & {sci(ev[i],3)} & {'--' if i==0 else _sl(sv[i-1])}" for i, n in enumerate(ns)]
    tab = _tab("rrrrr", r"n & h\ (\mathrm{s}) & |x^{num}-x|\ (\mathrm{m}) & |v^{num}-v|\ (\mathrm{m/s}) & p", rows)
    dts = (0.1, 0.05, 0.025, 0.0125, 0.00625)
    ef = [abs(c7.final_speed_numeric(dt=d) - va) for d in dts]
    sf = _slopes(ef)
    rows2 = [rf"{sci(d,3)} & {sci(e,3)} & {'--' if i==0 else _sl(sf[i-1])}" for i, (d, e) in enumerate(zip(dts, ef))]
    tab2 = _tab("rrr", r"\Delta t\ (\mathrm{s}) & |v_f^{num}-v_f|\ (\mathrm{m/s}) & p", rows2)
    wsimp = [(n, abs(c7.work_integral_numeric(n) - w_an)) for n in (4, 40, 400)]
    wrow = [rf"{n} & {fx(L/n,4)} & {sci(e,2)}" for n, e in wsimp]
    wtab = _tab("rrr", r"n & h\ (\mathrm{m}) & |W^{num}-30|\ (\mathrm{J})", wrow)
    vn = c7.final_speed_numeric()
    wk = c7.work_integral_numeric()
    return dict(
        numeric_intro=[
            "Aquí se comprueba el teorema trabajo-energía con dos caminos numéricos que no se tocan. El primero integra la segunda ley de Newton $m\\ddot x=F(x)$ con Runge-Kutta de orden 4, sin usar energía en absoluto, y lee la rapidez cuando $x=L$. El segundo calcula el trabajo $W=\\int_0^LF\\,dx$ con la regla de Simpson. El teorema afirma que $\\tfrac12mv^2=W$; si las dos cantidades numéricas coinciden, el teorema queda verificado sin hacer ninguna integral a mano.",
            "RK4 avanza el estado $y=(x,v)$ con $y_{n+1}=y_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$, donde las cuatro pendientes se evalúan en el inicio, dos veces en el punto medio y en el final del paso; su error global es $O(h^4)$. Simpson aplica $W\\approx\\tfrac h3[F_0+4F_1+2F_2+\\dots+F_n]$ con error $O(h^4)$ y es exacta para polinomios de grado $\\le3$. La fuerza $F=F_0(1-x/L)$ es lineal, así que Simpson no tiene error de truncamiento aquí; el error de RK4 en la trayectoria sí lo tiene, aunque diminuto.",
            f"Hay una fuente de error adicional que no es de RK4: el bucle se detiene cuando el paso sobrepasa $x=L$, y el estado en $x=L$ se obtiene por interpolación lineal entre los dos últimos puntos, lo que introduce un error $O(h^2)$ en la rapidez final. Por eso el error de $v_f$ no sigue el orden $4$ de RK4 (la tabla lo muestra). Con $\\Delta t=10^{{-3}}$ s este error es ${sci(abs(vn-va),2)}$ m/s; para el análisis limpio del orden se integra hasta un tiempo fijo, donde la solución exacta es el movimiento armónico $x=L(1-\\cos\\omega t)$ con $\\omega=\\sqrt{{F_0/(mL)}}={om:.4f}$ rad/s.",
            f"Los parámetros: $\\Delta t=10^{{-3}}$ s (unos $1000$ pasos hasta $x=L$) y $n=400$ subintervalos para Simpson ($h=0.005$ m). Los dos métodos son independientes entre sí y del analítico: RK4 no conoce el teorema y Simpson no conoce la trayectoria. El resultado esperado, $W={w_an:.2f}$ J y $v_f={va:.4f}$ m/s, sale de cualquiera de los dos y debe concordar con $\\tfrac12mv_f^2$.",
        ],
        numeric_code="""m, F0, L = 2.0, 30.0, 2.0                         # masa (kg), fuerza inicial (N), recorrido (m)
F = lambda x: F0*(1 - x/L)                        # fuerza variable F(x)
def f(t, y): return np.array([y[1], F(y[0])/m])   # y=(x, v): x'=v ; v'=F(x)/m

y, t, dt = np.array([0.0, 0.0]), 0.0, 1e-3        # parte del reposo, h = 1 ms
hist = [y]
while y[0] < L:                                   # avanza hasta cruzar x = L
    k1 = f(t, y); k2 = f(t + dt/2, y + dt/2*k1)   # cuatro pendientes de RK4
    k3 = f(t + dt/2, y + dt/2*k2); k4 = f(t + dt, y + dt*k3)
    y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)          # promedio ponderado 1,2,2,1
    t += dt; hist.append(y)
ya, yb = hist[-2], hist[-1]
frac = (L - ya[0])/(yb[0] - ya[0])                # fraccion del paso hasta x = L
v_final = ya[1] + frac*(yb[1] - ya[1])            # interpolacion lineal (error O(h^2))

n = 400; x = np.linspace(0, L, n + 1); Fx = F(x)  # Simpson con 400 subintervalos
h = L/n
W = h/3*(Fx[0] + Fx[-1] + 4*Fx[1:-1:2].sum() + 2*Fx[2:-1:2].sum())   # trabajo
# teorema: 0.5*m*v_final**2 debe coincidir con W
""",
        numeric_converge=[
            ("Orden de RK4 a tiempo fijo", f"Para aislar el error del integrador se resolvió el mismo problema hasta $t={tf}$ s con $n=5,\\dots,80$ pasos y se comparó con la solución armónica exacta $x=L(1-\\cos\\omega t)$, $v=L\\omega\\sin\\omega t$. La pendiente observada tiende a $4$, el orden teórico de RK4.",
             [tab]),
            ("Rapidez final: el error lo domina la interpolación", f"Con el criterio de parada $x=L$ y paso $\\Delta t$ variable, el error de $v_f$ ya no sigue $O(h^4)$ sino la interpolación lineal del cruce, que es $O(\\Delta x^2)$ con $\\Delta x$ el tramo del último paso, y depende de en qué fracción del paso cae $x=L$ (de ahí las pendientes irregulares, de {_sl(min(p for p in sf if p))} a {_sl(max(p for p in sf if p))}). Con el $\\Delta t=10^{{-3}}$ s del código el resultado es $v_f^{{num}}={vn:.6f}$ m/s contra $v_f={va:.6f}$ m/s.",
             [tab2]),
            ("Simpson y verificación del teorema", f"Como $F$ es lineal, Simpson es exacta y el único error es de redondeo. Cerrando el círculo, la energía cinética numérica y el trabajo numérico coinciden hasta el error de la interpolación final: criterio de confianza = acuerdo entre dos métodos independientes dentro de la tolerancia del más grosero (aquí el de la trayectoria).",
             [wtab, rf"\tfrac12m(v_f^{{num}})^2 = {0.5*m*vn**2:.6f}\ \mathrm{{J}},\qquad W^{{num}} = {wk:.6f}\ \mathrm{{J}},\qquad \text{{diferencia}} = {sci(abs(0.5*m*vn**2-wk),2)}\ \mathrm{{J}}"]),
        ],
    )


# =================================================================== PendulumEnergy
def _pend() -> dict:
    L, g, th0 = c8.LENGTH, c8.G, c8.THETA0_DEG
    ns = (150, 300, 600, 1200, 2400)
    dr = {mth: [c8.energy_drift(mth, n=n) for n in ns] for mth in ("euler", "verlet", "rk4")}
    sl = {mth: _slopes(dr[mth]) for mth in dr}
    rows = []
    for i, n in enumerate(ns):
        rows.append(rf"{n} & {fx(6.0/n,4)} & {sci(dr['euler'][i],2)} & {sci(dr['verlet'][i],2)} & {sci(dr['rk4'][i],2)}")
    tab = _tab("rrrrr", r"n & h\ (\mathrm{s}) & \text{Euler} & \text{Verlet} & \text{RK4}", rows)
    mean_sl = {m_: float(np.mean([s_ for s_ in sl[m_] if s_ is not None])) for m_ in sl}
    va = c8.bottom_speed_analytic()
    ns2 = (150, 600, 2400, 6000)
    vrows = []
    for n in ns2:
        _, y = c8.pendulum_run("rk4", n=n)
        vm = float(np.max(np.abs(y[:, 1])) * L)
        vrows.append(rf"{n} & {fx(6.0/n,4)} & {vm:.7f} & {sci(abs(vm-va),2)}")
    vtab = _tab("rrrr", r"n & h\ (\mathrm{s}) & v_{max}^{num}\ (\mathrm{m/s}) & |v_{max}^{num}-v|\ (\mathrm{m/s})", vrows)
    return dict(
        numeric_intro=[
            "El péndulo con ángulo grande ($60^\\circ$ aquí) no tiene solución elemental: $\\ddot\\theta=-(g/L)\\sin\\theta$ lleva a una integral elíptica. La conservación de la energía permite hallar la rapidez en el punto más bajo sin resolver la ecuación, pero para ver la trayectoria completa y comprobar que la energía de verdad se conserva hay que integrar numéricamente. El objetivo del método aquí es doble: obtener $\\theta(t)$ y medir cuánto se desvía la energía numérica de la energía exacta (que es constante).",
            "Se comparan tres integradores sobre el estado $y=(\\theta,\\omega)$. Euler, $y_{n+1}=y_n+hf(y_n)$, tiene error global $O(h)$ y no conserva la energía: la deriva crece en cada paso. RK4 usa cuatro pendientes con pesos $1,2,2,1$ y tiene error $O(h^4)$. Verlet de velocidad actualiza $\\theta_{n+1}=\\theta_n+\\omega_nh+\\tfrac12a_nh^2$ y $\\omega_{n+1}=\\omega_n+\\tfrac12(a_n+a_{n+1})h$; tiene error global $O(h^2)$ pero es simpléctico, de modo que la energía no deriva sino que oscila acotada en torno al valor exacto.",
            f"La limitación de cada uno es distinta: Euler es inservible para integraciones largas (gana energía y el péndulo “se acelera”); RK4 es muy exacto pero su deriva, aunque pequeña, crece lentamente con el tiempo y no está acotada en integraciones muy largas; Verlet es menos exacto por paso pero estructuralmente correcto. Con el paso del código ($n=600$ en $6$ s, $h=0.01$ s) el periodo es de unos $2.5$ s, así que hay unos $250$ pasos por oscilación: suficiente para que RK4 y Verlet queden por debajo de $10^{{-3}}$ en deriva relativa. El resultado es la rapidez máxima $v=\\sqrt{{2gL(1-\\cos\\theta_0)}}={va:.4f}$ m/s en $\\theta=0$.",
            "Es independiente del análisis energético porque la ecuación integrada es la dinámica $\\ddot\\theta=-(g/L)\\sin\\theta$ y la energía solo se evalúa a posteriori como diagnóstico; no se impone. Así, que $K+U$ salga constante no es una suposición sino un resultado, y la magnitud de su deriva mide la calidad de la integración.",
        ],
        numeric_code="""g, L, m, th0 = 9.81, 1.5, 1.0, np.deg2rad(60)     # parametros del pendulo
acc = lambda th: -(g/L)*np.sin(th)                # theta'' = -(g/L) sin(theta), sin aproximar
def f(t, y): return np.array([y[1], acc(y[0])])   # y=(theta, omega)

def energia(th, om):
    K = 0.5*m*(L*om)**2                           # cinetica
    U = m*g*L*(1 - np.cos(th))                    # potencial respecto al punto bajo
    return K + U

def euler(y, h): return y + h*f(0, y)             # error global O(h)
def rk4(y, h):
    k1 = f(0, y); k2 = f(0, y + h/2*k1)           # cuatro pendientes
    k3 = f(0, y + h/2*k2); k4 = f(0, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)        # error global O(h^4)
def verlet(y, h):
    th, om = y; a = acc(th)
    th2 = th + om*h + 0.5*a*h*h                   # posicion con aceleracion actual
    om2 = om + 0.5*(a + acc(th2))*h               # velocidad con promedio de aceleraciones
    return np.array([th2, om2])                   # simplectico: energia acotada

n, T = 600, 6.0; h = T/n; y = np.array([th0, 0.0]) # 600 pasos de 0.01 s
E0 = energia(*y); deriva = 0.0
for i in range(n):
    y = rk4(y, h)                                 # cambiar por euler o verlet para comparar
    deriva = max(deriva, abs(energia(*y) - E0)/E0)   # deriva relativa maxima
""",
        numeric_converge=[
            ("Deriva de energía contra n", f"Se ejecutó `energy_drift` con los tres integradores durante $6$ s variando el número de pasos. Se reporta $\\max_t|E-E_0|/E_0$ (valor del código en $n=600$: la fila central).",
             [tab]),
            ("Órdenes observados", f"Al duplicar $n$ la deriva de Euler baja por un factor cercano a $2$ (pendiente media {mean_sl['euler']:.2f}, orden $1$), la de Verlet por $4$ (pendiente media {mean_sl['verlet']:.2f}, orden $2$) y la de RK4 por $16$ o más (pendiente media {mean_sl['rk4']:.2f}, orden $4$; la pendiente decrece al acercarse al piso de redondeo). Con $n=600$ la deriva de Euler (${sci(dr['euler'][2],2)}$) es ${sci(dr['euler'][2]/dr['rk4'][2],2)}$ veces la de RK4 (${sci(dr['rk4'][2],2)}$). La fuente de error es de truncamiento; el redondeo acumulado ($\\sim n\\varepsilon_{{mq}}={sci(600*EPS,1)}$) solo domina a RK4 con pasos muy finos.",
             [rf"\varepsilon_E\propto h^{{{mean_sl['euler']:.2f}}},\qquad \varepsilon_V\propto h^{{{mean_sl['verlet']:.2f}}},\qquad \varepsilon_{{RK4}}\propto h^{{{mean_sl['rk4']:.2f}}}"]),
            ("Rapidez máxima y criterio de confianza", f"La rapidez máxima se lee como $L\\max|\\omega|$ sobre la malla; como el instante $\\theta=0$ no cae en un nodo, el muestreo añade un error que depende de cuánto se acerque algún nodo a $\\theta=0$, por lo que no es monótono (en la tabla $n=6000$ es peor que $n=2400$); el código usa $n=6000$. Criterio de confianza: la deriva de energía es menor que la tolerancia (por ejemplo $10^{{-6}}$), y la rapidez máxima no cambia en la cifra deseada al duplicar $n$.",
             [vtab]),
        ],
    )


# =================================================================== SpringFriction
def _springfric() -> dict:
    an = c8.turning_points_analytic()
    dts = (2e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 1e-4)
    errs, tfin, npas, nturn = [], [], [], []
    for dt in dts:
        ts, xs, vs, tu = c8.spring_friction_numeric(dt=dt)
        errs.append(max(abs(a_ - b_) for a_, b_ in zip(an, tu)))
        tfin.append(ts[-1])
        npas.append(len(ts) - 1)
        nturn.append(len(tu))
    sl = _slopes(errs)
    rows = [rf"{sci(d,2)} & {npas[i]} & {sci(errs[i],3)} & {'--' if i==0 else _sl(sl[i-1])}" for i, d in enumerate(dts)]
    tab = _tab("rrrr", r"\Delta t\ (\mathrm{s}) & \text{pasos} & \max_n|x_n^{num}-x_n|\ (\mathrm{m}) & p", rows)
    ts, xs, vs, tu = c8.spring_friction_numeric()
    k, m, mu, g, x0 = c8.KS, c8.M, c8.MU, c8.G, c8.X0
    path = float(np.sum(np.abs(np.diff(tu))))
    wf = mu * m * g * path
    de = 0.5 * k * x0**2 - 0.5 * k * tu[-1] ** 2
    d_ = c8.delta()
    tpar = 2 * np.pi * np.sqrt(m / k)
    return dict(
        numeric_intro=[
            "Con fricción cinética la fuerza $-\\mu_kmg\\,\\mathrm{sgn}(v)$ cambia de signo en cada retorno, así que la ecuación diferencial es diferente en cada semiciclo y la solución global se arma pegando trozos de senoides. El análisis a mano lo hace con la recurrencia $x_{n+1}=x_n-2\\delta$ (con $\\delta=\\mu_kmg/k$), pero una integración numérica permite seguir el movimiento sin suponer esa estructura: se resuelve $m\\ddot x=-kx-d\\,\\mu_kmg$ con una variable de dirección $d$ que se actualiza al detectar cada retorno.",
            "El integrador es RK4, $y_{n+1}=y_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$ para $y=(x,v)$, con error global $O(h^4)$ cuando $f$ es suave. Pero aquí $f$ es discontinua en $v=0$, y RK4 solo tiene su orden dentro de cada semiciclo: el paso en que $v$ cambia de signo se detecta y se resuelve con interpolación lineal, $x_{ret}=x+\\tfrac{v}{v-v_{nuevo}}(x_{nuevo}-x)$, que es $O(h^2)$. La precisión total queda entonces limitada por esa interpolación y por la forma en que se acumula el desfase del tiempo de retorno.",
            f"Los parámetros del código son $\\Delta t=10^{{-4}}$ s y $t_{{max}}=5$ s. El periodo del oscilador sin fricción es $T=2\\pi\\sqrt{{m/k}}={tpar:.4f}$ s, así que $\\Delta t$ da unos {int(tpar/1e-4)} pasos por periodo, mucho más de lo necesario para RK4. El criterio de parada es físico: cuando $v=0$ y $k|x|\\le\\mu_kmg$ el resorte ya no puede vencer la fricción, y la integración termina (aquí tras {nturn[-1]-1} retornos, en $t={ts[-1]:.4f}$ s). Un error típico del método es que el paso cruce $v=0$ y la fricción “empuje” en sentido contrario durante parte del paso, que es justo lo que corrige la detección.",
            f"Es independiente del análisis porque nunca usa la recurrencia ni $\\delta={d_:.5f}$ m para producir los retornos: los obtiene detectando cambios de signo en la integración. La recurrencia sirve después como referencia para medir el error, y además se verifica con un balance de energía distinto, el del trabajo disipado por la fricción.",
        ],
        numeric_code="""k, m, mu, g, x0 = 100.0, 1.0, 0.25, 9.81, 0.20   # resorte, masa, friccion, gravedad, x inicial
fr = mu*m*g                                       # fuerza de friccion cinetica (N)
dt, t, x, v, d = 1e-4, 0.0, x0, 0.0, 0.0
turning = [x0]                                    # lista de retornos
while t < 5.0:
    if v == 0.0 and abs(k*x) <= fr:               # el resorte no vence la friccion: parar
        break
    if v == 0.0:
        d = -np.sign(x)                           # al partir del reposo el resorte empuja hacia 0
    f = lambda t_, y: np.array([y[1], (-k*y[0] - d*fr)/m])   # m x'' = -k x - d mu m g
    y = np.array([x, v])
    k1 = f(t, y); k2 = f(t, y + dt/2*k1)          # cuatro pendientes de RK4
    k3 = f(t, y + dt/2*k2); k4 = f(t, y + dt*k3)
    yn = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    if v != 0.0 and np.sign(yn[1]) != np.sign(v): # la velocidad cambio de signo: retorno
        frac = v/(v - yn[1])                      # fraccion del paso hasta v = 0
        x = x + frac*(yn[0] - x); v = 0.0         # interpolacion lineal del retorno
        t += frac*dt; turning.append(x)
        continue
    x, v = yn; t += dt                            # paso normal
# turning[n] se compara con x_{n+1} = |x_n| - 2*delta (referencia analitica)
""",
        numeric_converge=[
            ("Error de los retornos contra el paso", f"Se ejecutó `spring_friction_numeric` con distintos $\\Delta t$ y se midió el mayor desvío de los retornos respecto de la recurrencia analítica $x_{{n+1}}=x_n-2\\delta$ ({len(an)} puntos, $\\delta={d_:.5f}$ m). En todos los casos se detectan {nturn[-1]} retornos (incluido el inicial).",
             [tab]),
            ("Lectura del estudio", f"El error cae unas {errs[0]/errs[-2]:.0f} veces al pasar de $\\Delta t=2\\times10^{{-2}}$ s a $10^{{-3}}$ s, pero no con una pendiente limpia (observadas entre {_sl(min(sl))} y {_sl(max(sl))}): la fricción discontinua y la interpolación lineal del retorno rompen el orden $4$ de RK4 y el error fluctúa según en qué punto del paso cae cada retorno. Con el valor del código, $\\Delta t=10^{{-4}}$ s, el error máximo es ${sci(errs[-1],2)}$ m, unas ${sci(errs[-1]/d_,1)}$ veces $\\delta$. La fuente dominante es la detección del retorno (truncamiento de la interpolación); el redondeo es despreciable.",
             [rf"\max_n|x_n^{{num}}-x_n| = {sci(errs[-1],2)}\ \mathrm{{m}}\ \ (\Delta t=10^{{-4}}\ \mathrm{{s}}),\qquad \dfrac{{\varepsilon}}{{\delta}} = {sci(errs[-1]/d_,2)}"]),
            ("Verificación cruzada por energía y criterio de confianza", f"Una prueba independiente de la recurrencia es el balance de energía: la pérdida de energía elástica debe ser igual al trabajo de la fricción, $\\mu_kmg$ por la distancia total recorrida (suma de $|\\Delta x|$ entre retornos). Criterio: el balance cierra con un residuo mucho menor que la energía perdida, y los retornos no cambian en la cifra deseada al reducir $\\Delta t$ a la mitad.",
             [rf"\Delta E_{{el}} = \tfrac12k(x_0^2-x_f^2) = {de:.6f}\ \mathrm{{J}},\qquad W_f = \mu_kmg\sum|\Delta x| = {wf:.6f}\ \mathrm{{J}}",
              rf"|\Delta E_{{el}}-W_f| = {sci(abs(de-wf),2)}\ \mathrm{{J}},\qquad \dfrac{{|\Delta E_{{el}}-W_f|}}{{\Delta E_{{el}}}} = {sci(abs(de-wf)/de,2)}"]),
        ],
    )


DETAIL = {
    "StaticEquilibrium": _static(),
    "InclinePulley": _pulley(),
    "FrictionIncline": _friction(),
    "DragFall": _drag(),
    "SpringWork": _spring_work(),
    "WorkEnergyTheorem": _wet(),
    "PendulumEnergy": _pend(),
    "SpringFriction": _springfric(),
}
