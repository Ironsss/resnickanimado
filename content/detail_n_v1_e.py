"""Explicacion del metodo numerico (intro, codigo, convergencia) de las escenas VenturiTube ... BeatsAndPipes.

Todas las cifras de las tablas de convergencia salen de ejecutar las funciones de `physics/`.
"""
from __future__ import annotations

import warnings

import numpy as np

from content.texfmt import fx, sci
from physics import ch17_fluid_dynamics as c17
from physics import ch18_oscillations as c18
from physics import ch19_waves as c19
from physics import ch20_sound as c20
from physics.integrators import euler_step, integrate, rk4_step

warnings.filterwarnings("ignore")


def _arr(cols: str, head: str, rows: list[str]) -> str:
    return r"\begin{array}{" + cols + "} " + head + r" \\ \hline " + r" \\ ".join(rows) + r" \end{array}"


def _order(xs, es) -> float:
    """Pendiente (positiva) de log(error) frente a log(1/x): orden observado cuando x es el numero de pasos."""
    return float(-np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(es, float)), 1)[0])


# =============================================================== VenturiTube
def _venturi() -> dict:
    a1, a2 = c17.areas()
    rho = c17.RHO
    dp = c17.P1 - c17.venturi_analytic()["p2"]
    qa = c17.flow_from_pressure_drop_analytic(dp)
    qn = c17.flow_from_pressure_drop_numeric(dp)
    f = lambda q: 0.5 * rho * ((q / a2) ** 2 - (q / a1) ** 2) - dp  # noqa: E731

    def bis(n):
        lo, hi = 0.0, 1.0
        for _ in range(n):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    ks = [5, 10, 20, 30, 40, 53, 80]
    errs = [abs(bis(k) - qa) for k in ks]
    rows = [f"{k} & {sci(bis(k), 8)} & {sci(e, 3)} & {sci(2.0 ** -(k + 1), 3)}" for k, e in zip(ks, errs)]
    tab = _arr("c|c|c|c", r"k & Q_k\ [\mathrm{m^3/s}] & |Q_k - Q_{ana}| & 2^{-(k+1)}", rows)
    # tolerancia -> iteraciones
    tols = [1e-3, 1e-6, 1e-9, 1e-12]
    its = [int(np.ceil(np.log2(1.0 / t))) for t in tols]
    rows2 = [f"{sci(t, 1)} & {n} & {sci(abs(bis(n) - qa), 3)}" for t, n in zip(tols, its)]
    tab2 = _arr("c|c|c", r"\text{tolerancia } \tau & k = \lceil\log_2(1/\tau)\rceil & |Q_k - Q_{ana}|", rows2)
    # sensibilidad al dato dp
    qp = c17.flow_from_pressure_drop_numeric(dp * 1.01)
    sens = (qp - qn) / qn
    # cota de raiz (Q* ~ sqrt(dp)): condicionamiento
    return dict(
        numeric_intro=[
            "El problema inverso del tubo de Venturi pide el caudal $Q$ a partir de la caída de presión medida $\\Delta p = p_1 - p_2$. Combinando continuidad ($v = Q/A$) con Bernoulli, $Q$ queda dentro de una ecuación $f(Q)=0$. Aquí esa ecuación es cuadrática y tiene solución cerrada, pero el método numérico no la usa: trata $f$ como una caja negra que solo se puede evaluar, que es lo que se haría si hubiera pérdidas por fricción o un coeficiente de descarga que dependiera de $Q$ y no hubiera fórmula.",
            "El método es la bisección. Si $f$ es continua y $f(a)<0<f(b)$, el teorema del valor intermedio garantiza una raíz en $[a,b]$. Se calcula el punto medio $m=(a+b)/2$ y se conserva la mitad donde $f$ cambia de signo: si $f(m)<0$ entonces $a \\leftarrow m$, y si no $b \\leftarrow m$. Para este problema $f(Q)=\\tfrac12\\rho\\,Q^2\\,(1/A_2^2 - 1/A_1^2) - \\Delta p$, que crece con $Q$ y por eso tiene una sola raíz positiva.",
            "Es correcto porque el invariante «la raíz está dentro de $[a_k,b_k]$» se conserva en cada paso, y la longitud se reduce exactamente a la mitad: $b_k - a_k = (b_0-a_0)/2^k$. Tomando el punto medio como estimación, el error cumple $|Q_k - Q^*|\\le (b_0-a_0)/2^{k+1}$. La convergencia es lineal (orden 1, razón $1/2$): cada iteración gana un dígito binario, es decir unos $0.30$ dígitos decimales. Es lenta frente a Newton, que duplica los dígitos correctos por paso, pero no necesita derivada y nunca diverge. Falla si $f$ no cambia de signo (raíz doble) o si es discontinua.",
            f"Los parámetros se escogen así. El intervalo $[0,\\,1]$ m³/s encierra el caudal real con holgura ($f(0)=-\\Delta p<0$ y $f(1)>0$). Para una tolerancia $\\tau$ bastan $k=\\lceil\\log_2((b_0-a_0)/\\tau)\\rceil$ iteraciones; el código usa 80, que reduce el intervalo a $2^{{-80}}$, mucho menos que el redondeo de la aritmética de 64 bits ($\\approx 10^{{-17}}$ relativo), de modo que las últimas ~25 iteraciones ya no cambian nada. Con $\\Delta p = {sci(dp, 5)}\\ \\mathrm{{Pa}}$ del enunciado se obtiene $Q = {sci(qn, 8)}\\ \\mathrm{{m^3/s}}$.",
            "Es independiente de la solución analítica porque esta despeja $Q = A_2\\sqrt{2\\Delta p/[\\rho(1-(A_2/A_1)^2)]}$ de forma algebraica, mientras que la bisección jamás invierte nada: solo evalúa $f$ en puntos y compara signos. Que ambas coincidan a ~16 cifras comprueba el álgebra, aunque no la física: las dos parten del mismo modelo (Bernoulli más continuidad, fluido ideal).",
        ],
        numeric_code="""def caudal_desde_dp(dp, rho=1000.0, d1=0.20, d2=0.10):
    a1, a2 = pi*d1**2/4, pi*d2**2/4        # areas de la seccion ancha y de la garganta
    f = lambda q: 0.5*rho*((q/a2)**2 - (q/a1)**2) - dp   # f(Q) = p1 - p2(Q) - dp (Bernoulli + continuidad)
    lo, hi = 0.0, 1.0                      # f(lo) < 0 < f(hi): la raiz esta encerrada
    for _ in range(80):                    # 80 bisecciones: intervalo final = 1/2**80
        mid = 0.5*(lo + hi)                # punto medio del intervalo actual
        if f(mid) < 0:                     # f negativa: la raiz esta a la derecha
            lo = mid                       # descartar la mitad izquierda
        else:                              # f >= 0: la raiz esta a la izquierda
            hi = mid                       # descartar la mitad derecha
    return 0.5*(lo + hi)                   # estimacion = centro del ultimo intervalo

dp = P1 - p2_analitico                     # caida de presion del enunciado [Pa]
Q = caudal_desde_dp(dp)                    # caudal numerico [m^3/s]
Q_ref = a2*sqrt(2*dp/(rho*(1 - (a2/a1)**2)))   # formula cerrada, solo para comparar
error = abs(Q - Q_ref)                     # discrepancia entre ambos metodos""",
        numeric_converge=[
            ("Error frente al número de bisecciones", "Se ejecuta la bisección con $k$ iteraciones y se compara con la fórmula cerrada. El error baja casi exactamente como la cota $2^{-(k+1)}$ hasta que topa con el redondeo.",
             [tab, rf"Q_{{ana}} = {sci(qa, 10)}\ \mathrm{{m^3/s}}"]),
            ("Orden observado y suelo de redondeo", f"Entre $k=10$ y $k=40$ el error cae un factor $2^{{30}}\\approx 10^9$, es decir {fx(np.log2(errs[1] / errs[4]) / 30, 2)} bits por iteración (teoría: 1, convergencia lineal con razón $1/2$). A partir de $k\\approx 53$ el error ya es del tamaño de la unidad de redondeo de 64 bits ($\\approx 10^{{-17}}\\ \\mathrm{{m^3/s}}$) y más iteraciones no mejoran nada: es la fuente de error de redondeo.",
             [rf"\varepsilon_{{k=40}}/\varepsilon_{{k=10}} = {sci(errs[4] / errs[1], 3)},\qquad \varepsilon_{{k=80}} = {sci(errs[6], 2)}"]),
            ("Elegir las iteraciones según una tolerancia", "Como la cota es conocida de antemano, el número de iteraciones sale de la tolerancia deseada, sin mirar el resultado:",
             [tab2]),
            ("Sensibilidad al dato y criterio de confianza", f"La bisección es robusta, pero el resultado nunca es mejor que los datos. Como $Q\\propto\\sqrt{{\\Delta p}}$, un error relativo $\\delta$ en la presión produce $\\delta/2$ en el caudal: con $\\Delta p$ aumentado 1 % el caudal cambia {fx(sens * 100, 4)} %. Criterio práctico: el resultado es confiable cuando el intervalo final es menor que la incertidumbre del dato ($\\tau \\ll \\delta Q$) y $f$ cambia de signo en el intervalo inicial; aquí el error de método ($10^{{-17}}$) es despreciable frente a la incertidumbre de cualquier manómetro real.",
             [rf"\dfrac{{\delta Q}}{{Q}} = \dfrac12\dfrac{{\delta(\Delta p)}}{{\Delta p}},\qquad \dfrac{{Q(1.01\,\Delta p)-Q}}{{Q}} = {fx(sens, 6)}"]),
        ],
    )


# ============================================================= TorricelliDrain
def _torricelli() -> dict:
    tv = c17.drain_time_analytic()
    ns = [100, 200, 400, 800]
    te = 0.9 * tv

    def err_h(n, tend):
        t, y = c17.height_numeric(n, t_end=tend)
        return float(np.max(np.abs(y[:, 0] - c17.height_analytic(t))))

    e_clean = [err_h(n, te) for n in ns]
    e_full = [err_h(n, 1.05 * tv) for n in ns]
    p_clean = _order(ns, e_clean)
    rows = [f"{n} & {fx(te / n, 3)} & {sci(a, 3)} & {sci(b, 3)}" for n, a, b in zip(ns, e_clean, e_full)]
    tab = _arr("c|c|c|c", r"n & \Delta t\ [\mathrm{s}] & \max|h-h_{ana}|\ (t\le 0.9\,t_{vac}) & \max|h-h_{ana}|\ (t\le 1.05\,t_{vac})", rows)
    nd = [500, 1000, 2000, 5000, 20000]
    ed = [abs(c17.drain_time_numeric(n) - tv) for n in nd]
    rows2 = [f"{n} & {sci(e, 3)}" for n, e in zip(nd, ed)]
    tab2 = _arr("c|c", r"n & |t_{vac,num} - t_{vac,ana}|\ [\mathrm{s}]", rows2)
    p_d = _order(nd[:3], ed[:3])
    t, y = c17.height_numeric(4000)
    dt0 = t[1] - t[0]
    # Euler para comparar al mismo n
    f = lambda _t, yy: np.array([-(c17.A_HOLE / c17.A_TANK) * np.sqrt(2 * c17.G * max(yy[0], 0.0))])  # noqa: E731
    te_e, ye = integrate(f, np.array([c17.H0]), 0.0, te, 400, euler_step)
    ee = float(np.max(np.abs(ye[:, 0] - c17.height_analytic(te_e))))
    return dict(
        numeric_intro=[
            "El vaciado de un tanque con un orificio obedece $\\dfrac{dh}{dt} = -\\dfrac{a}{A}\\sqrt{2gh}$: la rapidez de salida de Torricelli, $v=\\sqrt{2gh}$, multiplicada por la relación de áreas. Esta ecuación es separable y tiene solución cerrada, pero solo porque el tanque es cilíndrico. Si la sección $A(h)$ cambiara con la altura (un cono, una esfera) la integral seguiría siendo una ecuación diferencial perfectamente escribible, y el método numérico la resolvería sin cambios; por eso se aprende con este caso donde sí hay respuesta exacta para comparar.",
            "El método es Runge-Kutta de cuarto orden (RK4). Para $y' = f(t,y)$ con paso $\\Delta t$ se calculan cuatro pendientes, $k_1 = f(t_n,y_n)$, $k_2 = f(t_n+\\tfrac{\\Delta t}2,\\,y_n+\\tfrac{\\Delta t}2k_1)$, $k_3 = f(t_n+\\tfrac{\\Delta t}2,\\,y_n+\\tfrac{\\Delta t}2k_2)$ y $k_4 = f(t_n+\\Delta t,\\,y_n+\\Delta t\\,k_3)$, y se avanza con el promedio ponderado $y_{n+1} = y_n + \\tfrac{\\Delta t}{6}(k_1+2k_2+2k_3+k_4)$. Equivale a la regla de Simpson aplicada a la integral de $f$ a lo largo del paso.",
            "Su desarrollo en serie coincide con el de Taylor de la solución hasta $\\Delta t^4$, así que el error local es $O(\\Delta t^5)$ y el error global tras recorrer un tiempo fijo es $O(\\Delta t^4)$: dividir el paso a la mitad reduce el error unas 16 veces. Esa garantía exige que $f$ sea suficientemente suave, y aquí no lo es en el punto final: $f\\propto\\sqrt{h}$ tiene derivada infinita en $h=0$, por lo que RK4 pierde el orden cuando $h\\to0$. Además se debe evitar la raíz de un número negativo, por eso el código usa $\\max(h,0)$.",
            f"Los parámetros se escogen con esa limitación en mente. La integración llega hasta $1.05\\,t_{{vac}}$ con $n=4000$ pasos ($\\Delta t = {fx(dt0, 3)}\\ \\mathrm{{s}}$), lo que da altura con error del orden de $10^{{-7}}\\ \\mathrm{{m}}$ en toda la curva. El tiempo de vaciado no se lee del último paso: con $n=20000$ se detecta el primer paso con $h<1\\ \\mathrm{{mm}}$ y desde ahí se aplica la solución local $h=(\\kappa\\,(t_e-t))^2$ con $\\kappa = (a/A)\\sqrt{{g/2}}$, que da $t_e = t+\\sqrt h/\\kappa$. El tanque tarda $t_{{vac}} = {fx(tv, 2)}\\ \\mathrm{{s}}$ ($\\approx {fx(tv / 60, 1)}$ min).",
            "Es independiente del resultado analítico porque este se obtiene integrando $dh/\\sqrt h$ en forma exacta, mientras que RK4 solo evalúa la derivada en puntos intermedios y no conoce la forma $h(t)=(\\sqrt{h_0}-\\kappa t)^2$. Que reproduzca esa parábola a $10^{-7}$ m confirma tanto la integración como la derivación de $t_{vac}$.",
        ],
        numeric_code="""def altura_numerica(n=4000, h0=1.5, a=2e-4, A=0.5, g=9.81):
    t_vac = (A/a)*sqrt(2*h0/g)                          # tiempo de vaciado analitico (solo fija t_final)
    f = lambda t, y: [-(a/A)*sqrt(2*g*max(y[0], 0))]    # dh/dt = -(a/A) sqrt(2 g h); max evita raiz de negativo
    y, t, dt = [h0], 0.0, 1.05*t_vac/n                  # paso fijo hasta 1.05 t_vac
    for i in range(n):
        k1 = f(t, y)                                    # pendiente al inicio del paso
        k2 = f(t + dt/2, y + dt/2*k1)                   # pendiente en el punto medio (con k1)
        k3 = f(t + dt/2, y + dt/2*k2)                   # pendiente en el punto medio (con k2)
        k4 = f(t + dt, y + dt*k3)                       # pendiente al final del paso
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)            # promedio ponderado 1,2,2,1
        t += dt
        guardar(t, y)
    return t_hist, h_hist

def tiempo_vaciado(n=20000):
    t, h = altura_numerica(n)
    i = primer_indice(h < 1e-3)                         # primer paso con h < 1 mm
    kappa = (a/A)*sqrt(g/2)                             # cerca de h=0: h = (kappa (te - t))^2
    return t[i-1] + sqrt(h[i-1])/kappa                  # te = t + sqrt(h)/kappa""",
        numeric_converge=[
            ("Orden de RK4 y el efecto de la raíz", f"Se integra con $n$ pasos y se mide el error máximo respecto de la solución exacta. Si se corta antes del final ($t\\le 0.9\\,t_{{vac}}$, donde $f$ es suave) la pendiente observada en escala log-log es {fx(p_clean, 2)}, que confirma el orden 4 teórico. Si se llega hasta $1.05\\,t_{{vac}}$ el error se estanca en ~$10^{{-7}}$ m: la singularidad de $\\sqrt h$ en $h=0$ impide que mejore al refinar.",
             [tab]),
            ("Comparación con Euler", f"Para apreciar el valor de RK4, el método de Euler con 400 pasos (el mismo $\\Delta t$ de la fila $n=400$) en el mismo tramo tiene un error máximo de ${sci(ee, 3)}\\ \\mathrm{{m}}$, frente a ${sci(err_h(400, te), 3)}\\ \\mathrm{{m}}$ de RK4: unas ${sci(ee / err_h(400, te), 2)}$ veces peor, porque Euler solo tiene orden 1.",
             [rf"\varepsilon_{{Euler}}(n=400) = {sci(ee, 3)}\ \mathrm{{m}},\qquad \varepsilon_{{RK4}}(n=400) = {sci(err_h(400, te), 3)}\ \mathrm{{m}}"]),
            ("Error del tiempo de vaciado con la corrección local", f"La estimación de $t_{{vac}}$ usa el último paso antes de llegar a 1 mm y la solución local. El error baja con $n$ con pendiente {fx(p_d, 2)} entre $n=500$ y $n=2000$ (cercana a 4) y llega a ${sci(ed[-1], 2)}\\ \\mathrm{{s}}$ con los $n=20000$ del código, sobre un tiempo de {fx(tv, 1)} s. Sin la corrección, el error sería del orden de $\\Delta t$.",
             [tab2]),
            ("Fuentes de error y criterio de confianza", f"Hay tres: truncamiento de RK4 ($\\propto\\Delta t^4$, dominante lejos del final), la singularidad de la raíz (limita a ~$10^{{-7}}$ m cerca de $h=0$) y redondeo (~$10^{{-16}}$ por paso, acumulado en 20000 pasos queda en ~$10^{{-12}}$, irrelevante). Criterio: el resultado es confiable si al duplicar $n$ el error cae ~16 veces fuera de la zona $h<1$ mm; si no cae, domina la singularidad o el redondeo y hay que cambiar de estrategia, no de paso.",
             [rf"\dfrac{{\varepsilon(n=100)}}{{\varepsilon(n=200)}} = {fx(e_clean[0] / e_clean[1], 2)},\qquad \dfrac{{\varepsilon(n=200)}}{{\varepsilon(n=400)}} = {fx(e_clean[1] / e_clean[2], 2)}"]),
        ],
    )


# ============================================================== SimpleHarmonic
def _shm() -> dict:
    w = c18.omega0()
    T = 2 * np.pi / w
    ns = [25, 50, 100, 200, 400, 800]
    er = []
    ee = []
    for n in ns:
        t, y = c18.shm_numeric(1.0, n)
        er.append(float(np.max(np.abs(y[:, 0] - c18.x_analytic(t)))))
        t2, y2 = integrate(lambda _t, yy: np.array([yy[1], -(c18.K / c18.M) * yy[0]]), np.array([c18.AMP, 0.0]), 0.0, 1.0, n, euler_step)
        ee.append(float(np.max(np.abs(y2[:, 0] - c18.x_analytic(t2)))))
    p_rk = _order(ns, er)
    p_eu = _order(ns, ee)
    rows = [f"{n} & {sci(1.0 / n, 3)} & {w / n:.4f} & {sci(a, 3)} & {sci(b, 3)}" for n, a, b in zip(ns, er, ee)]
    tab = _arr("c|c|c|c|c", r"n & \Delta t\ [\mathrm{s}] & \omega\Delta t & \varepsilon_{RK4} & \varepsilon_{Euler}", rows)
    per_err = abs(c18.period_numeric() - T)
    t_, y_ = c18.shm_numeric(1.0, 2000)
    e2000 = float(np.max(np.abs(y_[:, 0] - c18.x_analytic(t_))))
    p_eu_tail = _order(ns[-3:], ee[-3:])
    # factor de amplificacion de RK4 para z = i w dt
    z = 1j * w * (1.0 / 2000)
    g = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    # periodo vs n con interpolacion (informativo)
    pn = []
    for n in (500, 2000, 8000, 40000):
        t, y = c18.shm_numeric(2.0, n)
        x = y[:, 0]
        idx = [i for i in range(1, len(x)) if x[i - 1] > 0 >= x[i]]
        tc = lambda i: t[i] - x[i] * (t[i] - t[i - 1]) / (x[i] - x[i - 1])  # noqa: E731
        pn.append((n, abs(tc(idx[1]) - tc(idx[0]) - T)))
    rows2 = [f"{n} & {sci(2.0 / n, 3)} & {sci(e, 3)}" for n, e in pn]
    tab2 = _arr("c|c|c", r"n & \Delta t\ [\mathrm{s}] & |T_{num} - T_{ana}|\ [\mathrm{s}]", rows2)
    return dict(
        numeric_intro=[
            f"El oscilador armónico simple, $m\\ddot x = -kx$, se resuelve en papel con $x=A\\cos\\omega t$. El método numérico no usa esa solución: reescribe la ecuación de segundo orden como un sistema de primer orden para el vector $y=(x,v)$, con $y' = (v,\\,-(k/m)x)$, y lo avanza paso a paso. Esto es útil porque el mismo código sirve sin cambios para fuerzas no lineales (el péndulo real, resortes duros), donde ya no existe fórmula con funciones elementales; aquí se usa el caso lineal porque permite medir el error exactamente.",
            "El integrador es RK4: en cada paso $\\Delta t$ se evalúan cuatro pendientes $k_1,\\dots,k_4$ (al inicio, dos en el punto medio y al final) y se combinan como $y_{n+1} = y_n + \\tfrac{\\Delta t}{6}(k_1+2k_2+2k_3+k_4)$. Para una ecuación lineal $y'=\\lambda y$ el paso equivale a multiplicar por $1+z+\\tfrac{z^2}{2}+\\tfrac{z^3}{6}+\\tfrac{z^4}{24}$ con $z=\\lambda\\Delta t$, es decir, los cuatro primeros términos de la serie de $e^{z}$.",
            f"De ahí sale el orden: el error local es $z^5/120 = (\\omega\\Delta t)^5/120$ y el error global tras un tiempo fijo es $O(\\Delta t^4)$. Con $\\lambda = \\pm i\\omega$ el factor de amplificación tiene módulo ligeramente menor que 1 ($|G| = {sci(abs(g), 10)}$ para $\\omega\\Delta t = {fx(w / 2000, 3)}$), por lo que la amplitud se amortigua artificialmente muy poco; Euler, en cambio, la hace crecer ($|G|=\\sqrt{{1+(\\omega\\Delta t)^2}}>1$) y la energía numérica aumenta sin parar. RK4 falla para pasos grandes: si $\\omega\\Delta t$ supera ~2.8 el método es inestable.",
            f"Se escogieron $n=2000$ pasos en $1$ s, de modo que $\\Delta t = {sci(1 / 2000, 2)}\\ \\mathrm{{s}}$ y $\\omega\\Delta t = {fx(w / 2000, 3)}$, es decir unos {fx(T / (1 / 2000), 0)} pasos por período $T={fx(T, 4)}\\ \\mathrm{{s}}$. Para medir el período se integran $2$ s con $n=40000$ y se buscan dos cruces descendentes por cero con interpolación lineal; la diferencia entre ambos es el período, sin usar $T=2\\pi\\sqrt{{m/k}}$.",
            "Es independiente del analítico porque nunca se usa $\\omega$ ni el coseno: el integrador solo ve $k/m$ dentro de la derivada. Que el período medido coincida con $2\\pi\\sqrt{m/k}$ y que la trayectoria reproduzca $A\\cos\\omega t$ valida la integración numérica; el mismo código aplicado al péndulo da el período exacto, donde el analítico simple ya no vale.",
        ],
        numeric_code="""def shm_numerico(t_end=1.0, n=2000, amp=0.08, k=200.0, m=0.5):
    f = lambda t, y: [y[1], -(k/m)*y[0]]       # y = (x, v): x' = v ; v' = -(k/m) x
    y, dt = [amp, 0.0], t_end/n                # parte del reposo en x = amplitud
    traj = [y]
    for i in range(n):
        t = i*dt
        k1 = f(t, y)                           # pendiente al inicio
        k2 = f(t + dt/2, y + dt/2*k1)          # pendiente en el punto medio
        k3 = f(t + dt/2, y + dt/2*k2)          # otra en el punto medio, corregida
        k4 = f(t + dt, y + dt*k3)              # pendiente al final
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)   # promedio ponderado 1-2-2-1
        traj.append(y)
    return traj

def periodo(amp=0.08, k=200.0, m=0.5):
    t, x = shm_numerico(2.0, 40000)            # 2 s con dt = 5e-5 s
    cruces = []
    for i in range(1, len(x)):
        if x[i-1] > 0 >= x[i]:                 # cruce descendente por cero
            cruces.append(t[i] - x[i]*(t[i]-t[i-1])/(x[i]-x[i-1]))   # interpolacion lineal
    return cruces[1] - cruces[0]               # un periodo completo""",
        numeric_converge=[
            ("Error global frente al número de pasos", f"Se integra 1 s con $n$ pasos y se toma el máximo de $|x_k - A\\cos\\omega t_k|$. La pendiente en escala log-log es {fx(p_rk, 2)} para RK4 (teoría: 4) y {fx(p_eu, 2)} para Euler en todo el rango. Esta última cifra engaña: con pasos grandes Euler está aún fuera de su régimen asintótico (la amplitud numérica crece como $(1+\omega^2\Delta t^2)^{{n/2}}$ y el error es enorme); entre las tres últimas filas la pendiente baja a {fx(p_eu_tail, 2)}, camino del valor teórico 1. RK4 gasta cuatro evaluaciones de $f$ por paso y Euler una, pero la ganancia en error supera por mucho ese costo.",
             [tab]),
            ("Qué significa el orden en la práctica", f"Con $n=2000$ (el valor del código) RK4 llega a ${sci(e2000, 2)}\ \mathrm{{m}}$; cada duplicación de $n$ divide el error por ~{fx(er[0] / er[1], 0)}-{fx(er[3] / er[4], 0)}. Para que Euler alcanzara esa precisión haría falta $n\approx{sci(ee[-1] * ns[-1] / e2000, 1)}$ pasos, dado que su error final decrece como $1/n$. La regla para elegir el paso es mantener $\\omega\\Delta t\\ll1$; aquí $\\omega\\Delta t={fx(w / 2000, 3)}$.",
             [rf"\dfrac{{\varepsilon(n=100)}}{{\varepsilon(n=200)}} = {fx(er[2] / er[3], 1)},\qquad \dfrac{{\varepsilon(n=200)}}{{\varepsilon(n=400)}} = {fx(er[3] / er[4], 1)}"]),
            ("Error del período medido", f"El período numérico usa $n=40000$ y sale con un error de ${sci(per_err, 2)}\\ \\mathrm{{s}}$ sobre $T={fx(T, 4)}\\ \\mathrm{{s}}$. La tabla muestra que baja con $n$ más rápido que $\\Delta t^2$: la interpolación lineal de cada cruce comete un error $O(\\Delta t^2)$, pero como la señal es periódica ambos cruces tienen el mismo error y se cancelan casi por completo al restarlos; queda el error de fase de RK4 ($O(\\Delta t^4)$) y, al final, el redondeo.",
             [tab2]),
            ("Fuentes de error y criterio de confianza", "Truncamiento de RK4 (dominante hasta $n\\sim10^3$), interpolación lineal del cruce por cero y redondeo (~$10^{-16}$ por paso, ~$10^{-12}$ tras $4\\times10^4$ pasos). Criterio: el resultado es confiable si $\\omega\\Delta t\\lesssim0.05$ y si al duplicar $n$ el error baja ~16 veces; si además la energía $\\tfrac12 mv^2+\\tfrac12kx^2$ se mantiene constante a varias cifras, no hay amortiguamiento numérico apreciable.",
             [rf"|G|-1 = {sci(abs(g) - 1, 3)}\ \ (\omega\Delta t = {fx(w / 2000, 3)})"]),
        ],
    )


# ============================================================= DrivenResonance
def _driven() -> dict:
    wd = 9.0
    aa = float(c18.driven_amplitude_analytic(wd))
    m, k, b, f0 = c18.M_D, c18.K_D, c18.B_D, c18.F0

    def amp(n, tend=24.0, w=wd):
        f = lambda t, y: np.array([y[1], (f0 * np.cos(w * t) - b * y[1] - k * y[0]) / m])  # noqa: E731
        t, y = integrate(f, np.zeros(2), 0.0, tend, n, rk4_step)
        tail = t > tend - max(3 * 2 * np.pi / w, 4.0)
        return float(np.max(np.abs(y[tail, 0])))

    ns = [1000, 2000, 4000, 8000, 16000, 32000]
    en = [abs(amp(n) - aa) for n in ns]
    rows = [f"{n} & {sci(24.0 / n, 3)} & {sci(a, 3)}" for n, a in zip(ns, en)]
    tab = _arr("c|c|c", r"n & \Delta t\ [\mathrm{s}] & |A_{num} - A_{ana}|\ [\mathrm{m}]", rows)
    tes = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    et = [abs(amp(64000, te) - aa) for te in tes]
    rows2 = [f"{te:.0f} & {sci(np.exp(-te), 2)} & {sci(e, 3)}" for te, e in zip(tes, et)]
    tab2 = _arr("c|c|c", r"t_{fin}\ [\mathrm{s}] & e^{-t_{fin}} & |A_{num} - A_{ana}|\ [\mathrm{m}]", rows2)
    wsc = [6.0, 9.0, 12.0]
    rows3 = [f"{w:.0f} & {sci(float(c18.driven_amplitude_analytic(w)), 6)} & {sci(c18.driven_amplitude_numeric(w), 6)} & {sci(abs(c18.driven_amplitude_numeric(w) - float(c18.driven_amplitude_analytic(w))), 2)}" for w in wsc]
    tab3 = _arr("c|c|c|c", r"\omega\ [\mathrm{rad/s}] & A_{ana}\ [\mathrm{m}] & A_{num}\ [\mathrm{m}] & |\Delta A|", rows3)
    dt0 = 24.0 / 20000
    bound = aa * wd**2 * (dt0 / 2) ** 2 / 2
    return dict(
        numeric_intro=[
            "El oscilador forzado y amortiguado $m\\ddot x + b\\dot x + kx = F_0\\cos\\omega t$ tiene solución analítica, pero esa solución se obtiene suponiendo que el transitorio ya desapareció y que la respuesta es $x_p = A\\cos(\\omega t-\\varphi)$, con $A = (F_0/m)/\\sqrt{(\\omega_0^2-\\omega^2)^2+(\\gamma\\omega)^2}$. La simulación numérica no supone nada: parte del reposo, deja evolucionar el sistema completo (transitorio incluido) y mide qué amplitud adquiere. Por eso sirve también para casos con fuerzas no lineales o con ruido, donde no hay fórmula.",
            "Se usa RK4 sobre $y=(x,v)$, con $y' = \\big(v,\\ [F_0\\cos\\omega t - bv - kx]/m\\big)$. La diferencia con el caso autónomo es que ahora $f$ depende explícitamente del tiempo, así que $k_2$ y $k_3$ se evalúan en $t+\\Delta t/2$ y $k_4$ en $t+\\Delta t$. El paso combina las pendientes con pesos $1,2,2,1$, con error local $O(\\Delta t^5)$ y error global $O(\\Delta t^4)$.",
            f"Aparecen dos fuentes de error además del integrador. La primera es el transitorio: la solución completa es $x = x_p + C e^{{-\\gamma t/2}}\\cos(\\omega_d t+\\phi)$, y con $\\gamma=b/m=2\\ \\mathrm{{s^{{-1}}}}$ el residuo decae como $e^{{-t}}$; por eso se simula hasta $t=24\\ \\mathrm{{s}}$ ($e^{{-24}}={sci(np.exp(-24), 2)}$). La segunda es que la amplitud se mide como el máximo de muestras discretas, que subestima el pico verdadero hasta $\\approx A\\,\\omega^2(\\Delta t/2)^2/2={sci(bound, 2)}\\ \\mathrm{{m}}$ en el peor caso.",
            f"Los parámetros: al menos $n=20000$ pasos en $24$ s ($\\Delta t \\le {sci(dt0, 3)}\\ \\mathrm{{s}}$, unos {fx(2 * np.pi / wd / dt0, 0)} pasos por ciclo de la fuerza a $\\omega=9$); la ventana de medida abarca el mayor entre tres períodos de la fuerza y $4$ s para cubrir varios ciclos completos. El resultado es la curva de resonancia: con $\\omega_0=10$ y $\\gamma=2$ el máximo está en $\\omega_r=\\sqrt{{\\omega_0^2-\\gamma^2/2}} = {fx(c18.resonance_frequency_analytic(), 3)}\\ \\mathrm{{rad/s}}$, algo por debajo de $\\omega_0$.",
            "Es independiente porque nunca se usa la fórmula de $A(\\omega)$ ni se calcula el desfase: se integra la ecuación de movimiento tal como está. Coincidir con la expresión analítica a 7 u 8 cifras confirma tanto la fórmula de resonancia como la simulación. La limitación es el costo: cada punto de la curva exige una simulación entera, mientras que la fórmula se evalúa al instante.",
        ],
        numeric_code="""def amplitud_estacionaria(w, m=1.0, k=100.0, b=2.0, f0=5.0):
    t_fin = 24.0                                   # e^{-t_fin} = 4e-11: el transitorio ya murio
    n = max(20000, ...)                            # al menos 20000 pasos (dt <= 1.2e-3 s)
    dt = t_fin/n
    f = lambda t, y: [y[1], (f0*cos(w*t) - b*y[1] - k*y[0])/m]   # x' = v ; m v' = F - b v - k x
    y, x = [0.0, 0.0], []                          # parte del reposo
    for i in range(n):
        t = i*dt
        k1 = f(t, y)                               # pendiente al inicio (fuerza en t)
        k2 = f(t + dt/2, y + dt/2*k1)              # fuerza evaluada en t + dt/2
        k3 = f(t + dt/2, y + dt/2*k2)
        k4 = f(t + dt, y + dt*k3)                  # fuerza evaluada en t + dt
        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        x.append(y[0])
    ventana = t > t_fin - max(3*2*pi/w, 4.0)       # varios ciclos al final
    return max(abs(x[ventana]))                    # amplitud = maximo de |x| en la ventana

curva = [amplitud_estacionaria(w) for w in omegas]    # una simulacion por cada frecuencia""",
        numeric_converge=[
            ("Error frente al número de pasos", f"Se repite la simulación con distinto $n$ a $\\omega=9\\ \\mathrm{{rad/s}}$ y se compara con $A_{{ana}} = {sci(aa, 8)}\\ \\mathrm{{m}}$. El error baja de ${sci(en[0], 2)}$ a ${sci(en[-1], 2)}\\ \\mathrm{{m}}$, pero de forma irregular: al medir un máximo de muestras, el resultado depende de lo cerca que caiga una muestra del pico, y eso añade un ruido de hasta ~$10^{{-6}}$ m. En promedio la caída es de ~16 veces por duplicación, como corresponde a RK4.",
             [tab]),
            ("Error por transitorio incompleto", f"Con $n=64000$ (para que el integrador no domine) se varía el tiempo final. El error decrece siguiendo la tendencia de $e^{{-t_{{fin}}}}$ (con una constante que depende de la fase) hasta aplanarse en ~$10^{{-9}}$ m, donde domina el muestreo del máximo. Con $t_{{fin}}=24$ s el transitorio aporta menos de ${sci(np.exp(-24), 1)}$ relativo, despreciable.",
             [tab2]),
            ("Resultado del código para tres frecuencias", "Con los valores del código ($n\\ge20000$, $t_{fin}=24$ s) el acuerdo con la fórmula es de unas 7 cifras significativas en las tres frecuencias, incluida la zona cercana al máximo, donde la sensibilidad a $\\omega$ es mayor:",
             [tab3]),
            ("Fuentes de error y criterio de confianza", f"Las fuentes son: truncamiento de RK4 (~$10^{{-8}}$ m a $n=20000$), transitorio residual (< $10^{{-10}}$), muestreo del pico (hasta ${sci(bound, 1)}$ m como cota) y redondeo (< $10^{{-13}}$). Una frecuencia con amortiguamiento débil requiere más tiempo, porque el transitorio decae como $e^{{-\\gamma t/2}}$ y hay que tomar $t_{{fin}}\\gg 2/\\gamma$. Criterio: el resultado es confiable si al duplicar tanto $n$ como $t_{{fin}}$ la amplitud medida no cambia en las cifras que se reportan.",
             [rf"e^{{-\gamma t_{{fin}}/2}} = e^{{-24}} = {sci(np.exp(-24), 3)}"]),
        ],
    )


# ================================================================== WavePulse
def _pulse() -> dict:
    ns = [100, 200, 400, 800, 1600]
    en = [c19.fdtd_error(0.9, n) for n in ns]
    p = _order(ns, en)
    rows = [f"{n} & {fx(c19.LENGTH / n, 4)} & {sci(e, 3)}" for n, e in zip(ns, en)]
    tab = _arr("c|c|c", r"N & \Delta x\ [\mathrm{m}] & \max|y_i - y_{dAl}|", rows)
    cs = [1.0, 0.9, 0.7, 0.5, 0.3]
    ec = [c19.fdtd_error(c, 800) for c in cs]
    rows2 = [f"{c} & {sci(c * c19.LENGTH / 800 / c19.V, 3)} & {sci(e, 3)}" for c, e in zip(cs, ec)]
    tab2 = _arr("c|c|c", r"C & \Delta t\ [\mathrm{s}] & \max|y_i - y_{dAl}|", rows2)
    ebad = c19.fdtd_error(1.05, 200)
    dx = c19.LENGTH / 800
    dt = 0.9 * dx / c19.V
    return dict(
        numeric_intro=[
            "La ecuación de onda $\\partial_t^2 y = v^2\\partial_x^2 y$ de una cuerda tiene solución de d'Alembert, $y = \\tfrac12[f(x-vt)+f(x+vt)]$, pero solo para una cuerda uniforme e infinita. En cuanto la tensión o la densidad varían, hay fronteras o amortiguamiento, esa fórmula desaparece y se recurre al método de diferencias finitas (FDTD): se reemplazan las derivadas parciales por cocientes de diferencias sobre una malla en $x$ y en $t$ y se avanza la solución en el tiempo.",
            "El esquema usa diferencias centradas en ambas variables, $\\dfrac{y_i^{n+1}-2y_i^n+y_i^{n-1}}{\\Delta t^2} = v^2\\dfrac{y_{i+1}^n-2y_i^n+y_{i-1}^n}{\\Delta x^2}$, y se despeja el valor futuro: $y_i^{n+1} = 2y_i^n - y_i^{n-1} + C^2\\,(y_{i+1}^n - 2y_i^n + y_{i-1}^n)$ con el número de Courant $C = v\\Delta t/\\Delta x$. Como necesita dos niveles de tiempo, el primer paso se calcula con un desarrollo de Taylor y velocidad inicial nula: $y_i^1 = y_i^0 + \\tfrac12C^2(y_{i+1}^0-2y_i^0+y_{i-1}^0)$.",
            "Cada diferencia centrada tiene error de truncamiento $O(\\Delta x^2)$ y $O(\\Delta t^2)$, así que el error global es $O(\\Delta x^2+\\Delta t^2)$: duplicar $N$ reduce el error unas 4 veces. Dos condiciones lo limitan. La estabilidad exige $C\\le1$ (condición de Courant-Friedrichs-Lewy: la información numérica debe viajar al menos tan rápido como la física); con $C>1$ los errores crecen exponencialmente. Y con $C<1$ aparece dispersión numérica: las longitudes de onda cortas viajan más lento que $v$, y el pulso se deforma con una cola oscilante. Con $C=1$ el esquema es exacto en la malla, lo que no es posible en dimensiones superiores.",
            f"Los parámetros: cuerda de $20\\ \\mathrm{{m}}$ con $N=800$ intervalos ($\\Delta x = {fx(dx, 4)}\\ \\mathrm{{m}}$, es decir {fx(c19.SIGMA / dx, 0)} puntos por cada $\\sigma={c19.SIGMA}\\ \\mathrm{{m}}$ del pulso gaussiano) y $C=0.9$, que da $\\Delta t = {sci(dt, 4)}\\ \\mathrm{{s}}$ y {int(round(0.15 / dt))} pasos hasta $t=0.15\\ \\mathrm{{s}}$ ($v={c19.V:.0f}\\ \\mathrm{{m/s}}$, de modo que cada mitad del pulso recorre $6\\ \\mathrm{{m}}$). El pulso es mucho más ancho que la malla, que es la condición para que la dispersión sea pequeña. Los extremos están lejos y no afectan en ese tiempo.",
            "Es independiente de la fórmula de d'Alembert: el esquema solo usa la ecuación diferencial y la condición inicial, nunca la idea de que el pulso se divide en dos mitades que viajan. Esa coincidencia entre dos caminos distintos es la verificación; y como d'Alembert es exacto, el error medido es puramente el del esquema numérico.",
        ],
        numeric_code="""def fdtd(N=800, C=0.9, v=40.0, L=20.0, t_final=0.15):
    x = linspace(0, L, N + 1)                      # malla espacial, dx = L/N
    dx = x[1] - x[0]
    dt = C*dx/v                                    # Courant: C = v dt/dx <= 1 para ser estable
    y_old = pulso_gauss(x)                         # y^0: pulso inicial en x = 10 m
    y = y_old.copy()
    y[1:-1] = y_old[1:-1] + 0.5*C**2*(y_old[2:] - 2*y_old[1:-1] + y_old[:-2])   # y^1 (velocidad nula)
    for paso in range(1, round(t_final/dt)):
        y_new = zeros_like(y)
        y_new[1:-1] = (2*y[1:-1] - y_old[1:-1]     # leapfrog: 2 y^n - y^(n-1)
                       + C**2*(y[2:] - 2*y[1:-1] + y[:-2]))   # + C^2 * laplaciano discreto
        y_old, y = y, y_new                        # avanzar un nivel de tiempo
    return x, y                                    # (los extremos quedan fijos en 0)

x, y = fdtd()
error = max(abs(y - dalembert(x, t_final)))        # max|y_i - d'Alembert| sobre la malla""",
        numeric_converge=[
            ("Error frente al tamaño de la malla (C = 0.9)", f"Se calcula $\\varepsilon=\\max_i|y_i-y_{{dAl}}(x_i,t)|$ en $t=0.15$ s para $N$ creciente manteniendo $C=0.9$. La pendiente log-log da un orden observado de {fx(p, 2)}, que confirma el $O(\\Delta x^2)$ teórico: cada duplicación de $N$ divide el error por ~{fx(en[1] / en[2], 1)}.",
             [tab]),
            ("Efecto del número de Courant", "Con la malla fija en $N=800$ se varía $C$. El error mínimo ocurre en $C=1$, donde el esquema es exacto en la malla y solo queda el redondeo; al alejarse de 1 el término de dispersión numérica (proporcional a $1-C^2$) crece:",
             [tab2]),
            ("Qué pasa si se viola la estabilidad", f"Con $C=1.05$ y $N=200$ el mismo código da un error máximo de ${sci(ebad, 2)}$ (en unidades de la amplitud inicial 1), tras solo {int(round(0.15 / (1.05 * 0.1 / c19.V)))} pasos, y el error crece exponencialmente al seguir avanzando: cada paso amplifica las componentes de longitud de onda más corta por un factor mayor que 1. Por eso en la escena se usa $C=0.9$ y no un valor más cercano al límite sin necesidad.",
             [rf"C = 1.05:\quad \varepsilon = {sci(ebad, 3)}\ \ (\text{{inestable}})"]),
            ("Fuentes de error y criterio de confianza", f"Hay tres: truncamiento espacial y temporal ($O(\\Delta x^2)$, ${sci(en[3], 2)}$ con el $N$ de la escena), dispersión numérica (el mismo origen, visible como cola) y redondeo (~${sci(ec[0], 1)}$ en $C=1$). Criterio: el resultado es confiable si $C\\le1$, el pulso abarca al menos ~10 celdas ($\\sigma/\\Delta x\\gtrsim10$) y al duplicar $N$ el error baja ~4 veces.",
             [rf"\dfrac{{\varepsilon(N=400)}}{{\varepsilon(N=800)}} = {fx(en[2] / en[3], 2)},\qquad \dfrac{{\varepsilon(N=800)}}{{\varepsilon(N=1600)}} = {fx(en[3] / en[4], 2)}"]),
        ],
    )


# ============================================================== StandingModes
def _modes() -> dict:
    L, v = c19.STRING_L, c19.V
    fa = np.array([c19.mode_frequency_analytic(j) for j in range(1, 6)])
    Ns = [25, 50, 100, 200, 400, 800]
    res = []
    for N in Ns:
        fr, _, _ = c19.modes_numeric(N)
        res.append((abs(fr[0] - fa[0]) / fa[0], abs(fr[4] - fa[4]) / fa[4]))
    r1 = [r[0] for r in res]
    r5 = [r[1] for r in res]
    p1, p5 = _order(Ns, r1), _order(Ns, r5)
    pred = lambda N, j: (j * np.pi / (N + 1)) ** 2 / 24  # noqa: E731
    rows = [f"{N} & {sci(a, 3)} & {sci(pred(N, 1), 3)} & {sci(b, 3)} & {sci(pred(N, 5), 3)}" for N, (a, b) in zip(Ns, res)]
    tab = _arr("c|c|c|c|c", r"N & \varepsilon_1 & \text{teoría}_1 & \varepsilon_5 & \text{teoría}_5", rows)
    fr, x, shapes = c19.modes_numeric(400)
    shape_err = float(np.max(np.abs(shapes[0] - np.sin(np.pi * x / L))))
    # numero de condicion aproximado
    cond = 4.0 * 401**2 / np.pi**2
    rows2 = [f"{j+1} & {fx(fa[j], 4)} & {fx(fr[j], 4)} & {sci(fr[j] - fa[j], 3)}" for j in range(5)]
    tab2 = _arr("c|c|c|c", r"n & f_{ana}\ [\mathrm{Hz}] & f_{num}\ [\mathrm{Hz}] & f_{num}-f_{ana}", rows2)
    dx = L / 401
    return dict(
        numeric_intro=[
            "Los modos normales de una cuerda fija en ambos extremos son soluciones de $X'' = -k^2X$ con $X(0)=X(L)=0$, y analíticamente salen $k_n=n\\pi/L$ y $f_n = nv/2L$. El método numérico convierte ese problema de ecuación diferencial en uno de álgebra lineal: se discretiza el intervalo y se buscan autovalores de una matriz. Esto es lo que se hace en la práctica cuando la cuerda no es uniforme ($\\mu(x)$ variable) o cuando se estudian membranas y cavidades de forma arbitraria, donde no hay fórmula.",
            "Se toman $N$ puntos interiores con $\\Delta x = L/(N+1)$ y se aproxima la segunda derivada con $X''_j\\approx (X_{j+1}-2X_j+X_{j-1})/\\Delta x^2$. Como $X_0=X_{N+1}=0$ (extremos fijos), el operador se escribe como la matriz tridiagonal $\\mathbf A = \\Delta x^{-2}\\,\\mathrm{tridiag}(-1,\\,2,\\,-1)$ y el problema queda $\\mathbf A X=k^2X$. Los autovalores $k^2$ dan las frecuencias $f=v\\sqrt{k^2}/2\\pi$ y los autovectores las formas de los modos. Se diagonaliza con `eigh`, el algoritmo de LAPACK para matrices simétricas.",
            f"Para esta matriz los autovalores se conocen exactamente, $\\lambda_j = \\dfrac{{4}}{{\\Delta x^2}}\\sin^2\\dfrac{{j\\pi}}{{2(N+1)}}$, y al compararlos con $(j\\pi/L)^2$ se ve que el error relativo en frecuencia es $\\approx (j\\pi\\Delta x/L)^2/24$: orden $O(\\Delta x^2)$ y creciente con el número de modo $j$, porque los modos altos oscilan más y quedan peor muestreados. La aproximación subestima siempre la frecuencia. El método falla (resuelve mal) cuando un modo tiene menos de unos 10 puntos por longitud de onda.",
            f"Se escogió $N=400$: $\\Delta x = {sci(dx, 4)}\\ \\mathrm{{m}}$ y el modo 5 tiene {fx(2 * 401 / 5, 0)} puntos por longitud de onda, más que suficiente. La matriz es de $400\\times400$ y se diagonaliza en una fracción de segundo. El error de diagonalización propiamente dicho es $\\approx\\epsilon_{{mach}}\\,\\|\\mathbf A\\|/\\lambda_1 \\sim 10^{{-11}}$ (para este número de condición de ~{sci(cond, 1)}), de modo que lo que domina es el error de discretización de la tabla de abajo.",
            "Es independiente de la solución analítica porque esta usa que las soluciones son senoidales con $k_n = n\\pi/L$, mientras que la diagonalización no sabe nada de senos: solo ve una matriz de números. Que sus autovectores reproduzcan $\\sin(n\\pi x/L)$ y los autovalores se acerquen a $(n\\pi/L)^2$ es la comprobación.",
        ],
        numeric_code="""def modos(N=400, L=1.2, v=40.0 * 1.0, n_modos=5):
    dx = L/(N + 1)                                 # N puntos interiores; los extremos (fijos) no son incognitas
    A = zeros((N, N))
    for j in range(N):
        A[j, j] = 2.0                              # diagonal: 2
        if j > 0:     A[j, j-1] = -1.0             # subdiagonal: -1
        if j < N-1:   A[j, j+1] = -1.0             # superdiagonal: -1
    A = A/dx**2                                    # aproxima -X'' con X(0) = X(L) = 0
    k2, vectores = eigh(A)                         # autovalores ordenados de menor a mayor
    f = v*sqrt(k2[:n_modos])/(2*pi)                # f = v k/(2 pi) para los n_modos menores
    x = linspace(dx, L - dx, N)                    # posiciones de los puntos interiores
    formas = [vec/max(abs(vec)) for vec in vectores.T[:n_modos]]   # normalizar a maximo 1
    return f, x, formas

f_num, x, formas = modos()
f_ana = [n*v/(2*L) for n in range(1, 6)]           # solo para comparar
error_rel = abs(f_num - f_ana)/f_ana               # error relativo por modo""",
        numeric_converge=[
            ("Error frente al número de puntos", f"Se diagonaliza para $N$ creciente y se compara con $f_n=nv/2L$. La pendiente log-log vale {fx(p1, 2)} para el modo 1 y {fx(p5, 2)} para el modo 5 (teoría: 2), y el error observado coincide con la predicción $(j\\pi/(N+1))^2/24$ a 3 cifras:",
             [tab]),
            ("Dependencia con el modo", f"A $N$ fijo el error crece con $j^2$: el modo 5 es 25 veces peor que el modo 1. Con $N=400$ (el valor del código) el error absoluto en la frecuencia del primer modo es de ${sci(fr[0] - fa[0], 2)}\\ \\mathrm{{Hz}}$ y en el quinto de ${sci(fr[4] - fa[4], 2)}\\ \\mathrm{{Hz}}$, ambos despreciables.",
             [tab2]),
            ("Forma del modo y verificación cruzada", f"El autovector del modo 1, normalizado a máximo 1, se compara con $\\sin(\\pi x/L)$ en los mismos puntos. Con $N=400$ la diferencia máxima es ${sci(shape_err, 2)}$.",
             [rf"\max_j\left|X^{{(1)}}_j - \sin(\pi x_j/L)\right| = {sci(shape_err, 3)}"]),
            ("Fuentes de error y criterio de confianza", f"La discretización $O(\\Delta x^2)$ domina; el redondeo de `eigh` es ~$10^{{-11}}$ relativo (negligible hasta que $N\\sim10^{{4}}$ o más). Criterio práctico: el modo $j$ es confiable si $N\\gtrsim 10\\,j$ y si al duplicar $N$ su frecuencia cambia unas 4 veces menos que antes; también se puede extrapolar de Richardson, $f \\approx (4f_{{2N}}-f_N)/3$, que elimina el término $\\Delta x^2$.",
             [rf"f_1^{{(N=400)}} = {fr[0]:.6f}\ \mathrm{{Hz}},\qquad f_1^{{(N=800)}} = {c19.modes_numeric(800)[0][0]:.6f}\ \mathrm{{Hz}},\qquad f_1^{{ana}} = {fa[0]:.6f}\ \mathrm{{Hz}}"]),
        ],
    )


# ============================================================== DopplerEffect
def _doppler() -> dict:
    fa = c20.doppler_analytic()
    f0, vs, c = c20.F0, c20.VS, c20.C
    ns = [10, 100, 400, 4000, 40000]
    en = [abs(c20.doppler_numeric(n_pulses=n) - fa) / fa for n in ns]
    rows = [f"{n} & {fx(c20.doppler_numeric(n_pulses=n), 9)} & {sci(e, 2)}" for n, e in zip(ns, en)]
    tab = _arr("c|c|c", r"N_{pulsos} & f_{num}\ [\mathrm{Hz}] & |f_{num}-f_{ana}|/f_{ana}", rows)
    te = np.arange(400) / f0
    t_arr = (1000.0 - vs * te + c * te) / c
    dta = float(np.mean(np.diff(t_arr)))
    ulp = float(np.spacing(t_arr[-1]))
    bound = 2 * ulp / dta
    cases = [("fuente se acerca ($v_s=120$)", dict(), c20.doppler_analytic()),
             ("fuente se aleja ($v_s=-120$)", dict(vs=-120.0), c20.doppler_analytic(vs=-120.0)),
             ("observador se acerca ($v_o=30$)", dict(vs=0.0, vo=30.0), c20.doppler_analytic(vs=0.0, vo=30.0))]
    rows2 = [f"\\text{{{n}}} & {fx(c20.doppler_numeric(**kw), 8)} & {fx(a, 8)} & {sci(abs(c20.doppler_numeric(**kw) - a) / a, 2)}" for n, kw, a in cases]
    tab2 = _arr("l|c|c|c", r"\text{caso} & f_{num} & f_{ana} & \text{error rel.}", rows2)
    dfd = f0 * c / (c - vs) ** 2
    return dict(
        numeric_intro=[
            "El efecto Doppler se deduce analíticamente contando frentes de onda: si la fuente emite a $f_0$ y se acerca a velocidad $v_s$, la distancia entre frentes se acorta y $f' = f_0\\,c/(c-v_s)$. El método numérico no usa esa fórmula; hace una simulación geométrica de los pulsos. Es un experimento mental ejecutado en el computador: se emiten pulsos a intervalos de $1/f_0$ desde una fuente que avanza, se calcula en qué instante llega cada uno al observador y se mide el intervalo medio entre llegadas.",
            "La regla es una ecuación de alcance para cada pulso $k$. Emitido en $t_{e,k}=k/f_0$ desde $x_{s,k} = x_0 - v_s t_{e,k}$, viaja a $c$ y el observador se mueve como $x_o=v_o t$; llegan a la vez cuando $x_{s,k}-c\\,(t-t_{e,k}) = v_ot$, de donde $t_{a,k} = (x_{s,k}+c\\,t_{e,k})/(c+v_o)$. La frecuencia percibida es el inverso del intervalo medio de llegadas, $f = 1/\\overline{\\Delta t_a}$ con $\\overline{\\Delta t_a} = (t_{a,N-1}-t_{a,0})/(N-1)$.",
            f"Aquí no hay error de truncamiento: la ecuación de alcance es lineal en $t$ y se resuelve de forma exacta, así que cada $t_{{a,k}}$ es exacto salvo redondeo. La diferencia entre llegadas sucesivas es constante, $\\Delta t_a=(1-v_s/c)/f_0 = {sci(dta, 6)}\\ \\mathrm{{s}}$, y el promedio telescópico solo conserva el primero y el último. El único error es de redondeo de punto flotante (~$10^{{-16}}$ relativo en cada tiempo) amplificado por la resta de tiempos casi iguales: cota $\\approx 2\\,\\mathrm{{ulp}}(t)/\\Delta t_a = {sci(bound, 2)}$. El método falla si la fuente supera la rapidez del sonido ($v_s\\ge c$, no hay llegada ordenada) o si pasa por el observador durante la simulación.",
            f"Los parámetros: la fuente parte de $x_0=1000\\ \\mathrm{{m}}$ y se emiten $400$ pulsos ({fx(399 / f0, 3)} s de emisión, durante los cuales la fuente recorre {fx(vs * 399 / f0, 1)} m y nunca llega al observador). Con $f_0=440$ Hz, $v_s=120$ m/s y $c=343$ m/s el resultado es $f = {fx(c20.doppler_numeric(), 4)}\\ \\mathrm{{Hz}}$, un {fx((c20.doppler_numeric() / f0 - 1) * 100, 0)} % por encima de $f_0$ (el sonido se percibe mucho más agudo, cerca de una quinta justa).",
            "Es independiente en el sentido de que no usa la fórmula $f_0c/(c-v_s)$: la frecuencia aparece como un resultado del conteo de llegadas. Pero el modelo físico es el mismo (velocidad del sonido constante en el medio en reposo, fuente y observador con velocidad uniforme), así que valida la deducción de la fórmula, no la suposición de medio estático.",
        ],
        numeric_code="""def doppler_numerico(f0=440.0, vs=120.0, vo=0.0, c=343.0, n_pulsos=400):
    x0 = 1000.0                                    # la fuente parte a 1000 m del observador (que esta en x = 0)
    te = arange(n_pulsos)/f0                       # instante de emision del pulso k: k/f0
    xs = x0 - vs*te                                # posicion de la fuente al emitir (se acerca si vs > 0)
    # el pulso sale de xs en te con rapidez c; el observador va como x_o(t) = vo*t
    # llegada: xs - c*(t - te) = vo*t  =>  t = (xs + c*te)/(c + vo)
    t_llegada = (xs + c*te)/(c + vo)               # instante de llegada de cada pulso al observador
    intervalo = diferencias(t_llegada)             # tiempo entre llegadas consecutivas
    return 1.0/promedio(intervalo)                 # frecuencia percibida = 1 / intervalo medio

f_num = doppler_numerico()                         # 676.7713... Hz
f_ana = 440.0*343.0/(343.0 - 120.0)                # formula cerrada, solo para comparar
error_rel = abs(f_num - f_ana)/f_ana               # ~1e-16: redondeo""",
        numeric_converge=[
            ("Error frente al número de pulsos", f"A diferencia de los métodos de discretización, aquí aumentar el número de pulsos no mejora el resultado, porque no hay error de truncamiento. El error relativo se queda en el nivel del redondeo ($\\sim10^{{-16}}$ a $10^{{-14}}$), sin tendencia clara:",
             [tab]),
            ("Tres configuraciones", "Se comprueba el modelo cambiando el signo de $v_s$ (la fuente se aleja: el sonido se vuelve grave) y moviendo el observador en lugar de la fuente (la fórmula es distinta, $f_0(c+v_o)/c$, y la simulación la reproduce sin cambiar de código):",
             [tab2]),
            ("Cota teórica del redondeo", f"Cada $t_a$ vale ~{fx(t_arr[-1], 2)} s y una resta de dos tiempos así tiene un error de ~$2\\,\\mathrm{{ulp}} = {sci(2 * ulp, 2)}\\ \\mathrm{{s}}$ frente a un intervalo de ${sci(dta, 3)}\\ \\mathrm{{s}}$. Eso da una cota de ${sci(bound, 2)}$ en la frecuencia relativa, que los errores de la tabla anterior respetan holgadamente; el promedio de diferencias consecutivas colapsa al primer y último tiempo, y por eso el error no crece con el número de pulsos.",
             [rf"\varepsilon_{{rel}} \lesssim \dfrac{{2\,\mathrm{{ulp}}(t_a)}}{{\overline{{\Delta t_a}}}} = {sci(bound, 3)}"]),
            ("Sensibilidad a los datos y criterio de confianza", f"La fuente de incertidumbre real es la de los datos, no la del método: $\\partial f/\\partial v_s = f_0c/(c-v_s)^2 = {fx(dfd, 3)}\\ \\mathrm{{Hz/(m/s)}}$, así que un error de $1\\ \\mathrm{{m/s}}$ en la velocidad de la fuente cambia la frecuencia en ~{fx(dfd, 1)} Hz, trece órdenes de magnitud más que el redondeo. Criterio: el resultado es confiable si $v_s<c$ y los pulsos son suficientes para que el intervalo medio sea estable (basta con ≥ 2), y entonces la exactitud la fija la precisión de $c$, $v_s$ y $f_0$.",
             [rf"\dfrac{{\partial f}}{{\partial v_s}} = \dfrac{{f_0\,c}}{{(c-v_s)^2}} = {fx(dfd, 4)}\ \mathrm{{Hz\,s/m}}"]),
        ],
    )


# ============================================================== BeatsAndPipes
def _beats() -> dict:
    fb = c20.beat_frequency_analytic()
    Ts = [1.0, 2.0, 4.0, 8.0, 16.0]
    ef = [abs(c20.beat_frequency_numeric(16000.0, T) - fb) for T in Ts]
    rows = [f"{T:g} & {fx(1 / T, 4)} & {sci(e, 3)}" for T, e in zip(Ts, ef)]
    tab = _arr("c|c|c", r"T\ [\mathrm{s}] & \Delta f_{bin} = 1/T\ [\mathrm{Hz}] & |f_{b,num}-f_{b,ana}|\ [\mathrm{Hz}]", rows)
    try:
        c20.beat_frequency_numeric(16000.0, 0.5)
        fallo = "resuelve"
    except Exception:
        fallo = "no resuelve"
    fss = [2000.0, 4000.0, 8000.0, 16000.0, 32000.0]
    efs = [abs(c20.beat_frequency_numeric(fs, 4.0) - fb) for fs in fss]
    rows2 = [f"{fs:.0f} & {sci(e, 3)}" for fs, e in zip(fss, efs)]
    tab2 = _arr("c|c", r"f_s\ [\mathrm{Hz}] & |f_{b,num}-f_{b,ana}|\ [\mathrm{Hz}]", rows2)
    eenv = [abs(c20.beat_envelope_frequency_numeric(16000.0, T) - fb) for T in (1.0, 2.0, 4.0)]
    Ns = [25, 100, 400, 1200]
    fa_o = c20.pipe_frequencies_analytic("open")
    fa_c = c20.pipe_frequencies_analytic("closed")
    eo = [np.abs(c20.pipe_frequencies_numeric("open", n_points=N) - fa_o) / fa_o for N in Ns]
    ec = [np.abs(c20.pipe_frequencies_numeric("closed", n_points=N) - fa_c) / fa_c for N in Ns]
    po = _order(Ns, [e[0] for e in eo])
    pc = _order(Ns, [e[0] for e in ec])
    rows3 = [f"{N} & {sci(a[0], 3)} & {sci(a[3], 3)} & {sci(b[0], 3)} & {sci(b[3], 3)}" for N, a, b in zip(Ns, eo, ec)]
    tab3 = _arr("c|c|c|c|c", r"N & \text{abierto } n=1 & \text{abierto } n=4 & \text{cerrado } n=1 & \text{cerrado } n=4", rows3)
    fn = c20.pipe_frequencies_numeric("open")
    t, s = c20.beat_signal(16000.0, 4.0)
    return dict(
        numeric_intro=[
            "Cuando se suman dos tonos de frecuencias cercanas $f_1$ y $f_2$ se oyen pulsaciones a $|f_2-f_1|$, porque $\\sin a+\\sin b = 2\\cos\\tfrac{a-b}2\\sin\\tfrac{a+b}2$: un tono promedio con una envolvente lenta. El método numérico mide esa frecuencia sin usar la identidad, analizando la señal muestreada de dos formas independientes: con la transformada de Fourier discreta (FFT), localizando los dos picos del espectro, y en el dominio del tiempo, midiendo el ritmo de los máximos de la envolvente. Una tercera parte de la escena obtiene las frecuencias de resonancia de tubos resolviendo un problema de autovalores.",
            "La FFT calcula $S_m=\\sum_{j=0}^{N-1}s_jw_j\\,e^{-2\\pi i jm/N}$, con la ventana de Hann $w_j = \\tfrac12[1-\\cos(2\\pi j/(N-1))]$ que reduce las fugas espectrales. Los máximos del espectro están en $f_m=m\\,f_s/N$, de modo que la resolución es el inverso de la duración, $\\Delta f=1/T$. Para ubicar el pico entre bines se ajusta una parábola a $\\ln|S|$ en tres puntos ($\\alpha,\\beta,\\gamma$) y el desplazamiento es $\\delta = \\tfrac12(\\alpha-\\gamma)/(\\alpha-2\\beta+\\gamma)$. La envolvente se obtiene de la señal analítica: se anulan las frecuencias negativas del espectro, se invierte y se toma el módulo.",
            "El error de la FFT no es de truncamiento de una serie, sino de resolución y fuga. Sin interpolación el error de un pico es hasta $\\Delta f/2=1/(2T)$; la ventana y la interpolación parabólica lo reducen mucho más, y dos picos solo se separan si $T\\gtrsim 2\\text{-}4/|f_2-f_1|$ (el lóbulo principal de Hann mide cuatro bines). Hay que cumplir además Nyquist, $f_s>2f_2$. En los tubos, el problema $p''=-k^2p$ se discretiza como en la cuerda y el error de la frecuencia es $O(\\Delta x^2)$, aproximadamente $(n\\pi\\Delta x/L)^2/24$.",
            f"Los parámetros: $f_s=16000$ Hz (muy por encima de Nyquist para 444 Hz) y $T=4$ s para la FFT, de modo que $\\Delta f = {fx(1 / 4, 2)}$ Hz es una fracción de los $4$ Hz que se quieren medir; la envolvente usa $T=2$ s para ver $8$ pulsaciones. Con ellos se obtiene $f_b = {fx(c20.beat_frequency_numeric(), 8)}\\ \\mathrm{{Hz}}$ (FFT) y {fx(c20.beat_envelope_frequency_numeric(), 8)} Hz (envolvente). Para los tubos, $N=600$ puntos y los cuatro menores autovalores: p. ej. el tubo abierto de $0.75$ m da $f_1={fx(fn[0], 3)}\\ \\mathrm{{Hz}}$ frente a $c/2L = {fx(fa_o[0], 3)}$ Hz.",
            "Los tres métodos son independientes del analítico porque nunca usan $|f_2-f_1|$ ni $nc/2L$: el primero lee la separación entre picos de un espectro, el segundo cuenta el ritmo de la envolvente, y el tercero diagonaliza una matriz. Que los dos primeros coincidan entre sí y con la resta simple da confianza en las herramientas de análisis de señal, que sirven aun cuando la señal no sea una suma limpia de dos tonos.",
        ],
        numeric_code="""def pulsacion_fft(fs=16000.0, T=4.0, f1=440.0, f2=444.0):
    t = arange(0, T, 1/fs)                         # muestras: N = fs*T
    s = sin(2*pi*f1*t) + sin(2*pi*f2*t)            # suma de los dos tonos
    S = abs(rfft(s*hann(len(s))))                  # espectro con ventana de Hann (menos fugas)
    f = rfftfreq(len(s), 1/fs)                     # frecuencia de cada bin: m*fs/N
    picos = [i for i in range(1, len(S)-1)
             if S[i] > S[i-1] and S[i] >= S[i+1] and S[i] > 0.2*max(S)]   # maximos locales altos
    def refinar(i):                                # parabola en ln|S| (alfa, beta, gamma)
        a, b, c = log(S[i-1]), log(S[i]), log(S[i+1])
        return f[i] + 0.5*(a - c)/(a - 2*b + c)*(f[1] - f[0])
    f_a, f_b = sorted(refinar(i) for i in picos)[:2]
    return f_b - f_a                               # pulsacion = separacion de picos

def pulsacion_envolvente(fs=16000.0, T=2.0):
    t, s = senal(fs, T)
    env = abs(ifft(fft(s)*[1, 2, ..., 2, 1, 0, ...]))   # envolvente de la senal analitica (Hilbert)
    picos = maximos_locales(env, umbral=1.8)       # un maximo por pulsacion
    return 1/promedio(diferencias(t[picos]))       # frecuencia = 1 / periodo medio

def frecuencias_tubo(tipo, L=0.75, c=343.0, N=600):
    A = tridiagonal(N, 2, -1)/dx**2                # -p'' = k^2 p; dx = L/(N+1) (abierto) o L/(N+0.5)
    if tipo == "cerrado": A[0, 0] = 1/dx**2        # p' = 0 en el extremo cerrado
    return c*sqrt(eigvalsh(A)[:4])/(2*pi)          # f = c k/(2 pi)""",
        numeric_converge=[
            ("Error de la FFT frente a la duración", f"Se mide la pulsación con $f_s=16000$ Hz y duración $T$ creciente. La resolución del bin es $1/T$, pero el error con interpolación parabólica y ventana es muchísimo menor (cae unas 30 veces por duplicación de $T$ en esta señal). Con $T=0.5$ s los dos picos de 4 Hz de separación quedan unidos en el lóbulo principal de la ventana y el método {fallo} los picos.",
             [tab]),
            ("Error frente a la frecuencia de muestreo y por envolvente", f"A $T=4$ s fijo, el error de la FFT baja con $f_s$ (aproximadamente en proporción inversa), pero ya con $f_s=2000$ Hz está por debajo de $10^{{-7}}$ Hz: lo que importa es $T$, no $f_s$, mientras $f_s>2f_2$. El método de la envolvente da error {'nulo' if max(eenv) == 0 else 'menor que $' + sci(max(eenv), 1) + '$ Hz'} con $T=1,2,4$ s: los máximos caen exactamente sobre muestras porque $0.25\\ \\mathrm{{s}}\\cdot16000=4000$ es entero; con otra $f_s$ el error sería $\\lesssim f_b^2/f_s = {sci(fb ** 2 / 16000, 1)}\\ \\mathrm{{Hz}}$ por intervalo.",
             [tab2]),
            ("Tubos: error frente al número de puntos", f"Para los tubos se mide el error relativo de los modos $n=1$ y $n=4$ de ambos tipos. La pendiente log-log del modo 1 es {fx(po, 2)} (abierto) y {fx(pc, 2)} (cerrado), compatible con el $O(\\Delta x^2)$ teórico; el modo $n=4$ abierto es $\\approx16$ veces peor, como predice $\\propto n^2$, y el cuarto modo del tubo cerrado (el armónico 7) unas 49 veces peor que el primero. Con $N=600$ el error del modo 1 abierto es ${sci(abs(fn[0] - fa_o[0]) / fa_o[0], 2)}$.",
             [tab3]),
            ("Fuentes de error y criterio de confianza", f"FFT: resolución y fuga espectral ($\\sim{sci(ef[2], 1)}$ Hz con $T=4$ s), cuantización en el tiempo para la envolvente ($\\sim1/f_s$) y redondeo de la FFT (~$10^{{-13}}$ Hz). Tubos: discretización $O(\\Delta x^2)$ (hasta ${sci(np.max(eo[2]), 1)}$ relativo en el tubo abierto con $N=400$) y redondeo de `eigvalsh` (~$10^{{-11}}$). Criterio: la pulsación por FFT es confiable si $T\\gtrsim 4/f_b$ (dos picos separados) y $f_s>2f_2$; los modos del tubo, si hay más de ~10 puntos por longitud de onda y el error cae ~4 veces al duplicar $N$.",
             [rf"T_{{min}} \approx \dfrac{{4}}{{f_b}} = {fx(4 / fb, 2)}\ \mathrm{{s}},\qquad f_s > 2f_2 = {fx(2 * c20.F2, 0)}\ \mathrm{{Hz}}"]),
        ],
    )


DETAIL = {
    "VenturiTube": _venturi(),
    "TorricelliDrain": _torricelli(),
    "SimpleHarmonic": _shm(),
    "DrivenResonance": _driven(),
    "WavePulse": _pulse(),
    "StandingModes": _modes(),
    "DopplerEffect": _doppler(),
    "BeatsAndPipes": _beats(),
}
