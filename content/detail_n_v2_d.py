"""Explicacion ampliada del metodo numerico (numeric_intro, numeric_code, numeric_converge) de las escenas
YeeFDTD, PoyntingDisplacement, SnellFermat, PrismDispersion, ThinLensImage, TwoLensSystem, YoungSlits,
ThinFilmCoating, SlitAndGrating y PolarizersMalus. Todas las cifras salen de ejecutar `physics/`."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch14_maxwell as m14
from physics import v2ch15_light as m15
from physics import v2ch16_lenses as m16
from physics import v2ch17_interference as m17
from physics import v2ch18_diffraction as m18
from physics.ch07_work_energy import simpson
from physics.constants import C_LIGHT, EPS0, MU0

EPS = float(np.finfo(float).eps)
PHI = (np.sqrt(5) - 1) / 2


def _tab(cols: str, rows: list[list[str]]) -> str:
    return r"\begin{array}{" + cols + r"}" + r" \\ ".join(" & ".join(r) for r in rows) + r"\end{array}"


def ps(x: float, d: int = 3) -> str:
    """Numero en notacion cientifica para PROSA (con $...$)."""
    return "$" + sci(x, d) + "$"


def _slope(x1: float, e1: float, x2: float, e2: float) -> float:
    return float(np.log(e2 / e1) / np.log(x2 / x1))


def _golden(f, a: float, b: float, k: int):
    """Seccion aurea con k iteraciones (misma logica que physics/); devuelve (punto medio, ancho del intervalo)."""
    c, e = b - PHI * (b - a), a + PHI * (b - a)
    for _ in range(k):
        if f(c) < f(e):
            b = e
        else:
            a = c
        c, e = b - PHI * (b - a), a + PHI * (b - a)
    return 0.5 * (a + b), b - a


# ===================================================================== Capitulo 14
def _yee() -> dict:
    dx, sig, steps = 1e-3, 20e-3, 600
    cs = (0.25, 0.5, 0.75, 0.9, 1.0)
    v = {s: m14.speed_numeric(s) for s in cs}
    err = {s: abs(v[s] - C_LIGHT) / C_LIGHT for s in cs}
    pred = {s: (1 - s * s) / 8 * (dx / sig) ** 2 for s in cs}
    tab = _tab("rrrrr", [[r"S", r"v_{num}\,(\mathrm{m/s})", r"\varepsilon_{rel}", r"\varepsilon_{pred}", r"\varepsilon/\varepsilon_{pred}"]] +
               [[str(s), sci(v[s], 8), sci(err[s], 3), sci(pred[s], 3), (fx(err[s] / pred[s], 4) if s < 1 else r"-")] for s in cs])
    dist = 0.5 * dx * steps
    lag = err[0.5] * dist
    return dict(
        numeric_intro=[
            r"El método de diferencias finitas en el dominio del tiempo (FDTD, esquema de Yee) integra directamente las dos ecuaciones de rotacional de Maxwell sobre una malla, sin suponer de antemano que la solución es una onda de velocidad $c$. Se necesita porque la solución analítica $f(x-ct)$ solo existe en el vacío ilimitado: en cuanto hay materiales, fronteras o fuentes, el único camino general es avanzar los campos paso a paso. Aquí se usa en el caso más simple para comprobar que las ecuaciones discretas, por sí solas, producen un pulso que viaja a $c$.",
            r"El esquema coloca $E_z$ en nodos enteros $x_j=j\,\Delta x$ y $H_y$ en medios nodos, y los avanza de forma alternada (salto de rana): $H_y^{n+1/2}[j]=H_y^{n-1/2}[j]+\dfrac{\Delta t}{\mu_0\Delta x}(E_z^n[j+1]-E_z^n[j])$ y $E_z^{n+1}[j]=E_z^{n}[j]+\dfrac{\Delta t}{\varepsilon_0\Delta x}(H_y^{n+1/2}[j]-H_y^{n+1/2}[j-1])$. Cada derivada es una diferencia centrada sobre una media celda, por lo que no hay que resolver ningún sistema: cada actualización es una operación vectorizada.",
            r"Al sustituir una onda plana $e^{i(kx-\omega t)}$ en el esquema se obtiene la relación de dispersión discreta $\sin(\omega\Delta t/2)=S\,\sin(k\Delta x/2)$, con el número de Courant $S=c\Delta t/\Delta x$. De ella se deduce que el error es de segundo orden, $O(\Delta x^2,\Delta t^2)$: la velocidad de fase relativa es $-(k\Delta x)^2(1-S^2)/24$ y la de grupo, que es la que sigue un pulso, es tres veces mayor en magnitud, $-(k\Delta x)^2(1-S^2)/8$. Además hay estabilidad solo si $S\le1$ (condición CFL); con $S=1$ en una dimensión el error de dispersión se anula y el esquema es exacto.",
            rf"Los parámetros se escogieron así: $\Delta x=1$ mm con anchura del pulso $\sigma=20\,\Delta x$, de modo que el número de onda dominante cumple $k\Delta x\approx\Delta x/\sigma=1/20$ (unas 20 celdas por característica del pulso, el mínimo razonable es 10); $S=0.5$ es un valor típico y estable; y 600 pasos llevan el pico {fx(dist, 2)} m a la derecha sin llegar al borde de la malla de 1.2 m, donde la condición de frontera reflejaría el pulso. La velocidad se obtiene ajustando una recta a las posiciones del pico (refinadas con una parábola por tres nodos), no leyendo ningún valor de la fórmula.",
            r"El método es independiente del analítico porque $c=1/\sqrt{\mu_0\varepsilon_0}$ nunca se usa en las actualizaciones: solo aparecen $\mu_0$ y $\varepsilon_0$ por separado en los coeficientes. La constante $c$ interviene únicamente para fijar $\Delta t$ y la razón inicial entre $E$ y $H$; si los coeficientes de Faraday o de Ampère-Maxwell estuvieran mal, el pulso se deformaría o iría a otra velocidad, y la pendiente medida lo delataría.",
        ],
        numeric_code=f"""dx, S, pasos = 1e-3, 0.5, 600            # celda de 1 mm, Courant 0.5, 600 pasos
dt = S * dx / c                          # paso temporal fijado por Courant (S <= 1: estable)
ez = exp(-(x - x0)**2 / (2*sigma**2))    # E_z en nodos enteros: pulso gaussiano, sigma = 20 dx
hy = -exp(-(xh - x0 + c*dt/2)**2 / (2*sigma**2)) / (mu0*c)   # H_y en medios nodos, medio paso antes
for k in range(pasos):
    hy += dt/(mu0*dx) * (ez[1:] - ez[:-1])             # Faraday: rotacional discreto de E
    ez[1:-1] += dt/(eps0*dx) * (hy[1:] - hy[:-1])      # Ampere-Maxwell con el H recien calculado
    if k % 20 == 0:                                    # cada 20 pasos se anota el pico
        j = argmax(ez)                                 # nodo del maximo
        y0, y1, y2 = ez[j-1], ez[j], ez[j+1]
        delta = 0.5*(y0 - y2) / (y0 - 2*y1 + y2)       # parabola por 3 nodos: posicion sub-celda
        picos.append(((k+1)*dt, (j + delta)*dx))       # pareja (t, x_pico)
v_num = polyfit(t_picos, x_picos, 1)[0]                # pendiente de la recta = velocidad medida
""",
        numeric_converge=[
            ("Error de la velocidad frente al número de Courant",
             f"Se repite la simulación completa (600 pasos, $\\Delta x=1$ mm) para cinco valores de $S$ y se compara la pendiente medida con $c$. La columna $\\varepsilon_{{pred}}=(1-S^2)(\\Delta x/\\sigma)^2/8$ es la predicción teórica de la velocidad de grupo con $k\\approx1/\\sigma$; el cociente entre lo medido y lo predicho vale 1 con tres o cuatro cifras, lo que confirma el orden $O(\\Delta x^2)$ y el factor $(1-S^2)$ de la teoría.",
             [tab]),
            ("Fuentes de error y su magnitud",
             f"El error dominante es de truncamiento (dispersión numérica): con $S=0.5$ es {ps(err[0.5], 3)}, es decir el pulso se retrasa {ps(lag, 2)} m = {fx(lag / dx, 3)} celdas al recorrer {fx(dist, 2)} m. Con $S=1$ el error cae a {ps(err[1.0], 2)}, que es puro redondeo de doble precisión acumulado en 600 pasos y en la regresión, así que el seguimiento del pico y el ajuste lineal no añaden error apreciable. Como el error escala con $(\\Delta x/\\sigma)^2$, duplicar la resolución lo divide entre 4.",
             [rf"\varepsilon_{{trunc}}(S=0.5)={sci(err[0.5], 3)},\qquad \varepsilon_{{red}}(S=1)={sci(err[1.0], 2)}"]),
            ("Criterio práctico de confianza",
             "El resultado es confiable si (i) $S\\le1$ (si no, el error crece exponencialmente en lugar de decaer), (ii) hay al menos 10 a 20 celdas por la escala más pequeña del pulso, y (iii) el pulso no ha tocado las fronteras. Aquí las tres condiciones se cumplen y el error relativo de $c$ es de unas dos partes en $10^4$, mucho menor que la precisión con que se mide $c$ en un laboratorio, de modo que la verificación numérica de $c=1/\\sqrt{\\mu_0\\varepsilon_0}$ es concluyente.",
             [rf"v_{{num}}(S=0.5)={sci(v[0.5], 8)}\ \mathrm{{m/s}}\ \ \text{{frente a}}\ \ c={sci(C_LIGHT, 8)}\ \mathrm{{m/s}}"]),
        ],
    )


def _poynting() -> dict:
    ia = m14.intensity_analytic()
    ic = m14.conduction_current_analytic()
    ns = (2, 4, 6, 8, 16, 4000)
    es = {n: abs(m14.intensity_numeric(n=n) - ia) / ia for n in ns}
    tabs = _tab("rrr", [[r"n", r"I_{num}\,(\mathrm{W/m^2})", r"\varepsilon_{rel}"]] +
                [[str(n), fx(m14.intensity_numeric(n=n), 6), sci(es[n], 3)] for n in ns])
    hs = (1e-3, 1e-6, 1e-9, 1e-12, 1e-15)
    t0 = 1e-6
    eh = {h: abs(m14.displacement_current_numeric(h) - ic) / ic for h in hs}
    ph = {h: EPS * max(1.0, t0 / h) for h in hs}
    tabh = _tab("rrrr", [[r"h\,(\mathrm{s})", r"I_{d,num}\,(\mathrm{A})", r"\varepsilon_{rel}", r"\varepsilon_{red}\approx\epsilon_m\max(1,t_0/h)"]] +
                [[sci(h, 1), sci(m14.displacement_current_numeric(h), 8), sci(eh[h], 2), sci(ph[h], 2)] for h in hs])
    return dict(
        numeric_intro=[
            r"Esta escena usa dos herramientas numéricas elementales. La primera es la regla de Simpson compuesta, que calcula el promedio temporal de $S(t)=E(t)B(t)/\mu_0$ integrando sobre un período; la segunda es la derivada por diferencias centradas, que calcula $I_d=\varepsilon_0\,d\Phi_E/dt$ a partir de muestras de $\Phi_E(t)$. Se usan porque un detector o un medidor real entrega muestras, no fórmulas, y estas dos reglas son la forma estándar de convertir muestras en integrales y derivadas.",
            r"Con $n$ subintervalos (par) de ancho $h=T/n$, Simpson aproxima $\int_0^T f\,dt\approx\dfrac{h}{3}\left[f_0+4f_1+2f_2+4f_3+\cdots+4f_{n-1}+f_n\right]$, y la derivada centrada es $f'(t)\approx\dfrac{f(t+h)-f(t-h)}{2h}$. Ambas salen de ajustar un polinomio local (una parábola por tres puntos en Simpson, una recta por dos puntos simétricos en la derivada) y calcular de forma exacta la integral o la pendiente de ese polinomio.",
            r"El error de truncamiento de Simpson es $-\dfrac{(b-a)}{180}h^4f^{(4)}(\xi)$, es decir $O(h^4)$, y el de la derivada centrada es $\dfrac{h^2}{6}f'''(\xi)$, es decir $O(h^2)$. Pero la derivada tiene otro enemigo: el redondeo, que aporta aproximadamente $\epsilon_m\,|f|/h$ con $\epsilon_m\approx2.2\times10^{-16}$; si $h$ es demasiado pequeño, la resta $f(t+h)-f(t-h)$ pierde cifras significativas y el error vuelve a crecer. Existe un $h$ óptimo que equilibra truncamiento y redondeo.",
            r"Aquí el integrando $\sin^2(2\pi t/T)$ es un polinomio trigonométrico con un solo armónico (frecuencia doble), así que la regla del trapecio es exacta en cuanto hay más de dos subintervalos y Simpson, que combina los resultados con $n$ y $n/2$ subintervalos, lo es desde $n=6$; se emplean $n=4000$ muestras, muy por encima del mínimo. En la derivada, el flujo $\Phi_E=(V/d)A$ con $V=(dV/dt)\,t$ es lineal en $t$, de modo que el truncamiento es cero y todo el error es redondeo; se usa $h=10^{-9}$ s, frente a $t_0=10^{-6}$ s.",
            r"Ambos resultados son independientes de la teoría: el promedio se obtiene muestreando $E\sin(2\pi t)$ y $B=E/c$ y multiplicando punto a punto, sin usar $\tfrac12 c\varepsilon_0E_0^2$ (que contiene el factor $\tfrac12$ del promedio de $\sin^2$), y la corriente de desplazamiento se obtiene derivando el flujo, sin usar la fórmula cerrada $\varepsilon_0 A\,\dot V/d$.",
        ],
        numeric_code=f"""# --- promedio temporal del vector de Poynting (Simpson compuesto) ---
n = 4000                                 # subintervalos (par) en un periodo
t = linspace(0, 1, n + 1)                # malla uniforme, T = 1 s
E = E0 * sin(2*pi*t)                     # campo electrico muestreado
B = E / c                                # onda plana: B = E / c
S = E * B / mu0                          # |S| = E x B / mu0 en cada instante
h = (t[-1] - t[0]) / n                   # ancho de subintervalo
I = h/3 * (S[0] + S[-1] + 4*S[1:-1:2].sum() + 2*S[2:-1:2].sum())   # pesos 1,4,2,4,...,4,1 de Simpson
# --- corriente de desplazamiento (derivada centrada del flujo) ---
phi = lambda t: (dVdt * t / d) * A       # flujo electrico a traves de una placa: E = V/d, V = dVdt t
t0, h = 1e-6, 1e-9                       # instante de evaluacion y paso de la derivada
Id = eps0 * (phi(t0 + h) - phi(t0 - h)) / (2*h)   # pendiente simetrica, error O(h^2) + redondeo
""",
        numeric_converge=[
            ("Simpson: error del promedio contra el número de subintervalos",
             f"Se calcula $\\langle S\\rangle$ con $n$ subintervalos y se compara con $\\tfrac12c\\varepsilon_0E_0^2={fx(ia, 6)}\\ \\mathrm{{W/m^2}}$. Con $n=2$ y $n=4$ el muestreo no resuelve el segundo armónico y el error es enorme; desde $n=6$ la regla es exacta y el error ({ps(es[6], 2)}) es puro redondeo, que no mejora al aumentar $n$ (con $n=4000$ vale {ps(es[4000], 2)}). El orden $h^4$ no se observa porque la integral de un armónico completo no tiene error de truncamiento mientras se resuelva.",
             [tabs]),
            ("Derivada centrada: el error contra el paso h",
             f"Como $\\Phi_E$ es lineal, el truncamiento $h^2f'''/6$ es nulo y la tabla muestra solo redondeo, que crece cuando $h$ disminuye: cada fila sigue la estimación $\\epsilon_m\\max(1,t_0/h)$ dentro de un factor de 1 a 4. Con $h=10^{{-9}}$ s el error es {ps(eh[1e-9], 2)}, y con $h=10^{{-15}}$ s el error sube a {ps(eh[1e-15], 2)}.",
             [tabh]),
            ("Fuentes de error y criterio de confianza",
             "Aquí no hay error estadístico ni de discretización relevante: la integral y la derivada son exactas salvo redondeo, y su magnitud (de $10^{-14}$ a $10^{-13}$ en relativo) es despreciable frente a la incertidumbre de cualquier medición de $E_0$. En general, el resultado de Simpson es confiable cuando al duplicar $n$ el cambio es menor que la tolerancia, y el de la derivada cuando se está cerca del paso óptimo $h\\sim\\epsilon_m^{1/3}$ (para derivadas centradas de funciones no lineales) y no se pierden cifras en la resta.",
             [rf"\varepsilon_{{Simpson}}(n=4000)={sci(es[4000], 2)},\qquad \varepsilon_{{deriv}}(h=10^{{-9}}\,\mathrm{{s}})={sci(eh[1e-9], 2)}"]),
        ],
    )


# ===================================================================== Capitulo 15
def _snell() -> dict:
    d, _ = m15.target_point()
    xs = m15.refraction_point_analytic()
    n1, n2, h1, h2 = m15.N1, m15.N2, m15.H1, m15.H2
    tt = lambda x: m15.travel_time(x, d)  # noqa: E731
    ks = (5, 10, 20, 30, 40, 60)
    gold = {k: _golden(tt, 0.0, d, k) for k in ks}
    ratio = (gold[60][1] / gold[10][1]) ** (1 / 50)
    tabg = _tab("rrrr", [[r"k", r"\mathrm{ancho}", r"|x_k-x_{Snell}|\,(\mathrm{m})", r"(\mathrm{ancho}_k/\mathrm{ancho}_{k_{prev}})^{1/\Delta k}"]] +
                [[str(k), sci(gold[k][1], 3), sci(abs(gold[k][0] - xs), 3), (fx((gold[k][1] / gold[ks[i - 1]][1]) ** (1 / (k - ks[i - 1])), 4) if i else r"-")] for i, k in enumerate(ks)])
    tc = m15.critical_angle_analytic()
    kb = (5, 10, 20, 30, 40, 60)
    eb = {}
    for k in kb:
        lo, hi = 0.0, 90.0
        for _ in range(k):
            mid = 0.5 * (lo + hi)
            if n2 * np.sin(np.deg2rad(mid)) <= n1:
                lo = mid
            else:
                hi = mid
        eb[k] = abs(0.5 * (lo + hi) - tc)
    tabb = _tab("rrr", [[r"k", r"\mathrm{ancho}=90^\circ/2^k", r"|\theta_k-\theta_c|\,(^\circ)"]] +
                [[str(k), sci(90.0 / 2**k, 3), sci(eb[k], 3)] for k in kb])
    r1, r2 = np.hypot(xs, h1), np.hypot(d - xs, h2)
    lpp = n1 * h1**2 / r1**3 + n2 * h2**2 / r2**3
    l0 = float(tt(xs) * C_LIGHT)
    floor_pred = float(np.sqrt(2 * EPS * l0 / lpp))
    floor_obs = abs(m15.fermat_numeric(d) - xs)
    return dict(
        numeric_intro=[
            r"Aquí hay dos problemas que se resuelven buscando un punto sin despejar nada. El principio de Fermat se aplica minimizando el tiempo de viaje $T(x)$ sobre el punto de incidencia $x$ con la búsqueda de sección áurea, y el ángulo crítico se obtiene por bisección sobre el ángulo de incidencia. Se usan métodos de búsqueda porque la condición $T'(x)=0$ conduce a una ecuación trascendente en $x$ que no se puede despejar, mientras que evaluar $T(x)$ es trivial.",
            r"La sección áurea mantiene un intervalo $[a,b]$ que contiene al mínimo y dos puntos interiores $c=b-\varphi(b-a)$, $e=a+\varphi(b-a)$ con $\varphi=(\sqrt5-1)/2\approx0.618$. Si $T(c)<T(e)$ el mínimo está en $[a,e]$ (se hace $b\leftarrow e$); si no, en $[c,b]$ (se hace $a\leftarrow c$). La elección de $\varphi$ hace que uno de los dos puntos interiores se reutilice, de modo que cada iteración cuesta una sola evaluación nueva. La bisección es más simple: $\theta_{mid}=(\theta_{lo}+\theta_{hi})/2$ y se queda con la mitad donde cambia el signo de $n\sin\theta-n_t$.",
            r"El ancho del intervalo se multiplica por $0.618$ en cada iteración de la sección áurea y por $1/2$ en cada paso de la bisección: ambas convergen linealmente, con certeza, pero sin acelerar. Los dos métodos exigen que el mínimo (o el cambio de signo) esté dentro del intervalo inicial y que la función sea unimodal ahí; si $T(x)$ tuviera varios mínimos, el método podría quedarse con uno local. La limitación más sutil es el redondeo: cerca de un mínimo $T$ es plana, así que la sección áurea no puede distinguir puntos más cercanos que $\sqrt{\epsilon_m}$ en relativo.",
            rf"Se tomó el intervalo inicial $[0,d]={fx(d, 4)}$ m, que contiene seguro al punto de incidencia (un rayo no puede cruzar la interfaz fuera del segmento entre fuente y destino), 200 iteraciones para la sección áurea y 80 para la bisección, muy por encima de lo necesario para llegar al piso de redondeo. El resultado se interpreta como el punto de incidencia que verifica Fermat, y la coincidencia con el punto de la ley de Snell ({fx(xs, 6)} m) es lo que se pretende comprobar.",
            r"El método es independiente de la ley de Snell porque solo evalúa longitudes de camino ópticas $n_1\sqrt{x^2+h_1^2}+n_2\sqrt{(d-x)^2+h_2^2}$ y compara valores; los ángulos nunca aparecen. Que el mínimo caiga justo en el punto que predice $n_1\sin\theta_1=n_2\sin\theta_2$ es un resultado, no una hipótesis. Lo mismo ocurre con el ángulo crítico: la bisección solo pregunta si $n\sin\theta\le1$ y nunca calcula $\arcsin(1/n)$.",
        ],
        numeric_code=f"""g = (sqrt(5) - 1) / 2                     # razon aurea 0.618
a, b = 0.0, d                             # intervalo que seguro contiene al punto de incidencia
c, e = b - g*(b - a), a + g*(b - a)       # dos puntos interiores
for _ in range(200):                      # iteraciones de la busqueda
    if T(c) < T(e):                       # T(x) = (n1*hypot(x,h1) + n2*hypot(d-x,h2)) / c_luz
        b = e                             # el minimo esta en [a, e]
    else:
        a = c                             # el minimo esta en [c, b]
    c, e = b - g*(b - a), a + g*(b - a)   # intervalo mas corto: ancho * 0.618
x_fermat = 0.5 * (a + b)                  # punto de incidencia que minimiza el tiempo
# --- angulo critico por biseccion ---
lo, hi = 0.0, 90.0                        # angulos de incidencia en grados
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if n_denso * sin(radians(mid)) <= n_fino:   # todavia existe rayo refractado
        lo = mid                          # el critico esta arriba de mid
    else:
        hi = mid                          # el critico esta abajo de mid
theta_c = 0.5 * (lo + hi)                 # el ancho se divide entre 2 en cada paso
""",
        numeric_converge=[
            ("Sección áurea: ancho del intervalo y error del punto",
             f"Se repite la búsqueda con $k$ iteraciones y se compara con el punto de Snell $x={fx(xs, 6)}$ m. El ancho decrece como $0.618^k$ (la última columna es la razón por iteración entre filas, siempre {fx(ratio, 4)}) y el error sigue al ancho hasta cerca de $k=33$; después se estanca en {ps(floor_obs, 2)} m, aunque el ancho siga bajando.",
             [tabg]),
            ("Bisección del ángulo crítico",
             f"Aquí el error es siempre $\\le$ el ancho $90^\\circ/2^k$ y la pendiente de $\\log_2$(error) contra $k$ es $-1$; cada 10 iteraciones ganan unas 3 cifras decimales. El error sigue al ancho mientras es mayor que el redondeo de un número cercano a $41.8^\\circ$ (unos $7\\times10^{{-15}}$ grados); con $k=60$ vale {ps(eb[60], 2)} grados, ya sin mejora posible. A diferencia de la sección áurea, aquí no hay piso de $\\sqrt{{\\epsilon_m}}$ porque la bisección solo compara un signo, no valores casi iguales.",
             [tabb]),
            ("Por qué se estanca la sección áurea",
             f"Cerca del mínimo $T(x)\\approx T_0+\\tfrac12T''(x-x^*)^2$, y una diferencia entre $T(c)$ y $T(e)$ menor que $\\epsilon_mT_0$ no se puede ver. El piso predicho es $\\delta x\\approx\\sqrt{{2\\epsilon_mL_0/L''}}$, con la longitud óptica $L_0={fx(l0, 4)}$ m y $L''={fx(lpp, 4)}\\ \\mathrm{{m^{{-1}}}}$; da {ps(floor_pred, 2)} m, del mismo orden que lo observado ({ps(floor_obs, 2)} m). Es un límite del tipo de método, no un error de programación, y es mucho menor que cualquier tolerancia práctica.",
             [rf"\delta x_{{pred}}=\sqrt{{\dfrac{{2\epsilon_mL_0}}{{L''}}}}={sci(floor_pred, 2)}\ \mathrm{{m}},\qquad |x_{{num}}-x_{{Snell}}|={sci(floor_obs, 2)}\ \mathrm{{m}}"]),
            ("Criterio de confianza",
             "El resultado se acepta si el intervalo final es mucho menor que la tolerancia deseada y por encima del piso $\\sqrt{\\epsilon_m}$ relativo. Aquí unas 35 iteraciones (de las 200 ejecutadas) ya dan un punto de incidencia con error menor que $10^{-7}$ m; más iteraciones no añaden información. La verificación de Fermat es concluyente: el mínimo del tiempo cae en el punto de Snell dentro de esa tolerancia.",
             [rf"x_{{Fermat}}={m15.fermat_numeric(d):.8f}\ \mathrm{{m}},\qquad x_{{Snell}}={xs:.8f}\ \mathrm{{m}};\qquad \theta_{{c,num}}={m15.critical_angle_numeric():.8f}^\circ,\ \theta_{{c,anal}}={tc:.8f}^\circ"]),
        ],
    )


def _prism() -> dict:
    n = float(m15.cauchy(550))
    A = m15.APEX_DEG
    da = float(m15.min_deviation_analytic(n))
    th_m = float(np.degrees(np.arcsin(n * np.sin(np.deg2rad(A / 2)))))
    dev = lambda t: m15.deviation(t, n)  # noqa: E731
    ks = (5, 10, 20, 30, 60, 100)
    res = {k: _golden(dev, 10.0, 85.0, k) for k in ks}
    tab = _tab("rrrr", [[r"k", r"\mathrm{ancho}\,(^\circ)", r"|\theta_k-\theta_m|\,(^\circ)", r"|\delta_k-\delta_{min}|\,(^\circ)"]] +
               [[str(k), sci(res[k][1], 3), sci(abs(res[k][0] - th_m), 3), sci(abs(dev(res[k][0]) - da), 3)] for k in ks])
    h = 1e-3
    dpp = (dev(th_m + h) - 2 * dev(th_m) + dev(th_m - h)) / h**2
    floor_pred = float(np.sqrt(2 * EPS * da / dpp))
    lams = (400, 450, 550, 700)
    rows = []
    for lm in lams:
        nl = float(m15.cauchy(lm))
        rows.append([str(lm), fx(nl, 5), fx(float(m15.min_deviation_analytic(nl)), 6), fx(m15.min_deviation_numeric(nl), 6),
                     sci(abs(m15.min_deviation_numeric(nl) - float(m15.min_deviation_analytic(nl))), 2)])
    tabl = _tab("rrrrr", [[r"\lambda\,(\mathrm{nm})", r"n", r"\delta_{anal}\,(^\circ)", r"\delta_{num}\,(^\circ)", r"|\Delta|\,(^\circ)"]] + rows)
    dn = 1e-4
    sens = (float(m15.min_deviation_analytic(n + dn)) - da) / dn
    return dict(
        numeric_intro=[
            r"Aquí se pide la desviación mínima de un prisma, que es el mínimo de la función $\delta(\theta_1)$ obtenida al trazar el rayo a través de las dos caras con la ley de Snell. Se resuelve numéricamente porque se pretende comprobar la fórmula $\delta_{min}=2\arcsin[n\sin(\alpha/2)]-\alpha$ sin suponer que el mínimo ocurre en el trayecto simétrico: la función se evalúa trazando el rayo y se busca su valle con la sección áurea.",
            r"La función de mérito $\delta(\theta_1)=\theta_1+\theta_2-\alpha$ se calcula en tres pasos: $\sin r_1=\sin\theta_1/n$ (primera cara), $r_2=\alpha-r_1$ (geometría del prisma) y $\sin\theta_2=n\sin r_2$ (segunda cara). La sección áurea reduce el intervalo $[a,b]$ con las reglas $c=b-\varphi(b-a)$, $e=a+\varphi(b-a)$ y $\varphi=0.618$: si $\delta(c)<\delta(e)$ se descarta $[e,b]$, y si no, $[a,c]$.",
            r"El ancho se reduce por $0.618$ en cada iteración (convergencia lineal). Pero hay una propiedad útil: cerca del mínimo $\delta\approx\delta_{min}+\tfrac12\delta''(\theta-\theta_m)^2$, así que el error del valor mínimo es cuadrático en el error del ángulo y se pierde más rápido: el valor $\delta_{min}$ llega al redondeo con la mitad de las iteraciones, aunque el ángulo $\theta_m$ queda limitado a $\sqrt{\epsilon_m}$ relativo. El método falla si el intervalo inicial no contiene el mínimo o si el rayo sufre reflexión total interna dentro del prisma ($n\sin r_2\ge1$), caso en que el código devuelve NaN.",
            r"Se tomó el intervalo $[10^\circ,85^\circ]$, que contiene el mínimo (cerca de $49^\circ$) y donde $\delta(\theta_1)$ tiene un solo valle para los índices usados, y 100 iteraciones, suficientes para llegar al piso de redondeo. El índice se toma de la fórmula de Cauchy $n=1.50+5000/\lambda^2$ para 400, 550 y 700 nm; el resultado es la desviación de cada color, cuya diferencia es la dispersión angular.",
            r"La independencia con el analítico es clara: el método jamás usa la condición de simetría $\theta_1=\theta_2$ ni la fórmula de $\delta_{min}$. Que el mínimo encontrado caiga exactamente en el ángulo simétrico $\arcsin[n\sin(\alpha/2)]$ es un resultado de la búsqueda, no una entrada.",
        ],
        numeric_code="""g = (sqrt(5) - 1) / 2                     # razon aurea
def desviacion(t1_grados, n, apice=60.0):
    t1, A = radians(t1_grados), radians(apice)
    r1 = arcsin(sin(t1) / n)              # Snell en la primera cara
    r2 = A - r1                           # el angulo interno de la segunda cara
    s = n * sin(r2)
    if s >= 1: return nan                 # reflexion total interna: sin rayo emergente
    return degrees(t1 + arcsin(s) - A)    # desviacion total delta
lo, hi = 10.0, 85.0                       # intervalo de angulos de incidencia
c, e = hi - g*(hi - lo), lo + g*(hi - lo) # puntos interiores
for _ in range(100):
    if desviacion(c, n) < desviacion(e, n):
        hi = e                            # el minimo esta en [lo, e]
    else:
        lo = c                            # el minimo esta en [c, hi]
    c, e = hi - g*(hi - lo), lo + g*(hi - lo)
delta_min = desviacion(0.5*(lo + hi), n)  # valor en el minimo hallado
""",
        numeric_converge=[
            ("Error del ángulo y del valor mínimo contra las iteraciones",
             f"Con $n={fx(n, 5)}$ ($\\lambda=550$ nm) se repite la búsqueda con $k$ iteraciones. El ancho cae como $0.618^k$, el ángulo converge igual hasta un piso de {ps(abs(res[100][0] - th_m), 2)} grados, y el valor de la desviación converge al doble de velocidad (error cuadrático) hasta {ps(abs(dev(res[60][0]) - da), 2)} grados con $k=60$, el redondeo. Es decir, aunque el ángulo del mínimo sea impreciso, la cantidad que se pide ya es exacta a 12 cifras con $k=30$.",
             [tab]),
            ("Piso de redondeo del ángulo",
             f"Con la segunda derivada numérica $\\delta''={fx(dpp, 4)}\\ ^\\circ/(^\\circ)^2$ en el mínimo, el ángulo no se puede fijar mejor que $\\sqrt{{2\\epsilon_m\\delta_{{min}}/\\delta''}}={sci(floor_pred, 2)}\\ \\mathrm{{grados}}$. El valor medido ({ps(abs(res[100][0] - th_m), 2)} grados) es del mismo orden; es donde se estanca la columna del ángulo y es mucho menor que cualquier precisión angular de laboratorio.",
             [rf"\theta_m={th_m:.6f}^\circ,\qquad \delta_{{min}}={da:.8f}^\circ"]),
            ("Resultado para varios colores y sensibilidad al índice",
             f"Con 100 iteraciones la diferencia con la fórmula analítica es del orden del redondeo para todos los colores. La mayor fuente de incertidumbre real no es el método, sino el índice: la derivada $\\partial\\delta_{{min}}/\\partial n={fx(sens, 3)}\\ ^\\circ$ por unidad de índice implica que un error de $10^{{-4}}$ en $n$ mueve la desviación unos {fx(sens * 1e-4, 4)} grados.",
             [tabl]),
        ],
    )


# ===================================================================== Capitulo 16
def _lens() -> dict:
    f = m16.focal_lensmaker()
    s = 0.3
    sp = m16.image_distance_analytic(s, f)
    e0 = m16.lens_image_numeric(s, h=1e-6) - sp
    hs = (1e-3, 5e-4, 2.5e-4, 1.25e-4)
    ab = {h: m16.lens_image_numeric(s, h=h) - sp - e0 for h in hs}
    rows = []
    for i, h in enumerate(hs):
        rows.append([sci(h, 3), fx(m16.lens_image_numeric(s, h=h), 8), sci(ab[h], 3), (fx(_slope(hs[i - 1], abs(ab[hs[i - 1]]), h, abs(ab[h])), 3) if i else r"-")])
    tab = _tab("rrrr", [[r"h\,(\mathrm{m})", r"s'_{num}\,(\mathrm{m})", r"s'_{num}-s'_{num}(h\to0)\,(\mathrm{m})", r"\mathrm{pendiente}"]] + rows)
    coef = ab[1e-3] / 1e-3**2
    rm, sm = 0.4, 0.6
    spm = m16.mirror_image_analytic(sm, rm)
    hm = (4e-3, 2e-3, 1e-3, 5e-4)
    em = {h: m16.mirror_image_numeric(sm, rm, h) - spm for h in hm}
    rowsm = []
    for i, h in enumerate(hm):
        rowsm.append([sci(h, 3), sci(m16.mirror_image_numeric(sm, rm, h), 8), sci(em[h], 3), (fx(_slope(hm[i - 1], abs(em[hm[i - 1]]), h, abs(em[h])), 3) if i else r"-")])
    tabm = _tab("rrrr", [[r"h\,(\mathrm{m})", r"s'_{num}\,(\mathrm{m})", r"s'_{num}-s'_{anal}\,(\mathrm{m})", r"\mathrm{pendiente}"]] + rowsm)
    ts = (5e-6, 1e-5, 2e-5, 4e-5)
    th_shift = {t: m16.lens_image_numeric(s, h=1e-5, thickness=t) - sp for t in ts}
    tabt = _tab("rrr", [[r"t\,(\mathrm{m})", r"s'_{num}-s'_{anal}\,(\mathrm{m})", r"(s'_{num}-s'_{anal})/t"]] +
                [[sci(t, 2), sci(th_shift[t], 4), fx(th_shift[t] / t, 4)] for t in ts])
    return dict(
        numeric_intro=[
            r"En lugar de usar la ecuación de las lentes delgadas, el método traza un rayo real: parte del objeto axial, llega a la primera superficie esférica, se refracta con la ley de Snell en forma vectorial, atraviesa el vidrio, se refracta de nuevo en la segunda superficie, y el punto donde el rayo corta el eje es la imagen. Se necesita porque $1/s+1/s'=1/f$ es una aproximación (paraxial y de lente delgada) y la única forma de ver cuánto se equivoca es calcular el trayecto exacto de un rayo.",
            r"En cada superficie esférica de centro $\vec C$ y radio $R$ se resuelve la intersección rayo-esfera $|\vec p+t\vec d-\vec C|^2=R^2$ (una ecuación cuadrática en $t$) y se toma la raíz que corresponde al vértice. Luego se aplica la ley de Snell vectorial: con $\eta=n_1/n_2$, normal unitaria $\hat n$ y $\cos\theta_i=-\hat n\cdot\hat d$, el rayo refractado es $\hat d'=\eta\hat d+\left(\eta\cos\theta_i-\sqrt{1-\eta^2(1-\cos^2\theta_i)}\right)\hat n$. El espejo es análogo, con la reflexión $\hat d'=\hat d-2(\hat d\cdot\hat n)\hat n$.",
            r"No hay un parámetro de discretización: la traza es exacta salvo redondeo. El error del resultado frente a la lente delgada viene del modelo, no de la aritmética. Para un rayo a altura $h$ aparece la aberración esférica longitudinal, que crece como $h^2$, y por el espesor finito de la lente (la distancia entre vértices), que desplaza la imagen una cantidad proporcional al espesor. El método falla si el rayo no corta la superficie (discriminante negativo) o si $\eta\sin\theta_i>1$ (reflexión total).",
            rf"Se eligió un rayo a $h=1$ mm del eje (el 1 % de la distancia focal $f={fx(f, 2)}$ m, lo bastante paraxial para que el error sea pequeño pero con tamaño suficiente para que el corte con el eje no sea ruidoso) y un espesor de $2\times10^{{-5}}$ m, tan delgado que la lente es prácticamente delgada. El espejo cóncavo de $R=40$ cm se traza igual con $h=1$ mm. El resultado es la distancia imagen $s'$ medida desde el vértice.",
            r"El método es independiente porque nunca usa $f$, ni el aumento, ni la aproximación $\sin\theta\approx\theta$: usa solo la geometría de dos esferas y la ley de Snell exacta, y la ecuación de las lentes aparece solo para comparar. Si ambos coinciden a una parte en $10^4$, eso valida la aproximación paraxial y la ecuación del fabricante de lentes.",
        ],
        numeric_code="""def refractar(p, d, cx, R, n_ini, n_fin):
    c = array([cx, 0.0]); oc = p - c           # centro de la esfera y posicion relativa
    b = oc @ d
    disc = b*b - (oc @ oc - R**2)              # discriminante de |p + t d - c|^2 = R^2
    ts = [-b - sqrt(disc), -b + sqrt(disc)]    # dos cortes del rayo con la esfera
    hit = min((p + t*d for t in ts if t > 0), key=lambda q: abs(q[0] - (cx - R)))   # el mas cercano al vertice
    nrm = (hit - c) / abs(R)                   # normal unitaria
    if nrm @ d > 0: nrm = -nrm                 # orientada contra el rayo
    cos_i = -(nrm @ d)
    r = n_ini / n_fin
    k = 1 - r**2 * (1 - cos_i**2)              # k < 0: reflexion total
    return hit, r*d + (r*cos_i - sqrt(k)) * nrm   # Snell vectorial
p = array([-s, 0.0])                           # objeto axial a distancia s
d = array([s, h]) / hypot(s, h)                # rayo que sube con altura h en la lente
hit1, d1 = refractar(p, d, R1, R1, 1.0, n)     # primera superficie (vertice en x = 0)
hit2, d2 = refractar(hit1, d1, espesor + R2, R2, n, 1.0)   # segunda superficie
t = -hit2[1] / d2[1]                           # parametro donde el rayo cruza el eje (y = 0)
s_imagen = hit2[0] + t * d2[0]                 # posicion de la imagen
""",
        numeric_converge=[
            ("Aberración esférica contra la altura del rayo",
             f"Con la ecuación de las lentes $s'={fx(sp, 5)}$ m (para $s=0.3$ m). El trazado converge a un valor límite cuando $h\\to0$, que difiere en {ps(e0, 3)} m (el efecto del espesor, ver el siguiente paso). Restando ese límite, la desviación debida a la altura es la aberración esférica, y la pendiente en escala log-log es 2: al reducir $h$ a la mitad la aberración cae a la cuarta parte, con $\\Delta s'\\approx{sci(coef, 3)}\\,h^2$.",
             [tab]),
            ("Efecto del espesor de la lente",
             f"La lente de $2\\times10^{{-5}}$ m es una lente delgada solo de forma aproximada. Con $h=10^{{-5}}$ m (aberración despreciable) el corrimiento de la imagen es proporcional al espesor $t$: el cociente vale prácticamente {fx(th_shift[2e-5] / 2e-5, 3)} para todos los espesores, así que la corrección es lineal, $\\Delta s'\\approx{fx(th_shift[2e-5] / 2e-5, 3)}\\,t$.",
             [tabt]),
            ("Espejo cóncavo y criterio de confianza",
             f"El espejo ($R=0.4$ m, $s=0.6$ m, $s'_{{anal}}={fx(spm, 5)}$ m) muestra la misma ley, con pendiente 2. Para el valor usado $h=1$ mm el error es {ps(em[1e-3], 2)} m (relativo {ps(abs(em[1e-3]) / spm, 2)}), de modo que la ecuación paraxial es confiable cuando $h/R\\lesssim10^{{-2}}$, y la desviación crece rápidamente: con $h=4$ mm es 16 veces mayor. La traza es confiable mientras el rayo no esté cerca del borde y el discriminante sea positivo.",
             [tabm]),
        ],
    )


def _twolens() -> dict:
    sys_ = lambda s, d=m16.SEP: m16.thin_lens_matrix(m16.F2) @ m16.free_matrix(d) @ m16.thin_lens_matrix(m16.F1) @ m16.free_matrix(s)  # noqa: E731
    ss = (0.3, 0.5, 1.0, 5.0)
    rows = []
    for s in ss:
        za, ma = m16.two_lens_analytic(s)
        zn, mn = m16.two_lens_matrix(s)
        mm = sys_(s)
        z = zn
        bres = mm[0, 1] + z * mm[1, 1]
        rows.append([fx(s, 1), fx(za, 8), fx(zn, 8), sci(abs(za - zn), 2), sci(abs(ma - mn), 2), sci(abs(bres), 2)])
    tab1 = _tab("rrrrrr", [[r"s", r"s'_{anal}", r"s'_{mat}", r"|\Delta s'|", r"|\Delta m|", r"|B_{res}|"]] + rows)
    m5 = sys_(0.5)
    det = float(np.linalg.det(m5))
    cond = float(np.linalg.cond(m5))
    ds = (0.15, 0.30, 0.40, 0.43, 0.4333)
    rows2 = []
    for dd in ds:
        zn, mn = m16.two_lens_matrix(0.5, d=dd)
        mm = sys_(0.5, dd)
        rows2.append([fx(dd, 4), fx(float(mm[1, 1]), 5), fx(zn, 4), fx(m16.two_lens_analytic(0.5, d=dd)[0], 4), sci(abs(zn - m16.two_lens_analytic(0.5, d=dd)[0]) / abs(zn), 2)])
    tab2 = _tab("rrrrr", [[r"d\,(\mathrm{m})", r"D=\mathrm{Mat}_{22}", r"s'_{mat}\,(\mathrm{m})", r"s'_{anal}\,(\mathrm{m})", r"\varepsilon_{rel}"]] + rows2)
    s0 = m16.two_lens_matrix(0.5)[0]
    rel = 1e-3
    f1p = m16.F1 * (1 + rel)
    sp1 = m16.image_distance_analytic(0.5, f1p)
    s2p = m16.image_distance_analytic(m16.SEP - sp1, m16.F2)
    sens = (s2p - s0) / s0 / rel
    return dict(
        numeric_intro=[
            r"El método matricial (óptica de rayos $ABCD$) representa cada elemento óptico por una matriz $2\times2$ que actúa sobre el vector $(y,\theta)$ de altura y pendiente de un rayo. Se usa porque encadenar muchos elementos con la fórmula de las lentes obliga a llevar la cuenta de signos de imágenes virtuales y objetos virtuales, mientras que el producto de matrices lo hace automáticamente y sirve igual para dos que para veinte elementos.",
            r"La propagación libre una distancia $d$ es $T(d)=\begin{pmatrix}1&d\\0&1\end{pmatrix}$ y la lente delgada es $L(f)=\begin{pmatrix}1&0\\-1/f&1\end{pmatrix}$. La matriz del sistema objeto $\to$ $L_1\to$ $L_2$ es $M=L(f_2)\,T(d)\,L(f_1)\,T(s)$ (el orden es inverso al del recorrido porque las matrices actúan por la izquierda). Para llegar a un plano a distancia $z$ de la segunda lente, se multiplica por $T(z)$ y se pide que el elemento $B$ de $T(z)M$ sea cero: $B+z\,D=0$, lo que da $z=-B/D$ sin resolver ninguna iteración, y el aumento es $A+zC$.",
            r"No hay error de truncamiento porque la aritmética es exacta en la aproximación paraxial: solo hay redondeo, de orden $\epsilon_m\approx10^{-16}$ por multiplicación. La limitación es física y de condicionamiento: la fórmula $z=-B/D$ diverge cuando $D\to0$, es decir cuando la imagen se va al infinito (el objeto intermedio cae en el foco de la segunda lente), y en ese caso el problema está mal condicionado. Tampoco incluye aberraciones.",
            rf"Los parámetros son los de la escena: $f_1=20$ cm, $f_2=10$ cm, separación $d=15$ cm y objeto a $s=50$ cm, con el objeto intermedio virtual ($s_2<0$) que la fórmula en cadena tendría que tratar con cuidado. La matriz del sistema tiene determinante {fx(det, 6)} (la unidad, como corresponde a un sistema en el mismo medio), y número de condición {fx(cond, 1)}, moderado.",
            r"Es independiente de la solución analítica porque esta resuelve dos veces la ecuación $1/s+1/s'=1/f$, pasando la imagen de la primera lente como objeto de la segunda, mientras que el método matricial nunca calcula esa imagen intermedia: solo multiplica matrices. Que los dos concuerden hasta el redondeo comprueba el manejo de signos del objeto virtual.",
        ],
        numeric_code=f"""def lente(f):   return array([[1.0, 0.0], [-1.0/f, 1.0]])   # matriz de lente delgada (y, theta)
def libre(d):   return array([[1.0, d], [0.0, 1.0]])         # propagacion libre una distancia d
f1, f2, sep, s = 0.20, 0.10, 0.15, 0.50      # lentes de 20 y 10 cm, separadas 15 cm, objeto a 50 cm
M = lente(f2) @ libre(sep) @ lente(f1) @ libre(s)   # orden inverso al del recorrido de la luz
A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]  # elementos del sistema objeto -> lente 2
z = -B / D                                   # el plano imagen anula B: B + z D = 0
aumento = A + z * C                          # elemento A del sistema completo: aumento lateral
residuo = B + z * D                          # debe ser ~1e-17 (comprobacion)
det_M = A*D - B*C                            # debe valer 1 (sistema en un solo medio)
""",
        numeric_converge=[
            ("Verificación cruzada contra la solución analítica",
             "No hay parámetro de discretización que variar, así que el estudio es una verificación cruzada: se resuelve el mismo sistema con el método en cadena y con matrices para distintas distancias objeto $s$. La columna $|B_{res}|$ es el residuo de la condición que define el plano imagen. Todas las discrepancias están en el nivel del redondeo.",
             [tab1]),
            ("Condicionamiento al variar la separación",
             f"La matriz es sencilla de multiplicar pero la imagen puede ser muy sensible a la geometría. Con $d$ cercano a $0.4333$ m, el objeto intermedio ($s_1'=1/3$ m) queda a $f_2=0.10$ m de la segunda lente, $D\\to0$ y $s'_2=-B/D\\to\\infty$. La tabla muestra cómo $D$ se aproxima a cero y $s'$ crece; el error relativo entre matriz y cadena sube de $10^{{-16}}$ a unos $8\\times10^{{-13}}$ por la pérdida de cifras al dividir entre un $D$ pequeño.",
             [tab2]),
            ("Sensibilidad a las incertidumbres y criterio de confianza",
             f"Un error relativo de $10^{{-3}}$ en $f_1$ cambia la posición de la imagen final en una fracción relativa de {fx(sens * 1e-3, 5)} (amplificación {fx(sens, 2)}), así que la incertidumbre de la imagen la dominan las distancias focales medidas, no el método. El resultado es confiable si $\\det M=1$ con tolerancia de redondeo ({fx(det, 8)} aquí), si el residuo $|B+zD|\\lesssim10^{{-15}}$ y si $D$ no es pequeño.",
             [rf"\det M={fx(det, 10)},\qquad \mathrm{{cond}}(M)={fx(cond, 2)},\qquad s'_{{mat}}={s0:.8f}\ \mathrm{{m}}"]),
        ],
    )


# ===================================================================== Capitulo 17
def _young() -> dict:
    lam, d, L = m17.LAM, m17.D_SLIT, m17.L_SCR
    ya = m17.fringe_position_analytic(1)
    yn = m17.bright_fringe_numeric(1)
    f = lambda y, big_l=L: np.hypot(big_l, y + d / 2) - np.hypot(big_l, y - d / 2) - lam  # noqa: E731
    ks = (5, 10, 20, 30, 40)
    rows = []
    for k in ks:
        lo, hi = 0.0, 3 * lam * L / d + 1e-9
        for _ in range(k):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        rows.append([str(k), sci(hi - lo, 3), sci(abs(0.5 * (lo + hi) - yn), 3), sci(abs(0.5 * (lo + hi) - ya), 3)])
    tab1 = _tab("rrrr", [[r"k", r"\mathrm{ancho}\,(\mathrm{m})", r"|y_k-y_\infty|\,(\mathrm{m})", r"|y_k-y_{anal}|\,(\mathrm{m})"]] + rows)
    Ls = (1.5, 0.15, 0.015)
    rows2 = []
    for bl in Ls:
        a, n_ = m17.fringe_position_analytic(1, big_l=bl), m17.bright_fringe_numeric(1, big_l=bl)
        rows2.append([fx(bl, 3), sci(a, 7), sci(n_, 7), sci(abs(a - n_), 3), sci(abs(a - n_) * bl**2, 3)])
    tab2 = _tab("rrrrr", [[r"L\,(\mathrm{m})", r"y_{anal}\,(\mathrm{m})", r"y_{num}\,(\mathrm{m})", r"|\Delta y|", r"|\Delta y|\,L^2"]] + rows2)
    # redondeo: biseccion repetida en precision extendida
    fl = lambda y: np.hypot(np.longdouble(L), y + np.longdouble(d) / 2) - np.hypot(np.longdouble(L), y - np.longdouble(d) / 2) - np.longdouble(lam)  # noqa: E731
    lo, hi = np.longdouble(0), np.longdouble(3 * lam * L / d + 1e-9)
    for _ in range(100):
        mid = (lo + hi) / 2
        if fl(mid) < 0:
            lo = mid
        else:
            hi = mid
    yl = (lo + hi) / 2
    round_obs = float(abs(np.longdouble(yn) - yl))
    cancel = EPS * L / lam
    return dict(
        numeric_intro=[
            r"La posición de la franja brillante se obtiene sin la aproximación de campo lejano: se halla por bisección la raíz de la diferencia exacta de caminos $f(y)=r_2(y)-r_1(y)-m\lambda$, con $r_{1,2}=\sqrt{L^2+(y\mp d/2)^2}$. Esta ecuación no se puede despejar en $y$ (hay dos raíces cuadradas), pero $f$ es monótona creciente en $y$, de modo que la bisección converge con certeza.",
            r"El algoritmo mantiene un intervalo $[y_{lo},y_{hi}]$ con $f(y_{lo})<0<f(y_{hi})$ y lo divide a la mitad: se evalúa $y_{mid}$, y si $f(y_{mid})<0$ la raíz está arriba ($y_{lo}\leftarrow y_{mid}$), y si no, abajo. Tras $k$ pasos el ancho es $(y_{hi}-y_{lo})_0/2^k$, y la raíz queda acotada con error $\le$ ancho. Es lineal (ganancia de $\log_{10}2\approx0.3$ cifras por paso) pero infalible mientras $f$ cambie de signo en el intervalo inicial.",
            rf"Se partió de $[0,\,3m\lambda L/d]$, tres veces la posición aproximada de la franja, porque ahí $f(y_{{hi}})>0$ seguro; se hicieron 100 iteraciones, mucho más de las 53 que agotan la precisión doble. La limitación numérica es la cancelación: $f$ es la diferencia de dos longitudes de $\approx{fx(L, 1)}$ m que debe resolverse a $\lambda=600$ nm, y eso pierde unas $\log_{{10}}(L/\lambda)\approx6$ de las 16 cifras: queda un error relativo de $\epsilon_mL/\lambda\approx{sci(cancel, 1)}$ en $f$.",
            rf"El resultado es la posición de la franja de orden $m$, por ejemplo $y_1={yn * 1e3:.6f}$ mm para $m=1$. Difiere de la fórmula analítica $L\tan(\arcsin(m\lambda/d))$ en {ps(abs(yn - ya), 2)} m, pero esa diferencia no es error numérico: la fórmula analítica supone que $r_2-r_1=d\sin\theta$ (aproximación de campo lejano), y la bisección usa el camino exacto. La diferencia mide cuánto vale esa aproximación.",
            r"El método es independiente del analítico porque nunca usa $d\sin\theta=m\lambda$, ni $\tan$ ni $\arcsin$: solo calcula distancias a las dos rendijas con el teorema de Pitágoras y busca el cero de su diferencia. La coincidencia hasta $10^{-11}$ m comprueba que el modelo geométrico de las dos fuentes y la fórmula de los máximos describen lo mismo.",
        ],
        numeric_code=f"""lam, d, L, m = 600e-9, 0.25e-3, 1.5, 1       # longitud de onda, separacion de rendijas, distancia a pantalla, orden
def f(y):                                      # diferencia de camino exacta menos m*lambda
    r1 = hypot(L, y - d/2)                     # distancia de la rendija superior al punto y
    r2 = hypot(L, y + d/2)                     # distancia de la rendija inferior
    return r2 - r1 - m*lam                     # cero cuando hay interferencia constructiva de orden m
lo, hi = 0.0, 3*m*lam*L/d + 1e-9              # intervalo: f(lo) < 0 < f(hi) (3 veces la posicion aproximada)
for _ in range(100):                           # 100 pasos: el ancho cae a ~1e-3/2^100 (agota la doble precision)
    mid = 0.5 * (lo + hi)                      # punto medio
    if f(mid) < 0:                             # todavia falta camino: la raiz esta mas arriba
        lo = mid
    else:                                      # ya pasamos: la raiz esta mas abajo
        hi = mid
y_franja = 0.5 * (lo + hi)                     # posicion de la franja brillante de orden m
""",
        numeric_converge=[
            ("Bisección: el error se divide por 2 en cada paso",
             f"Se ejecutan $k$ pasos y se compara con el valor convergido $y_\\infty={yn * 1e3:.9f}$ mm. El ancho cae a la mitad en cada paso (de {ps(3 * lam * L / d, 2)} m a {ps(3 * lam * L / d / 2**40, 2)} m en 40 pasos), y el error acotado por él lo sigue. El error contra la fórmula analítica se estanca en un valor fijo de {ps(abs(yn - ya), 2)} m, que es la aproximación de campo lejano y no tiene que ver con $k$.",
             [tab1]),
            ("Aproximación de campo lejano: la diferencia contra el analítico",
             "La fórmula analítica es exacta solo para $L\\gg d$. Al reducir $L$ la discrepancia entre los dos métodos crece como $1/L^2$ (la última columna es constante), con lo que se confirma que proviene de la aproximación y que el método numérico, que no la hace, es el más fiel. Con $L=1.5$ m, donde estamos, la aproximación se cumple a $10^{-11}$ m.",
             [tab2]),
            ("Redondeo por cancelación y criterio de confianza",
             f"La diferencia entre $y$ en doble precisión y la misma bisección en precisión extendida (64 bits de mantisa) es {ps(round_obs, 2)} m, consistente con la cota $\\epsilon_m(L/\\lambda)\\,y\\approx{sci(cancel * yn, 2)}$ m. El resultado es confiable cuando el ancho del intervalo baja de $10^{{-12}}$ m (unos 40 pasos aquí) y cuando $f(y_{{lo}})$ y $f(y_{{hi}})$ tienen signos opuestos al principio. Si alguna franja diera una raíz fuera del intervalo, el código dejaría de converger al valor correcto sin avisar.",
             [rf"y_1^{{num}}={yn * 1e3:.9f}\ \mathrm{{mm}},\qquad y_1^{{anal}}={ya * 1e3:.9f}\ \mathrm{{mm}}"]),
        ],
    )


def _film() -> dict:
    lams = np.array([400, 450, 500, 550, 600, 650, 700]) * 1e-9
    ra, rm = m17.reflectance_analytic(lams), m17.reflectance_matrix(lams)
    tab1 = _tab("rrrr", [[r"\lambda\,(\mathrm{nm})", r"R_{matriz}\,(\%)", r"R_{Airy}\,(\%)", r"|\Delta R|"]] +
                [[str(int(lams[i] * 1e9)), fx(rm[i] * 100, 6), fx(ra[i] * 100, 6), sci(abs(rm[i] - ra[i]), 2)] for i in range(len(lams))])
    eps_s = (-0.10, -0.05, 0.0, 0.05, 0.10)
    r_t = {e: float(m17.reflectance_matrix(550e-9, t=m17.T_FILM * (1 + e))[0]) for e in eps_s}
    tab2 = _tab("rrr", [[r"\varepsilon_t", r"R(550\ \mathrm{nm})\,(\%)", r"R-R_0\,(\%)"]] +
                [[f"{e * 100:+.0f}\\,\\%", fx(r_t[e] * 100, 4), fx((r_t[e] - r_t[0.0]) * 100, 4)] for e in eps_s])
    curv = (r_t[0.10] - r_t[0.0]) / (r_t[0.05] - r_t[0.0])
    nfs = (1.36, 1.38, 1.40)
    r_n = {nf: float(m17.reflectance_matrix(550e-9, n_film=nf)[0]) for nf in nfs}
    r0 = m17.bare_reflectance()
    dr01 = float(m17.reflectance_matrix(550e-9, t=m17.T_FILM * 1.001)[0]) - r_t[0.0]
    return dict(
        numeric_intro=[
            r"Se calcula la reflectancia de una película delgada con el método de la matriz de transferencia: cada capa se describe por una matriz característica $2\times2$ que relaciona los campos eléctrico y magnético en sus dos caras, y el sistema completo es el producto de las matrices. Se usa porque la suma de múltiples reflexiones internas en la película es una serie infinita; la matriz la resume en un solo producto y, a diferencia de la fórmula de Airy, se generaliza sin cambio a decenas de capas (espejos dieléctricos, filtros).",
            r"La matriz de una capa de índice $n_f$ y fase $\delta=2\pi n_ft/\lambda$ es $M=\begin{pmatrix}\cos\delta & i\sin\delta/n_f\\ i\,n_f\sin\delta & \cos\delta\end{pmatrix}$. Con el sustrato de índice $n_s$ se calcula $\begin{pmatrix}B\\C\end{pmatrix}=M\begin{pmatrix}1\\n_s\end{pmatrix}$, y el coeficiente de reflexión desde el aire ($n_0=1$) es $r=(n_0B-C)/(n_0B+C)$, de modo que $R=|r|^2$. Esto equivale a imponer la continuidad de los campos tangenciales en cada interfaz.",
            r"El método es exacto: no se corta ninguna serie ni se discretiza nada, y el único error es el de redondeo de la aritmética compleja (de orden $\epsilon_m\approx10^{-16}$ por operación). Las limitaciones son del modelo, no del cálculo: se supone incidencia normal, capas planas y paralelas, índices reales sin dispersión ni absorción, y luz perfectamente coherente en la capa (no hay promediado sobre un espesor irregular ni sobre el ancho de banda).",
            r"Los parámetros son los de un recubrimiento antirreflejante clásico: una capa de $n_f=1.38$ (como el fluoruro de magnesio) sobre vidrio de $n_s=1.52$, con espesor de cuarto de onda para 550 nm, $t=\lambda_0/(4n_f)$, y se evalúa a varias longitudes de onda para ver la banda de bajas pérdidas. El resultado, el porcentaje de luz reflejada, se compara con el 4.3 % del vidrio desnudo.",
            r"El método es independiente del analítico porque la fórmula de Airy suma explícitamente la serie geométrica de reflexiones con los coeficientes de Fresnel $r_{01}$ y $r_{12}$, mientras que la matriz parte de la continuidad de los campos y nunca calcula esos coeficientes por separado. Que ambos coincidan hasta el redondeo es una prueba cruzada de que las dos formulaciones describen la misma física.",
        ],
        numeric_code=f"""n_f, n_s, lam0 = 1.38, 1.52, 550e-9           # pelicula, sustrato, longitud de diseno
t = lam0 / (4 * n_f)                           # espesor: cuarto de onda en la pelicula
def reflectancia(lam):
    delta = 2*pi * n_f * t / lam               # fase de una pasada por la pelicula
    M = array([[cos(delta),         1j*sin(delta)/n_f],
               [1j*n_f*sin(delta),  cos(delta)]])    # matriz caracteristica de la capa
    B, C = M @ array([1.0, n_s])               # campos equivalentes en la entrada (sustrato semiinfinito)
    r = (1*B - C) / (1*B + C)                  # coeficiente de reflexion visto desde el aire (n0 = 1)
    return abs(r)**2                           # reflectancia = |r|^2
for lam in (400e-9, 450e-9, 550e-9, 700e-9):   # barrido de longitudes de onda
    R = reflectancia(lam)                      # una matriz 2x2 por longitud de onda
    print(lam, R)
R0 = ((1 - n_s) / (1 + n_s))**2                # vidrio desnudo, para comparar
""",
        numeric_converge=[
            ("Concordancia entre la matriz y la fórmula de Airy",
             "Como no hay parámetro de discretización, el estudio es una verificación cruzada entre dos formulaciones algebraicamente equivalentes sobre todo el rango visible. La última columna es la diferencia absoluta de reflectancias (en fracción, no en porcentaje): del orden de $10^{-17}$, o sea el último bit de la doble precisión.",
             [tab1]),
            ("Sensibilidad al espesor de la película",
             f"El diseño de cuarto de onda es un mínimo de $R$, así que un error de espesor lo aumenta de forma cuadrática: el aumento con $\\pm10\\,\\%$ es {fx(curv, 2)} veces el de $\\pm5\\,\\%$ (4 si fuese exactamente parabólico). Este tipo de tabla es el estudio de tolerancias real: un error de depósito del 5 % solo eleva la reflectancia de {fx(r_t[0.0] * 100, 3)} % a {fx(r_t[0.05] * 100, 3)} %.",
             [tab2]),
            ("Sensibilidad al índice y criterio de confianza",
             f"Con $n_f=1.36$, $1.38$ y $1.40$ (mismo espesor) la reflectancia a 550 nm es {fx(r_n[1.36] * 100, 3)} %, {fx(r_n[1.38] * 100, 3)} % y {fx(r_n[1.40] * 100, 3)} %: la incertidumbre del índice de la película pesa más que cualquier error numérico. El resultado es confiable cuando la película cumple las hipótesis (normal, sin absorción) y la reflectancia queda en $[0,1]$; el redondeo es $10^{{-17}}$, unos {fx(np.log10(dr01 / 1e-17), 0)} órdenes de magnitud por debajo de lo que cambia $R$ con un 0.1 % de error en el espesor ({ps(dr01, 2)}). Se reduce de {fx(r0 * 100, 3)} % (vidrio desnudo) a {fx(r_t[0.0] * 100, 3)} %.",
             [rf"R_0={r0 * 100:.4f}\,\%,\qquad R(550\ \mathrm{{nm}})={r_t[0.0] * 100:.4f}\,\%"]),
        ],
    )


# ===================================================================== Capitulo 18
def _grating() -> dict:
    lam, a, d, N = m18.LAM, m18.A_SLIT, m18.D_GRAT, m18.N_SLITS
    s0 = 0.2
    ex = float(m18.single_slit_analytic(s0))
    ns = (2, 4, 8, 16, 32, 64)
    es = {n: abs(float(m18.single_slit_numeric([s0], n=n)[0]) - ex) for n in ns}
    rows = [[str(n), fx(float(m18.single_slit_numeric([s0], n=n)[0]), 8), sci(es[n], 3), (fx(np.log2(es[ns[i - 1]] / es[n]), 2) if i else r"-")] for i, n in enumerate(ns)]
    tab1 = _tab("rrrr", [[r"n", r"I_{num}", r"|I_{num}-I_{anal}|", r"\log_2(\varepsilon_{n/2}/\varepsilon_n)"]] + rows)
    sin_t = np.array([0.05, 0.1, 0.2, 0.3])
    ms = (1, 2, 4, 8, 16, 32, 64)
    em = {m: float(np.max(np.abs(m18.grating_numeric(sin_t, m=m) - m18.grating_analytic(sin_t)))) for m in ms}
    rows2 = [[str(m), sci(em[m], 3), (fx(np.log2(em[ms[i - 1]] / em[m]), 2) if i else r"-")] for i, m in enumerate(ms)]
    tab2 = _tab("rrr", [[r"M", r"\max|I_{num}-I_{anal}|", r"\log_2(\varepsilon_{M/2}/\varepsilon_M)"]] + rows2)
    _, ang = m18.principal_maxima_analytic()

    def peaks(npts: int):
        th = np.linspace(0, np.deg2rad(89.0), npts)
        i = m18.grating_numeric(np.sin(th), lam, 1e-9, d, N, 1)
        idx = [j for j in range(1, npts - 1) if i[j] > i[j - 1] and i[j] >= i[j + 1] and i[j] > 0.9]
        return np.degrees(th[[0] + idx]), np.degrees(th[1] - th[0])

    gs = (2001, 20001, 200001)
    rows3 = []
    for g in gs:
        pk, dth = peaks(g)
        k = min(len(pk), len(ang))
        rows3.append([str(g), sci(dth, 3), str(len(pk)), sci(float(np.max(np.abs(pk[:k] - ang[:k]))), 3)])
    tab3 = _tab("rrrr", [[r"N_\theta", r"\Delta\theta\,(^\circ)", r"\mathrm{picos}", r"\max|\theta_m-\theta_{m,anal}|\,(^\circ)"]] + rows3)
    return dict(
        numeric_intro=[
            r"Esta escena aplica el principio de Huygens como algoritmo: cada rendija se divide en fuentes puntuales y la amplitud en una dirección es la suma coherente de sus fases. Es el método natural cuando el patrón de difracción no tiene una fórmula cerrada (rendijas de forma arbitraria, aberturas con perfil de transmisión) y, aquí, sirve para comprobar que las fórmulas $\mathrm{sinc}^2$ y $[\sin N\gamma/(N\sin\gamma)]^2$ salen de una suma pura de fases. Los máximos principales se encuentran buscando máximos locales en una malla de ángulos.",
            r"Para una rendija de ancho $a$ la amplitud es $A(\theta)=\dfrac1a\int_{-a/2}^{a/2}e^{ikx\sin\theta}dx$, que se calcula con Simpson compuesto, $\int\approx\dfrac h3[f_0+4f_1+2f_2+\cdots+f_n]$. Para la red de $N$ rendijas se suma la regla del punto medio, $A=\dfrac1{NM}\sum_{j=1}^{NM}e^{ikx_j\sin\theta}$ con $x_j$ en el centro de cada una de las $M$ celdas de cada rendija, y la intensidad es $I=|A|^2$. Los máximos principales son los nodos de la malla de $\theta$ donde $I$ es un máximo local y supera 0.9.",
            r"Simpson tiene error $O(h^4)$ y la regla del punto medio $O(h^2)$, porque el integrando es suave (analítico); pero el error crece con la oscilación del integrando: es proporcional a $(k\,a\sin\theta)^{p}$, con $p=4$ y $p=2$ respectivamente, así que se necesitan más nodos a ángulos grandes o con rendijas anchas. La búsqueda de picos tiene un error de malla: un máximo solo se puede localizar con una resolución de $\Delta\theta$, no mejor, así que su error es de orden $\Delta\theta/2$ (convergencia lineal con la malla). Además se supone aproximación de Fraunhofer (pantalla lejana).",
            rf"Se escogieron $n=400$ subintervalos de Simpson (la fase $kx\sin\theta$ da a lo sumo $a/\lambda=3.6$ ciclos en toda la rendija, de modo que bastan pocos nodos por ciclo) y $M=200$ puntos por rendija en la red, para que el error sea despreciable frente al paso angular; para los picos se usó una malla de 20 001 ángulos entre $0^\circ$ y $89^\circ$ ($\Delta\theta=0.00445^\circ$) con una sola fuente puntual por rendija (una rendija estrecha), pues los máximos principales no dependen del ancho. Con $d/\lambda={d / lam:.2f}$ hay órdenes $m=0,\ldots,10$, y como $d/a=3$ los órdenes 3, 6 y 9 quedan suprimidos.",
            r"El método es independiente de la fórmula analítica porque suma amplitudes complejas y nunca usa $\mathrm{sinc}$ ni la identidad $\sum e^{in\gamma}=\sin N\gamma/\sin\gamma$ ni la condición $d\sin\theta=m\lambda$. Por eso el acuerdo con las fórmulas cerradas confirma que ese comportamiento (picos estrechos, orden suprimido) sale de la interferencia de fuentes coherentes.",
        ],
        numeric_code=f"""k = 2*pi / lam                                  # numero de onda
# --- una rendija: Simpson con n subintervalos (n par) ---
x = linspace(-a/2, a/2, n + 1)                  # nodos en la rendija
amp = simpson(cos(k * x * sin_t), x) / a        # parte real de la amplitud (parte imaginaria es 0 por simetria)
I_rendija = amp**2                              # intensidad relativa
# --- red de N rendijas con M fuentes por rendija (punto medio) ---
centros = (arange(N) - (N-1)/2) * d             # centros de las rendijas
xs = (arange(M) + 0.5) / M * a - a/2            # fuentes puntuales dentro de cada rendija
pos = (centros[:, None] + xs[None, :]).ravel()  # N*M posiciones de las fuentes
amp = exp(1j * k * pos * sin_t).sum() / (N*M)   # suma coherente de fases
I_red = abs(amp)**2                             # intensidad relativa
# --- maximos principales por busqueda en malla ---
th = linspace(0, radians(89), 20001)            # malla fina de angulos
I = intensidad_red(sin(th), a=1e-9, M=1)        # 1 fuente por rendija: sin envolvente
picos = [j for j in range(1, len(th)-1) if I[j] > I[j-1] and I[j] >= I[j+1] and I[j] > 0.9]   # maximo local alto
""",
        numeric_converge=[
            ("Simpson en una rendija: orden 4",
             f"Con $\\sin\\theta={s0}$ (el argumento de la envolvente es $\\beta={np.pi * a * s0 / lam:.3f}$ rad) se compara con $\\mathrm{{sinc}}^2\\beta={fx(ex, 8)}$. Al duplicar $n$ el error se divide por 16 (la última columna tiende a 4), con lo que se confirma $O(h^4)$. Con los $n=400$ del código el error de truncamiento estimado es {ps(es[64] * (64 / 400) ** 4, 1)}, despreciable.",
             [tab1]),
            ("Punto medio en la red: orden 2",
             f"Se compara la red de $N=5$ rendijas con la fórmula analítica en cuatro ángulos y se toma el peor error. Con $M$ fuentes por rendija el error cae como $1/M^2$ (la última columna tiende a 2), como corresponde a la regla del punto medio. Con $M=200$ el error estimado es {ps(em[64] * (64 / 200) ** 2, 1)}, mucho menor que la resolución de los picos.",
             [tab2]),
            ("Búsqueda de máximos principales: error de la malla",
             f"El error de cada orden es un nodo de la malla, así que el peor error ({ps(float(np.max(np.abs(peaks(20001)[0][:min(len(peaks(20001)[0]), len(ang))] - ang[:min(len(peaks(20001)[0]), len(ang))]))), 2)} grados con la malla usada) queda acotado por $\\Delta\\theta/2$ (en la tabla entre $\\Delta\\theta/4$ y $\\Delta\\theta/2.5$) y cae casi linealmente con $N_\\theta$: cada factor 10 en los puntos reduce el error entre 5 y 16 veces, sin el orden limpio de los otros métodos porque depende de dónde caiga el nodo respecto del máximo.",
             [tab3]),
            ("Fuentes de error y criterio de confianza",
             "El error de truncamiento (Simpson, punto medio) se controla con $n$ y $M$ y es despreciable con los valores usados; el error de malla angular se controla con $N_\\theta$ y es la fuente dominante, y el redondeo vale unos $10^{-16}$. El resultado es confiable cuando al duplicar $n$, $M$ o $N_\\theta$ el cambio es menor que la tolerancia deseada, y cuando el paso angular es varias veces menor que el ancho de un pico principal (aproximadamente $\\lambda/(Nd)$).",
             [rf"\Delta\theta_{{pico}}\approx\dfrac{{\lambda}}{{N d}}\approx{np.degrees(lam / (N * d)):.3f}^\circ\ \gg\ \Delta\theta={np.degrees(np.deg2rad(89) / 20000):.5f}^\circ"]),
        ],
    )


def _pol() -> dict:
    ths = np.deg2rad([10, 20, 30, 45, 60, 80])
    rows = []
    for t in ths:
        ij, ia = m18.jones_intensity([0, t, np.pi / 2]), m18.three_polarizers_analytic(t)
        rows.append([f"{np.degrees(t):.0f}^\\circ", fx(ij, 10), fx(ia, 10), sci(abs(ij - ia), 2)])
    tab1 = _tab("rrrr", [[r"\theta", r"I_{Jones}", r"I_{anal}", r"|\Delta I|"]] + rows)
    ds = (0, 1, 2, 5, 10)
    imax = m18.jones_intensity([0, np.pi / 4, np.pi / 2])
    rows2 = []
    for dd in ds:
        v = m18.jones_intensity([0, np.pi / 4 + np.deg2rad(dd), np.pi / 2])
        rows2.append([f"{dd}^\\circ", fx(v, 6), fx(v / imax, 6), fx(np.cos(2 * np.deg2rad(dd)) ** 2, 6)])
    tab2 = _tab("rrrr", [[r"\delta", r"I(45^\circ+\delta)", r"I/I_{max}", r"\cos^22\delta"]] + rows2)
    nn = (1, 2, 10, 100, 1000)
    rows3 = []
    for n in nn:
        angs = [k * np.pi / (2 * n) for k in range(n + 1)]
        iv = m18.jones_intensity(angs)
        ia = float(np.cos(np.pi / (2 * n)) ** (2 * n))
        rows3.append([str(n), fx(iv, 10), fx(ia, 10), sci(abs(iv - ia), 2)])
    tab3 = _tab("rrrr", [[r"N_{int}", r"I_{Jones}", r"\cos^{2N}(\pi/2N)", r"|\Delta I|"]] + rows3)
    p = m18.polarizer_jones(np.deg2rad(30))
    idem = float(np.max(np.abs(p @ p - p)))
    det = float(np.linalg.det(p))
    return dict(
        numeric_intro=[
            r"El cálculo de Jones representa el estado de polarización de la luz por un vector complejo $\vec E=(E_x,E_y)^T$ y cada elemento óptico por una matriz $2\times2$; la luz que atraviesa varios elementos se obtiene multiplicando el vector por las matrices en orden. Se usa porque la ley de Malus $I=I_0\cos^2\theta$ vale para un par de polarizadores, mientras que una cadena de tres o más obliga a encadenar la ley con cuidado de que la luz que sale ya esté polarizada en la dirección del eje anterior, que es lo que la matriz hace automáticamente.",
            r"Un polarizador lineal con eje a ángulo $\theta$ es el proyector $P(\theta)=\begin{pmatrix}\cos^2\theta&\cos\theta\sin\theta\\ \cos\theta\sin\theta&\sin^2\theta\end{pmatrix}=\hat u\hat u^T$ con $\hat u=(\cos\theta,\sin\theta)$. Para la cadena de ejes $\theta_1,\theta_2,\theta_3$ se calcula $\vec E_{out}=P(\theta_3)P(\theta_2)P(\theta_1)\vec E_{in}$ y la intensidad es $I=\vec E^\dagger\vec E$. Como la matriz proyecta, deja pasar solo la componente sobre el eje.",
            r"No hay truncamiento: el resultado es exacto en aritmética exacta y solo se acumula redondeo, que crece con el número $N$ de matrices multiplicadas (del orden de $10^{-16}$ por producto). Una limitación notable es que $P$ es singular ($\det P=0$, rango 1): no se puede invertir, lo que refleja que el polarizador destruye información sobre la componente absorbida. Además, el modelo supone polarizadores ideales (extinción perfecta, sin pérdidas ni reflexión) y luz monocromática.",
            r"Los parámetros son los de la escena: campo incidente polarizado a 0° ($\vec E_{in}=(1,0)^T$), primer polarizador a 0°, segundo a $\theta$ (30° y 45° en los ejemplos) y analizador a 90°. El resultado, $I/I_0$, es la fracción transmitida, con máximo $1/4$ a 45°. No hay semilla ni paso: el estudio de error consiste en comparar con la expresión cerrada y en medir cómo cambia $I$ si el polarizador intermedio se desalinea.",
            r"El método es independiente del analítico porque la fórmula $I_3/I_0=\cos^2\theta\,\sin^2\theta$ usa la ley de Malus dos veces seguidas, mientras que Jones calcula vectores de campo y proyecta, sin invocar $\cos^2$ para la intensidad (solo para construir las matrices). La coincidencia hasta $10^{-17}$ confirma que la ley de Malus es consecuencia del producto escalar de amplitudes.",
        ],
        numeric_code="""def polarizador(theta):                       # proyector sobre el eje a angulo theta
    c, s = cos(theta), sin(theta)
    return array([[c*c, c*s],
                  [c*s, s*s]])                # P = u u^T con u = (cos, sin)
def intensidad(angulos, E=(1.0, 0.0)):
    E = array(E, dtype=complex)               # campo incidente polarizado a 0 grados
    for a in angulos:                         # la luz atraviesa los elementos en orden
        E = polarizador(a) @ E                # cada polarizador proyecta el campo sobre su eje
    return vdot(E, E).real                    # I = E^dagger E
I30 = intensidad([0, radians(30), pi/2])      # cadena 0 -> 30 -> 90 grados
I45 = intensidad([0, radians(45), pi/2])      # el medio a 45: maximo
I_malus = intensidad([0, radians(60)])        # dos polarizadores = ley de Malus
P = polarizador(radians(30))
res = abs(P @ P - P).max()                    # proyector: P^2 = P (comprobacion)
""",
        numeric_converge=[
            ("Verificación cruzada contra la expresión cerrada",
             "Como no hay parámetro de discretización, el estudio es una verificación en varios ángulos: la cadena $0^\\circ\\to\\theta\\to90^\\circ$ por Jones frente a $\\cos^2\\theta\\sin^2\\theta$. La diferencia está en el nivel del último bit de doble precisión.",
             [tab1]),
            ("Sensibilidad a la desalineación del polarizador intermedio",
             f"El máximo de transmisión ($1/4$ en $45^\\circ$) es un extremo, así que una desalineación $\\delta$ lo reduce solo de forma cuadrática: $I/I_{{max}}=\\cos^22\\delta$. Con $5^\\circ$ la caída es de {(1 - m18.jones_intensity([0, np.pi / 4 + np.deg2rad(5), np.pi / 2]) / imax) * 100:.2f} %. A $30^\\circ$, lejos del máximo, la pendiente $dI/d\\theta$ es mucho mayor (unos {fx(0.5 * np.sin(4 * np.pi / 6), 3)} por radián, frente a 0 en el máximo); por eso, para alinear dos polarizadores se busca el mínimo de transmisión (ejes cruzados) y no el máximo.",
             [tab2]),
            ("Acumulación de redondeo en cadenas largas",
             f"Para estudiar cómo crece el error con el número de matrices se coloca una cadena de $N_{{int}}$ polarizadores igualmente espaciados entre $0^\\circ$ y $90^\\circ$, cuya transmisión es $\\cos^{{2N}}(\\pi/2N)$ y tiende a 1 cuando $N\\to\\infty$ (efecto Zenón). Con 1000 matrices el error absoluto es {ps(abs(m18.jones_intensity([k * np.pi / 2000 for k in range(1001)]) - float(np.cos(np.pi / 2000) ** 2000)), 2)}: el redondeo crece, pero es despreciable frente a una escala de $I\\sim1$.",
             [tab3]),
            ("Propiedades de control y criterio de confianza",
             f"Se comprueba que el proyector sea idempotente, $P^2=P$, con residuo {ps(idem, 2)}, y que sea singular, $\\det P={sci(det, 2)}$. El resultado es confiable cuando estos residuos son del orden de $\\epsilon_m$, cuando $0\\le I\\le I_0$ (energía no creada) y cuando coincide con la expresión cerrada dentro del redondeo; no se espera ninguna otra fuente de error numérico.",
             [rf"\|P^2-P\|_\infty={sci(idem, 2)},\qquad \det P(30^\circ)={sci(det, 2)}"]),
        ],
    )


DETAIL = {
    "YeeFDTD": _yee(),
    "PoyntingDisplacement": _poynting(),
    "SnellFermat": _snell(),
    "PrismDispersion": _prism(),
    "ThinLensImage": _lens(),
    "TwoLensSystem": _twolens(),
    "YoungSlits": _young(),
    "ThinFilmCoating": _film(),
    "SlitAndGrating": _grating(),
    "PolarizersMalus": _pol(),
}
