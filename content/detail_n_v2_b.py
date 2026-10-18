"""Explicacion del METODO NUMERICO (numeric_intro, numeric_code, numeric_converge) para:
CoaxialLaplace, CapacitorNetwork, DrudeElectrons, TaperedWire, MeshCircuit, RCCharging, HelicalMotion, HallEffect.
Todas las cifras de los estudios de error se calculan al importar con las funciones de `physics/`."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch05_capacitance as e5
from physics import v2ch06_current as e6
from physics import v2ch07_circuits as e7
from physics import v2ch08_magnetic_field as m8
from physics.constants import E_CH, M_E
from physics.integrators import euler_step, integrate, rk4_step


def _table(head: list[str], rows: list[list[str]]) -> str:
    cols = "c" * len(head)
    body = r"\\ ".join(" & ".join(r) for r in rows)
    return r"\begin{array}{" + cols + r"}" + " & ".join(head) + r"\\ \hline " + body + r"\end{array}"


def _slope(hs, es) -> float:
    """Pendiente de log(error) contra log(h) (orden observado)."""
    return float(np.polyfit(np.log(np.asarray(hs, float)), np.log(np.asarray(es, float)), 1)[0])


# ------------------------------------------------------------------------------------------ CoaxialLaplace
def _coax() -> dict:
    a, b = e5.A_IN, e5.B_OUT
    ca = e5.coax_capacitance_per_length_analytic()
    runs = [(21, 300), (41, 300), (81, 500), (161, 800)]
    hs, errs, rows = [], [], []
    for n, its in runs:
        h = 2 * 1.02 * b / (n - 1)
        c = e5.coax_capacitance_numeric(n=n, iterations=its)
        er = abs(c - ca) / ca
        hs.append(h)
        errs.append(er)
        rows.append([str(n), fx(h * 1e3, 3), fx(c * 1e12, 3), fx(er * 100, 2)])
    p_h = _slope(hs, errs)
    tab_h = _table([r"n", r"h\ (\mathrm{mm})", r"C'_{num}\ (\mathrm{pF/m})", r"\text{error}\ (\%)"], rows)
    n0 = 161
    ref = e5.coax_capacitance_numeric(n=n0, iterations=800)
    rows2 = []
    for k in (25, 50, 100, 200, 400):
        c = e5.coax_capacitance_numeric(n=n0, iterations=k)
        rows2.append([str(k), fx(c * 1e12, 3), sci(abs(c - ref) / ref, 2)])
    tab_it = _table([r"\text{iteraciones}", r"C'_{num}\ (\mathrm{pF/m})", r"|C'_k-C'_{800}|/C'_{800}"], rows2)
    omega = 2.0 / (1.0 + np.sin(np.pi / n0))
    return dict(
        numeric_intro=[
            "La ecuación de Laplace $\\nabla^2 V=0$ entre dos cilindros concéntricos tiene solución cerrada sólo porque la simetría la reduce a una ecuación en $r$. En cuanto la geometría pierde simetría (conductores excéntricos, ranuras, bordes) ya no hay fórmula, y se resuelve en una malla. Aquí se resuelve sobre una malla cuadrada aunque el problema sí tenga solución exacta, para poder medir cuánto se equivoca el método.",
            "Con diferencias centrales, $\\partial^2V/\\partial x^2\\approx[V(x+h)-2V(x)+V(x-h)]/h^2$, la ecuación se vuelve la regla $V_{i,j}=\\tfrac14\\left(V_{i+1,j}+V_{i-1,j}+V_{i,j+1}+V_{i,j-1}\\right)$: cada nodo libre es el promedio de sus cuatro vecinos. Es un sistema lineal enorme que se resuelve iterando (relajación). El método SOR acelera la iteración con $V\\leftarrow V+\\omega\\,(\\bar V-V)$, y el orden rojo-negro (tablero de ajedrez) permite actualizar todos los nodos de un color a la vez porque sus vecinos son del otro color.",
            "Hay dos errores distintos. El de discretización viene de sustituir derivadas por diferencias y de aproximar un círculo con nodos cuadrados: es $O(h^2)$ en el interior pero, por la frontera escalonada, el efecto sobre la capacitancia resulta de orden cercano a 1 en $h$. El de iteración es cuánto falta para que el sistema lineal esté resuelto del todo, y baja geométricamente con cada barrido; con $\\omega$ óptimo el radio espectral es aproximadamente $\\omega-1$, así que el número de barridos crece como $n$ y no como $n^2$ (Jacobi).",
            f"Los parámetros: $n=161$ nodos por lado (cada nodo mide $h={sci(2*1.02*b/(n0-1)*1e3,3)}$ mm y el radio interno $a=5$ mm abarca unos 20 nodos), $\\omega=2/(1+\\sin(\\pi/n))={fx(omega,4)}$, y 4000 barridos, muchos más de los necesarios (la tabla de abajo muestra que bastan unos cientos). El resultado se interpreta como la capacitancia por unidad de longitud $C'=2U'/V_0^2$ calculada con la energía del campo $U'=\\tfrac12\\varepsilon_0\\int|E|^2dA$.",
            "El método es independiente del analítico: nunca usa $\\ln(b/a)$ ni la ley de Gauss. Sólo usa las condiciones de frontera, la regla del promedio y el gradiente de $V$; la fórmula $C'=2\\pi\\varepsilon_0/\\ln(b/a)$ aparece únicamente al final como referencia para medir el error.",
        ],
        numeric_code="""\
xs = linspace(-1.02*b, 1.02*b, n); h = xs[1] - xs[0]    # malla cuadrada n x n
R = hypot(X, Y)                                          # distancia al eje de cada nodo
fijo = (R <= a) | (R >= b)                               # nodos dentro de los conductores
V = where(R <= a, V0, 0.0)                               # semilla: interior a V0, resto 0
omega = 2 / (1 + sin(pi / n))                            # factor SOR (casi optimo)
rojo  = ((i + j) % 2 == 0) & ~fijo                       # tablero de ajedrez
negro = ((i + j) % 2 == 1) & ~fijo
for it in range(iteraciones):                            # barridos de relajacion
    for color in (rojo, negro):                          # un color usa vecinos del otro
        prom = 0.25 * (V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])   # promedio de 4 vecinos
        V = where(color, V + omega * (prom - V), V)      # sobrerrelajacion en nodos libres
gx, gy = gradient(V, h, h)                               # E = -grad V (dif. centrales)
U = 0.5 * eps0 * sum((gx**2 + gy**2)[~fijo]) * h * h     # energia por unidad de longitud
C = 2 * U / V0**2                                        # de U = 1/2 C V0^2
""",
        numeric_converge=[
            ("Error contra tamaño de malla", f"Se repite el cálculo duplicando la resolución ($n=21,41,81,161$, con barridos suficientes en cada caso) y se compara con $C'_{{an}}={sci(ca,5)}$ F/m. El error no cae como $h^2$ sino aproximadamente como $h^{{{fx(p_h,2)}}}$ (pendiente de $\\log\\text{{error}}$ contra $\\log h$), porque la frontera circular se aproxima con escalones.",
             [tab_h, rf"\text{{pendiente observada}}\ p={fx(p_h,2)}"]),
            ("Error contra número de barridos", f"Con $n={n0}$ fijo se mira cuánto cambia $C'$ al seguir iterando. El valor se estabiliza en unos cientos de barridos y no depende de los 4000 que usa la escena: ese error de iteración se vuelve despreciable frente al de discretización.",
             [tab_it]),
            ("Fuentes de error y criterio de confianza", f"Dominan la discretización de la frontera (del orden de {fx(errs[-1]*100,1)} % a $n=161$) y el cálculo de $\\vec E$ por diferencias centrales; el redondeo ($\\sim10^{{-16}}$ por operación) y el error de iteración son mucho menores. Criterio práctico: el resultado es confiable cuando duplicar $n$ reduce el error en la proporción observada y cuando más barridos ya no cambian $C'$ en las cifras citadas.",
             [rf"\text{{error a }}n=161:\ {fx(errs[-1]*100,2)}\ \%\ \Rightarrow\ \text{{extrapolando con }}p={fx(p_h,2)},\ n=321\ \text{{daría aprox. }}{fx(errs[-1]*100/2**p_h,2)}\ \%"]),
        ],
    )


# ------------------------------------------------------------------------------------------ CapacitorNetwork
def _network() -> dict:
    c1, c2, c3, v = e5.C1, e5.C2, e5.C3, e5.V_SRC
    ceq_a = e5.network_ceq_analytic()
    r = e5.network_nodal_numeric()
    res_node = c1 * (v - r["vx"]) - (c2 + c3) * r["vx"]
    q_bal = r["q"] - r["q2"] - r["q3"]
    rel = abs(r["c_eq"] - ceq_a) / ceq_a
    rows = []
    el_sum = 0.0
    for name, idx in (("C_1", 0), ("C_2", 1), ("C_3", 2)):
        cs = [c1, c2, c3]
        cs[idx] *= 1.01
        r2 = e5.network_nodal_numeric(c1=cs[0], c2=cs[1], c3=cs[2])
        el = (r2["c_eq"] - r["c_eq"]) / r["c_eq"] / 0.01
        el_sum += el
        rows.append([name, fx(r2["c_eq"] * 1e6, 5), fx(el, 3)])
    tab = _table([r"\text{se cambia}\ +1\%", r"C_{eq}\ (\mu\mathrm{F})", r"\text{elasticidad}\ \dfrac{\Delta C_{eq}/C_{eq}}{\Delta C/C}"], rows)
    return dict(
        numeric_intro=[
            "Para una red pequeña la fórmula serie-paralelo es inmediata, pero el método de nodos no depende de reconocer qué está en serie y qué en paralelo, y por eso escala a redes de cientos de capacitores. La idea es tratar los potenciales de los nodos como incógnitas y exigir que la carga neta de cada nodo aislado sea cero, lo que produce un sistema lineal $\\mathbf A\\vec V=\\vec b$.",
            "Aquí hay un solo nodo interno, de potencial $V_x$. La carga que entra por $C_1$ es $C_1(V-V_x)$ y la que sale por $C_2$ y $C_3$ es $(C_2+C_3)V_x$; igualarlas da $(C_1+C_2+C_3)V_x=C_1V$. Con $V_x$ se obtienen las cargas $q_k=C_k\\Delta V_k$ y, de la carga total, $C_{eq}=q_1/V$. En una red general la matriz $A_{ii}=\\sum C$ (capacitores del nodo $i$) y $A_{ij}=-C_{ij}$ es simétrica y definida positiva.",
            "Un solve directo (factorización LU con pivoteo) no es iterativo: no tiene error de truncamiento, sólo de redondeo, y éste se controla con el número de condición: el error relativo de la solución está acotado por $\\kappa(A)\\,\\varepsilon_{m}$ con $\\varepsilon_m\\approx10^{-16}$. Con una matriz $1\\times1$ se tiene $\\kappa=1$ exactamente, así que el resultado es exacto hasta redondeo; en redes mal planteadas (capacitores que difieren en muchos órdenes de magnitud) $\\kappa$ crece y habría que revisarlo.",
            f"No hay paso ni tolerancia que elegir: los valores son los datos del problema ($V={fx(v,0)}$ V, $C_1=2$, $C_2=3$, $C_3=6\\ \\mu$F). El resultado es el potencial del nodo, $V_x={fx(r['vx'],4)}$ V, del que sale todo lo demás. Para estudiar la fiabilidad no se varía $h$ sino que se comprueba el residuo de la ecuación nodal, la conservación de carga y la sensibilidad de $C_{{eq}}$ a los datos.",
            "Es independiente del método analítico porque éste combina $1/C$ de las ramas en serie y suma las ramas en paralelo, mientras que aquí sólo se impone conservación de la carga y se calcula $C_{eq}$ a partir de la carga total entregada por la fuente. Que ambos coincidan es una verificación cruzada real de dos razonamientos diferentes.",
        ],
        numeric_code="""\
# datos: fuente V, capacitores C1 (fuente->nodo), C2 y C3 (nodo->tierra)
A = array([[C1 + C2 + C3]])             # matriz nodal (1 incognita: potencial del nodo x)
b = array([C1 * V])                     # termino independiente: carga aportada por la fuente
Vx = linalg.solve(A, b)[0]              # LU con pivoteo (no hay iteracion ni paso h)
q1 = C1 * (V - Vx)                      # carga en C1 = carga total entregada
q2 = C2 * Vx                            # carga en C2
q3 = C3 * Vx                            # carga en C3
residuo = C1*(V - Vx) - (C2 + C3)*Vx    # ecuacion nodal: debe ser ~ 0
balance = q1 - q2 - q3                  # conservacion de carga en el nodo: ~ 0
Ceq = q1 / V                            # capacitancia equivalente
U = 0.5 * q1 * V                        # energia almacenada en toda la red
""",
        numeric_converge=[
            ("Residuo y conservación de carga", "Como no hay iteraciones, el estudio de error es una verificación a posteriori: se sustituye la solución en la ecuación nodal y se suman las cargas del nodo. Ambos residuos deben ser del orden del redondeo.",
             [rf"C_1(V-V_x)-(C_2+C_3)V_x={sci(res_node,2)}\ \mathrm{{C}},\qquad q_1-q_2-q_3={sci(q_bal,2)}\ \mathrm{{C}}",
              rf"\dfrac{{|C_{{eq}}^{{num}}-C_{{eq}}^{{an}}|}}{{C_{{eq}}^{{an}}}}={sci(rel,2)},\qquad \kappa(A)={fx(float(np.linalg.cond(np.array([[c1+c2+c3]]))),1)}"]),
            ("Sensibilidad a los datos", "El otro error posible es el de los datos (tolerancias reales de los capacitores, a menudo del 5 al 20 %). Se aumenta cada capacitancia un 1 % y se mide cuánto cambia $C_{eq}$. La elasticidad dice qué capacitor domina la incertidumbre; como $C_{eq}$ es homogénea de grado 1, las tres elasticidades deben sumar 1 (aquí, con un incremento finito del 1 %, suman casi 1).",
             [tab, rf"\sum\text{{elasticidades}}={fx(el_sum,3)}"]),
            ("Criterio de confiabilidad", "El resultado es confiable si el residuo está al nivel del redondeo, la carga se conserva y $\\kappa$ es pequeño. Aquí el error numérico es despreciable ($\\sim10^{-16}$) frente a la incertidumbre de los datos: una tolerancia del 1 % en $C_1$ ya mueve $C_{eq}$ unos catorce órdenes de magnitud por encima de ese error, de modo que no tiene sentido buscar más precisión del cálculo.",
             [rf"\text{{error numérico}}\sim{sci(max(rel,1e-16),1)}\ \ll\ \text{{error por tolerancia}}\sim{fx(abs(float(rows[0][2]))*1,2)}\ \%"]),
        ],
    )


# ------------------------------------------------------------------------------------------ DrudeElectrons
def _drude() -> dict:
    mu_a = e6.drift_mobility_analytic()
    tau, es = e6.TAU, e6.E_SIM
    rows, ns, rms = [], [], []
    for ne in (500, 2000, 8000, 32000):
        errs, sigs = [], []
        for seed in range(1, 17 if ne < 20000 else 7):
            mu, sg, _ = e6.drift_mobility_monte_carlo(n_e=ne, steps=2000, seed=seed)
            errs.append(mu - mu_a)
            sigs.append(sg)
        rm = float(np.sqrt(np.mean(np.square(errs))))
        ns.append(ne)
        rms.append(rm)
        rows.append([str(ne), sci(float(np.mean(sigs)), 3), sci(rm, 3), fx(rm / mu_a * 100, 2)])
    p = _slope(ns, rms)
    tab = _table([r"N_e", r"\sigma_{est}\ (\mathrm{m^2/Vs})", r"\text{RMS real entre semillas}", r"\text{error relativo}\ (\%)"], rows)
    mu9, sg9, ser = e6.drift_mobility_monte_carlo()
    dt = tau / 50
    return dict(
        numeric_intro=[
            "El modelo de Drude describe un electrón que se acelera con el campo y de vez en cuando choca y pierde la memoria de su velocidad. El resultado analítico $v_d=eE\\tau/m$ es un promedio sobre muchos electrones y muchas colisiones aleatorias. Una simulación Monte Carlo reproduce directamente ese proceso aleatorio y mide el promedio, sin suponer de entrada que el efecto de las colisiones sea simplemente restar un término $-m\\vec v/\\tau$.",
            "Cada electrón se avanza en pasos $\\Delta t=\\tau/50$: primero $v\\leftarrow v+a\\,\\Delta t$ con $a=-eE/m$, y después, con probabilidad $p=\\Delta t/\\tau$, choca y su velocidad se sortea de una distribución normal de dispersión $v_{th}=\\sqrt{k_BT/m}$. La movilidad se estima como $\\hat\\mu=-\\langle v\\rangle/E$, promediando sobre los $N_e$ electrones y sobre la segunda mitad del tiempo (cuando ya se llegó al régimen estacionario).",
            "El estimador de Monte Carlo es insesgado (en el límite $\\Delta t\\to0$) y su error estadístico baja como $1/\\sqrt{N}$, con $N$ el número de muestras independientes: para ganar un dígito se necesitan 100 veces más electrones. El otro error es de discretización en el tiempo, de orden $\\Delta t/\\tau=0.02$ en la probabilidad de choque; en esta escena queda por debajo del ruido y no se distingue. Limitación: la señal $v_d$ es minúscula frente a $v_{th}$, así que el ruido de cada electrón (que es de unos $10^5$ m/s) tapa la deriva real si el campo es pequeño.",
            f"Por eso se usa $E_{{sim}}={sci(es,2)}$ V/m en lugar de los 100 V/m del problema: el modelo es lineal en $E$ y $\\mu=v_d/E$ no cambia, pero la señal sube lo suficiente para que 20 000 electrones la midan con ruido relativo del orden de {fx(sg9/mu9*100,1)} %. La semilla fija (9) hace reproducible la figura; el resultado se interpreta como la movilidad $\\mu$ que, multiplicada por $E=100$ V/m, da $v_d$.",
            "Es independiente del analítico: la fórmula $\\mu=e\\tau/m$ no entra en la simulación; sólo entran $a=-eE/m$, la probabilidad de choque y la distribución térmica inicial. Que la media sorteada reproduzca $e\\tau/m$ es lo que valida la interpretación de $\\tau$ como tiempo medio entre colisiones.",
        ],
        numeric_code="""\
rng = default_rng(semilla)                        # generador reproducible (semilla = 9)
dt = tau / 50                                     # 50 pasos por tiempo de colision
vth = sqrt(kB * T / m)                            # dispersion termica de la velocidad
v = rng.normal(0, vth, Ne)                        # velocidades iniciales de Ne electrones
a = -e * E / m                                    # aceleracion constante (electron: signo -)
for k in range(pasos):
    v += a * dt                                   # el campo acelera a todos
    choque = rng.uniform(size=Ne) < dt / tau      # prob. de choque en este paso
    v[choque] = rng.normal(0, vth, choque.sum())  # el choque borra la memoria de v
    media[k] = v.mean()                           # velocidad media del conjunto
cola = -media[pasos//2:] / E                      # segunda mitad: ya estacionario
mu = cola.mean()                                  # movilidad estimada
sigma = vth / sqrt(Ne * (pasos//2) / 50) / E      # error estadistico estimado
""",
        numeric_converge=[
            ("Error estadístico contra número de electrones", f"Se cuadruplica $N_e$ y cada caso se repite con 16 semillas distintas (6 en el último, por costo) para obtener un error cuadrático medio real, que se compara con el estimado $\\sigma_{{est}}$ de la función. Si el orden es $1/\\sqrt N$, cada cuadruplicación debe dividir el error entre 2. Se observa además que $\\sigma_{{est}}$ subestima el error real por un factor cercano a 1.5 o 2 (las velocidades de pasos vecinos están correlacionadas), de modo que debe leerse como orden de magnitud.",
             [tab, rf"\text{{pendiente observada}}\ p={fx(p,2)}\ \ (\text{{teórica}}\ -0.5)"]),
            ("Fuentes de error en la escena", f"Con $N_e=20\\,000$ y 4000 pasos la escena obtiene $\\hat\\mu={sci(mu9,4)}$ frente a $\\mu_{{an}}={sci(mu_a,4)}$ m$^2$/(V s), con error real {fx(abs(mu9-mu_a)/mu_a*100,2)} % y estimado {fx(sg9/mu_a*100,2)} %. Además, la media sólo se promedia en el régimen estacionario (segunda mitad), así que el transitorio inicial no sesga el resultado. El redondeo ($\\sim10^{{-16}}$) es irrelevante frente al ruido estadístico.",
             [rf"\hat\mu={sci(mu9,4)}\pm{sci(sg9,2)},\qquad \mu_{{an}}={sci(mu_a,4)}\ \mathrm{{m^2/(V\,s)}}"]),
            ("Criterio práctico", "El resultado es confiable cuando (i) el error real entre semillas es comparable con $\\sigma_{est}$, (ii) la diferencia con el valor teórico es menor que 2 o 3 veces el error real entre semillas y (iii) al aumentar $N_e$ el error baja con pendiente $-1/2$. Para precisar un dígito más hay que pagar 100 veces más electrones, por lo que Monte Carlo no compite con la fórmula cuando ésta existe: sólo vale la pena cuando las colisiones no son tratables analíticamente.",
             []),
        ],
    )


# ------------------------------------------------------------------------------------------ TaperedWire
def _tapered() -> dict:
    ra = e6.tapered_resistance_analytic()
    rho, L, r1, r2 = e6.RHO, e6.LENGTH, e6.R1, e6.R2
    ns = [2, 4, 8, 16, 32]
    es_s, es_t, rows = [], [], []
    for n in ns:
        rs = e6.tapered_resistance_numeric(n=n)
        x = np.linspace(0, L, n + 1)
        y = rho / (np.pi * (r1 + (r2 - r1) * x / L) ** 2)
        rt = (L / n) * (y.sum() - 0.5 * (y[0] + y[-1]))
        es_s.append(abs(rs - ra) / ra)
        es_t.append(abs(rt - ra) / ra)
        rows.append([str(n), sci(es_t[-1], 3), sci(es_s[-1], 3)])
    hs = [L / n for n in ns]
    ps, pt = _slope(hs[-3:], es_s[-3:]), _slope(hs[-3:], es_t[-3:])
    tab = _table([r"n", r"\text{error trapecio}", r"\text{error Simpson}"], rows)
    r400 = e6.tapered_resistance_numeric(n=400)
    r4000 = e6.tapered_resistance_numeric(n=4000)
    return dict(
        numeric_intro=[
            "La resistencia de un cable de sección variable es $R=\\int_0^L\\rho\\,dx/A(x)$: se suman en serie rebanadas delgadas, cada una de resistencia $\\rho\\,dx/(\\pi r(x)^2)$. Aquí la integral sí tiene primitiva, pero en un cable real $r(x)$ viene de una medición o de un perfil complicado y no hay primitiva; la integración numérica sólo necesita evaluar el integrando en puntos.",
            "La regla de Simpson aproxima el integrando en cada par de subintervalos por una parábola. Con $n$ par y $h=L/n$: $\\int_a^b f\\,dx\\approx\\tfrac h3\\left[f_0+4f_1+2f_2+4f_3+\\dots+4f_{n-1}+f_n\\right]$. El patrón de pesos $1,4,2,4,\\dots,4,1$ viene de integrar exactamente la parábola que pasa por tres puntos consecutivos.",
            "Su error es $-\\dfrac{(b-a)}{180}h^4f^{(4)}(\\xi)$, es decir $O(h^4)$ (el trapecio sólo $O(h^2)$), y la regla es exacta para polinomios hasta grado 3. Falla cuando el integrando no es suave (saltos, esquinas, singularidades) o cuando $f^{(4)}$ es enorme, y exige $n$ par. Aquí $f=\\rho/[\\pi(r_1+kx)^2]$ es analítico en $[0,L]$ porque $r$ nunca se anula, así que las hipótesis se cumplen bien.",
            f"Se eligió $n=400$ subintervalos ($h={sci(L/400*1e3,3)}$ mm): como el error escala con $h^4$ ya a $n=16$ se tienen varias cifras, y $n=400$ lleva el error a unas diez cifras correctas con costo ínfimo. El resultado, $R={sci(r400,6)}\\ \\Omega$, es la resistencia total del cono; comparado con un cable uniforme de radio $r_1$ resulta mucho menor porque la parte ancha conduce mejor.",
            "Es independiente del método analítico: éste sustituye $r(x)$ lineal y usa la primitiva $-1/(k\\,r)$ para obtener $\\rho L/(\\pi r_1r_2)$; Simpson nunca calcula esa primitiva, sólo muestrea $\\rho/(\\pi r^2)$ en 401 puntos y los combina con pesos fijos.",
        ],
        numeric_code="""\
n = 400                                          # subintervalos (par, requisito de Simpson)
x = linspace(0, L, n + 1)                        # nodos a lo largo del cable
r = r1 + (r2 - r1) * x / L                       # radio lineal en cada nodo
f = rho / (pi * r**2)                            # integrando: resistencia por unidad de longitud
h = L / n                                        # paso uniforme
impares = f[1:-1:2].sum()                        # nodos 1, 3, 5, ... (peso 4)
pares = f[2:-1:2].sum()                          # nodos 2, 4, 6, ... (peso 2)
R = h / 3 * (f[0] + f[-1] + 4*impares + 2*pares) # regla de Simpson compuesta
# comparacion con el trapecio, de orden 2:
R_trap = h * (f.sum() - 0.5*(f[0] + f[-1]))
""",
        numeric_converge=[
            ("Error contra número de subintervalos", f"Se duplica $n$ y se compara con $R_{{an}}={sci(ra,6)}\\ \\Omega$. Simpson reduce el error por un factor cercano a 16 cada vez (orden 4) y el trapecio sólo por 4 (orden 2). Con los pasos iniciales todavía fuera del régimen asintótico (la razón de errores sube 7.7, 11.2, 14.0, 15.4 hacia 16), las pendientes de $\\log\\text{{error}}$ contra $\\log h$ en los tres refinamientos finales son: Simpson {fx(ps,2)}, trapecio {fx(pt,2)}.",
             [tab, rf"p_{{Simpson}}={fx(ps,2)},\qquad p_{{trapecio}}={fx(pt,2)}"]),
            ("Fuentes de error y el piso del redondeo", f"Al pasar de $n=400$ a $n=4000$ el error cae por un factor $\\approx10^4$, justo lo que predice $h^4$: sigue siendo de truncamiento. El piso del redondeo ($\\sim10^{{-16}}$ por operación, acumulado en miles de sumas) se alcanza recién alrededor de $10^{{-15}}$; pasar de ahí no mejora el resultado. A $n=400$ la escena tiene unas diez cifras significativas correctas, más que suficientes.",
             [rf"\dfrac{{|R_{{400}}-R_{{an}}|}}{{R_{{an}}}}={sci(abs(r400-ra)/ra,2)},\qquad \dfrac{{|R_{{4000}}-R_{{an}}|}}{{R_{{an}}}}={sci(abs(r4000-ra)/ra,2)}"]),
            ("Criterio práctico", "Sin conocer la respuesta, se estima el error comparando $S_n$ con $S_{2n}$: $|S_{2n}-S_n|/15$ aproxima el error de $S_{2n}$ cuando el orden es 4 (extrapolación de Richardson). Se acepta el resultado cuando esa diferencia es menor que la tolerancia pedida y el integrando es suave en todo el intervalo.",
             [rf"\dfrac{{|S_{{32}}-S_{{16}}|}}{{15}}={sci(abs(e6.tapered_resistance_numeric(n=32)-e6.tapered_resistance_numeric(n=16))/15,2)}\ \Omega\ \ (\text{{error real de }}S_{{32}}={sci(abs(e6.tapered_resistance_numeric(n=32)-ra),2)}\ \Omega)"]),
        ],
    )


# ------------------------------------------------------------------------------------------ MeshCircuit
def _mesh() -> dict:
    e1, e2, r1, r2, r3 = e7.E1, e7.E2, e7.R1, e7.R2, e7.R3
    a = np.array([[r1 + r3, -r3], [-r3, r2 + r3]])
    i = e7.mesh_currents_numeric()
    ia = e7.mesh_currents_analytic()
    kap = float(np.linalg.cond(a))
    res = a @ i - np.array([e1, -e2])
    kr = e7.kirchhoff_residuals(i)
    rows = []
    for name, kw in (("R_1", dict(r1=r1 * 1.01)), ("R_2", dict(r2=r2 * 1.01)), ("R_3", dict(r3=r3 * 1.01)), ("E_1", dict(e1=e1 * 1.01))):
        j = e7.mesh_currents_numeric(**kw)
        d = (j - i) / i * 100
        rows.append([name, fx(d[0], 3), fx(d[1], 3)])
    tab = _table([r"\text{se cambia}\ +1\%", r"\Delta i_1/i_1\ (\%)", r"\Delta i_2/i_2\ (\%)"], rows)
    return dict(
        numeric_intro=[
            "El método de mallas convierte un circuito en un sistema lineal: se asigna una corriente cíclica a cada malla independiente y se aplica la ley de voltajes de Kirchhoff a cada una. Para dos mallas se resuelve a mano con la regla de Cramer, pero con diez mallas esa fórmula es impracticable (su costo crece como $n!$) y se usa eliminación gaussiana, que cuesta $O(n^3)$.",
            f"El sistema es $\\mathbf R\\,\\vec i=\\vec\\varepsilon$ con $R_{{kk}}$ la suma de resistencias de la malla $k$ y $R_{{kl}}=-$(resistencias compartidas); aquí $\\mathbf R=\\begin{{pmatrix}}{r1+r3:.0f}&-{r3:.0f}\\\\-{r3:.0f}&{r2+r3:.0f}\\end{{pmatrix}}$. El código llama a `linalg.solve`, que factoriza $\\mathbf R=\\mathbf P\\mathbf L\\mathbf U$ con pivoteo parcial y resuelve dos sistemas triangulares por sustitución hacia adelante y hacia atrás.",
            "Un método directo no tiene error de truncamiento: la solución es exacta salvo por el redondeo. Ese error está acotado por $\\|\\delta\\vec i\\|/\\|\\vec i\\|\\lesssim\\kappa(\\mathbf R)\\,\\varepsilon_m$, con $\\kappa$ el número de condición y $\\varepsilon_m\\approx2\\times10^{-16}$. Las matrices de mallas con resistencias positivas son simétricas definidas positivas y suelen estar bien condicionadas; el método falla (matriz singular) si el circuito está mal planteado, por ejemplo mallas dependientes.",
            f"No hay parámetros que ajustar: los datos son $\\varepsilon_1={e1:.0f}$ V, $\\varepsilon_2={e2:.0f}$ V, $R_1={r1:.0f}$, $R_2={r2:.0f}$, $R_3={r3:.0f}\\ \\Omega$. El resultado son las corrientes de malla; las de rama se obtienen restando, $I_3=i_1-i_2$ en la resistencia compartida. La confianza se evalúa con residuos y conservación de potencia, no con un paso $h$.",
            "Es independiente de la solución analítica: ésta aplica Cramer con el determinante explícito, mientras que aquí el algoritmo es una factorización con pivoteo y las comprobaciones (Kirchhoff en cada malla y balance de potencia entre fuentes y resistencias) usan relaciones físicas que no participaron en el cálculo.",
        ],
        numeric_code="""\
# mallas: izquierda (E1, R1, R3) y derecha (E2, R2, R3); R3 es compartida
R = array([[R1 + R3, -R3],
           [-R3,     R2 + R3]])                  # matriz de resistencias de malla
E = array([E1, -E2])                             # fuentes (la derecha va en sentido contrario)
i = linalg.solve(R, E)                           # LU con pivoteo: corrientes de malla [i1, i2]
I_R1, I_R2, I_R3 = i[0], i[1], i[0] - i[1]       # corrientes de rama (R3 recibe la diferencia)
res_malla_izq = E1 - R1*I_R1 - R3*I_R3          # LVK malla izquierda: debe dar ~ 0
res_malla_der = -E2 - R2*I_R2 + R3*I_R3         # LVK malla derecha: debe dar ~ 0
P_fuentes = E1*i[0] - E2*i[1]                    # potencia entregada por las fuentes
P_resist = R1*I_R1**2 + R2*I_R2**2 + R3*I_R3**2  # potencia disipada (Joule)
balance = P_fuentes - P_resist                   # conservacion de energia: ~ 0
kappa = linalg.cond(R)                           # numero de condicion de la matriz
""",
        numeric_converge=[
            ("Residuos y verificación cruzada", "Al no haber iteraciones, se verifica la solución sustituyéndola en todo lo que debe cumplir: la ecuación matricial, los voltajes de Kirchhoff en cada malla y el balance de potencia. Los residuos deben ser del orden del redondeo ($\\sim10^{-15}$ en estas magnitudes).",
             [rf"\|\mathbf R\vec i-\vec\varepsilon\|={sci(float(np.linalg.norm(res)),2)}\ \mathrm{{V}},\qquad \max|i^{{num}}-i^{{an}}|={sci(float(np.max(np.abs(i-ia))),2)}\ \mathrm{{A}}",
              rf"\text{{LVK izq.}}={sci(kr['loop_left'],2)}\ \mathrm{{V}},\quad \text{{LVK der.}}={sci(kr['loop_right'],2)}\ \mathrm{{V}},\quad P_{{fuentes}}-P_{{Joule}}={sci(kr['power_balance'],2)}\ \mathrm{{W}}"]),
            ("Condicionamiento y sensibilidad a los datos", f"El número de condición es $\\kappa={fx(kap,3)}$, de modo que el redondeo se amplifica a lo sumo unas {fx(kap,1)} veces: error relativo $\\lesssim{sci(kap*2.2e-16,1)}$. El error que sí importa en la práctica es el de los datos: aquí se sube cada parámetro un 1 % y se recalculan las corrientes.",
             [tab]),
            ("Criterio de confiabilidad", "El resultado es confiable cuando (i) la matriz no es casi singular ($\\kappa$ moderado), (ii) los residuos de LVK y de potencia están en el nivel del redondeo, y (iii) coincide con una segunda vía independiente. Las tres condiciones se cumplen aquí, así que el error numérico es despreciable frente a cualquier tolerancia real de resistencias (1 % o mayor).",
             [rf"\text{{error numérico}}\sim{sci(kap*2.2e-16,1)}\ \ll\ \text{{efecto de }}1\%\ \text{{en un dato}}\sim{fx(max(abs(float(r_[1])) for r_ in rows),2)}\ \%"]),
        ],
    )


# ------------------------------------------------------------------------------------------ RCCharging
def _rc() -> dict:
    eps, r, c = e7.EPS, e7.R, e7.C
    tau = r * c
    qinf = c * eps
    ns = [6, 12, 24, 48, 96]
    rows, e_rk, e_eu = [], [], []
    for n in ns:
        t, y = e7.rc_numeric(1.2, n)
        rhs = lambda _t, yy: np.array([(eps - yy[0] / c) / r])  # noqa: E731
        _, ye = integrate(rhs, np.array([0.0]), 0.0, 1.2, n, euler_step)
        qa = e7.q_analytic(t)
        e_rk.append(float(np.abs(y[:, 0] - qa).max() / qinf))
        e_eu.append(float(np.abs(ye[:, 0] - qa).max() / qinf))
        rows.append([str(n), fx(1.2 / n / tau, 4), sci(e_eu[-1], 3), sci(e_rk[-1], 3)])
    hs = [1.2 / n for n in ns]
    p_rk, p_eu = _slope(hs, e_rk), _slope(hs, e_eu)
    tab = _table([r"n", r"h/\tau", r"\text{Euler}", r"\text{RK4}"], rows)
    en = e7.rc_energies_numeric()
    e_ref = 0.5 * c * eps**2
    t_num = e7.time_to_fraction_numeric()
    t_an = -tau * np.log(1 - 0.632)
    return dict(
        numeric_intro=[
            "La carga del capacitor obedece a la ecuación diferencial $dq/dt=(\\varepsilon-q/C)/R$. Esta ecuación sí se integra a mano, pero un integrador numérico sirve igual para circuitos no lineales (diodos, resistencias que dependen de la temperatura) donde no hay solución cerrada; aquí se prueba con un caso en que podemos medir su error exacto.",
            "El método de Runge-Kutta de cuarto orden (RK4) avanza con cuatro evaluaciones de la pendiente $f(t,q)$ por paso: $k_1=f(t_n,q_n)$, $k_2=f(t_n+h/2,\\,q_n+\\tfrac h2k_1)$, $k_3=f(t_n+h/2,\\,q_n+\\tfrac h2k_2)$, $k_4=f(t_n+h,\\,q_n+hk_3)$, y luego $q_{n+1}=q_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$. Equivale a una cuadratura de Simpson de la pendiente a lo largo del paso.",
            "El esquema coincide con el desarrollo de Taylor de la solución hasta $h^4$: el error local es $O(h^5)$ y el global $O(h^4)$ (Euler: $O(h)$). Para este problema lineal, $q_{n+1}=q_n\\,R(z)+\\dots$ con $z=-h/\\tau$ y $R(z)=1+z+z^2/2+z^3/6+z^4/24$ aproximando $e^z$; funciona bien mientras $h$ no sea mucho mayor que $\\tau$ (límite de estabilidad $|z|\\lesssim2.8$) y falla si la solución tiene cambios bruscos que el paso no resuelve.",
            f"Se integra hasta $t=1.2$ s con $n=2400$ pasos, es decir $h={sci(1.2/2400,3)}$ s $=\\tau/{fx(tau/(1.2/2400),0)}$ con $\\tau=RC={fx(tau,2)}$ s. Es una resolución mucho más fina de la necesaria (la tabla muestra que con $h=\\tau/16$ ya se tiene un error de $5\\times10^{{-8}}\\,q_\\infty$), pero deja el error muy por debajo de lo que se dibuja. Con la misma solución se calculan las energías con Simpson y el tiempo en que $q$ llega al 63.2 % por interpolación lineal.",
            "Es independiente del analítico: el integrador sólo conoce el lado derecho $f(t,q)$ de la ecuación diferencial; nunca usa $e^{-t/RC}$. Esa exponencial aparece únicamente al final para medir el error.",
        ],
        numeric_code="""\
f = lambda t, q: (eps - q / C) / R             # dq/dt segun la ley de Kirchhoff de la malla
h = t_fin / n                                  # paso fijo (n = 2400 pasos)
q = 0.0                                        # capacitor descargado
for k in range(n):
    k1 = f(t, q)                               # pendiente al inicio del paso
    k2 = f(t + h/2, q + h/2 * k1)              # pendiente en el punto medio (con k1)
    k3 = f(t + h/2, q + h/2 * k2)              # pendiente en el punto medio (con k2)
    k4 = f(t + h,   q + h   * k3)              # pendiente al final del paso
    q = q + h/6 * (k1 + 2*k2 + 2*k3 + k4)      # promedio ponderado tipo Simpson
    t = t + h
    guardar(t, q)                              # trayectoria q(t)
i = (eps - q_array / C) / R                    # corriente a partir de q
E_disip = simpson(i**2 * R, t)                 # energia disipada en R
E_cap = q_final**2 / (2 * C)                   # energia final del capacitor
""",
        numeric_converge=[
            ("Error contra paso", f"Se integra la misma ecuación con Euler y con RK4 para $n=6,\\dots,96$ pasos y se mide el error máximo de $q(t)$ respecto de $q_{{an}}$, normalizado con $q_\\infty={sci(qinf*1e3,3)}$ mC. Pendientes observadas: Euler {fx(p_eu,2)} (tiende a 1 al refinar), RK4 {fx(p_rk,2)}, compatibles con órdenes 1 y 4.",
             [tab, rf"p_{{Euler}}={fx(p_eu,2)},\qquad p_{{RK4}}={fx(p_rk,2)}"]),
            ("Verificaciones físicas con Simpson e interpolación", f"Con $n=6000$ pasos hasta $3$ s las energías deben cumplir que la batería entrega $\\varepsilon q_\\infty$, la mitad queda en el capacitor y la mitad se disipa. El tiempo al 63.2 % sale de interpolar entre los dos pasos que cruzan el valor.",
             [rf"E_{{cap}}={sci(en['capacitor']*1e3,5)}\ \mathrm{{mJ}},\ \ E_{{disip}}={sci(en['dissipated']*1e3,5)}\ \mathrm{{mJ}},\ \ E_{{bat}}={sci(en['battery']*1e3,5)}\ \mathrm{{mJ}}\ \ (\tfrac12C\varepsilon^2={sci(e_ref*1e3,5)}\ \mathrm{{mJ}})",
              rf"t_{{63.2\%}}^{{num}}={sci(t_num,6)}\ \mathrm{{s}},\qquad t^{{an}}={sci(t_an,6)}\ \mathrm{{s}},\qquad \text{{error rel.}}={sci(abs(t_num-t_an)/t_an,2)}"]),
            ("Fuentes de error y criterio", f"Para $h=\\tau/16$ el truncamiento de RK4 ya es $\\sim10^{{-7}}$ de $q_\\infty$ y a $h=\\tau/400$ (la escena) es muchísimo menor; el error del tiempo al 63.2 % proviene sobre todo de la interpolación lineal entre pasos, $O(h^2)$. El redondeo ($\\sim10^{{-16}}$ por paso) sólo se nota con $h$ minúsculo. Se acepta el resultado si al duplicar $n$ el error baja en un factor cercano a 16 y si el balance de energías se cumple dentro del error esperado.",
             [rf"\text{{error máx. a }}n=96:\ {sci(e_rk[-1],2)}\ q_\infty"]),
        ],
    )


# ------------------------------------------------------------------------------------------ HelicalMotion
def _helical() -> dict:
    ra, ta = m8.radius_analytic(), m8.period_analytic()
    spts = [10, 20, 50, 100, 200]
    rows, drb, drr, ept_b, ept_r = [], [], [], [], []
    for s in spts:
        tb, yb = m8.helix_boris(steps_per_turn=s)
        tr, yr = m8.helix_rk4(steps_per_turn=s)
        drb.append(m8.speed_drift(yb))
        drr.append(m8.speed_drift(yr))
        ept_b.append(abs(m8.measured_period(tb, yb) / ta - 1))
        ept_r.append(abs(m8.measured_period(tr, yr) / ta - 1))
        rows.append([str(s), sci(drr[-1], 2), sci(drb[-1], 2), sci(ept_r[-1], 2), sci(ept_b[-1], 2)])
    hs = [1 / s for s in spts]
    pb, pr = _slope(hs, ept_b), _slope(hs, ept_r)
    tab = _table([r"\text{pasos/vuelta}", r"\Delta|v|/|v|\ \text{RK4}", r"\Delta|v|/|v|\ \text{Boris}", r"\text{error }T\ \text{RK4}", r"\text{error }T\ \text{Boris}"], rows)
    return dict(
        numeric_intro=[
            "Una partícula en un campo magnético uniforme se resuelve a mano, pero en campos no uniformes (botellas magnéticas, tokamaks, aceleradores) no hay trayectoria cerrada y se integran las ecuaciones de movimiento durante millones de vueltas. Ahí importa que el integrador respete la propiedad física clave: el campo magnético no hace trabajo, así que $|\\vec v|$ debe permanecer constante.",
            "El esquema de Boris separa el paso en una rotación pura. Con $\\vec t=\\tfrac{q\\vec B}{m}\\tfrac{\\Delta t}{2}$ y $\\vec s=\\dfrac{2\\vec t}{1+t^2}$ se calcula $\\vec v'=\\vec v_n+\\vec v_n\\times\\vec t$, luego $\\vec v_{n+1}=\\vec v_n+\\vec v'\\times\\vec s$ y por último $\\vec x_{n+1}=\\vec x_n+\\vec v_{n+1}\\Delta t$. La transformación $\\vec v_n\\to\\vec v_{n+1}$ es una rotación exacta (de ángulo $2\\arctan t$), por eso conserva el módulo para cualquier $\\Delta t$.",
            "Su orden de precisión es 2: el ángulo girado por paso es $2\\arctan(\\omega\\Delta t/2)=\\omega\\Delta t-(\\omega\\Delta t)^3/12+\\dots$, así que la frecuencia de giro tiene un error relativo $O(\\Delta t^2)$ y el radio y el periodo medidos heredan ese error. El RK4 tiene orden 4 en la trayectoria, pero no conserva la rapidez: pierde o gana una fracción pequeña que se acumula vuelta tras vuelta, que es lo que lo hace malo para simulaciones muy largas.",
            "Se usan 50 pasos por vuelta ($\\Delta t=T/50$) y 4 vueltas. Es un compromiso: a ese paso el error del periodo con Boris es del orden de 0.1 % (suficiente para dibujar), mientras que la rapidez se conserva a precisión de máquina, y las 200 iteraciones son baratas. El resultado se interpreta midiendo en la trayectoria el radio (mitad del rango en $y$) y el periodo (cruces ascendentes con interpolación) y comparándolos con $mv_\\perp/(qB)$ y $2\\pi m/(qB)$.",
            "Es independiente del analítico: Boris sólo usa la regla de rotación y $q\\vec B/m$; nunca aparecen $r=mv/qB$ ni $T=2\\pi m/qB$. El radio y el periodo se miden después sobre los puntos calculados, como se haría con datos experimentales.",
        ],
        numeric_code="""\
T = 2*pi*m / (q*B); n = 4 * 50; dt = 4*T / n      # 4 vueltas, 50 pasos por vuelta
v = [v_perp, 0, v_par]; x = [0, 0, 0]             # condiciones iniciales
tvec = (q/m) * [0, 0, B] * dt / 2                 # medio angulo de giro por paso
svec = 2*tvec / (1 + dot(tvec, tvec))             # factor que hace la rotacion exacta
for k in range(n):
    vprima = v + cross(v, tvec)                   # primer medio giro
    v = v + cross(vprima, svec)                   # giro completo (conserva |v|)
    x = x + v * dt                                # avance de la posicion
    guardar(x, v)
r_num = (y.max() - y.min()) / 2                   # radio: mitad del rango en y
T_num = t[cruce2] - t[cruce1]                     # periodo: dos cruces ascendentes por y medio
deriva = max(abs(norm(v) - norm(v0))) / norm(v0)  # conservacion de la rapidez
""",
        numeric_converge=[
            ("Rapidez y periodo contra pasos por vuelta", "Se repite la integración con 10 a 200 pasos por vuelta, con Boris y con RK4, y se mide la deriva relativa de la rapidez y el error del periodo respecto de $2\\pi m/(qB)$. Boris conserva $|v|$ al redondeo en todos los casos, mientras RK4 la degrada cada vez menos al refinar.",
             [tab]),
            ("Orden observado", f"La pendiente de $\\log(\\text{{error en }}T)$ contra $\\log\\Delta t$ confirma los órdenes teóricos: Boris es de orden 2 y RK4 de orden 4. Esto explica que a 50 pasos por vuelta RK4 sea más preciso en el periodo (${sci(ept_r[2],2)}$ contra ${sci(ept_b[2],2)}$) pero Boris conserve la energía exactamente.",
             [rf"p_{{Boris}}={fx(pb,2)}\ (\text{{teórico }}2),\qquad p_{{RK4}}={fx(pr,2)}\ (\text{{teórico }}4)"]),
            ("Fuentes de error y criterio", f"En esta escena el error dominante es el de fase de Boris (${sci(ept_b[2],2)}$ en el periodo) más el de medida del radio por muestreo discreto; el redondeo ($\\sim10^{{-16}}$) es irrelevante. Criterio práctico: el resultado es confiable si el error del periodo cae por 4 al duplicar los pasos por vuelta y la deriva de $|v|$ permanece al nivel de máquina; para trayectorias muy largas Boris es preferible aunque RK4 gane en una sola vuelta.",
             [rf"\text{{deriva Boris a 50 pasos/vuelta: }}{sci(drb[2],2)},\qquad \text{{RK4: }}{sci(drr[2],2)}"]),
        ],
    )


# ------------------------------------------------------------------------------------------ HallEffect
def _hall_vh(steps: int = 800, iters: int = 80, t_tau: float = 40.0) -> float:
    """Copia parametrizable de m8.hall_voltage_numeric (mismo algoritmo)."""
    i_, b, n, t, w, tau = m8.I_HALL, m8.B_HALL, m8.N_CU, m8.THICK, m8.WIDTH, m8.TAU
    vd = i_ / (n * E_CH * w * t)
    ex = M_E * vd / (E_CH * tau)
    k = E_CH / M_E

    def vy(ey):
        def rhs(_t, v):
            return np.array([-k * (ex + v[1] * b) - v[0] / tau, -k * (ey - v[0] * b) - v[1] / tau])
        _, y = integrate(rhs, np.zeros(2), 0.0, t_tau * tau, steps, rk4_step)
        return float(y[-1, 1])

    lo, hi = -2 * vd * b, 2 * vd * b
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if vy(mid) > 0:
            lo = mid
        else:
            hi = mid
    return abs(0.5 * (lo + hi)) * w


def _hall() -> dict:
    va = m8.hall_voltage_analytic()
    vn = m8.hall_voltage_numeric()
    vd = m8.I_HALL / (m8.N_CU * E_CH * m8.WIDTH * m8.THICK)
    width0 = 4 * vd * m8.B_HALL
    rows_b = []
    for it in (5, 10, 20, 30, 40):
        v = _hall_vh(iters=it)
        rows_b.append([str(it), sci(width0 / 2**it, 3), sci(abs(v - va) / va, 3)])
    tab_b = _table([r"k", r"\text{ancho del intervalo}\ (\mathrm{V/m})", r"|V_H-V_H^{an}|/V_H^{an}"], rows_b)
    rows_s = []
    for st in (4, 8, 16, 50, 200):
        v = _hall_vh(steps=st)
        rows_s.append([str(st), fx(40.0 / st, 2), fx(abs(v - va) / va, 4) if abs(v - va) / va > 1e-6 else sci(max(abs(v - va) / va, 1e-16), 1)])
    tab_s = _table([r"\text{pasos RK4}", r"h/\tau", r"\text{error relativo}"], rows_s)
    rows_t = []
    for tt in (3, 5, 10, 20, 40):
        v = _hall_vh(steps=20 * tt, t_tau=tt)
        rows_t.append([str(tt), sci(max(abs(v - va) / va, 1e-16), 2)])
    tab_t = _table([r"T_{int}/\tau", r"\text{error relativo}"], rows_t)
    return dict(
        numeric_intro=[
            "El voltaje de Hall analítico sale de igualar la fuerza eléctrica transversal con la magnética, $eE_H=ev_dB$, en el estado estacionario. Aquí se busca el mismo campo transversal de otra manera: se simula la dinámica de los electrones con la ecuación de Drude con campo magnético, $m\\,d\\vec v/dt=-e(\\vec E+\\vec v\\times\\vec B)-m\\vec v/\\tau$, y se busca el valor de $E_y$ que anula la corriente transversal en régimen estacionario.",
            "Son dos algoritmos anidados. El interno integra la ecuación con RK4 desde $\\vec v=0$ hasta un tiempo largo ($40\\tau$) y devuelve $v_y$ final. El externo es una bisección sobre $E_y$: con un intervalo $[a,b]$ donde $v_y$ cambia de signo, se evalúa el punto medio y se descarta la mitad que no contiene el cero, de modo que tras $k$ iteraciones el intervalo mide $(b-a)/2^k$. La bisección siempre converge si la función es continua y cambia de signo, pero sólo gana un bit por evaluación (convergencia lineal).",
            "Los errores son tres. La bisección: tras 80 iteraciones el intervalo mide menos que $10^{-24}$ de su ancho inicial, muy por debajo del redondeo. El de RK4: como la ecuación es lineal, el punto fijo del esquema (donde $k_1=0$) coincide exactamente con el equilibrio verdadero, de modo que el estado estacionario no tiene error de truncamiento; $h$ sólo determina si el transitorio se resuelve y si el paso es estable ($h/\\tau\\lesssim2.8$ para la parte de relajación). El de tiempo finito: se corta en $T$ y queda un residuo $\\sim e^{-T/\\tau}$.",
            f"Los parámetros salen de estas condiciones: 800 pasos en $40\\tau$ da $h=\\tau/20$, claramente estable y que resuelve la rotación del ciclotrón (para $B=0.5$ T, $\\omega_c\\tau\\approx{sci(E_CH*m8.B_HALL/M_E*m8.TAU,2)}$, muy pequeño); $40\\tau$ vuelve el transitorio $\\sim10^{{-17}}$; 80 bisecciones superan la precisión doble. El resultado, $V_H={sci(vn,5)}$ V, es el voltaje transversal para $I=5$ A, $B=0.5$ T, $t=0.1$ mm.",
            "Es independiente del analítico: éste no resuelve ninguna dinámica, supone el balance de fuerzas. La simulación obtiene el balance sin imponerlo, pues sólo exige que no haya corriente transversal ($v_y=0$) y la dinámica produce el campo que lo compensa. La coincidencia con $IB/(nqt)$ confirma que la hipótesis de equilibrio de fuerzas es correcta.",
        ],
        numeric_code="""\
vd = I / (n*q*w*t)                                # deriva longitudinal impuesta por la corriente
Ex = m * vd / (q * tau)                           # campo longitudinal que sostiene esa deriva
def vy_estacionaria(Ey):                          # integrador interno (RK4)
    f = lambda t, v: [-(q/m)*(Ex + v[1]*B) - v[0]/tau,   # dvx/dt (fuerza + friccion de Drude)
                      -(q/m)*(Ey - v[0]*B) - v[1]/tau]   # dvy/dt
    v = rk4(f, [0, 0], 0, 40*tau, 800)            # 800 pasos en 40 tau (h = tau/20)
    return v[-1][1]                               # vy al final: ya estacionaria
lo, hi = -2*vd*B, 2*vd*B                          # intervalo que encierra la raiz
for k in range(80):                               # biseccion: el intervalo se divide por 2
    mid = (lo + hi) / 2
    if vy_estacionaria(mid) > 0: lo = mid         # la raiz esta en [mid, hi]
    else: hi = mid                                # la raiz esta en [lo, mid]
Ey = (lo + hi) / 2                                # campo de Hall que anula la corriente lateral
VH = abs(Ey) * w                                  # voltaje de Hall = Ey por el ancho
""",
        numeric_converge=[
            ("Error contra iteraciones de la bisección", f"Se fija el integrador interno y se varía el número $k$ de bisecciones. El ancho del intervalo se divide entre 2 en cada una, y el error de $V_H$ cae en esa proporción hasta llegar al redondeo (con $k=40$ ya se alcanza $\\sim10^{{-12}}$; las 80 de la escena sobran).",
             [tab_b]),
            ("Error contra paso del integrador y contra tiempo de integración", "Se varía el número de pasos de RK4 en $40\\tau$. Con pocos pasos ($h/\\tau\\gtrsim2.5$) el esquema es inestable o no relaja y el resultado es erróneo; con $h\\lesssim\\tau$ el error de truncamiento desaparece porque el estado estacionario es el punto fijo de RK4. Después se fija $h=\\tau/20$ y se acorta el tiempo $T$: el error es el transitorio residual y cae exponencialmente.",
             [tab_s, tab_t]),
            ("Fuentes de error y criterio de confianza", f"En la escena el valor numérico es $V_H={sci(vn,6)}$ V frente a $V_H^{{an}}={sci(va,6)}$ V (diferencia relativa ${sci(abs(vn-va)/va,1)}$, nivel de redondeo). Las fuentes son: bisección (insignificante con 80 iteraciones), truncamiento de RK4 (nulo en el equilibrio) y tiempo finito (residuo $e^{{-40}}$). Se acepta el resultado cuando (i) $h<\\tau$, (ii) $T\\gtrsim 20\\tau$ y (iii) el intervalo inicial cambia de signo; si no se cumple (i), el signo de $v_y$ no es fiable y la bisección converge a un valor falso.",
             []),
        ],
    )


DETAIL = {
    "CoaxialLaplace": _coax(),
    "CapacitorNetwork": _network(),
    "DrudeElectrons": _drude(),
    "TaperedWire": _tapered(),
    "MeshCircuit": _mesh(),
    "RCCharging": _rc(),
    "HelicalMotion": _helical(),
    "HallEffect": _hall(),
}
