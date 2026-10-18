"""Explicacion del metodo numerico (intro, codigo, convergencia). Volumen 2, caps. 23 a 25:
BandStructure, DiodeCircuit, DecayChain, BindingEnergy, TwoBodyDecay, FriedmannAge.
Todas las cifras de convergencia se calculan aqui ejecutando las funciones de physics/ (o replicando su algoritmo)."""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch23_solids as m23
from physics import v2ch24_nuclear as m24
from physics import v2ch25_particles as m25
from physics.integrators import rk4_step


def _tab(head: list[str], rows: list[list[str]]) -> str:
    """Tabla TeX (\\begin{array}) con encabezado y filas ya formateadas."""
    spec = "c" * len(head)
    body = r" \\ ".join(" & ".join(r) for r in rows)
    return r"\begin{array}{" + spec + "}" + " & ".join(head) + r" \\ \hline " + body + r"\end{array}"


def _slope(x, y) -> float:
    """Pendiente de log|y| contra log x (orden observado)."""
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.abs(np.asarray(y, float))), 1)[0])


# ====================================================================== BandStructure
def _band() -> dict:
    g = m23.G_STRENGTH
    ana = m23.band_edges_analytic(3)
    ns = [11, 21, 51, 101, 201, 401, 801]
    errs, per = [], {}
    for n in ns:
        e = m23.band_edges_numeric(3, n)
        per[n] = e
        errs.append(float(np.abs(e - ana).max()))
    sl = _slope(ns, errs)
    rows = [[str(n), fx(float(per[n][0]), 5), sci(er, 3), fx(errs[i - 1] / er, 2) if i else "-"] for i, (n, er) in enumerate(zip(ns, errs))]
    t1 = _tab([r"N", r"E_1\ (\mathrm{Ha})", r"\max|\Delta E|", r"\text{cociente}"], rows)
    n1, n2 = 201, 401
    e1, e2 = per[n1], per[n2]
    extra = (n2 * e2 - n1 * e1) / (n2 - n1)  # E(N) = E_inf + c/N
    err_ex = np.abs(extra - ana)
    e_side = per[201] - ana
    # residuo en la relacion de dispersion: |F(q)| - 1 en cada borde numerico
    fq = lambda e: np.cos(np.sqrt(2 * e)) + g / np.sqrt(2 * e) * np.sin(np.sqrt(2 * e))  # noqa: E731
    res = np.abs(np.abs(fq(per[801])) - 1.0)
    rows3 = [[str(i + 1), fx(float(ana[i]), 6), sci(float(e_side[i]), 3), sci(float(err_ex[i]), 3), sci(float(res[i]), 3)] for i in range(6)]
    t3 = _tab([r"\text{borde}", r"E_{\mathrm{anal}}", r"E_{201}-E_{\mathrm{anal}}", r"\text{error extrapolado}", r"\big||F|-1\big|\ (N=801)"], rows3)
    ev = np.linalg.eigvalsh(np.full((201, 201), g) + np.diag(0.5 * (2 * np.pi * np.arange(-100, 101)) ** 2))
    ruido = float(np.finfo(float).eps * np.abs(ev).max())  # perturbacion absoluta tipica de un autovalor por redondeo
    ex_ok = [i + 1 for i in range(6) if abs(e_side[i]) < 1e-9]
    return dict(
        numeric_intro=[
            r"En lugar de resolver la ecuación trascendente de Kronig-Penney, el código convierte la ecuación de Schrödinger en un problema de álgebra lineal: se expande la función de onda de Bloch en ondas planas $e^{i(k+2\pi m)x}$ y se diagonaliza el hamiltoniano resultante. Es el método estándar de los cálculos de estructura de bandas reales, donde no existe ninguna relación de dispersión cerrada y solo queda diagonalizar. Se necesita porque, para un potencial periódico cualquiera, no hay fórmula para $E(k)$.",
            r"La regla es el método variacional de Ritz sobre una base truncada: con $m=-M,\dots,M$ (aquí $N=2M+1$ ondas) se construye la matriz $H_{mm'}=\tfrac12(k+2\pi m)^2\,\delta_{mm'}+g$, pues el peine de deltas acopla todas las ondas con el mismo elemento $V_{mm'}=g$ (con $a=1$). Los autovalores de esa matriz hermítica $N\times N$ son las energías $E_n(k)$. Los bordes de banda son los autovalores en $k=0$ y $k=\pi$.",
            r"Como la base truncada es un subespacio, los autovalores obtenidos son cotas superiores de los exactos (principio min-max) y bajan monótonamente al agregar ondas. El error, sin embargo, converge lento: la función de onda tiene un quiebre en cada delta, sus coeficientes de Fourier decaen como $1/m^2$ y el error de la energía baja como $O(1/N)$, no de forma exponencial como ocurriría con un potencial suave. Esta lentitud es la principal limitación del método con potenciales singulares.",
            r"Se eligió $N=201$ ($M=100$) porque da un error del orden de $10^{-2}$ Ha en las tres primeras bandas con una matriz que se diagonaliza en milisegundos, suficiente para ver con claridad las brechas (de varios Ha). Como se conoce la ley $E(N)\approx E_\infty+c/N$, dos tamaños permiten extrapolar y ganar un orden de magnitud. El resultado físico es la lista de seis bordes de banda y, de ahí, las brechas del cristal.",
            r"El método es independiente del analítico: no usa $\cos k=\cos q+(g/q)\sin q$ en ningún momento, solo la energía cinética de las ondas planas y el acoplamiento $g$. Que ambos caminos coincidan al aumentar $N$ valida a la vez la relación de dispersión y la diagonalización.",
        ],
        numeric_code="""import numpy as np
g = 4.0                                   # intensidad de cada delta (a = m = hbar = 1)
def bandas(k, n_bandas=3, N=201):
    m = np.arange(-(N // 2), N // 2 + 1)  # indices de onda plana: -100 ... 100
    H = np.full((N, N), g)                # V_mm' = g para todo par (peine de deltas)
    H[np.diag_indices(N)] += 0.5 * (k + 2*np.pi*m)**2   # energia cinetica en la diagonal
    return np.linalg.eigvalsh(H)[:n_bandas]             # autovalores ordenados; los 3 menores
lo = bandas(0.0)                          # bordes en el centro de zona k = 0
hi = bandas(np.pi)                        # bordes en el borde de zona k = pi
bordes = np.sort(np.concatenate([lo, hi]))   # 6 energias: extremos de las 3 bandas
# comparacion independiente con la solucion de Kronig-Penney (brentq sobre |F(q)| = 1)
analitico = band_edges_analytic(3)
error = np.abs(bordes - analitico).max()  # error maximo; baja como 1/N
extrap = (N2*E_N2 - N1*E_N1) / (N2 - N1)  # extrapolacion de Richardson para c/N
""",
        numeric_converge=[
            ("Error contra el tamaño de la base", "Se repite la diagonalización con $N$ ondas planas y se compara cada uno de los seis bordes con el valor analítico; se muestra el mayor error. El cociente entre errores consecutivos refleja cuánto se reduce al cambiar $N$.",
             [t1, rf"\text{{pendiente de }}\log|\Delta E|\text{{ contra }}\log N = {fx(sl, 3)}"]),
            ("Lo que indica la pendiente", rf"La pendiente observada es {fx(sl, 2)}, es decir $|\Delta E|\propto N^{{{fx(sl, 2)}}}\approx 1/N$, como predice la teoría para un potencial con quiebres. Duplicar $N$ solo divide el error entre dos, de modo que bajar de $10^{{-2}}$ a $10^{{-4}}$ exigiría matrices de miles de ondas. Los errores son positivos (el numérico queda por encima), señal de la cota variacional. Los bordes {", ".join(map(str, ex_ok))} salen exactos hasta el redondeo: son estados antisimétricos que se anulan en cada delta y por eso no sienten el potencial, así que la base truncada los reproduce sin error; el error de $1/N$ vive solo en los bordes simétricos.",
             [rf"E_{{201}}-E_{{\mathrm{{anal}}}}\ge0\ \text{{(salvo redondeo)}};\qquad \max = {sci(float(np.abs(e_side).max()), 3)}\ \mathrm{{Ha}}"]),
            ("Extrapolación y verificación cruzada", r"Como $E(N)\approx E_\infty+c/N$, con $N=201$ y $N=401$ se elimina el término $c/N$ (Richardson: $E_\infty\approx(N_2E_{N_2}-N_1E_{N_1})/(N_2-N_1)$), con lo que el error baja unos tres órdenes de magnitud. Además se comprueba cada borde numérico sustituyéndolo en la relación de dispersión (residuo $||F|-1|$), un criterio que no depende del valor analítico tabulado.",
             [t3, rf"\text{{Piso de redondeo de la diagonalización: }}\varepsilon\,\|H\|\approx{sci(ruido, 2)}\ \mathrm{{Ha}}\ \ (\|H\|={sci(float(np.abs(ev).max()), 3)}\ \mathrm{{Ha}})"]),
        ],
    )


# ====================================================================== DiodeCircuit
def _diode() -> dict:
    e, r = 5.0, m23.R_SERIES
    nvt = m23.N_IDEAL * m23.V_T
    v_ex = m23.diode_voltage_analytic(e)
    v_num = m23.diode_voltage_numeric(e)
    f = lambda v: m23.I_S * np.expm1(v / nvt) - (e - v) / r  # noqa: E731

    def biseccion(k: int) -> float:
        lo, hi = 0.0, e
        for _ in range(k):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    ks = [5, 10, 15, 20, 30, 40, 50]
    rows = [[str(k), sci(e / 2 ** (k + 1), 3), sci(abs(biseccion(k) - v_ex), 3)] for k in ks]
    t1 = _tab([r"k", r"\text{cota }E/2^{k+1}", r"|V_k-V_{\mathrm{Lambert}}|"], rows)
    # numero de iteraciones para llegar a una tolerancia
    tol = 1e-12
    k_tol = int(np.ceil(np.log2(e / tol)))
    # sensibilidades
    dv_de = (m23.diode_voltage_numeric(e + 1e-3) - m23.diode_voltage_numeric(e - 1e-3)) / 2e-3
    i_d = m23.diode_current(e, v_num)
    i_r = (e - v_num) / r
    i_dio = m23.I_S * np.expm1(v_num / nvt)
    old = m23.I_S
    try:
        m23.I_S = 2 * old
        dv_is = m23.diode_voltage_numeric(e) - v_num
    finally:
        m23.I_S = old
    fp = m23.I_S / nvt * np.exp(v_num / nvt) + 1 / r
    return dict(
        numeric_intro=[
            r"El circuito exige resolver $E=V_d+R\,I_S\,(e^{V_d/nV_T}-1)$, una ecuación trascendente: la tensión aparece fuera y dentro de una exponencial, y no se puede despejar con funciones elementales (la solución exacta usa la función $W$ de Lambert). Un método numérico de búsqueda de raíces evita ese recurso y funciona igual para cualquier curva $I(V)$ que no tenga forma cerrada, por ejemplo un diodo real con resistencia interna.",
            r"El método es la bisección. Se define $f(V)=I_S(e^{V/nV_T}-1)-(E-V)/R$, la diferencia entre la corriente que el diodo deja pasar y la que la resistencia permite. En $V=0$ vale $f<0$ y en $V=E$ vale $f>0$; como $f$ es continua y estrictamente creciente, hay una única raíz en $[0,E]$. Se toma el punto medio $m$, se conserva la mitad donde $f$ cambia de signo y se repite.",
            r"Cada iteración divide el intervalo a la mitad, así que tras $k$ pasos el error es a lo sumo $E/2^{k+1}$: convergencia lineal, un bit de precisión por iteración (orden 1, razón $1/2$). Es lento comparado con Newton, pero nunca falla mientras haya cambio de signo, no necesita derivadas y no se desborda, cosa que sí le ocurre a Newton partiendo de $V$ grande, donde $e^{V/nV_T}$ es astronómico.",
            rf"Los parámetros son un intervalo seguro $[0,E]$ (el diodo nunca supera la fuente) y 200 iteraciones fijas. Bastan {k_tol} para una tolerancia de $10^{{-12}}$ V; el resto solo agota la precisión de coma flotante, de unos $10^{{-16}}$ en relativo. El resultado, $V_d={fx(v_num, 6)}$ V para $E={e:g}$ V, es la tensión del diodo, de la que sale la corriente $I=(E-V_d)/R$.",
            r"El método es independiente del analítico porque nunca evalúa $W(x)$: solo compara signos de $f$ en puntos que él mismo elige. Coincidir con la expresión de Lambert hasta la precisión de máquina valida las dos soluciones y la ecuación de Shockley planteada.",
        ],
        numeric_code="""I_S, nVT, R = 1.0e-12, 0.025852, 100.0   # corriente de saturacion, N*V_T y resistencia
E = 5.0                                   # fuente (V)
def f(V):                                 # corriente del diodo menos corriente de la resistencia
    return I_S * np.expm1(V / nVT) - (E - V) / R   # expm1 evita perdida de cifras cerca de 0
lo, hi = 0.0, E                           # f(lo) < 0 y f(hi) > 0: la raiz esta dentro
for _ in range(200):                      # iteraciones fijas (la cota E/2^k llega al redondeo)
    mid = 0.5 * (lo + hi)                 # punto medio del intervalo
    if f(mid) > 0:                        # el diodo conduce de mas: la raiz esta a la izquierda
        hi = mid                          # nos quedamos con la mitad inferior
    else:
        lo = mid                          # la raiz esta a la derecha
V_d = 0.5 * (lo + hi)                     # tension del diodo
I = (E - V_d) / R                         # corriente del circuito (ley de mallas)
# verificacion independiente: V_d = E + I_S R - nVT * W(I_S R/nVT * exp((E + I_S R)/nVT))
""",
        numeric_converge=[
            ("Error contra número de iteraciones", rf"Se replica el bucle con $k$ iteraciones y se mide la distancia a la solución de Lambert ($V={fx(v_ex, 12)}$ V). El error observado se mantiene por debajo de la cota teórica $E/2^{{k+1}}$ y se estanca en el nivel del redondeo.",
             [t1, rf"\text{{Cada 10 iteraciones el error baja }}2^{{10}}={2**10}\text{{ veces; para }}10^{{-12}}\ \mathrm{{V}}\text{{ se necesitan }}k={k_tol}."]),
            ("Fuentes de error y piso de redondeo", rf"Con 200 iteraciones el intervalo llega a un ancho de un solo número de máquina y el error de truncamiento (el del método) desaparece. Queda el error de redondeo al evaluar $f$: del orden de $\varepsilon\cdot V\approx{sci(2.2e-16 * v_num, 2)}$ V. La pendiente de $f$ en la raíz es $f'={sci(float(fp), 3)}\ \mathrm{{A/V}}$ (diodo más resistencia), de modo que un error absoluto $\delta f$ en la corriente se traduce en $\delta V=\delta f/f'$, del mismo orden: el problema está bien condicionado.",
             [rf"|V_{{200}}-V_{{\mathrm{{Lambert}}}}|={sci(abs(v_num - v_ex), 3)}\ \mathrm{{V}}"]),
            ("Verificación cruzada y sensibilidad", r"Un criterio práctico de confianza es que las dos corrientes coincidan: la del diodo (ecuación de Shockley) y la de la resistencia (ley de Ohm). También se estudia cuánto cambia $V_d$ al variar los datos: la tensión del diodo es casi insensible a la fuente (casi toda la variación de $E$ cae en $R$) y se desplaza $-nV_T\ln2$ si se duplica $I_S$, porque la corriente es casi la misma y el diodo la conduce con menos tensión.",
             [rf"I_R=\dfrac{{E-V_d}}{{R}}={sci(float(i_r), 6)}\ \mathrm{{A}},\qquad I_D=I_S(e^{{V_d/nV_T}}-1)={sci(float(i_dio), 6)}\ \mathrm{{A}}",
              rf"\dfrac{{\mathrm dV_d}}{{\mathrm dE}}={fx(float(dv_de), 5)},\qquad V_d(2I_S)-V_d(I_S)={sci(float(dv_is), 3)}\ \mathrm{{V}}\ \ (-nV_T\ln2={sci(-nvt * np.log(2), 3)}\ \mathrm{{V}})"]),
        ],
    )


# ====================================================================== DecayChain
def _chain() -> dict:
    la, lb = m24.LAM_A, m24.LAM_B
    ns = [50, 100, 200, 400, 800, 1600, 4000]
    hs, errs = [], []
    for n in ns:
        t, y = m24.chain_numeric(20.0, n)
        hs.append(20.0 / n)
        errs.append(float(np.abs(y[:, 1] - m24.bateman_b(t)).max()))
    sl = _slope(hs, errs)
    rows = [[str(n), fx(h, 4), sci(er, 3), fx(errs[i - 1] / er, 1) if i else "-"] for i, (n, h, er) in enumerate(zip(ns, hs, errs))]
    t1 = _tab([r"n", r"h\ (\mathrm{s})", r"\max|N_B-N_B^{\mathrm{Bateman}}|", r"\text{cociente}"], rows)
    t, y = m24.chain_numeric(20.0, 4000)
    cons = float(np.abs(y.sum(axis=1) - 1.0).max())
    # maximo de N_B: afinado parabolico
    tp = m24.peak_time_analytic()
    def pico(n):
        tt, yy = m24.chain_numeric(20.0, n)
        k = int(np.argmax(yy[:, 1]))
        y0, y1, y2 = yy[k - 1, 1], yy[k, 1], yy[k + 1, 1]
        return float(tt[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (tt[1] - tt[0]))
    npk = [400, 2000, 10000, 40000]
    rows2 = [[str(n), fx(20.0 / n, 5), sci(abs(pico(n) - tp), 3)] for n in npk]
    t2 = _tab([r"n", r"h", r"|t_{\max}^{\mathrm{num}}-t_{\max}^{\mathrm{anal}}|"], rows2)
    # Monte Carlo
    nn = [100, 1000, 10000, 50000, 500000]
    rowsm = []
    sig = []
    for n in nn:
        mu, s = m24.mean_life_monte_carlo(n=n)
        sig.append(s)
        rowsm.append([f"{n}", fx(mu, 4), fx(s, 4), fx(abs(mu - 1 / lb), 4), fx(abs(mu - 1 / lb) / s, 2)])
    tm = _tab([r"N", r"\bar t\ (\mathrm{s})", r"\hat\sigma/\sqrt N", r"|\bar t-4|", r"\text{error}/\hat\sigma_{\bar t}"], rowsm)
    slm = _slope(nn, sig)
    # rms sobre semillas para N=1000
    nseeds = 300
    n_chk = 1000
    devs = np.array([m24.mean_life_monte_carlo(n=n_chk, seed=s)[0] - 1 / lb for s in range(nseeds)])
    rms = float(np.sqrt((devs**2).mean()))
    return dict(
        numeric_intro=[
            r"Las ecuaciones de la cadena $A\to B\to C$ son lineales y tienen solución cerrada (Bateman), pero esa solución solo existe porque el sistema es sencillo. El código integra directamente las ecuaciones diferenciales $\dot N_A=-\lambda_AN_A$, $\dot N_B=\lambda_AN_A-\lambda_BN_B$, $\dot N_C=\lambda_BN_B$, método que sirve igual para cadenas largas, ramificaciones o tasas que dependen del tiempo, donde Bateman ya no vale.",
            r"El integrador es Runge-Kutta de cuarto orden. Para $\dot{\vec y}=\vec F(t,\vec y)$ y paso $h$: $\vec k_1=\vec F(t_n,\vec y_n)$, $\vec k_2=\vec F(t_n+h/2,\vec y_n+h\vec k_1/2)$, $\vec k_3=\vec F(t_n+h/2,\vec y_n+h\vec k_2/2)$, $\vec k_4=\vec F(t_n+h,\vec y_n+h\vec k_3)$, y $\vec y_{n+1}=\vec y_n+\tfrac h6(\vec k_1+2\vec k_2+2\vec k_3+\vec k_4)$. Coincide con el desarrollo de Taylor hasta $h^4$, de modo que el error local es $O(h^5)$ y el global $O(h^4)$.",
            r"Con tasas $\lambda_A=1$ y $\lambda_B=0.25$ s$^{-1}$ el sistema no es rígido y RK4 es estable si $h\lambda\lesssim2.8$; aquí $h\lambda\sim5\times10^{-3}$ con $n=4000$ en $[0,20]$ s, muy por debajo. Si hubiera tasas muy distintas (rigidez) habría que bajar $h$ o usar un método implícito. Los 4000 pasos dan un error del orden de $10^{-12}$, más que suficiente; con el afinado parabólico del máximo y $n=40000$ el instante del pico queda determinado con unas ocho cifras (la interpolación parabólica del máximo tiene error $O(h^2)$).",
            r"La vida media se estudia por separado con Monte Carlo: cada tiempo de decaimiento se genera por inversión de la función de distribución, $t=-\ln(1-u)/\lambda_B$ con $u$ uniforme, y la vida media es el promedio. Es un estimador insesgado cuyo error estadístico es $\sigma/\sqrt N$ (orden $N^{-1/2}$, independiente de la dimensión del problema): para ganar un dígito hay que multiplicar $N$ por 100. Se usan $N=50000$ y semilla fija 11 para que el resultado sea reproducible.",
            r"Ninguno de los dos caminos usa la fórmula de Bateman ni la integral $\int t\,\lambda e^{-\lambda t}\mathrm dt=1/\lambda$: RK4 solo evalúa el lado derecho de las ecuaciones y Monte Carlo solo sortea números aleatorios. Por eso su acuerdo con las fórmulas cerradas es una verificación real.",
        ],
        numeric_code="""LA, LB = 1.0, 0.25                         # constantes de decaimiento (1/s)
def F(t, y):                               # lado derecho dy/dt del sistema A -> B -> C
    return np.array([-LA*y[0], LA*y[0] - LB*y[1], LB*y[1]])
def rk4(F, t, y, h):
    k1 = F(t, y)                           # pendiente al inicio del paso
    k2 = F(t + h/2, y + h/2*k1)            # pendiente en el punto medio usando k1
    k3 = F(t + h/2, y + h/2*k2)            # pendiente en el punto medio usando k2
    k4 = F(t + h, y + h*k3)                # pendiente al final usando k3
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4) # promedio ponderado: error global O(h^4)
n, T = 4000, 20.0; h = T / n               # 4000 pasos en [0, 20] s
y = np.array([1.0, 0.0, 0.0])              # N_A = 1, N_B = N_C = 0 al inicio
for i in range(n):
    y = rk4(F, i*h, y, h)                  # avanza un paso; se guarda la trayectoria
# pico de N_B: argmax + parabola por los tres puntos vecinos (afinado sub-paso)
t_pico = t[k] + 0.5*(y0 - y2)/(y0 - 2*y1 + y2) * h
# Monte Carlo de la vida media: inversion de la CDF
rng = np.random.default_rng(11)            # semilla fija: resultado reproducible
t_dec = -np.log(1 - rng.uniform(size=50000)) / LB   # tiempos exponenciales
vida = t_dec.mean(); err = t_dec.std(ddof=1)/np.sqrt(50000)   # media y error estadistico
""",
        numeric_converge=[
            ("RK4: error contra paso", rf"Se integra con distintos $n$ y se toma el mayor desvío de $N_B$ respecto de la solución de Bateman en toda la malla. Al duplicar $n$ el error se divide entre aproximadamente $2^4=16$.",
             [t1, rf"\text{{pendiente de }}\log\text{{error}}\text{{ contra }}\log h = {fx(sl, 3)}\ \ (\text{{teoría: }}4)"]),
            ("RK4: fuentes de error y conservación", rf"El error de truncamiento domina hasta $n=4000$ (${sci(errs[-1], 3)}$); el redondeo de cada paso es del orden de $10^{{-16}}$ y se acumula como $\sqrt n\,\varepsilon\sim10^{{-14}}$, todavía muy por debajo. Como criterio práctico de confianza, la conservación $N_A+N_B+N_C=1$ (que RK4 respeta de forma exacta por ser lineal el sistema, salvo redondeo) y el error del instante del pico tras el afinado parabólico (aquí cae como $h^2$, no como $h^4$, por la interpolación):",
             [rf"\max|N_A+N_B+N_C-1|={sci(cons, 3)}", t2]),
            ("Monte Carlo: error contra N", rf"La media de $N$ tiempos exponenciales se acerca a $1/\lambda_B=4$ s con un error que se reduce como $N^{{-1/2}}$ (pendiente de $\hat\sigma/\sqrt N$ contra $N$: {fx(slm, 3)}). El error real de una sola corrida es del orden de una desviación estándar, de ahí la última columna.",
             [tm]),
            ("Monte Carlo: error esperado y criterio", rf"Repitiendo la estimación con {nseeds} semillas distintas y $N={n_chk}$, el error rms es {fx(rms, 4)} s frente a la predicción $\sigma/\sqrt N=4/\sqrt{{{n_chk}}}={fx(4 / np.sqrt(n_chk), 4)}$ s. El resultado es confiable cuando se informa junto con su $\hat\sigma/\sqrt N$ y el valor exacto queda dentro de unas dos desviaciones; con $N=50000$ la incertidumbre relativa es del {fx(100 * sig[3] / 4, 2)} %.",
             [rf"\text{{rms}}={fx(rms, 4)}\ \mathrm{{s}}\ \approx\ \dfrac{{\sigma}}{{\sqrt N}}={fx(4 / np.sqrt(n_chk), 4)}\ \mathrm{{s}}"]),
        ],
    )


# ====================================================================== BindingEnergy
def _binding() -> dict:
    a_list = [20, 56, 90, 120, 160, 208, 238]
    zstar = lambda a: (4 * m24.A_A + m24.A_C / a ** (1 / 3)) / (2 * m24.A_C / a ** (1 / 3) + 8 * m24.A_A / a)  # noqa: E731
    rows = []
    for a in a_list:
        z0 = m24.z_stable_analytic(a)
        zn = m24.z_stable_numeric(a, False)
        rows.append([str(a), fx(z0, 3), fx(zstar(a), 3), str(zn), fx(abs(zn - zstar(a)), 3),
                     fx(m24.b_per_a(a, False), 5), fx(float(m24.b_per_a_analytic(a)), 5),
                     sci(float(m24.b_per_a_analytic(a)) - m24.b_per_a(a, False), 2)])
    t1 = _tab([r"A", r"Z_0", r"Z^*", r"Z_{\mathrm{num}}", r"|Z_{\mathrm{num}}-Z^*|", r"(B/A)_{\mathrm{num}}", r"(B/A)_{\mathrm{cont}}", r"\text{dif.}"], rows)
    aa = np.arange(20, 240)
    dz = np.array([abs(m24.z_stable_numeric(int(a), False) - m24.z_stable_analytic(int(a))) for a in aa])
    dzs = np.array([abs(m24.z_stable_numeric(int(a), False) - zstar(int(a))) for a in aa])
    dba = np.array([float(m24.b_per_a_analytic(int(a))) - m24.b_per_a(int(a), False) for a in aa])
    pk = m24.peak_a_numeric()
    # sensibilidad a un coeficiente y al apareamiento
    old = m24.A_C
    pks = {}
    try:
        for fac in (0.9, 1.0, 1.1):
            m24.A_C = old * fac
            pks[fac] = (m24.peak_a_numeric(), m24.z_stable_numeric(56, False))
    finally:
        m24.A_C = old
    z_con = m24.z_stable_numeric(56, True)
    z_sin = m24.z_stable_numeric(56, False)
    rows3 = [[fx(fac, 1), str(p[0]), str(p[1])] for fac, p in pks.items()]
    t3 = _tab([r"a_C/a_C^{0}", r"A_{\mathrm{pico}}", r"Z(A=56)"], rows3)
    return dict(
        numeric_intro=[
            r"La fórmula semiempírica da $B(A,Z)$ para cualquier núcleo, pero la pregunta física es otra: dado $A$, ¿qué $Z$ es el más estable y cuál es el pico de $B/A$ a lo largo de la tabla? Analíticamente se resuelve $\partial B/\partial Z=0$ tratando $Z$ como continuo y despreciando el apareamiento; en el código se prefiere una búsqueda directa que sirve incluso con el término de apareamiento, que depende de la paridad y por tanto no es derivable.",
            r"El método es una búsqueda exhaustiva (fuerza bruta) sobre un conjunto finito: para cada $A$ se evalúa $B(A,Z)=a_VA-a_SA^{2/3}-a_C\,Z(Z-1)A^{-1/3}-a_A(A-2Z)^2/A+\delta$ para $Z=1,\dots,A-1$ y se toma el $Z$ del máximo (`argmax`). Repetido para $A=20,\dots,239$, el pico de $B/A$ es el máximo de esa lista. No hay iteración ni tolerancia: el resultado es exacto en el conjunto de enteros.",
            r"Como $B$ es una parábola cóncava en $Z$, el máximo entero cae a una distancia de a lo sumo $1/2$ del óptimo continuo exacto $Z^*$ de $Z(Z-1)$ (que difiere en unas décimas del $Z_0$ de la fórmula, que usa $Z^2$); la pérdida en energía es de segundo orden, $\Delta B\approx\tfrac12|B''|\,(Z-Z^*)^2$, con $|B''|=8a_A/A+2a_C/A^{1/3}$ del orden de MeV, es decir, un error de redondeo y no de truncamiento. La búsqueda por fuerza bruta cuesta $O(A)$ evaluaciones por núcleo, trivial aquí.",
            r"Las limitaciones son de modelo, no de método: los coeficientes están ajustados a datos, la fórmula ignora capas cerradas (números mágicos) y deja de ser fiable para $A\lesssim20$. Por eso el resultado numérico no es verdad nuclear, sino la solución exacta del modelo. Los parámetros de código (rango $Z=1,\dots,A-1$, $A\in[20,239]$) se eligieron para cubrir todo el rango de núcleos estables.",
            r"Es independiente del analítico porque nunca usa la expresión cerrada $Z_0=\tfrac A2/(1+a_CA^{2/3}/4a_A)$: solo evalúa $B$ y compara. Que ambos coincidan hasta el redondeo a entero valida la derivada del modelo y la implementación de $B$.",
        ],
        numeric_code="""aV, aS, aC, aA, aP = 15.75, 17.8, 0.711, 23.7, 11.18   # coeficientes en MeV
def B(A, Z, apareamiento=True):
    N = A - Z                                    # numero de neutrones
    d = 0.0
    if apareamiento:
        if Z % 2 == 0 and N % 2 == 0: d = +aP / A**0.5     # par-par: mas ligado
        elif Z % 2 == 1 and N % 2 == 1: d = -aP / A**0.5   # impar-impar: menos ligado
    return (aV*A - aS*A**(2/3)                   # volumen y superficie
            - aC*Z*(Z-1)/A**(1/3)                # repulsion coulombiana
            - aA*(A-2*Z)**2/A + d)               # asimetria y apareamiento
def Z_estable(A, apareamiento=True):
    Zs = range(1, A)                             # todos los enteros posibles
    return max(Zs, key=lambda Z: B(A, Z, apareamiento))   # el de mayor energia de enlace
def B_por_A(A, ap=True):
    return B(A, Z_estable(A, ap), ap) / A        # energia de enlace por nucleon
pico = max(range(20, 240), key=lambda A: B_por_A(A, False))   # A del maximo de B/A
Z0 = (A/2) / (1 + aC*A**(2/3)/(4*aA))            # solo para comparar: optimo continuo
""",
        numeric_converge=[
            ("El error es de redondeo a entero", "No hay un paso $h$ que reducir: el conjunto de $Z$ es discreto. El error del método es cuánto se aleja el entero óptimo del óptimo continuo, y cuánto pierde $B/A$ por ello. Tabla para siete valores de $A$ (sin apareamiento); $Z^*$ es el óptimo continuo de la propia $B$ con $Z(Z-1)$:",
             [t1]),
            ("Barrido completo de A", rf"Para los 220 valores de $A$ entre 20 y 239 la distancia entre $Z_{{\mathrm{{num}}}}$ y $Z_0$ llega a {fx(float(dz.max()), 3)}, algo más de 0.5 porque la fórmula de $Z_0$ usa $Z^2$ en lugar de $Z(Z-1)$; respecto del óptimo exacto $Z^*$ la distancia máxima es {fx(float(dzs.max()), 3)} (cota teórica 0.5). La diferencia de $B/A$ entre el valor continuo y el entero va de ${sci(float(dba.min()), 2)}$ a ${sci(float(dba.max()), 2)}$ MeV (mayor a $A$ pequeño, donde $|B''|$ pesa más). El pico de $B/A$ está en $A={pk}$.",
             [rf"\max|Z_{{\mathrm{{num}}}}-Z_0|={fx(float(dz.max()), 3)},\quad \max|Z_{{\mathrm{{num}}}}-Z^*|={fx(float(dzs.max()), 3)},\quad \max\left[(B/A)_{{\mathrm{{cont}}}}-(B/A)_{{\mathrm{{num}}}}\right]={sci(float(dba.max()), 3)}\ \mathrm{{MeV}}"]),
            ("Sensibilidad del modelo y criterio de confiabilidad", rf"La incertidumbre dominante no es numérica sino la de los coeficientes. Variando $a_C$ en $\pm10\,\%$ se mueve mucho el pico de $B/A$ (tabla) y poco el $Z$ estable de $A=56$; activar el apareamiento cambia la paridad del $Z$ óptimo (A=56: $Z={z_sin}$ sin apareamiento, $Z={z_con}$ con él). Frente a ese efecto del 10 % en $a_C$, el redondeo a entero es inocuo. El resultado se considera confiable cuando el óptimo entero está dentro de $\pm1$ del óptimo continuo y la diferencia de $B/A$ es menor que la incertidumbre de los coeficientes.",
             [t3]),
        ],
    )


# ====================================================================== TwoBodyDecay
def _twobody() -> dict:
    m, m1, m2 = m25.M_PARENT, m25.M_1, m25.M_2
    ps = m25.p_star_analytic()
    f = lambda p: np.hypot(p, m1) + np.hypot(p, m2) - m  # noqa: E731

    def bis(k: int) -> float:
        lo, hi = 0.0, m
        for _ in range(k):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    ks = [5, 10, 20, 30, 40, 50]
    rows = [[str(k), sci(m / 2 ** (k + 1), 3), sci(abs(bis(k) - ps), 3)] for k in ks]
    t1 = _tab([r"k", r"\text{cota }M/2^{k+1}", r"|p_k-p^*_{\mathrm{anal}}|"], rows)
    k_tol = int(np.ceil(np.log2(m / 1e-9)))
    pmax_ex, pmin_ex = m25.lab_momentum_limits_analytic()
    from scipy.optimize import minimize_scalar
    ps0 = m25.p_star_analytic()
    gg = np.hypot(2000.0, m) / m
    bb = 2000.0 / np.hypot(2000.0, m)
    e1_0 = np.hypot(ps0, m1)
    pfun = lambda c: np.sqrt(ps0**2 * (1 - c * c) + (gg * (ps0 * c + bb * e1_0)) ** 2)  # noqa: E731
    pmin_true = float(minimize_scalar(pfun, bounds=(-1, 0), method="bounded", options=dict(xatol=1e-14)).fun)
    ns = [100, 1000, 10000, 100000, 1000000]
    rows2 = []
    ee = []
    for n in ns:
        mu, sd, pmx, pmn = m25.decay_monte_carlo(n=n)
        ee.append(abs(pmx - pmax_ex))
        rows2.append([f"{n}", sci(abs(mu - m), 2), sci(sd, 2), sci(abs(pmx - pmax_ex), 3), sci(abs(pmn - pmin_ex), 3)])
    t2 = _tab([r"N", r"|\bar M-M|", r"\text{desv.}(M)", r"|p_{\max}^{\mathrm{MC}}-p_{\max}|", r"|p_{\min}^{\mathrm{MC}}-p_{\min}|"], rows2)
    sl_ext = _slope(ns, ee)
    # media de la energia: estimador con error sigma/sqrt N
    gamma = np.hypot(2000.0, m) / m
    beta = 2000.0 / np.hypot(2000.0, m)
    e1 = np.hypot(ps, m1)
    mean_ex = gamma * e1
    sig_th = gamma * beta * ps / np.sqrt(3)
    rows3 = []
    for n in (1000, 10000, 100000, 200000):
        es = m25.lab_energy_samples(n=n)
        rows3.append([f"{n}", fx(float(es.mean()), 3), fx(float(sig_th / np.sqrt(n)), 3), fx(abs(float(es.mean()) - mean_ex), 3)])
    t3 = _tab([r"N", r"\bar E_1\ (\mathrm{MeV})", r"\sigma/\sqrt N", r"|\bar E_1-\gamma E^*|"], rows3)
    return dict(
        numeric_intro=[
            r"Hay dos preguntas numéricas distintas. La primera: ¿cuánto vale el impulso $p^*$ de las hijas en el sistema en reposo? Se plantea como la conservación de energía $\sqrt{p^2+m_1^2}+\sqrt{p^2+m_2^2}=M$, una ecuación en $p$ que se resuelve por bisección sin usar la fórmula cerrada de $p^*$. La segunda: ¿qué observa un detector en el laboratorio, donde el padre viaja con impulso $p_{\mathrm{lab}}$? Eso se estudia con Monte Carlo, generando muchos decaimientos con dirección aleatoria.",
            r"La bisección parte de $[0,M]$: la función $f(p)=E_1(p)+E_2(p)-M$ es estrictamente creciente, vale $m_1+m_2-M<0$ en $p=0$ y es positiva en $p=M$, de modo que hay una única raíz. Cada paso conserva la mitad con cambio de signo, así que el error tras $k$ pasos es $\le M/2^{k+1}$ (convergencia lineal, razón $1/2$). El Monte Carlo muestrea $\cos\theta^*\in[-1,1]$ y $\varphi\in[0,2\pi)$ de forma uniforme (decaimiento isótropo), arma $\vec p_1=-\vec p_2$ con módulo $p^*$ y aplica el boost $E'=\gamma(E+\beta p_z)$, $p_z'=\gamma(p_z+\beta E)$.",
            r"El error del Monte Carlo depende de qué se calcula. Una media (como la energía media de la hija) converge como $\sigma/\sqrt N$, orden $N^{-1/2}$. Pero la masa invariante reconstruida es, evento por evento, la misma para todos (la del padre) porque la cinemática es exacta: solo queda el redondeo de coma flotante, sin error estadístico. Y los extremos (impulso máximo y mínimo en el laboratorio) se acercan a su valor exacto más rápido que $N^{-1/2}$, del orden de $1/N$ en promedio, pues dependen del evento más extremo, no del promedio (con fluctuaciones grandes entre corridas).",
            rf"Los parámetros: 200 iteraciones de bisección (bastan {k_tol} para $10^{{-9}}$ MeV) y $N=10^5$ eventos con semilla 5, con $p_{{\mathrm{{lab}}}}=2000$ MeV. El resultado más importante es el control: la masa invariante reconstruida debe reproducir $M=1115.683$ MeV con dispersión nula, lo que verifica que el boost y el sorteo conservan energía e impulso.",
            r"Ambos métodos son independientes de la fórmula $p^*=\sqrt{[M^2-(m_1+m_2)^2][M^2-(m_1-m_2)^2]}/2M$: la bisección usa solo la conservación de energía y el Monte Carlo usa $p^*$ como dato pero comprueba la invariancia relativista de la masa del sistema, algo que ninguna fórmula del bloque analítico calcula.",
        ],
        numeric_code="""M, m1, m2 = 1115.683, 938.272, 139.570          # MeV, c = 1 (Lambda -> p + pi-)
f = lambda p: np.hypot(p, m1) + np.hypot(p, m2) - M   # E1(p) + E2(p) - M, creciente en p
lo, hi = 0.0, M                                  # f(lo) < 0 < f(hi)
for _ in range(200):
    mid = 0.5 * (lo + hi)                        # punto medio
    if f(mid) < 0: lo = mid                      # energia insuficiente: p* esta a la derecha
    else:          hi = mid                      # energia de mas: p* esta a la izquierda
p_estrella = 0.5 * (lo + hi)                     # impulso de las hijas en reposo
rng = np.random.default_rng(5)                   # semilla fija
cost = rng.uniform(-1, 1, N); phi = rng.uniform(0, 2*np.pi, N)   # direccion isotropa
sint = np.sqrt(1 - cost**2)
p1 = p_estrella * np.stack([sint*np.cos(phi), sint*np.sin(phi), cost], axis=1)  # hija 1; hija 2 = -p1
gamma = np.hypot(p_lab, M) / M; beta = p_lab / np.hypot(p_lab, M)   # boost del padre
E1l = gamma * (E1 + beta * p1[:, 2])             # energia de la hija 1 en el laboratorio
E2l = gamma * (E2 - beta * p1[:, 2])             # energia de la hija 2 (su pz es -p1z)
pz  = gamma * (p1[:, 2] + beta*E1) + gamma * (-p1[:, 2] + beta*E2)   # pz total
masa = np.sqrt((E1l + E2l)**2 - pz**2)           # masa invariante: debe salir M para todo evento
""",
        numeric_converge=[
            ("Bisección: error contra iteraciones", rf"Se replica el bucle con $k$ iteraciones y se compara con $p^*={fx(ps, 9)}$ MeV de la fórmula cerrada. El error cae por debajo de la cota $M/2^{{k+1}}$ y se estanca en el redondeo (alrededor de $10^{{-13}}$ MeV, con 15-16 cifras relativas).",
             [t1]),
            ("Monte Carlo: masa invariante y extremos", rf"La masa invariante reconstruida reproduce $M$ con error de redondeo, que no depende de $N$ (no hay error estadístico, porque el valor es el mismo en todo evento). En cambio el impulso máximo y mínimo de la hija se acercan a los valores de las fórmulas con $\cos\theta^*=\pm1$ ($p_{{\max}}={fx(pmax_ex, 3)}$, $p_{{\min}}={fx(pmin_ex, 3)}$ MeV; el mínimo verdadero, hallado minimizando, difiere de esa fórmula en ${sci(abs(pmin_true - pmin_ex), 2)}$ MeV) de forma irregular, porque dependen de la muestra más extrema: la pendiente del ajuste de log error contra log N del máximo es {fx(sl_ext, 2)}. El error del mínimo se queda fijo mientras la muestra más extrema no cambia (el generador es el mismo flujo de números, así que las corridas con más $N$ contienen a las anteriores).",
             [t2]),
            ("Monte Carlo de una media y criterio de confianza", rf"Para el promedio de la energía de la hija en el laboratorio, con $\cos\theta^*$ uniforme, el valor esperado es $\gamma E^*={fx(mean_ex, 3)}$ MeV y la desviación $\sigma=\gamma\beta p^*/\sqrt3={fx(sig_th, 3)}$ MeV. El error observado queda en torno a $\sigma/\sqrt N$ (orden $N^{{-1/2}}$). El resultado se acepta si la masa invariante coincide con $M$ a ~$10^{{-12}}$ relativo y la media está a menos de unas dos desviaciones de $\sigma/\sqrt N$ de su valor exacto.",
             [t3]),
        ],
    )


# ====================================================================== FriedmannAge
def _friedmann() -> dict:
    h0, om, ol = m25.H0_PER_GYR, m25.OMEGA_M, m25.OMEGA_L
    t_ex = m25.age_analytic()
    ns = [4, 8, 16, 32, 64, 128, 256]

    def trap(n):
        u = np.linspace(0.0, 1.0, n + 1)
        a = u**2
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(u > 0, 2 * u / (a * h0 * np.sqrt(om / a**3 + ol)), 0.0)
        g[0] = 0.0
        return float(np.trapezoid(g, u)) if hasattr(np, "trapezoid") else float(np.trapz(g, u))

    es = [abs(m25.age_numeric(n=n) - t_ex) for n in ns]
    et = [abs(trap(n) - t_ex) for n in ns]
    rows = [[str(n), sci(et[i], 3), sci(es[i], 3), fx(es[i - 1] / es[i], 1) if i and es[i] > 0 else "-"] for i, n in enumerate(ns)]
    t1 = _tab([r"N", r"|T_N-t_0|\ (\text{trapecio})", r"|S_N-t_0|\ (\text{Simpson})", r"\text{cociente}_S"], rows)
    # pendiente solo en la zona de truncamiento
    sl_s = _slope(ns[2:], es[2:])
    sl_t = _slope(ns[2:], et[2:])
    t_num = m25.age_numeric()
    # RK4
    n_rk = [25, 50, 100, 200, 400, 800]
    ea = float(m25.scale_factor_analytic(25.0))
    erk = []
    for n in n_rk:
        tt, yy = m25.scale_factor_numeric(25.0, n)
        erk.append(abs(float(yy[-1, 0]) - float(m25.scale_factor_analytic(tt[-1]))))
    rows2 = [[str(n), fx(25.0 / n, 4), sci(e_, 3), fx(erk[i - 1] / e_, 1) if i else "-"] for i, (n, e_) in enumerate(zip(n_rk, erk))]
    t2 = _tab([r"n", r"h\ (\mathrm{Gyr})", r"|a_{\mathrm{num}}(25)-a_{\mathrm{anal}}|", r"\text{cociente}"], rows2)
    sl_rk = _slope([25.0 / n for n in n_rk], erk)
    sl_loc = float(np.log(erk[-2] / erk[-1]) / np.log(n_rk[-1] / n_rk[-2]))
    e5k, e20k = [], []
    for n_, lst in ((5000, e5k), (20000, e20k)):
        tt, yy = m25.scale_factor_numeric(25.0, n_)
        lst.append(abs(float(yy[-1, 0]) - float(m25.scale_factor_analytic(tt[-1]))))
    r_asint = e5k[0] / e20k[0]
    # sensibilidad a H0 y Omega_m
    d = 1e-4
    dt_dh = (m25.age_numeric(h0=h0 * (1 + d)) - m25.age_numeric(h0=h0 * (1 - d))) / (2 * d * h0)
    dom = 0.01
    t_om = m25.age_numeric(om=om + dom, ol=ol - dom)
    return dict(
        numeric_intro=[
            r"La edad del universo es $t_0=\int_0^1\dfrac{\mathrm da}{a\,H(a)}$ con $H(a)=H_0\sqrt{\Omega_m/a^3+\Omega_\Lambda}$. Esta integral sí tiene primitiva (con $\operatorname{arcsinh}$), pero solo para un universo plano con materia y constante cosmológica; con radiación o curvatura deja de ser elemental. La cuadratura numérica no depende de que la primitiva exista, y la integración de $\mathrm da/\mathrm dt$ con RK4 da además toda la historia $a(t)$.",
            r"Para la integral se usa la regla de Simpson compuesta: con $N$ subintervalos pares y $h=(b-a)/N$, $S_N=\tfrac h3\left[f_0+4f_1+2f_2+4f_3+\dots+4f_{N-1}+f_N\right]$. Ajusta parábolas por cada tres nodos y su error es $-\tfrac{(b-a)}{180}h^4f^{(4)}(\xi)$, es decir $O(h^4)$ (el trapecio, en cambio, es $O(h^2)$). Para $a(t)$ se usa Runge-Kutta de cuarto orden, también $O(h^4)$ global (salvo cerca de $t=0$, donde la solución es singular, como se verá en la convergencia).",
            r"El integrando $1/(aH)$ diverge en $a\to0$ como $a^{1/2}$ y Simpson, que supone un integrando suave, perdería su orden. La sustitución $a=u^2$, $\mathrm da=2u\,\mathrm du$ lo convierte en $2u^2/\big(H_0\sqrt{\Omega_m+\Omega_\Lambda u^6}\big)$, finito y suave en $[0,1]$ (vale 0 en $u=0$). Para RK4 el problema análogo es que $\mathrm da/\mathrm dt\to\infty$ en $a=0$; por eso se arranca en $t=10^{-3}$ Gyr con la solución analítica de ese instante.",
            rf"Se eligen $N=20000$ para la integral (mucho más fino de lo necesario, para quedar en el piso del redondeo) y $n=5000$ pasos de RK4 hasta 25 Gyr ($h=5\times10^{{-3}}$ Gyr). Con $H_0=0.0703$ Gyr$^{{-1}}$, $\Omega_m=0.31$ y $\Omega_\Lambda=0.69$ se obtiene $t_0={fx(t_num, 5)}$ Gyr. Hay que advertir que la edad está dominada por la incertidumbre de $H_0$ y no por la numérica.",
            r"Ambos cálculos son independientes del resultado cerrado $t_0=\tfrac{2}{3H_0\sqrt{\Omega_\Lambda}}\operatorname{arcsinh}\sqrt{\Omega_\Lambda/\Omega_m}$ y de la solución $a(t)\propto\sinh^{2/3}$: solo usan la ecuación de Friedmann $H(a)$. Por eso el acuerdo entre las dos rutas (cuadratura y EDO) y la fórmula cerrada es una verificación genuina.",
        ],
        numeric_code="""H0, Om, OL = 0.0703, 0.31, 0.69                  # 1/Gyr y fracciones de densidad (universo plano)
def simpson(y, x):                               # regla compuesta 1-4-2-4-...-4-1
    n = len(x) - 1; h = (x[-1] - x[0]) / n       # n debe ser par
    return h/3 * (y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())
u = np.linspace(0.0, 1.0, 20001)                 # N = 20000 subintervalos en u
a = u**2                                         # cambio a = u^2: elimina la singularidad en a = 0
f = 2*u / (a * H0 * np.sqrt(Om/a**3 + OL))       # da/(a H) = 2u du / (a H); finito en u -> 0
f[0] = 0.0                                       # limite del integrando en u = 0
t0 = simpson(f, u)                               # edad = integral de 0 a 1; error O(h^4)
def dadt(t, y):                                  # ecuacion de Friedmann para a(t)
    return np.array([H0 * np.sqrt(Om/y[0] + OL*y[0]**2)])
t, y, h = 1e-3, np.array([a_inicial]), 25.0/5000 # arranca con la solucion de materia en t = 1e-3 Gyr
for i in range(5000):
    y = rk4_step(dadt, t, y, h); t += h          # RK4: k1..k4 y promedio ponderado; error O(h^4)
# comparacion: t0 cerrado = 2/(3 H0 sqrt(OL)) * arcsinh(sqrt(OL/Om)); a(t) = (Om/OL)^(1/3) sinh^(2/3)(1.5 sqrt(OL) H0 t)
""",
        numeric_converge=[
            ("Integral: error contra N", rf"Se calcula la edad con distintos $N$ y se compara con la fórmula cerrada ($t_0={fx(t_ex, 8)}$ Gyr). Se muestran trapecio y Simpson sobre el mismo integrando: al duplicar $N$ el trapecio mejora $\sim4$ veces y Simpson $\sim16$ veces, hasta llegar al piso del redondeo.",
             [t1, rf"\text{{pendientes (N=16 a 256): trapecio }}{fx(sl_t, 2)},\quad \text{{Simpson }}{fx(sl_s, 2)}\ \ (\text{{teoría: }}-2\ \text{{y}}\ -4)"]),
            ("Valor usado y fuentes de error", rf"Con $N=20000$ el error es ${sci(abs(t_num - t_ex), 2)}$ Gyr, es decir, ya dominado por el redondeo al sumar 20000 términos (${sci(abs(t_num - t_ex) / t_ex, 1)}$ relativo). Extrapolando la ley $h^4$ desde $N=256$, el truncamiento de Simpson a $N=20000$ sería de unos ${sci(es[-1] * (256 / 20000) ** 4, 1)}$ Gyr, muy por debajo del redondeo; seguir refinando no mejora nada.",
             [rf"|t_{{0,\mathrm{{num}}}}-t_{{0,\mathrm{{anal}}}}|={sci(abs(t_num - t_ex), 2)}\ \mathrm{{Gyr}},\qquad t_0={fx(t_num, 8)}\ \mathrm{{Gyr}}"]),
            ("RK4: error contra paso", rf"Se integra hasta 25 Gyr con distintos $n$ y se compara $a(25)$ con la solución analítica. El orden $h^4$ no se ve con pasos grandes: la pendiente ajustada en esta tabla es solo {fx(sl_rk, 2)} (local entre los dos últimos: {fx(sl_loc, 2)}) porque $a\propto t^{{2/3}}$ tiene derivada infinita en $t=0$ y, con $h\gg10^{{-3}}$ Gyr, los primeros pasos no resuelven ese comportamiento. Con $h$ pequeño se recupera el régimen de orden 4: entre $n=5000$ (el valor del código) y $n=20000$ el error baja {fx(r_asint, 0)} veces ($4^4=256$ sería el límite).",
             [t2]),
            ("Sensibilidad física y criterio", rf"El error numérico de la edad (${sci(abs(t_num - t_ex), 1)}$ Gyr) y el de $a(25)$ con $n=5000$ (${sci(e5k[0], 2)}$, relativo ${sci(e5k[0] / ea, 1)}$) son despreciables frente a las incertidumbres físicas: como $t_0\propto1/H_0$, un 1 % de error en $H_0$ cambia la edad en 1 %. Cambiar $\Omega_m$ de {om:g} a {om + dom:g} (planitud conservada) mueve $t_0$ a {fx(t_om, 4)} Gyr. Un resultado es confiable cuando duplicar $N$ (o $n$) no cambia las cifras que se reportan.",
             [rf"\dfrac{{\partial\ln t_0}}{{\partial\ln H_0}}={fx(dt_dh * h0 / t_num, 4)},\qquad t_0(\Omega_m={om + dom:g})={fx(t_om, 4)}\ \mathrm{{Gyr}}"]),
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
