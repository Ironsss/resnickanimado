"""Metodo numerico explicado: Volumen 1, capitulos 1 a 4 (bloque A).

Escenas: DimensionalAnalysis, ErrorPropagation, VectorAddition, VectorProducts,
InstantaneousVelocity, ConstantAcceleration, Projectile, CircularMotion.
Campos: numeric_intro, numeric_code, numeric_converge. Las cifras salen de ejecutar `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch01_measurement as c1
from physics import ch02_vectors as vec
from physics import ch03_motion_1d as m1d
from physics import ch04_motion_2d as c4
from physics.integrators import euler_step, integrate, rk4_step, riemann_left, trapezoid


def _tab(head, rows, align) -> str:
    """Tabla TeX (array) con encabezado y filas ya formateadas como cadenas."""
    return r"\begin{array}{" + align + "}" + " & ".join(head) + r" \\ \hline " + r" \\ ".join(" & ".join(r) for r in rows) + r" \end{array}"


def _slopes(xs, es):
    """Pendiente log-log entre filas sucesivas (primera fila: guion)."""
    out = ["--"]
    for i in range(1, len(xs)):
        out.append(f"{np.log(es[i] / es[i - 1]) / np.log(xs[i] / xs[i - 1]):.2f}")
    return out


def _dec_polar(mod: int, deg: int):
    """(x, y) de un vector polar en aritmetica decimal de 50 cifras (series de Taylor), como referencia exacta."""
    from decimal import Decimal, getcontext
    getcontext().prec = 50
    pi = Decimal("3.14159265358979323846264338327950288419716939937510")
    x = Decimal(deg) * pi / 180
    c = s_ = Decimal(0)
    term_c, term_s = Decimal(1), x
    for k in range(40):
        c += term_c
        s_ += term_s
        term_c *= -x * x / ((2 * k + 1) * (2 * k + 2))
        term_s *= -x * x / ((2 * k + 2) * (2 * k + 3))
    return mod * c, mod * s_


# ====================================================================== cap. 1
def _dimensional() -> dict:
    a = np.column_stack([c1.DIM["L"], c1.DIM["g"], c1.DIM["m"]])
    d = c1.DIM["T"]
    cond = float(np.linalg.cond(a))
    sv = np.linalg.svd(a, compute_uv=False)
    x = np.array(list(c1.pendulum_exponents().values()))
    res = float(np.linalg.norm(a @ x - d))
    # sensibilidad: se perturba d en 1e-6 y se mide el cambio relativo de la solucion
    dp = d + np.array([1e-6, -1e-6, 1e-6])
    xp, *_ = np.linalg.lstsq(a, dp, rcond=None)
    rel_in = np.linalg.norm(dp - d) / np.linalg.norm(d)
    rel_out = np.linalg.norm(xp - x) / np.linalg.norm(x)
    # convergencia RK4 + interpolacion lineal del cruce
    ns = [50, 100, 200, 400, 800]
    ref = c1.pendulum_period_numeric(0.01, n=25600)
    es = [abs(c1.pendulum_period_numeric(0.01, n=n) - ref) for n in ns]
    tmax = 2 * np.pi * np.sqrt(1 / 9.81)
    # pendiente respecto de h = tmax/n (h se reduce al aumentar n)
    sl = ["--"] + [f"{np.log(es[i] / es[i - 1]) / np.log(ns[i - 1] / ns[i]):.2f}" for i in range(1, len(ns))]
    t_tab = _tab([r"n", r"h\ \mathrm{[s]}", r"|T_n-T_{ref}|\ \mathrm{[s]}", r"\text{pendiente}"],
                 [[str(n), fx(tmax / n, 5), sci(e, 3), s] for n, e, s in zip(ns, es, sl)], "rrrr")
    # amplitud finita
    amps = [0.01, 0.1, 0.3, 0.6]
    k_rows = []
    for th in amps:
        k = c1.pendulum_period_numeric(th, n=4000) / np.sqrt(1 / 9.81)
        k_rows.append([str(th), fx(k, 6), sci(k / (2 * np.pi) - 1, 3), sci(th**2 / 16, 3)])
    k_tab = _tab([r"\theta_0\ \mathrm{[rad]}", r"k_{num}", r"k_{num}/2\pi-1", r"\theta_0^2/16"], k_rows, "rrrr")
    return dict(
        numeric_intro=[
            "El análisis dimensional exige que el periodo $T=L^{a}g^{b}m^{c}$ tenga las dimensiones de un tiempo. Igualar los exponentes de $\\mathsf M$, $\\mathsf L$ y $\\mathsf T$ produce un sistema lineal $A\\vec x=\\vec d$ de $3\\times3$ cuya solución son los exponentes. Aquí no se resuelve por eliminación a mano sino con álgebra lineal numérica, porque es la forma que escala a problemas con muchas magnitudes donde el sistema ya no es cuadrado ni invertible.",
            "El código usa mínimos cuadrados con la descomposición en valores singulares, $A=U\\Sigma V^{T}$. La solución es $\\vec x=V\\Sigma^{+}U^{T}\\vec d$, donde $\\Sigma^{+}$ invierte los valores singulares no nulos. Si el sistema tiene solución exacta, minimiza $\\|A\\vec x-\\vec d\\|$ y da residuo del orden de la precisión de máquina; si es singular elige la solución de menor norma. El error de la solución está acotado por $\\kappa(A)\\,\\epsilon_{m}$, con $\\kappa=\\sigma_{\\max}/\\sigma_{\\min}$ el número de condición.",
            "Para fijar la constante $k$ de $T=k\\sqrt{L/g}$ el código integra la ecuación del péndulo, $\\ddot\\theta=-(g/L)\\sin\\theta$, con Runge-Kutta de cuarto orden. RK4 combina cuatro pendientes por paso, $y_{n+1}=y_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$, y su error global es $O(h^4)$. El periodo se obtiene del cruce por cero de $\\theta$ (un cuarto de periodo) interpolado linealmente entre dos pasos. Esa interpolación sería $O(h^2)$ en general, pero justo en el cruce $\\ddot\\theta=0$, así que no degrada el orden y la convergencia observada es de cuarto orden.",
            "Los parámetros se eligieron así: amplitud pequeña $\\theta_0=0.01$ rad, para que $\\sin\\theta\\approx\\theta$ y el resultado deba ser $2\\pi$; y $n=4000$ pasos hasta una cota $t_{\\max}=2\\pi\\sqrt{L/g}$, con lo que el error de discretización queda por debajo de la corrección física de amplitud finita, $\\theta_0^2/16$. El resultado numérico $k$ debe aproximar $2\\pi$ sin que el código conozca ese valor.",
            "El método es independiente del analítico porque el exponente sale de un sistema lineal con las dimensiones de las magnitudes y la constante $k$ sale de integrar la dinámica, sin usar la fórmula $T=2\\pi\\sqrt{L/g}$. Falla cuando faltan magnitudes (el sistema queda subdeterminado y hay una familia de soluciones, la constante adimensional queda libre) o cuando hay magnitudes dimensionalmente dependientes.",
        ],
        numeric_code="""A = columnas(DIM[L], DIM[g], DIM[m])      # filas = M, L, T; columnas = L, g, m
d = DIM[T]                                # el periodo tiene dimension T^1
x = lstsq(A, d)                           # minimos cuadrados por SVD: x = V S+ U^T d
residuo = norma(A @ x - d)                # ~1e-16 si el sistema es consistente

def periodo(theta0, L=1.0, g=9.81, n=4000):
    f = lambda t, y: [y[1], -(g/L)*sin(y[0])]   # theta' = w, w' = -(g/L) sin(theta)
    tmax = 2*pi*sqrt(L/g)                       # cota holgada del cuarto de periodo
    t, y = integrar_rk4(f, [theta0, 0], 0, tmax, n)   # n pasos de tamano h = tmax/n
    i = primer indice con y[i,0] <= 0           # primer cruce por cero de theta
    tc = t[i-1] + (0 - y[i-1,0])*(t[i]-t[i-1])/(y[i,0]-y[i-1,0])   # interpolacion lineal
    return 4*tc                                 # cuarto de periodo -> periodo

k = periodo(0.01) / (L**x[L] * g**x[g])   # constante adimensional; esperado 2*pi
""",
        numeric_converge=[
            ("Condicionamiento y residuo del sistema lineal", "El sistema dimensional es pequeño y está bien condicionado: sus valores singulares no son muy distintos, así que los errores de redondeo casi no se amplifican. Al perturbar el vector $\\vec d$ en una parte en $10^{6}$ la solución cambia en proporción parecida.",
             [rf"\sigma(A) = ({fx(sv[0], 4)},\ {fx(sv[1], 4)},\ {fx(sv[2], 4)}),\qquad \kappa(A) = \dfrac{{\sigma_{{\max}}}}{{\sigma_{{\min}}}} = {fx(cond, 4)}",
              rf"\|A\vec x-\vec d\| = {sci(res, 2)},\qquad \dfrac{{\|\delta\vec d\|}}{{\|\vec d\|}} = {sci(rel_in, 2)}\ \Rightarrow\ \dfrac{{\|\delta\vec x\|}}{{\|\vec x\|}} = {sci(rel_out, 2)}"]),
            ("Error de discretización de RK4 con paso decreciente", "Se calcula el periodo con $n$ pasos y se compara con un cálculo de referencia de $n=25\\,600$. La pendiente se mide respecto de $h$ entre filas sucesivas y resulta cercana a 4, el orden de RK4: la interpolación lineal del cruce no la degrada porque en el cruce $\\ddot\\theta=-(g/L)\\sin\\theta=0$ y la recta es exacta hasta segundo orden. La última pendiente baja porque el error se acerca al de la propia referencia.",
             [t_tab,
              rf"T_{{ref}} = {fx(ref, 9)}\ \mathrm{{s}},\qquad T_{{teor}} = 2\pi\sqrt{{L/g}} = {fx(2 * np.pi * np.sqrt(1 / 9.81), 9)}\ \mathrm{{s}}"]),
            ("Error físico: la amplitud finita", "Aun con $h\\to0$ queda una diferencia con $2\\pi$ que no es error del método: el péndulo real con amplitud $\\theta_0$ tiene $T\\approx T_0(1+\\theta_0^2/16)$. La tabla muestra que $k_{num}/2\\pi-1$ sigue esa ley, así que con $\\theta_0=0.01$ rad la discrepancia $6\\times10^{-6}$ es física, no numérica.",
             [k_tab,
              rf"\text{{con }}n=4000:\ \ \dfrac{{|T_n-T_{{ref}}|}}{{T_{{ref}}}} = {sci(abs(c1.pendulum_period_numeric(0.01, n=4000) - ref) / ref, 2)}\ \ll\ \dfrac{{\theta_0^2}}{{16}} = {sci(0.01**2 / 16, 2)}"]),
            ("Fuentes de error y criterio de confianza", "Hay tres: el redondeo del álgebra lineal (residuo del orden $10^{-16}$), la discretización temporal (decrece como $h^4$, el orden de RK4) y la amplitud finita (física, $\\theta_0^2/16$). El resultado es confiable cuando el residuo es del orden de $\\epsilon_m\\kappa$ y al duplicar $n$ el periodo cambia menos que la corrección de amplitud.",
             [rf"\epsilon_m\,\kappa \approx {sci(np.finfo(float).eps * cond, 2)},\qquad \Delta T(n\to2n) = {sci(abs(c1.pendulum_period_numeric(0.01, n=4000) - c1.pendulum_period_numeric(0.01, n=8000)), 2)}\ \mathrm{{s}}"]),
        ],
    )


def _error_prop() -> dict:
    an = c1.density_analytic()
    rho0 = an["rho"]
    ns = [100, 1000, 10_000, 100_000]
    seeds = range(40)
    rows, stds = [], []
    for n in ns:
        means = np.array([c1.density_monte_carlo(n, seed=s)["mean"] for s in seeds])
        sds = np.array([c1.density_monte_carlo(n, seed=s)["std"] for s in seeds])
        stds.append(means.std(ddof=1))
        rows.append([f"{n:,}".replace(",", r"\,"), sci(means.std(ddof=1), 3), sci(an["sigma_quadrature"] / np.sqrt(n), 3), sci(sds.std(ddof=1), 3)])
    sl = _slopes(ns, stds)
    for r_, s in zip(rows, sl):
        r_.insert(3, s)
    tab = _tab([r"N", r"\text{dispersión de }\bar\rho", r"\sigma_\rho/\sqrt N", r"\text{pendiente}", r"\text{dispersión de }s"], rows, "rrrrr")
    big = c1.density_monte_carlo(400_000, seed=7)
    bias = big["mean"] - rho0
    se = big["std"] / np.sqrt(400_000)
    return dict(
        numeric_intro=[
            "Cuando una cantidad se calcula a partir de varias mediciones con incertidumbre, la propagación analítica de primer orden linealiza la fórmula alrededor de los valores medios. Monte Carlo evita esa linealización: cada medición se trata como una variable aleatoria, se sortean muchas realizaciones, se evalúa la fórmula completa en cada una y la dispersión de los resultados es la incertidumbre. Se necesita aquí porque $\\rho=m/(\\pi r^2h)$ es no lineal en $r$ y conviene comprobar que la linealización es suficiente.",
            "El estimador es el promedio muestral $\\bar\\rho=\\tfrac1N\\sum\\rho_i$ y la desviación $s=\\sqrt{\\sum(\\rho_i-\\bar\\rho)^2/(N-1)}$. Por la ley de los grandes números $\\bar\\rho\\to E[\\rho]$, y por el teorema central del límite el error estadístico del promedio es $s/\\sqrt N$. Es decir, el error de Monte Carlo es $O(N^{-1/2})$: para ganar una cifra decimal hay que multiplicar $N$ por 100.",
            "Las limitaciones son dos. La convergencia es lenta y no depende de la dimensión del problema (por eso es útil con muchas variables), y el resultado depende de que las distribuciones de entrada sean las correctas: aquí se supone normales independientes. El método también incluye el sesgo por no linealidad, $E[\\rho]-\\rho(\\bar m,\\bar r,\\bar h)$, que la fórmula de primer orden ignora.",
            "Se escogió $N=400\\,000$ para que el error estadístico del promedio, $s/\\sqrt N$, sea de solo unas milésimas de la incertidumbre buscada, y se fijó la semilla 7 para que el resultado sea reproducible. El resultado $s\\approx0.0227$ g/cm$^3$ es la incertidumbre de la densidad y debe coincidir con la propagación en cuadratura sin haberla usado.",
            "El método es independiente del analítico: nunca usa derivadas parciales ni la suma en cuadratura. Solo evalúa $\\rho(m,r,h)$ en puntos sorteados, de modo que si ambos coinciden, la linealización y el sorteo se validan mutuamente.",
        ],
        numeric_code="""rng = generador_aleatorio(semilla=7)               # semilla fija: resultado reproducible
N = 400_000                                        # numero de realizaciones
m = rng.normal(111.4, 0.1,   N)                    # masa [g] ~ Normal(media, sigma)
r = rng.normal(1.250, 0.005, N)                    # radio [cm]
h = rng.normal(8.40,  0.02,  N)                    # altura [cm]
rho = m / (pi * r**2 * h)                          # formula completa en cada terna (sin linealizar)
media = rho.mean()                                 # estimador de la densidad
sigma = rho.std(ddof=1)                            # incertidumbre: desviacion muestral (N-1)
err_media = sigma / sqrt(N)                        # error estadistico del promedio ~ N^(-1/2)

# comparacion con la propagacion de primer orden (solo como verificacion)
rel = [0.1/111.4, 2*0.005/1.250, 0.02/8.40]        # contribuciones relativas
sigma_cuad = rho_nominal * sqrt(sum(x**2 for x in rel))
""",
        numeric_converge=[
            ("Error estadístico frente a N", "Para cada $N$ se repite el experimento con 40 semillas distintas y se mide la dispersión del promedio $\\bar\\rho$. Teóricamente es $\\sigma_\\rho/\\sqrt N$; la pendiente en escala log-log debe ser $-0.5$.",
             [tab]),
            ("Resultado final y sesgo por no linealidad", "Con $N=400\\,000$ el promedio difiere del valor nominal en unas pocas veces el error estadístico: es el sesgo por no linealidad, que para $1/r^2$ vale aproximadamente $3(\\sigma_r/r)^2\\rho$ y es mucho menor que $\\sigma_\\rho$. La desviación coincide con la propagación en cuadratura.",
             [rf"\bar\rho-\rho_{{nom}} = {sci(bias, 2)}\ \mathrm{{g/cm^3}},\qquad \dfrac{{s}}{{\sqrt N}} = {sci(se, 2)}\ \mathrm{{g/cm^3}}",
              rf"3(\sigma_r/r)^2\rho = {sci(3 * (0.005 / 1.25) ** 2 * rho0, 2)}\ \mathrm{{g/cm^3}}",
              rf"s = {fx(big['std'], 5)},\qquad \sigma_{{cuad}} = {fx(an['sigma_quadrature'], 5)},\qquad \dfrac{{s-\sigma_{{cuad}}}}{{\sigma_{{cuad}}}} = {sci((big['std'] - an['sigma_quadrature']) / an['sigma_quadrature'], 2)}"]),
            ("Fuentes de error y criterio de confianza", "El error es solo estadístico (no hay discretización) más el error de modelo: supone entradas normales e independientes. El redondeo de coma flotante es despreciable. El resultado es confiable cuando $s/\\sqrt N$ es mucho menor que la precisión con que se quiere dar $\\sigma_\\rho$ y cambiar de semilla no altera las cifras reportadas.",
             [rf"\dfrac{{s/\sqrt N}}{{s}} = \dfrac{{1}}{{\sqrt N}} = {sci(1 / np.sqrt(400_000), 2)}\qquad\text{{(error relativo de }}\bar\rho\text{{ respecto de }}s\text{{)}}"]),
        ],
    )


# ====================================================================== cap. 2
def _vec_add() -> dict:
    rn = vec.resultant_numeric()
    ra = vec.resultant_analytic()
    # referencia en precision extendida
    (x1, y1), (x2, y2) = _dec_polar(40, 30), _dec_polar(25, 120)
    rx, ry = x1 + x2, y1 + y2
    rmag_ld = (rx * rx + ry * ry).sqrt()
    cases = [(40, 25, 30.0, 120.0), (40, 25, 30.0, 30.0), (40, 25, 30.0, 209.0), (25, 40, 30.0, 180.0), (25, 40, 30.0, 135.0)]
    rows = []
    for a, b, t1, t2 in cases:
        r = vec.resultant_components((vec.polar(a, t1), vec.polar(b, t2)))
        mag_n, ang_n = float(np.hypot(*r)), float(np.rad2deg(np.arctan2(r[1], r[0])))
        g = np.deg2rad(t2 - t1)
        mag_a = float(np.sqrt(a * a + b * b + 2 * a * b * np.cos(g)))
        ang_s = float(np.rad2deg(np.arcsin(np.clip(b * np.sin(g) / mag_a, -1, 1))) + t1) if mag_a > 1e-9 else float("nan")
        rows.append([f"{a},{b}", f"{t2 - t1:.0f}^\\circ", sci(abs(mag_n - mag_a), 2), fx(ang_n, 3), fx(ang_s, 3)])
    tab = _tab([r"a,b", r"\gamma", r"|R_{comp}-R_{cos}|", r"\theta_{atan2}\ [^\circ]", r"\theta_{arcsin}\ [^\circ]"], rows, "rrrrr")
    rng = np.random.default_rng(3)
    mx = 0.0
    for _ in range(2000):
        a_, b_ = rng.uniform(1, 50, 2)
        t1, t2 = rng.uniform(0, 360, 2)
        r = vec.resultant_components((vec.polar(a_, t1), vec.polar(b_, t2)))
        g = np.deg2rad(t2 - t1)
        m_a = np.sqrt(a_**2 + b_**2 + 2 * a_ * b_ * np.cos(g))
        mx = max(mx, abs(np.hypot(*r) - m_a) / m_a)
    return dict(
        numeric_intro=[
            "Sumar vectores dados por módulo y ángulo se puede hacer con las leyes de los cosenos y de los senos, pero esas leyes cambian de forma según el cuadrante y el número de vectores. El método numérico usa la representación en componentes: cada vector se proyecta sobre los ejes, las componentes se suman como números y al final se reconstruyen módulo y ángulo. No es un método iterativo sino un algoritmo directo.",
            "La regla es $R_x=\\sum A_i\\cos\\theta_i$, $R_y=\\sum A_i\\sin\\theta_i$, seguida de $R=\\sqrt{R_x^2+R_y^2}$ y $\\theta_R=\\mathrm{atan2}(R_y,R_x)$. Es correcta porque la suma vectorial es lineal y las componentes sobre ejes ortogonales son independientes. No hay error de truncamiento: el resultado es exacto salvo el redondeo de coma flotante, que es del orden de $\\epsilon_m\\approx2\\times10^{-16}$ relativo en cada operación.",
            "La limitación principal no es el error sino el cuadrante: con $\\arctan(R_y/R_x)$ se pierde el signo de $R_x$, y con $\\arcsin$ (ley de los senos) el ángulo queda ambiguo cuando el segundo vector es mucho mayor que el primero. Por eso el código usa atan2, que devuelve el ángulo en el cuadrante correcto, y hypot, que evita desbordamientos intermedios.",
            "Aquí no hay parámetros que sintonizar (ni paso, ni $N$, ni semilla): los datos son $40$ m a $30^\\circ$ y $25$ m a $120^\\circ$. El resultado es el desplazamiento neto, y su error se estudia comparando con la ley de los cosenos en muchos casos y mediante precisión extendida.",
            "Es independiente del analítico porque no usa el ángulo entre vectores ni la ley de los cosenos; solo seno, coseno y suma de números. La coincidencia entre ambos caminos es la verificación cruzada de la escena.",
        ],
        numeric_code="""def polar_a_xy(mod, ang_grados):
    a = radianes(ang_grados)                  # conversion: grados -> radianes
    return mod * [cos(a), sin(a)]             # componentes x, y

D1 = polar_a_xy(40, 30)                       # primer desplazamiento
D2 = polar_a_xy(25, 120)                      # segundo desplazamiento
R  = D1 + D2                                  # suma componente a componente (lineal, exacta)
mod = hipot(R[0], R[1])                       # sqrt(Rx^2 + Ry^2) sin desbordamiento
ang = grados(atan2(R[1], R[0]))               # angulo en el cuadrante correcto
# verificacion cruzada con la ley de los cosenos (solo para comparar)
gamma = radianes(120 - 30)
mod_cos = sqrt(40**2 + 25**2 + 2*40*25*cos(gamma))   # modulo por cosenos
""",
        numeric_converge=[
            ("Error de redondeo frente a una referencia decimal", "Como no hay parámetro de discretización, se mide el error con aritmética decimal de 50 cifras (series de Taylor para seno y coseno). El módulo de doble precisión coincide hasta el redondeo.",
             [rf"R_{{double}} = {fx(rn['magnitude'], 12)}\ \mathrm{{m}},\qquad R_{{ref}} = {float(rmag_ld):.12f}\ \mathrm{{m}}",
              rf"|R_{{double}}-R_{{ref}}| = {sci(abs(rn['magnitude'] - float(rmag_ld)), 2)}\ \mathrm{{m}},\qquad |R_{{comp}}-R_{{cos}}| = {sci(abs(rn['magnitude'] - ra['magnitude']), 2)}\ \mathrm{{m}}"]),
            ("Verificación cruzada en varios casos y ángulos", "Se compara con la ley de los cosenos. El módulo coincide siempre al redondeo. El ángulo por arcsin, en cambio, falla en el caso $a=25$, $b=40$, $\\gamma=150^\\circ$ (el segundo vector es mayor y apunta hacia atrás: el seno es el mismo para $\\delta$ y $180^\\circ-\\delta$): atan2 acierta.",
             [tab]),
            ("Barrido aleatorio y criterio de confianza", "Con 2000 pares aleatorios (módulos de 1 a 50 m, ángulos de 0 a $360^\\circ$) el mayor error relativo del módulo frente a la ley de los cosenos es del orden de la precisión de máquina. El resultado es confiable mientras no se restan cantidades casi iguales: si $\\vec D_1\\approx-\\vec D_2$ la resultante pierde cifras significativas por cancelación.",
             [rf"\max\dfrac{{|R_{{comp}}-R_{{cos}}|}}{{R_{{cos}}}} = {sci(mx, 2)},\qquad \epsilon_m = {sci(np.finfo(float).eps, 2)}"]),
        ],
    )


def _vec_prod() -> dict:
    area_true = abs(vec.cross_z())
    ns = [100, 1000, 10_000, 100_000]
    rows, sd = [], []
    for n in ns:
        ests = np.array([vec.parallelogram_area_monte_carlo(n=n, seed=s)[0] for s in range(60)])
        sd.append(float(ests.std(ddof=1)))
        rows.append([f"{n:,}".replace(",", r"\,"), fx(float(ests.mean()), 4), sci(float(ests.std(ddof=1)), 3), sci(24 * np.sqrt(area_true / 24 * (1 - area_true / 24) / n), 3)])
    sl = _slopes(ns, sd)
    for r_, s in zip(rows, sl):
        r_.append(s)
    tab = _tab([r"N", r"\overline{\text{Área}}", r"\text{dispersión}", r"24\sqrt{p(1-p)/N}", r"\text{pendiente}"], rows, "rrrrr")
    area, pts, inside = vec.parallelogram_area_monte_carlo()
    p = area_true / 24
    # condicionamiento del sistema 2x2
    m = np.column_stack([vec.A, vec.B])
    cond = float(np.linalg.cond(m))
    return dict(
        numeric_intro=[
            "El área del paralelogramo es $|\\vec A\\times\\vec B|$, pero el código la estima sin usar el producto vectorial: encierra la figura en una caja y cuenta la fracción de puntos aleatorios que caen dentro. Esa es la integración de Monte Carlo \"de acierto o fallo\" (hit or miss) y sirve para áreas de formas complicadas donde no hay fórmula, por eso se usa aquí como prueba independiente de que el producto cruz mide un área.",
            "El estimador es $\\widehat{A}=A_{caja}\\,N_{dentro}/N$. Cada punto cae dentro con probabilidad $p=A/A_{caja}$, así que $N_{dentro}$ es binomial y el error estándar es $A_{caja}\\sqrt{p(1-p)/N}$, es decir, $O(N^{-1/2})$. Para decidir si un punto está dentro se resuelve $\\vec p=s\\vec A+t\\vec B$ y se exige $0\\le s,t\\le1$.",
            "Las limitaciones son la lenta convergencia (cada cifra decimal cuesta 100 veces más puntos) y que el resultado es aleatorio: depende de la semilla. Si la caja es mucho mayor que la figura, $p$ es pequeña y el error relativo crece como $1/\\sqrt{Np}$.",
            "Se eligió $N=400\\,000$ y semilla 11. La caja es $[0,6]\\times[0,4]$ de área 24, con $p\\approx0.42$, así que el error estadístico esperado es de unas milésimas de unidad de área. El resultado es una estimación del área $|A_xB_y-A_yB_x|=10$.",
            "No reutiliza la fórmula del determinante: usa solo la geometría de la pertenencia y el conteo. La coincidencia con 10 dentro del error estadístico es una verificación independiente del producto cruz.",
        ],
        numeric_code="""rng = generador(semilla=11)                         # semilla fija
vertices = [(0,0), A, B, A+B]                       # esquinas del paralelogramo
lo, hi = minimo(vertices), maximo(vertices)         # caja envolvente alineada con los ejes
puntos = rng.uniforme(lo, hi, tamano=(N, 2))        # N puntos uniformes dentro de la caja
M = matriz_columnas(A, B)                           # p = s*A + t*B  ->  M [s,t]^T = p
st = resolver(M, puntos.T).T                        # (s, t) de cada punto (sistema 2x2)
dentro = todos(0 <= st <= 1, eje=1)                 # cae dentro si ambos coef. estan en [0,1]
area_caja = producto(hi - lo)                       # 6 x 4 = 24
area = area_caja * dentro.mean()                    # fraccion dentro * area de la caja
# error estandar esperado (solo informativo)
err = area_caja * sqrt(p*(1-p)/N)                   # p = area/area_caja
""",
        numeric_converge=[
            ("Error frente a N con muchas semillas", "Para cada $N$ se repite el experimento con 60 semillas. La dispersión de las estimaciones coincide con $A_{caja}\\sqrt{p(1-p)/N}$ y su pendiente es $-0.5$ en escala log-log.",
             [tab]),
            ("Resultado con N = 400 000", "La estimación con la semilla 11 queda dentro de una desviación estándar teórica del valor exacto 10, de acuerdo con el error binomial.",
             [rf"\widehat A = {fx(area, 4)},\qquad |\widehat A-10| = {sci(abs(area - area_true), 2)},\qquad 24\sqrt{{p(1-p)/N}} = {sci(24 * np.sqrt(p * (1 - p) / 400_000), 2)}"]),
            ("Fuentes de error y criterio de confianza", "Hay error estadístico (dominante) y redondeo despreciable al resolver el sistema $2\\times2$, que está bien condicionado. Los puntos en el borde se clasifican por la desigualdad no estricta, un efecto de medida cero. El resultado es confiable si el error estándar teórico es menor que la tolerancia deseada y varias semillas dan valores compatibles.",
             [rf"\kappa\begin{{pmatrix}}4&2\\1&3\end{{pmatrix}} = {fx(cond, 3)},\qquad \dfrac{{24\sqrt{{p(1-p)/N}}}}{{A}} = {sci(24 * np.sqrt(p * (1 - p) / 400_000) / area_true, 2)}"]),
        ],
    )


# ====================================================================== cap. 3
def _inst_vel() -> dict:
    t0 = 2.0
    ve = float(m1d.v_analytic(t0))
    dts = [1.0, 0.1, 0.01, 0.001]
    errs = [abs(m1d.secant_slope(t0, dt) - ve) for dt in dts]
    sl = _slopes([1 / d for d in dts], errs)
    sl = ["--"] + [f"{np.log(errs[i] / errs[i - 1]) / np.log(dts[i] / dts[i - 1]):.2f}" for i in range(1, len(dts))]
    rows = [[sci(dt, 1) if dt < 1 else "1", fx(m1d.secant_slope(t0, dt), 6), sci(e, 3), s] for dt, e, s in zip(dts, errs, sl)]
    t1 = _tab([r"\Delta t\ \mathrm{[s]}", r"\text{pendiente secante}", r"\text{error}", r"\text{orden}"], rows, "rrrr")
    # roundoff
    hs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    r2 = [[sci(h, 1), sci(abs(m1d.secant_slope(t0, h) - ve), 3)] for h in hs]
    t2 = _tab([r"\Delta t\ \mathrm{[s]}", r"\text{error de la secante}"], r2, "rr")
    a_ex = float(m1d.a_analytic(t0))
    hh = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    r3 = [[sci(h, 1), sci(abs(m1d.central_second_difference(t0, h) - a_ex), 3)] for h in hh]
    t3 = _tab([r"h\ \mathrm{[s]}", r"|a_{fd}-a_{exacta}|"], r3, "rr")
    nn = [4, 8, 16, 32, 64]
    er, et = [], []
    for n in nn:
        d = m1d.displacement_from_velocity(0.0, 4.0, n)
        er.append(abs(d["riemann"] - d["exact"]))
        et.append(abs(d["trapezoid"] - d["exact"]))
    sr = ["--"] + [f"{np.log(er[i] / er[i - 1]) / np.log(nn[i - 1] / nn[i]):.2f}" for i in range(1, len(nn))]
    st_ = ["--"] + [f"{np.log(et[i] / et[i - 1]) / np.log(nn[i - 1] / nn[i]):.2f}" for i in range(1, len(nn))]
    t4 = _tab([r"n", r"\text{err. Riemann}", r"\text{orden}", r"\text{err. trapecio}", r"\text{orden}"],
              [[str(n), sci(a, 3), s1, sci(b, 3), s2] for n, a, s1, b, s2 in zip(nn, er, sr, et, st_)], "rrrrr")
    hopt = float(np.sqrt(np.finfo(float).eps))
    return dict(
        numeric_intro=[
            "La velocidad instantánea es el límite de la pendiente de la secante cuando el intervalo tiende a cero. Con una computadora no se puede tomar el límite: se evalúa el cociente con $\\Delta t$ pequeño pero finito y se estudia cómo se acerca al valor verdadero. Es el mismo problema de derivar numéricamente un dato, cuando solo se conocen valores de $x(t)$ y no su fórmula derivada.",
            "Con el desarrollo de Taylor, $x(t+\\Delta t)=x+v\\Delta t+\\tfrac12a\\Delta t^2+\\dots$, la secante hacia adelante vale $v+\\tfrac12a\\Delta t+O(\\Delta t^2)$: el error es de primer orden, $O(\\Delta t)$, y proporcional a la aceleración. La segunda diferencia centrada, $a\\approx[x(t+h)-2x(t)+x(t-h)]/h^2$, cancela los términos impares y tiene error $O(h^2)$, exacto para polinomios de grado 2 (el error depende de la cuarta derivada, nula aquí salvo redondeo).",
            "El área bajo $v(t)$, el camino inverso, se aproxima con trapecios, $\\Delta x\\approx h[\\tfrac12v_0+v_1+\\dots+\\tfrac12v_n]$, con error global $O(h^2)$, frente a $O(h)$ de la suma de Riemann por la izquierda. La limitación común de las diferencias es el redondeo: restar números casi iguales y dividir por $h$ o $h^2$ amplifica el error, así que hay un $h$ óptimo; no sirve hacerlo arbitrariamente pequeño.",
            "Los parámetros: se evalúa en $t_0=2$ s con $\\Delta t$ de 1 a 0.001 s para ver el límite, $h=0.01$ s en la segunda diferencia (equilibrio entre truncamiento y redondeo) y $n=4$ trapecios en la versión reducida a mano. El resultado es $v(2)\\approx2.2$ m/s y $a(2)\\approx0.2$ m/s$^2$, que son derivadas de $x(t)$ obtenidas sin derivar la fórmula.",
            "Es independiente del analítico porque solo evalúa $x(t)$ en puntos; la función $v(t)=3-t+0.3t^2$ se usa únicamente para medir el error. Las tres técnicas se verifican entre sí: derivar la posición, integrar la velocidad y volver a $\\Delta x$.",
        ],
        numeric_code="""def x(t):                                       # posicion [m] en funcion del tiempo [s]
    return 2 + 3*t - 0.5*t**2 + 0.1*t**3

def secante(t0, dt):                            # velocidad media en [t0, t0+dt]
    return (x(t0 + dt) - x(t0)) / dt            # error O(dt): ~ a*dt/2

for dt in (1, 0.5, 0.1, 0.01, 0.001):           # se acerca el limite dt -> 0
    v = secante(2.0, dt)
    error = abs(v - 2.2)                        # solo para medir (2.2 = v(2) exacta)

def segunda_dif(t0, h):                         # aceleracion con tres puntos simetricos
    return (x(t0 + h) - 2*x(t0) + x(t0 - h)) / h**2    # error O(h^2)

a = segunda_dif(2.0, 0.01)

t = linspace(0, 4, n + 1)                       # n trapecios de ancho h
dx = suma(0.5 * (v(t[1:]) + v(t[:-1])) * diff(t))  # area bajo v(t): O(h^2)
""",
        numeric_converge=[
            ("Secante hacia adelante: orden 1", "Se reduce $\\Delta t$ en factores de 10 y se mide el error. La pendiente log-log mide el orden y debe valer 1. El error observado es aproximadamente $a\\Delta t/2$ con $a(2)=0.2$ m/s$^2$, más un término $x'''\\Delta t^2/6=0.1\\Delta t^2$ que explica el 0.011 de la segunda fila.",
             [t1, rf"\dfrac{{a\,\Delta t}}{{2}}\Big|_{{\Delta t=0.1}} = {sci(a_ex * 0.1 / 2, 3)}\ \mathrm{{m/s}}"]),
            ("Redondeo: por qué no basta con achicar $\\Delta t$", "Más allá de cierto punto el error vuelve a crecer porque $x(t+\\Delta t)-x(t)$ pierde cifras. El mínimo está cerca de $\\sqrt{\\epsilon_m}\\approx10^{-8}$ s; para $10^{-12}$ s el cociente ya es inútil.",
             [t2, rf"\sqrt{{\epsilon_m}} = {sci(hopt, 2)}"]),
            ("Segunda diferencia centrada y área bajo $v(t)$", "En la segunda diferencia el truncamiento es nulo para este polinomio (grado 3 con cuarta derivada cero) y todo el error es redondeo, que crece como $\\epsilon_m/h^2$. En las áreas, la suma de Riemann tiene orden 1 y el trapecio orden 2, medidos sobre $[0,4]$ s.",
             [t3, t4]),
            ("Fuentes de error y criterio de confianza", "El truncamiento domina con $\\Delta t$ grande y el redondeo con $\\Delta t$ muy pequeño. Es confiable un valor cuando cambia poco al pasar de $\\Delta t$ a $\\Delta t/2$ (la pendiente log-log ya es la teórica) y está lejos de la zona de redondeo.",
             [rf"v_{{sec}}(0.01) = {fx(m1d.secant_slope(t0, 0.01), 6)},\quad v_{{sec}}(0.001) = {fx(m1d.secant_slope(t0, 0.001), 6)},\quad a_{{fd}}(0.01) = {fx(m1d.central_second_difference(t0, 0.01), 5)}"]),
        ],
    )


def _const_acc() -> dict:
    ns = [10, 20, 40, 80, 160]
    rows = m1d.free_fall_convergence(ns=tuple(ns))
    ee = [r["err_euler"] for r in rows]
    se = ["--"] + [f"{np.log(ee[i] / ee[i - 1]) / np.log(ns[i - 1] / ns[i]):.2f}" for i in range(1, len(ns))]
    t1 = _tab([r"n", r"h\ \mathrm{[s]}", r"\text{err. Euler [m]}", r"\text{orden}", r"\text{err. RK4 [m]}"],
              [[str(r["n"]), fx(r["dt"], 4), sci(r["err_euler"], 3), s, sci(r["err_rk4"], 2)] for r, s in zip(rows, se)], "rrrrr")
    g = m1d.G
    pred = 0.5 * g * 3.0 * rows[0]["dt"]

    # prueba adicional con arrastre cuadratico (no hay solucion cerrada simple): RK4 si tiene orden 4
    def rhs(_t, s):
        return np.array([s[1], -g - 0.2 * s[1] * abs(s[1])])
    def run(n, step):
        return integrate(rhs, np.array([0.0, 20.0]), 0.0, 3.0, n, step)[1][-1, 0]
    ref = run(20_000, rk4_step)
    nn = [10, 20, 40, 80]
    eu = [abs(run(n, euler_step) - ref) for n in nn]
    rk = [abs(run(n, rk4_step) - ref) for n in nn]
    s_eu = ["--"] + [f"{np.log(eu[i] / eu[i - 1]) / np.log(nn[i - 1] / nn[i]):.2f}" for i in range(1, len(nn))]
    s_rk = ["--"] + [f"{np.log(rk[i] / rk[i - 1]) / np.log(nn[i - 1] / nn[i]):.2f}" for i in range(1, len(nn))]
    t2 = _tab([r"n", r"\text{err. Euler}", r"\text{orden}", r"\text{err. RK4}", r"\text{orden}"],
              [[str(n), sci(a, 3), s1, sci(b, 3), s2] for n, a, s1, b, s2 in zip(nn, eu, s_eu, rk, s_rk)], "rrrrr")
    _, yr = m1d.free_fall_numeric(10, "rk4")
    return dict(
        numeric_intro=[
            "Si la aceleración es constante se puede integrar a mano, pero el código integra la ecuación diferencial $\\dot y=v$, $\\dot v=-g$ paso a paso. El motivo es que ese mismo esquema funciona cuando la aceleración depende de la posición o la velocidad (arrastre, resortes, gravitación) y ya no hay fórmula cerrada. Aquí el caso sencillo sirve para validar el integrador antes de usarlo en problemas difíciles.",
            "Euler avanza con la pendiente inicial: $\\vec s_{n+1}=\\vec s_n+h\\,f(\\vec s_n)$; su error global es $O(h)$. Runge-Kutta de cuarto orden evalúa cuatro pendientes ($k_1$ al inicio, $k_2$ y $k_3$ en el punto medio, $k_4$ al final) y avanza con $\\vec s_{n+1}=\\vec s_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$; su error local es $O(h^5)$ y el global $O(h^4)$ porque coincide con el desarrollo de Taylor hasta $h^4$.",
            "Por eso, para $y(t)=v_0t-\\tfrac12gt^2$, que es un polinomio de grado 2, RK4 es exacto salvo redondeo: sus términos de error involucran derivadas de orden 3 y superior, que valen cero. Euler en cambio acumula un error $\\tfrac12 g\\,T\\,h$ por usar siempre la pendiente del inicio. La limitación de RK4 es que con pasos demasiado grandes o ecuaciones rígidas puede ser inestable, y que cuesta cuatro evaluaciones por paso.",
            "Se tomaron $n=10$ pasos de $h=0.3$ s hasta $T=3$ s para poder hacer la tabla a mano; la convergencia se estudia con $n$ de 10 a 160. El resultado es la posición y la velocidad en cualquier instante, que se comparan con el máximo $y_{\\max}=v_0^2/2g$ y el regreso al suelo.",
            "Es independiente del analítico: nunca evalúa $v_0t-\\tfrac12gt^2$; solo usa $f(\\vec s)=(v,-g)$ repetidamente. La fórmula cerrada se usa después, únicamente para medir el error.",
        ],
        numeric_code="""def f(t, s):                              # lado derecho: s = (y, v)
    return [s[1], -g]                     # y' = v ;  v' = -g  (aceleracion constante)

def euler(f, t, s, h):
    return s + h * f(t, s)                # una pendiente (la inicial): O(h) global

def rk4(f, t, s, h):
    k1 = f(t, s)                          # pendiente al inicio
    k2 = f(t + h/2, s + h/2 * k1)         # pendiente en el punto medio (con k1)
    k3 = f(t + h/2, s + h/2 * k2)         # pendiente en el punto medio (con k2)
    k4 = f(t + h,   s + h * k3)           # pendiente al final
    return s + h/6 * (k1 + 2*k2 + 2*k3 + k4)   # promedio ponderado 1,2,2,1: O(h^4)

s = [0.0, 20.0]                           # y(0)=0 m, v(0)=20 m/s
h = 3.0 / n                               # n pasos hasta T = 3 s
for i in range(n):
    s = rk4(f, i*h, s, h)                 # cada paso parte del resultado anterior
""",
        numeric_converge=[
            ("Euler contra RK4 en la caída libre", "El error en $T=3$ s respecto de la fórmula cerrada se mide al duplicar $n$. Euler tiene orden 1 y su error coincide con $\\tfrac12 gTh$. RK4 es exacto al redondeo, así que su error no baja con $h$: ya está en el piso de $\\epsilon_m$.",
             [t1, rf"\tfrac12 g\,T\,h\big|_{{n=10}} = \tfrac12({g})(3)({rows[0]['dt']:.1f}) = {pred:.4f}\ \mathrm{{m}}"]),
            ("Prueba adicional con arrastre cuadrático", "Para ver el orden 4 de RK4 se necesita un problema no polinómico. Se agrega un arrastre $-0.2\\,v|v|$ y se compara con una solución de referencia de $20\\,000$ pasos. La pendiente aparente de Euler tiende a 1 y la de RK4 a 4 al achicar $h$ (con $h$ grande aún no es asintótica).",
             [t2]),
            ("Fuentes de error y criterio de confianza", "Hay truncamiento (Euler lineal en $h$, RK4 en $h^4$) y redondeo, que se acumula como $\\sim n\\epsilon_m$ y es mucho menor. El resultado de RK4 es confiable cuando al duplicar $n$ el cambio es menor que la tolerancia o cuando las cantidades conservadas del problema (aquí la energía por unidad de masa) no derivan.",
             [rf"y_{{RK4}}(3) - y_{{exacto}}(3) = {sci(float(yr[-1, 0]) - float(m1d.free_fall_analytic(3.0)[0]), 2)}\ \mathrm{{m}},\qquad \epsilon_m = {sci(np.finfo(float).eps, 2)}"]),
        ],
    )


# ====================================================================== cap. 4
def _projectile() -> dict:
    an = c4.projectile_analytic()
    dts = [0.08, 0.04, 0.02, 0.01, 0.005]
    errs, env = [], []
    for dt in dts:
        t, s = c4.projectile_numeric(dt=dt)
        errs.append(abs(s[-1, 0] - an["range"]))
        # el error de la interpolacion depende de la fraccion del paso donde cae el impacto:
        # se toma el maximo sobre 24 pasos distintos entre dt y 2dt (misma escala de h)
        env.append(max(abs(c4.projectile_numeric(dt=dt * (1 + j / 24))[1][-1, 0] - an["range"]) for j in range(24)))
    sl = ["--"] + [f"{np.log(env[i] / env[i - 1]) / np.log(dts[i] / dts[i - 1]):.2f}" for i in range(1, len(dts))]
    vy_land = c4.V0 * np.sin(np.deg2rad(c4.THETA_DEG))
    c_bound = c4.V0 * np.cos(np.deg2rad(c4.THETA_DEG)) * c4.G / (8 * vy_land)
    tab = _tab([r"h\ \mathrm{[s]}", r"|R_{num}-R|\ \mathrm{[m]}", r"\max_{h'\in[h,2h]}|R_{num}-R|", r"\text{orden}", r"C\,h^2"],
              [[fx(dt, 3), sci(e, 3), sci(en, 3), s, sci(c_bound * (2 * dt) ** 2, 3)] for dt, e, en, s in zip(dts, errs, env, sl)], "rrrrr")

    # busqueda de seccion aurea con k iteraciones
    def golden(k):
        phi = (np.sqrt(5) - 1) / 2
        a, b = 5.0, 85.0
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = c4.range_numeric(c), c4.range_numeric(d)
        for _ in range(k):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = c4.range_numeric(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = c4.range_numeric(d)
        return (a + b) / 2, b - a
    ks = [5, 10, 15, 20, 30, 40]
    grow = []
    for k in ks:
        th, w = golden(k)
        grow.append([str(k), sci(w, 3), sci(abs(th - 45.0), 3)])
    t2 = _tab([r"k", r"\text{ancho del intervalo }[^\circ]", r"|\theta_k-45^\circ|\ [^\circ]"], grow, "rrr")
    t_n, s_n = c4.projectile_numeric(dt=0.005)
    n_steps = len(t_n) - 1
    d_r = abs(c4.range_numeric(45.0) - c4.V0**2 / c4.G)
    d_th = float(np.rad2deg(np.sqrt(d_r / (c4.V0**2 / c4.G))))
    th_opt = c4.optimal_angle_numeric()
    return dict(
        numeric_intro=[
            "El tiro parabólico con aceleración constante tiene solución exacta, pero el código lo integra numéricamente con RK4 para validar el esquema y porque el mismo programa vale cuando se agrega arrastre del aire, donde ya no hay parábola. Se integra el vector de estado $\\vec q=(x,y,v_x,v_y)$ con $\\dot{\\vec q}=(v_x,v_y,0,-g)$ hasta que $y$ cruza el suelo, y luego se refina el instante de impacto.",
            "RK4 usa $\\vec q_{n+1}=\\vec q_n+\\tfrac h6(k_1+2k_2+2k_3+k_4)$ y tiene error global $O(h^4)$; para este movimiento polinómico es exacto salvo redondeo. El error que sí aparece viene de detectar el impacto: se localiza el primer paso con $y<0$ y se interpola linealmente entre los dos estados vecinos. Como $y(t)$ es una parábola y no una recta, esa interpolación comete un error $O(h^2)$, que es el que se observa en el alcance.",
            "Para el ángulo óptimo se maximiza $R(\\theta)$ con búsqueda de sección áurea: se mantiene un intervalo $[a,b]$ y cada iteración lo reduce por el factor $\\varphi=0.618$ reutilizando una evaluación. Sirve para funciones unimodales sin necesitar derivadas, pero cerca del máximo $R(\\theta)$ es plana, $R\\approx R_{\\max}-\\tfrac12|R''|\\,\\delta\\theta^2$, así que la precisión del ángulo queda limitada por el error con que se calcula $R$: un error $\\delta R$ en el alcance limita el ángulo a $\\delta\\theta\\sim\\sqrt{\\delta R/R}$ (en radianes).",
            "Los parámetros: $h=0.005$ s en la solución principal (unos 650 pasos de vuelo), $h=0.002$ s para evaluar $R(\\theta)$ en la búsqueda, y 40 iteraciones áureas ($0.618^{40}\\approx4\\times10^{-9}$ del intervalo inicial). El resultado es el alcance, la altura máxima, el tiempo de vuelo y el ángulo que maximiza el alcance.",
            "Es independiente del analítico: solo se usa $f(\\vec q)$ y un criterio geométrico de impacto; la fórmula $v_0^2\\sin2\\theta/g$ y el ángulo de $45^\\circ$ se emplean exclusivamente para medir el error. Que la búsqueda áurea encuentre $45^\\circ$ por sí sola confirma la teoría del alcance.",
        ],
        numeric_code="""def f(t, q):                                  # q = (x, y, vx, vy)
    return [q[2], q[3], 0.0, -g]              # x'=vx, y'=vy, vx'=0, vy'=-g

q = [0, 0, v0*cos(th), v0*sin(th)]            # estado inicial: (25 m/s, 40 grados)
t = 0
while True:
    q_nuevo = rk4(f, t, q, h)                 # un paso RK4 de tamano h
    if q_nuevo[1] < 0 and t > 0:              # primer paso que cruza y = 0
        frac = q[1] / (q[1] - q_nuevo[1])     # fraccion del paso hasta el suelo
        q_hit = q + frac*(q_nuevo - q)        # interpolacion lineal del estado
        t_hit = t + frac*h                    # instante de impacto
        break
    q, t = q_nuevo, t + h                     # avanzar

def mejor_angulo():                           # maximizar alcance (seccion aurea)
    a, b = 5, 85                              # intervalo inicial en grados
    for _ in range(40):                       # cada iteracion reduce por 0.618
        c, d = b - 0.618*(b-a), a + 0.618*(b-a)
        a, b = (a, d) if alcance(c) > alcance(d) else (c, b)
    return (a + b) / 2
""",
        numeric_converge=[
            ("Error del alcance y del tiempo con el paso h", "Se integra con distintos $h$ y se compara con $R=v_0^2\\sin2\\theta/g$. El error de un solo $h$ es irregular porque depende de en qué fracción del paso cae el impacto; por eso se toma también el máximo sobre 24 valores de $h'$ entre $h$ y $2h$, que sí decrece con la pendiente 2 de la interpolación lineal (no con 4 de RK4) y queda bajo la cota $C h^2$ con $C=v_x g/(8v_{y0})$ (ya con $(2h)^2$ en la columna).",
             [tab,
              rf"R = {fx(an['range'], 5)}\ \mathrm{{m}},\qquad T = {fx(an['t_flight'], 5)}\ \mathrm{{s}}"]),
            ("Convergencia de la búsqueda áurea", "El ancho del intervalo se reduce por 0.618 en cada iteración, pero el error del ángulo se estanca en unos $10^{-4}$ grados: cerca del máximo $R$ es plana y el error propio de $R$ (por el paso $h=0.002$ s) impide distinguir ángulos más cercanos.",
             [t2, rf"\theta_{{opt}} = {fx(th_opt, 6)}^\circ\ (\text{{teoría }}45^\circ)"]),
            ("Fuentes de error y criterio de confianza", "Hay truncamiento por la interpolación del impacto ($h^2$), redondeo despreciable y, en el ángulo, un piso fijado por el error de $R$ y la planicie del máximo. El resultado es confiable cuando al reducir $h$ a la mitad el alcance cambia menos que la tolerancia y la búsqueda áurea da el mismo ángulo con otro $h$.",
             [rf"h=0.005:\ \ |R_{{num}}-R| = {sci(abs(s_n[-1, 0] - an['range']), 2)}\ \mathrm{{m}},\qquad {n_steps}\ \text{{pasos completos}}",
              rf"\delta R(\theta=45^\circ,\ h=0.002) = {sci(d_r, 2)}\ \mathrm{{m}}\ \Rightarrow\ \delta\theta_{{min}}\sim\sqrt{{\delta R/R}} = {sci(d_th, 2)}\ \mathrm{{grados}}"]),
        ],
    )


def _circular() -> dict:
    ca = c4.circular_analytic()
    hs = [0.5, 0.25, 0.1, 0.05, 0.01]
    tab_ = c4.fd_error_table(hs=tuple(hs))
    ev = [r["err_speed"] for r in tab_]
    ea = [r["err_a"] for r in tab_]
    sv = ["--"] + [f"{np.log(ev[i] / ev[i - 1]) / np.log(hs[i] / hs[i - 1]):.2f}" for i in range(1, len(hs))]
    sa = ["--"] + [f"{np.log(ea[i] / ea[i - 1]) / np.log(hs[i] / hs[i - 1]):.2f}" for i in range(1, len(hs))]
    t1 = _tab([r"h\ \mathrm{[s]}", r"|v_{fd}|-v", r"\text{orden}", r"|a_{fd}|-a_c", r"\text{orden}"],
              [[str(h), sci(r["speed"] - ca["speed"], 3), s1, sci(r["a_c"] - ca["a_c"], 3), s2] for h, r, s1, s2 in zip(hs, tab_, sv, sa)], "rrrrr")
    h2 = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    tab2 = c4.fd_error_table(hs=tuple(h2))
    t2 = _tab([r"h\ \mathrm{[s]}", r"\text{err. velocidad}", r"\text{err. aceleración}"],
              [[sci(h, 1), sci(r["err_speed"], 3), sci(r["err_a"], 3)] for h, r in zip(h2, tab2)], "rrr")
    w = ca["omega"]
    h0 = 0.01
    pred_v = ca["speed"] * w**2 * h0**2 / 6
    pred_a = ca["a_c"] * w**2 * h0**2 / 12
    h_opt = float(np.finfo(float).eps ** 0.25)
    return dict(
        numeric_intro=[
            "En el movimiento circular se conoce la posición $\\vec r(t)$ y se quiere la velocidad y la aceleración. El código estima las derivadas con diferencias finitas, usando solo valores de $\\vec r$ en instantes cercanos. Esto es lo que se hace con datos experimentales (posiciones medidas por video o sensores), donde no existe una fórmula que se pueda derivar a mano.",
            "Las fórmulas son $\\vec v\\approx[\\vec r(t+h)-\\vec r(t-h)]/(2h)$ y $\\vec a\\approx[\\vec r(t+h)-2\\vec r(t)+\\vec r(t-h)]/h^2$. Con Taylor, los términos impares se cancelan por la simetría de los puntos, de modo que ambas tienen error $O(h^2)$: $\\vec v_{fd}=\\vec v+\\tfrac16\\vec r'''h^2+\\dots$. En este movimiento $\\vec r'''=-\\omega^2\\vec v$ y $\\vec r''''=\\omega^4\\vec r$, así que el error relativo es $\\omega^2h^2/6$ en la velocidad y $\\omega^2h^2/12$ en la aceleración.",
            "La limitación es el redondeo: el numerador resta cantidades casi iguales, y su error absoluto $\\sim\\epsilon_m|\\vec r|$ se divide por $h$ en la velocidad y por $h^2$ en la aceleración. El error total es truncamiento $\\propto h^2$ más redondeo $\\propto\\epsilon_m/h$ (o $/h^2$), con un mínimo en un $h$ óptimo. Para la segunda derivada ese $h$ está cerca de $\\epsilon_m^{1/4}\\approx10^{-4}$ s.",
            "Se evalúa en $t=0.7$ s con $h=0.01$ s: es pequeño frente al periodo de 4 s (error relativo $\\omega^2h^2/6\\approx4\\times10^{-5}$) y mayor que el $h_{opt}$ de la aceleración, así que domina el truncamiento. Los resultados son $v=\\omega R=3.1416$ m/s y $a_c=\\omega^2R=4.9348$ m/s$^2$, con la aceleración apuntando hacia el centro.",
            "Es independiente del analítico: no se usan $\\omega R$ ni $\\omega^2R$ para derivar, solo restas de posiciones. Que la aceleración numérica sea antiparalela a $\\vec r$ y de módulo $v^2/R$ confirma la teoría de la aceleración centrípeta.",
        ],
        numeric_code="""R, T = 2.0, 4.0                                  # radio [m] y periodo [s]
w = 2*pi / T                                     # velocidad angular (solo para construir r)

def r(t):                                        # posicion: unica informacion que se usa
    return R * [cos(w*t), sin(w*t)]

t, h = 0.7, 0.01                                 # instante de evaluacion y paso
v = (r(t + h) - r(t - h)) / (2*h)                # diferencia centrada: error O(h^2)
a = (r(t + h) - 2*r(t) + r(t - h)) / h**2        # segunda diferencia: error O(h^2)

rapidez = norma(v)                               # debe ser w*R
a_c = norma(a)                                   # debe ser w^2*R = v^2/R
cos_ang = dot(a, r(t)) / (a_c * R)               # -1 si a apunta hacia el centro
""",
        numeric_converge=[
            ("Truncamiento: orden 2", "Al reducir $h$ en el rango $0.5$ a $0.01$ s, el error de ambas diferencias baja con pendiente 2. El signo negativo indica que la diferencia centrada subestima el módulo.",
             [t1,
              rf"\text{{teoría con }}h=0.01:\ \ \dfrac{{\omega^2h^2}}{{6}}v = {sci(pred_v, 2)},\qquad \dfrac{{\omega^2h^2}}{{12}}a_c = {sci(pred_a, 2)}"]),
            ("Redondeo: el error vuelve a crecer", "Con $h$ más pequeño el truncamiento ya es despreciable, pero el redondeo crece como $\\epsilon_m/h$ en la velocidad y como $\\epsilon_m/h^2$ en la aceleración. La segunda derivada se degrada mucho antes.",
             [t2, rf"h_{{opt}}\sim\epsilon_m^{{1/4}} = {sci(h_opt, 2)}\ \mathrm{{s}}\ \text{{(aceleración)}}"]),
            ("Fuentes de error y criterio de confianza", "Con $h=0.01$ s el truncamiento es mucho mayor que el redondeo, como muestran las cifras de abajo. Un resultado es confiable cuando está en la zona donde el error baja con pendiente 2 y lejos de la zona de redondeo; una prueba práctica es comparar $h$ con $h/2$ y ver que el cambio es cuatro veces menor.",
             [rf"\text{{truncamiento relativo en }}a:\ \dfrac{{\omega^2h^2}}{{12}} = {sci(w**2 * 0.01**2 / 12, 2)},\qquad \text{{redondeo relativo en }}a:\ \dfrac{{\epsilon_m}}{{h^2}} = {sci(np.finfo(float).eps / 0.01**2, 2)}",
              rf"v_{{fd}} = {fx(np.linalg.norm(c4.velocity_fd(0.7, 0.01)), 6)},\quad v = {fx(ca['speed'], 6)};\qquad a_{{fd}} = {fx(np.linalg.norm(c4.acceleration_fd(0.7, 0.01)), 6)},\quad a_c = {fx(ca['a_c'], 6)}"]),
        ],
    )


DETAIL = {
    "DimensionalAnalysis": _dimensional(),
    "ErrorPropagation": _error_prop(),
    "VectorAddition": _vec_add(),
    "VectorProducts": _vec_prod(),
    "InstantaneousVelocity": _inst_vel(),
    "ConstantAcceleration": _const_acc(),
    "Projectile": _projectile(),
    "CircularMotion": _circular(),
}
