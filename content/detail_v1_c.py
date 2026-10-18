"""Explicacion y pasos a mano: Volumen 1, capitulos 9 a 12.

Escenas: CenterOfMass, ExplodingProjectile, Collision1D, BallisticPendulum,
SpinUp, RollingCycloid, MomentOfInertia, HangingMassDisc.
Todos los numeros salen de las funciones de `physics/` o de aritmetica en Python.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch09_particle_systems as c9
from physics import ch10_collisions as c10
from physics import ch11_rotational_kinematics as c11
from physics import ch12_rotational_dynamics as c12
from physics.integrators import rk4_step
from physics.ch07_work_energy import simpson


# ----------------------------------------------------------------- ayudas de formato
def vec(v, d: int = 4) -> str:
    return r"\left(" + r",\ ".join(sci(float(x), d) for x in v) + r"\right)"


def table(header: list[str], rows: list[list[str]]) -> str:
    cols = "c" * len(header)
    out = r"\begin{array}{" + cols + "}" + " & ".join(header) + r"\\ \hline "
    out += r"\\ ".join(" & ".join(r) for r in rows)
    return out + r"\end{array}"


def rk4_ks(rhs, t, y, h):
    k1 = rhs(t, y)
    k2 = rhs(t + h / 2, y + h / 2 * k1)
    k3 = rhs(t + h / 2, y + h / 2 * k2)
    k4 = rhs(t + h, y + h * k3)
    return k1, k2, k3, k4, y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simpson_terms(y, h) -> str:
    """Escribe la suma de Simpson h/3 (y0 + 4 y1 + 2 y2 + ... + yn) con cifras."""
    n = len(y) - 1
    parts = []
    for i, v in enumerate(y):
        c = 1 if i in (0, n) else (4 if i % 2 else 2)
        parts.append((f"{c}\\," if c > 1 else "") + f"({sci(float(v), 5)})")
    return r"\dfrac{" + sci(h, 5) + r"}{3}\left[" + " + ".join(parts) + r"\right]"


# =========================================================================== CAP. 9
def _cm() -> dict:
    ms, ps = c9.MASSES, c9.POSITIONS
    mt = ms.sum()
    cma, cmn = c9.cm_discrete_analytic(), c9.cm_discrete_numeric()
    L = c9.ROD_L
    xr = c9.rod_cm_analytic()
    rod_n = c9.rod_cm_numeric()
    mc, xs = c9.rod_cm_monte_carlo()
    n4 = 4
    x4 = np.linspace(0, L, n4 + 1)
    lam4 = x4 / L
    h4 = L / n4
    num4, den4 = simpson(x4 * lam4, x4), simpson(lam4, x4)
    rng = np.random.default_rng(5)
    u = rng.uniform(size=5)
    x5 = L * np.sqrt(u)
    sig = L / np.sqrt(18)
    return dict(
        explanation=[
            "El centro de masa es el punto que representa la posición promedio de un sistema, pero ponderada por cuánta masa hay en cada lugar. Si el sistema fuera una sola partícula con toda la masa $M$ en ese punto, respondería a las fuerzas externas exactamente igual que el sistema real. Por eso reduce un problema de muchos cuerpos a uno de un solo cuerpo.",
            "Para partículas puntuales se calcula como un promedio ordinario con pesos $m_i/M$: las partículas más pesadas \"jalan\" el punto hacia ellas. Para una distribución continua la suma se convierte en una integral, donde cada trozo de masa es $dm=\\lambda\\,dx$ con $\\lambda$ la densidad lineal. El resultado no tiene por qué caer sobre materia: el centro de masa de un aro está en su centro vacío.",
            "En la barra del problema la densidad crece linealmente desde cero en un extremo, así que la masa se acumula hacia el extremo pesado y el centro de masa queda más allá de la mitad de la barra. Una barra uniforme daría $L/2$; aquí el desplazamiento hacia el lado denso lleva el resultado a $2L/3$. Es una buena comprobación de sentido común: el valor debe estar siempre entre los extremos y más cerca del lado más masivo.",
            "El orden de magnitud también se verifica por simetría y por cotas: las coordenadas del centro de masa de tres partículas nunca salen del triángulo que forman. La idea numérica es la misma que la analítica, solo que se calculan las integrales con una regla de cuadratura o con muestras aleatorias distribuidas según la densidad.",
        ],
        analytic_steps=[
            ("Plantear la definición para partículas", "El centro de masa es el promedio de las posiciones ponderado por las masas. Se calcula cada coordenada por separado.",
             [r"x_{cm}=\dfrac{\sum m_i x_i}{M},\qquad y_{cm}=\dfrac{\sum m_i y_i}{M},\qquad M=\sum m_i"]),
            ("Sumar las masas", "La masa total es el denominador común de ambas coordenadas.",
             [rf"M = 2+3+5 = {fx(mt, 0)}\ \mathrm{{kg}}"]),
            ("Calcular x del centro de masa", "Se multiplica cada masa por su coordenada $x$ y se suma; la partícula en el origen no aporta.",
             [rf"x_{{cm}} = \dfrac{{(2)(0)+(3)(4)+(5)(2)}}{{{fx(mt, 0)}}} = \dfrac{{{fx(float((ms * ps[:, 0]).sum()), 1)}}}{{{fx(mt, 0)}}} = {cma[0]:.4f}\ \mathrm{{m}}"]),
            ("Calcular y del centro de masa", "Solo la tercera partícula está fuera del eje $x$, así que es la única que contribuye.",
             [rf"y_{{cm}} = \dfrac{{(2)(0)+(3)(0)+(5)(3)}}{{{fx(mt, 0)}}} = \dfrac{{{fx(float((ms * ps[:, 1]).sum()), 1)}}}{{{fx(mt, 0)}}} = {cma[1]:.4f}\ \mathrm{{m}}"]),
            ("Plantear la integral de la barra", "Con $\\lambda(x)=\\lambda_0 x/L$ cada elemento es $dm=\\lambda\\,dx$. El centro de masa es el cociente de dos integrales; $\\lambda_0$ se cancela.",
             [r"x_{cm}=\dfrac{\int_0^L x\,\lambda_0\frac{x}{L}\,dx}{\int_0^L \lambda_0\frac{x}{L}\,dx}"]),
            ("Evaluar numerador y denominador", "Se integran potencias de $x$ con la regla $\\int x^n dx = x^{n+1}/(n+1)$.",
             [r"\int_0^L \dfrac{x^2}{L}\,dx=\dfrac{L^2}{3},\qquad \int_0^L \dfrac{x}{L}\,dx=\dfrac{L}{2}"]),
            ("Dividir y sustituir L", "El cociente da dos tercios de la longitud, y con $L=2$ m se obtiene el valor numérico.",
             [r"x_{cm}=\dfrac{L^2/3}{L/2}=\dfrac{2L}{3}", rf"x_{{cm}}=\dfrac{{2(2)}}{{3}}={xr:.4f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Partículas: producto matricial", "El código forma el vector de masas $(2,3,5)$ y la matriz de posiciones de $3\\times2$, y calcula el producto $\\mathbf m^{T}\\mathbf P$, que entrega $(\\sum m_i x_i,\\ \\sum m_i y_i)$ de una sola vez; luego divide entre $M$.",
             [rf"\mathbf m^{{T}}\mathbf P=\left({fx(float((ms @ ps)[0]), 1)},\ {fx(float((ms @ ps)[1]), 1)}\right),\qquad \mathbf r_{{cm}}=\dfrac{{1}}{{{fx(mt, 0)}}}\left({fx(float((ms @ ps)[0]), 1)},\ {fx(float((ms @ ps)[1]), 1)}\right)=({cmn[0]:.4f},\ {cmn[1]:.4f})\ \mathrm{{m}}"]),
            ("Barra: preparar la regla de Simpson", "Se divide $[0,L]$ en $N$ subintervalos pares de ancho $h$ y se evalúan los integrandos $x\\lambda$ (numerador) y $\\lambda$ (denominador) con $\\lambda=x/L$. Para ver la mecánica a mano se usa $N=4$; el código usa $N=200$.",
             [rf"h=\dfrac{{L}}{{N}}=\dfrac{{2}}{{4}}={h4}\ \mathrm{{m}},\qquad \int f\,dx\approx\dfrac{{h}}{{3}}\left[f_0+4f_1+2f_2+4f_3+f_4\right]",
              table(["x_i", "0", "0.5", "1.0", "1.5", "2.0"],
                    [[r"\lambda_i=x_i/L"] + [fx(v, 3) for v in lam4], [r"x_i\lambda_i"] + [fx(v, 4) for v in x4 * lam4]])]),
            ("Barra: Simpson con N = 4", "Se aplican los pesos $1,4,2,4,1$ a cada columna de la tabla. Como los integrandos son polinomios de grado $\\le 3$, Simpson ya es exacto con tan pocos puntos.",
             [r"\text{num}=" + simpson_terms(x4 * lam4, h4) + "=" + fx(num4, 5),
              r"\text{den}=" + simpson_terms(lam4, h4) + "=" + fx(den4, 5),
              rf"x_{{cm}}=\dfrac{{{fx(num4, 5)}}}{{{fx(den4, 5)}}}={num4 / den4:.5f}\ \mathrm{{m}}"]),
            ("Barra: Simpson con N = 200", "El código repite exactamente lo mismo con 200 subintervalos ($h=0.01$ m). El resultado coincide con $2L/3$ a precisión de máquina porque Simpson integra exactamente cúbicas.",
             [rf"x_{{cm}}^{{(Simpson)}}={rod_n:.8f}\ \mathrm{{m}},\qquad \left|x_{{cm}}-\tfrac{{2L}}{{3}}\right|={sci(abs(rod_n - xr), 2)}\ \mathrm{{m}}"]),
            ("Monte Carlo: muestrear con la densidad", "Para obtener puntos distribuidos según $\\lambda\\propto x$ se usa la transformada inversa. La función de distribución acumulada es $F(x)=x^2/L^2$; igualarla a un número uniforme $u\\in[0,1)$ y despejar da $x=L\\sqrt{u}$.",
             [r"F(x)=\dfrac{\int_0^x \lambda\,dx'}{\int_0^L \lambda\,dx'}=\dfrac{x^2}{L^2}=u\;\Rightarrow\; x=L\sqrt{u}"]),
            ("Monte Carlo: primeras muestras", "Con la semilla 5 del generador de numpy salen los primeros cinco números uniformes; cada uno se transforma y el centro de masa estimado es el promedio simple de las posiciones (los pesos ya van en la densidad del muestreo).",
             [table(["i", "u_i", "x_i=2\\sqrt{u_i}"], [[str(i + 1), fx(u[i], 5), fx(x5[i], 5)] for i in range(5)]),
              rf"\bar x_{{(5)}}=\dfrac{{1}}{{5}}\sum x_i={x5.mean():.5f}\ \mathrm{{m}}"]),
            ("Monte Carlo: 400 000 muestras y su error", "El código promedia $n=400\\,000$ posiciones. El error típico decae como $\\sigma/\\sqrt n$ con $\\sigma=L/\\sqrt{18}$ (la desviación estándar de la distribución $\\propto x$), de modo que la diferencia con $2L/3$ debe ser del orden de $10^{-3}$ m.",
             [rf"\sigma=\dfrac{{L}}{{\sqrt{{18}}}}={sig:.4f}\ \mathrm{{m}},\qquad \dfrac{{\sigma}}{{\sqrt n}}={sci(sig / np.sqrt(len(xs)), 3)}\ \mathrm{{m}}",
              rf"\bar x_{{(400\,000)}}={mc:.6f}\ \mathrm{{m}},\qquad \bar x-\tfrac{{2L}}{{3}}={sci(mc - xr, 3)}\ \mathrm{{m}}"]),
        ],
    )


def _explode() -> dict:
    g = c9.G
    ap = c9.apex_state()
    an = c9.fragments_analytic()
    nu = c9.fragments_numeric()
    R = c9.range_analytic()
    th = np.deg2rad(c9.ANGLE_DEG)
    vy0 = c9.V0 * np.sin(th)
    ta, ya = nu["A"]
    tb, yb = nu["B"]
    h = 1e-3
    rhs = lambda _t, s: np.array([s[2], s[3], 0.0, -g])  # noqa: E731
    s0 = np.array([ap["x"], ap["y"], 0.0, 0.0])
    k1, k2, k3, k4, s1 = rk4_ks(rhs, 0.0, s0, h)
    frac = ya[-2, 1] / (ya[-2, 1] - rk4_step(rhs, 0.0, ya[-2], h)[1])
    return dict(
        explanation=[
            "Una explosión es un proceso interno: las fuerzas que despedazan el proyectil son pares de acción y reacción entre fragmentos. Por eso no cambian el momento lineal total ni la aceleración del centro de masa, que solo responde a las fuerzas externas (aquí el peso). Esta es la idea central: el centro de masa sigue la misma parábola que habría seguido el proyectil entero, sin importar cómo se fragmente.",
            "La explosión se produce en el punto más alto, donde la velocidad vertical es cero y solo queda la componente horizontal $v_x=v_0\\cos\\theta$. Como el fragmento A parte del reposo, cae en vertical; para que el momento horizontal se conserve, el fragmento B, más liviano en proporción, debe salir con una velocidad mayor que la original, cuatro tercios de $v_x$.",
            "Después de la explosión cada fragmento es un proyectil con su propia velocidad horizontal pero la misma altura inicial, así que ambos tardan lo mismo en caer (el tiempo de caída libre desde la altura del vértice). Lo único que difiere es cuánto avanzan horizontalmente durante ese tiempo.",
            "La comprobación elegante es que el centro de masa de los fragmentos llega al suelo exactamente en el alcance $R$ del proyectil sin explotar: lo que A pierde de alcance, B lo gana, ponderado por las masas. Si el resultado no cumpliera esto, habría un error de planteamiento.",
        ],
        analytic_steps=[
            ("Descomponer la velocidad inicial", "Se separan las componentes horizontal y vertical con el ángulo de 60°. La horizontal permanece constante durante todo el vuelo.",
             [rf"v_x=v_0\cos\theta=20\cos 60^\circ={ap['vx']:.4f}\ \mathrm{{m/s}},\qquad v_{{y0}}=v_0\sin\theta={vy0:.4f}\ \mathrm{{m/s}}"]),
            ("Estado en el punto más alto", "En el vértice $v_y=0$. El tiempo de subida y la altura salen de la cinemática vertical con $g=9.81$ m/s².",
             [rf"t_a=\dfrac{{v_{{y0}}}}{{g}}=\dfrac{{{vy0:.4f}}}{{{g}}}={ap['t']:.4f}\ \mathrm{{s}}",
              rf"x_a=v_xt_a={ap['x']:.4f}\ \mathrm{{m}},\qquad y_a=v_{{y0}}t_a-\tfrac12gt_a^2={ap['y']:.4f}\ \mathrm{{m}}"]),
            ("Conservar el momento horizontal en la explosión", "Antes de explotar el momento horizontal es $Mv_x$. Después, A (1 kg) tiene velocidad horizontal nula y solo B (3 kg) lo lleva.",
             [r"M v_x = m_A(0)+m_B v_B\;\Rightarrow\; v_B=\dfrac{M}{m_B}v_x",
              rf"v_B=\dfrac{{4}}{{3}}({ap['vx']:.4f})={an['vb']:.4f}\ \mathrm{{m/s}}"]),
            ("Tiempo de caída de los fragmentos", "Ambos parten de la altura $y_a$ con velocidad vertical nula, así que caen en caída libre durante el mismo tiempo.",
             [rf"t_f=\sqrt{{\dfrac{{2y_a}}{{g}}}}=\sqrt{{\dfrac{{2({ap['y']:.4f})}}{{{g}}}}}={an['t_fall']:.4f}\ \mathrm{{s}}"]),
            ("Posiciones de caída", "A no se mueve horizontalmente; B avanza con $v_B$ durante $t_f$.",
             [rf"x_A=x_a={an['xa']:.4f}\ \mathrm{{m}}",
              rf"x_B=x_a+v_Bt_f={ap['x']:.4f}+({an['vb']:.4f})({an['t_fall']:.4f})={an['xb']:.4f}\ \mathrm{{m}}"]),
            ("Centro de masa final", "Se promedian las posiciones de caída con las masas de los fragmentos.",
             [rf"x_{{cm}}=\dfrac{{m_Ax_A+m_Bx_B}}{{M}}=\dfrac{{(1)({an['xa']:.4f})+(3)({an['xb']:.4f})}}{{4}}={an['x_cm']:.4f}\ \mathrm{{m}}"]),
            ("Verificar con el alcance sin explosión", "Si no hubiera explosión, el alcance sería $R=v_0^2\\sin2\\theta/g$. Debe coincidir con $x_{cm}$.",
             [rf"R=\dfrac{{(20)^2\sin120^\circ}}{{{g}}}={R:.4f}\ \mathrm{{m}}={an['x_cm']:.4f}\ \mathrm{{m}}=x_{{cm}}"]),
        ],
        numeric_steps=[
            ("Plantear el sistema de primer orden", "Cada fragmento se integra por separado desde el vértice con el vector de estado $\\mathbf s=(x,y,u,w)$, donde $u,w$ son las velocidades. No hay aire, así que sólo actúa la gravedad.",
             [r"\dot{\mathbf s}=\mathbf f(\mathbf s)=(u,\ w,\ 0,\ -g)"]),
            ("Condiciones iniciales y paso", "El código usa $h=1$ ms. A parte de $(x_a,y_a)$ con velocidad nula; B con $u=v_B$.",
             [rf"\mathbf s_0^{{A}}={vec(s0, 5)},\qquad \mathbf s_0^{{B}}=({sci(ap['x'], 5)},\ {sci(ap['y'], 5)},\ {sci(an['vb'], 5)},\ 0),\qquad h={sci(h, 1)}\ \mathrm{{s}}"]),
            ("Primer paso RK4 del fragmento A: pendientes", "Se evalúan cuatro pendientes: al inicio, dos en el punto medio y una al final. Las pendientes $k_2,k_3,k_4$ usan la velocidad vertical corregida por $-g\\,h/2$ o $-g\\,h$.",
             [rf"k_1={vec(k1, 5)},\qquad k_2={vec(k2, 5)}",
              rf"k_3={vec(k3, 5)},\qquad k_4={vec(k4, 5)}"]),
            ("Primer paso RK4 del fragmento A: nuevo estado", "Se combinan las pendientes con pesos $1,2,2,1$ sobre 6. Como la aceleración es constante, RK4 reproduce la solución exacta de caída libre.",
             [r"\mathbf s_1=\mathbf s_0+\dfrac{h}{6}\left(k_1+2k_2+2k_3+k_4\right)",
              rf"\mathbf s_1={vec(s1, 8)}"]),
            ("Repetir hasta tocar el suelo", "El lazo avanza pasos de 1 ms mientras $y>0$. Al primer paso que deja $y\\le 0$ se detiene.",
             [rf"\text{{pasos de A}}={len(ta) - 1},\qquad \text{{pasos de B}}={len(tb) - 1}"]),
            ("Interpolar el instante de impacto", "El último paso se pasó del suelo; se interpola linealmente entre los dos últimos estados con la fracción $f=y_{n-1}/(y_{n-1}-y_n)$ para ubicar el impacto.",
             [rf"f=\dfrac{{y_{{n-1}}}}{{y_{{n-1}}-y_n}}={frac:.5f},\qquad \mathbf s_{{impacto}}=\mathbf s_{{n-1}}+f(\mathbf s_n-\mathbf s_{{n-1}})",
              rf"t_A={ta[-1]:.5f}\ \mathrm{{s}},\qquad t_B={tb[-1]:.5f}\ \mathrm{{s}}\ \ (\text{{analítico }}{an['t_fall']:.5f})"]),
            ("Resultados y error", "Las posiciones de impacto y el centro de masa reconstruido se comparan con las analíticas.",
             [rf"x_A={nu['xa']:.5f}\ \mathrm{{m}},\quad x_B={nu['xb']:.5f}\ \mathrm{{m}},\quad x_{{cm}}={nu['x_cm']:.5f}\ \mathrm{{m}}",
              rf"|x_A-x_A^{{an}}|={sci(abs(nu['xa'] - an['xa']), 2)},\quad |x_B-x_B^{{an}}|={sci(abs(nu['xb'] - an['xb']), 2)},\quad |x_{{cm}}-R|={sci(abs(nu['x_cm'] - R), 2)}\ \mathrm{{m}}"]),
        ],
    )


# ========================================================================== CAP. 10
def _contact_trace(zeta, k=1.0e4, dt=1e-5):
    """Repite contact_simulation guardando el primer paso dentro del contacto."""
    m1, m2, v1 = c10.M1, c10.M2, c10.V1
    meff = m1 * m2 / (m1 + m2)
    c = 2 * zeta * np.sqrt(k * meff)

    def rhs(_t, s):
        x1, x2, u1, u2 = s
        ov = x1 - x2
        if ov <= 0.0:
            return np.array([u1, u2, 0.0, 0.0])
        f = k * ov + c * (u1 - u2)
        return np.array([u1, u2, -f / m1, f / m2])

    s = np.array([-0.02, 0.0, v1, 0.0])
    t, n, touched, first = 0.0, 0, False, None
    s_first_free = rk4_step(rhs, t, s, dt)
    while True:
        prev = s.copy()
        s = rk4_step(rhs, t, s, dt)
        t += dt
        n += 1
        if first is None and prev[0] - prev[1] > 0:
            first = (n, prev, s.copy(), rhs(t, prev))
        if s[0] - s[1] > 0:
            touched = True
        if touched and s[0] - s[1] <= 0:
            break
        if t > 1.0:
            break
    return dict(rhs=rhs, steps=n, t=t, s=s, first=first, free=s_first_free, c=c, meff=meff)


def _collision() -> dict:
    m1, m2, v1 = c10.M1, c10.M2, c10.V1
    p = m1 * v1
    vcm = p / (m1 + m2)
    k0 = 0.5 * m1 * v1**2
    out = {e: c10.outcome_analytic(e) for e in (1.0, 0.5, 0.0)}
    zs = (0.0, 0.3, 0.7)
    k = 1.0e4
    tr = _contact_trace(0.3)
    meff = tr["meff"]
    w0 = np.sqrt(k / meff)
    z = 0.3
    wd = w0 * np.sqrt(1 - z**2)
    n0, prev, nxt, kk = tr["first"]
    v1f, v2f = c10.contact_simulation(0.3)
    rows_e = [[sci(z_, 2), sci(c10.restitution_from_damping(z_), 5), sci(c10.restitution_measured(z_), 5),
               sci(abs(c10.restitution_from_damping(z_) - c10.restitution_measured(z_)), 2)] for z_ in zs]
    return dict(
        explanation=[
            "En una colisión corta las fuerzas entre los cuerpos son enormes comparadas con cualquier fuerza externa, así que el momento lineal total se conserva en todos los casos. Lo que distingue a unas colisiones de otras es qué pasa con la energía cinética: en una colisión elástica se conserva, en una inelástica parte se transforma en calor, sonido y deformación, y en la perfectamente inelástica los cuerpos quedan unidos.",
            "El coeficiente de restitución $e$ resume esa diferencia con un solo número: es el cociente entre la rapidez relativa de separación y la de aproximación. Vale 1 si el choque es elástico, 0 si los cuerpos quedan pegados y está entre ambos valores en los choques reales. Junto con la conservación del momento, es una segunda ecuación que permite despejar las dos velocidades finales.",
            "Con el cuerpo pesado golpeando al ligero (2 kg contra 1 kg) ambos siguen hacia adelante en el caso elástico, y el ligero sale con más rapidez que la incidente: $4$ m/s frente a $3$ m/s. Esto no viola la energía porque el pesado cede parte de su movimiento. En el caso opuesto $e=0$ ambos terminan a la velocidad del centro de masa, $2$ m/s.",
            "Para entender de dónde sale $e$ se modela el contacto como un resorte con amortiguador. El resorte devuelve la energía comprimida y el amortiguador la disipa; el cociente entre la velocidad de salida y la de entrada depende solo del amortiguamiento relativo $\\zeta$ y vale $e=\\exp\\!\\left(-\\pi\\zeta/\\sqrt{1-\\zeta^2}\\right)$. Así, un contacto sin amortiguamiento es elástico y uno muy amortiguado es casi plástico.",
        ],
        analytic_steps=[
            ("Conservar el momento lineal", "Se calcula el momento total antes del choque; es el mismo después, sin importar $e$. También se obtiene la velocidad del centro de masa.",
             [rf"P=m_1v_1+m_2(0)=(2)(3)={p:.1f}\ \mathrm{{kg\,m/s}},\qquad v_{{cm}}=\dfrac{{P}}{{m_1+m_2}}=\dfrac{{{p:.1f}}}{{3}}={vcm:.1f}\ \mathrm{{m/s}}"]),
            ("Escribir la condición de restitución", "Con $m_2$ en reposo, la rapidez relativa antes es $v_1$ y después es $v_2'-v_1'$.",
             [r"e=\dfrac{v_2'-v_1'}{v_1-0}\;\Rightarrow\; v_2'=v_1'+e\,v_1"]),
            ("Despejar las velocidades finales", "Se sustituye $v_2'$ en la ecuación de momento y se despeja $v_1'$; después se obtiene $v_2'$.",
             [r"m_1v_1=m_1v_1'+m_2(v_1'+ev_1)\;\Rightarrow\; v_1'=\dfrac{m_1-em_2}{m_1+m_2}\,v_1,\qquad v_2'=\dfrac{(1+e)m_1}{m_1+m_2}\,v_1"]),
            ("Caso elástico, e = 1", "Se sustituyen $m_1=2$, $m_2=1$ y $v_1=3$ m/s. El cuerpo ligero sale más rápido que la rapidez incidente.",
             [rf"v_1'=\dfrac{{2-1(1)}}{{3}}(3)={out[1.0][0]:.3f}\ \mathrm{{m/s}},\qquad v_2'=\dfrac{{(2)(2)}}{{3}}(3)={out[1.0][1]:.3f}\ \mathrm{{m/s}}"]),
            ("Caso intermedio, e = 0.5", "Misma sustitución con $e=0.5$.",
             [rf"v_1'=\dfrac{{2-0.5(1)}}{{3}}(3)={out[0.5][0]:.3f}\ \mathrm{{m/s}},\qquad v_2'=\dfrac{{(1.5)(2)}}{{3}}(3)={out[0.5][1]:.3f}\ \mathrm{{m/s}}"]),
            ("Caso perfectamente inelástico, e = 0", "Los cuerpos salen juntos con la velocidad del centro de masa.",
             [rf"v_1'=\dfrac{{2-0}}{{3}}(3)={out[0.0][0]:.3f}\ \mathrm{{m/s}},\qquad v_2'=\dfrac{{(1)(2)}}{{3}}(3)={out[0.0][1]:.3f}\ \mathrm{{m/s}}"]),
            ("Energía perdida y verificación", "La energía cinética inicial es $K_0=\\tfrac12m_1v_1^2$ y la pérdida es $\\tfrac12\\mu v_1^2(1-e^2)$ con $\\mu=m_1m_2/(m_1+m_2)$. Para $e=0.5$ se comprueba directamente.",
             [rf"K_0=\tfrac12(2)(3)^2={k0:.2f}\ \mathrm{{J}},\qquad \mu={meff:.4f}\ \mathrm{{kg}}",
              rf"K_f=\tfrac12(2)({out[0.5][0]:.3f})^2+\tfrac12(1)({out[0.5][1]:.3f})^2={0.5 * m1 * out[0.5][0]**2 + 0.5 * m2 * out[0.5][1]**2:.4f}\ \mathrm{{J}},\qquad \Delta K=\tfrac12\mu v_1^2(1-e^2)={0.5 * meff * v1**2 * (1 - 0.25):.4f}\ \mathrm{{J}}"]),
            ("Relacionar e con el amortiguamiento", "En el modelo resorte-amortiguador el contacto dura media oscilación amortiguada, y la velocidad relativa decae un factor $e^{-\\zeta\\omega_0 t}$ en ese lapso con $t=\\pi/\\omega_d$. Eso da la fórmula con la que se compara la simulación.",
             [r"e=\exp\!\left(-\zeta\omega_0\dfrac{\pi}{\omega_d}\right)=\exp\!\left(-\dfrac{\pi\zeta}{\sqrt{1-\zeta^2}}\right)",
              rf"\zeta=0.3:\quad e=\exp\!\left(-\dfrac{{\pi(0.3)}}{{\sqrt{{1-0.09}}}}\right)={c10.restitution_from_damping(0.3):.5f}"]),
        ],
        numeric_steps=[
            ("Modelo de contacto", "Las partículas son puntos con posiciones $x_1<x_2$. Hay contacto cuando $x_1-x_2>0$ (solape), y la fuerza es resorte más amortiguador. El amortiguamiento se fija con $\\zeta$ mediante $c=2\\zeta\\sqrt{k\\mu}$.",
             [r"F=k(x_1-x_2)+c(u_1-u_2),\qquad \dot u_1=-F/m_1,\quad \dot u_2=+F/m_2",
              rf"k={sci(k, 1)}\ \mathrm{{N/m}},\quad \mu={meff:.4f}\ \mathrm{{kg}},\quad \omega_0=\sqrt{{k/\mu}}={w0:.3f}\ \mathrm{{rad/s}},\quad c(\zeta=0.3)={tr['c']:.4f}\ \mathrm{{N\,s/m}}"]),
            ("Paso temporal y condiciones iniciales", "El paso es $h=10\\ \\mu$s, mucho menor que el periodo de contacto, y el estado inicial es $(x_1,x_2,u_1,u_2)=(-0.02,\\,0,\\,3,\\,0)$: la partícula 1 está 2 cm antes del contacto.",
             [rf"h={sci(1e-5, 1)}\ \mathrm{{s}},\qquad \text{{tiempo hasta el contacto}}\approx\dfrac{{0.02}}{{3}}={0.02 / 3 * 1000:.3f}\ \mathrm{{ms}}"]),
            ("Primer paso RK4 (vuelo libre)", "Mientras no hay solape las aceleraciones son nulas: las cuatro pendientes de velocidad valen cero y las de posición valen $u$. La partícula 1 solo avanza $u_1h$.",
             [rf"\mathbf s_1=\mathbf s_0+h\,(u_1,u_2,0,0)=({tr['free'][0]:.8f},\ {tr['free'][1]:.1f},\ {tr['free'][2]:.1f},\ {tr['free'][3]:.1f})"]),
            ("Primer paso con contacto", "Tras unos 667 pasos de vuelo libre aparece solape. Si en el paso número $n$ ya hay solape al inicio, $k_1$ incluye la fuerza del resorte y del amortiguador.",
             [rf"n={n0},\quad x_1-x_2={sci(prev[0] - prev[1], 4)}\ \mathrm{{m}},\quad F=k(x_1-x_2)+c(u_1-u_2)={sci(float(-kk[2] * m1), 4)}\ \mathrm{{N}}",
              rf"k_1=({sci(kk[0], 4)},\ {sci(kk[1], 4)},\ {sci(kk[2], 4)},\ {sci(kk[3], 4)})\quad\Rightarrow\quad \mathbf s={vec(nxt, 6)}"]),
            ("Criterio de parada", "Se sigue integrando hasta que, habiendo tocado, el solape vuelva a ser nulo o negativo. En ese instante las velocidades son las finales.",
             [rf"\text{{pasos totales}}={tr['steps']},\qquad t_{{fin}}={tr['t'] * 1000:.3f}\ \mathrm{{ms}},\qquad \text{{duración del contacto}}\approx\dfrac{{\pi}}{{\omega_d}}={np.pi / wd * 1000:.3f}\ \mathrm{{ms}}"]),
            ("Medir la restitución", "Con las velocidades finales se aplica la definición $e=(v_2'-v_1')/v_1$. Para $\\zeta=0.3$:",
             [rf"v_1'={v1f:.5f}\ \mathrm{{m/s}},\quad v_2'={v2f:.5f}\ \mathrm{{m/s}},\quad e=\dfrac{{{v2f:.5f}-({v1f:.5f})}}{{3}}={(v2f - v1f) / v1:.5f}"]),
            ("Comparar con la fórmula para tres amortiguamientos", "El mismo lazo se repite para $\\zeta=0,\\,0.3,\\,0.7$. La pequeña diferencia viene del paso finito y de que el contacto termina al cruzar el solape cero entre dos pasos.",
             [table([r"\zeta", r"e_{an}", r"e_{num}", r"|\Delta e|"], rows_e)]),
        ],
    )


def _pendulum() -> dict:
    m, M, v, g, L = c10.BULLET, c10.BLOCK, c10.BULLET_V, c10.G, c10.ROPE
    an, nu = c10.bullet_block_analytic(), c10.bullet_block_numeric()
    V = an["V"]
    k0 = 0.5 * m * v**2
    k1_ = 0.5 * (m + M) * V**2
    cosb = 1 - an["h"] / L
    beta = np.degrees(np.arccos(cosb))
    h = 1e-4
    rhs = lambda _t, y: np.array([y[1], -(g / L) * np.sin(y[0])])  # noqa: E731
    y0 = np.array([0.0, V / L])
    a1, a2, a3, a4, y1 = rk4_ks(rhs, 0.0, y0, h)
    # repetir el lazo del codigo para contar pasos y tiempo del maximo
    y, t, hmax, n = y0.copy(), 0.0, 0.0, 0
    while t < 3.0:
        y = rk4_step(rhs, t, y, h)
        t += h
        n += 1
        hmax = max(hmax, L * (1 - np.cos(y[0])))
        if y[1] < 0:
            break
    return dict(
        explanation=[
            "El péndulo balístico mide la rapidez de una bala sin necesidad de cronometrar algo tan rápido. El truco es dividir el proceso en dos fases con leyes distintas: la incrustación, que es tan breve que el momento lineal se conserva pero la energía mecánica no, y el ascenso, que es lento y conserva la energía mecánica porque la tensión de la cuerda no realiza trabajo.",
            "En la primera fase el choque es perfectamente inelástico: bala y bloque quedan unidos y se mueven con velocidad común $V=mv/(m+M)$. Se pierde casi toda la energía cinética en deformación y calor; como el bloque es 100 veces más pesado que la bala, la fracción perdida es $M/(m+M)\\approx 99\\%$. Usar la energía en esta fase sería un error clásico.",
            "En la segunda fase solo cambia la energía cinética del conjunto por energía potencial gravitatoria, de modo que la altura alcanzada es $h=V^2/(2g)$ y no depende de la masa total. Medir $h$ (o el ángulo de la cuerda) permite recorrer el camino al revés y obtener la rapidez de la bala.",
            "El orden de magnitud tiene sentido: una bala de 250 m/s produce un conjunto que se mueve a unos 2.5 m/s, cifra comparable a una caminata rápida, y que sube apenas unos 31 cm. Por eso el aparato es práctico: una velocidad enorme se convierte en un desplazamiento fácil de medir.",
        ],
        analytic_steps=[
            ("Identificar las dos fases", "Durante la incrustación se conserva el momento lineal horizontal; durante el ascenso se conserva la energía mecánica. Cada principio se aplica solo donde es válido.",
             [r"\text{(1) } mv=(m+M)V\qquad\text{(2) } \tfrac12(m+M)V^2=(m+M)gh"]),
            ("Despejar la velocidad del conjunto", "Se divide el momento inicial entre la masa total, con $m=0.020$ kg, $M=2$ kg y $v=250$ m/s.",
             [r"V=\dfrac{m\,v}{m+M}", rf"V=\dfrac{{(0.020)(250)}}{{0.020+2}}=\dfrac{{{m * v:.3f}}}{{{m + M:.3f}}}={V:.4f}\ \mathrm{{m/s}}"]),
            ("Calcular la altura", "En la ecuación de energía se cancela la masa total y se despeja $h$.",
             [r"h=\dfrac{V^2}{2g}", rf"h=\dfrac{{({V:.4f})^2}}{{2({g})}}=\dfrac{{{V**2:.4f}}}{{{2 * g:.2f}}}={an['h']:.5f}\ \mathrm{{m}}"]),
            ("Energías antes y después de la colisión", "Se calcula la energía cinética de la bala y la del conjunto justo después de incrustarse.",
             [rf"K_0=\tfrac12(0.020)(250)^2={k0:.3f}\ \mathrm{{J}}",
              rf"K_1=\tfrac12(2.020)({V:.4f})^2={k1_:.4f}\ \mathrm{{J}}"]),
            ("Fracción de energía perdida", "La fracción disipada es la diferencia relativa de energías; algebraicamente es $M/(m+M)$.",
             [rf"\dfrac{{\Delta K}}{{K_0}}=1-\dfrac{{K_1}}{{K_0}}=1-\dfrac{{{k1_:.4f}}}{{{k0:.3f}}}={an['loss_fraction']:.5f}",
              rf"\dfrac{{M}}{{m+M}}=\dfrac{{2}}{{2.020}}={an['loss_fraction'] * 100:.2f}\%"]),
            ("Ángulo de la cuerda", "La altura se relaciona con el ángulo máximo mediante la geometría de la cuerda de longitud $L=1.2$ m.",
             [r"h=L(1-\cos\beta)\;\Rightarrow\;\cos\beta=1-\dfrac{h}{L}", rf"\cos\beta=1-\dfrac{{{an['h']:.5f}}}{{1.2}}={cosb:.5f}\;\Rightarrow\;\beta={beta:.3f}^\circ"]),
            ("Comprobación inversa", "A partir de la altura se recupera la rapidez de la bala, que es lo que haría el experimentador.",
             [rf"v=\dfrac{{m+M}}{{m}}\sqrt{{2gh}}=\dfrac{{2.020}}{{0.020}}\sqrt{{2({g})({an['h']:.5f})}}={an['v_from_h']:.3f}\ \mathrm{{m/s}}"]),
        ],
        numeric_steps=[
            ("Fase 1 en el código", "La colisión se trata con la fórmula exacta del momento (no se integra), y se obtiene la velocidad inicial del péndulo. Con esa velocidad se integra solo el ascenso.",
             [rf"V=\dfrac{{mv}}{{m+M}}={nu['V']:.5f}\ \mathrm{{m/s}},\qquad \dot\theta_0=\dfrac{{V}}{{L}}=\dfrac{{{V:.5f}}}{{1.2}}={V / L:.5f}\ \mathrm{{rad/s}}"]),
            ("Ecuación del péndulo como sistema de primer orden", "Se integra el péndulo rígido completo, sin aproximar $\\sin\\theta\\approx\\theta$. Con $\\mathbf y=(\\theta,\\omega)$:",
             [rf"\dot\theta=\omega,\qquad \dot\omega=-\dfrac{{g}}{{L}}\sin\theta,\qquad \dfrac{{g}}{{L}}={g / L:.4f}\ \mathrm{{s^{{-2}}}},\qquad h={sci(h, 1)}\ \mathrm{{s}}"]),
            ("Primer paso RK4: pendientes", "Partiendo de $\\mathbf y_0=(0,\\,V/L)$ se calculan las cuatro pendientes. La aceleración angular inicial es nula porque $\\sin0=0$; las siguientes ya sienten la inclinación.",
             [rf"k_1={vec(a1, 6)},\qquad k_2={vec(a2, 6)}",
              rf"k_3={vec(a3, 6)},\qquad k_4={vec(a4, 6)}"]),
            ("Primer paso RK4: nuevo estado", "Se promedian las pendientes con pesos $1,2,2,1$.",
             [r"\mathbf y_1=\mathbf y_0+\dfrac{h}{6}(k_1+2k_2+2k_3+k_4)", rf"\mathbf y_1=({y1[0]:.9f},\ {y1[1]:.9f})"]),
            ("Seguimiento de la altura", "En cada paso se calcula $h_j=L(1-\\cos\\theta_j)$ y se guarda el máximo. El bucle termina cuando $\\omega<0$, es decir, cuando el péndulo empieza a bajar.",
             [rf"h_1=L(1-\cos\theta_1)={L * (1 - np.cos(y1[0])):.4e}\ \mathrm{{m}}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}" if False else rf"h_1=L(1-\cos\theta_1)={sci(L * (1 - np.cos(y1[0])), 4)}\ \mathrm{{m}}",
              rf"\text{{pasos hasta }}\omega<0:\ {n}\quad(t={n * h:.4f}\ \mathrm{{s}})"]),
            ("Resultado numérico de la altura", "El máximo registrado es la altura numérica; difiere de $V^2/(2g)$ solo por la discretización del tiempo (el máximo se detecta con resolución $h$).",
             [rf"h_{{num}}={nu['h']:.7f}\ \mathrm{{m}},\qquad h_{{an}}={an['h']:.7f}\ \mathrm{{m}},\qquad |\Delta h|={sci(abs(nu['h'] - an['h']), 2)}\ \mathrm{{m}}"]),
            ("Fracción de energía perdida", "El código calcula las energías antes y después de la colisión y forma $1-K_1/K_0$.",
             [rf"1-\dfrac{{K_1}}{{K_0}}={nu['loss_fraction']:.6f},\qquad \dfrac{{M}}{{m+M}}={an['loss_fraction']:.6f},\qquad |\Delta|={sci(abs(nu['loss_fraction'] - an['loss_fraction']), 2)}"]),
        ],
    )


# ========================================================================== CAP. 11
def _spinup() -> dict:
    a, tf, r = c11.ALPHA, c11.T_END, c11.RIM
    rim = c11.rim_accelerations(tf)
    t, w, th = c11.spin_up_numeric()
    n = len(t) - 1
    dt = tf / n
    t2, w2, th2 = c11.spin_up_numeric_variable()
    dt2 = tf / (len(t2) - 1)
    ex = tf**3 / 24
    rows = [[str(i), fx(t[i], 2), fx(w[i], 4), fx(th[i], 5)] for i in (0, 1, 2, 3, n)]
    return dict(
        explanation=[
            "La cinemática rotacional describe un cuerpo que gira alrededor de un eje fijo con las mismas ideas que el movimiento rectilíneo, cambiando posición, velocidad y aceleración por ángulo $\\theta$, velocidad angular $\\omega$ y aceleración angular $\\alpha$. Si $\\alpha$ es constante, valen las mismas ecuaciones del movimiento uniformemente acelerado.",
            "Cada punto de la rueda recorre una circunferencia cuyo radio es su distancia al eje. Su rapidez lineal es $v=\\omega r$, su aceleración tangencial $a_t=\\alpha r$ cambia el módulo de la velocidad, y su aceleración centrípeta $a_c=\\omega^2r$ cambia su dirección. Las dos son perpendiculares, de modo que la aceleración total es $\\sqrt{a_t^2+a_c^2}$.",
            "Un comentario de orden de magnitud: tras 8 s la rueda gira a 12 rad/s (unas 115 rpm) y el borde pasa de 4.8 m/s. La aceleración tangencial es modesta (0.6 m/s²), pero la centrípeta crece con el cuadrado de $\\omega$ y llega a 57.6 m/s², casi seis veces $g$; por eso al final la aceleración del borde es casi completamente radial.",
            "Numéricamente, el ángulo y la velocidad angular se obtienen integrando la aceleración angular. Es un buen ejemplo de integración encadenada: se integra $\\alpha(t)$ para obtener $\\omega(t)$ y luego se integra $\\omega(t)$ para obtener $\\theta(t)$.",
        ],
        analytic_steps=[
            ("Datos y ecuaciones", "La rueda parte del reposo ($\\omega_0=0$, $\\theta_0=0$) con aceleración angular constante, así que se aplican las ecuaciones de cinemática con $\\alpha$ fija.",
             [rf"\alpha={a}\ \mathrm{{rad/s^2}},\quad t={tf:.0f}\ \mathrm{{s}},\quad r={r}\ \mathrm{{m}}", r"\omega=\omega_0+\alpha t,\qquad \theta=\theta_0+\omega_0t+\tfrac12\alpha t^2"]),
            ("Velocidad angular final", "Se sustituye el tiempo y la aceleración.",
             [rf"\omega=0+({a})({tf:.0f})={c11.omega_analytic(tf):.1f}\ \mathrm{{rad/s}}"]),
            ("Ángulo girado", "Con $\\omega_0=0$ solo queda el término cuadrático. Dividiendo entre $2\\pi$ se obtienen las vueltas.",
             [rf"\theta=\tfrac12({a})({tf:.0f})^2={c11.theta_analytic(tf):.1f}\ \mathrm{{rad}}={c11.theta_analytic(tf) / (2 * np.pi):.3f}\ \text{{vueltas}}"]),
            ("Rapidez del borde", "La rapidez lineal de un punto del borde es el producto de la velocidad angular por el radio.",
             [rf"v=\omega r=({c11.omega_analytic(tf):.1f})({r})={rim['v']:.2f}\ \mathrm{{m/s}}"]),
            ("Aceleración tangencial", "No depende del tiempo porque $\\alpha$ es constante.",
             [rf"a_t=\alpha r=({a})({r})={rim['a_t']:.2f}\ \mathrm{{m/s^2}}"]),
            ("Aceleración centrípeta", "Depende de $\\omega^2$, por lo que crece rápidamente.",
             [rf"a_c=\omega^2r=({c11.omega_analytic(tf):.1f})^2({r})={rim['a_c']:.1f}\ \mathrm{{m/s^2}}"]),
            ("Aceleración total", "Las componentes son perpendiculares, así que se suman cuadráticamente.",
             [rf"|\vec a|=\sqrt{{a_t^2+a_c^2}}=\sqrt{{({rim['a_t']:.2f})^2+({rim['a_c']:.1f})^2}}={rim['a']:.3f}\ \mathrm{{m/s^2}}"]),
        ],
        numeric_steps=[
            ("Malla de tiempo", "El intervalo $[0,8]$ s se divide en $N=400$ pasos iguales. Las incógnitas son $\\omega_k$ y $\\theta_k$ en cada instante $t_k=k\\,\\Delta t$.",
             [rf"\Delta t=\dfrac{{{tf:.0f}}}{{{n}}}={dt:.2f}\ \mathrm{{s}},\qquad t_k=k\,\Delta t"]),
            ("Regla del trapecio acumulada", "La integral de una función sobre un paso se aproxima por el área del trapecio con los valores en los extremos; las áreas se acumulan con una suma corrida.",
             [r"\omega_{k+1}=\omega_k+\tfrac12(\alpha_k+\alpha_{k+1})\Delta t,\qquad \theta_{k+1}=\theta_k+\tfrac12(\omega_k+\omega_{k+1})\Delta t"]),
            ("Primer paso a mano", "Con $\\omega_0=\\theta_0=0$ y $\\alpha=1.5$ en todos los puntos:",
             [rf"\omega_1=0+\tfrac12({a}+{a})({dt:.2f})={w[1]:.4f}\ \mathrm{{rad/s}}",
              rf"\theta_1=0+\tfrac12(0+{w[1]:.4f})({dt:.2f})={th[1]:.6f}\ \mathrm{{rad}}"]),
            ("Siguientes pasos", "Se repite la misma fórmula. Como $\\alpha$ es constante, el trapecio es exacto para $\\omega$ y el único error en $\\theta$ aparece porque $\\omega$ es lineal (el trapecio también es exacto).",
             [table(["k", "t_k\\ (\\mathrm{s})", r"\omega_k\ (\mathrm{rad/s})", r"\theta_k\ (\mathrm{rad})"], rows)]),
            ("Resultado final con N = 400", "El último punto de la malla corresponde a $t=8$ s.",
             [rf"\omega_{{400}}={w[-1]:.6f}\ \mathrm{{rad/s}},\qquad \theta_{{400}}={th[-1]:.6f}\ \mathrm{{rad}}",
              rf"|\omega-\omega_{{an}}|={sci(abs(w[-1] - c11.omega_analytic(tf)), 2)},\qquad |\theta-\theta_{{an}}|={sci(abs(th[-1] - c11.theta_analytic(tf)), 2)}"]),
            ("Aceleración variable: prueba del método", "Para comprobar que el método no solo funciona por casualidad, se repite con $\\alpha(t)=0.25\\,t$ y $N=4000$ ($\\Delta t=0.002$ s). Analíticamente $\\omega=0.125\\,t^2$ y $\\theta=t^3/24$.",
             [rf"\omega_1=\tfrac12(0+{0.25 * t2[1]:.4f})({dt2:.3f})={w2[1]:.8f}\ \mathrm{{rad/s}},\qquad \theta_1=\tfrac12(0+{w2[1]:.8f})({dt2:.3f})={sci(th2[1], 4)}\ \mathrm{{rad}}"]),
            ("Error con aceleración variable", "El trapecio tiene error de orden $\\Delta t^2$, así que ahora sí aparece una diferencia pequeña, que se achica al refinar la malla.",
             [rf"\theta_{{4000}}={th2[-1]:.6f}\ \mathrm{{rad}},\qquad \theta_{{an}}=\dfrac{{8^3}}{{24}}={ex:.6f}\ \mathrm{{rad}},\qquad |\Delta\theta|={sci(abs(th2[-1] - ex), 2)}\ \mathrm{{rad}}"]),
        ],
    )


def _cycloid() -> dict:
    R = c11.R_WHEEL
    la, aa = c11.arch_length_analytic(), c11.arch_area_analytic()
    ln, an_ = c11.arch_length_numeric(), c11.arch_area_numeric()
    lp = c11.arch_length_polyline()
    n4 = 4
    th4 = np.linspace(0, 2 * np.pi, n4 + 1)
    h4 = 2 * np.pi / n4
    sp4 = 2 * R * np.abs(np.sin(th4 / 2))
    ar4 = (R * (1 - np.cos(th4))) ** 2
    l4, a4 = simpson(sp4, th4), simpson(ar4, th4)
    n = 2000
    dth = 2 * np.pi / n
    x0, y0 = c11.cycloid(np.array([0.0, dth]), R)
    seg1 = float(np.hypot(x0[1] - x0[0], y0[1] - y0[0]))
    return dict(
        explanation=[
            "Rodar sin deslizar significa que el punto de la rueda en contacto con el suelo está momentáneamente en reposo. Eso liga el movimiento de traslación y el de rotación: el centro avanza $v_{cm}=\\omega R$ y, en un giro completo, recorre una distancia igual a la circunferencia $2\\pi R$.",
            "Un punto del borde combina ambos movimientos: avanza con el centro y a la vez gira alrededor de él. Su trayectoria es la cicloide, $x=R(\\theta-\\sin\\theta)$ e $y=R(1-\\cos\\theta)$, que toca el suelo con velocidad nula (cúspide) y alcanza el doble de la rapidez del centro, $2v_{cm}$, en el punto más alto.",
            "Dos resultados clásicos describen un arco de cicloide: su longitud es $8R$, es decir, cuatro diámetros, y el área bajo él es $3\\pi R^2$, tres veces el área de la rueda. Ambos pueden obtenerse con cálculo sencillo y no dependen de la velocidad de rodadura, solo del radio.",
            "Para el cálculo numérico es útil el parámetro $\\theta$: la rapidez de la curva es $|d\\vec r/d\\theta|=2R\\sin(\\theta/2)$, de modo que la longitud es una integral de una función suave, adecuada para la regla de Simpson. Una alternativa independiente es aproximar la curva por una poligonal y sumar la longitud de sus segmentos.",
        ],
        analytic_steps=[
            ("Parametrizar la cicloide", "Se toma como parámetro el ángulo girado $\\theta$ y se usa la condición de rodadura: el centro está en $x_c=R\\theta$. El punto del borde se desplaza respecto del centro en $(-R\\sin\\theta,\\,-R\\cos\\theta)$.",
             [r"x=R(\theta-\sin\theta),\qquad y=R(1-\cos\theta),\qquad R=0.50\ \mathrm{m}"]),
            ("Derivar respecto del parámetro", "Las derivadas dan las componentes de la velocidad por unidad de ángulo.",
             [r"\dfrac{dx}{d\theta}=R(1-\cos\theta),\qquad \dfrac{dy}{d\theta}=R\sin\theta"]),
            ("Rapidez de la curva", "Se forma el módulo y se simplifica con la identidad $1-\\cos\\theta=2\\sin^2(\\theta/2)$.",
             [r"\left|\dfrac{d\vec r}{d\theta}\right|=R\sqrt{(1-\cos\theta)^2+\sin^2\theta}=R\sqrt{2(1-\cos\theta)}=2R\sin\dfrac{\theta}{2}"]),
            ("Integrar la longitud", "Un arco completo corresponde a $\\theta$ de $0$ a $2\\pi$. La integral de $\\sin(\\theta/2)$ es $-2\\cos(\\theta/2)$.",
             [r"s=\int_0^{2\pi}2R\sin\dfrac{\theta}{2}\,d\theta=2R\left[-2\cos\dfrac{\theta}{2}\right]_0^{2\pi}=2R(2+2)=8R",
              rf"s=8(0.50)={la:.3f}\ \mathrm{{m}}"]),
            ("Plantear el área", "El área bajo la curva es $\\int y\\,dx$; se cambia a la variable $\\theta$ con $dx=R(1-\\cos\\theta)d\\theta$.",
             [r"A=\int y\,dx=\int_0^{2\pi}R(1-\cos\theta)\,R(1-\cos\theta)\,d\theta=R^2\int_0^{2\pi}(1-2\cos\theta+\cos^2\theta)\,d\theta"]),
            ("Evaluar la integral", "Los términos en $\\cos\\theta$ se anulan en un periodo y $\\int_0^{2\\pi}\\cos^2\\theta\\,d\\theta=\\pi$.",
             [r"A=R^2\left(2\pi-0+\pi\right)=3\pi R^2"]),
            ("Sustituir el radio", "Con $R=0.50$ m:",
             [rf"A=3\pi(0.50)^2=3\pi(0.25)={aa:.5f}\ \mathrm{{m^2}}"]),
        ],
        numeric_steps=[
            ("Elegir la regla y la malla", "Se integra con Simpson sobre $\\theta\\in[0,2\\pi]$ con $N$ subintervalos pares. Para verlo a mano se usa $N=4$ y después se da el resultado con el $N=2000$ real del código.",
             [rf"h=\dfrac{{2\pi}}{{N}}=\dfrac{{2\pi}}{{4}}={h4:.5f},\qquad \int f\,d\theta\approx\dfrac{{h}}{{3}}\left[f_0+4f_1+2f_2+4f_3+f_4\right]"]),
            ("Tabla de valores para N = 4", "Se evalúan los dos integrandos: la rapidez $2R\\sin(\\theta/2)$ para la longitud, y $y\\,dx/d\\theta=R^2(1-\\cos\\theta)^2$ para el área.",
             [table([r"\theta_i", "0", r"\pi/2", r"\pi", r"3\pi/2", r"2\pi"],
                    [[r"2R\sin(\theta_i/2)"] + [fx(v, 5) for v in sp4], [r"R^2(1-\cos\theta_i)^2"] + [fx(v, 5) for v in ar4]])]),
            ("Longitud con N = 4", "Se aplican los pesos $1,4,2,4,1$.",
             [r"s\approx" + simpson_terms(sp4, h4) + "=" + fx(l4, 5) + r"\ \mathrm{m}",
              rf"|s-8R|={abs(l4 - la):.5f}\ \mathrm{{m}}\ \ (\text{{con solo 4 intervalos}})"]),
            ("Área con N = 4", "Misma regla sobre el segundo integrando. La función es un polinomio trigonométrico y con solo 4 intervalos aún hay error visible.",
             [r"A\approx" + simpson_terms(ar4, h4) + "=" + fx(a4, 5) + r"\ \mathrm{m^2}",
              rf"|A-3\pi R^2|={abs(a4 - aa):.5f}\ \mathrm{{m^2}}"]),
            ("Refinar a N = 2000", "El código usa $h=2\\pi/2000$. El error de Simpson decrece como $h^4$, por lo que con 2000 intervalos queda por debajo de la precisión mostrada.",
             [rf"h={sci(dth, 5)},\qquad s_{{Simpson}}={ln:.8f}\ \mathrm{{m}},\qquad A_{{Simpson}}={an_:.8f}\ \mathrm{{m^2}}",
              rf"|s-8R|={sci(abs(ln - la), 2)}\ \mathrm{{m}},\qquad |A-3\pi R^2|={sci(abs(an_ - aa), 2)}\ \mathrm{{m^2}}"]),
            ("Método independiente: poligonal", "Se evalúan $2001$ puntos de la cicloide y se suma la distancia entre puntos consecutivos. El primer segmento va de $\\theta=0$ a $\\theta=\\Delta\\theta$:",
             [rf"\Delta x=R(\Delta\theta-\sin\Delta\theta)={sci(float(x0[1] - x0[0]), 4)},\quad \Delta y=R(1-\cos\Delta\theta)={sci(float(y0[1] - y0[0]), 4)}",
              rf"\ell_1=\sqrt{{\Delta x^2+\Delta y^2}}={sci(seg1, 5)}\ \mathrm{{m}}"]),
            ("Suma de la poligonal y error", "La poligonal siempre subestima la longitud (las cuerdas son más cortas que el arco), con error de orden $h^2$.",
             [rf"s_{{pol}}=\sum_{{i=1}}^{{2000}}\ell_i={lp:.8f}\ \mathrm{{m}},\qquad s_{{an}}-s_{{pol}}={sci(la - lp, 3)}\ \mathrm{{m}}"]),
        ],
    )


# ========================================================================== CAP. 12
def _inertia() -> dict:
    m, L = c12.M_ROD, c12.L_ROD
    ie, ic = c12.rod_inertia_analytic("end"), c12.rod_inertia_analytic("center")
    ien, icn = c12.rod_inertia_numeric("end"), c12.rod_inertia_numeric("center")
    md, rd = 4.0, 0.25
    idc = c12.disc_inertia_analytic()
    idn = c12.disc_inertia_numeric()
    ipa = c12.parallel_axis(icn, m, L / 2)
    n4 = 4
    x4 = np.linspace(0, L, n4 + 1)
    h4 = L / n4
    lam = m / L
    f_end = x4**2 * lam
    f_cen = (x4 - L / 2) ** 2 * lam
    s4 = np.linspace(0, rd, n4 + 1)
    sig = md / (np.pi * rd**2)
    f_disc = s4**2 * sig * 2 * np.pi * s4
    hd = rd / n4
    i_end4, i_cen4, i_disc4 = simpson(f_end, x4), simpson(f_cen, x4), simpson(f_disc, s4)
    return dict(
        explanation=[
            "El momento de inercia es la versión rotacional de la masa: mide cuánto se resiste un cuerpo a cambiar su velocidad angular. A diferencia de la masa, no depende solo de cuánta materia hay sino de dónde está: un elemento de masa aporta $dm\\,r^2$, de modo que lo que está lejos del eje cuenta mucho más que lo que está cerca.",
            "Por eso $I$ depende del eje elegido. La misma barra tiene cuatro veces más inercia al girar alrededor de un extremo que alrededor de su centro: $ML^2/3$ frente a $ML^2/12$. El teorema de los ejes paralelos conecta ambos casos: $I=I_{cm}+Md^2$, donde $d$ es la distancia entre el eje por el centro de masa y el eje paralelo; el centro de masa es siempre el eje de menor inercia.",
            "Para cuerpos continuos se pasa de la suma a la integral $I=\\int r^2\\,dm$ eligiendo un elemento de masa conveniente. En la barra se usan trozos $dm=(M/L)\\,dx$; en el disco, anillos delgados de radio $s$ y ancho $ds$ con $dm=\\sigma\\,2\\pi s\\,ds$, porque todos los puntos de un anillo están a la misma distancia del eje.",
            "Como orden de magnitud, el disco de 4 kg y 25 cm tiene $I=0.125$ kg m², menor que $MR^2=0.25$ kg m² de un aro de igual masa y radio, porque en el disco parte de la masa se acerca al eje. La forma de la distribución importa tanto como la masa.",
        ],
        analytic_steps=[
            ("Densidad lineal de la barra", "La barra es uniforme, así que la masa por unidad de longitud es constante. Es el factor que convierte $dm$ en $dx$.",
             [rf"\lambda=\dfrac{{M}}{{L}}=\dfrac{{{m:.0f}}}{{{L}}}={lam:.3f}\ \mathrm{{kg/m}},\qquad dm=\lambda\,dx"]),
            ("Eje en el extremo", "Con el origen en el eje, la distancia de cada trozo es $x$.",
             [r"I_{ext}=\int_0^L x^2\lambda\,dx=\lambda\dfrac{L^3}{3}=\dfrac{ML^2}{3}", rf"I_{{ext}}=\dfrac{{({m:.0f})({L})^2}}{{3}}={ie:.4f}\ \mathrm{{kg\,m^2}}"]),
            ("Eje en el centro", "Ahora la distancia es $x-L/2$, y se integra de $0$ a $L$ (equivale a integrar de $-L/2$ a $L/2$).",
             [r"I_{cm}=\int_{-L/2}^{L/2}u^2\lambda\,du=\lambda\dfrac{2}{3}\left(\dfrac{L}{2}\right)^3=\dfrac{ML^2}{12}", rf"I_{{cm}}=\dfrac{{({m:.0f})({L})^2}}{{12}}={ic:.4f}\ \mathrm{{kg\,m^2}}"]),
            ("Verificar con ejes paralelos", "El eje del extremo está a $d=L/2=0.6$ m del centro de la barra. Debe cumplirse $I_{ext}=I_{cm}+Md^2$.",
             [rf"I_{{cm}}+M\left(\dfrac{{L}}{{2}}\right)^2={ic:.4f}+({m:.0f})(0.6)^2={ic:.4f}+{m * 0.36:.4f}={ic + m * 0.36:.4f}\ \mathrm{{kg\,m^2}}={ie:.4f}"]),
            ("Densidad superficial del disco", "El disco es uniforme: la masa se reparte en el área $\\pi R^2$.",
             [rf"\sigma=\dfrac{{M}}{{\pi R^2}}=\dfrac{{{md:.0f}}}{{\pi(0.25)^2}}={sig:.4f}\ \mathrm{{kg/m^2}}"]),
            ("Integrar anillos", "Un anillo de radio $s$ tiene área $2\\pi s\\,ds$ y todos sus puntos están a distancia $s$ del eje.",
             [r"I=\int_0^R s^2\,\sigma\,2\pi s\,ds=2\pi\sigma\dfrac{R^4}{4}=\dfrac{1}{2}MR^2"]),
            ("Sustituir en el disco", "Con $M=4$ kg y $R=0.25$ m:",
             [rf"I_{{disco}}=\tfrac12({md:.0f})(0.25)^2={idc:.5f}\ \mathrm{{kg\,m^2}}"]),
        ],
        numeric_steps=[
            ("Plantear la integral discreta", "El código aproxima $I=\\int r^2\\,dm$ con la regla de Simpson sobre el integrando $r^2\\lambda$ (barra) o $s^2\\sigma2\\pi s$ (disco). Con $N$ subintervalos pares y ancho $h$:",
             [r"I\approx\dfrac{h}{3}\left[f_0+4f_1+2f_2+\cdots+4f_{N-1}+f_N\right]"]),
            ("Barra con eje en el extremo, N = 4", "Para ver el mecanismo a mano se toman 4 intervalos de ancho $h=0.3$ m y $f(x)=x^2\\lambda$.",
             [table(["x_i", "0", "0.3", "0.6", "0.9", "1.2"], [[r"f_i=x_i^2\lambda"] + [fx(v, 5) for v in f_end]]),
              r"I\approx" + simpson_terms(f_end, h4) + "=" + fx(i_end4, 5) + r"\ \mathrm{kg\,m^2}"]),
            ("Barra con eje en el centro, N = 4", "Ahora $f(x)=(x-0.6)^2\\lambda$. Simpson integra exactamente polinomios hasta grado 3, de modo que ya con 4 intervalos coincide con el valor analítico.",
             [table(["x_i", "0", "0.3", "0.6", "0.9", "1.2"], [[r"f_i"] + [fx(v, 5) for v in f_cen]]),
              r"I\approx" + simpson_terms(f_cen, h4) + "=" + fx(i_cen4, 5) + r"\ \mathrm{kg\,m^2}"]),
            ("Disco por anillos, N = 4", "Se toman 4 anillos entre $s=0$ y $s=0.25$ m, con $f(s)=s^2\\sigma\\,2\\pi s$ (cúbica, también exacta con Simpson).",
             [table(["s_i", "0", "0.0625", "0.125", "0.1875", "0.25"], [[r"f_i"] + [fx(v, 5) for v in f_disc]]),
              r"I\approx" + simpson_terms(f_disc, hd) + "=" + fx(i_disc4, 6) + r"\ \mathrm{kg\,m^2}"]),
            ("Valores con N = 400", "El código repite el mismo cálculo con $h=L/400=0.003$ m (barra) y $h=R/400$ (disco).",
             [rf"I_{{ext}}={ien:.8f},\quad I_{{cm}}={icn:.8f},\quad I_{{disco}}={idn:.8f}\ \ (\mathrm{{kg\,m^2}})"]),
            ("Error respecto del analítico", "Como los integrandos son polinomios de grado $\\le3$, el error de Simpson es cero salvo el redondeo de punto flotante.",
             [rf"|I_{{ext}}-I_{{an}}|={sci(abs(ien - ie), 2)},\quad |I_{{cm}}-I_{{an}}|={sci(abs(icn - ic), 2)},\quad |I_{{disco}}-I_{{an}}|={sci(abs(idn - idc), 2)}"]),
            ("Comprobación numérica de ejes paralelos", "El código suma $I_{cm}^{num}+M(L/2)^2$ y lo compara con la integral directa respecto del extremo.",
             [rf"I_{{cm}}^{{num}}+M(L/2)^2={icn:.8f}+{m * (L / 2) ** 2:.4f}={ipa:.8f}\ \mathrm{{kg\,m^2}},\qquad |{ipa:.8f}-I_{{ext}}^{{num}}|={sci(abs(ipa - ien), 2)}"]),
        ],
    )


def _hanging() -> dict:
    m, M, r, g = c12.M_HANG, c12.M_DISC, c12.R_DISC, c12.G
    I = 0.5 * M * r**2
    an, nu = c12.hanging_analytic(), c12.hanging_numeric()
    mat = np.array([[m, 1.0], [I / r, -r]])
    rhs_v = np.array([m * g, 0.0])
    det = np.linalg.det(mat)
    sol = np.linalg.solve(mat, rhs_v)
    h = 1.0 / 1000
    alpha = an["alpha"]
    rhs = lambda _t, s: np.array([s[1], an["a"], s[3], an["a"] / r])  # noqa: E731
    k1, k2, k3, k4, s1 = rk4_ks(rhs, 0.0, np.zeros(4), h)
    yf = nu["y"][-1]
    e_in = m * g * yf[0]
    e_out = 0.5 * m * yf[1] ** 2 + 0.5 * I * yf[3] ** 2
    return dict(
        explanation=[
            "Este sistema une un cuerpo que se traslada (la masa colgante) con un cuerpo que gira (el disco). La cuerda, que no desliza ni se estira, los obliga a moverse de forma acoplada: cuando la masa baja una distancia $y$, el borde del disco gira ese mismo arco, y la aceleración lineal y la angular quedan ligadas por $a=\\alpha R$.",
            "Se aplica la segunda ley a cada cuerpo por separado. Para la masa, $mg-T=ma$: el peso menos la tensión da la aceleración. Para el disco, la tensión actúa en el borde y produce el torque $TR=I\\alpha$. Es esencial entender que la tensión no es $mg$: si lo fuera, la masa no aceleraría, y como el disco tiene inercia, $T<mg$.",
            "Combinando las ecuaciones aparece una masa efectiva: el disco se comporta, para el movimiento de la cuerda, como una masa adicional $I/R^2=M/2$ que debe acelerarse junto con $m$. Así $a=mg/(m+M/2)$. Con un disco sin inercia se recuperaría $a=g$ (caída libre); un disco muy pesado hace que la masa descienda lentamente.",
            "Con los datos, $m=2$ kg y $M/2=2$ kg dan exactamente la mitad de $g$: $a=4.905$ m/s². La tensión resulta $T=9.81$ N, justo la mitad del peso de 19.62 N. Una verificación independiente es la energía: la energía potencial perdida debe repartirse entre la energía cinética de traslación y la de rotación.",
        ],
        analytic_steps=[
            ("Datos y momento de inercia", "El disco es uniforme, con $I=\\tfrac12MR^2$ respecto de su eje.",
             [rf"m={m:.0f}\ \mathrm{{kg}},\quad M={M:.0f}\ \mathrm{{kg}},\quad R={r}\ \mathrm{{m}},\quad I=\tfrac12({M:.0f})({r})^2={I:.4f}\ \mathrm{{kg\,m^2}}"]),
            ("Segunda ley para la masa colgante", "Tomando positivo hacia abajo, actúan el peso y la tensión.",
             [r"mg-T=ma"]),
            ("Segunda ley para el disco", "Solo la tensión produce torque respecto del eje (el peso del disco y la reacción del eje pasan por él).",
             [r"TR=I\alpha"]),
            ("Condición de cuerda sin deslizar", "El punto del borde y la masa tienen la misma aceleración tangencial. Con $I=\\tfrac12MR^2$ el torque da una tensión simple.",
             [r"a=\alpha R\;\Rightarrow\;T=\dfrac{I\alpha}{R}=\dfrac{I\,a}{R^2}=\dfrac{M}{2}\,a"]),
            ("Despejar la aceleración", "Se sustituye $T=\\tfrac{M}{2}a$ en la ecuación de la masa y se agrupan términos en $a$.",
             [r"mg=\left(m+\dfrac{M}{2}\right)a\;\Rightarrow\;a=\dfrac{mg}{m+M/2}", rf"a=\dfrac{{({m:.0f})({g})}}{{{m:.0f}+{M / 2:.0f}}}={an['a']:.4f}\ \mathrm{{m/s^2}}"]),
            ("Tensión y aceleración angular", "Se evalúa la tensión con $T=m(g-a)$ y se divide la aceleración entre el radio.",
             [rf"T=m(g-a)=({m:.0f})({g}-{an['a']:.4f})={an['T']:.4f}\ \mathrm{{N}}",
              rf"\alpha=\dfrac{{a}}{{R}}=\dfrac{{{an['a']:.4f}}}{{{r}}}={alpha:.3f}\ \mathrm{{rad/s^2}}"]),
            ("Comprobar con el torque", "La tensión por el radio debe igualar $I\\alpha$.",
             [rf"TR={an['T']:.4f}({r})={an['T'] * r:.5f}\ \mathrm{{N\,m}},\qquad I\alpha=({I:.4f})({alpha:.3f})={I * alpha:.5f}\ \mathrm{{N\,m}}"]),
        ],
        numeric_steps=[
            ("Escribir el sistema lineal 2x2", "Las incógnitas son la aceleración $a$ y la tensión $T$. Las dos ecuaciones se arreglan con las incógnitas a la izquierda; la segunda se obtiene de $T R=I\\,a/R$.",
             [r"\begin{cases} m\,a+T=mg\\ \dfrac{I}{R}\,a-R\,T=0\end{cases}\quad\Rightarrow\quad \begin{pmatrix} m&1\\ I/R&-R\end{pmatrix}\begin{pmatrix}a\\T\end{pmatrix}=\begin{pmatrix}mg\\0\end{pmatrix}",
              rf"\begin{{pmatrix}} {m:.0f}&1\\ {I / r:.2f}&{-r:.2f}\end{{pmatrix}}\begin{{pmatrix}}a\\T\end{{pmatrix}}=\begin{{pmatrix}}{m * g:.2f}\\0\end{{pmatrix}}"]),
            ("Resolver a mano por la regla de Cramer", "El determinante de la matriz es $\\Delta=-mR-I/R$. Cada incógnita es un cociente de determinantes.",
             [rf"\Delta=({m:.0f})({-r:.2f})-(1)({I / r:.2f})={det:.4f}",
              rf"a=\dfrac{{(mg)(-R)-(1)(0)}}{{\Delta}}=\dfrac{{({m * g:.2f})({-r:.2f})}}{{{det:.4f}}}={sol[0]:.5f}\ \mathrm{{m/s^2}}",
              rf"T=\dfrac{{m(0)-(I/R)(mg)}}{{\Delta}}=\dfrac{{-({I / r:.2f})({m * g:.2f})}}{{{det:.4f}}}={sol[1]:.5f}\ \mathrm{{N}}"]),
            ("Qué hace numpy.linalg.solve", "El código llama a esta rutina, que realiza eliminación gaussiana con pivoteo parcial (descomposición LU). Reproduce el resultado de la regla de Cramer a precisión de máquina.",
             [rf"a_{{num}}={nu['a']:.6f}\ \mathrm{{m/s^2}},\qquad T_{{num}}={nu['T']:.6f}\ \mathrm{{N}}",
              rf"|a_{{num}}-a_{{an}}|={sci(abs(nu['a'] - an['a']), 2)},\qquad |T_{{num}}-T_{{an}}|={sci(abs(nu['T'] - an['T']), 2)}"]),
            ("Integrar el movimiento: ecuaciones de estado", "Con $a$ conocida en cada instante se integra el estado $\\mathbf y=(y,v,\\theta,\\omega)$ durante 1 s con $N=1000$ pasos, usando la cuerda inextensible como ligadura ($\\dot\\omega=a/R$).",
             [rf"\dot y=v,\quad \dot v=a,\quad \dot\theta=\omega,\quad \dot\omega=\dfrac{{a}}{{R}},\qquad h={sci(h, 1)}\ \mathrm{{s}},\quad \mathbf y_0=\mathbf 0"]),
            ("Primer paso RK4: pendientes", "Como $a$ es constante las pendientes de velocidad son idénticas; las de posición cambian ligeramente porque la velocidad crece dentro del paso.",
             [rf"k_1={vec(k1, 6)},\qquad k_2={vec(k2, 6)}",
              rf"k_3={vec(k3, 6)},\qquad k_4={vec(k4, 6)}"]),
            ("Primer paso RK4: nuevo estado", "Se promedian con pesos $1,2,2,1$. El resultado es exactamente $y=\\tfrac12at^2$, $v=at$, porque RK4 es exacto para polinomios de grado bajo.",
             [rf"\mathbf y_1=\mathbf y_0+\dfrac{{h}}{{6}}(k_1+2k_2+2k_3+k_4)={vec(s1, 6)}"]),
            ("Estado en t = 1 s", "Tras 1000 pasos:",
             [rf"y={yf[0]:.5f}\ \mathrm{{m}},\quad v={yf[1]:.5f}\ \mathrm{{m/s}},\quad \omega={yf[3]:.5f}\ \mathrm{{rad/s}}"]),
            ("Balance de energía", "La energía potencial perdida por la masa debe ser igual a la suma de la energía cinética de traslación y la de rotación del disco. El error mide qué tan bien se cumple en la simulación.",
             [rf"mgy={m:.0f}({g})({yf[0]:.5f})={e_in:.5f}\ \mathrm{{J}}",
              rf"\tfrac12mv^2+\tfrac12I\omega^2={0.5 * m * yf[1] ** 2:.5f}+{0.5 * I * yf[3] ** 2:.5f}={e_out:.5f}\ \mathrm{{J}}",
              rf"\Delta E={sci(nu['energy_error'], 3)}\ \mathrm{{J}}\ \ (\text{{redondeo de máquina}})"]),
        ],
    )


DETAIL = {
    "CenterOfMass": _cm(),
    "ExplodingProjectile": _explode(),
    "Collision1D": _collision(),
    "BallisticPendulum": _pendulum(),
    "SpinUp": _spinup(),
    "RollingCycloid": _cycloid(),
    "MomentOfInertia": _inertia(),
    "HangingMassDisc": _hanging(),
}
