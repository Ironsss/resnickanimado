"""Explicacion del metodo numerico: Volumen 1, capitulos 9 a 12.

Escenas: CenterOfMass, ExplodingProjectile, Collision1D, BallisticPendulum,
SpinUp, RollingCycloid, MomentOfInertia, HangingMassDisc.
Campos: numeric_intro, numeric_code, numeric_converge. Las cifras salen de ejecutar `physics/`.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch09_particle_systems as c9
from physics import ch10_collisions as c10
from physics import ch11_rotational_kinematics as c11
from physics import ch12_rotational_dynamics as c12
from physics.integrators import euler_step, integrate, rk4_step, trapezoid


def table(header: list[str], rows: list[list[str]]) -> str:
    out = r"\begin{array}{" + "c" * len(header) + "}" + " & ".join(header) + r"\\ \hline "
    out += r"\\ ".join(" & ".join(r) for r in rows)
    return out + r"\end{array}"


def slope(h, err) -> float:
    """Pendiente de log(err) contra log(h) por minimos cuadrados."""
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


# ====================================================================== CenterOfMass
def _cm() -> dict:
    xr = c9.rod_cm_analytic()
    ns = [2, 4, 10, 200]
    simp = [abs(c9.rod_cm_numeric(n) - xr) for n in ns]
    mcn = [100, 1000, 10_000, 100_000, 1_000_000]
    rms = []
    for n in mcn:
        e = [c9.rod_cm_monte_carlo(n, seed=s)[0] - xr for s in range(40)]
        rms.append(float(np.sqrt(np.mean(np.square(e)))))
    sig = c9.ROD_L / np.sqrt(18)
    s_mc = slope(mcn, rms)
    one = [abs(c9.rod_cm_monte_carlo(n)[0] - xr) for n in mcn]
    return dict(
        numeric_intro=[
            "El centro de masa de partículas discretas es un promedio ponderado y se calcula con un producto matriz-vector sin aproximación alguna. La barra con densidad $\\lambda(x)=\\lambda_0 x/L$ es distinta: el centro de masa es un cociente de integrales y el programa lo evalúa con dos técnicas numéricas, la regla de Simpson y un estimador Monte Carlo. Se usan dos porque tienen perfiles de error opuestos: una es determinista y de orden alto, la otra es estadística y lenta pero no depende de la dimensión.",
            "Simpson aproxima el integrando por parábolas sobre pares de subintervalos de ancho $h=L/N$: $\\int_a^b f\\,dx\\approx\\frac{h}{3}\\,[f_0+4f_1+2f_2+\\cdots+4f_{N-1}+f_N]$, con $N$ par. Su error global es $-\\frac{(b-a)}{180}h^4 f^{(4)}(\\xi)$, es decir $O(h^4)$ y exacto para polinomios de grado $\\le3$. Aquí $x\\lambda$ y $\\lambda$ son de grado 2 y 1, así que cualquier $N$ par da el resultado exacto salvo redondeo; $N=200$ es simplemente un valor holgado que sería necesario si $\\lambda$ no fuese polinómica.",
            "El estimador Monte Carlo usa la transformada inversa: si $u$ es uniforme en $[0,1)$ entonces $x=L\\sqrt{u}$ tiene densidad proporcional a $x$, y el centro de masa es el valor esperado $\\langle x\\rangle$, que se estima con el promedio $\\bar x=\\frac1N\\sum x_i$. Por el teorema del límite central el error típico es $\\sigma/\\sqrt N$ con $\\sigma=L/\\sqrt{18}$ para esta distribución: cada factor 100 en muestras solo gana un factor 10 en precisión. Con $N=4\\times10^5$ y semilla 5 se espera un error cercano a $10^{-3}$ m, y la semilla fija hace que el resultado sea reproducible.",
            "Ambos métodos son independientes del análisis: este nunca integró $x^2/L$ ni usó la fórmula $2L/3$; el código solo evalúa el integrando en puntos (Simpson) o sortea posiciones con la densidad (Monte Carlo) y compara al final con $2L/3$. Monte Carlo falla cuando la varianza es enorme o la densidad no se puede muestrear; Simpson falla con integrandos no suaves (picos, singularidades), donde su orden $h^4$ ya no se cumple.",
        ],
        numeric_code="""# Sistema discreto: promedio ponderado por producto matricial
cm = (masas @ posiciones) / masas.sum()          # (sum m_i r_i) / M

# Barra con lambda(x) = x/L: cociente de dos integrales
x = linspace(0, L, N + 1)                        # N subintervalos pares, h = L/N
lam = x / L                                      # densidad relativa
def simpson(y, x):
    h = (x[-1] - x[0]) / (len(x) - 1)            # ancho del subintervalo
    return h/3 * (y[0] + y[-1]                   # extremos con peso 1
                  + 4*y[1:-1:2].sum()            # impares con peso 4
                  + 2*y[2:-1:2].sum())           # pares con peso 2
x_cm = simpson(x*lam, x) / simpson(lam, x)       # numerador / denominador

# Monte Carlo con transformada inversa
rng = default_rng(semilla)                       # semilla fija = reproducible
u = rng.uniform(size=n)                          # uniformes en [0, 1)
xs = L * sqrt(u)                                 # F(x)=x^2/L^2 = u  ->  x = L sqrt(u)
x_cm_mc = xs.mean()                              # promedio de muestras
error_tipico = (L / sqrt(18)) / sqrt(n)          # sigma / sqrt(n)""",
        numeric_converge=[
            ("Simpson: exactitud independiente de N", "Se repite la regla con varios $N$ y se compara con $2L/3$. Como el integrando es polinómico de grado $\\le3$, el error no baja como $h^4$ sino que ya vale cero salvo redondeo desde $N=2$; lo que se ve es el piso de precisión de la máquina.",
             [table(["N", r"|x_{cm}-2L/3|\ (\mathrm{m})"], [[str(n), sci(e, 2)] for n, e in zip(ns, simp)])]),
            ("Monte Carlo: el error baja como 1/sqrt(N)", "Para cada $N$ se calcula el error cuadrático medio sobre 40 semillas distintas (así el resultado no depende de la suerte de una sola) y se compara con $\\sigma/\\sqrt N$. La pendiente de $\\log(\\text{error})$ contra $\\log N$ confirma el exponente $-1/2$.",
             [table(["N", r"\text{RMS}\ (\mathrm{m})", r"\sigma/\sqrt{N}\ (\mathrm{m})", r"\text{semilla }5\ (\mathrm{m})"],
                    [[f"{n:,}".replace(",", r"\,"), sci(r, 3), sci(sig / np.sqrt(n), 3), sci(o, 3)] for n, r, o in zip(mcn, rms, one)]),
              rf"\text{{pendiente observada}}={s_mc:.3f}\quad(\text{{teoría }}-0.5)"]),
            ("Fuentes de error y criterio de confianza", "En Simpson el error de truncamiento es nulo para este integrando y solo queda el redondeo, del orden de $10^{-16}$ m. En Monte Carlo el error es estadístico y domina todo: con $N=400\\,000$ vale unos milímetros o menos. Un resultado Monte Carlo es confiable cuando la diferencia entre dos semillas distintas es comparable a $\\sigma/\\sqrt N$ y concuerda con Simpson dentro de ese margen; si hubiera que ganar un dígito habría que multiplicar $N$ por 100.",
             [rf"\dfrac{{\sigma}}{{\sqrt{{N}}}}\Big|_{{N=4\times10^5}}={sci(sig / np.sqrt(4e5), 3)}\ \mathrm{{m}}\ \ll\ x_{{cm}}={xr:.4f}\ \mathrm{{m}}"]),
        ],
    )


# ============================================================== ExplodingProjectile
def _explode() -> dict:
    ref = c9.fragments_analytic()["x_cm"]
    dts = [0.1, 0.02, 0.01, 0.002, 0.001]
    errs = [abs(c9.fragments_numeric(dt=d)["x_cm"] - ref) for d in dts]
    sl = slope(dts, errs)
    ap = c9.apex_state()
    d0 = 1e-3
    tn = c9.fragments_numeric(dt=d0)
    return dict(
        numeric_intro=[
            "Tras la explosión cada fragmento es un proyectil que solo siente la gravedad. Se podría escribir la caída libre en forma cerrada, pero la simulación numérica integra las ecuaciones de movimiento $\\dot x=u,\\ \\dot y=w,\\ \\dot u=0,\\ \\dot w=-g$ con Runge-Kutta de cuarto orden y detecta cuándo cada trozo toca el suelo. Así el centro de masa se reconstruye a partir de dónde caen realmente los fragmentos, sin usar $t=\\sqrt{2y/g}$ ni $x=x_0+v t$.",
            "RK4 avanza el estado con cuatro evaluaciones de la pendiente: $\\mathbf y_{n+1}=\\mathbf y_n+\\frac h6(\\mathbf k_1+2\\mathbf k_2+2\\mathbf k_3+\\mathbf k_4)$, donde $\\mathbf k_1=f(t_n,\\mathbf y_n)$, $\\mathbf k_2$ y $\\mathbf k_3$ usan el punto medio y $\\mathbf k_4$ el extremo. Coincide con el desarrollo de Taylor hasta $h^4$, de modo que el error global es $O(h^4)$. En este problema la aceleración es constante, así que el desarrollo termina antes y RK4 reproduce la parábola exactamente; el único error proviene de la detección del suelo.",
            "El lazo avanza hasta que $y$ cambia de signo y entonces interpola linealmente dentro del último paso para estimar el instante y la posición del impacto. Esa interpolación sustituye un arco de parábola por una recta, lo que introduce un error $O(h^2)$ en el punto de aterrizaje. Con $h=10^{-3}$ s el error en $x_{cm}$ queda del orden de $10^{-6}$ m, mucho menor que los 35 m de distancia recorrida.",
            "El método es independiente del análisis porque no calcula el tiempo de caída ni la velocidad $v_B=(m_A+m_B)v_x/m_B$ como cantidad derivada de una fórmula de caída: solo las usa como condición inicial (conservación del momento en la explosión) y deja que la integración encuentre el aterrizaje. La limitación es el umbral: si $h$ fuera comparable al tiempo de vuelo la interpolación lineal ya no sería válida.",
        ],
        numeric_code="""def caida(estado, dt):                           # estado = (x, y, vx, vy)
    f = lambda t, s: [s[2], s[3], 0.0, -g]       # derivadas: x'=vx, y'=vy, vx'=0, vy'=-g
    ts, ys = [0.0], [estado]
    while ys[-1][1] > 0:                         # avanzar hasta cruzar el suelo
        ys.append(rk4_step(f, ts[-1], ys[-1], dt))   # un paso de RK4
        ts.append(ts[-1] + dt)
    frac = ys[-2][1] / (ys[-2][1] - ys[-1][1])   # fraccion del ultimo paso hasta y = 0
    ys[-1] = ys[-2] + frac * (ys[-1] - ys[-2])   # interpolacion lineal al suelo
    return ys

vb = (mA + mB) * vx_apice / mB                   # momento horizontal conservado
A = caida([x_ap, y_ap, 0.0, 0.0], dt)            # A cae desde el reposo
B = caida([x_ap, y_ap, vb, 0.0], dt)             # B sale con velocidad vb
x_cm = (mA * A[-1][0] + mB * B[-1][0]) / (mA + mB)  # CM reconstruido en el suelo""",
        numeric_converge=[
            ("Error en x_cm al reducir el paso", "Se ejecuta la simulación con cinco pasos y se compara con el centro de masa analítico. El error cae aproximadamente como una potencia de $h$ y la pendiente de ajuste lo cuantifica. No es perfectamente monótono porque depende de dónde cae el último paso respecto del suelo.",
             [table([r"h\ (\mathrm{s})", r"|x_{cm}-x_{cm}^{an}|\ (\mathrm{m})"], [[sci(d, 2), sci(e, 3)] for d, e in zip(dts, errs)]),
              rf"\text{{pendiente de ajuste}}={sl:.2f}\quad(\text{{interpolación lineal: orden }}2)"]),
            ("Por qué no se observa orden 4", "RK4 por sí solo sería $O(h^4)$, pero con aceleración constante es exacto; el error medido es el de la interpolación del último paso. Eso muestra una regla general: el orden de un esquema completo lo fija su eslabón más débil, no su integrador más sofisticado.",
             [rf"x_{{cm}}^{{num}}({sci(d0, 1)}\ \mathrm{{s}})={tn['x_cm']:.6f}\ \mathrm{{m}},\qquad x_{{cm}}^{{an}}={ref:.6f}\ \mathrm{{m}}"]),
            ("Criterio de confianza", "El resultado es fiable cuando al dividir $h$ entre 10 el error baja unas 100 veces (orden 2) y $x_{cm}$ deja de cambiar en la cifra buscada. Aquí el paso es 1000 veces menor que el tiempo de vuelo; el redondeo (del orden de $10^{-14}$ m acumulado en unos miles de pasos) es despreciable frente al truncamiento.",
             [rf"t_{{vuelo}}\approx{tn['A'][0][-1]:.3f}\ \mathrm{{s}}\ \Rightarrow\ \dfrac{{t_{{vuelo}}}}{{h}}\approx{tn['A'][0][-1] / d0:.0f}\ \text{{pasos}}"]),
        ],
    )


# ==================================================================== Collision1D
def _collision() -> dict:
    z = 0.3
    ea = c10.restitution_from_damping(z)
    dts = [4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1e-5]
    errs = [abs(c10.restitution_measured(z, dt=d) - ea) for d in dts]
    sl = slope(dts, errs)
    zs = [0.0, 0.3, 0.7]
    rows = []
    for zz in zs:
        en = c10.restitution_measured(zz, dt=1e-5)
        ean = c10.restitution_from_damping(zz)
        rows.append([fx(zz, 1), fx(ean, 5), fx(en, 5), sci(abs(en - ean), 2)])
    kk, mu = 1.0e4, c10.M1 * c10.M2 / (c10.M1 + c10.M2)
    w0 = np.sqrt(kk / mu)
    return dict(
        numeric_intro=[
            "La fórmula de colisión con coeficiente de restitución $e$ es un postulado: resume qué ocurre durante el contacto sin describirlo. La simulación va un nivel más abajo y modela el contacto como un resorte más un amortiguador lineal, $F=k\\,\\delta+c\\,\\dot\\delta$, con $\\delta=x_1-x_2$ el solape. Integrar esas ecuaciones y medir las velocidades finales permite descubrir $e$ en lugar de suponerlo.",
            "El sistema $\\dot x_i=u_i,\\ m_1\\dot u_1=-F,\\ m_2\\dot u_2=+F$ se resuelve con Runge-Kutta de cuarto orden, $\\mathbf y_{n+1}=\\mathbf y_n+\\frac h6(\\mathbf k_1+2\\mathbf k_2+2\\mathbf k_3+\\mathbf k_4)$, cuyo error global es $O(h^4)$ si el lado derecho es suave. Aquí no lo es: la fuerza se enciende de golpe cuando $\\delta>0$ y se apaga al separarse, de modo que el esquema pierde orden en los pasos que cruzan esos dos instantes y el error efectivo es menor que el teórico.",
            "El amortiguamiento se fija con la razón $\\zeta$ mediante $c=2\\zeta\\sqrt{k\\mu}$ con $\\mu=m_1m_2/(m_1+m_2)$ la masa reducida. La solución del oscilador amortiguado en medio periodo da $e=\\exp(-\\pi\\zeta/\\sqrt{1-\\zeta^2})$, y esa es la predicción contra la que se compara la medición $e=(v_2'-v_1')/v_1$. Esa relación proviene de la ecuación diferencial del contacto, no de la fórmula de colisión, por eso la comparación valida el programa.",
            "Los parámetros se eligieron así: $k=10^4$ N/m es suficientemente rígido para que el contacto sea corto (unos 25 ms) pero no tanto como para exigir pasos diminutos; $h=10^{-5}$ s da miles de pasos dentro del contacto; y el estado inicial se coloca 2 cm antes del choque para que el programa vea primero vuelo libre. Con $h$ demasiado grande el contacto se resolvería con pocos puntos y el instante de separación se detectaría con retraso.",
        ],
        numeric_code="""meff = m1 * m2 / (m1 + m2)                       # masa reducida
c = 2 * zeta * sqrt(k * meff)                    # amortiguamiento a partir de zeta
def rhs(t, s):                                   # s = (x1, x2, u1, u2)
    x1, x2, u1, u2 = s
    solape = x1 - x2                             # >0 significa contacto
    if solape <= 0:
        return [u1, u2, 0, 0]                    # vuelo libre: sin fuerza
    F = k * solape + c * (u1 - u2)               # resorte + amortiguador
    return [u1, u2, -F / m1, +F / m2]            # accion y reaccion

s = [-0.02, 0.0, v1, 0.0]                        # 2 cm antes del contacto
tocó = False
while t < 1.0:                                   # limite de seguridad
    s = rk4_step(rhs, t, s, dt)                  # un paso de RK4
    t += dt
    if s[0] - s[1] > 0: tocó = True              # ya hubo contacto
    if tocó and s[0] - s[1] <= 0: break          # se separaron: terminar
e_medido = (s[3] - s[2]) / v1                    # definicion de restitucion""",
        numeric_converge=[
            ("Error de e al reducir el paso", "Con $\\zeta=0.3$ se repite la simulación con seis pasos y se compara $e$ medido con $e=\\exp(-\\pi\\zeta/\\sqrt{1-\\zeta^2})$. El error disminuye en promedio con una pendiente menor que 4 porque el contacto es no suave y el final se detecta sin interpolar, de modo que la separación se registra con una granularidad de hasta un paso.",
             [table([r"h\ (\mathrm{s})", r"|e_{num}-e_{an}|"], [[sci(d, 2), sci(e, 3)] for d, e in zip(dts, errs)]),
              rf"e_{{an}}={ea:.5f},\qquad \text{{pendiente de ajuste}}={sl:.2f}"]),
            ("Barrido en el amortiguamiento", "Con el paso final $h=10^{-5}$ s se repite para tres valores de $\\zeta$. El error absoluto es del orden de $10^{-5}$ a $10^{-4}$ y, por lo tanto, mucho menor que las diferencias entre las restituciones, que es lo que se quiere resolver.",
             [table([r"\zeta", r"e_{an}", r"e_{num}", r"|\Delta e|"], rows)]),
            ("Fuentes de error y confianza", "El error de truncamiento domina y viene sobre todo de los dos pasos que cruzan el inicio y el final del contacto. El redondeo es despreciable (unos $10^{-16}$ por paso). Además de esto hay un error de modelo: la restitución del resorte lineal amortiguado no depende de la velocidad de impacto, algo que no es cierto en materiales reales. Un criterio práctico es exigir al menos 100 pasos dentro del contacto, es decir $h\\lesssim T_c/100$ con $T_c\\approx\\pi/\\omega_0$ (cota superior de la duración del contacto).",
             [rf"\omega_0=\sqrt{{k/\mu}}={w0:.3f}\ \mathrm{{rad/s}},\qquad T_c\approx\dfrac{{\pi}}{{\omega_0}}={np.pi / w0 * 1000:.2f}\ \mathrm{{ms}}\ \Rightarrow\ \dfrac{{T_c}}{{h}}\approx{np.pi / w0 / 1e-5:.0f}\ \text{{pasos}}"]),
        ],
    )


# ============================================================== BallisticPendulum
def _pendulum() -> dict:
    an = c10.bullet_block_analytic()
    V, h_an = an["V"], an["h"]
    dts = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 1e-4]
    errs = [abs(c10.swing_numeric(V, dt=d) - h_an) for d in dts]
    sl = slope(dts, errs)
    th_max = np.arccos(1 - h_an / c10.ROPE)
    return dict(
        numeric_intro=[
            "En el péndulo balístico el choque bala-bloque es perfectamente inelástico y se resuelve exactamente con la conservación del momento, pero la altura que sube el bloque se calcula numéricamente integrando la ecuación del péndulo rígido $\\ddot\\theta=-(g/\\ell)\\sin\\theta$ sin la aproximación de ángulos pequeños. La altura máxima es $h=\\ell(1-\\cos\\theta_{max})$ y se obtiene siguiendo la trayectoria hasta que la velocidad angular cambia de signo.",
            "Se convierte la ecuación de segundo orden en un sistema de primer orden $\\dot\\theta=\\omega,\\ \\dot\\omega=-(g/\\ell)\\sin\\theta$ y se integra con Runge-Kutta de cuarto orden: $\\mathbf y_{n+1}=\\mathbf y_n+\\frac h6(\\mathbf k_1+2\\mathbf k_2+2\\mathbf k_3+\\mathbf k_4)$, con error global $O(h^4)$. Esto es apropiado porque $\\sin\\theta$ es suave y la ecuación no tiene solución elemental (el período exacto involucra integrales elípticas), así que no se reutiliza la fórmula $h=V^2/2g$ del análisis.",
            "La energía mecánica es una cantidad conservada de la ecuación que RK4 no impone: la simulación no sabe de $h=V^2/2g$. Por eso, que la altura numérica coincida con esa expresión es una verificación de que la dinámica del péndulo discretizada conserva la energía; la discrepancia mide cuánto se degrada esa conservación por truncamiento.",
            "El paso $h=10^{-4}$ s equivale a unas $10^{4}$ evaluaciones por segundo de movimiento. La altura se toma como el máximo de $\\ell(1-\\cos\\theta)$ sobre los instantes muestreados, no sobre el tiempo continuo, lo que añade un error $O(h^2)$ alrededor de la cumbre (donde $\\omega\\approx0$ y $h(t)$ es plana). Con $V=2.4752$ m/s el ángulo máximo es de unos 42°, una amplitud en la que $\\sin\\theta\\approx\\theta$ ya empieza a fallar, por lo que la integración con el seno completo es justificada.",
        ],
        numeric_code="""def altura_maxima(V, L, dt):
    f = lambda t, y: [y[1], -(g / L) * sin(y[0])]    # theta' = omega, omega' = -(g/L) sin(theta)
    y = [0.0, V / L]                             # parte del punto mas bajo con omega0 = V/L
    t, hmax = 0.0, 0.0
    while t < 3.0:                               # limite de seguridad
        y = rk4_step(f, t, y, dt)                # un paso de RK4
        t += dt
        hmax = max(hmax, L * (1 - cos(y[0])))    # altura sobre el punto mas bajo
        if y[1] < 0: break                       # omega cambio de signo: ya paso la cumbre
    return hmax

V = m * v / (m + M)                              # conservacion del momento en el choque
h = altura_maxima(V, L, 1e-4)                    # altura por integracion
K_antes = 0.5 * m * v**2                         # energia cinetica de la bala
K_despues = 0.5 * (m + M) * V**2                 # energia cinetica justo tras el choque
perdida = 1 - K_despues / K_antes                # fraccion disipada en el choque""",
        numeric_converge=[
            ("Altura numérica contra altura de energía", "Se repite la integración con seis pasos y se compara con $h=V^2/2g$, que es el valor que da la conservación de la energía. La pendiente de ajuste no es 4 porque el máximo se lee sobre una malla discreta de tiempos y el criterio $\\omega<0$ detiene el lazo con un paso de retraso.",
             [table([r"h_t\ (\mathrm{s})", r"|h_{num}-V^2/2g|\ (\mathrm{m})"], [[sci(d, 2), sci(e, 3)] for d, e in zip(dts, errs)]),
              rf"V={V:.5f}\ \mathrm{{m/s}},\quad h_{{an}}={h_an:.6f}\ \mathrm{{m}},\quad \text{{pendiente de ajuste}}={sl:.2f}"]),
            ("Qué significa ese error", "Incluso con el paso más grueso el error de altura es de unos micrómetros, y con $h=10^{-4}$ s baja a unos $10^{-10}$ m. El resultado es consistente con la energía: el sistema discretizado conserva $\\tfrac12\\omega^2\\ell^2-g\\ell\\cos\\theta$ con un error mucho menor que cualquier medida experimental.",
             [rf"\theta_{{max}}=\arccos\!\left(1-\dfrac{{h}}{{\ell}}\right)={np.degrees(th_max):.3f}^\circ"]),
            ("Fuentes de error y confianza", "Hay tres: el truncamiento de RK4 (muy pequeño con $\\omega h\\ll1$), la lectura discreta del máximo, $O(h^2)$, y el redondeo (unos $10^{-16}$ por paso). El resultado se considera fiable cuando la altura deja de cambiar al reducir $h$ a la mitad y cuando coincide con la energía dentro de esa tolerancia. La pérdida de energía en el choque ($99\\%$ de la cinética inicial) no es error numérico: es física, la que absorbe la deformación del bloque.",
             [rf"\text{{fracción disipada}}=1-\dfrac{{K_f}}{{K_i}}={c10.bullet_block_numeric()['loss_fraction']:.5f}"]),
        ],
    )


# ============================================================================ SpinUp
def _spinup() -> dict:
    ns = [50, 100, 200, 400, 800, 4000]
    tf = c11.T_END
    th_ex = tf**3 / 24
    w_ex = 0.125 * tf**2
    rows, hs, es = [], [], []
    for n in ns:
        _, w, th = c11.spin_up_numeric_variable(n)
        d = tf / n
        hs.append(d)
        es.append(abs(th[-1] - th_ex))
        pred = tf * d**2 * 0.25 / 12
        rows.append([str(n), sci(d, 2), sci(abs(w[-1] - w_ex), 2), sci(es[-1], 4), sci(pred, 4)])
    sl = slope(hs, es)
    _, wc, thc = c11.spin_up_numeric(400)
    return dict(
        numeric_intro=[
            "La velocidad angular y el ángulo se obtienen integrando la aceleración angular: $\\omega(t)=\\int\\alpha\\,dt$ y $\\theta(t)=\\int\\omega\\,dt$. El código no usa las fórmulas $\\omega=\\alpha t$ y $\\theta=\\tfrac12\\alpha t^2$; construye cada valor sumando áreas bajo la curva en una malla de tiempos, un procedimiento que funcionaría igual si $\\alpha$ cambiara con el tiempo o viniera de datos medidos.",
            "La regla del trapecio acumulada aproxima el área de cada tramo por un trapecio: $\\omega_{k+1}=\\omega_k+\\tfrac12(\\alpha_k+\\alpha_{k+1})\\Delta t$ y $\\theta_{k+1}=\\theta_k+\\tfrac12(\\omega_k+\\omega_{k+1})\\Delta t$. Reemplaza la curva por segmentos rectos, así que es exacta para funciones lineales y tiene error local $-\\frac{\\Delta t^3}{12}f''$, es decir un error global $O(\\Delta t^2)$: cada vez que se divide el paso a la mitad el error baja cuatro veces.",
            "Con $\\alpha$ constante, $\\omega$ es lineal y $\\theta$ es el área bajo una recta, de modo que el trapecio da el resultado exacto salvo redondeo y no se ve ningún error de truncamiento. Para probar que el método de verdad funciona se usa también $\\alpha(t)=0.25\\,t$ con $N=4000$ pasos: ahora $\\omega$ es parabólica, $\\theta$ cúbica y aparece un error medible. La malla de $N=400$ pasos (con $\\Delta t=0.02$ s) del caso constante es solo una elección cómoda.",
            "El método es independiente del análisis porque jamás multiplica $\\alpha$ por $t$ ni eleva $t$ al cuadrado: solo suma áreas de trapecios. Falla cuando la función oscila más rápido que el paso (aliasing) o tiene saltos, y su error crece si se integra por mucho tiempo, porque el error de $\\omega$ se acumula en $\\theta$.",
        ],
        numeric_code="""t = linspace(0, T, N + 1)                        # malla uniforme, dt = T/N
dt = diff(t)                                     # anchos de los N tramos
a = alfa(t)                                      # aceleracion angular en cada nodo
# regla del trapecio acumulada (area = promedio de extremos x ancho)
w = concat([0.0], cumsum(0.5 * (a[1:] + a[:-1]) * dt))     # omega(t_k) = integral de alfa
th = concat([0.0], cumsum(0.5 * (w[1:] + w[:-1]) * dt))    # theta(t_k) = integral de omega
# comprobacion con aceleracion variable alfa = 0.25 t  (omega=0.125 t^2, theta=t^3/24)
err_theta = abs(th[-1] - T**3 / 24)              # error global en theta
pred = T * dt[0]**2 * 0.25 / 12                  # estimacion teorica (b-a) h^2 f''/12""",
        numeric_converge=[
            ("Error contra N con aceleración variable", "Para $\\alpha=0.25\\,t$ la integral de $\\omega$ es exacta con el trapecio (porque $\\alpha$ es lineal) y el error de $\\theta$ proviene del término $f''=0.25$ de $\\omega$. Se compara el error medido con la predicción teórica $(b-a)\\,h^2 f''/12$.",
             [table(["N", r"\Delta t\ (\mathrm{s})", r"|\Delta\omega|", r"|\Delta\theta|\ (\mathrm{rad})", r"\text{teoría}"], rows),
              rf"\text{{pendiente observada}}={sl:.3f}\quad(\text{{trapecio: orden }}2)"]),
            ("Caso de aceleración constante", "Con $\\alpha=1.5$ rad/s$^2$ y $N=400$ el integrando es lineal, el trapecio no comete error de truncamiento y lo único que se observa es el redondeo acumulado de 400 sumas.",
             [rf"|\omega_N-\omega_{{an}}|={sci(abs(wc[-1] - c11.omega_analytic(tf)), 2)},\qquad |\theta_N-\theta_{{an}}|={sci(abs(thc[-1] - c11.theta_analytic(tf)), 2)}"]),
            ("Fuentes de error y criterio de confianza", "El truncamiento es $O(\\Delta t^2)$ y la predicción teórica coincide con la medición a varias cifras; el redondeo (del orden de $10^{-15}$) es despreciable. El resultado es confiable si al duplicar $N$ el error baja cerca de cuatro veces y se queda dentro de la tolerancia que se busca; si no, el integrando no es suficientemente suave para el paso elegido.",
             [rf"\text{{error de }}\theta\text{{ con }}N=4000:\ {sci(es[-1], 3)}\ \mathrm{{rad}}\ \ll\ \theta={th_ex:.4f}\ \mathrm{{rad}}"]),
        ],
    )


# ================================================================== RollingCycloid
def _cycloid() -> dict:
    ns = [8, 16, 32, 64, 128]
    la = c11.arch_length_analytic()
    aa = c11.arch_area_analytic()
    es_s = [abs(c11.arch_length_numeric(n) - la) for n in ns]
    es_p = [abs(c11.arch_length_polyline(n) - la) for n in ns]
    hs = [2 * np.pi / n for n in ns]
    sl_s, sl_p = slope(hs, es_s), slope(hs, es_p)
    rows = [[str(n), sci(a, 3), sci(b, 3)] for n, a, b in zip(ns, es_s, es_p)]
    ea = {n: abs(c11.arch_area_numeric(n) - aa) for n in (4, 8, 2000)}
    return dict(
        numeric_intro=[
            "La longitud de un arco de cicloide es una integral de longitud de arco, $s=\\int|d\\vec r/d\\theta|\\,d\\theta$, y el área bajo el arco es $\\int y\\,dx$. El análisis las resuelve con identidades trigonométricas ($1-\\cos\\theta=2\\sin^2(\\theta/2)$); el código no hace ninguna sustitución y evalúa el integrando $2r\\sin(\\theta/2)$ en una malla de 2000 intervalos sobre $[0,2\\pi]$ aplicando la regla de Simpson.",
            "Simpson sustituye el integrando por parábolas sobre cada par de subintervalos: $\\int_a^b f\\,d\\theta\\approx\\frac h3\\,[f_0+4f_1+2f_2+\\cdots+4f_{N-1}+f_N]$ con $h=2\\pi/N$. Su error global es $-\\frac{(b-a)}{180}h^4 f^{(4)}(\\xi)$, es decir $O(h^4)$: duplicar $N$ reduce el error 16 veces. Para contraste, la poligonal (suma de las longitudes de los segmentos entre puntos de la curva) es equivalente al trapecio y tiene error $O(h^2)$.",
            "La función $\\sin(\\theta/2)$ es suave y no negativa en $[0,2\\pi]$ (el módulo del código solo importaría fuera de ese intervalo), de modo que se cumplen las hipótesis de Simpson y el error observado sigue la ley $h^4$. Con $N=2000$ el error ya es del orden de $10^{-13}$ m. En el área, el integrando $(1-\\cos\\theta)^2$ es un polinomio trigonométrico de grado 2 y Simpson lo integra exactamente cuando $N$ es suficientemente grande.",
            "Es un método independiente porque jamás usa $8r$ ni $3\\pi r^2$: solo suma valores del integrando. Su límite es la suavidad: si el integrando tuviera un pico o un salto dentro del intervalo (por ejemplo al integrar varios arcos seguidos con $|\\sin(\\theta/2)|$, que se anula con pendiente distinta de cero), el orden $h^4$ se perdería. Aquí las cúspides de la cicloide quedan en los extremos del intervalo, donde la velocidad se anula, y no afectan.",
        ],
        numeric_code="""th = linspace(0, 2*pi, N + 1)                    # N subintervalos pares, h = 2pi/N
# longitud: rapidez del punto del borde por unidad de theta (omega = 1)
rapidez = 2 * r * abs(sin(th / 2))               # |dr/dtheta| = 2r sin(theta/2)
L_arco = simpson(rapidez, th)                    # h/3 (f0 + 4f1 + 2f2 + ... + fN)
# area bajo el arco: int y dx con x' = r(1 - cos th)
y = r * (1 - cos(th))                            # altura del punto del borde
dx = r * (1 - cos(th))                           # dx/dtheta
A_arco = simpson(y * dx, th)                     # integral de y dx
# poligonal (orden 2) para comparar
x, y = r*(th - sin(th)), r*(1 - cos(th))         # puntos de la curva
L_poli = sum(hypot(diff(x), diff(y)))            # suma de segmentos""",
        numeric_converge=[
            ("Longitud del arco: Simpson contra poligonal", "Con $N$ creciente se mide el error de la longitud respecto de $8r=4$ m. Cada duplicación de $N$ divide el error de Simpson por 16 y el de la poligonal por 4; las pendientes de ajuste lo confirman.",
             [table(["N", r"\text{Simpson}\ (\mathrm{m})", r"\text{poligonal}\ (\mathrm{m})"], rows),
              rf"\text{{pendiente Simpson}}={sl_s:.2f}\ (\text{{teoría }}4),\qquad \text{{pendiente poligonal}}={sl_p:.2f}\ (\text{{teoría }}2)"]),
            ("Área: exactitud espectral", "El integrando del área es un polinomio trigonométrico de grado 2 sobre un período completo, por lo que cualquier regla de este tipo es exacta desde un número pequeño de puntos. Con $N=4$ todavía falla, con $N\\ge8$ ya se llega al redondeo.",
             [rf"|A_N-3\pi r^2|:\quad N=4:\ {sci(ea[4], 3)},\quad N=8:\ {sci(ea[8], 2)},\quad N=2000:\ {sci(ea[2000], 2)}\ \mathrm{{m^2}}"]),
            ("Fuentes de error y criterio de confianza", "Con $N=2000$ el truncamiento de Simpson es menor que el redondeo acumulado, del orden de $10^{-13}$ m en la longitud. La poligonal, en cambio, daría unos $4\\times10^{-7}$ m. Un resultado es confiable cuando la pendiente medida entre dos refinamientos sucesivos se acerca al orden teórico; si baja, el integrando no es suave o ya se alcanzó el suelo del redondeo.",
             [rf"\text{{longitud con }}N=2000:\ {c11.arch_length_numeric(2000):.10f}\ \mathrm{{m}}\ \ (8r={la:.1f}\ \mathrm{{m}})"]),
        ],
    )


# =============================================================== MomentOfInertia
def _inertia() -> dict:
    ns = [4, 8, 16, 32]
    ia = c12.rod_inertia_analytic("end")
    ds = c12.disc_inertia_analytic()
    rows, hs, tr_err = [], [], []
    for n in ns:
        x = np.linspace(0, c12.L_ROD, n + 1)
        trap = trapezoid(x**2 * (c12.M_ROD / c12.L_ROD), x)
        es = abs(c12.rod_inertia_numeric("end", n=n) - ia)
        ed = abs(c12.disc_inertia_numeric(n=n) - ds)
        tr_err.append(abs(trap - ia))
        hs.append(c12.L_ROD / n)
        rows.append([str(n), sci(es, 2), sci(ed, 2), sci(tr_err[-1], 3)])
    sl = slope(hs, tr_err)
    return dict(
        numeric_intro=[
            "El momento de inercia se define como $I=\\int r^2\\,dm$: se suma, sobre todos los elementos de masa, su masa por el cuadrado de la distancia al eje. El análisis lo evalúa con una primitiva; el programa en cambio discretiza el objeto y suma. Para la barra, $dm=(m/L)\\,dx$ con $r=x-x_{eje}$; para el disco, se cortan anillos de radio $s$ y ancho $ds$ con $dm=\\sigma\\,2\\pi s\\,ds$. En ambos casos el resultado es una integral unidimensional.",
            "Se usa la regla de Simpson, $\\int f\\,dx\\approx\\frac h3\\,[f_0+4f_1+2f_2+\\cdots+f_N]$ con $h$ el ancho de cada subintervalo, cuyo error global es $-\\frac{(b-a)}{180}h^4 f^{(4)}(\\xi)$. Como el integrando de la barra ($x^2$) es de grado 2 y el del disco ($s^3$) es de grado 3, la cuarta derivada es nula y Simpson es exacta con cualquier $N$ par: lo único que se ve es el redondeo. Se eligió $N=400$ sin necesidad, solo para tener holgura si cambiara la densidad.",
            "Esa exactitud es una propiedad del problema, no una casualidad: con una densidad no uniforme (por ejemplo $\\sigma\\propto s^2$) el integrando sería $s^5$ y aparecería el error $O(h^4)$ habitual, que se reduciría refinando $N$. Para contrastar se incluye el trapecio, que reemplaza el integrando por segmentos rectos y tiene error $O(h^2)$.",
            "El método es independiente del análisis porque no emplea las fórmulas $mL^2/3$ ni $\\tfrac12 mR^2$ y trata el objeto como una suma de elementos; además, el teorema del eje paralelo, $I=I_{cm}+md^2$, se verifica cruzando los dos ejes de la barra. Limitaciones: para objetos tridimensionales la integral es doble o triple y el costo crece con la dimensión, caso en el que se prefiere Monte Carlo.",
        ],
        numeric_code="""# Barra: I = int (x - x_eje)^2 (m/L) dx
x = linspace(0, L, N + 1)                        # N subintervalos pares
eje = 0.0 if extremo else L / 2                  # posicion del eje
I_barra = simpson((x - eje)**2 * (m / L), x)    # lambda = m/L uniforme

# Disco: anillos de radio s y ancho ds, dm = sigma 2 pi s ds
s = linspace(0, R, N + 1)
sigma = m / (pi * R**2)                          # densidad superficial
I_disco = simpson(s**2 * sigma * 2*pi * s, s)    # integrando s^3: grado 3

# teorema del eje paralelo como comprobacion cruzada
I_extremo = I_centro + m * (L / 2)**2            # I_cm + m d^2 debe igualar el calculo directo""",
        numeric_converge=[
            ("Simpson contra trapecio", "Se calcula el error de la barra (eje en el extremo) y del disco con Simpson, y el de la barra con la regla del trapecio, para cuatro valores de $N$. Simpson está en el piso de redondeo para cualquier $N$ mientras que el trapecio baja como $h^2$.",
             [table(["N", r"\text{Simpson, barra}\ (\mathrm{kg\,m^2})", r"\text{Simpson, disco}", r"\text{trapecio, barra}"], rows),
              rf"\text{{pendiente del trapecio}}={sl:.2f}\quad(\text{{teoría }}2)"]),
            ("Fuentes de error", "En Simpson no hay error de truncamiento porque el integrando tiene grado $\\le3$; queda el redondeo, de magnitud $10^{-16}$ en unidades de $I$. En el trapecio el error es $(b-a)h^2 f''/12$ y se puede predecir: para la barra vale $\\tfrac{m}{L}\\cdot\\tfrac{L\\,h^2}{12}\\cdot2$.",
             [rf"\text{{trapecio, }}N=32:\ \text{{medido}}={sci(tr_err[-1], 4)},\quad \text{{teoría}}={sci(c12.M_ROD / c12.L_ROD * c12.L_ROD * (c12.L_ROD / 32) ** 2 / 12 * 2, 4)}\ \mathrm{{kg\,m^2}}"]),
            ("Criterio de confianza y comprobación cruzada", "El resultado es confiable cuando los dos métodos de integración coinciden y la prueba del eje paralelo se cumple. Aquí la verificación cruzada con el eje paralelo reproduce el valor del extremo con un error de redondeo.",
             [rf"I_{{cm}}+m\left(\dfrac{{L}}{{2}}\right)^2={c12.parallel_axis(c12.rod_inertia_numeric('center'), c12.M_ROD, c12.L_ROD / 2):.8f}\ \mathrm{{kg\,m^2}},\qquad I_{{extremo}}={c12.rod_inertia_numeric('end'):.8f}\ \mathrm{{kg\,m^2}}"]),
        ],
    )


# ============================================================== HangingMassDisc
def _hanging() -> dict:
    an = c12.hanging_analytic()
    m, big_m, r, g = c12.M_HANG, c12.M_DISC, c12.R_DISC, c12.G
    inertia = 0.5 * big_m * r**2
    mat = np.array([[m, 1.0], [inertia / r, -r]])
    rhs_v = np.array([m * g, 0.0])
    sol = np.linalg.solve(mat, rhs_v)
    cond = np.linalg.cond(mat)
    resid = np.linalg.norm(mat @ sol - rhs_v)

    def rhs(_t, s):
        a, _ = np.linalg.solve(mat, rhs_v)
        return np.array([s[1], a, s[3], a / r])

    ns = [10, 100, 1000, 10000]
    rows, y_an = [], an["a"] / 2
    for n in ns:
        res = c12.hanging_numeric(n=n)
        _, ye = integrate(rhs, np.zeros(4), 0.0, 1.0, n, euler_step)
        rows.append([str(n), sci(abs(res["y"][-1, 0] - y_an), 2), sci(res["energy_error"], 2), sci(abs(ye[-1, 0] - y_an), 3)])
    return dict(
        numeric_intro=[
            "Una masa colgante unida por una cuerda a un disco que gira lleva a dos ecuaciones acopladas: $m\\,a=mg-T$ para la masa y $I\\,\\alpha=Tr$ para el disco, con la ligadura $a=\\alpha r$ de la cuerda inextensible. El código no despeja $a=mg/(m+M/2)$: en cada instante arma las dos ecuaciones como un sistema lineal $2\\times2$ en las incógnitas $(a,T)$ y lo resuelve con álgebra lineal numérica.",
            "El sistema lineal se resuelve por eliminación gaussiana con pivoteo parcial (descomposición LU), que es un método directo y exacto salvo redondeo; su fiabilidad se mide con el número de condición $\\kappa(A)=\\|A\\|\\,\\|A^{-1}\\|$, que indica cuánto se amplifica un error relativo en los datos. Una vez conocida la aceleración, el estado $(y,v,\\theta,\\omega)$ se avanza con Runge-Kutta de cuarto orden, $\\mathbf y_{n+1}=\\mathbf y_n+\\frac h6(\\mathbf k_1+2\\mathbf k_2+2\\mathbf k_3+\\mathbf k_4)$, de error global $O(h^4)$.",
            "Este problema tiene una particularidad: la aceleración es constante, así que la solución $y=\\tfrac12at^2$ es un polinomio de grado 2 y RK4, que es exacto hasta grado 4, la reproduce sin error de truncamiento. Eso es lo que muestra la tabla de convergencia: el error está en el piso del redondeo y crece lentamente con el número de pasos. Para ver el orden de un método hay que compararlo con uno peor, como Euler explícito, de orden 1.",
            "Se usan $N=1000$ pasos durante 1 s ($h=10^{-3}$ s). El resultado es independiente del análisis porque se obtiene al resolver el sistema numéricamente y la verificación más sólida no usa la fórmula cerrada: la conservación de la energía, $mgy=\\tfrac12mv^2+\\tfrac12I\\omega^2$, que no se impone en ninguna ecuación y por eso detecta cualquier error de planteamiento.",
        ],
        numeric_code="""I = 0.5 * M * R**2                               # momento de inercia del disco
A = [[m, 1.0],                                   # m a + T = m g
     [I / R, -R]]                                # (I/R) a - R T = 0  (de I alfa = T R, alfa = a/R)
b = [m * g, 0.0]
def aceleracion():
    a, T = solve(A, b)                           # sistema lineal 2x2 por LU
    return a, T
def rhs(t, s):                                   # s = (y, v, theta, omega)
    a, _ = aceleracion()
    return [s[1], a, s[3], a / R]                # y'=v, v'=a, theta'=omega, omega'=a/R
t, Y = integrate(rhs, zeros(4), 0.0, 1.0, N, rk4_step)   # N pasos de RK4
energia_in = m * g * Y[-1, 0]                    # trabajo del peso
energia_out = 0.5*m*Y[-1,1]**2 + 0.5*I*Y[-1,3]**2   # cinetica de traslacion + rotacion
error_energia = abs(energia_in - energia_out)    # verificacion independiente""",
        numeric_converge=[
            ("Condicionamiento y residuo del sistema lineal", "El número de condición mide qué tan sensible es la solución a perturbaciones; un valor cercano a 1 es excelente. El residuo $\\|A\\mathbf x-\\mathbf b\\|$ mide qué tan bien se cumplen las ecuaciones planteadas.",
             [rf"\kappa(A)={cond:.3f},\qquad \|A\mathbf x-\mathbf b\|={sci(resid, 2)},\qquad (a,T)=({sol[0]:.6f}\ \mathrm{{m/s^2}},\ {sol[1]:.6f}\ \mathrm{{N}})"]),
            ("RK4 contra Euler al aumentar N", "Se integra 1 s con $N$ pasos y se compara la posición final con $y=\\tfrac12at^2$. RK4 reproduce la parábola exacta y solo muestra redondeo (que crece lentamente con $N$) y un error de energía igualmente de redondeo, mientras que Euler baja linealmente con $h$, como corresponde a un método de orden 1.",
             [table(["N", r"|\Delta y|_{RK4}\ (\mathrm{m})", r"\Delta E_{RK4}\ (\mathrm{J})", r"|\Delta y|_{Euler}\ (\mathrm{m})"], rows)]),
            ("Fuentes de error y criterio de confianza", "La solución lineal tiene error de redondeo de orden $\\kappa\\,\\varepsilon_{mach}\\approx10^{-16}$; la integración no tiene truncamiento porque el movimiento es parabólico. Una solución es confiable cuando el número de condición es pequeño, el residuo cercano al redondeo y la energía se conserva; con otro modelo (cuerda elástica o rozamiento) habría que añadir la comprobación del orden contra Euler como la anterior.",
             [rf"a_{{an}}={an['a']:.6f}\ \mathrm{{m/s^2}},\qquad a_{{num}}={sol[0]:.6f}\ \mathrm{{m/s^2}},\qquad \Delta E(N=1000)={sci(c12.hanging_numeric(n=1000)['energy_error'], 2)}\ \mathrm{{J}}"]),
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
