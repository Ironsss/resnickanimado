"""Explicacion y pasos a mano: Volumen 1, capitulos 13 a 16 (SkaterSpin ... FloatingCylinder).

Todas las cifras se calculan con las funciones de `physics/` o replicando el algoritmo de esas funciones.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch13_angular_momentum as c13
from physics import ch14_equilibrium as c14
from physics import ch15_gravitation as c15
from physics import ch16_fluid_statics as c16
from physics.integrators import rk4_step


def _rk4(f, t, y, h):
    """Un paso RK4 devolviendo (k1, k2, k3, k4, y1) para mostrarlo a mano."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return k1, k2, k3, k4, y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _v(v, d=5) -> str:
    """Vector fila en TeX."""
    return r"\left(" + ",\\ ".join(sci(float(x), d) for x in np.atleast_1d(v)) + r"\right)"


# ----------------------------------------------------------------------------- SkaterSpin
def _skater() -> dict:
    ii, if_, wi, tp = c13.I_I, c13.I_F, c13.W_I, c13.T_PULL
    wf = c13.omega_final_analytic()
    dk = c13.kinetic_energy_change_analytic()
    ki, kf = 0.5 * ii * wi**2, 0.5 * if_ * wf**2
    t, w, inertia, ang, k = c13.skater_numeric()
    n, tend = 4000, 4.0
    h = tend / n
    f = lambda tt, y: np.array([-(c13.inertia_rate(tt) / c13.inertia_profile(tt)) * y[0]])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, 0.0, [wi], h)
    i_h2, ir_h2 = float(c13.inertia_profile(h / 2)), float(c13.inertia_rate(h / 2))
    i_h, ir_h = float(c13.inertia_profile(h)), float(c13.inertia_rate(h))
    idx = [500, 1000, 1500, 2000, 4000]
    tabla = r"\begin{array}{c|cccc} t\,(\mathrm{s}) & I\,(\mathrm{kg\,m^2}) & \omega\,(\mathrm{rad/s}) & L\,(\mathrm{kg\,m^2/s}) & K\,(\mathrm{J})\\ \hline " + r"\\ ".join(
        f"{t[i]:.2f} & {inertia[i]:.4f} & {w[i]:.5f} & {ang[i]:.6f} & {k[i]:.4f}" for i in idx) + r"\end{array}"
    return dict(
        explanation=[
            "Cuando no actúa ningún torque externo sobre un cuerpo que gira, su momento angular $L = I\\omega$ no cambia. La patinadora sobre el hielo es un caso casi ideal: el roce con la pista es mínimo y el peso y la normal pasan por el eje de giro, de modo que el torque externo respecto a ese eje es prácticamente nulo.",
            "El momento de inercia $I$ mide cómo está repartida la masa respecto al eje: cuanto más lejos del eje, mayor es $I$. Al recoger los brazos la patinadora acerca masa al eje y $I$ disminuye; como el producto $I\\omega$ debe permanecer constante, la velocidad angular $\\omega$ tiene que aumentar en la misma proporción.",
            "La energía cinética de rotación, $K = \\tfrac12 I\\omega^2 = L^2/(2I)$, sí cambia. Con $L$ fijo, reducir $I$ aumenta $K$: la diferencia la aporta el trabajo interno de los músculos al jalar los brazos hacia adentro contra la tendencia a salir despedidos (fuerza centrífuga aparente en el sistema que gira). Por eso la conservación de $L$ no implica conservación de $K$.",
            "La conclusión se aplica solo mientras $\\tau_{ext}=0$. Es el mismo principio por el que un clavadista se encoge para dar más vueltas en el aire o un gato gira para caer de pie. Aquí la reducción de $I$ por un factor 2.5 produce un aumento de $\\omega$ por el mismo factor, una cifra muy típica en patinaje.",
        ],
        analytic_steps=[
            ("Identificar la magnitud conservada", "Respecto al eje vertical no hay torque externo, así que $dL/dt = \\tau_{ext} = 0$ y el momento angular se mantiene igual antes y después de recoger los brazos.",
             [r"\tau_{ext}=0\;\Rightarrow\; L_i = L_f\;\Rightarrow\; I_i\,\omega_i = I_f\,\omega_f"]),
            ("Calcular el momento angular inicial", "Se sustituyen los datos del enunciado con sus unidades.",
             [rf"L_i = I_i\,\omega_i = ({ii}\ \mathrm{{kg\,m^2}})({wi}\ \mathrm{{rad/s}}) = {ii*wi:.2f}\ \mathrm{{kg\,m^2/s}}"]),
            ("Despejar la velocidad angular final", "De $I_i\\omega_i = I_f\\omega_f$ se despeja $\\omega_f$ y se sustituye $I_f$.",
             [r"\omega_f = \dfrac{I_i\,\omega_i}{I_f}",
              rf"\omega_f = \dfrac{{({ii}\ \mathrm{{kg\,m^2}})({wi}\ \mathrm{{rad/s}})}}{{{if_}\ \mathrm{{kg\,m^2}}}} = \dfrac{{{ii*wi:.2f}}}{{{if_}}} = {wf:.2f}\ \mathrm{{rad/s}}"]),
            ("Energía cinética inicial", "Se evalúa $K = \\tfrac12 I\\omega^2$ con los valores iniciales.",
             [rf"K_i = \tfrac12 I_i\,\omega_i^2 = \tfrac12 ({ii})({wi})^2 = {ki:.2f}\ \mathrm{{J}}"]),
            ("Energía cinética final", "Mismo cálculo con $I_f$ y la $\\omega_f$ recién obtenida.",
             [rf"K_f = \tfrac12 I_f\,\omega_f^2 = \tfrac12 ({if_})({wf:.2f})^2 = {kf:.2f}\ \mathrm{{J}}"]),
            ("Variación de energía y su origen", "La diferencia es positiva: la energía extra no sale de ningún torque externo, sino del trabajo que realizan los músculos de la patinadora.",
             [rf"\Delta K = K_f - K_i = {kf:.2f} - {ki:.2f} = {dk:.1f}\ \mathrm{{J}}",
              rf"\dfrac{{K_f}}{{K_i}} = \dfrac{{I_i}}{{I_f}} = {ii/if_:.2f}"]),
        ],
        numeric_steps=[
            ("Plantear la ecuación diferencial", "Si $I$ varía en el tiempo y $\\tau_{ext}=0$, entonces $d(I\\omega)/dt = I'\\omega + I\\omega' = 0$, y por tanto $\\omega' = -(I'/I)\\,\\omega$. Esa ecuación se integra sin usar la solución analítica.",
             [r"\dfrac{d\omega}{dt} = -\dfrac{\dot I(t)}{I(t)}\,\omega"]),
            ("Modelar $I(t)$ de forma suave", "Los brazos se recogen entre $t=0$ y $t=2$ s con una curva cúbica cuya derivada es nula en los extremos, con $s = t/2$.",
             [rf"I(t) = I_i + (I_f - I_i)(3s^2 - 2s^3),\qquad \dot I(t) = (I_f - I_i)\,\dfrac{{6s - 6s^2}}{{2}}",
              rf"I_i = {ii},\quad I_f = {if_},\quad t_{{pull}} = {tp}\ \mathrm{{s}}"]),
            ("Parámetros del método RK4", "Se integra de $t=0$ a $t=4$ s con 4000 pasos iguales; la condición inicial es $\\omega(0)=2$ rad/s.",
             [rf"h = \dfrac{{{tend}}}{{{n}}} = {sci(h,3)}\ \mathrm{{s}},\qquad \omega_0 = {wi}\ \mathrm{{rad/s}}"]),
            ("Pendientes del primer paso", "RK4 evalúa $f(t,\\omega) = -(\\dot I/I)\\omega$ cuatro veces. En $t=0$ el código fija $\\dot I=0$ (solo se aplica estrictamente dentro de $0<t<2$); $k_2$ y $k_3$ se evalúan en $t=h/2$ y $k_4$ en $t=h$.",
             [rf"I(h/2) = {i_h2:.8f},\quad \dot I(h/2) = {sci(ir_h2,5)};\qquad I(h) = {i_h:.8f},\quad \dot I(h) = {sci(ir_h,5)}",
              rf"k_1 = {sci(float(k1[0]),5)},\quad k_2 = -\dfrac{{\dot I(h/2)}}{{I(h/2)}}\left(\omega_0 + \tfrac h2 k_1\right) = {sci(float(k2[0]),5)}",
              rf"k_3 = {sci(float(k3[0]),5)},\quad k_4 = {sci(float(k4[0]),5)}"]),
            ("Primer valor actualizado", "Se combinan las cuatro pendientes con pesos $1,2,2,1$.",
             [rf"\omega_1 = \omega_0 + \dfrac h6\,(k_1 + 2k_2 + 2k_3 + k_4) = {wi} + {sci(float(y1[0]-wi),5)} = {float(y1[0]):.9f}\ \mathrm{{rad/s}}",
              rf"L_1 = I(h)\,\omega_1 = ({i_h:.8f})({float(y1[0]):.9f}) = {i_h*float(y1[0]):.8f}\ \mathrm{{kg\,m^2/s}}"]),
            ("Resultados en pasos posteriores", "Se repite el mismo paso 3999 veces más. La tabla muestra que $\\omega$ crece mientras $I$ baja, y que $L=I\\omega$ no se altera.",
             [tabla]),
            ("Verificar conservación y comparar", "Tras $t=2$ s, $I=I_f$ y $\\dot I=0$, así que $\\omega$ queda constante. El error es el residuo del integrador.",
             [rf"\omega_f^{{num}} = {float(w[-1]):.9f}\ \mathrm{{rad/s}},\qquad \omega_f^{{an}} = {wf:.9f}\ \mathrm{{rad/s}}",
              rf"\Delta L = L_{{max}} - L_{{min}} = {sci(float(np.ptp(ang)),3)},\qquad \Delta K^{{num}} = {float(k[-1]-k[0]):.6f}\ \mathrm{{J}}\ \ (\text{{analítico }}{dk:.1f})"]),
        ],
    )


# ----------------------------------------------------------------------------- GyroscopePrecession
def _gyro() -> dict:
    m, r, inertia, w, g = c13.M_GYRO, c13.R_GYRO, c13.I_GYRO, c13.W_SPIN, c13.G
    big_l = inertia * w
    tau = m * g * r
    om = c13.precession_analytic()
    om2 = c13.precession_analytic(400.0)
    t, y, phi, om_n = c13.precession_numeric()
    _, _, _, om2_n = c13.precession_numeric(w_spin=400.0)
    n, tend = 6000, 6.0
    h = tend / n

    def rhs(_t, s):
        axis = s / np.linalg.norm(s)
        return np.cross(r * axis, np.array([0.0, 0.0, -m * g]))

    s0 = np.array([big_l, 0.0, 0.0])
    k1, k2, k3, k4, s1 = _rk4(rhs, 0.0, s0, h)
    phi1 = float(np.arctan2(s1[1], s1[0]))
    return dict(
        explanation=[
            "Un giróscopo que gira rápido tiene un momento angular $\\vec L = I\\omega\\,\\hat e$ a lo largo de su eje. Si el eje es horizontal y se apoya en un pivote, el peso aplica un torque $\\vec\\tau = \\vec r\\times m\\vec g$ respecto al pivote. Intuitivamente uno esperaría que el giróscopo se caiga, pero no lo hace.",
            "La clave es la segunda ley en forma rotacional, $d\\vec L/dt = \\vec\\tau$. El torque es horizontal y perpendicular a $\\vec L$, así que no cambia el módulo de $\\vec L$, solo su dirección: en un intervalo $dt$ el vector $\\vec L$ gira un ángulo $d\\phi = \\tau\\,dt/L$ en el plano horizontal. El resultado es un movimiento de precesión alrededor del eje vertical.",
            "La velocidad de precesión es $\\Omega = \\tau/L = mgr/(I\\omega)$. Cuanto más rápido gira el rotor (mayor $L$), más lenta es la precesión; duplicar $\\omega$ reduce $\\Omega$ a la mitad. Es la razón por la que un trompo lento se bambolea mucho y uno rápido parece quieto.",
            "Esta fórmula es una aproximación válida cuando $\\omega$ es mucho mayor que $\\Omega$ (aquí $200$ frente a menos de $1$ rad/s), de modo que el momento angular total se confunde con el del giro propio y se ignora la nutación y el momento angular de la propia precesión.",
        ],
        analytic_steps=[
            ("Datos y momento angular del giro", "Se calcula $L$ asociado al giro propio, que apunta a lo largo del eje horizontal.",
             [rf"m = {m}\ \mathrm{{kg}},\quad r = {r}\ \mathrm{{m}},\quad I = {sci(inertia,3)}\ \mathrm{{kg\,m^2}},\quad \omega = {w:.0f}\ \mathrm{{rad/s}}",
              rf"L = I\omega = ({sci(inertia,3)})({w:.0f}) = {big_l:.2f}\ \mathrm{{kg\,m^2/s}}"]),
            ("Torque del peso respecto al pivote", "El peso actúa en el centro de masa, a distancia $r$ del pivote y perpendicular al eje, por lo que el brazo de palanca es $r$.",
             [rf"\tau = m g r = ({m})({g})({r}) = {tau:.3f}\ \mathrm{{N\,m}}"]),
            ("Aplicar la segunda ley rotacional", "El torque es perpendicular a $\\vec L$: solo la rota. En un tiempo $dt$ el vector $\\vec L$ cambia $dL = \\tau\\,dt$ y su dirección gira $d\\phi = dL/L$.",
             [r"dL = \tau\,dt,\qquad d\phi = \dfrac{dL}{L} = \dfrac{\tau}{L}\,dt"]),
            ("Despejar la velocidad de precesión", "La razón de cambio del ángulo azimutal es la velocidad de precesión.",
             [r"\Omega = \dfrac{d\phi}{dt} = \dfrac{\tau}{L} = \dfrac{mgr}{I\omega}"]),
            ("Sustituir con $\\omega = 200$ rad/s", "Se usa el torque y el momento angular obtenidos antes.",
             [rf"\Omega = \dfrac{{{tau:.3f}\ \mathrm{{N\,m}}}}{{{big_l:.2f}\ \mathrm{{kg\,m^2/s}}}} = {om:.3f}\ \mathrm{{rad/s}}",
              rf"T_{{prec}} = \dfrac{{2\pi}}{{\Omega}} = {2*np.pi/om:.2f}\ \mathrm{{s}}"]),
            ("Dependencia con $\\omega$", "Como $\\Omega\\propto 1/\\omega$, al duplicar la velocidad de giro la precesión se reduce a la mitad.",
             [rf"\omega = 400\ \mathrm{{rad/s}}:\quad \Omega = \dfrac{{({m})({g})({r})}}{{({sci(inertia,3)})(400)}} = {om2:.3f}\ \mathrm{{rad/s}}"]),
        ],
        numeric_steps=[
            ("Plantear el sistema vectorial", "El código no usa la fórmula de $\\Omega$: integra la ecuación $d\\vec L/dt = \\vec r\\times m\\vec g$ en 3D, con $\\vec r = r\\,\\hat L$ (el centro de masa está sobre el eje, que sigue a $\\vec L$) y $m\\vec g = (0,0,-mg)$.",
             [r"\dfrac{d\vec L}{dt} = r\,\hat L\times(0,0,-mg),\qquad \hat L = \dfrac{\vec L}{|\vec L|}"]),
            ("Parámetros", "Se integra con RK4 6 s en 6000 pasos, partiendo de $\\vec L$ horizontal a lo largo de $x$.",
             [rf"h = \dfrac{{{tend:.0f}}}{{{n}}} = {sci(h,3)}\ \mathrm{{s}},\qquad \vec L_0 = ({big_l:.2f},\,0,\,0)\ \mathrm{{kg\,m^2/s}}"]),
            ("Primera evaluación de la pendiente", "Con $\\hat L = (1,0,0)$, el producto cruz es $\\hat x\\times(-\\hat z) = \\hat y$, por lo que $\\vec k_1$ apunta en $+y$ con magnitud $mgr$.",
             [rf"\vec k_1 = {_v(k1,5)}\ \mathrm{{N\,m}}"]),
            ("Pendientes intermedias y primer paso", "Para $\\vec k_2,\\vec k_3,\\vec k_4$ se normaliza el $\\vec L$ provisional; como el cambio en un paso es diminuto, la dirección casi no varía.",
             [rf"\vec k_2 = {_v(k2,5)},\quad \vec k_3 = {_v(k3,5)},\quad \vec k_4 = {_v(k4,5)}",
              rf"\vec L_1 = \vec L_0 + \dfrac h6(\vec k_1 + 2\vec k_2 + 2\vec k_3 + \vec k_4) = {_v(s1,8)}"]),
            ("Ángulo azimutal tras el primer paso", "En cada paso se obtiene $\\phi = \\operatorname{atan2}(L_y, L_x)$. Su incremento por paso debe ser $\\Omega h$.",
             [rf"\phi_1 = \arctan\dfrac{{L_y}}{{L_x}} = {sci(phi1,6)}\ \mathrm{{rad}},\qquad \dfrac{{\phi_1}}{{h}} = {phi1/h:.5f}\ \mathrm{{rad/s}}"]),
            ("Evolución y módulo de $\\vec L$", "Se repite hasta $t=6$ s. El módulo de $\\vec L$ se mantiene y el ángulo crece casi linealmente.",
             [rf"|\vec L(6\ \mathrm{{s}})| = {np.linalg.norm(y[-1]):.6f}\ \mathrm{{kg\,m^2/s}},\qquad \phi(6\ \mathrm{{s}}) = {phi[-1]:.5f}\ \mathrm{{rad}}"]),
            ("Ajuste lineal de $\\phi(t)$", "La pendiente de la recta de mínimos cuadrados $\\phi = \\Omega t + \\phi_0$ (numpy.polyfit, grado 1) da la precesión numérica; se repite con $\\omega=400$ rad/s.",
             [rf"\Omega_{{num}}(200) = {om_n:.5f}\ \mathrm{{rad/s}}\ \ (\text{{analítico }}{om:.5f}),\qquad \Omega_{{num}}(400) = {om2_n:.5f}\ \ (\text{{analítico }}{om2:.5f})",
              rf"\dfrac{{|\Omega_{{num}} - \Omega_{{an}}|}}{{\Omega_{{an}}}} = {sci(abs(om_n-om)/om,2)}"]),
        ],
    )


# ----------------------------------------------------------------------------- LadderFriction
def _ladder() -> dict:
    L, wl, wh, th, mu = c14.L_LAD, c14.W_LAD, c14.W_MAN, c14.THETA_DEG, c14.MU
    rth = np.deg2rad(th)
    dmax = c14.d_max_analytic()
    amin = c14.min_angle_analytic()
    dn, an = c14.d_max_numeric(), c14.min_angle_numeric()
    nfl = wl + wh
    fmax = mu * nfl
    tan = float(np.tan(rth))
    f25a = c14.ladder_forces_analytic(2.5)
    f25n = c14.ladder_forces_numeric(2.5)
    rhs3 = (wl * L / 2 + wh * 2.5) * np.cos(rth)
    # bisecciones
    lo, hi = 0.0, L
    filas = []
    for i in range(1, 7):
        mid = 0.5 * (lo + hi)
        fr = c14.ladder_forces_numeric(mid, th, L)
        ok = fr["f"] <= mu * fr["N_floor"]
        filas.append((i, lo, hi, mid, fr["f"], ok))
        if ok:
            lo = mid
        else:
            hi = mid
    tabla = r"\begin{array}{c|ccccc} k & d_{lo} & d_{hi} & d_{mid} & f\,(\mathrm{N}) & f\le f_{max}?\\ \hline " + r"\\ ".join(
        f"{i} & {a:.4f} & {b:.4f} & {c:.4f} & {f:.3f} & \\text{{{'sí' if ok else 'no'}}}" for i, a, b, c, f, ok in filas) + r"\end{array}"
    tabla = tabla.replace("c|ccccc", "c|ccccc")
    # angulo minimo: primera biseccion
    mida = 0.5 * (20.0 + 89.0)
    fra = c14.ladder_forces_numeric(L, mida, L)
    return dict(
        explanation=[
            "Una escalera apoyada en una pared lisa está en equilibrio estático: la suma de fuerzas y la suma de torques valen cero. La pared lisa solo puede empujar horizontalmente (no hay fricción vertical), mientras que el suelo ofrece una normal vertical y una fuerza de fricción horizontal que impide que el pie de la escalera resbale.",
            "Las tres ecuaciones de equilibrio ($\\sum F_x$, $\\sum F_y$ y $\\sum\\tau$ respecto al pie) permiten hallar las tres incógnitas $N_{suelo}$, $f$ y $N_{pared}$. Se elige el pie como eje para eliminar del torque a $N_{suelo}$ y a $f$, que pasan por ese punto. Lo decisivo es que la fricción necesaria es igual al empuje de la pared, y este crece a medida que el hombre sube, porque aumenta el torque de su peso respecto al pie.",
            "El suelo solo puede suministrar una fricción estática máxima $f_{max} = \\mu_s N_{suelo}$. Cuando la fricción requerida alcanza ese valor, la escalera está a punto de resbalar. De la igualdad se despeja la distancia máxima $d_{max}$ a la que el hombre puede subir por la escalera.",
            "Una escalera más vertical necesita menos fricción, porque el empuje de la pared disminuye con $\\cot\\theta$. Por eso existe un ángulo mínimo con el que se puede llegar hasta el extremo: es el valor de $\\theta$ en que $f = \\mu_s N_{suelo}$ con el hombre en $d = L$.",
        ],
        analytic_steps=[
            ("Diagrama de fuerzas y condiciones de equilibrio", "Sobre la escalera actúan: su peso $W_e$ en el centro (a $L/2$), el peso del hombre $W_h$ a una distancia $d$ del pie, la normal del suelo, la fricción en el pie y el empuje horizontal de la pared en el extremo superior.",
             [r"\sum F_y = 0:\ N_{suelo} = W_e + W_h,\qquad \sum F_x = 0:\ f = N_{pared}"]),
            ("Fuerza normal del suelo", "Del equilibrio vertical, con los datos del enunciado.",
             [rf"N_{{suelo}} = {wl:.0f} + {wh:.0f} = {nfl:.0f}\ \mathrm{{N}},\qquad f_{{max}} = \mu_s N_{{suelo}} = ({mu})({nfl:.0f}) = {fmax:.0f}\ \mathrm{{N}}"]),
            ("Torque respecto al pie", "Los brazos de palanca de los pesos son $(L/2)\\cos\\theta$ y $d\\cos\\theta$; el brazo de $N_{pared}$ es $L\\sin\\theta$.",
             [r"N_{pared}\,L\sin\theta = W_e\,\dfrac L2\cos\theta + W_h\,d\cos\theta",
              r"f = N_{pared} = \dfrac{(W_eL/2 + W_hd)\cos\theta}{L\sin\theta}"]),
            ("Ejemplo: fricción con el hombre en $d = 2.5$ m", "Se sustituyen valores en la expresión anterior para tener una idea de la fricción requerida.",
             [rf"f = \dfrac{{(({wl:.0f})({L}/2) + ({wh:.0f})(2.5))\cos {th:.0f}^\circ}}{{({L})\sin {th:.0f}^\circ}} = \dfrac{{{wl*L/2+wh*2.5:.1f}\cdot{np.cos(rth):.4f}}}{{{L*np.sin(rth):.4f}}} = {f25a['f']:.3f}\ \mathrm{{N}}"]),
            ("Condición de deslizamiento inminente", "Se iguala la fricción requerida con $\\mu_s N_{suelo}$ y se despeja $d$.",
             [r"\dfrac{(W_eL/2 + W_hd)\cos\theta}{L\sin\theta} = \mu_s(W_e+W_h)\;\Rightarrow\; d_{max} = \dfrac{\mu_s(W_e+W_h)L\tan\theta - W_eL/2}{W_h}"]),
            ("Distancia máxima", "Se sustituye con $\\tan 60^\\circ$.",
             [rf"d_{{max}} = \dfrac{{({mu})({nfl:.0f})({L})({tan:.4f}) - ({wl:.0f})({L})/2}}{{{wh:.0f}}} = \dfrac{{{mu*nfl*L*tan:.2f} - {wl*L/2:.1f}}}{{{wh:.0f}}} = {dmax:.3f}\ \mathrm{{m}}"]),
            ("Ángulo mínimo para llegar al extremo", "Con $d = L$ el torque da $\\tan\\theta_{min} = (W_e/2 + W_h)/(\\mu_s(W_e+W_h))$.",
             [rf"\tan\theta_{{min}} = \dfrac{{{wl/2:.0f} + {wh:.0f}}}{{({mu})({nfl:.0f})}} = {(wl/2+wh)/fmax:.4f}\;\Rightarrow\;\theta_{{min}} = {amin:.3f}^\circ"]),
        ],
        numeric_steps=[
            ("Sistema lineal de equilibrio", "El código plantea las tres ecuaciones como $A\\,\\vec x = \\vec b$ con incógnitas $\\vec x = (N_{suelo}, f, N_{pared})$ y lo resuelve con numpy.linalg.solve (eliminación gaussiana con pivoteo parcial).",
             [r"\begin{pmatrix}0&1&-1\\1&0&0\\0&0&L\sin\theta\end{pmatrix}\begin{pmatrix}N_{suelo}\\f\\N_{pared}\end{pmatrix}=\begin{pmatrix}0\\W_e+W_h\\(W_eL/2+W_hd)\cos\theta\end{pmatrix}"]),
            ("Resolver a mano para $d = 2.5$ m", "Las filas ya están casi triangulares: la fila 2 da $N_{suelo}$, la fila 3 da $N_{pared}$ y la fila 1 da $f = N_{pared}$.",
             [rf"b = (0,\ {nfl:.0f},\ {rhs3:.3f}),\qquad L\sin\theta = {L*np.sin(rth):.5f}",
              rf"N_{{suelo}} = {f25n['N_floor']:.1f}\ \mathrm{{N}},\quad N_{{pared}} = \dfrac{{{rhs3:.3f}}}{{{L*np.sin(rth):.5f}}} = {f25n['N_wall']:.3f}\ \mathrm{{N}},\quad f = {f25n['f']:.3f}\ \mathrm{{N}}"]),
            ("Plan de bisección para $d_{max}$", "Se busca la raíz de $g(d) = f(d) - \\mu_s N_{suelo}$ en $[0, L]$. En cada paso se prueba el punto medio: si $f\\le\\mu_s N_{suelo}$ la escalera aguanta y se sube el extremo inferior; si no, se baja el superior.",
             [rf"d_{{lo}}=0,\quad d_{{hi}}={L:.0f}\ \mathrm{{m}},\quad f_{{max}} = {fmax:.0f}\ \mathrm{{N}},\quad 60\ \text{{iteraciones}}"]),
            ("Primera bisección", "En $d = 2.5$ m la fricción requerida es menor que $f_{max}$, de modo que la escalera no resbala y se descarta la mitad inferior.",
             [rf"d_{{mid}} = {filas[0][3]:.2f}\ \mathrm{{m}}:\quad f = {filas[0][4]:.3f}\ \mathrm{{N}}\ \le\ {fmax:.0f}\ \mathrm{{N}}\;\Rightarrow\; d_{{lo}} = {filas[0][3]:.2f}"]),
            ("Siguientes iteraciones", "El intervalo se divide a la mitad cada vez. Tras seis pasos el tamaño del intervalo es $L/2^6$.",
             [tabla]),
            ("Convergencia y resultado", "Después de 60 iteraciones el intervalo mide $L/2^{60}\\approx 4\\times10^{-18}$ m, es decir, está al límite de la precisión de doble precisión.",
             [rf"d_{{max}}^{{num}} = {dn:.9f}\ \mathrm{{m}},\qquad d_{{max}}^{{an}} = {dmax:.9f}\ \mathrm{{m}},\qquad |\Delta| = {sci(abs(dn-dmax),2)}\ \mathrm{{m}}"]),
            ("Bisección sobre el ángulo", "Se repite con $d = L$ fijo y $\\theta\\in[20^\\circ, 89^\\circ]$. Si $\\theta$ medio exige demasiada fricción, hay que subir el ángulo.",
             [rf"\theta_{{mid}} = {mida:.2f}^\circ:\ f = {fra['f']:.3f}\ \mathrm{{N}}\ \le\ {fmax:.0f}\ \mathrm{{N}}\;\Rightarrow\;\theta_{{hi}} = {mida:.2f}^\circ",
              rf"\theta_{{min}}^{{num}} = {an:.7f}^\circ,\qquad \theta_{{min}}^{{an}} = {amin:.7f}^\circ,\qquad |\Delta| = {sci(abs(an-amin),2)}^\circ"]),
        ],
    )


# ----------------------------------------------------------------------------- BeamReactions
def _beam() -> dict:
    L, wb, wp = c14.BEAM_L, c14.BEAM_W, c14.PERSON_W
    x = 2.0
    ra, rb = c14.beam_reactions_analytic(x)
    tot = wb + wp
    p0 = c14.beam_reactions_numeric(x, 0.0)
    p3 = c14.beam_reactions_numeric(x, 3.0)
    p6 = c14.beam_reactions_numeric(x, 6.0)
    xs = [0.0, 1.5, 3.0, 4.5, 6.0]
    tab = r"\begin{array}{c|cc} x\,(\mathrm{m}) & R_A\,(\mathrm{N}) & R_B\,(\mathrm{N})\\ \hline " + r"\\ ".join(
        f"{xx:.1f} & {c14.beam_reactions_analytic(xx)[0]:.1f} & {c14.beam_reactions_analytic(xx)[1]:.1f}" for xx in xs) + r"\end{array}"
    b0 = wb * (L / 2 - 0.0) + wp * (x - 0.0)
    b3 = wb * (L / 2 - 3.0) + wp * (x - 3.0)
    b6 = wb * (L / 2 - 6.0) + wp * (x - 6.0)
    return dict(
        explanation=[
            "Una viga apoyada en sus extremos recibe dos fuerzas de reacción verticales, $R_A$ y $R_B$, que sostienen el peso de la viga y el de la persona. Es un problema estáticamente determinado: dos incógnitas y dos ecuaciones independientes (suma de fuerzas y suma de torques).",
            "El peso de la viga uniforme se concentra en su centro, y por simetría reparte la mitad en cada apoyo. La persona agrega una carga puntual cuya reacción se reparte según su posición: cuanto más cerca está de un apoyo, mayor es la fracción de su peso que sostiene ese apoyo, como en una palanca.",
            "Un resultado importante es que el torque se puede calcular respecto a cualquier punto: si el cuerpo está en equilibrio, la suma de torques es cero respecto a todos los puntos a la vez. Elegir un apoyo como pivote elimina una incógnita y simplifica la cuenta, pero cualquier otra elección conduce a la misma respuesta.",
            "Como comprobación de sentido común, la suma $R_A + R_B$ siempre es igual al peso total ($1100$ N) y cuando la persona está en un extremo ese apoyo sostiene casi todo su peso, mientras que cuando está en el centro ambos apoyos cargan igual.",
        ],
        analytic_steps=[
            ("Plantear las incógnitas y el equilibrio vertical", "Con $x$ medida desde el apoyo A, las fuerzas verticales son $R_A$, $R_B$, el peso de la viga $W_b$ (en $L/2$) y el de la persona $W_p$ (en $x$).",
             [rf"R_A + R_B = W_b + W_p = {wb:.0f} + {wp:.0f} = {tot:.0f}\ \mathrm{{N}}"]),
            ("Torques respecto al apoyo A", "Con A como pivote, $R_A$ no produce torque. El torque de $R_B$ (antihorario) compensa los pesos (horarios).",
             [r"R_B\,L = W_b\,\dfrac L2 + W_p\,x"]),
            ("Despejar $R_B$", "Se divide por $L$.",
             [r"R_B = \dfrac{W_b}{2} + W_p\,\dfrac xL"]),
            ("Despejar $R_A$", "Se resta $R_B$ del peso total.",
             [r"R_A = W_b + W_p - R_B = \dfrac{W_b}{2} + W_p\left(1 - \dfrac xL\right)"]),
            ("Sustituir para $x = 2$ m", "Se reemplazan los datos con sus unidades.",
             [rf"R_B = \dfrac{{{wb:.0f}}}{{2}} + ({wp:.0f})\dfrac{{{x:.0f}}}{{{L:.0f}}} = {wb/2:.1f} + {wp*x/L:.1f} = {rb:.1f}\ \mathrm{{N}}",
              rf"R_A = {wb/2:.1f} + ({wp:.0f})\left(1 - \dfrac{{{x:.0f}}}{{{L:.0f}}}\right) = {wb/2:.1f} + {wp*(L-x)/L:.1f} = {ra:.1f}\ \mathrm{{N}}"]),
            ("Comprobación y tabla de posiciones", "La suma debe dar el peso total y $R_A$ debe decrecer linealmente con $x$ mientras $R_B$ crece.",
             [rf"R_A + R_B = {ra:.1f} + {rb:.1f} = {ra+rb:.1f}\ \mathrm{{N}}", tab]),
        ],
        numeric_steps=[
            ("Formular el sistema 2x2", "Para un pivote arbitrario en $x_p$, las dos ecuaciones son $R_A + R_B = W_b + W_p$ y $-x_pR_A + (L - x_p)R_B = W_b(L/2 - x_p) + W_p(x - x_p)$. El código las resuelve con numpy.linalg.solve.",
             [r"\begin{pmatrix}1&1\\-x_p&L-x_p\end{pmatrix}\begin{pmatrix}R_A\\R_B\end{pmatrix}=\begin{pmatrix}W_b+W_p\\W_b(L/2-x_p)+W_p(x-x_p)\end{pmatrix}"]),
            ("Pivote en $x_p = 0$", "La segunda fila se reduce a $6R_B = 2600$; se despeja $R_B$ y luego $R_A$ de la fila 1.",
             [rf"\begin{{pmatrix}}1&1\\0&{L:.0f}\end{{pmatrix}}\begin{{pmatrix}}R_A\\R_B\end{{pmatrix}}=\begin{{pmatrix}}{tot:.0f}\\{b0:.0f}\end{{pmatrix}}",
              rf"R_B = \dfrac{{{b0:.0f}}}{{{L:.0f}}} = {p0[1]:.4f}\ \mathrm{{N}},\qquad R_A = {tot:.0f} - {p0[1]:.4f} = {p0[0]:.4f}\ \mathrm{{N}}"]),
            ("Pivote en $x_p = 3$ m (centro)", "Ahora los brazos de palanca cambian de signo, pero el sistema es igualmente de rango completo. Eliminación: se suma 3 veces la fila 1 a la fila 2.",
             [rf"\begin{{pmatrix}}1&1\\-3&3\end{{pmatrix}}\begin{{pmatrix}}R_A\\R_B\end{{pmatrix}}=\begin{{pmatrix}}{tot:.0f}\\{b3:.0f}\end{{pmatrix}}",
              rf"6R_B = {b3:.0f} + 3({tot:.0f})={b3+3*tot:.0f}\;\Rightarrow\;R_B = {p3[1]:.4f}\ \mathrm{{N}},\quad R_A = {p3[0]:.4f}\ \mathrm{{N}}"]),
            ("Pivote en $x_p = 6$ m (apoyo B)", "El torque de $R_B$ desaparece y queda $R_A$ despejado directamente: $-6R_A = b_2$.",
             [rf"\begin{{pmatrix}}1&1\\-6&0\end{{pmatrix}}\begin{{pmatrix}}R_A\\R_B\end{{pmatrix}}=\begin{{pmatrix}}{tot:.0f}\\{b6:.0f}\end{{pmatrix}}",
              rf"R_A = \dfrac{{{b6:.0f}}}{{-6}} = {p6[0]:.4f}\ \mathrm{{N}},\qquad R_B = {tot:.0f} - {p6[0]:.4f} = {p6[1]:.4f}\ \mathrm{{N}}"]),
            ("Independencia del pivote", "Los tres pivotes dan el mismo resultado, como exige el equilibrio: el torque total es nulo respecto a cualquier punto.",
             [rf"R_A:\ {p0[0]:.6f},\ {p3[0]:.6f},\ {p6[0]:.6f}\ \mathrm{{N}};\qquad R_B:\ {p0[1]:.6f},\ {p3[1]:.6f},\ {p6[1]:.6f}\ \mathrm{{N}}"]),
            ("Comparación con la solución analítica", "El error es solo el redondeo de la aritmética de doble precisión.",
             [rf"|R_A^{{num}}-R_A^{{an}}|\le {sci(max(abs(p[0]-ra) for p in (p0,p3,p6)),2)}\ \mathrm{{N}},\qquad |R_B^{{num}}-R_B^{{an}}|\le {sci(max(abs(p[1]-rb) for p in (p0,p3,p6)),2)}\ \mathrm{{N}}"]),
        ],
    )


# ----------------------------------------------------------------------------- KeplerOrbit
def _kepler() -> dict:
    a, e, gm = c15.A_ORBIT, c15.E_ORBIT, c15.GM
    T = c15.period_analytic()
    vp = c15.perihelion_speed()
    rp, ra_ = a * (1 - e), a * (1 + e)
    va = vp * rp / ra_
    b = a * np.sqrt(1 - e**2)
    dAdt = 0.5 * rp * vp
    area_ell = np.pi * a * b
    Tn = c15.period_numeric()
    t, y = c15.orbit_numeric(a, e, 1.3, 20000)
    n_per = 20000
    h = T / n_per
    rhs = lambda _t, s: np.array([s[2], s[3], -gm * s[0] / np.hypot(s[0], s[1]) ** 3, -gm * s[1] / np.hypot(s[0], s[1]) ** 3])  # noqa: E731
    y0 = np.array([rp, 0.0, 0.0, vp])
    k1, k2, k3, k4, y1 = _rk4(rhs, 0.0, y0, h)
    t8, y8 = c15.orbit_numeric(revolutions=1.0)
    area = c15.areal_velocity(t8, y8)
    en = c15.energy(y8)
    r = np.hypot(y[:, 0], y[:, 1])
    idx = [i for i in range(1, len(r) - 1) if r[i] < r[i - 1] and r[i] <= r[i + 1]][0]
    r0, r1, r2 = r[idx - 1], r[idx], r[idx + 1]
    shift = 0.5 * (r0 - r2) / (r0 - 2 * r1 + r2)
    e0 = float(c15.energy(y0[None, :])[0])
    return dict(
        explanation=[
            "La fuerza de atracción gravitacional del Sol sobre un planeta es central (dirigida siempre hacia el Sol) y varía como $1/r^2$. Una fuerza central no produce torque respecto al Sol, así que el momento angular del planeta se conserva; esa es la razón de la segunda ley de Kepler: el radio vector barre áreas iguales en tiempos iguales, $dA/dt = L/(2m)$ constante.",
            "La órbita es una elipse con el Sol en un foco. El semieje mayor $a$ y la excentricidad $e$ fijan las distancias extremas, $r_p = a(1-e)$ en el perihelio y $r_a = a(1+e)$ en el afelio. Para $e = 0.5$ el planeta pasa de 1 UA a 3 UA del Sol y su rapidez cambia en la misma proporción inversa, porque $r\\,v_\\perp$ es constante.",
            "La tercera ley establece que $T^2$ es proporcional a $a^3$, con una constante que depende solo de la masa del Sol: $T^2 = 4\\pi^2a^3/(GM)$. En unidades de UA y años, $GM = 4\\pi^2$ y la ley queda $T^2 = a^3$, que reproduce el año terrestre para $a = 1$ UA. Obsérvese que el periodo no depende de $e$, solo de $a$.",
            "También se conserva la energía mecánica $E = \\tfrac12v^2 - GM/r$, que es negativa en una órbita ligada y vale $-GM/(2a)$. Esa conservación y la de $dA/dt$ son dos invariantes con los que se puede comprobar la calidad de una simulación numérica.",
        ],
        analytic_steps=[
            ("Unidades y constante gravitacional", "Se trabaja en UA y años. La Tierra (a = 1 UA, T = 1 año) en órbita casi circular obliga a $GM = 4\\pi^2\\,\\mathrm{UA^3/año^2}$.",
             [rf"GM = 4\pi^2 = {gm:.4f}\ \mathrm{{UA^3/a\tilde{{n}}o^2}}"]),
            ("Tercera ley: periodo", "Con $T^2 = 4\\pi^2a^3/(GM) = a^3$ se despeja $T$.",
             [rf"T = a^{{3/2}} = ({a:.0f}\ \mathrm{{UA}})^{{3/2}} = {T:.4f}\ \text{{años}}"]),
            ("Distancias extremas", "Se calculan el perihelio y el afelio con la excentricidad dada.",
             [rf"r_p = a(1-e) = {a:.0f}(1-{e}) = {rp:.1f}\ \mathrm{{UA}},\qquad r_a = a(1+e) = {a:.0f}(1+{e}) = {ra_:.1f}\ \mathrm{{UA}}"]),
            ("Velocidad en el perihelio", "De la ecuación vis-viva $v^2 = GM(2/r - 1/a)$ con $r = a(1-e)$ resulta $v_p^2 = GM(1+e)/[a(1-e)]$.",
             [rf"v_p = \sqrt{{\dfrac{{GM(1+e)}}{{a(1-e)}}}} = \sqrt{{\dfrac{{({gm:.4f})(1+{e})}}{{({a:.0f})(1-{e})}}}} = \sqrt{{{gm*(1+e)/(a*(1-e)):.4f}}} = {vp:.3f}\ \mathrm{{UA/a\tilde{{n}}o}}"]),
            ("Velocidad en el afelio", "La conservación de $L$ en los extremos de la órbita ($r\\vec v\\perp$) da $r_pv_p = r_av_a$.",
             [rf"v_a = v_p\,\dfrac{{r_p}}{{r_a}} = ({vp:.3f})\dfrac{{{rp:.1f}}}{{{ra_:.1f}}} = {va:.3f}\ \mathrm{{UA/a\tilde{{n}}o}}"]),
            ("Segunda ley: velocidad areal", "El área barrida por unidad de tiempo en el perihelio es la mitad de $r_pv_p$ y debe ser igual en todo punto.",
             [rf"\dfrac{{dA}}{{dt}} = \tfrac12 r_pv_p = \tfrac12({rp:.1f})({vp:.3f}) = {dAdt:.3f}\ \mathrm{{UA^2/a\tilde{{n}}o}}"]),
            ("Comprobación con el área de la elipse", "El área total de la elipse dividida entre el periodo debe reproducir la velocidad areal.",
             [rf"b = a\sqrt{{1-e^2}} = {b:.4f}\ \mathrm{{UA}},\qquad \dfrac{{\pi ab}}{{T}} = \dfrac{{\pi({a:.0f})({b:.4f})}}{{{T:.4f}}} = {area_ell/T:.3f}\ \mathrm{{UA^2/a\tilde{{n}}o}}"]),
        ],
        numeric_steps=[
            ("Ecuaciones de movimiento", "Se integra el sistema de primer orden para $\\vec s = (x, y, v_x, v_y)$ con $\\ddot{\\vec r} = -GM\\,\\vec r/|\\vec r|^3$, sin usar la elipse.",
             [r"\dot x = v_x,\quad \dot y = v_y,\quad \dot v_x = -\dfrac{GM\,x}{r^3},\quad \dot v_y = -\dfrac{GM\,y}{r^3}"]),
            ("Condiciones iniciales y paso", "El planeta parte del perihelio en el eje $x$ con velocidad tangencial. Para medir el periodo el código usa 20000 pasos por órbita y 1.3 órbitas; las gráficas de áreas y energía usan 8000 pasos por órbita.",
             [rf"\vec s_0 = (r_p,\,0,\,0,\,v_p) = ({rp:.1f},\,0,\,0,\,{vp:.5f}),\qquad h = \dfrac{{T}}{{20000}} = {sci(h,5)}\ \text{{años}}"]),
            ("Primer paso RK4: pendientes", "En $\\vec s_0$, $r=r_p=1$ y la aceleración es $-GM/r_p^2$ en $x$. $k_2,k_3,k_4$ se calculan con estados provisionales a $h/2$ y $h$.",
             [rf"\vec k_1 = {_v(k1,5)}",
              rf"\vec k_2 = {_v(k2,5)},\qquad \vec k_3 = {_v(k3,5)}",
              rf"\vec k_4 = {_v(k4,5)}"]),
            ("Primer paso RK4: nuevo estado", "Se combinan las pendientes con la fórmula $\\vec s_1 = \\vec s_0 + \\tfrac h6(\\vec k_1 + 2\\vec k_2 + 2\\vec k_3 + \\vec k_4)$.",
             [rf"\vec s_1 = {_v(y1,8)}",
              rf"E_0 = \tfrac12 v_p^2 - \dfrac{{GM}}{{r_p}} = {e0:.6f}\ \mathrm{{UA^2/a\tilde{{n}}o^2}} = -\dfrac{{GM}}{{2a}} = {-gm/(2*a):.6f}"]),
            ("Repetir y vigilar las invariantes", "Con 8000 pasos por vuelta se integra una órbita completa. Las magnitudes que deben ser constantes se vigilan con su variación relativa.",
             [rf"\dfrac{{dA}}{{dt}}\Big|_{{inicial}} = {area[0]:.6f},\qquad \dfrac{{\Delta(dA/dt)}}{{|dA/dt|}} = {sci(float(np.ptp(area)/abs(area[0])),2)}",
              rf"\dfrac{{\Delta E}}{{|E|}} = {sci(float(np.ptp(en)/abs(en[0])),2)}"]),
            ("Medir el periodo: segundo paso por el perihelio", "Se detecta el primer mínimo local de $r(t)$ después del inicio (el paso 1 por el perihelio es t = 0, el siguiente corresponde a una vuelta completa). Su posición se afina con una parábola por tres puntos.",
             [rf"r_{{i-1}} = {r0:.9f},\quad r_i = {r1:.9f},\quad r_{{i+1}} = {r2:.9f}\ \mathrm{{UA}}",
              rf"\delta = \dfrac12\,\dfrac{{r_{{i-1}} - r_{{i+1}}}}{{r_{{i-1}} - 2r_i + r_{{i+1}}}} = {shift:.5f},\qquad T = t_i + \delta h = {t[idx]:.6f} + ({shift:.5f})({sci(h,5)})"]),
            ("Resultado y error", "Se compara con la tercera ley.",
             [rf"T_{{num}} = {Tn:.6f}\ \text{{años}},\qquad T_{{an}} = {T:.6f}\ \text{{años}},\qquad |\Delta T| = {sci(abs(Tn-T),2)}\ \text{{años}}"]),
        ],
    )


# ----------------------------------------------------------------------------- EscapeVelocity
def _escape_test(v0, mu, radius, max_steps=400000):
    """Replica del criterio de la biseccion: devuelve (escapa, pasos)."""
    s = np.array([radius, v0])
    dt = 2.0
    for k in range(max_steps):
        s = rk4_step(lambda _t, yy: np.array([yy[1], -mu / yy[0] ** 2]), 0.0, s, dt)
        if s[1] <= 0:
            return False, k + 1
        if s[0] > 1e4 * radius:
            return True, k + 1
        dt = min(1e5, max(2.0, 0.002 * s[0] / max(abs(s[1]), 1.0)))
    return True, max_steps


def _escape() -> dict:
    G_, M_, R_ = c15.G_N, c15.M_EARTH, c15.R_EARTH
    mu = G_ * M_
    v0 = 8000.0
    rm = c15.rmax_analytic(v0)
    rm10 = c15.rmax_analytic(10000.0)
    va = c15.escape_speed_analytic()
    vn = c15.escape_speed_numeric()
    e_s = 0.5 * v0**2 - mu / R_
    fr = v0**2 * R_ / (2 * mu)
    rn8 = c15.rmax_numeric(v0, t_end=40000.0, n=40000)
    rn10 = c15.rmax_numeric(10000.0, t_end=40000.0, n=40000)
    h = 1.0
    f = lambda _t, s: np.array([s[1], -mu / s[0] ** 2])  # noqa: E731
    k1, k2, k3, k4, s1 = _rk4(f, 0.0, [R_, v0], h)
    t, y = c15.radial_launch_numeric(v0, t_end=40000.0, n=40000)
    imax = int(np.argmax(y[:, 0]))
    lo, hi = 0.5 * np.sqrt(mu / R_), 2.0 * np.sqrt(mu / R_)
    filas = []
    for _ in range(4):
        mid = 0.5 * (lo + hi)
        esc, ns = _escape_test(mid, mu, R_)
        filas.append((lo, hi, mid, esc, ns))
        if esc:
            hi = mid
        else:
            lo = mid
    tab = r"\begin{array}{c|cccc} k & v_{lo}\,(\mathrm{m/s}) & v_{hi}\,(\mathrm{m/s}) & v_{mid}\,(\mathrm{m/s}) & \text{¿escapa?}\\ \hline " + r"\\ ".join(
        f"{i+1} & {a:.1f} & {b:.1f} & {c:.1f} & \\text{{{'sí' if esc else 'no'}}}" for i, (a, b, c, esc, ns) in enumerate(filas)) + r"\end{array}"
    return dict(
        explanation=[
            "Cerca de la superficie terrestre se suele usar la energía potencial $mgh$, pero esa aproximación falla cuando la altura es comparable al radio $R$ de la Tierra. En general la energía potencial gravitacional es $U(r) = -GMm/r$ (cero en el infinito y negativa en cualquier punto finito), y la energía mecánica $E = \\tfrac12mv^2 - GMm/r$ se conserva mientras solo actúe la gravedad.",
            "Si un proyectil se lanza radialmente con rapidez $v_0$, sube hasta que se detiene en un radio $r_{max}$, donde su energía cinética es cero. La conservación de $E$ entre la superficie y ese punto permite hallar $r_{max}$ sin integrar la ecuación de movimiento. Con $E<0$ el proyectil está ligado y regresa; con $E\\ge0$ no vuelve.",
            "La velocidad de escape es la rapidez mínima con la cual el proyectil llega al infinito con velocidad cero, o sea con $E = 0$. Resulta $v_{esc} = \\sqrt{2GM/R}$, que para la Tierra es unos 11.2 km/s, y no depende de la masa del proyectil. Es $\\sqrt2$ veces la velocidad de una órbita circular rasante.",
            "La expresión para $r_{max}$ muestra que la altura alcanzada crece muy rápido cuando $v_0$ se acerca a $v_{esc}$ (el denominador se anula). A 8 km/s, el proyectil sube más de un radio terrestre; la fórmula $v_0^2/2g$ daría una altura distinta, porque $g$ disminuye con la altura.",
        ],
        analytic_steps=[
            ("Datos", "Se usan la constante de gravitación, la masa y el radio de la Tierra.",
             [rf"G = {sci(G_,4)}\ \mathrm{{N\,m^2/kg^2}},\quad M = {sci(M_,4)}\ \mathrm{{kg}},\quad R = {sci(R_,4)}\ \mathrm{{m}}",
              rf"GM = {sci(mu,5)}\ \mathrm{{m^3/s^2}}"]),
            ("Conservación de la energía", "Entre la superficie ($r=R$, $v=v_0$) y el punto más alto ($r=r_{max}$, $v=0$) la energía total es la misma.",
             [r"\tfrac12 v_0^2 - \dfrac{GM}{R} = -\dfrac{GM}{r_{max}}"]),
            ("Despejar $r_{max}$", "Se pasa $GM/R$ al otro miembro y se invierte la ecuación.",
             [r"\dfrac{1}{r_{max}} = \dfrac{1}{R} - \dfrac{v_0^2}{2GM}\;\Rightarrow\; r_{max} = \dfrac{R}{1 - v_0^2R/(2GM)}"]),
            ("Evaluar para $v_0 = 8$ km/s", "Primero el cociente adimensional del denominador, luego el radio.",
             [rf"\dfrac{{v_0^2R}}{{2GM}} = \dfrac{{({sci(v0,3)})^2({sci(R_,4)})}}{{2({sci(mu,5)})}} = {fr:.4f}",
              rf"r_{{max}} = \dfrac{{R}}{{1-{fr:.4f}}} = {rm/R_:.3f}\,R = {sci(rm,4)}\ \mathrm{{m}},\qquad h_{{max}} = r_{{max}}-R = {(rm-R_)/1000:.0f}\ \mathrm{{km}}"]),
            ("Otro caso: $v_0 = 10$ km/s", "El mismo procedimiento muestra el crecimiento rápido cerca del escape.",
             [rf"\dfrac{{v_0^2R}}{{2GM}} = {1e8*R_/(2*mu):.4f},\qquad r_{{max}} = {rm10/R_:.3f}\,R"]),
            ("Condición de escape", "Con $E = 0$ el proyectil llega al infinito ($r_{max}\\to\\infty$, denominador nulo).",
             [r"\tfrac12 v_{esc}^2 - \dfrac{GM}{R} = 0\;\Rightarrow\; v_{esc} = \sqrt{\dfrac{2GM}{R}}"]),
            ("Velocidad de escape terrestre", "Se sustituye con los datos.",
             [rf"v_{{esc}} = \sqrt{{\dfrac{{2({sci(mu,5)})}}{{{sci(R_,4)}}}}} = \sqrt{{{sci(2*mu/R_,5)}}} = {sci(va,5)}\ \mathrm{{m/s}} = {va/1000:.3f}\ \mathrm{{km/s}}"]),
        ],
        numeric_steps=[
            ("Ecuación a integrar", "Para el movimiento radial se integra $\\ddot r = -GM/r^2$ como sistema de primer orden en $(r, v)$, sin usar la conservación de la energía.",
             [r"\dot r = v,\qquad \dot v = -\dfrac{GM}{r^2}"]),
            ("Parámetros para $r_{max}$", "Se integra con RK4 de 0 a 40000 s en 40000 pasos; la condición inicial es la superficie con $v_0 = 8$ km/s.",
             [rf"h = {h:.0f}\ \mathrm{{s}},\qquad (r_0, v_0) = ({sci(R_,4)}\ \mathrm{{m}},\ {v0:.0f}\ \mathrm{{m/s}})"]),
            ("Primer paso RK4", "Se evalúan las pendientes $(\\dot r, \\dot v)$ en los cuatro puntos del paso.",
             [rf"\vec k_1 = {_v(k1,6)},\qquad \vec k_2 = {_v(k2,6)}",
              rf"\vec k_3 = {_v(k3,6)},\qquad \vec k_4 = {_v(k4,6)}",
              rf"(r_1, v_1) = (r_0, v_0) + \tfrac h6(\vec k_1+2\vec k_2+2\vec k_3+\vec k_4) = ({s1[0]:.4f}\ \mathrm{{m}},\ {s1[1]:.5f}\ \mathrm{{m/s}})"]),
            ("Buscar el máximo", "Tras 40000 pasos la altura sube, se frena y empieza a descender; el código toma el máximo de $r$ en la trayectoria.",
             [rf"t_{{max}} \approx {t[imax]:.0f}\ \mathrm{{s}},\qquad r_{{max}}^{{num}} = {sci(float(y[imax,0]),6)}\ \mathrm{{m}} = {y[imax,0]/R_:.4f}\,R",
              rf"r_{{max}}^{{an}} = {rm/R_:.4f}\,R;\qquad 10\ \mathrm{{km/s}}:\ {rn10/R_:.4f}\,R\ \text{{(num.)}},\ {rm10/R_:.4f}\,R\ \text{{(an.)}}"]),
            ("Bisección para la velocidad de escape", "Se acota $v_{esc}$ entre $\\tfrac12\\sqrt{GM/R}$ y $2\\sqrt{GM/R}$. Para cada $v_0$ de prueba se integra con paso adaptativo $h = \\max(2,\\,0.002\\,r/|v|)$ hasta que $v\\le0$ (no escapa) o $r>10^4R$ (escapa).",
             [rf"v_{{lo}} = {filas[0][0]:.1f}\ \mathrm{{m/s}},\qquad v_{{hi}} = {filas[0][1]:.1f}\ \mathrm{{m/s}},\qquad 45\ \text{{iteraciones}}"]),
            ("Primeras iteraciones", "En cada paso se prueba el punto medio. Si escapa, ese valor pasa a ser el nuevo extremo superior; si no, el inferior.",
             [tab]),
            ("Resultado y error", "El intervalo inicial de longitud $\\approx 1.5\\sqrt{GM/R}$ se reduce $2^{45}$ veces. Existe un pequeño sesgo del criterio ($r > 10^4R$ tolera una velocidad ligeramente inferior a la de escape).",
             [rf"v_{{esc}}^{{num}} = {vn:.3f}\ \mathrm{{m/s}},\qquad v_{{esc}}^{{an}} = {va:.3f}\ \mathrm{{m/s}},\qquad \dfrac{{|\Delta|}}{{v}} = {sci(abs(vn-va)/va,2)}"]),
        ],
    )


# ----------------------------------------------------------------------------- PressureDepth
def _pressure() -> dict:
    p0, g, rho = c16.P0, c16.G, c16.RHO_WATER
    M, R, T = c16.M_AIR, c16.R_GAS, c16.T_AIR
    d = 100.0
    pw = c16.water_pressure(d)
    kk = M * g / (R * T)
    p5 = float(c16.atmosphere_analytic(5000.0))
    p10 = float(c16.atmosphere_analytic(10000.0))
    pc5 = float(c16.atmosphere_constant_density(5000.0))
    rho0 = p0 * M / (R * T)
    hh, pn = c16.atmosphere_numeric()
    p5n = float(np.interp(5000.0, hh, pn))
    n, hend = 200, 10000.0
    step = hend / n
    f = lambda _h, y: np.array([-y[0] * kk])  # noqa: E731
    k1, k2, k3, k4, y1 = _rk4(f, 0.0, [p0], step)
    z = kk * step
    fac = 1 - z + z**2 / 2 - z**3 / 6 + z**4 / 24
    tab = r"\begin{array}{c|cc} h\,(\mathrm{m}) & p_{num}\,(\mathrm{Pa}) & p_{an}\,(\mathrm{Pa})\\ \hline " + r"\\ ".join(
        f"{hh[i]:.0f} & {pn[i]:.2f} & {float(c16.atmosphere_analytic(hh[i])):.2f}" for i in (1, 2, 20, 100, 200)) + r"\end{array}"
    return dict(
        explanation=[
            "En un fluido en reposo, cada capa debe sostener el peso del fluido que tiene encima. Al considerar un elemento horizontal delgado de espesor $dy$, el equilibrio vertical da $dp = -\\rho g\\,dy$: la presión aumenta al descender y disminuye al subir. Esta ecuación vale para cualquier fluido en reposo, sea líquido o gas.",
            "Para un líquido casi incompresible, $\\rho$ es constante y la ecuación se integra directamente: $p = p_0 + \\rho g h$. Cada 10 m de agua añaden aproximadamente una atmósfera de presión, de manera que a 100 m la presión absoluta es unas 10.7 atm, una cifra relevante para el buceo y los submarinos.",
            "Un gas es muy compresible: su densidad depende de la presión. Si se modela el aire como gas ideal a temperatura constante, $\\rho = pM/(RT)$ y la ecuación queda $dp/dh = -(Mg/RT)\\,p$, cuya solución es una exponencial decreciente, $p = p_0e^{-Mgh/RT}$. La escala de altura es $RT/(Mg)\\approx 8.4$ km.",
            "El modelo isotermo es aproximado (la temperatura real baja con la altura), pero capta lo esencial. Si por error se supusiera densidad constante, la presión sería lineal y daría un valor demasiado bajo a 5000 m, porque ignoraría que el aire se hace menos denso al subir y, por tanto, pesa menos por metro.",
        ],
        analytic_steps=[
            ("Ecuación fundamental de la hidrostática", "El equilibrio de un elemento de fluido de espesor $dy$ relaciona la presión con la densidad.",
             [r"\dfrac{dp}{dy} = -\rho g"]),
            ("Líquido a densidad constante", "Se integra desde la superficie (presión $p_0$) hasta la profundidad $h$, con $y$ creciendo hacia arriba.",
             [r"p(h) = p_0 + \rho g h"]),
            ("Presión a 100 m en agua", "Se sustituye $p_0$, $\\rho$ y $g$.",
             [rf"p = {p0:.0f}\ \mathrm{{Pa}} + ({rho:.0f}\ \mathrm{{kg/m^3}})({g}\ \mathrm{{m/s^2}})({d:.0f}\ \mathrm{{m}}) = {p0:.0f} + {rho*g*d:.0f} = {pw:.0f}\ \mathrm{{Pa}} = {pw/1e6:.3f}\ \mathrm{{MPa}}",
              rf"\dfrac{{p}}{{p_0}} = {pw/p0:.2f}\ \mathrm{{atm}}"]),
            ("Atmósfera: densidad del gas ideal", "Con temperatura constante, $\\rho = pM/(RT)$. Se sustituye en la ecuación fundamental.",
             [r"\dfrac{dp}{dh} = -\dfrac{Mg}{RT}\,p"]),
            ("Resolver por separación de variables", "Se escribe $dp/p = -(Mg/RT)\\,dh$ y se integra desde $h=0$ ($p=p_0$).",
             [r"\ln\dfrac{p}{p_0} = -\dfrac{Mg}{RT}\,h\;\Rightarrow\; p = p_0\,e^{-Mgh/(RT)}"]),
            ("Coeficiente del exponente", "Se evalúa con $M = 0.02897$ kg/mol, $R = 8.314$ J/(mol K) y $T = 288.15$ K (15 °C).",
             [rf"\dfrac{{Mg}}{{RT}} = \dfrac{{({M})({g})}}{{({R})({T})}} = {sci(kk,5)}\ \mathrm{{m^{{-1}}}},\qquad \dfrac{{RT}}{{Mg}} = {1/kk/1000:.3f}\ \mathrm{{km}}"]),
            ("Presión a 5000 m", "Se calcula el exponente y se evalúa la exponencial.",
             [rf"p(5000) = {p0:.0f}\,\exp\!\left(-({sci(kk,5)})(5000)\right) = {p0:.0f}\,e^{{-{kk*5000:.4f}}} = {p5:.0f}\ \mathrm{{Pa}} = {p5/1000:.2f}\ \mathrm{{kPa}}"]),
            ("Contraste con densidad constante", "Con $\\rho_0 = p_0M/(RT)$ fijo la presión disminuiría linealmente.",
             [rf"\rho_0 = {rho0:.4f}\ \mathrm{{kg/m^3}},\qquad p_{{cte}}(5000) = {p0:.0f} - ({rho0:.4f})({g})(5000) = {pc5/1000:.2f}\ \mathrm{{kPa}}\ \ (\text{{vs. }}{p5/1000:.2f})"]),
        ],
        numeric_steps=[
            ("Ecuación diferencial a integrar", "El código integra $dp/dh = -\\rho(p)g$ con $\\rho = pM/(RT)$ usando RK4, sin la fórmula exponencial.",
             [rf"\dfrac{{dp}}{{dh}} = f(p) = -\kappa\,p,\qquad \kappa = \dfrac{{Mg}}{{RT}} = {sci(kk,5)}\ \mathrm{{m^{{-1}}}}"]),
            ("Parámetros", "Se avanza desde $h=0$ hasta $10$ km en 200 pasos iguales, con $p(0)=101325$ Pa.",
             [rf"\Delta h = \dfrac{{{hend:.0f}}}{{{n}}} = {step:.0f}\ \mathrm{{m}},\qquad \kappa\,\Delta h = {z:.6f}"]),
            ("Primer paso: las cuatro pendientes", "Como $f$ solo depende de $p$, cada pendiente usa la presión provisional del subpaso anterior.",
             [rf"k_1 = -\kappa p_0 = {k1[0]:.5f}\ \mathrm{{Pa/m}}",
              rf"k_2 = -\kappa\left(p_0 + \tfrac{{\Delta h}}{{2}}k_1\right) = {k2[0]:.5f},\qquad k_3 = -\kappa\left(p_0 + \tfrac{{\Delta h}}{{2}}k_2\right) = {k3[0]:.5f}",
              rf"k_4 = -\kappa\left(p_0 + \Delta h\,k_3\right) = {k4[0]:.5f}"]),
            ("Primer valor actualizado", "Se combinan las pendientes. En este problema lineal es equivalente a multiplicar por los cuatro primeros términos de $e^{-z}$.",
             [rf"p_1 = p_0 + \dfrac{{\Delta h}}{{6}}(k_1+2k_2+2k_3+k_4) = {y1[0]:.4f}\ \mathrm{{Pa}}",
              rf"p_1/p_0 = 1 - z + \dfrac{{z^2}}{{2}} - \dfrac{{z^3}}{{6}} + \dfrac{{z^4}}{{24}} = {fac:.9f}"]),
            ("Iteraciones siguientes", "El paso se repite 200 veces. La tabla compara con la exponencial analítica en algunas alturas.",
             [tab]),
            ("Interpolar en 5000 m", "5000 m coincide con el nodo 100, pero el código interpola linealmente (numpy.interp) en la malla, de modo que lo calculado es exactamente el valor del nodo.",
             [rf"p_{{num}}(5000) = {p5n/1000:.5f}\ \mathrm{{kPa}},\qquad p_{{an}}(5000) = {p5/1000:.5f}\ \mathrm{{kPa}}"]),
            ("Error y presión a 10 km", "El error de RK4 por paso es $O(\\Delta h^5)$ y, acumulado, $O(\\Delta h^4)$: muy pequeño.",
             [rf"p_{{num}}(10\ \mathrm{{km}}) = {pn[-1]/1000:.5f}\ \mathrm{{kPa}},\quad p_{{an}}(10\ \mathrm{{km}}) = {p10/1000:.5f}\ \mathrm{{kPa}}",
              rf"\dfrac{{|p_{{num}}-p_{{an}}|}}{{p_{{an}}}} = {sci(abs(pn[-1]-p10)/p10,2)}"]),
        ],
    )


# ----------------------------------------------------------------------------- FloatingCylinder
def _cylinder() -> dict:
    rho, A, Lc, rf_, g = c16.RHO_OBJ, c16.AREA, c16.LEN, c16.RHO_WATER, c16.G
    m = c16.MASS
    h_a = c16.submerged_depth_analytic()
    h_b = c16.equilibrium_bisection()
    w = c16.omega_analytic()
    T_a = 2 * np.pi / w
    T_n = c16.period_numeric()
    mg = m * g
    kk = rf_ * g * A
    lo, hi = 0.0, Lc
    filas = []
    for i in range(1, 7):
        mid = 0.5 * (lo + hi)
        b = c16.buoyancy(mid)
        filas.append((i, lo, hi, mid, b, b < mg))
        if b < mg:
            lo = mid
        else:
            hi = mid
    tab = r"\begin{array}{c|ccccc} k & h_{lo} & h_{hi} & h_{mid} & B(h_{mid})\,(\mathrm{N}) & B<mg?\\ \hline " + r"\\ ".join(
        f"{i} & {a:.5f} & {bb:.5f} & {c:.5f} & {b:.4f} & \\text{{{'sí' if s else 'no'}}}" for i, a, bb, c, b, s in filas) + r"\end{array}"
    n, tend, y0 = 40000, 2.0, 0.02
    hstep = tend / n
    rhs = lambda _t, s: np.array([s[1], (m * g - c16.buoyancy(s[0])) / m])  # noqa: E731
    k1, k2, k3, k4, s1 = _rk4(rhs, 0.0, [h_a + y0, 0.0], hstep)
    t, s = c16.bobbing_numeric(0.0, y0, tend, n)
    dd = s[:, 0] - h_a
    cross = [i for i in range(1, len(dd)) if dd[i - 1] > 0 >= dd[i]]
    c1, c2 = cross[0], cross[1]
    t1 = t[c1] - dd[c1] * (t[c1] - t[c1 - 1]) / (dd[c1] - dd[c1 - 1])
    t2 = t[c2] - dd[c2] * (t[c2] - t[c2 - 1]) / (dd[c2] - dd[c2 - 1])
    return dict(
        explanation=[
            "El principio de Arquímedes establece que un cuerpo sumergido, total o parcialmente, recibe un empuje vertical hacia arriba igual al peso del fluido que desaloja: $B = \\rho_f g V_{sum}$. El origen físico es la diferencia de presiones entre la cara inferior y la superior del cuerpo, que crece con la profundidad.",
            "Un cuerpo flota en equilibrio cuando el empuje iguala a su peso. Para un cilindro vertical de sección $A$ y longitud $L$, el volumen sumergido es $Ah$ y el peso es $\\rho A L g$. Cancelando $A$ y $g$ resulta $h/L = \\rho/\\rho_f$: la fracción sumergida es igual al cociente de densidades. La madera con $\\rho=600$ kg/m$^3$ queda sumergida al 60 %.",
            "Si se empuja el cilindro una pequeña distancia $\\delta$ hacia abajo, el empuje aumenta en $\\rho_fgA\\,\\delta$ y aparece una fuerza neta restauradora proporcional al desplazamiento, $F = -\\rho_fgA\\,\\delta$. Es la ley de Hooke con una constante efectiva $k = \\rho_fgA$ y masa $m$, de modo que el cilindro oscila armónicamente con $\\omega = \\sqrt{k/m}$.",
            "El modelo desprecia la viscosidad del agua y el oleaje generado, por lo que es válido para amplitudes pequeñas (el cilindro no debe salir del agua ni hundirse del todo). Con los datos del problema el periodo es del orden de un segundo, un valor fácil de comprobar en un experimento casero.",
        ],
        analytic_steps=[
            ("Datos del cilindro", "Se calculan masa y peso a partir de la densidad y de las dimensiones.",
             [rf"m = \rho A L = ({rho:.0f})({A})({Lc}) = {m:.2f}\ \mathrm{{kg}},\qquad mg = {m:.2f}\times{g} = {mg:.3f}\ \mathrm{{N}}"]),
            ("Condición de equilibrio", "El empuje con profundidad sumergida $h$ iguala al peso.",
             [r"\rho_f\,g\,A\,h = \rho\,A\,L\,g\;\Rightarrow\; h = \dfrac{\rho}{\rho_f}\,L"]),
            ("Profundidad de equilibrio", "Se sustituyen densidades y longitud.",
             [rf"h = \dfrac{{{rho:.0f}\ \mathrm{{kg/m^3}}}}{{{rf_:.0f}\ \mathrm{{kg/m^3}}}}({Lc}\ \mathrm{{m}}) = {h_a:.3f}\ \mathrm{{m}}"]),
            ("Fuerza neta al desplazar $\\delta$", "Con la profundidad $h+\\delta$ el empuje es $\\rho_fgA(h+\\delta)$; el peso no cambia. Se descuenta la parte que ya compensa el peso.",
             [r"F = mg - \rho_fgA(h+\delta) = -\rho_fgA\,\delta"]),
            ("Constante efectiva", "La fuerza restauradora es lineal, con constante $k$.",
             [rf"k = \rho_f g A = ({rf_:.0f})({g})({A}) = {kk:.2f}\ \mathrm{{N/m}}"]),
            ("Frecuencia angular", "La ecuación del movimiento es $m\\ddot\\delta = -k\\delta$.",
             [rf"\omega = \sqrt{{\dfrac{{k}}{{m}}}} = \sqrt{{\dfrac{{{kk:.2f}}}{{{m:.2f}}}}} = \sqrt{{{kk/m:.4f}}} = {w:.3f}\ \mathrm{{rad/s}}"]),
            ("Periodo", "Se pasa de frecuencia angular a periodo. El resultado no depende de la amplitud.",
             [rf"T = \dfrac{{2\pi}}{{\omega}} = \dfrac{{2\pi}}{{{w:.3f}}} = {T_a:.3f}\ \mathrm{{s}}",
              rf"T = 2\pi\sqrt{{\dfrac{{\rho L}}{{\rho_f g}}}} = 2\pi\sqrt{{\dfrac{{h}}{{g}}}}"]),
        ],
        numeric_steps=[
            ("Equilibrio por bisección", "Se busca la raíz de $B(h) - mg = 0$ en $[0, L]$. $B(h) = \\rho_fgAh$ es creciente, así que si $B<mg$ la raíz está a la derecha del punto medio.",
             [rf"mg = {mg:.3f}\ \mathrm{{N}},\qquad h_{{lo}} = 0,\ h_{{hi}} = {Lc}\ \mathrm{{m}},\qquad 60\ \text{{iteraciones}}"]),
            ("Primeras iteraciones", "La primera: $h_{mid}=0.25$ m da $B = 24.525$ N, menor que el peso, así que se sube $h_{lo}$. Cada paso divide el intervalo a la mitad.",
             [tab]),
            ("Resultado de la bisección", "Tras 60 pasos el intervalo es del orden de $10^{-19}$ m.",
             [rf"h_{{num}} = {h_b:.9f}\ \mathrm{{m}},\qquad h_{{an}} = {h_a:.9f}\ \mathrm{{m}},\qquad |\Delta| = {sci(abs(h_b-h_a),2)}\ \mathrm{{m}}"]),
            ("Oscilaciones: ecuación y parámetros", "Con $y$ = profundidad sumergida, $m\\ddot y = mg - \\rho_fgAy$. Se escribe como sistema $(y, v)$ y se integra con RK4, partiendo de $y_0 = h_{eq}+0.02$ m en reposo.",
             [rf"\dot y = v,\quad \dot v = g - \dfrac{{\rho_fgA}}{{m}}\,y;\qquad h = \dfrac{{{tend:.0f}}}{{{n}}} = {sci(hstep,3)}\ \mathrm{{s}},\quad y_0 = {h_a+y0:.2f}\ \mathrm{{m}}"]),
            ("Primer paso RK4", "En $t=0$ la fuerza neta es $-k\\cdot0.02$, así que la aceleración inicial es $-\\omega^2(0.02)$.",
             [rf"\vec k_1 = {_v(k1,5)},\qquad \vec k_2 = {_v(k2,5)}",
              rf"\vec k_3 = {_v(k3,5)},\qquad \vec k_4 = {_v(k4,5)}",
              rf"(y_1, v_1) = ({s1[0]:.9f}\ \mathrm{{m}},\ {s1[1]:.7f}\ \mathrm{{m/s}})"]),
            ("Detectar los cruces ascendentes", "Se sigue $\\delta = y - h_{eq}$ durante 2 s. Un cruce es el paso en que $\\delta$ pasa de positivo a no positivo; se interpola linealmente entre las dos muestras para refinar el instante.",
             [rf"t_1 = t_{{i}} - \delta_i\,\dfrac{{t_i - t_{{i-1}}}}{{\delta_i - \delta_{{i-1}}}} = {t1:.6f}\ \mathrm{{s}},\qquad t_2 = {t2:.6f}\ \mathrm{{s}}"]),
            ("Periodo y comparación", "El periodo es la diferencia entre dos cruces consecutivos de igual sentido. Se compara con $2\\pi/\\omega$.",
             [rf"T_{{num}} = t_2 - t_1 = {T_n:.6f}\ \mathrm{{s}},\qquad T_{{an}} = {T_a:.6f}\ \mathrm{{s}},\qquad |\Delta| = {sci(abs(T_n-T_a),2)}\ \mathrm{{s}}"]),
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
