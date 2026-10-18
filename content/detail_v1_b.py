"""Explicacion y pasos a mano: Volumen 1, capitulos 5 a 8.

Escenas: StaticEquilibrium, InclinePulley, FrictionIncline, DragFall,
SpringWork, WorkEnergyTheorem, PendulumEnergy, SpringFriction.
Todos los numeros se calculan con las funciones de `physics/` o replicando su algoritmo.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import ch05_newton as c5
from physics import ch06_dynamics as c6
from physics import ch07_work_energy as c7
from physics import ch08_energy_conservation as c8
from physics.integrators import euler_step, rk4_step


def _ks(f, t, y, h):
    """Etapas de RK4 (igual que physics.integrators.rk4_step) y el valor y1."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return k1, k2, k3, k4, y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _v(k, d=5):
    return rf"({sci(float(k[0]), d)},\ {sci(float(k[1]), d)})"


# ------------------------------------------------------------------ capitulo 5
def _static() -> dict:
    w, al, be = c5.WEIGHT, c5.ALPHA_DEG, c5.BETA_DEG
    a, b = np.deg2rad(al), np.deg2rad(be)
    t1, t2 = c5.tensions_analytic()
    n1, n2 = c5.tensions_numeric()
    sab = np.sin(a + b)
    tana = np.tan(a)
    piv = np.sin(b) + tana * np.cos(b)
    t2m = w / piv
    t1m = np.cos(b) * t2m / np.cos(a)
    res = c5.net_force(n1, n2)
    sweep = {x: c5.tensions_numeric(w, x, be) for x in (15.0, 30.0, 60.0)}
    return dict(
        explanation=[
            "Una partícula está en equilibrio cuando su aceleración es nula, y por la segunda ley eso ocurre si y solo si la suma vectorial de las fuerzas es cero. Esa única condición vectorial equivale a dos ecuaciones escalares independientes, una por cada eje, y con ellas se pueden hallar dos incógnitas.",
            "En este problema el letrero es la partícula y sobre él actúan tres fuerzas: su peso vertical hacia abajo y las tensiones de dos cables, que solo pueden jalar a lo largo de la dirección del cable. Dibujar el diagrama de cuerpo libre y elegir ejes horizontal y vertical convierte el problema en un sistema lineal de $2\\times2$.",
            "La componente horizontal obliga a que los dos cables se jalen mutuamente de lado con la misma magnitud, y la componente vertical obliga a que las componentes hacia arriba sumen exactamente el peso. Por eso un cable más inclinado carga una tensión mayor que el peso mismo: los cables casi horizontales son muy desfavorables.",
            "El método solo vale si el cuerpo realmente está en reposo (o con velocidad constante) y si los cables son ideales, sin masa y siempre tensos. Como orden de magnitud, con ángulos de 30° y 45° las tensiones salen del mismo orden que el peso de 500 N, pero no iguales a la mitad de él como ocurriría con cables verticales.",
        ],
        analytic_steps=[
            ("Diagrama de cuerpo libre", "Se aísla el letrero y se dibujan las tres fuerzas que actúan sobre él. El cable izquierdo forma $\\alpha$ con la horizontal y el derecho forma $\\beta$.",
             [rf"W = {fx(w,1)}\ \mathrm{{N}},\qquad \alpha = {fx(al,0)}^\circ,\qquad \beta = {fx(be,0)}^\circ"]),
            ("Plantear equilibrio en x", "Con $a=0$, las componentes horizontales de las tensiones deben cancelarse: el cable izquierdo jala hacia la izquierda y el derecho hacia la derecha.",
             [r"\sum F_x = 0:\qquad -T_1\cos\alpha + T_2\cos\beta = 0\;\Rightarrow\; T_2 = T_1\,\dfrac{\cos\alpha}{\cos\beta}"]),
            ("Plantear equilibrio en y", "Las componentes verticales de ambas tensiones sostienen el peso.",
             [r"\sum F_y = 0:\qquad T_1\sin\alpha + T_2\sin\beta - W = 0"]),
            ("Despejar T₁ simbólicamente", r"Se sustituye $T_2$ de la primera ecuación en la segunda y se factoriza $T_1$. Se usa $\sin\alpha\cos\beta+\cos\alpha\sin\beta=\sin(\alpha+\beta)$.",
             [r"T_1\left(\sin\alpha + \dfrac{\cos\alpha\sin\beta}{\cos\beta}\right) = W\;\Rightarrow\; T_1 = \dfrac{W\cos\beta}{\sin(\alpha+\beta)},\qquad T_2 = \dfrac{W\cos\alpha}{\sin(\alpha+\beta)}"]),
            ("Sustituir cifras en T₁", "Primero el denominador común, $\\sin(30^\\circ+45^\\circ)=\\sin75^\\circ$.",
             [rf"\sin(\alpha+\beta) = \sin 75^\circ = {sab:.6f}",
              rf"T_1 = \dfrac{{({fx(w,1)}\ \mathrm{{N}})\cos 45^\circ}}{{{sab:.6f}}} = \dfrac{{({fx(w,1)})({np.cos(b):.6f})}}{{{sab:.6f}}} = {t1:.2f}\ \mathrm{{N}}"]),
            ("Sustituir cifras en T₂", "Mismo denominador, con el coseno del ángulo del otro cable.",
             [rf"T_2 = \dfrac{{({fx(w,1)}\ \mathrm{{N}})\cos 30^\circ}}{{{sab:.6f}}} = \dfrac{{({fx(w,1)})({np.cos(a):.6f})}}{{{sab:.6f}}} = {t2:.2f}\ \mathrm{{N}}"]),
            ("Comprobar el resultado", "Se verifican ambas ecuaciones de equilibrio con los valores obtenidos; deben dar cero.",
             [rf"\sum F_x = -({t1:.2f})({np.cos(a):.5f}) + ({t2:.2f})({np.cos(b):.5f}) \approx 0",
              rf"\sum F_y = ({t1:.2f})({np.sin(a):.5f}) + ({t2:.2f})({np.sin(b):.5f}) - {fx(w,1)} \approx 0"]),
        ],
        numeric_steps=[
            ("Escribir el sistema como matriz", r"Las dos ecuaciones de equilibrio son lineales en $(T_1,T_2)$ y se agrupan como $M\,\vec T=\vec b$. El código construye exactamente esta matriz con los cosenos y senos de los ángulos.",
             [rf"\begin{{pmatrix}}\cos\alpha & -\cos\beta\\ \sin\alpha & \sin\beta\end{{pmatrix}}\begin{{pmatrix}}T_1\\T_2\end{{pmatrix}}=\begin{{pmatrix}}0\\W\end{{pmatrix}}",
              rf"M = \begin{{pmatrix}}{np.cos(a):.6f} & {-np.cos(b):.6f}\\ {np.sin(a):.6f} & {np.sin(b):.6f}\end{{pmatrix}},\qquad \vec b = \begin{{pmatrix}}0\\{fx(w,1)}\end{{pmatrix}}"]),
            ("Eliminación hacia adelante", r"`numpy.linalg.solve` hace una factorización LU, que a mano es eliminación gaussiana. Se resta a la fila 2 la fila 1 multiplicada por $m_{21}=\sin\alpha/\cos\alpha$ para anular el elemento bajo la diagonal.",
             [rf"m_{{21}} = \dfrac{{{np.sin(a):.6f}}}{{{np.cos(a):.6f}}} = {tana:.6f}",
              rf"\left(\sin\beta + m_{{21}}\cos\beta\right)T_2 = W - m_{{21}}\cdot 0\;\Rightarrow\;({np.sin(b):.6f} + {tana:.6f}\cdot{np.cos(b):.6f})\,T_2 = {fx(w,1)}"]),
            ("Sustitución hacia atrás: T₂", "El nuevo coeficiente de $T_2$ es el pivote; con él se despeja $T_2$ directamente.",
             [rf"T_2 = \dfrac{{{fx(w,1)}}}{{{piv:.6f}}} = {t2m:.4f}\ \mathrm{{N}}"]),
            ("Sustitución hacia atrás: T₁", "Con $T_2$ conocido, la primera fila da $T_1$.",
             [rf"T_1 = \dfrac{{\cos\beta}}{{\cos\alpha}}\,T_2 = \dfrac{{{np.cos(b):.6f}}}{{{np.cos(a):.6f}}}({t2m:.4f}) = {t1m:.4f}\ \mathrm{{N}}"]),
            ("Resultado numérico y error", "El resultado de `solve` coincide con el analítico hasta el redondeo de doble precisión.",
             [rf"T_1^{{num}} = {n1:.4f}\ \mathrm{{N}},\quad T_2^{{num}} = {n2:.4f}\ \mathrm{{N}}",
              rf"|T_1^{{num}}-T_1| = {sci(abs(n1-t1),2)}\ \mathrm{{N}},\qquad |T_2^{{num}}-T_2| = {sci(abs(n2-t2),2)}\ \mathrm{{N}}"]),
            ("Fuerza neta residual", "Se evalúa la suma de fuerzas con las tensiones numéricas: el residuo debe ser del orden del redondeo.",
             [rf"\sum \vec F = ({sci(float(res[0]),2)},\ {sci(float(res[1]),2)})\ \mathrm{{N}},\qquad \|\sum\vec F\| = {sci(float(np.linalg.norm(res)),2)}\ \mathrm{{N}}"]),
            ("Barrido del ángulo α", r"Repetir la solución del sistema para muchos $\alpha$ (con $\beta=45^\circ$ fijo) produce las curvas de tensión de la animación. Tres puntos de ejemplo:",
             [rf"\alpha={k:.0f}^\circ:\ \ T_1={v[0]:.2f}\ \mathrm{{N}},\ T_2={v[1]:.2f}\ \mathrm{{N}}" for k, v in sweep.items()]),
        ],
    )


def _pulley() -> dict:
    m1, m2, th, g = c5.M1, c5.M2, c5.THETA_DEG, c5.G
    s = np.sin(np.deg2rad(th))
    a, t = c5.pulley_analytic()
    an, tn = c5.pulley_numeric()
    thb = c5.balance_angle_analytic()
    thn = c5.balance_angle_numeric()
    # RK4 sobre (s, v) con aceleracion constante an
    rhs = lambda _t, y: np.array([y[1], an])  # noqa: E731
    n, t1 = 220, 2.2
    h = t1 / n
    k1, k2, k3, k4, y1 = _ks(rhs, 0.0, np.array([0.0, 0.0]), h)
    _, ys = c5.pulley_trajectory(t1=t1, n=n)
    # biseccion
    lo, hi = 0.0, 89.9
    rows = []
    for i in range(1, 6):
        mid = 0.5 * (lo + hi)
        flo, fmid = c5.pulley_numeric(th_deg := lo)[0], c5.pulley_numeric(m1, m2, mid)[0]
        flo = c5.pulley_numeric(m1, m2, lo)[0]
        if flo * fmid <= 0:
            hi = mid
        else:
            lo = mid
        rows.append(rf"{i} & {mid:.4f} & {fmid:.5f} & [{lo:.4f},\ {hi:.4f}]")
    tab = r"\begin{array}{cccc}k & \theta_{mid}\,(^\circ) & a(\theta_{mid})\,(\mathrm{m/s^2}) & \text{intervalo nuevo}\\\hline " + r"\\ ".join(rows) + r"\end{array}"
    return dict(
        explanation=[
            "Cuando varios cuerpos están unidos por una cuerda, cada uno obedece su propia segunda ley, y la cuerda los acopla mediante dos hechos cinemáticos: si es inextensible los dos tienen la misma aceleración en magnitud, y si la polea es ideal (sin masa ni rozamiento) la tensión es la misma en todo el tramo. Con esas dos ideas el problema se reduce a un sistema de dos ecuaciones con incógnitas $a$ y $T$.",
            "El truco es elegir un sentido positivo común: hacia donde se movería el sistema, es decir, subiendo el bloque del plano y bajando el colgante. Sobre el plano inclinado solo la componente del peso paralela al plano, $m_1 g\\sin\\theta$, compite con la tensión; la componente perpendicular queda equilibrada por la normal y no afecta el movimiento por no haber fricción.",
            "El resultado $a=\\dfrac{m_2-m_1\\sin\\theta}{m_1+m_2}g$ tiene una lectura clara: el numerador es la fuerza neta que impulsa al sistema (peso colgante menos la componente del peso en el plano) y el denominador es la masa total que hay que acelerar. Si el numerador se anula hay equilibrio, lo que ocurre para $\\sin\\theta=m_2/m_1$.",
            "Con $m_1=4$ kg, $m_2=3$ kg y $\\theta=30^\\circ$ la aceleración es una fracción pequeña de $g$ porque casi se compensan las fuerzas impulsoras: $m_2g$ y $m_1g\\sin\\theta$ difieren poco. La tensión queda entre ambos pesos, como debe ser para que el bloque colgante acelere hacia abajo.",
        ],
        analytic_steps=[
            ("Elegir sentidos y dibujar los diagramas", "Se toma positiva la aceleración que sube al bloque 1 por el plano y baja al bloque 2. Sobre 1 actúan la tensión (hacia arriba del plano), $m_1g\\sin\\theta$ (hacia abajo del plano) y la normal; sobre 2, la tensión hacia arriba y su peso.",
             [rf"m_1 = {fx(m1,1)}\ \mathrm{{kg}},\quad m_2 = {fx(m2,1)}\ \mathrm{{kg}},\quad \theta = {fx(th,0)}^\circ,\quad g = {g}\ \mathrm{{m/s^2}}"]),
            ("Segunda ley en cada cuerpo", "Una ecuación para cada bloque, con la misma $a$ y la misma $T$ por la cuerda inextensible y la polea ideal.",
             [r"m_1:\quad T - m_1 g\sin\theta = m_1 a", r"m_2:\quad m_2 g - T = m_2 a"]),
            ("Sumar para eliminar T", "Al sumar miembro a miembro la tensión se cancela, y queda la fuerza impulsora igual a la masa total por $a$.",
             [r"m_2 g - m_1 g\sin\theta = (m_1+m_2)\,a\;\Rightarrow\; a = \dfrac{m_2 - m_1\sin\theta}{m_1+m_2}\,g"]),
            ("Sustituir cifras en a", rf"Se usa $\sin 30^\circ={s:.1f}$.",
             [rf"a = \dfrac{{{fx(m2,1)} - ({fx(m1,1)})({s:.1f})}}{{{fx(m1,1)}+{fx(m2,1)}}}({g}) = \dfrac{{{m2 - m1*s:.2f}}}{{{m1+m2:.1f}}}({g}) = {a:.4f}\ \mathrm{{m/s^2}}"]),
            ("Despejar la tensión", "Se vuelve a la ecuación del bloque colgante y se despeja $T$ con la $a$ ya conocida.",
             [r"T = m_2\,(g - a)", rf"T = ({fx(m2,1)})({g} - {a:.4f}) = {m2*(g-a):.3f}\ \mathrm{{N}}"]),
            ("Comprobar con el otro cuerpo", "La misma $T$ debe salir del bloque sobre el plano: $T=m_1(a+g\\sin\\theta)$.",
             [rf"T = ({fx(m1,1)})({a:.4f} + {g}\cdot{s:.1f}) = {t:.3f}\ \mathrm{{N}}"]),
            ("Ángulo de equilibrio", "El sistema no acelera cuando $a=0$, es decir cuando el peso colgante iguala a la componente del peso en el plano.",
             [r"m_2 = m_1\sin\theta_{eq}\;\Rightarrow\;\theta_{eq} = \arcsin\dfrac{m_2}{m_1}", rf"\theta_{{eq}} = \arcsin\dfrac{{{fx(m2,1)}}}{{{fx(m1,1)}}} = \arcsin {m2/m1:.2f} = {thb:.4f}^\circ"]),
        ],
        numeric_steps=[
            ("Sistema lineal en (a, T)", "Se escriben las dos leyes de Newton como ecuaciones lineales en las incógnitas $(a,T)$, que es la forma que usa `numpy.linalg.solve`.",
             [r"\begin{pmatrix} m_1 & -1\\ m_2 & 1\end{pmatrix}\begin{pmatrix}a\\T\end{pmatrix}=\begin{pmatrix}-m_1 g\sin\theta\\ m_2 g\end{pmatrix}",
              rf"\begin{{pmatrix}} {fx(m1,1)} & -1\\ {fx(m2,1)} & 1\end{{pmatrix}}\begin{{pmatrix}}a\\T\end{{pmatrix}}=\begin{{pmatrix}}{-m1*g*s:.4f}\\ {m2*g:.4f}\end{{pmatrix}}"]),
            ("Eliminación gaussiana", "Se suma la fila 1 a la fila 2 (equivale al multiplicador $m_{21}=m_2/m_1$ de la eliminación general) para anular $T$ de la segunda fila, y se retrocede.",
             [rf"(m_1+m_2)\,a = {m2*g:.4f} - {m1*g*s:.4f} = {m2*g-m1*g*s:.4f}\;\Rightarrow\; a = {an:.5f}\ \mathrm{{m/s^2}}",
              rf"T = m_2 g - m_2 a = {m2*g:.4f} - ({fx(m2,1)})({an:.5f}) = {tn:.4f}\ \mathrm{{N}}"]),
            ("Integrar el movimiento con RK4", rf"Con $a$ constante se integra $(s,v)$ con $\dot s=v$, $\dot v=a$ durante {t1} s en {n} pasos, así que $h={t1}/{n}={h:.3f}$ s. Cada paso de RK4 calcula cuatro pendientes.",
             [rf"k_1 = f(0,y_0) = {_v(k1)}", rf"k_2 = f(\tfrac h2,\ y_0+\tfrac h2k_1) = {_v(k2)}",
              rf"k_3 = {_v(k3)},\qquad k_4 = {_v(k4)}"]),
            ("Primer paso completo", "Se combinan las pendientes con los pesos $1,2,2,1$. Para una aceleración constante el resultado coincide con la cinemática exacta $s=\\tfrac12ah^2$, $v=ah$.",
             [rf"y_1 = y_0 + \dfrac h6\,(k_1+2k_2+2k_3+k_4) = ({sci(float(y1[0]),4)}\ \mathrm{{m}},\ {sci(float(y1[1]),4)}\ \mathrm{{m/s}})"]),
            ("Resultado de la trayectoria", rf"Tras los {n} pasos se compara el desplazamiento con $\tfrac12 a t^2$ con $a$ analítica.",
             [rf"s^{{num}}(2.2\ \mathrm{{s}}) = {float(ys[-1,0]):.4f}\ \mathrm{{m}},\qquad \tfrac12({a:.4f})(2.2)^2 = {0.5*a*2.2**2:.4f}\ \mathrm{{m}}"]),
            ("Ángulo de equilibrio por bisección", "Se busca el cero de $a(\\theta)$ en $[0^\\circ, 89.9^\\circ]$. En cada iteración se evalúa $a$ en el punto medio y se conserva la mitad donde cambia el signo. Primeras 5 iteraciones:",
             [tab]),
            ("Convergencia y error", "Cada bisección reduce el intervalo a la mitad; con 60 iteraciones el ancho es $89.9/2^{60}$, muy por debajo del redondeo.",
             [rf"\theta_{{eq}}^{{num}} = {thn:.4f}^\circ,\qquad |\theta^{{num}}_{{eq}}-\theta_{{eq}}| = {sci(abs(thn-thb),2)}^\circ",
              rf"|a^{{num}}-a| = {sci(abs(an-a),2)}\ \mathrm{{m/s^2}},\qquad |T^{{num}}-T| = {sci(abs(tn-t),2)}\ \mathrm{{N}}"]),
        ],
    )


# ------------------------------------------------------------------ capitulo 6
def _friction() -> dict:
    th, mus, muk, g = c6.THETA_DEG, c6.MU_S, c6.MU_K, c6.G
    r = np.deg2rad(th)
    a = c6.incline_acceleration_analytic(th)
    thc = c6.critical_angle_analytic()
    _, ss, vs = c6.simulate_incline(th, 2.0)
    thn = c6.critical_angle_numeric()
    dt = 1e-3
    rhs = lambda _t, y: np.array([y[1], g * (np.sin(r) - muk * np.cos(r))])  # noqa: E731
    k1, k2, k3, k4, y1 = _ks(rhs, 0.0, np.array([0.0, 0.0]), dt)
    lo, hi = 1.0, 80.0
    rows = []
    for i in range(1, 7):
        mid = 0.5 * (lo + hi)
        _, s, _ = c6.simulate_incline(mid, 1.0, dt=0.01)
        moves = s[-1] > 1e-9
        if moves:
            hi = mid
        else:
            lo = mid
        rows.append(rf"{i} & {mid:.4f} & {s[-1]:.4f} & {'si' if moves else 'no'} & [{lo:.4f},\ {hi:.4f}]")
    tab = r"\begin{array}{ccccc}k & \theta_{mid}\,(^\circ) & s(1\,\mathrm{s})\,(\mathrm{m}) & \text{desliza} & \text{intervalo}\\\hline " + r"\\ ".join(rows) + r"\end{array}"
    return dict(
        explanation=[
            "La fricción seca es la fuerza de contacto paralela a la superficie que se opone al deslizamiento relativo. Mientras el bloque no se mueve, la fricción estática se ajusta exactamente para equilibrar lo que lo empuja, pero tiene un tope $f_s\\le\\mu_sN$. Una vez que desliza, la fricción pasa a ser cinética con valor casi constante $f_k=\\mu_kN$, que por lo general es menor que el máximo estático.",
            "En un plano inclinado la normal vale $N=mg\\cos\\theta$ y la componente del peso a lo largo del plano es $mg\\sin\\theta$. El bloque está a punto de deslizar cuando la fricción estática necesaria iguala su tope, $mg\\sin\\theta=\\mu_s mg\\cos\\theta$, lo que da $\\tan\\theta_c=\\mu_s$: el ángulo crítico no depende de la masa.",
            "Si el ángulo supera $\\theta_c$ el bloque desliza y su aceleración es $a=g(\\sin\\theta-\\mu_k\\cos\\theta)$, también independiente de la masa. Como $\\mu_k<\\mu_s$, la aceleración inicial es positiva apenas se sobrepasa el ángulo crítico: el bloque arranca con un pequeño salto brusco, que es la sensación familiar del deslizamiento ``pegajoso''.",
            "Con $\\mu_s=0.45$ el ángulo crítico es cercano a 24°, de modo que a 35° el bloque desliza sin duda. La fricción es un modelo empírico: los coeficientes dependen de los materiales y de la superficie, y la proporcionalidad a $N$ deja de valer a cargas o velocidades extremas.",
        ],
        analytic_steps=[
            ("Plantear fuerzas y ejes", "Se toma un eje a lo largo del plano (positivo hacia abajo) y otro perpendicular. Actúan el peso, la normal y la fricción, que se opone al movimiento posible.",
             [rf"\theta = {fx(th,0)}^\circ,\qquad \mu_s = {mus},\qquad \mu_k = {muk},\qquad g = {g}\ \mathrm{{m/s^2}}"]),
            ("Equilibrio perpendicular al plano", "No hay aceleración perpendicular, así que la normal compensa la componente del peso en esa dirección.",
             [r"N - mg\cos\theta = 0\;\Rightarrow\; N = mg\cos\theta"]),
            ("Decidir si desliza", "El bloque se mantendría en reposo si la fricción estática necesaria $mg\\sin\\theta$ no supera su tope $\\mu_s mg\\cos\\theta$. Dividiendo por $mg\\cos\\theta$ se compara $\\tan\\theta$ con $\\mu_s$.",
             [rf"\tan {fx(th,0)}^\circ = {np.tan(r):.4f}\ \ >\ \ \mu_s = {mus}\;\Rightarrow\;\text{{el bloque desliza}}"]),
            ("Segunda ley a lo largo del plano", "Una vez en movimiento actúa la fricción cinética $\\mu_kN$ hacia arriba del plano.",
             [r"mg\sin\theta - \mu_k\,mg\cos\theta = ma\;\Rightarrow\; a = g\,(\sin\theta - \mu_k\cos\theta)"]),
            ("Sustituir cifras en a", "Se evalúan seno y coseno de 35° y se multiplica.",
             [rf"a = ({g})\left({np.sin(r):.5f} - ({muk})({np.cos(r):.5f})\right) = ({g})({np.sin(r)-muk*np.cos(r):.5f}) = {a:.4f}\ \mathrm{{m/s^2}}"]),
            ("Desplazamiento a los 2 s", "Con aceleración constante desde el reposo, $s=\\tfrac12at^2$ y $v=at$.",
             [rf"s(2) = \tfrac12({a:.4f})(2)^2 = {0.5*a*4:.4f}\ \mathrm{{m}},\qquad v(2) = {a*2:.4f}\ \mathrm{{m/s}}"]),
            ("Ángulo crítico", "El umbral de deslizamiento es la igualdad en la condición de equilibrio.",
             [r"\tan\theta_c = \mu_s\;\Rightarrow\;\theta_c = \arctan\mu_s", rf"\theta_c = \arctan({mus}) = {thc:.3f}^\circ"]),
        ],
        numeric_steps=[
            ("Máquina de estados", "El código integra con paso $\\Delta t=10^{-3}$ s y dos estados: reposo y deslizamiento. En reposo se comprueba en cada paso si $g\\sin\\theta>\\mu_s g\\cos\\theta$; si no, el bloque sigue quieto y $(s,v)$ no cambian.",
             [rf"g\sin\theta = {g*np.sin(r):.4f},\qquad \mu_s g\cos\theta = {mus*g*np.cos(r):.4f}\ \Rightarrow\ {g*np.sin(r):.4f} > {mus*g*np.cos(r):.4f}"]),
            ("Cambio a estado deslizante", "La condición se cumple en el primer paso, así que el estado pasa a ``moviéndose'' y la ecuación es $\\dot s=v$, $\\dot v=g(\\sin\\theta-\\mu_k\\cos\\theta)$ con $y=(s,v)$.",
             [rf"f(y) = \left(v,\ {g*(np.sin(r)-muk*np.cos(r)):.5f}\right)"]),
            ("Primer paso RK4 (cuatro pendientes)", rf"Con $y_0=(0,0)$ y $h={dt}$ s. Como $f$ no depende de $s$ y la aceleración es constante, las pendientes solo difieren en la componente $s$.",
             [rf"k_1 = {_v(k1)},\quad k_2 = {_v(k2)}", rf"k_3 = {_v(k3)},\quad k_4 = {_v(k4)}"]),
            ("Actualizar", "Se promedian con pesos $1,2,2,1$.",
             [rf"y_1 = y_0 + \dfrac h6(k_1+2k_2+2k_3+k_4) = ({sci(float(y1[0]),4)}\ \mathrm{{m}},\ {sci(float(y1[1]),4)}\ \mathrm{{m/s}})",
              r"\text{y se repite 2000 veces hasta } t=2\ \mathrm{s}."]),
            ("Resultado de la simulación", "La aceleración numérica se toma como $v(2)/2$ y el desplazamiento es la última muestra.",
             [rf"s^{{num}}(2) = {float(ss[-1]):.4f}\ \mathrm{{m}},\quad v^{{num}}(2) = {float(vs[-1]):.4f}\ \mathrm{{m/s}},\quad a^{{num}} = {float(vs[-1]/2):.4f}\ \mathrm{{m/s^2}}",
              rf"|a^{{num}}-a| = {sci(abs(float(vs[-1]/2)-a),2)}\ \mathrm{{m/s^2}}"]),
            ("Ángulo crítico por bisección", "Entre $1^\\circ$ y $80^\\circ$ se simula 1 s (con $\\Delta t=0.01$ s); si el bloque se movió más de $10^{-9}$ m, el umbral está por debajo del punto medio. Primeras iteraciones:",
             [tab]),
            ("Convergencia", "Tras 40 bisecciones el intervalo mide $79/2^{40}$ grados. El valor coincide con $\\arctan\\mu_s$ porque el estado deslizante se activa justo cuando se cumple la condición estática.",
             [rf"\theta_c^{{num}} = {thn:.4f}^\circ,\qquad \theta_c = {thc:.4f}^\circ,\qquad |\Delta| = {sci(abs(thn-thc),2)}^\circ"]),
        ],
    )


def _drag() -> dict:
    m, b, g = c6.MASS, c6.DRAG_B, c6.G
    vt, ta = c6.terminal_speed(), c6.tau()
    t95 = c6.time_to_fraction_analytic()
    t95n = c6.time_to_fraction_numeric()
    f = lambda _t, y: np.array([y[1], g - (b / m) * y[1]])  # noqa: E731
    n, te = 8, 1.0
    h = te / n
    y0 = np.array([0.0, 0.0])
    k1, k2, k3, k4, y1 = _ks(f, 0.0, y0, h)
    ye1 = euler_step(f, 0.0, y0, h)
    vex = float(c6.drag_velocity_analytic(h))
    conv = c6.drag_convergence()
    # tiempo al 95 %
    tt, yy = c6.drag_numeric(4000, "rk4", 6.0)
    target = 0.95 * vt
    i = int(np.argmax(yy[:, 1] >= target))
    t0_, t1_, v0_, v1_ = tt[i-1], tt[i], yy[i-1, 1], yy[i, 1]
    _, y8 = c6.drag_numeric(4000, "rk4", 8.0)
    crow = r"\begin{array}{rcc}n & \varepsilon_{Euler}\,(\mathrm{m/s}) & \varepsilon_{RK4}\,(\mathrm{m/s})\\\hline " + r"\\ ".join(
        rf"{c['n']} & {sci(float(c['err_euler']),3)} & {sci(float(c['err_rk4']),3)}" for c in conv) + r"\end{array}"
    return dict(
        explanation=[
            "Un cuerpo que cae en un fluido sufre una fuerza de resistencia que se opone a la velocidad. A velocidades bajas (flujo laminar, esferas pequeñas o medios viscosos) un buen modelo es la resistencia lineal $F=-bv$: crece con la rapidez y apunta contra ella.",
            "Al principio la resistencia es nula y el cuerpo acelera con $g$. Conforme gana rapidez la resistencia aumenta, la fuerza neta $mg-bv$ disminuye y la aceleración se reduce, hasta que ambas fuerzas se igualan. Desde ahí la velocidad ya no cambia: es la rapidez terminal $v_t=mg/b$.",
            "La ecuación $m\\,dv/dt=mg-bv$ es lineal de primer orden y su solución es una aproximación exponencial a $v_t$ con constante de tiempo $\\tau=m/b$. Después de un tiempo $\\tau$ la rapidez ha recorrido el 63 % del camino hacia $v_t$, y tras $3\\tau$ el 95 %; por eso $\\tau$ mide ``qué tan rápido'' el medio frena al cuerpo.",
            "Con $m=0.5$ kg y $b=0.8$ kg/s la constante de tiempo es de poco más de medio segundo y la rapidez terminal es de unos 6 m/s, de modo que en pocos segundos la caída ya es prácticamente uniforme. A velocidades altas la resistencia deja de ser lineal y pasa a depender de $v^2$.",
        ],
        analytic_steps=[
            ("Plantear la segunda ley", "Con el eje $y$ hacia abajo actúan el peso, $+mg$, y la resistencia, $-bv$. La ecuación de movimiento es lineal en $v$.",
             [r"m\dfrac{dv}{dt} = mg - bv", rf"m={m}\ \mathrm{{kg}},\quad b={b}\ \mathrm{{kg/s}},\quad g={g}\ \mathrm{{m/s^2}}"]),
            ("Rapidez terminal", "Se impone aceleración nula: el peso iguala a la resistencia.",
             [r"mg - bv_t = 0\;\Rightarrow\; v_t = \dfrac{mg}{b}", rf"v_t = \dfrac{{({m})({g})}}{{{b}}} = {vt:.4f}\ \mathrm{{m/s}}"]),
            ("Constante de tiempo", "Se reescribe la ecuación como $dv/dt=(v_t-v)/\\tau$ con $\\tau=m/b$.",
             [rf"\tau = \dfrac{{m}}{{b}} = \dfrac{{{m}}}{{{b}}} = {ta:.4f}\ \mathrm{{s}}"]),
            ("Separar variables e integrar", "Se agrupa $v$ a un lado y $t$ al otro y se integra desde $v(0)=0$.",
             [r"\int_0^v\dfrac{dv'}{v_t-v'} = \int_0^t\dfrac{dt'}{\tau}\;\Rightarrow\;-\ln\dfrac{v_t-v}{v_t} = \dfrac t\tau",
              r"v(t) = v_t\left(1-e^{-t/\tau}\right)"]),
            ("Evaluar v en un instante", "Como ejemplo, a $t=\\tau$ la velocidad es el 63.2 % de la terminal, y a $t=1$ s:",
             [rf"v(\tau) = ({vt:.4f})(1-e^{{-1}}) = {vt*(1-np.exp(-1)):.4f}\ \mathrm{{m/s}},\qquad v(1) = ({vt:.4f})(1-e^{{-1/{ta:.4f}}}) = {float(c6.drag_velocity_analytic(1.0)):.4f}\ \mathrm{{m/s}}"]),
            ("Tiempo al 95 %", "Se iguala $v=0.95\\,v_t$ y se despeja $t$; $v_t$ se cancela.",
             [r"1 - e^{-t/\tau} = 0.95\;\Rightarrow\; t_{95} = -\tau\ln 0.05 = \tau\ln 20",
              rf"t_{{95}} = ({ta:.4f})({np.log(20):.5f}) = {t95:.4f}\ \mathrm{{s}}"]),
        ],
        numeric_steps=[
            ("Ecuaciones de estado", "Se integra el vector $y=(y,v)$ con $\\dot y=v$, $\\dot v=g-(b/m)v$. Para la tabla de convergencia se toma $t\\in[0,1]$ s con $n=8$ pasos, es decir $h=0.125$ s.",
             [rf"f(y,v) = \left(v,\ {g} - {b/m:.2f}\,v\right),\qquad y_0 = (0,0),\qquad h = {h}\ \mathrm{{s}}"]),
            ("Primer paso de Euler", "Euler usa solo la pendiente inicial: $y_1=y_0+hf(y_0)$.",
             [rf"v_1^{{E}} = 0 + ({h})({g}) = {ye1[1]:.5f}\ \mathrm{{m/s}},\qquad \text{{exacto: }}v(h) = {vex:.5f}\ \mathrm{{m/s}}"]),
            ("Primer paso de RK4", "RK4 calcula cuatro pendientes en el paso (solo la segunda componente, la de la aceleración, cambia):",
             [rf"k_1 = {_v(k1)}", rf"k_2 = f(y_0+\tfrac h2k_1) = {_v(k2)}", rf"k_3 = {_v(k3)},\qquad k_4 = {_v(k4)}"]),
            ("Combinar las pendientes", "Se pondera con $1,2,2,1$ y se multiplica por $h/6$. La velocidad de RK4 está mucho más cerca del valor exacto.",
             [rf"v_1^{{RK4}} = \dfrac h6\,(k_{{1v}}+2k_{{2v}}+2k_{{3v}}+k_{{4v}}) = {y1[1]:.7f}\ \mathrm{{m/s}}",
              rf"|v_1^{{E}}-v(h)| = {sci(abs(ye1[1]-vex),2)},\qquad |v_1^{{RK4}}-v(h)| = {sci(abs(y1[1]-vex),2)}\ \mathrm{{m/s}}"]),
            ("Repetir y comparar el orden", "Repitiendo hasta $t=1$ s con $n=8,16,\\dots,128$ el error de Euler se divide por 2 al duplicar $n$ (orden 1) y el de RK4 por 16 (orden 4).",
             [crow]),
            ("Rapidez terminal numérica", "Con $n=4000$ pasos hasta 8 s ($h=0.002$ s), $v$ ya está a $e^{-10}\\approx5\\times10^{-5}$ de $v_t$ en términos relativos.",
             [rf"v^{{num}}(8) = {float(y8[-1,1]):.5f}\ \mathrm{{m/s}},\qquad v_t = {vt:.5f}\ \mathrm{{m/s}},\qquad |\Delta|={sci(abs(float(y8[-1,1])-vt),2)}\ \mathrm{{m/s}}"]),
            ("Tiempo al 95 % por interpolación", rf"Con $n=4000$ en $[0,6]$ s ($h={6/4000}$ s) se busca el primer paso $i$ con $v_i\ge0.95v_t={target:.5f}$ y se interpola linealmente entre $i-1$ e $i$.",
             [rf"t_{{i-1}}={t0_:.5f},\ v_{{i-1}}={v0_:.6f};\qquad t_i={t1_:.5f},\ v_i={v1_:.6f}",
              rf"t_{{95}} = t_{{i-1}} + (v^\ast-v_{{i-1}})\dfrac{{t_i-t_{{i-1}}}}{{v_i-v_{{i-1}}}} = {t95n:.5f}\ \mathrm{{s}}",
              rf"|t_{{95}}^{{num}}-t_{{95}}| = {sci(abs(t95n-t95),2)}\ \mathrm{{s}}"]),
        ],
    )


# ------------------------------------------------------------------ capitulo 7
def _spring_work() -> dict:
    k, x1, x2 = c7.K, c7.X1, c7.X2
    w = c7.spring_work_analytic()
    n = 4
    x = np.linspace(x1, x2, n + 1)
    f = c7.spring_force(x)
    h = (x2 - x1) / n
    wl, wt, ws = (c7.spring_work_numeric(n, mth) for mth in ("riemann", "trapezoid", "simpson"))
    conv = c7.work_convergence()
    w32 = {mth: c7.spring_work_numeric(32, mth) for mth in ("riemann", "trapezoid", "simpson")}
    xs = r",\ ".join(f"{v:.2f}" for v in x)
    fs = r",\ ".join(f"{v:.0f}" for v in f)
    tab = r"\begin{array}{rccc}n & \varepsilon_{Riemann}\,(\mathrm{J}) & \varepsilon_{trap}\,(\mathrm{J}) & \varepsilon_{Simpson}\,(\mathrm{J})\\\hline " + r"\\ ".join(
        rf"{c['n']} & {sci(c['riemann'],3)} & {sci(c['trapezoid'],3)} & {sci(c['simpson'],3)}" for c in conv) + r"\end{array}"
    return dict(
        explanation=[
            "El trabajo de una fuerza que cambia con la posición no es simplemente fuerza por desplazamiento, porque la fuerza es distinta en cada punto del camino. Se divide el recorrido en tramos tan pequeños que en cada uno la fuerza es casi constante y se suman los trabajos parciales $F\\,dx$: eso es la integral $W=\\int F(x)\\,dx$, el área bajo la curva fuerza-posición.",
            "Un resorte ideal ejerce una fuerza restauradora $F=-kx$ (ley de Hooke). Para estirarlo lentamente hay que aplicar la fuerza opuesta, $+kx$, y por eso el trabajo del agente externo es positivo: la fuerza crece linealmente y el área es un trapecio entre $x_1$ y $x_2$, cuyo valor $\\tfrac12k(x_2^2-x_1^2)$ es la energía almacenada en el resorte.",
            "El trabajo no depende de qué tan rápido se estira, solo de las posiciones inicial y final, y es el cambio de la energía potencial elástica. Estirar de $0.10$ a $0.30$ m cuesta más que de $0$ a $0.20$ m aunque ambos tramos midan lo mismo, porque la fuerza es mayor mientras más estirado esté el resorte.",
            "Cuando no existe una fórmula cerrada para la integral se aproxima numéricamente. Las tres reglas del ejemplo (Riemann por la izquierda, trapecio y Simpson) reemplazan la curva por rectángulos, rectas o parábolas por tramos, y su error disminuye como $h$, $h^2$ y $h^4$ respectivamente.",
        ],
        analytic_steps=[
            ("Plantear el trabajo como integral", "La fuerza del agente externo que equilibra al resorte en cada posición es $F=kx$, y el trabajo es el área bajo $F(x)$.",
             [r"W = \int_{x_1}^{x_2}F(x)\,dx = \int_{x_1}^{x_2}kx\,dx", rf"k = {fx(k,0)}\ \mathrm{{N/m}},\quad x_1 = {x1}\ \mathrm{{m}},\quad x_2 = {x2}\ \mathrm{{m}}"]),
            ("Integrar", "La primitiva de $x$ es $x^2/2$; se evalúa entre los límites.",
             [r"W = k\left[\dfrac{x^2}{2}\right]_{x_1}^{x_2} = \dfrac12k\,(x_2^2 - x_1^2)"]),
            ("Sustituir las posiciones al cuadrado", "Se calculan primero los cuadrados con sus unidades.",
             [rf"x_2^2 = ({x2})^2 = {x2**2:.4f}\ \mathrm{{m^2}},\qquad x_1^2 = ({x1})^2 = {x1**2:.4f}\ \mathrm{{m^2}}"]),
            ("Resultado en joules", "Se multiplica por $k/2$.",
             [rf"W = \dfrac12({fx(k,0)}\ \mathrm{{N/m}})({x2**2:.4f} - {x1**2:.4f})\ \mathrm{{m^2}} = ({k/2:.0f})({x2**2-x1**2:.4f}) = {w:.2f}\ \mathrm{{J}}"]),
            ("Interpretación geométrica", "El área bajo la recta $F=kx$ es un trapecio de bases $F(x_1)$ y $F(x_2)$ y altura $x_2-x_1$; da el mismo valor.",
             [rf"W = \dfrac{{F(x_1)+F(x_2)}}{{2}}(x_2-x_1) = \dfrac{{{k*x1:.1f}+{k*x2:.1f}}}{{2}}({x2-x1:.2f}) = {(k*x1+k*x2)/2*(x2-x1):.2f}\ \mathrm{{J}}"]),
        ],
        numeric_steps=[
            ("Dividir el intervalo", rf"Para mostrar la mecánica se toma $n={n}$ subintervalos; el ancho es $h=(x_2-x_1)/n$ y los nodos son $x_i=x_1+ih$. El código usa los mismos nodos con `linspace`.",
             [rf"h = \dfrac{{{x2}-{x1}}}{{{n}}} = {h:.3f}\ \mathrm{{m}},\qquad x_i = {xs}\ \mathrm{{m}}"]),
            ("Evaluar la fuerza en los nodos", "Se calcula $F_i=kx_i$ en cada nodo.",
             [rf"F_i = {fs}\ \mathrm{{N}}"]),
            ("Regla de Riemann por la izquierda", "Cada tramo se aproxima por un rectángulo de altura igual a $F$ en su extremo izquierdo; como $F$ crece, subestima.",
             [rf"W_R = h\,(F_0+F_1+F_2+F_3) = ({h:.3f})({f[:-1].sum():.0f}) = {wl:.4f}\ \mathrm{{J}},\qquad \varepsilon={abs(wl-w):.4f}\ \mathrm{{J}}"]),
            ("Regla del trapecio", "Cada tramo se reemplaza por un trapecio, es decir, la curva por segmentos rectos.",
             [rf"W_T = h\left[\tfrac12F_0 + F_1+F_2+F_3 + \tfrac12F_4\right] = ({h:.3f})({0.5*f[0]+f[1:-1].sum()+0.5*f[-1]:.0f}) = {wt:.4f}\ \mathrm{{J}}"]),
            ("Regla de Simpson", "Se ajusta una parábola a cada par de tramos con pesos $1,4,2,4,1$ (exige $n$ par). Como $F$ es lineal, la parábola coincide con la recta y la regla es exacta.",
             [rf"W_S = \dfrac h3\left[F_0 + 4(F_1+F_3) + 2F_2 + F_4\right] = \dfrac{{{h:.3f}}}{{3}}\left[{f[0]:.0f}+4({f[1]+f[3]:.0f})+2({f[2]:.0f})+{f[4]:.0f}\right] = {ws:.4f}\ \mathrm{{J}}"]),
            ("Refinar la malla", "Se repite con $n=2,4,8,16,32$ y se mide el error absoluto $\\varepsilon=|W_{num}-W|$. Riemann baja a la mitad al duplicar $n$ (orden 1); trapecio y Simpson son exactos para una función lineal, y quedan solo en el redondeo.",
             [tab]),
            ("Resultado con n = 32", "Los valores de las filas de verificación.",
             [rf"W_R = {w32['riemann']:.4f},\quad W_T = {w32['trapezoid']:.4f},\quad W_S = {w32['simpson']:.4f}\ \mathrm{{J}}\qquad (W = {w:.4f}\ \mathrm{{J}})"]),
        ],
    )


def _wet() -> dict:
    m, f0, L = c7.MASS, c7.F0, c7.LENGTH
    va = c7.final_speed_analytic()
    vn = c7.final_speed_numeric()
    w_an = f0 * L / 2
    wnum = c7.work_integral_numeric()
    dt = 1e-3
    rhs = lambda _t, y: np.array([y[1], c7.force_profile(y[0], f0, L) / m])  # noqa: E731
    y0 = np.array([0.0, 0.0])
    k1, k2, k3, k4, y1 = _ks(rhs, 0.0, y0, dt)
    t, x, v = c7.newton_numeric()
    nst = len(t) - 1
    x4 = np.linspace(0, L, 5)
    F4 = c7.force_profile(x4)
    s4 = c7.simpson(F4, x4)
    pmax, xpmax = c7.max_power()
    return dict(
        explanation=[
            "El teorema trabajo-energía dice que el trabajo neto de todas las fuerzas sobre una partícula es igual al cambio de su energía cinética, $W_{neto}=\\Delta K$. Se deduce de la segunda ley: al integrar $F\\,dx=m v\\,dv$ a lo largo del camino, el resultado depende solo de las rapideces inicial y final.",
            "Su ventaja es que permite hallar la rapidez sin conocer el tiempo ni la aceleración instante a instante. Aquí la fuerza varía con la posición, $F(x)=30(1-x/2)$ N, de modo que Newton directamente exigiría resolver una ecuación diferencial, mientras que el teorema solo pide el área bajo la curva.",
            "La fuerza decrece linealmente y se anula en $x=2$ m, por lo que el área es un triángulo. Mientras la fuerza sea positiva el bloque sigue ganando rapidez, aunque cada vez más despacio; la potencia $P=Fv$ es cero al inicio (porque $v=0$) y al final (porque $F=0$), y alcanza un máximo en algún punto intermedio.",
            "El teorema vale para el trabajo de la fuerza neta, aquí la única fuerza horizontal porque no hay fricción y la normal y el peso no hacen trabajo. Como comprobación de orden de magnitud: 30 J de energía cinética para 2 kg dan unos 5 m/s.",
        ],
        analytic_steps=[
            ("Datos y planteamiento", "El bloque parte del reposo, así que $K_i=0$ y todo el trabajo se convierte en energía cinética final.",
             [rf"m = {m}\ \mathrm{{kg}},\quad F(x) = {fx(f0,0)}\left(1-\dfrac{{x}}{{{L:.0f}}}\right)\mathrm{{N}},\quad 0\le x\le {L:.0f}\ \mathrm{{m}}"]),
            ("Trabajo como integral", "Se integra la fuerza respecto de la posición entre $0$ y $2$ m.",
             [rf"W = \int_0^{{{L:.0f}}} {fx(f0,0)}\left(1-\dfrac{{x}}{{{L:.0f}}}\right)dx = {fx(f0,0)}\left[x - \dfrac{{x^2}}{{{2*L:.0f}}}\right]_0^{{{L:.0f}}}"]),
            ("Evaluar", "En $x=2$: $2-4/4=1$. El resultado coincide con el área del triángulo de base 2 m y altura 30 N.",
             [rf"W = ({fx(f0,0)})(1) = {w_an:.1f}\ \mathrm{{J}},\qquad \dfrac12(2)(30) = {w_an:.1f}\ \mathrm{{J}}"]),
            ("Aplicar trabajo-energía", "Con $v_0=0$ el trabajo es la energía cinética final.",
             [r"W = \tfrac12 m v^2 - 0\;\Rightarrow\; v = \sqrt{\dfrac{2W}{m}}"]),
            ("Sustituir cifras", "Se evalúa la raíz con $W=30$ J y $m=2$ kg.",
             [rf"v = \sqrt{{\dfrac{{2({w_an:.1f}\ \mathrm{{J}})}}{{{m}\ \mathrm{{kg}}}}}} = \sqrt{{{2*w_an/m:.1f}}} = {va:.4f}\ \mathrm{{m/s}}"]),
            ("Energía cinética a lo largo del camino", "Si se deja el límite superior $x$ como variable, $K(x)=30(x-x^2/4)$, con la que se puede hallar la potencia $P=Fv$ en cada punto.",
             [rf"K(x) = {fx(f0,0)}\left(x - \dfrac{{x^2}}{{{2*L:.0f}}}\right),\qquad K(1) = {float(c7.kinetic_energy_analytic(1.0)):.2f}\ \mathrm{{J}},\quad v(1) = {float(np.sqrt(2*c7.kinetic_energy_analytic(1.0)/m)):.4f}\ \mathrm{{m/s}}"]),
        ],
        numeric_steps=[
            ("Dos métodos independientes", "Por un lado se integra la ley de Newton $m\\ddot x=F(x)$ con RK4 (sin usar el teorema) y se lee la rapidez al llegar a $x=2$; por otro se calcula el trabajo con Simpson. Ambos deben reproducir 30 J y 5.477 m/s.",
             [rf"y=(x,v),\qquad \dot x=v,\qquad \dot v = \dfrac{{F(x)}}{{m}} = {f0/m:.0f}\left(1-\dfrac{{x}}{{{L:.0f}}}\right),\qquad h={dt}\ \mathrm{{s}}"]),
            ("Primer paso RK4", rf"Con $y_0=(0,0)$ la aceleración inicial es $F(0)/m={f0/m:.0f}$ m/s². Las cuatro pendientes son:",
             [rf"k_1 = {_v(k1)}", rf"k_2 = {_v(k2)}", rf"k_3 = {_v(k3)},\qquad k_4 = {_v(k4)}"]),
            ("Actualizar", "Se combinan con pesos $1,2,2,1$.",
             [rf"y_1 = y_0 + \dfrac h6(k_1+2k_2+2k_3+k_4) = ({sci(float(y1[0]),5)}\ \mathrm{{m}},\ {sci(float(y1[1]),5)}\ \mathrm{{m/s}})"]),
            ("Criterio de parada y refinamiento", "El bucle repite el paso mientras $x<L$. Como el último paso casi siempre sobrepasa $x=2$, se interpola linealmente entre los dos últimos puntos para ubicar exactamente $x=L$.",
             [rf"\text{{pasos}} \approx {nst},\qquad t_f = {t[-1]:.4f}\ \mathrm{{s}},\qquad x_f = {x[-1]:.5f}\ \mathrm{{m}}"]),
            ("Rapidez final numérica", "Se toma la velocidad en el punto interpolado.",
             [rf"v^{{num}} = {vn:.5f}\ \mathrm{{m/s}},\qquad v = {va:.5f}\ \mathrm{{m/s}},\qquad |\Delta| = {sci(abs(vn-va),2)}\ \mathrm{{m/s}}"]),
            ("Simpson reducido (n = 4)", rf"Para ver el mecanismo, con $h={L/4}$ m los nodos son $x=0,0.5,1,1.5,2$ y $F={F4[0]:.1f},{F4[1]:.1f},{F4[2]:.1f},{F4[3]:.1f},{F4[4]:.1f}$ N. Pesos $1,4,2,4,1$:",
             [rf"W_S = \dfrac{{{L/4}}}{{3}}\left[{F4[0]:.1f}+4({F4[1]:.1f}+{F4[3]:.1f})+2({F4[2]:.1f})+{F4[4]:.1f}\right] = {s4:.4f}\ \mathrm{{J}}"]),
            ("Simpson con 400 subintervalos y potencia", "El código usa $n=400$ ($h=0.005$ m). Como $F$ es lineal, Simpson es exacto salvo redondeo. Además se evalúa $P=Fv$ en cada paso de la trayectoria RK4.",
             [rf"W^{{num}} = {wnum:.6f}\ \mathrm{{J}},\qquad |W^{{num}}-30| = {sci(abs(wnum-w_an),2)}\ \mathrm{{J}}",
              rf"P_{{max}} = {pmax:.1f}\ \mathrm{{W}}\ \text{{en }}x = {xpmax:.2f}\ \mathrm{{m}}"]),
        ],
    )


# ------------------------------------------------------------------ capitulo 8
def _pend() -> dict:
    L, th0, g, m = c8.LENGTH, c8.THETA0_DEG, c8.G, c8.MASS
    r0 = np.deg2rad(th0)
    h0 = L * (1 - np.cos(r0))
    va = c8.bottom_speed_analytic()
    vn = c8.max_speed_numeric()
    n, te = 600, 6.0
    dt = te / n
    f = c8.pendulum_rhs()
    y0 = np.array([r0, 0.0])
    k1, k2, k3, k4, y1 = _ks(f, 0.0, y0, dt)
    ye = euler_step(f, 0.0, y0, dt)
    # verlet un paso
    acc = lambda x_: -(g / L) * np.sin(x_)  # noqa: E731
    a0 = acc(r0)
    thv = r0 + 0 * dt + 0.5 * a0 * dt**2
    omv = 0 + 0.5 * (a0 + acc(thv)) * dt
    K0, U0 = c8.energy(r0, 0.0)
    E0 = float(K0 + U0)
    def e_of(y):
        k, u = c8.energy(y[0], y[1])
        return float(k + u)
    drift = {mm: c8.energy_drift(mm, t_end=6.0, n=600) for mm in ("euler", "verlet", "rk4")}
    _, yr = c8.pendulum_run("rk4")
    return dict(
        explanation=[
            "Una fuerza es conservativa si el trabajo que realiza entre dos puntos no depende del camino. Para ellas se puede definir una energía potencial $U$ tal que $W=-\\Delta U$, y el teorema trabajo-energía se convierte en la conservación de la energía mecánica: $K+U$ no cambia durante el movimiento.",
            "En el péndulo actúan el peso, que es conservativo, y la tensión de la cuerda, que siempre es perpendicular a la trayectoria y por tanto no hace trabajo. Por eso la energía mecánica se conserva y la suma de la energía potencial gravitatoria $mgh$ y la cinética $\\tfrac12mv^2$ es constante en toda la oscilación.",
            "Medida desde el punto más bajo, la altura es $h=L(1-\\cos\\theta)$. Al soltar desde el reposo a $\\theta_0$ toda la energía es potencial, y al pasar por el punto más bajo toda es cinética: $v=\\sqrt{2gh}$. La masa se cancela, así que la rapidez es la misma que si el cuerpo cayera libremente desde la altura $h$.",
            "Numéricamente la conservación de la energía es un buen criterio para juzgar un integrador: un método que la deje derivar está acumulando error físico. Euler explícito la aumenta sistemáticamente, mientras que Verlet (simpléctico) la conserva en promedio con oscilaciones acotadas, y RK4 tiene una deriva minúscula.",
        ],
        analytic_steps=[
            ("Datos", "Se toma como referencia $U=0$ en el punto más bajo de la trayectoria.",
             [rf"L = {L}\ \mathrm{{m}},\quad m = {m}\ \mathrm{{kg}},\quad \theta_0 = {th0:.0f}^\circ,\quad g = {g}\ \mathrm{{m/s^2}}"]),
            ("Altura inicial", "La cuerda forma $\\theta_0$ con la vertical, de modo que la masa está $L\\cos\\theta_0$ por debajo del soporte, y por tanto a $L(1-\\cos\\theta_0)$ sobre el punto más bajo.",
             [rf"h = L(1-\cos\theta_0) = ({L})(1-\cos {th0:.0f}^\circ) = ({L})(1 - {np.cos(r0):.4f}) = {h0:.4f}\ \mathrm{{m}}"]),
            ("Energía en el punto de partida", "Parte del reposo, así que $K_i=0$.",
             [rf"E_i = U_i = mgh = ({m})({g})({h0:.4f}) = {m*g*h0:.4f}\ \mathrm{{J}}"]),
            ("Energía en el punto más bajo", "Allí $h=0$, luego $U_f=0$ y toda la energía es cinética.",
             [r"E_f = K_f = \tfrac12 m v^2"]),
            ("Igualar y despejar", "La masa se cancela.",
             [r"mgh = \tfrac12 mv^2\;\Rightarrow\; v = \sqrt{2gh} = \sqrt{2gL(1-\cos\theta_0)}"]),
            ("Sustituir cifras", "Se evalúa la raíz.",
             [rf"v = \sqrt{{2({g})({h0:.4f})}} = \sqrt{{{2*g*h0:.4f}}} = {va:.4f}\ \mathrm{{m/s}}"]),
            ("Tensión de la cuerda (comprobación)", "Aunque no hace trabajo, sí se puede calcular: en el punto más bajo $T-mg=mv^2/L$.",
             [rf"T = mg + \dfrac{{mv^2}}{{L}} = {m*g:.3f} + \dfrac{{({m})({va**2:.4f})}}{{{L}}} = {m*g+m*va**2/L:.3f}\ \mathrm{{N}}\ \ (= mg(3-2\cos\theta_0))"]),
        ],
        numeric_steps=[
            ("Ecuación de movimiento", "Se integra la ecuación exacta del péndulo, sin aproximar $\\sin\\theta\\approx\\theta$, con el estado $y=(\\theta,\\omega)$. Se usan $n=600$ pasos en 6 s, de modo que $h=0.01$ s.",
             [rf"\dot\theta = \omega,\qquad \dot\omega = -\dfrac gL\sin\theta = -{g/L:.4f}\sin\theta,\qquad y_0 = ({r0:.6f}\ \mathrm{{rad}},\ 0)"]),
            ("Energía inicial de referencia", "Con $K=\\tfrac12m(L\\omega)^2$ y $U=mgL(1-\\cos\\theta)$ se calcula $E_0$. La deriva es $\\max|E-E_0|/E_0$.",
             [rf"K_0 = 0,\quad U_0 = {float(U0):.5f}\ \mathrm{{J}},\quad E_0 = {E0:.5f}\ \mathrm{{J}}"]),
            ("Primer paso RK4", "Cuatro pendientes con $h=0.01$ s.",
             [rf"k_1 = {_v(k1,6)},\quad k_2 = {_v(k2,6)}", rf"k_3 = {_v(k3,6)},\quad k_4 = {_v(k4,6)}"]),
            ("Actualizar y medir la energía", "Se combina con pesos $1,2,2,1$ y se recalcula la energía.",
             [rf"y_1 = ({y1[0]:.7f},\ {y1[1]:.7f}),\qquad E_1 = {e_of(y1):.7f}\ \mathrm{{J}},\qquad \dfrac{{E_1-E_0}}{{E_0}} = {sci((e_of(y1)-E0)/E0,2)}"]),
            ("Mismo paso con Euler y con Verlet", "Euler usa solo $k_1$: $y_1=y_0+hk_1$. Verlet de velocidad actualiza $\\theta_1=\\theta_0+\\omega_0h+\\tfrac12a_0h^2$ y $\\omega_1=\\omega_0+\\tfrac12(a_0+a_1)h$.",
             [rf"\text{{Euler: }}y_1=({ye[0]:.7f},\ {ye[1]:.7f}),\quad \dfrac{{E_1-E_0}}{{E_0}} = {sci((e_of(ye)-E0)/E0,2)}",
              rf"\text{{Verlet: }}y_1=({thv:.7f},\ {omv:.7f}),\quad \dfrac{{E_1-E_0}}{{E_0}} = {sci((e_of([thv,omv])-E0)/E0,2)}"]),
            ("Deriva tras 600 pasos", "Se repite hasta 6 s y se toma el máximo de $|E-E_0|/E_0$. Euler gana energía en cada paso (el error se acumula), Verlet oscila acotado y RK4 es mínimo.",
             [rf"\text{{Euler: }}{sci(drift['euler'],3)},\qquad \text{{Verlet: }}{sci(drift['verlet'],3)},\qquad \text{{RK4: }}{sci(drift['rk4'],3)}"]),
            ("Rapidez en el punto más bajo", "Se repite RK4 con 6000 pasos y se toma $\\max|\\omega|\\,L$, que ocurre cuando $\\theta=0$.",
             [rf"v^{{num}} = L\max|\omega| = {vn:.5f}\ \mathrm{{m/s}},\qquad v = {va:.5f}\ \mathrm{{m/s}},\qquad |\Delta| = {sci(abs(vn-va),2)}\ \mathrm{{m/s}}"]),
        ],
    )


def _springfric() -> dict:
    k, m, mu, x0, g = c8.KS, c8.M, c8.MU, c8.X0, c8.G
    d = c8.delta()
    an = c8.turning_points_analytic()
    ts, xs, vs, turn = c8.spring_friction_numeric()
    f = mu * m * g
    dt = 1e-4
    direc = -np.sign(x0)
    rhs = lambda _t, y: np.array([y[1], (-k * y[0] - direc * mu * m * g) / m])  # noqa: E731
    y0 = np.array([x0, 0.0])
    k1, k2, k3, k4, y1 = _ks(rhs, 0.0, y0, dt)
    rowsA = r"\begin{array}{ccc}n & x_n^{analit}\,(\mathrm{m}) & x_n^{num}\,(\mathrm{m})\\\hline " + r"\\ ".join(
        rf"{i} & {an[i]:+.5f} & {turn[i]:+.5f}" for i in range(len(an))) + r"\end{array}"
    return dict(
        explanation=[
            "Una fuerza no conservativa, como la fricción cinética, no tiene energía potencial asociada: el trabajo que realiza depende del camino recorrido y se pierde como calor. El balance correcto es $W_{nc}=\\Delta E_{mec}$: la energía mecánica disminuye exactamente en lo que la fricción disipa.",
            "Un bloque en un resorte con fricción cinética ya no oscila indefinidamente. En cada recorrido la fricción realiza un trabajo $-\\mu_k mg\\,d$ y roba energía, de modo que cada semiciclo es más corto que el anterior. Aplicando el balance entre dos puntos de retorno sucesivos se obtiene que la amplitud disminuye una cantidad constante $2\\delta$ con $\\delta=\\mu_kmg/k$, es decir, decrece linealmente y no exponencialmente.",
            "La magnitud $\\delta$ tiene un significado claro: es el desplazamiento a partir del cual la fuerza elástica $k|x|$ ya no supera la fricción máxima. Dentro de la banda $|x|\\le\\delta$ la fricción estática puede sostener al bloque, y por eso el movimiento termina de forma abrupta en un punto que en general no es la posición de equilibrio del resorte, $x=0$.",
            "Con los datos del problema, $\\delta$ es de apenas unos 2.5 cm frente a una compresión inicial de 20 cm, así que el bloque da varios semiciclos antes de detenerse. El modelo supone $\\mu_k$ constante e igual a la fricción estática máxima de parada, una idealización razonable.",
        ],
        analytic_steps=[
            ("Datos y energía inicial", "El bloque parte del reposo con el resorte comprimido, de modo que toda la energía es elástica.",
             [rf"k = {fx(k,0)}\ \mathrm{{N/m}},\quad m = {m}\ \mathrm{{kg}},\quad \mu_k = {mu},\quad x_0 = {x0}\ \mathrm{{m}}",
              rf"E_0 = \tfrac12kx_0^2 = \tfrac12({fx(k,0)})({x0})^2 = {0.5*k*x0**2:.3f}\ \mathrm{{J}}"]),
            ("Fuerza de fricción cinética", "Es constante en módulo mientras el bloque se mueve.",
             [rf"f_k = \mu_k mg = ({mu})({m})({g}) = {f:.4f}\ \mathrm{{N}}"]),
            ("Balance de energía en un semiciclo", "Entre un retorno en $x_n$ y el siguiente en $-x_{n+1}$ el bloque recorre $|x_n|+|x_{n+1}|$ y la fricción disipa $f_k$ por esa distancia.",
             [r"\tfrac12kx_n^2 - \tfrac12kx_{n+1}^2 = \mu_k mg\,(x_n+x_{n+1})"]),
            ("Reducción de amplitud", "Se factoriza la diferencia de cuadrados y se cancela $(x_n+x_{n+1})$.",
             [r"\tfrac12k(x_n-x_{n+1})(x_n+x_{n+1}) = \mu_k mg(x_n+x_{n+1})\;\Rightarrow\; x_{n+1} = x_n - 2\delta,\qquad \delta=\dfrac{\mu_k mg}{k}"]),
            ("Valor de δ", "Se sustituyen los datos.",
             [rf"\delta = \dfrac{{{f:.4f}\ \mathrm{{N}}}}{{{fx(k,0)}\ \mathrm{{N/m}}}} = {d:.5f}\ \mathrm{{m}},\qquad 2\delta = {2*d:.5f}\ \mathrm{{m}}"]),
            ("Puntos de retorno sucesivos", "Los signos se alternan; la amplitud baja $2\\delta$ cada vez.",
             [rf"|x_n| = x_0-2n\delta:\quad " + r",\quad ".join(rf"|x_{i}| = {abs(an[i]):.5f}" for i in range(len(an)))]),
            ("Dónde se detiene", "El bloque se detiene cuando, en un punto de retorno, $k|x|\\le\\mu_s mg$, es decir $|x|\\le\\delta$ (aquí se toma $\\mu_s=\\mu_k$). El último punto es el que cumple esto por primera vez.",
             [rf"|x_{len(an)-1}| = {abs(an[-1]):.5f}\ \mathrm{{m}}\ <\ \delta = {d:.5f}\ \mathrm{{m}}\;\Rightarrow\;\text{{se detiene en }}x = {an[-1]:+.4f}\ \mathrm{{m}}",
              rf"N_{{semiciclos}} = {len(an)-1}"]),
        ],
        numeric_steps=[
            ("Ecuación con fricción de sentido variable", "La fricción siempre se opone a la velocidad, así que su signo cambia en cada retorno. El código mantiene una variable de dirección $d=\\mathrm{sgn}(v)$ (al partir del reposo, $d=-\\mathrm{sgn}(x)$, pues el resorte empuja hacia $x=0$) y resuelve $m\\ddot x=-kx-d\\mu_kmg$.",
             [rf"\dot x = v,\qquad \dot v = \dfrac{{-kx - d\,\mu_kmg}}{{m}},\qquad d={direc:+.0f},\qquad \Delta t = {sci(dt,2)}\ \mathrm{{s}}"]),
            ("Condición de arranque", "Si $v=0$ y $k|x|\\le\\mu m g$ el bloque no se mueve. Aquí $k|x_0|=20$ N supera a la fricción:",
             [rf"k|x_0| = {k*abs(x0):.3f}\ \mathrm{{N}}\ >\ \mu mg = {f:.4f}\ \mathrm{{N}}\;\Rightarrow\;\text{{se mueve}}"]),
            ("Primer paso RK4", rf"Aceleración inicial $a_0=(-kx_0-d\mu mg)/m={(-k*x0-direc*f)/m:.4f}$ m/s². Cuatro pendientes con $h={sci(dt,2)}$ s:",
             [rf"k_1 = {_v(k1)}", rf"k_2 = {_v(k2)}", rf"k_3 = {_v(k3)},\qquad k_4 = {_v(k4)}"]),
            ("Actualizar", "Combinación $1,2,2,1$.",
             [rf"y_1 = ({y1[0]:.7f}\ \mathrm{{m}},\ {y1[1]:.7f}\ \mathrm{{m/s}})"]),
            ("Detectar los retornos", "Después de cada paso se compara el signo de $v$. Si cambió, el retorno está dentro del paso y se interpola linealmente con $\\text{frac}=v/(v-v_{nuevo})$: $x_{ret}=x+\\text{frac}\\,(x_{nuevo}-x)$. Entonces $v=0$ y la dirección se invierte.",
             [r"x_{ret} = x + \dfrac{v}{v-v_{nuevo}}\,(x_{nuevo}-x)"]),
            ("Parada", "Tras cada retorno, con $v=0$, se evalúa si $k|x|\\le\\mu_kmg$. Si es así se termina la integración.",
             [rf"k|x_{len(an)-1}| = ({fx(k,0)})({abs(turn[-1]):.5f}) = {k*abs(turn[-1]):.4f}\ \mathrm{{N}}\ \le\ {f:.4f}\ \mathrm{{N}}\;\Rightarrow\;\text{{parada}}",
              rf"t_{{parada}} = {ts[-1]:.4f}\ \mathrm{{s}},\qquad \text{{pasos}}\approx {len(ts)-1}"]),
            ("Comparar con el análisis", "Los retornos medidos coinciden con la recurrencia $x_{n+1}=x_n-2\\delta$ a cinco decimales o mejor; la pequeña diferencia viene del error de la interpolación lineal.",
             [rowsA,
              rf"\max_n|x_n^{{num}}-x_n| = {sci(max(abs(a_-b_) for a_, b_ in zip(an, turn)),2)}\ \mathrm{{m}}"]),
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
