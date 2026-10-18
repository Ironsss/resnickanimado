"""Explicacion y pasos a mano, Volumen 2, capitulos 1 a 4 (Coulomb, campo, Gauss, potencial).

Todos los numeros se calculan con `physics/` o replicando su algoritmo en pasos pequenos.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch01_coulomb as e1
from physics import v2ch02_field as e2
from physics import v2ch03_gauss as e3
from physics import v2ch04_potential as e4
from physics.constants import EPS0, K_E
from physics.integrators import rk4_step


def _vec(v, d=4, f=sci) -> str:
    return "(" + ",\\ ".join(f(float(c), d) for c in v) + ")"


# =========================================================================== CoulombForces
def _coulomb() -> dict:
    q, pos = e1.Q, e1.POS
    q1, q2, q3 = q
    r31 = pos[2] - pos[0]
    r32 = pos[2] - pos[1]
    d31, d32 = float(np.linalg.norm(r31)), float(np.linalg.norm(r32))
    f31m, f32m = K_E * q3 * q1 / d31**2, K_E * q3 * q2 / d32**2
    u31, u32 = r31 / d31, r32 / d32
    f31, f32 = f31m * u31, f32m * u32
    fa = e1.force_analytic()
    fn = e1.force_numeric()
    f1, f2 = e1.force_analytic(0), e1.force_analytic(1)
    tot = f1 + f2 + fa
    h = 1e-6
    pp, pm = pos.copy(), pos.copy()
    pp[2, 0] += h
    pm[2, 0] -= h
    up, um = e1.potential_energy(pp), e1.potential_energy(pm)
    qp, qm = pos.copy(), pos.copy()
    qp[2, 1] += h
    qm[2, 1] -= h
    vp, vm = e1.potential_energy(qp), e1.potential_energy(qm)
    u0 = e1.potential_energy(pos)
    return dict(
        explanation=[
            "La ley de Coulomb describe la fuerza entre dos cargas puntuales en reposo: es proporcional al producto de las cargas, inversamente proporcional al cuadrado de la distancia y está dirigida a lo largo de la recta que las une. Las cargas del mismo signo se repelen y las de signo opuesto se atraen. La constante $k = 1/(4\\pi\\varepsilon_0) \\approx 8.99\\times10^{9}\\ \\mathrm{N\\,m^2/C^2}$ es enorme porque el coulomb es una carga muy grande: por eso las cargas de laboratorio se miden en microcoulombs.",
            "Cuando hay más de dos cargas, la fuerza sobre una de ellas se obtiene por superposición: se calcula la fuerza de cada una de las demás como si estuviera sola y se suman los vectores. La superposición es una propiedad experimental, no una consecuencia de la ley de Coulomb, y es lo que permite tratar cualquier distribución como una suma de cargas puntuales.",
            "Aquí es esencial sumar vectores y no magnitudes. Cada fuerza tiene la dirección del vector que va de la carga fuente a la carga sobre la que se actúa, de modo que el signo del producto $q_iq_j$ decide si ese vector se toma hacia afuera (repulsión) o hacia adentro (atracción) sin necesidad de dibujar flechas aparte.",
            "Como las fuerzas de Coulomb son pares de acción y reacción, la suma de todas las fuerzas internas de un sistema aislado de cargas es nula (tercera ley de Newton). Además, la fuerza electrostática es conservativa: $\\vec F = -\\nabla U$, donde $U$ es la energía potencial del conjunto. Esta segunda vía es la que usa la verificación numérica, y es independiente de la primera porque no se escribe en ningún momento la ley de Coulomb vectorial.",
        ],
        analytic_steps=[
            ("Plantear la geometría", "La fuerza sobre $q_3$ es la suma de la que ejerce $q_1$ y la que ejerce $q_2$. Para cada una se necesita el vector que apunta de la fuente a $q_3$ y su longitud.",
             [rf"\vec r_{{31}} = \vec r_3 - \vec r_1 = (0.2,\ 0.3)\ \mathrm{{m}},\qquad r_{{31}} = \sqrt{{0.2^2+0.3^2}} = {d31:.5f}\ \mathrm{{m}}",
              rf"\vec r_{{32}} = \vec r_3 - \vec r_2 = (-0.2,\ 0.3)\ \mathrm{{m}},\qquad r_{{32}} = {d32:.5f}\ \mathrm{{m}}"]),
            ("Escribir la ley de Coulomb en forma vectorial", "Con el vector unitario $\\hat r = \\vec r/r$ del paso anterior, la fuerza de la carga $j$ sobre la $i$ ya incluye la atracción o repulsión por el signo de $q_iq_j$.",
             [r"\vec F_{3j} = k\,\dfrac{q_3 q_j}{r_{3j}^2}\,\hat r_{3j} = k\,q_3 q_j\,\dfrac{\vec r_{3j}}{r_{3j}^3}"]),
            ("Fuerza de $q_1$ sobre $q_3$", "Las dos cargas son positivas, la fuerza es repulsiva y apunta en el sentido de $\\hat r_{31}$. Primero la magnitud y después las componentes.",
             [rf"F_{{31}} = \dfrac{{({sci(K_E,5)})({sci(q3,3)})({sci(q1,3)})}}{{({d31:.5f})^2}} = {f31m * 1000:.4f}\ \mathrm{{mN}}",
              rf"\vec F_{{31}} = {f31m * 1000:.4f}\,({u31[0]:.5f},\ {u31[1]:.5f})\ \mathrm{{mN}} = ({f31[0] * 1000:.4f},\ {f31[1] * 1000:.4f})\ \mathrm{{mN}}"]),
            ("Fuerza de $q_2$ sobre $q_3$", "Aquí $q_2q_3 < 0$, por lo que el resultado de la magnitud sale negativo: la fuerza es atractiva y el signo ya la dirige hacia $q_2$ al multiplicar por $\\hat r_{32}$.",
             [rf"F_{{32}} = \dfrac{{({sci(K_E,5)})({sci(q3,3)})({sci(q2,3)})}}{{({d32:.5f})^2}} = {f32m * 1000:.4f}\ \mathrm{{mN}}",
              rf"\vec F_{{32}} = {f32m * 1000:.4f}\,({u32[0]:.5f},\ {u32[1]:.5f})\ \mathrm{{mN}} = ({f32[0] * 1000:.4f},\ {f32[1] * 1000:.4f})\ \mathrm{{mN}}"]),
            ("Sumar por superposición", "Se suman las componentes por separado; la magnitud y el ángulo salen del resultado.",
             [rf"F_x = {f31[0] * 1000:.4f} + ({f32[0] * 1000:.4f}) = {fa[0] * 1000:.4f}\ \mathrm{{mN}},\qquad F_y = {f31[1] * 1000:.4f} + ({f32[1] * 1000:.4f}) = {fa[1] * 1000:.4f}\ \mathrm{{mN}}",
              rf"|\vec F_3| = \sqrt{{({fa[0] * 1000:.4f})^2 + ({fa[1] * 1000:.4f})^2}} = {np.linalg.norm(fa) * 1000:.4f}\ \mathrm{{mN}},\qquad \theta = {np.degrees(np.arctan2(fa[1], fa[0])):.2f}^\circ"]),
            ("Fuerzas sobre las otras dos cargas", "Para comprobar la tercera ley se repite el mismo procedimiento con $q_1$ y con $q_2$ como carga de prueba.",
             [rf"\vec F_1 = {_vec(f1 * 1000, 4, fx)}\ \mathrm{{mN}},\qquad \vec F_2 = {_vec(f2 * 1000, 4, fx)}\ \mathrm{{mN}}"]),
            ("Verificar que la suma interna es cero", "Las seis fuerzas se anulan por pares, de modo que la suma de las tres fuerzas netas debe ser cero. El residuo que queda es solo error de redondeo de la aritmética de punto flotante.",
             [rf"\vec F_1 + \vec F_2 + \vec F_3 = {_vec(tot * 1000, 3, sci)}\ \mathrm{{mN}} \approx \vec 0"]),
        ],
        numeric_steps=[
            ("Cambiar de método: derivar la energía", "En lugar de sumar fuerzas, se usa que $\\vec F = -\\nabla U$. La energía de las tres cargas es la suma de $k\\,q_iq_j/r_{ij}$ sobre los tres pares, y solo tres números dependen de la posición de $q_3$ en cada evaluación.",
             [r"U(\vec r_3) = k\left(\dfrac{q_1q_2}{r_{12}} + \dfrac{q_1q_3}{r_{13}} + \dfrac{q_2q_3}{r_{23}}\right)",
              rf"U_0 = {sci(u0, 8)}\ \mathrm{{J}}"]),
            ("Elegir el paso de la derivada", "La derivada parcial se aproxima con diferencias centradas, de error $O(h^2)$. Un paso demasiado pequeño amplifica el redondeo y uno grande amplifica el truncamiento; $h = 1\\ \\mu\\mathrm{m}$ es un compromiso razonable.",
             [r"F_x \approx -\dfrac{U(x_3+h,y_3) - U(x_3-h,y_3)}{2h},\qquad h = 10^{-6}\ \mathrm{m}"]),
            ("Calcular $F_x$", "Se mueve $q_3$ una distancia $h$ a la derecha y a la izquierda, se recalcula la energía completa y se resta.",
             [rf"U(x_3+h) = {sci(up, 10)}\ \mathrm{{J}},\qquad U(x_3-h) = {sci(um, 10)}\ \mathrm{{J}}",
              rf"F_x \approx -\dfrac{{{sci(up, 10)} - {sci(um, 10)}}}{{2\,({sci(h, 1)})}} = {fn[0] * 1000:.6f}\ \mathrm{{mN}}"]),
            ("Calcular $F_y$", "Se repite el procedimiento desplazando $q_3$ en la dirección $y$.",
             [rf"U(y_3+h) = {sci(vp, 10)}\ \mathrm{{J}},\qquad U(y_3-h) = {sci(vm, 10)}\ \mathrm{{J}}",
              rf"F_y \approx -\dfrac{{{sci(vp, 10)} - {sci(vm, 10)}}}{{2\,({sci(h, 1)})}} = {fn[1] * 1000:.6f}\ \mathrm{{mN}}"]),
            ("Comparar con el cálculo analítico", "Las dos componentes coinciden con las del cálculo vectorial hasta el error de la derivada numérica (y del redondeo, que domina al restar dos energías casi iguales).",
             [rf"\Delta F_x = {sci(abs(fn[0] - fa[0]) * 1000, 2)}\ \mathrm{{mN}},\qquad \Delta F_y = {sci(abs(fn[1] - fa[1]) * 1000, 2)}\ \mathrm{{mN}}"]),
            ("Verificación de la tercera ley", "Se calculan con el método vectorial las fuerzas sobre cada carga y se suman; el módulo del resultado es el residuo que aparece en la tabla de verificación.",
             [rf"\left|\vec F_1 + \vec F_2 + \vec F_3\right| = {sci(e1.newton_third_law_residual(), 3)}\ \mathrm{{N}}"]),
        ],
    )


# =========================================================================== EquilibriumPoint
def _bisect_rows(f_pos_goes_lo, lo, hi, n):
    rows = []
    for k in range(1, n + 1):
        mid = 0.5 * (lo + hi)
        up = f_pos_goes_lo(mid)
        rows.append((k, lo, hi, mid, up))
        if up:
            lo = mid
        else:
            hi = mid
    return rows, lo, hi


def _equilibrium() -> dict:
    q1, q2, d = e1.Q1, e1.Q2, e1.SEP
    xa, xn = e1.zero_force_point_analytic(), e1.zero_force_point_numeric()
    oa, on = e1.opposite_charges_point_analytic(), e1.opposite_charges_point_numeric()
    ratio = float(np.sqrt(q1 / q2))
    lo0, hi0 = 1e-6 * d, d * (1 - 1e-6)
    rows1, lo1, hi1 = _bisect_rows(lambda m: float(e1.net_force_on_test(m)) > 0, lo0, hi0, 6)
    tab1 = r"\begin{array}{c|cccc}k & x_{\min}\ (\mathrm{m}) & x_{\max}\ (\mathrm{m}) & x_{\mathrm{mid}}\ (\mathrm{m}) & F(x_{\mathrm{mid}})\\\hline " + r"\\ ".join(
        f"{k} & {a:.6f} & {b:.6f} & {m:.6f} & {'>0' if up else '<0'}" for k, a, b, m, up in rows1) + r"\end{array}"
    f_mid1 = float(e1.net_force_on_test(rows1[0][3]))
    f_lo = lambda x: q1 / x**2 + (-1e-6) / (x - d) ** 2  # noqa: E731
    lo2, hi2 = d * (1 + 1e-9), 100 * d
    rows2 = []
    for k in range(1, 6):
        mid = 0.5 * (lo2 + hi2)
        up = f_lo(mid) > 0
        rows2.append((k, lo2, hi2, mid, up))
        if up:
            hi2 = mid
        else:
            lo2 = mid
    tab2 = r"\begin{array}{c|cccc}k & x_{\min}\ (\mathrm{m}) & x_{\max}\ (\mathrm{m}) & x_{\mathrm{mid}}\ (\mathrm{m}) & \text{decisión}\\\hline " + r"\\ ".join(
        f"{k} & {a:.4f} & {b:.4f} & {m:.4f} & {'x_{\\max}=x_{\\mathrm{mid}}' if up else 'x_{\\min}=x_{\\mathrm{mid}}'}" for k, a, b, m, up in rows2) + r"\end{array}"
    return dict(
        explanation=[
            "Una carga de prueba está en equilibrio cuando la fuerza neta que recibe es cero. Con dos cargas fijas, cada una empuja o jala a la de prueba con una fuerza que decrece como $1/r^2$, así que el equilibrio exige que ambas fuerzas tengan igual magnitud y sentidos opuestos. La carga de prueba es pequeña y no perturba a las otras.",
            "Si las dos cargas fuente tienen el mismo signo, las fuerzas son opuestas solo en la región entre ellas: allí una repele hacia un lado y la otra hacia el otro. El punto de equilibrio queda más cerca de la carga menor, porque debe compensar con cercanía lo que le falta de carga. La razón de distancias es $\\sqrt{q_1/q_2}$ y no $q_1/q_2$, por la dependencia cuadrática.",
            "Si las cargas fuente tienen signos opuestos, entre ellas ambas fuerzas apuntan en el mismo sentido y no pueden cancelarse. El equilibrio solo puede existir fuera del segmento, y del lado de la carga de menor magnitud, donde la mayor está más lejos pero tiene más carga. Es un equilibrio inestable: si la carga de prueba se desvía un poco, la fuerza neta la aleja todavía más.",
            "Aunque la carga de prueba tenga un valor arbitrario, ni su signo ni su magnitud afectan la posición del equilibrio, porque se factoriza en ambas fuerzas. Por eso la posición es una propiedad del par de cargas fuente y no de la carga de prueba.",
        ],
        analytic_steps=[
            ("Plantear la fuerza neta sobre la carga de prueba", "Se coloca $q_1$ en el origen y $q_2$ en $x = d$. Una carga de prueba positiva $q_t$ en $0<x<d$ es repelida por $q_1$ hacia la derecha y por $q_2$ hacia la izquierda.",
             [rf"F(x) = k\,q_t\left(\dfrac{{q_1}}{{x^2}} - \dfrac{{q_2}}{{(d-x)^2}}\right),\qquad q_1 = {sci(q1,3)}\ \mathrm{{C}},\ q_2 = {sci(q2,3)}\ \mathrm{{C}},\ d = {d}\ \mathrm{{m}}"]),
            ("Imponer equilibrio", "Se iguala a cero. La constante $k\\,q_t$ se cancela, lo que confirma que la posición no depende de la carga de prueba.",
             [r"\dfrac{q_1}{x^2} = \dfrac{q_2}{(d-x)^2}"]),
            ("Sacar raíz cuadrada", "Se toma la raíz positiva, porque en esta región $x$ y $d-x$ son ambas positivas. Queda una ecuación lineal en $x$.",
             [r"\dfrac{x}{d-x} = \sqrt{\dfrac{q_1}{q_2}}",
              rf"\sqrt{{\dfrac{{{sci(q1,3)}}}{{{sci(q2,3)}}}}} = {ratio:.4f}\quad\Rightarrow\quad x = {ratio:.4f}\,(d - x)"]),
            ("Despejar $x$", "Se agrupan los términos en $x$ y se sustituyen los datos.",
             [r"x = d\,\dfrac{\sqrt{q_1}}{\sqrt{q_1}+\sqrt{q_2}}",
              rf"x = ({d})\,\dfrac{{{ratio:.4f}}}{{{ratio:.4f}+1}} = {xa:.6f}\ \mathrm{{m}}"]),
            ("Comprobar", "A $x = 0.4$ m las fuerzas por unidad de $kq_t$ deben ser iguales; la distancia a $q_2$ es $0.2$ m.",
             [rf"\dfrac{{q_1}}{{x^2}} = \dfrac{{{sci(q1,3)}}}{{({xa:.4f})^2}} = {sci(q1 / xa**2, 4)},\qquad \dfrac{{q_2}}{{(d-x)^2}} = \dfrac{{{sci(q2,3)}}}{{({d - xa:.4f})^2}} = {sci(q2 / (d - xa)**2, 4)}\ \ \mathrm{{C/m^2}}"]),
            ("Caso $q_2 = -1\\ \\mu\\mathrm{C}$: ¿dónde puede estar?", "Entre las cargas ambas fuerzas tienen el mismo sentido, así que solo queda fuera del segmento. Para $x>d$ la repulsión de $q_1$ apunta a la derecha y la atracción de $q_2$ a la izquierda, y se compensan.",
             [r"\dfrac{|q_1|}{x^2} = \dfrac{|q_2|}{(x-d)^2},\qquad \dfrac{x}{x-d} = \sqrt{\dfrac{|q_1|}{|q_2|}} = 2"]),
            ("Despejar y evaluar", "Se resuelve la ecuación lineal $x = 2(x-d)$. El punto está al otro lado de $q_2$, a una distancia $d$ de ella.",
             [r"x = d\,\dfrac{r}{r-1},\quad r = \sqrt{|q_1/q_2|}",
              rf"x = ({d})\,\dfrac{{{np.sqrt(4):.1f}}}{{{np.sqrt(4):.1f}-1}} = {oa:.6f}\ \mathrm{{m}}"]),
        ],
        numeric_steps=[
            ("Elegir el método: bisección", "Se busca la raíz de $F(x)$ en un intervalo donde la función cambia de signo. La bisección no necesita derivadas y siempre converge: cada iteración reduce el intervalo a la mitad.",
             [rf"\text{{Intervalo inicial: }}[x_{{\min}},x_{{\max}}] = [{lo0:.7f},\ {hi0:.7f}]\ \mathrm{{m}}\quad(10^{{-6}}d\ \text{{de cada extremo}})"]),
            ("Primera iteración", "El punto medio es el valor central del intervalo. En él se evalúa el signo de la fuerza (con $q_t = 1\\ \\mathrm{nC}$): positiva significa que la repulsión de $q_1$ domina, por lo que la raíz está más a la derecha y se descarta la mitad izquierda.",
             [rf"x_{{\mathrm{{mid}}}} = \dfrac{{{lo0:.7f}+{hi0:.7f}}}{{2}} = {rows1[0][3]:.6f}\ \mathrm{{m}}",
              rf"F(x_{{\mathrm{{mid}}}}) = {sci(f_mid1, 4)}\ \mathrm{{N}} > 0\ \Rightarrow\ x_{{\min}} = x_{{\mathrm{{mid}}}}"]),
            ("Primeras seis iteraciones", "El intervalo se encoge a la mitad cada vez y se acerca al valor analítico $0.4$ m.",
             [tab1]),
            ("Criterio de parada", "El código ejecuta 100 iteraciones fijas. La anchura del intervalo inicial se divide entre $2^{100}$, mucho menor que la precisión de un número de doble precisión, así que el resultado queda limitado por el redondeo.",
             [rf"\text{{ancho final}} = \dfrac{{{hi0 - lo0:.5f}}}{{2^{{100}}}} \approx {sci((hi0 - lo0) / 2**100, 2)}\ \mathrm{{m}}"]),
            ("Resultado del primer caso", "El punto medio del intervalo final es la estimación.",
             [rf"x_{{num}} = {xn:.10f}\ \mathrm{{m}},\qquad x_{{an}} = {xa:.10f}\ \mathrm{{m}},\qquad |\Delta x| = {sci(abs(xn - xa), 2)}\ \mathrm{{m}}"]),
            ("Segundo caso: otro intervalo", "Con $q_2<0$ la raíz está fuera del segmento, en $x>d$. Se parte de $[d(1+10^{-9}),\\,100\\,d]$ y se usa el signo de $q_1/x^2 + q_2/(x-d)^2$: si es positivo, domina $q_1$ y la raíz está más cerca, por lo que baja el extremo derecho.",
             [tab2]),
            ("Resultado del segundo caso", "Con 200 iteraciones el intervalo queda reducido hasta el límite de la precisión de máquina.",
             [rf"x_{{num}} = {on:.10f}\ \mathrm{{m}},\qquad x_{{an}} = {oa:.10f}\ \mathrm{{m}},\qquad |\Delta x| = {sci(abs(on - oa), 2)}\ \mathrm{{m}}"]),
        ],
    )


# =========================================================================== DipoleField
def _dipole() -> dict:
    q, d = e2.QD, e2.DSEP
    z = 0.5
    p = q * d
    t1, t2 = 1 / (z - d / 2) ** 2, 1 / (z + d / 2) ** 2
    exa, exf = float(e2.dipole_field_axis_exact(z)), float(e2.dipole_field_axis_far(z))
    x0, y0 = 0.3, 0.2
    h = 1e-6
    vxp, vxm = e2.potential_dipole(x0 + h, y0), e2.potential_dipole(x0 - h, y0)
    vyp, vym = e2.potential_dipole(x0, y0 + h), e2.potential_dipole(x0, y0 - h)
    en, ev = e2.field_from_potential_numeric(x0, y0), e2.field_dipole_vector(x0, y0)
    rp, rm = np.array([x0 - d / 2, y0]), np.array([x0 + d / 2, y0])
    dp, dm = float(np.linalg.norm(rp)), float(np.linalg.norm(rm))
    ep, em = K_E * q * rp / dp**3, -K_E * q * rm / dm**3
    # linea de campo: primer paso RK4
    ds = 5e-5
    start = np.array([d / 2, 0.0]) + 2e-4 * np.array([0.0, 1.0])
    rhs = lambda _s, r: (lambda e: e / np.linalg.norm(e))(e2.field_dipole_vector(r[0], r[1]))  # noqa: E731
    k1 = rhs(0, start)
    k2 = rhs(0, start + ds / 2 * k1)
    k3 = rhs(0, start + ds / 2 * k2)
    k4 = rhs(0, start + ds * k3)
    r1 = rk4_step(rhs, 0.0, start, ds)
    pts, ok = e2.trace_field_line(90)
    return dict(
        explanation=[
            "El campo eléctrico $\\vec E$ en un punto es la fuerza por unidad de carga que sentiría una carga de prueba positiva colocada allí. Una carga puntual produce un campo radial que decrece como $1/r^2$, y el campo de varias cargas es la suma vectorial de los campos individuales. Las líneas de campo son curvas cuya tangente en cada punto es $\\vec E$: salen de las cargas positivas, terminan en las negativas y su densidad indica la intensidad del campo.",
            "Un dipolo son dos cargas $+q$ y $-q$ separadas una distancia $d$. Su carga neta es cero, pero los campos de ambas no se cancelan exactamente porque las cargas están en puntos distintos. Sobre el eje, el campo de la carga más cercana gana ligeramente al de la más lejana, y esa pequeña diferencia es el campo del dipolo.",
            "Para distancias grandes frente a la separación, $z\\gg d$, el resultado se simplifica a $E\\approx 2kp/z^3$, con el momento dipolar $p = qd$. El campo cae como $1/z^3$, mucho más rápido que el de una carga aislada, justamente porque las dos contribuciones casi se cancelan. La corrección relativa es del orden de $d^2/(2z^2)$.",
            "La otra manera de obtener el campo es mediante el potencial, que es un escalar y por eso más fácil de sumar: $\\vec E = -\\nabla V$. Verificar que el gradiente numérico del potencial coincide con la suma vectorial de Coulomb es una comprobación independiente de que el campo es conservativo.",
        ],
        analytic_steps=[
            ("Plantear el campo sobre el eje", "Con $+q$ en $z = d/2$ y $-q$ en $z = -d/2$, un punto del eje con $z>d/2$ recibe el campo de $+q$ hacia afuera y el de $-q$ hacia adentro. Ambos son axiales y se restan.",
             [r"E(z) = kq\left[\dfrac{1}{(z-d/2)^2} - \dfrac{1}{(z+d/2)^2}\right]"]),
            ("Distancias a cada carga", "Se sustituyen $z = 0.50$ m y $d = 0.02$ m.",
             [rf"z - \dfrac d2 = {z - d / 2:.2f}\ \mathrm{{m}},\qquad z + \dfrac d2 = {z + d / 2:.2f}\ \mathrm{{m}}"]),
            ("Evaluar la fórmula exacta", "Los dos términos son casi iguales: cada uno vale unos 4 $\\mathrm{m^{-2}}$ y la resta es del orden de $0.03$ de ese valor. Hay que conservar varias cifras para no perder la diferencia.",
             [rf"E = ({sci(K_E,5)})({sci(q,3)})\left[\dfrac{{1}}{{({z - d / 2:.2f})^2}} - \dfrac{{1}}{{({z + d / 2:.2f})^2}}\right] = ({sci(K_E * q, 5)})\,[{t1:.5f} - {t2:.5f}]",
              rf"E_{{exacto}} = {exa:.5f}\ \mathrm{{N/C}}"]),
            ("Desarrollar para $z\\gg d$", "Se reúnen las dos fracciones: $1/(z-a)^2 - 1/(z+a)^2 = 4az/(z^2-a^2)^2$ con $a=d/2$. Si $a\\ll z$, el denominador es $\\approx z^4$ y el resultado es el campo dipolar.",
             [r"E = \dfrac{2kq\,d\,z}{(z^2 - d^2/4)^2} \approx \dfrac{2kqd}{z^3} = \dfrac{2kp}{z^3}"]),
            ("Evaluar el campo dipolar", "El momento dipolar es $p = qd$.",
             [rf"p = ({sci(q,3)})({d}) = {sci(p,3)}\ \mathrm{{C\,m}}",
              rf"E_{{lejano}} = \dfrac{{2({sci(K_E,5)})({sci(p,3)})}}{{({z})^3}} = {exf:.5f}\ \mathrm{{N/C}}"]),
            ("Estimar el error de la aproximación", "El primer término despreciado da una corrección relativa $d^2/(2z^2)$.",
             [rf"\dfrac{{E_{{exacto}} - E_{{lejano}}}}{{E_{{exacto}}}} = {(exa - exf) / exa:.3e}\ \ (\text{{teórico: }} -\dfrac{{d^2}}{{2z^2}}\approx {-d**2 / (2 * z**2):.3e})".replace("e-0", r"\times10^{-").replace("e+0", r"\times10^{") if False else
              rf"\dfrac{{E_{{exacto}} - E_{{lejano}}}}{{E_{{exacto}}}} = {sci((exa - exf) / exa, 3)},\qquad \dfrac{{d^2}}{{2z^2}} = {sci(d**2 / (2 * z**2), 3)}"]),
        ],
        numeric_steps=[
            ("Potencial como puente", "El potencial del dipolo en el plano es la suma escalar $V = kq(1/r_+ - 1/r_-)$ con $r_\\pm$ las distancias a cada carga. El campo se obtiene derivando: $E_x = -\\partial V/\\partial x$, $E_y = -\\partial V/\\partial y$.",
             [rf"V(x,y) = kq\left(\dfrac{{1}}{{\sqrt{{(x-d/2)^2+y^2}}}} - \dfrac{{1}}{{\sqrt{{(x+d/2)^2+y^2}}}}\right),\qquad (x,y) = ({x0},\ {y0})\ \mathrm{{m}}"]),
            ("Diferencia centrada en $x$", "Se evalúa $V$ un paso $h = 1\\ \\mu\\mathrm{m}$ a cada lado del punto y se divide por $2h$.",
             [rf"V(x+h) = {sci(float(vxp), 10)}\ \mathrm{{V}},\qquad V(x-h) = {sci(float(vxm), 10)}\ \mathrm{{V}}",
              rf"E_x \approx -\dfrac{{{sci(float(vxp), 10)} - {sci(float(vxm), 10)}}}{{2({sci(h, 1)})}} = {en[0]:.5f}\ \mathrm{{N/C}}"]),
            ("Diferencia centrada en $y$", "Igual, pero desplazando en $y$.",
             [rf"V(y+h) = {sci(float(vyp), 10)}\ \mathrm{{V}},\qquad V(y-h) = {sci(float(vym), 10)}\ \mathrm{{V}}",
              rf"E_y \approx -\dfrac{{{sci(float(vyp), 10)} - {sci(float(vym), 10)}}}{{2({sci(h, 1)})}} = {en[1]:.5f}\ \mathrm{{N/C}}"]),
            ("Comparar con la suma vectorial de Coulomb", "Como control independiente se suman los campos de las dos cargas en ese punto.",
             [rf"\vec r_+ = ({rp[0]:.2f},\ {rp[1]:.2f}),\ r_+ = {dp:.6f}\ \mathrm{{m}};\qquad \vec r_- = ({rm[0]:.2f},\ {rm[1]:.2f}),\ r_- = {dm:.6f}\ \mathrm{{m}}",
              rf"\vec E_+ = {_vec(ep, 5, fx)},\quad \vec E_- = {_vec(em, 5, fx)}\ \mathrm{{N/C}}",
              rf"\vec E = ({ev[0]:.5f},\ {ev[1]:.5f})\ \mathrm{{N/C}}\ \ \text{{frente a }}\ \ ({en[0]:.5f},\ {en[1]:.5f})\ \mathrm{{N/C}}"]),
            ("Líneas de campo: ecuación a integrar", "Una línea de campo satisface $d\\vec r/ds = \\vec E/|\\vec E|$ con $s$ la longitud de arco. Se integra con RK4 y paso $\\Delta s = 50\\ \\mu\\mathrm{m}$, partiendo a $0.2$ mm de la carga positiva en un ángulo dado (aquí $90^\\circ$).",
             [rf"\vec r_0 = ({start[0]:.4f},\ {start[1]:.4f})\ \mathrm{{m}},\qquad \Delta s = {sci(ds, 2)}\ \mathrm{{m}}"]),
            ("Primer paso de RK4", "Se evalúa la dirección unitaria del campo en cuatro puntos: el inicial, dos puntos intermedios y el final estimado, y se combinan con pesos $1,2,2,1$.",
             [rf"\vec k_1 = {_vec(k1, 5, fx)},\quad \vec k_2 = {_vec(k2, 5, fx)},\quad \vec k_3 = {_vec(k3, 5, fx)},\quad \vec k_4 = {_vec(k4, 5, fx)}",
              rf"\vec r_1 = \vec r_0 + \dfrac{{\Delta s}}{{6}}(\vec k_1 + 2\vec k_2 + 2\vec k_3 + \vec k_4) = ({r1[0]:.7f},\ {r1[1]:.7f})\ \mathrm{{m}}"]),
            ("Terminar la línea y resumir", "El proceso se repite hasta que la curva llega a menos de $0.4$ mm de la carga negativa (o se aleja demasiado). Para $90^\\circ$ la curva sale hacia arriba y regresa a la carga negativa.",
             [rf"\text{{pasos}} = {len(pts) - 1},\qquad \text{{llega a }}-q:\ {'sí' if ok else 'no'},\qquad \text{{altura máxima}}\ y = {pts[:, 1].max() * 100:.3f}\ \mathrm{{cm}}"]),
        ],
    )


# =========================================================================== ChargedRing
def _ring() -> dict:
    q, a = e2.RING_Q, e2.RING_R
    za, zn = e2.ring_max_field_position_analytic(), e2.ring_max_field_position_numeric()
    emax = float(e2.ring_axis_field_analytic(za))
    emax_closed = 2 * K_E * q / (3 * np.sqrt(3) * a**2)
    e5, e30 = float(e2.ring_axis_field_analytic(0.05)), float(e2.ring_axis_field_analytic(0.30))
    n5, n30 = e2.ring_axis_field_numeric(0.05), e2.ring_axis_field_numeric(0.30)
    z = 0.05
    dist = float(np.sqrt(z**2 + a**2))
    # Simpson mini N=4 en phi, z=0.05
    phi4 = np.linspace(0, 2 * np.pi, 5)
    dqrad = q / (2 * np.pi)
    src = np.stack([a * np.cos(phi4), a * np.sin(phi4), np.zeros(5)], axis=1)
    r = np.array([0.0, 0.0, z]) - src
    dd = np.linalg.norm(r, axis=1)
    c = K_E * dqrad * r / dd[:, None] ** 3
    h4 = 2 * np.pi / 4
    ez4 = h4 / 3 * (c[0, 2] + c[4, 2] + 4 * (c[1, 2] + c[3, 2]) + 2 * c[2, 2])
    ex4 = h4 / 3 * (c[0, 0] + c[4, 0] + 4 * (c[1, 0] + c[3, 0]) + 2 * c[2, 0])
    # fit parabolico
    zs = np.linspace(0.001, 3 * a, 2000)
    zc = zs[::10]
    ev = np.array([e2.ring_axis_field_numeric(float(zz), n=200) for zz in zc])
    i = int(np.argmax(ev))
    lo, hi = max(i - 3, 0), min(i + 4, len(zc))
    cf = np.polyfit(zc[lo:hi], ev[lo:hi], 2)
    return dict(
        explanation=[
            "Para una distribución continua de carga se divide en elementos infinitesimales $dq$, se trata cada uno como una carga puntual que aporta $d\\vec E = k\\,dq\\,\\hat r/r^2$ y se suman (integran) las contribuciones vectoriales. La dificultad práctica es elegir un elemento cuya contribución sea fácil de escribir y aprovechar la simetría para anular componentes.",
            "En un anillo uniforme, cada elemento tiene un compañero diametralmente opuesto cuyo campo perpendicular al eje es contrario. Las componentes transversales se cancelan por completo y sobreviven solo las axiales. Todos los elementos están a la misma distancia $\\sqrt{z^2+a^2}$ del punto del eje, así que la integral se reduce a multiplicar por la carga total.",
            "El campo axial $E_z = kQz/(z^2+a^2)^{3/2}$ se anula en el centro (las contribuciones se cancelan por simetría), crece al alejarse y a gran distancia decae como $kQ/z^2$, que es el campo de una carga puntual. Por tanto tiene un máximo en algún punto intermedio. Derivar e igualar a cero permite hallarlo.",
            "El orden de magnitud sale de $kQ/a^2$: con 10 nC y 10 cm es de unos $9\\times10^{3}$ N/C; el máximo vale cerca de $0.38$ de ese valor, y ocurre a $0.71$ del radio, más cerca del anillo de lo que sugeriría una intuición de carga puntual.",
        ],
        analytic_steps=[
            ("Elemento de carga y su campo", "Se toma un elemento $dq = Q\\,d\\varphi/2\\pi$ en el anillo. Su campo en el punto $(0,0,z)$ apunta a lo largo del vector que va del elemento al punto, de longitud $r = \\sqrt{z^2+a^2}$.",
             [r"d\vec E = k\,\dfrac{dq}{r^2}\,\hat r,\qquad r^2 = z^2 + a^2"]),
            ("Aprovechar la simetría", "Solo la componente axial $\\cos\\theta = z/r$ sobrevive; las perpendiculares se cancelan por pares opuestos. Como $r$ y $\\cos\\theta$ son constantes, se sacan de la integral.",
             [r"E_z = \oint k\,\dfrac{dq}{r^2}\,\dfrac{z}{r} = \dfrac{kz}{(z^2+a^2)^{3/2}}\oint dq = \dfrac{kQz}{(z^2+a^2)^{3/2}}"]),
            ("Evaluar a $z = 5$ cm", "Se sustituyen $Q = 10$ nC, $a = 10$ cm.",
             [rf"E_z = \dfrac{{({sci(K_E,5)})({sci(q,3)})({z})}}{{(({z})^2+({a})^2)^{{3/2}}}} = \dfrac{{{sci(K_E * q * z, 5)}}}{{({dist:.6f})^3}} = {e5:.2f}\ \mathrm{{N/C}}"]),
            ("Evaluar a $z = 30$ cm", "Mismos datos, más lejos del anillo. Allí el anillo se parece a una carga puntual: $kQ/z^2 = " + f"{K_E * q / 0.3**2:.1f}" + "$ N/C.",
             [rf"E_z = \dfrac{{({sci(K_E,5)})({sci(q,3)})(0.30)}}{{((0.30)^2+({a})^2)^{{3/2}}}} = {e30:.2f}\ \mathrm{{N/C}}"]),
            ("Derivar para encontrar el máximo", "Se deriva respecto de $z$ y se iguala a cero. El numerador se simplifica a $a^2 - 2z^2$.",
             [r"\dfrac{dE_z}{dz} = kQ\,\dfrac{(z^2+a^2) - 3z^2}{(z^2+a^2)^{5/2}} = kQ\,\dfrac{a^2 - 2z^2}{(z^2+a^2)^{5/2}} = 0"]),
            ("Posición y valor del máximo", "La raíz positiva es $z = a/\\sqrt2$. Sustituyendo, $z^2+a^2 = 3a^2/2$.",
             [rf"z_{{max}} = \dfrac{{a}}{{\sqrt2}} = \dfrac{{{a}}}{{1.41421}} = {za * 100:.4f}\ \mathrm{{cm}}",
              rf"E_{{max}} = \dfrac{{2kQ}}{{3\sqrt3\,a^2}} = \dfrac{{2({sci(K_E,5)})({sci(q,3)})}}{{3\sqrt3\,({a})^2}} = {emax_closed:.2f}\ \mathrm{{N/C}}"]),
        ],
        numeric_steps=[
            ("Discretizar el anillo", "El código integra sobre el ángulo $\\varphi\\in[0,2\\pi]$ con la regla de Simpson. Cada punto $\\varphi_j$ es un elemento de carga $dq/d\\varphi = Q/2\\pi$; se calcula la contribución vectorial completa (incluidas $x$ e $y$) y se integra cada componente.",
             [r"\vec E = \int_0^{2\pi} k\,\dfrac{Q}{2\pi}\,\dfrac{\vec r_{\mathrm{obs}} - \vec r_{\mathrm{anillo}}(\varphi)}{|\vec r_{\mathrm{obs}} - \vec r_{\mathrm{anillo}}(\varphi)|^3}\,d\varphi"]),
            ("Versión reducida a mano: $N = 4$", "Con 4 subintervalos, $h = \\pi/2$ y los nodos están en $0,\\ \\pi/2,\\ \\pi,\\ 3\\pi/2,\\ 2\\pi$. Para $z = 5$ cm, cada punto del anillo está a $0.1118$ m del observador y la componente $z$ de la contribución es la misma en los cinco nodos.",
             [rf"\text{{contribuciones}}\ \ (E_z,\ E_x)\text{{ por radián, en N/C: }}"+ r"\quad" + rf"j=0:\ ({c[0,2]:.3f},\ {c[0,0]:.3f}),\ \ j=1:\ ({c[1,2]:.3f},\ {c[1,0]:.3f}),\ \ j=2:\ ({c[2,2]:.3f},\ {c[2,0]:.3f})"]),
            ("Aplicar Simpson", "La regla pesa los nodos con $1,4,2,4,1$ y multiplica por $h/3$. Las contribuciones de $E_x$ valen $+c,0,-c,0,+c$ y la suma ponderada es cero: se comprueba que la componente transversal se cancela.",
             [r"I \approx \dfrac h3\left[f_0 + 4f_1 + 2f_2 + 4f_3 + f_4\right]",
              rf"E_z \approx \dfrac{{\pi/2}}{{3}}\,(12)\,({c[0, 2]:.3f}) = {ez4:.2f}\ \mathrm{{N/C}},\qquad E_x \approx {sci(float(ex4), 2)}\ \mathrm{{N/C}}"]),
            ("El código a escala: $N = 400$", "Se hace lo mismo con 400 subintervalos (401 nodos). El integrando es constante en $E_z$, así que Simpson es exacto salvo redondeo; el código además comprueba con un `assert` que $|E_x|$ y $|E_y|$ sean menores que $10^{-6}|E_z|$.",
             [rf"E_z(5\ \mathrm{{cm}}) = {n5:.4f}\ \mathrm{{N/C}}\ \ (\text{{analítico }}{e5:.4f}),\qquad E_z(30\ \mathrm{{cm}}) = {n30:.4f}\ \mathrm{{N/C}}\ \ (\text{{analítico }}{e30:.4f})",
              rf"\text{{error relativo a 5 cm: }}{sci(abs(n5 - e5) / e5, 2)}"]),
            ("Localizar el máximo: barrido", "Se calcula $E_z$ numérico (con $N=200$) en 200 valores de $z$ equiespaciados entre $1$ mm y $30$ cm (uno de cada diez de una malla de 2000), y se toma el mayor.",
             [rf"z_i = {zc[i]*100:.4f}\ \mathrm{{cm}},\qquad E_i = {ev[i]:.2f}\ \mathrm{{N/C}}\qquad(\text{{índice }}{i}\text{{ de }}{len(zc)})"]),
            ("Refinar con un ajuste parabólico", "Cerca del máximo la curva se parece a una parábola. Se ajusta $E \\approx c_2z^2 + c_1z + c_0$ por mínimos cuadrados a 7 puntos alrededor del mayor y se toma el vértice $z = -c_1/(2c_2)$.",
             [rf"c_2 = {sci(cf[0], 4)},\quad c_1 = {sci(cf[1], 4)},\quad c_0 = {sci(cf[2], 4)}",
              rf"z_{{max}} = -\dfrac{{c_1}}{{2c_2}} = {zn * 100:.4f}\ \mathrm{{cm}}\ \ (\text{{analítico }}{za * 100:.4f}\ \mathrm{{cm}}),\qquad \Delta z = {sci(abs(zn - za) * 100, 2)}\ \mathrm{{cm}}"]),
        ],
    )


# =========================================================================== GaussFlux
def _gauss_flux() -> dict:
    q, rr = e3.Q, 0.3
    phi = e3.flux_sphere_analytic()
    es = K_E * q / rr**2
    area = 4 * np.pi * rr**2
    nth, nph = 200, 400
    dth, dph = np.pi / nth, 2 * np.pi / nph
    th0, ph0 = 0.5 * dth, 0.5 * dph
    n0 = np.array([np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)])
    p0 = rr * n0
    e0 = e3._field(p0, np.zeros(3), q)
    en0 = float(e0 @ n0)
    da0 = rr**2 * np.sin(th0) * dth * dph
    cpos = np.array([0.1, 0.05, 0.0])
    ec = e3._field(p0, cpos, q)
    enc = float(ec @ n0)
    fs = e3.flux_sphere_numeric()
    fsd = e3.flux_sphere_numeric(charge_pos=(0.1, 0.05, 0.0))
    fc = e3.flux_cube_numeric()
    fout = e3.flux_sphere_numeric(radius=0.1, charge_pos=(0.3, 0.0, 0.0))
    side, nc = 0.4, 400
    s0 = (0.5) / nc * side - side / 2
    pc = np.array([s0, s0, side / 2])
    ecube = e3._field(pc, np.zeros(3), q)
    dac = (side / nc) ** 2
    return dict(
        explanation=[
            "El flujo eléctrico mide cuánto campo atraviesa una superficie: es la integral de $\\vec E\\cdot\\hat n\\,dA$, donde $\\hat n$ es la normal exterior. Una superficie perpendicular al campo recibe el flujo máximo; una paralela, ninguno. Si el campo sale de una superficie cerrada, el flujo es positivo; si entra, negativo.",
            "La ley de Gauss afirma que el flujo neto a través de cualquier superficie cerrada es $q_{enc}/\\varepsilon_0$, con $q_{enc}$ la carga encerrada. No importa la forma de la superficie ni dónde esté la carga dentro de ella. El motivo es geométrico: el campo de una carga puntual decrece como $1/r^2$ y el área de una esfera crece como $r^2$, de manera que el flujo por cada ángulo sólido es constante.",
            "Si la carga está fuera de la superficie, cada línea de campo que entra por un lado sale por el otro y el flujo neto es cero, aunque el campo no sea nulo en ningún punto de la superficie. Con la carga descentrada dentro de la esfera, el campo ya no es uniforme sobre la superficie (más intenso del lado cercano), pero el flujo total sigue siendo el mismo.",
            "En la práctica la ley de Gauss sirve para hallar $E$ solo cuando la simetría permite sacar el campo de la integral. Aquí se usa al revés: se calcula el flujo directamente, con superficies cuya integral no es trivial, para ver que el resultado es siempre $q/\\varepsilon_0$.",
        ],
        analytic_steps=[
            ("Enunciar la ley de Gauss", "El flujo total a través de una superficie cerrada depende solo de la carga neta encerrada.",
             [r"\Phi_E = \oint \vec E\cdot d\vec A = \dfrac{q_{enc}}{\varepsilon_0}"]),
            ("Esfera con la carga en el centro: integral directa", "Por simetría, $\\vec E$ es radial y de magnitud constante sobre la esfera, y es paralelo a $d\\vec A$. La integral se reduce a $E\\cdot A$.",
             [rf"E = \dfrac{{kq}}{{R^2}} = \dfrac{{({sci(K_E,5)})({sci(q,3)})}}{{({rr})^2}} = {es:.3f}\ \mathrm{{N/C}},\qquad A = 4\pi R^2 = {area:.5f}\ \mathrm{{m^2}}",
              rf"\Phi = EA = ({es:.3f})({area:.5f}) = {es * area:.3f}\ \mathrm{{N\,m^2/C}}"]),
            ("Comprobar con la ley de Gauss", "Se obtiene el mismo número a partir de $q/\\varepsilon_0$, lo cual confirma que $k = 1/4\\pi\\varepsilon_0$ es consistente con la ley.",
             [rf"\Phi = \dfrac{{q}}{{\varepsilon_0}} = \dfrac{{{sci(q,3)}}}{{{sci(EPS0,5)}}} = {phi:.3f}\ \mathrm{{N\,m^2/C}}"]),
            ("Cubo con la carga en el centro", "No hay que integrar sobre el cubo: por la simetría cúbica, cada una de las seis caras recibe el mismo flujo, por tanto un sexto del total. El resultado es el mismo que en la esfera.",
             [rf"\Phi_{{cara}} = \dfrac{{\Phi}}{{6}} = {phi / 6:.3f}\ \mathrm{{N\,m^2/C}},\qquad \Phi_{{cubo}} = 6\,\Phi_{{cara}} = {phi:.3f}\ \mathrm{{N\,m^2/C}}"]),
            ("Carga descentrada", "Sobre la esfera, $E$ ya no es uniforme ni perpendicular, pero la ley de Gauss no exige simetría: solo que la carga esté dentro. El flujo es el mismo.",
             [rf"\Phi_{{descentrada}} = \dfrac{{q}}{{\varepsilon_0}} = {phi:.3f}\ \mathrm{{N\,m^2/C}}"]),
            ("Carga fuera de la superficie", "Aquí $q_{enc} = 0$. Cada línea de campo que entra en la superficie sale por otro punto, y su aporte neto se cancela.",
             [r"\Phi = \dfrac{0}{\varepsilon_0} = 0"]),
        ],
        numeric_steps=[
            ("Elegir la malla", "El código integra $\\oint \\vec E\\cdot\\hat n\\,dA$ con la regla del punto medio. La esfera se parametriza con $\\theta\\in[0,\\pi]$ y $\\varphi\\in[0,2\\pi]$ con 200 y 400 celdas; el elemento de área es $dA = R^2\\sin\\theta\\,d\\theta\\,d\\varphi$.",
             [rf"\Delta\theta = \dfrac{{\pi}}{{200}} = {dth:.6f},\qquad \Delta\varphi = \dfrac{{2\pi}}{{400}} = {dph:.6f},\qquad 200\times400 = 80\,000\ \text{{celdas}}"]),
            ("Primera celda, carga centrada", "Se evalúa en el centro de la celda: $\\theta_0 = \\Delta\\theta/2$, $\\varphi_0 = \\Delta\\varphi/2$. Se calcula el punto, la normal, el campo, su componente normal y el área.",
             [rf"\hat n_0 = {_vec(n0, 5, fx)},\qquad \vec r_0 = R\hat n_0 = {_vec(p0, 5, fx)}\ \mathrm{{m}}",
              rf"\vec E_0 = {_vec(e0, 3, fx)}\ \mathrm{{N/C}},\qquad \vec E_0\cdot\hat n_0 = {en0:.4f}\ \mathrm{{N/C}}",
              rf"dA_0 = R^2\sin\theta_0\,\Delta\theta\,\Delta\varphi = {sci(da0, 5)}\ \mathrm{{m^2}},\qquad \Delta\Phi_0 = {sci(en0 * da0, 5)}\ \mathrm{{N\,m^2/C}}"]),
            ("Sumar todas las celdas (esfera centrada)", "Con la carga en el centro, $\\vec E\\cdot\\hat n$ es el mismo número en todas las celdas y la suma de las áreas es $4\\pi R^2$.",
             [rf"\Phi_{{esfera}} = \sum_{{i,j}}\Delta\Phi_{{ij}} = {fs:.4f}\ \mathrm{{N\,m^2/C}}\ \ (\text{{analítico }}{phi:.4f}),\qquad \text{{error rel.}} = {sci(abs(fs - phi) / phi, 2)}"]),
            ("Misma celda con la carga descentrada", "Con la carga en $(0.1,\\,0.05,\\,0)$ m, el campo en la misma celda ya no es radial ni de la misma magnitud; la componente normal cambia, pero la suma sobre toda la esfera se compensa.",
             [rf"\vec E_0 = {_vec(ec, 3, fx)}\ \mathrm{{N/C}},\qquad \vec E_0\cdot\hat n_0 = {enc:.4f}\ \mathrm{{N/C}}\ \ (\text{{antes: }}{en0:.4f})",
              rf"\Phi = {fsd:.4f}\ \mathrm{{N\,m^2/C}},\qquad \text{{error rel.}} = {sci(abs(fsd - phi) / phi, 2)}"]),
            ("Cubo: seis caras de $400\\times400$", "En cada cara se toma una malla de $400\\times400$ celdas de lado $0.4/400 = 1$ mm. Para la cara $z = +0.2$ m, la primera celda tiene su centro en $(-0.1995,\\,-0.1995,\\,0.2)$ m y solo cuenta la componente $E_z$ (normal exterior $+\\hat z$).",
             [rf"\vec E = {_vec(ecube, 3, fx)}\ \mathrm{{N/C}},\qquad dA = ({side / nc})^2 = {sci(dac, 3)}\ \mathrm{{m^2}},\qquad \Delta\Phi = E_z\,dA = {sci(float(ecube[2]) * dac, 4)}\ \mathrm{{N\,m^2/C}}",
              rf"\Phi_{{cubo}} = \sum_{{6\ \mathrm{{caras}}}}\sum_{{ij}} E_n\,dA = {fc:.4f}\ \mathrm{{N\,m^2/C}},\qquad \text{{error rel.}} = {sci(abs(fc - phi) / phi, 2)}"]),
            ("Carga fuera de la superficie", "Se usa una esfera de radio $0.1$ m con la carga a $0.3$ m del centro. Las celdas cercanas a la carga contribuyen con valores negativos grandes (el campo entra) y las del lado opuesto con positivos (sale); la suma deja un residuo de redondeo.",
             [rf"\Phi_{{fuera}} = {sci(fout, 3)}\ \mathrm{{N\,m^2/C}}\ \approx\ 0\quad\left(\text{{frente al valor típico }}{phi:.1f}\right)"]),
        ],
    )


# =========================================================================== SphereFieldProfile
def _sphere() -> dict:
    q, big_r = e3.Q_SPH, e3.R_SPH
    n = 200_000
    rng = np.random.default_rng(17)
    v = rng.normal(size=(n, 3))
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / nv
    u = rng.uniform(size=(n, 1))
    rad = big_r * u ** (1 / 3)
    src = v * rad
    rob = 0.05
    obs = np.array([rob, 0.0, 0.0])
    dd = obs - src
    dist = np.linalg.norm(dd, axis=1, keepdims=True)
    ex = (K_E * (q / n) * dd / dist**3)[:, 0]
    mc = {r: e3.sphere_field_monte_carlo(r, n=n) for r in (0.05, 0.10, 0.20)}
    an = {r: float(e3.sphere_field_analytic(r)) for r in (0.05, 0.10, 0.20)}
    conv = [(m, e3.sphere_field_monte_carlo(0.05, n=m)) for m in (2000, 20000, 200000)]
    cum10 = float(ex[:10].sum())
    return dict(
        explanation=[
            "Para una distribución con simetría esférica, el campo es radial y depende solo de la distancia $r$ al centro. Se elige como superficie gaussiana una esfera concéntrica de radio $r$: sobre ella $E$ es constante y paralelo a $d\\vec A$, de modo que el flujo vale $E\\,4\\pi r^2$. Basta entonces conocer cuánta carga queda dentro.",
            "Fuera de la esfera ($r>R$) la carga encerrada es toda la carga $Q$ y el campo es idéntico al de una carga puntual en el centro: $kQ/r^2$. Dentro ($r<R$) solo está encerrada la fracción $r^3/R^3$ de la carga, porque la densidad es uniforme y los volúmenes escalan como $r^3$; el campo crece linealmente, $kQr/R^3$.",
            "El campo es continuo en la superficie: ambas expresiones valen $kQ/R^2$ en $r=R$, y allí está el máximo. Esto es característico de una carga volumétrica en un aislante; en una superficie con densidad superficial (como un conductor) el campo sí daría un salto.",
            "El método numérico es un experimento independiente: se reparte la carga en muchas cargas puntuales colocadas al azar con densidad uniforme y se suma el campo coulombiano de todas. Es un cálculo de Monte Carlo, cuyo error disminuye como $1/\\sqrt N$ y que no usa la ley de Gauss en ningún momento.",
        ],
        analytic_steps=[
            ("Superficie gaussiana y simetría", "Por simetría esférica, $\\vec E = E(r)\\hat r$. Con una esfera gaussiana de radio $r$ el flujo es $E(r)\\,4\\pi r^2$ y la ley de Gauss queda en una ecuación para $E$.",
             [r"E(r)\,4\pi r^2 = \dfrac{q_{enc}(r)}{\varepsilon_0}"]),
            ("Fuera de la esfera", "Para $r\\geq R$, $q_{enc} = Q$.",
             [r"E_{out} = \dfrac{Q}{4\pi\varepsilon_0 r^2} = \dfrac{kQ}{r^2}"]),
            ("Dentro de la esfera", "Con densidad uniforme $\\rho = Q/(\\tfrac43\\pi R^3)$, la carga encerrada es $\\rho\\tfrac43\\pi r^3 = Q\\,r^3/R^3$.",
             [r"E_{in}\,4\pi r^2 = \dfrac{Q}{\varepsilon_0}\dfrac{r^3}{R^3}\ \Rightarrow\ E_{in} = \dfrac{kQ\,r}{R^3}"]),
            ("Evaluar a $r = 5$ cm (dentro)", "Con $Q = 5$ nC y $R = 10$ cm.",
             [rf"E = \dfrac{{({sci(K_E,5)})({sci(q,3)})(0.05)}}{{(0.10)^3}} = {an[0.05]:.3f}\ \mathrm{{N/C}}"]),
            ("Evaluar en la superficie", "Aquí ambas fórmulas deben coincidir; es el valor máximo.",
             [rf"E(R) = \dfrac{{kQ}}{{R^2}} = \dfrac{{({sci(K_E,5)})({sci(q,3)})}}{{(0.10)^2}} = {an[0.10]:.3f}\ \mathrm{{N/C}}"]),
            ("Evaluar a $r = 20$ cm (fuera)", "Fuera, como si toda la carga estuviera en el centro.",
             [rf"E = \dfrac{{({sci(K_E,5)})({sci(q,3)})}}{{(0.20)^2}} = {an[0.20]:.3f}\ \mathrm{{N/C}}"]),
        ],
        numeric_steps=[
            ("Muestrear la esfera", "Se generan $N = 200\\,000$ puntos distribuidos uniformemente en el volumen. La dirección se toma de un vector gaussiano normalizado (uniforme en la esfera unidad) y el radio como $R\\,u^{1/3}$ con $u$ uniforme en $[0,1]$; la raíz cúbica compensa que el volumen crece como $r^3$.",
             [rf"N = {n},\qquad \text{{semilla}} = 17,\qquad \vec r_i = R\,u_i^{{1/3}}\,\hat n_i,\qquad q_i = \dfrac{{Q}}{{N}} = {sci(q / n, 4)}\ \mathrm{{C}}"]),
            ("Primera muestra", "Se muestra la primera carga puntual generada y su aporte al campo en el punto de observación $(0.05,\\,0,\\,0)$ m.",
             [rf"\hat n_1 = {_vec(v[0], 5, fx)},\quad u_1 = {u[0, 0]:.5f},\quad r_1 = R u_1^{{1/3}} = {rad[0, 0]:.5f}\ \mathrm{{m}}",
              rf"\vec r_1 = {_vec(src[0], 5, fx)}\ \mathrm{{m}},\qquad \vec d_1 = \vec r_{{obs}} - \vec r_1 = {_vec(dd[0], 5, fx)}\ \mathrm{{m}},\qquad |\vec d_1| = {dist[0, 0]:.5f}\ \mathrm{{m}}",
              rf"E_{{x,1}} = k\,\dfrac{{q}}{{N}}\,\dfrac{{d_x}}{{|\vec d|^3}} = {sci(float(ex[0]), 4)}\ \mathrm{{N/C}}"]),
            ("Segunda y tercera muestras y suma parcial", "Se repite con las siguientes cargas. Los aportes individuales varían mucho (una carga muy próxima al punto de observación contribuye más), y el campo total es la suma de todos.",
             [rf"E_{{x,2}} = {sci(float(ex[1]), 4)},\quad E_{{x,3}} = {sci(float(ex[2]), 4)}\ \mathrm{{N/C}}",
              rf"\sum_{{i=1}}^{{10}} E_{{x,i}} = {sci(cum10, 4)}\ \mathrm{{N/C}}"]),
            ("Estimar el campo", "El campo es la suma de las $N$ contribuciones, $E_x=\\sum_i E_{x,i}$. Cada contribución individual es una variable aleatoria $X_i = N\\,E_{x,i}$, así que $E_x$ es la media de $X_i$ y su incertidumbre es $s/\\sqrt N$.",
             [rf"E_x = \sum_{{i=1}}^{{N}} E_{{x,i}} = {mc[0.05][0]:.3f}\ \mathrm{{N/C}},\qquad \sigma_E = \dfrac{{s}}{{\sqrt N}} = {mc[0.05][1]:.3f}\ \mathrm{{N/C}}",
              rf"\text{{analítico }}{an[0.05]:.3f}\ \mathrm{{N/C}},\qquad \Delta E = {abs(mc[0.05][0] - an[0.05]):.3f}\ \mathrm{{N/C}}"]),
            ("Convergencia con $N$", "El error estadístico baja como $1/\\sqrt N$: multiplicar $N$ por 10 reduce la incertidumbre aproximadamente $\\sqrt{10}\\approx3.2$ veces (punto $r = 5$ cm).",
             [r"\begin{array}{r|cc}N & E_x\ (\mathrm{N/C}) & \sigma_E\ (\mathrm{N/C})\\\hline " + r"\\ ".join(f"{m} & {val[0]:.2f} & {val[1]:.2f}" for m, val in conv) + r"\end{array}"]),
            ("Los tres puntos del perfil", "Se repite el procedimiento (misma semilla) para los tres radios. Las diferencias frente al valor analítico son compatibles con la incertidumbre estadística.",
             [r"\begin{array}{c|ccc}r\ (\mathrm{m}) & E_{MC}\ (\mathrm{N/C}) & \pm\sigma\ (\mathrm{N/C}) & E_{an}\ (\mathrm{N/C})\\\hline " + r"\\ ".join(
                 f"{r:.2f} & {mc[r][0]:.2f} & {mc[r][1]:.2f} & {an[r]:.2f}" for r in (0.05, 0.10, 0.20)) + r"\end{array}"]),
        ],
    )


# =========================================================================== DiskPotential
def _disk() -> dict:
    sg, rd, z = e4.SIGMA, e4.R_DISK, 0.05
    pref = sg / (2 * EPS0)
    sq = float(np.sqrt(z**2 + rd**2))
    van, vnu = float(e4.disk_potential_analytic(z)), e4.disk_potential_numeric(z)
    ean = float(e4.disk_field_analytic(z))
    vf = lambda zz: float(e4.disk_potential_analytic(zz))  # noqa: E731
    h = 1e-6
    enu = e4.field_from_potential_numeric(vf, z)
    # Simpson mini N=4
    s4 = np.linspace(0, rd, 5)
    f4 = K_E * sg * 2 * np.pi * s4 / np.sqrt(s4**2 + z**2)
    h4 = rd / 4
    v4 = h4 / 3 * (f4[0] + f4[4] + 4 * (f4[1] + f4[3]) + 2 * f4[2])
    # N=400 first terms
    n = 400
    s = np.linspace(0, rd, n + 1)
    hh = rd / n
    f = K_E * sg * 2 * np.pi * s / np.sqrt(s**2 + z**2)
    return dict(
        explanation=[
            "El potencial eléctrico $V$ es la energía potencial por unidad de carga. Se mide en voltios y es un escalar, lo que lo hace mucho más cómodo que el campo vectorial: para una distribución de carga basta sumar (o integrar) $k\\,dq/r$ sin preocuparse por componentes ni direcciones. Después el campo se recupera con $\\vec E = -\\nabla V$.",
            "Un disco uniformemente cargado se puede construir como una pila de anillos concéntricos de radio $s$ y ancho $ds$. Todos los puntos de un anillo están a la misma distancia $\\sqrt{s^2+z^2}$ del punto del eje, así que cada anillo aporta $k\\,dq/\\sqrt{s^2+z^2}$ con $dq = \\sigma\\,2\\pi s\\,ds$. Integrar de $0$ a $R$ da el potencial total.",
            "El resultado, $V(z) = \\tfrac{\\sigma}{2\\varepsilon_0}(\\sqrt{z^2+R^2}-z)$, decrece al alejarse. Derivarlo con signo menos da el campo, que en el centro del disco vale $\\sigma/2\\varepsilon_0$ y, muy cerca de él, es el de un plano infinito. A gran distancia $E\\to kQ/z^2$ y $V\\to kQ/z$ con $Q = \\sigma\\pi R^2$.",
            "El campo se obtiene de $V$ y no al revés porque derivar es más fácil que integrar vectorialmente: así se evita tener que proyectar componentes sobre el eje. La derivada numérica solo funciona bien con un paso moderado, porque restar valores casi iguales pierde cifras significativas.",
        ],
        analytic_steps=[
            ("Elemento de carga: un anillo", "Un anillo de radio $s$ y ancho $ds$ tiene área $2\\pi s\\,ds$ y carga $dq = \\sigma\\,2\\pi s\\,ds$. Todos sus puntos están a la distancia $\\sqrt{s^2+z^2}$ del punto del eje.",
             [r"dV = \dfrac{k\,dq}{\sqrt{s^2+z^2}} = \dfrac{k\sigma\,2\pi s\,ds}{\sqrt{s^2+z^2}}"]),
            ("Integrar sobre el disco", "Con $k = 1/4\\pi\\varepsilon_0$ el factor se simplifica. La integral es inmediata con $u = s^2+z^2$.",
             [r"V(z) = \dfrac{\sigma}{2\varepsilon_0}\int_0^R\dfrac{s\,ds}{\sqrt{s^2+z^2}} = \dfrac{\sigma}{2\varepsilon_0}\Big[\sqrt{s^2+z^2}\Big]_0^R = \dfrac{\sigma}{2\varepsilon_0}\left(\sqrt{z^2+R^2}-z\right)"]),
            ("Calcular el prefactor", "Con $\\sigma = 2\\ \\mu\\mathrm{C/m^2}$.",
             [rf"\dfrac{{\sigma}}{{2\varepsilon_0}} = \dfrac{{{sci(sg,3)}}}{{2({sci(EPS0,5)})}} = {sci(pref, 5)}\ \mathrm{{V/m}}"]),
            ("Evaluar el potencial a 5 cm", "Se sustituye $z = 0.05$ m y $R = 0.10$ m.",
             [rf"\sqrt{{z^2+R^2}} = \sqrt{{({z})^2+({rd})^2}} = {sq:.6f}\ \mathrm{{m}}",
              rf"V = ({sci(pref, 5)})({sq:.6f} - {z}) = {van:.3f}\ \mathrm{{V}} = {van / 1000:.4f}\ \mathrm{{kV}}"]),
            ("Obtener el campo derivando", "El campo es menos el gradiente. Sobre el eje solo hay componente $z$ y $d\\sqrt{z^2+R^2}/dz = z/\\sqrt{z^2+R^2}$.",
             [r"E_z = -\dfrac{dV}{dz} = \dfrac{\sigma}{2\varepsilon_0}\left(1 - \dfrac{z}{\sqrt{z^2+R^2}}\right)"]),
            ("Evaluar el campo a 5 cm", "El resultado es positivo: el campo apunta alejándose del disco (carga positiva).",
             [rf"E_z = ({sci(pref, 5)})\left(1 - \dfrac{{{z}}}{{{sq:.6f}}}\right) = ({sci(pref, 5)})({1 - z / sq:.6f}) = {ean:.3f}\ \mathrm{{V/m}} = {ean / 1000:.4f}\ \mathrm{{kV/m}}"]),
            ("Comprobaciones de límites", "En $z\\to 0$ el campo tiende al del plano infinito, $\\sigma/2\\varepsilon_0$. A gran distancia debe parecerse al de una carga puntual.",
             [rf"E_z(0) = {pref:.1f}\ \mathrm{{V/m}},\qquad Q = \sigma\pi R^2 = {sci(sg * np.pi * rd**2, 4)}\ \mathrm{{C}}"]),
        ],
        numeric_steps=[
            ("Plantear la integral numérica", "El código no usa la fórmula cerrada: integra $V = \\int_0^R k\\sigma\\,2\\pi s/\\sqrt{s^2+z^2}\\,ds$ con la regla de Simpson sobre $N=400$ subintervalos.",
             [rf"f(s) = \dfrac{{k\sigma\,2\pi s}}{{\sqrt{{s^2+z^2}}}},\qquad h = \dfrac{{R}}{{N}} = \dfrac{{{rd}}}{{{n}}} = {sci(hh, 3)}\ \mathrm{{m}}"]),
            ("Regla de Simpson", "Alterna pesos $4$ y $2$ en los nodos interiores y pesa $1$ los extremos. Es exacta para polinomios hasta grado 3 y su error decrece como $h^4$.",
             [r"I \approx \dfrac h3\Big[f_0 + 4(f_1+f_3+\cdots+f_{N-1}) + 2(f_2+f_4+\cdots+f_{N-2}) + f_N\Big]"]),
            ("Versión reducida a mano: $N = 4$", "Con 4 subintervalos, $h = 0.025$ m y nodos en $0,\\ 0.025,\\ 0.05,\\ 0.075,\\ 0.10$ m.",
             [r"\begin{array}{c|ccccc}j & 0 & 1 & 2 & 3 & 4\\\hline s_j\ (\mathrm{m}) & " + " & ".join(f"{x:.3f}" for x in s4) + r"\\ f_j\ (\mathrm{V/m}) & " + " & ".join(sci(float(x), 4) for x in f4) + r"\end{array}",
              rf"V_4 = \dfrac{{{h4}}}{{3}}\left[{sci(f4[0], 3)} + {sci(f4[4], 4)} + 4({sci(f4[1], 4)} + {sci(f4[3], 4)}) + 2({sci(f4[2], 4)})\right] = {v4:.3f}\ \mathrm{{V}}"]),
            ("La misma regla con $N = 400$", "Los primeros nodos son $s_1 = h = 0.25$ mm con peso 4 y $s_2 = 0.50$ mm con peso 2; se continúa hasta $s_{400} = R$.",
             [rf"f_1 = {sci(f[1], 5)},\quad f_2 = {sci(f[2], 5)},\quad f_{{400}} = {sci(f[-1], 5)}\ \mathrm{{V/m}}",
              rf"V_{{400}} = {vnu:.5f}\ \mathrm{{V}}\ \ (\text{{analítico }}{van:.5f}),\qquad \text{{error rel.}} = {sci(abs(vnu - van) / van, 2)}"]),
            ("Convergencia respecto de $N$", "El error de Simpson baja muy rápido al refinar la partición, porque el integrando es suave.",
             [r"\begin{array}{c|c}N & \text{error rel. de }V\\\hline " + r"\\ ".join(f"{m} & {sci(abs(e4.disk_potential_numeric(z, n=m) - van) / van, 2)}" for m in (4, 20, 100, 400)) + r"\end{array}"]),
            ("Campo por derivación numérica", "El campo se obtiene de $E_z = -dV/dz$ con diferencia centrada y $h = 1\\ \\mu\\mathrm{m}$. En el código la función derivada es la fórmula analítica de $V$, de modo que solo se prueba la derivada.",
             [rf"V(z+h) = {sci(vf(z + h), 12)}\ \mathrm{{V}},\qquad V(z-h) = {sci(vf(z - h), 12)}\ \mathrm{{V}}",
              rf"E_z \approx -\dfrac{{V(z+h) - V(z-h)}}{{2h}} = {enu:.3f}\ \mathrm{{V/m}}\ \ (\text{{analítico }}{ean:.3f}),\qquad \text{{error rel.}} = {sci(abs(enu - ean) / ean, 2)}"]),
        ],
    )


# =========================================================================== EquipotentialPair
def _pair() -> dict:
    qa, qb, d = e4.QA, e4.QB, e4.D
    u = e4.interaction_energy()
    extent, n = 0.5, 801
    xs = np.linspace(-extent, extent, n)
    dx = float(xs[1] - xs[0])
    i, j = 600, 480
    x0, y0 = float(xs[i]), float(xs[j])
    vxp, vxm = float(e4.potential_pair(xs[i + 1], xs[j])), float(e4.potential_pair(xs[i - 1], xs[j]))
    vyp, vym = float(e4.potential_pair(xs[i], xs[j + 1])), float(e4.potential_pair(xs[i], xs[j - 1]))
    exn, eyn = -(vxp - vxm) / (2 * dx), -(vyp - vym) / (2 * dx)
    exa, eya = e4.field_pair_analytic(x0, y0)
    exa, eya = float(exa), float(eya)
    err_max = e4.field_grid_error()
    xg, gx, gy = e4.field_grid_numeric()
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    eax, eay = e4.field_pair_analytic(X, Y)
    far = (np.hypot(X + d / 2, Y) > 0.12) & (np.hypot(X - d / 2, Y) > 0.12)
    rel = np.hypot(gx - eax, gy - eay) / np.hypot(eax, eay)
    rel_f = np.where(far, rel, 0.0)
    k = np.unravel_index(rel_f.argmax(), rel_f.shape)
    inner = rel_f[2:-2, 2:-2]
    err_in = float(inner.max())
    err_401 = e4.field_grid_error(n=401)
    ra, rb = np.hypot(x0 + d / 2, y0), np.hypot(x0 - d / 2, y0)
    v0 = float(e4.potential_pair(x0, y0))
    return dict(
        explanation=[
            "La diferencia de potencial entre dos puntos es el trabajo por unidad de carga necesario para llevar una carga de prueba de uno al otro. Para cargas puntuales, el potencial es la suma algebraica $V = k\\sum q_i/r_i$ (tomando $V = 0$ en el infinito). Una equipotencial es el conjunto de puntos con el mismo $V$: moverse sobre ella no cuesta trabajo.",
            "El campo eléctrico siempre es perpendicular a las equipotenciales y apunta hacia donde $V$ decrece más deprisa; su magnitud es la rapidez de ese cambio, $\\vec E = -\\nabla V$. Donde las equipotenciales están muy juntas el campo es intenso. Para un par de cargas iguales y opuestas, la mediatriz es una equipotencial de valor cero, porque cada punto está a la misma distancia de ambas y los aportes se cancelan.",
            "La energía de interacción del par, $U = kq_aq_b/d$, es el trabajo necesario para acercar las cargas desde el infinito hasta separarlas una distancia $d$. Es negativa cuando las cargas son opuestas: el sistema ligado tiene menos energía que las cargas separadas, y ese déficit es la energía de enlace.",
            "Calcular el campo derivando el potencial sobre una malla es un método muy general, porque funciona aunque no se sepa la expresión del campo, pero tiene límites: las diferencias finitas son poco precisas cerca de las cargas, donde $V$ varía bruscamente en una distancia comparable al paso de la malla, y en los bordes, donde solo se puede usar una diferencia unilateral.",
        ],
        analytic_steps=[
            ("Potencial de dos cargas puntuales", "El potencial es un escalar y se suma algebraicamente. Se colocan $q_a = +4$ nC en $x=-0.10$ m y $q_b = -4$ nC en $x=+0.10$ m.",
             [r"V(x,y) = k\left(\dfrac{q_a}{r_a} + \dfrac{q_b}{r_b}\right),\qquad r_a = \sqrt{(x+d/2)^2+y^2},\quad r_b = \sqrt{(x-d/2)^2+y^2}"]),
            ("La mediatriz es equipotencial", "Para $x=0$ se cumple $r_a=r_b$ y como $q_b=-q_a$ los dos términos se cancelan, sea cual sea $y$. Se evalúa en $(0,\\,0.1)$ m como ejemplo.",
             [rf"V(0,\,0.1) = ({sci(K_E,5)})\left(\dfrac{{{sci(qa,2)}}}{{{np.hypot(d / 2, 0.1):.5f}}} + \dfrac{{{sci(qb,2)}}}{{{np.hypot(d / 2, 0.1):.5f}}}\right) = {float(e4.potential_pair(0.0, 0.1)):.1f}\ \mathrm{{V}}"]),
            ("Campo como gradiente", "Derivando el potencial se obtiene el campo; coincide con la suma vectorial de Coulomb de cada carga. El gradiente es perpendicular a las curvas $V=\\text{const}$.",
             [r"\vec E = -\nabla V = k\sum_i q_i\,\dfrac{\vec r - \vec r_i}{|\vec r - \vec r_i|^3}"]),
            ("Energía de interacción", "Es el trabajo de formar el par: se trae $q_b$ desde el infinito hasta la distancia $d$ de $q_a$, sometida al potencial $kq_a/d$ de $q_a$.",
             [r"U = q_b\,V_a(d) = \dfrac{k\,q_aq_b}{d}",
              rf"U = \dfrac{{({sci(K_E,5)})({sci(qa,2)})({sci(qb,2)})}}{{{d}}} = {sci(u, 5)}\ \mathrm{{J}} = {u * 1e6:.4f}\ \mu\mathrm{{J}}"]),
            ("Interpretar el signo", "Un valor negativo significa que hay que entregar $|U|$ de energía para separar las cargas hasta el infinito; es la energía de enlace del par.",
             [rf"W_{{\mathrm{{separar}}}} = -U = {-u * 1e6:.4f}\ \mu\mathrm{{J}}"]),
            ("Valores de ejemplo para la malla", "El punto $(0.25,\\,0.10)$ m, de la malla numérica, tiene distancias y potencial como sigue; servirá para comparar luego el gradiente numérico.",
             [rf"r_a = {ra:.6f}\ \mathrm{{m}},\quad r_b = {rb:.6f}\ \mathrm{{m}},\quad V = {v0:.5f}\ \mathrm{{V}}",
              rf"\vec E_{{an}} = ({exa:.5f},\ {eya:.5f})\ \mathrm{{V/m}}"]),
        ],
        numeric_steps=[
            ("Construir la malla", "Se toma el cuadrado $[-0.5,\\,0.5]$ m con $801\\times801$ nodos, de modo que el paso es $\\Delta = 1/800 = 1.25$ mm. En cada nodo se evalúa $V(x_i,y_j)$ (primero $x$ y luego $y$ en el índice).",
             [rf"\Delta = \dfrac{{1.0}}{{800}} = {dx * 1000:.3f}\ \mathrm{{mm}},\qquad 801^2 = {801**2}\ \text{{nodos}}"]),
            ("Diferencia centrada en el nodo $(600, 480)$", "El nodo $i = 600$, $j = 480$ corresponde a $(0.25,\\,0.10)$ m. La derivada en $x$ usa los vecinos $i\\pm1$ y la de $y$ los vecinos $j\\pm1$; así lo hace `np.gradient` en el interior.",
             [rf"V_{{i+1,j}} = {vxp:.6f},\quad V_{{i-1,j}} = {vxm:.6f}\ \mathrm{{V}}",
              rf"E_x = -\dfrac{{V_{{i+1,j}} - V_{{i-1,j}}}}{{2\Delta}} = -\dfrac{{{vxp:.6f} - {vxm:.6f}}}{{{2 * dx:.5f}}} = {exn:.5f}\ \mathrm{{V/m}}"]),
            ("Componente $y$ del mismo nodo", "Igual para la dirección $y$.",
             [rf"V_{{i,j+1}} = {vyp:.6f},\quad V_{{i,j-1}} = {vym:.6f}\ \mathrm{{V}}",
              rf"E_y = -\dfrac{{{vyp:.6f} - {vym:.6f}}}{{{2 * dx:.5f}}} = {eyn:.5f}\ \mathrm{{V/m}}"]),
            ("Comparar con el campo exacto", "En este punto, lejos de las cargas, el gradiente numérico concuerda con la suma de Coulomb con un error relativo muy pequeño.",
             [rf"\vec E_{{an}} = ({exa:.5f},\ {eya:.5f})\ \mathrm{{V/m}},\qquad \dfrac{{|\vec E_{{num}} - \vec E_{{an}}|}}{{|\vec E_{{an}}|}} = {sci(float(np.hypot(exn - exa, eyn - eya) / np.hypot(exa, eya)), 2)}"]),
            ("Error en toda la malla", "Se calcula el error relativo $|\\vec E_{num}-\\vec E_{an}|/|\\vec E_{an}|$ en cada nodo y se toma el máximo, excluyendo los nodos a menos de $12$ cm de cualquiera de las cargas (el paso es demasiado grueso para resolver el potencial cerca de ellas).",
             [rf"\varepsilon_{{max}} = {err_max:.5f}\ \ \text{{en el nodo }}({xg[k[0]]:.3f},\ {xg[k[1]]:.3f})\ \mathrm{{m}}\ \text{{(borde de la malla)}}",
              rf"\text{{En el interior (sin las dos filas de borde): }}\varepsilon_{{max}} = {sci(err_in, 3)}"]),
            ("Por qué el máximo está en el borde", "En el borde `np.gradient` usa una diferencia unilateral de primer orden, con error $O(\\Delta)$, mientras que en el interior la diferencia centrada es de segundo orden, $O(\\Delta^2)$. Duplicar el paso ($401\\times401$ nodos) duplica aproximadamente el error del borde.",
             [rf"\varepsilon_{{max}}(401\times401) = {err_401:.5f},\qquad \varepsilon_{{max}}(801\times801) = {err_max:.5f}"]),
            ("Equipotenciales y energía", "Las equipotenciales se trazan como curvas de nivel de la malla de $V$; la mediatriz $x=0$ resulta ser el nivel $V = 0$ (con el código se obtiene un valor exactamente nulo en $(0,0.1)$), y la energía del par es la misma que se calculó analíticamente.",
             [rf"V(0,\,0.1) = {float(e4.potential_pair(0.0, 0.1)):.6f}\ \mathrm{{V}},\qquad U = {u * 1e6:.4f}\ \mu\mathrm{{J}}"]),
        ],
    )


DETAIL = {
    "CoulombForces": _coulomb(),
    "EquilibriumPoint": _equilibrium(),
    "DipoleField": _dipole(),
    "ChargedRing": _ring(),
    "GaussFlux": _gauss_flux(),
    "SphereFieldProfile": _sphere(),
    "DiskPotential": _disk(),
    "EquipotentialPair": _pair(),
}
