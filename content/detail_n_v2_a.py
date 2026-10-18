"""Explicacion del METODO NUMERICO, Volumen 2, capitulos 1 a 4 (Coulomb, campo, Gauss, potencial).

Cada entrada aporta numeric_intro, numeric_code y numeric_converge. Todas las cifras de los estudios de
error se calculan aqui mismo con las funciones de `physics/` (o replicando su algoritmo con otro parametro).
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

EPS = float(np.finfo(float).eps)


def _arr(cols: str, head: list[str], rows: list[list[str]]) -> str:
    """Tabla TeX (array) con encabezado y filas ya formateadas."""
    body = r" \\ ".join(" & ".join(r) for r in rows)
    return r"\begin{array}{" + cols + r"}\hline " + " & ".join(head) + r" \\ \hline " + body + r" \\ \hline \end{array}"


def _slope(x1: float, x2: float, e1_: float, e2_: float) -> float:
    """Pendiente log-log local entre dos puntos (x, error)."""
    return float(np.log(e1_ / e2_) / np.log(x1 / x2))


def _bisect(fun_lo_moves, lo: float, hi: float, k: int) -> float:
    """Biseccion con k iteraciones; fun_lo_moves(mid) True => la raiz esta a la derecha de mid."""
    for _ in range(k):
        mid = 0.5 * (lo + hi)
        if fun_lo_moves(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =========================================================================== CoulombForces
def _coulomb() -> dict:
    fa = e1.force_analytic()
    nf = float(np.linalg.norm(fa))
    u0 = e1.potential_energy(e1.POS)

    def rel(h: float) -> float:
        return float(np.linalg.norm(e1.force_numeric(h=h) - fa) / nf)

    hs = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    errs = [rel(h) for h in hs]
    s12, s23 = _slope(hs[0], hs[1], errs[0], errs[1]), _slope(hs[1], hs[2], errs[1], errs[2])
    c_trunc = errs[1] / hs[1] ** 2  # coeficiente de truncamiento C en err ~ C h^2
    c_round = EPS * abs(u0) / nf  # coeficiente de redondeo R en err ~ R / h  (U ~ |U| con error relativo eps)
    h_opt = (c_round / (2 * c_trunc)) ** (1 / 3)
    # derivada hacia adelante (orden 1) con el mismo h, para comparar
    h = 1e-6
    ffw = np.zeros(2)
    for ax in range(2):
        p = e1.POS.copy()
        p[2, ax] += h
        ffw[ax] = -(e1.potential_energy(p) - u0) / h
    err_fw = float(np.linalg.norm(ffw - fa) / nf)
    err_c = rel(h)
    res3 = e1.newton_third_law_residual()
    scale = max(float(np.linalg.norm(e1.force_analytic(i))) for i in range(3))

    intro = [
        "La fuerza sobre una carga se puede obtener de dos maneras: sumando vectorialmente las fuerzas de Coulomb (la solución analítica) o a partir de la energía, porque las fuerzas electrostáticas son conservativas y $\\vec F=-\\nabla U$. Aquí se usa la segunda, que no necesita descomponer nada en vectores unitarios: basta saber calcular un escalar, la energía potencial total $U$, y ver cómo cambia cuando se mueve la carga.",
        "El gradiente se aproxima con una diferencia centrada: $\\dfrac{\\partial U}{\\partial x}\\approx\\dfrac{U(x+h)-U(x-h)}{2h}$, y lo mismo en $y$. Al desarrollar $U$ en serie de Taylor, los términos pares se cancelan y queda un error de truncamiento $-\\dfrac{h^2}{6}\\,U'''(x)$, es decir, el método es de orden $O(h^2)$: al dividir $h$ entre 10 el error baja 100 veces. La diferencia hacia adelante, en cambio, es solo $O(h)$.",
        f"Reducir $h$ no mejora el resultado indefinidamente. Cada evaluación de $U$ tiene un error de redondeo relativo del orden de la precisión de máquina ($\\varepsilon\\approx 2.2\\times10^{{-16}}$) y al restar dos valores casi iguales y dividir entre $2h$ ese error crece como $\\varepsilon\\,|U|/h$. La suma de truncamiento (decrece con $h$) y redondeo (crece al achicar $h$) tiene un mínimo cerca de $h^*\\sim$ {h_opt * 1e6:.1f} µm; el código usa $h=1\\ \\mu\\mathrm{{m}}$, que queda en esa zona y da un error relativo de {errs[3]:.1e} en la fuerza.",
        "El método es independiente del analítico porque nunca usa la expresión $kq_iq_j\\,\\vec r/r^3$ de la fuerza: solo evalúa la energía $k\\,q_iq_j/r_{ij}$ de cada par, que es un escalar. Que ambos caminos coincidan confirma de paso que el campo electrostático es conservativo. Además, la suma de las fuerzas sobre las tres cargas debe anularse por la tercera ley de Newton, lo cual da una segunda verificación sin ningún parámetro libre.",
    ]
    code = """def U(pos, q):                           # energia potencial total del sistema
    u = 0.0
    for a in range(3):
        for b in range(a + 1, 3):         # cada par una sola vez
            u += K * q[a] * q[b] / norm(pos[a] - pos[b])   # k q_a q_b / r_ab
    return u

def fuerza_numerica(pos, q, i=2, h=1e-6):
    F = zeros(2)
    for eje in (0, 1):                    # derivada parcial en x y luego en y
        mas, menos = pos.copy(), pos.copy()
        mas[i, eje] += h                  # mueve q_i a +h sobre el eje
        menos[i, eje] -= h                # y a -h
        F[eje] = -(U(mas, q) - U(menos, q)) / (2 * h)   # F = -dU/dx_i, diferencia centrada
    return F

# verificacion: la suma de fuerzas internas debe ser 0 (tercera ley)
resto = norm(sum(fuerza_analitica(j) for j in range(3)))   # residuo ~ 0 (redondeo)
"""
    conv = [
        ("Error de la fuerza frente al paso h", f"Se repite el cálculo de $\\vec F_3=-\\nabla U$ con distintos $h$ y se mide el error relativo $\\|\\vec F_{{num}}-\\vec F_{{an}}\\|/\\|\\vec F_{{an}}\\|$. Mientras domina el truncamiento la pendiente en escala log-log debe ser $2$: entre $10^{{-2}}$ y $10^{{-3}}$ se observa {s12:.2f} y entre $10^{{-3}}$ y $10^{{-4}}$ se observa {s23:.2f}.",
         [_arr("ccc", [r"h\ (\mathrm{m})", r"\text{error relativo}", r"\text{régimen}"],
               [[sci(hs[0], 1), sci(errs[0], 3), r"\text{truncamiento}"], [sci(hs[1], 1), sci(errs[1], 3), r"\text{truncamiento}"],
                [sci(hs[2], 1), sci(errs[2], 3), r"\text{truncamiento}"], [sci(hs[3], 1), sci(errs[3], 3), r"\text{óptimo}"],
                [sci(hs[4], 1), sci(errs[4], 3), r"\text{redondeo}"]])]),
        ("Fuentes de error y paso óptimo", f"El error total se modela como $\\varepsilon_{{tot}}\\approx C h^2+R/h$. Con la fila $h=10^{{-3}}$ (puro truncamiento) se obtiene $C=\\varepsilon_{{rel}}/h^2$ y con la energía $|U|$ se estima $R=\\varepsilon_{{m}}|U|/\\|\\vec F\\|$. El mínimo está en $h^*=(R/2C)^{{1/3}}$.",
         [rf"C = {sci(c_trunc, 3)}\ \mathrm{{m^{{-2}}}},\qquad R = \dfrac{{\varepsilon_m |U|}}{{\|\vec F\|}} = {sci(c_round, 3)}\ \mathrm{{m}},\qquad |U| = {sci(abs(u0), 4)}\ \mathrm{{J}}",
          rf"h^* = \left(\dfrac{{R}}{{2C}}\right)^{{1/3}} = {sci(h_opt, 3)}\ \mathrm{{m}},\qquad \varepsilon_{{tot}}(h^*) \approx {sci(c_trunc * h_opt ** 2 + c_round / h_opt, 2)}"]),
        ("Por qué diferencia centrada y cuándo es confiable", f"Con el mismo $h=1\\ \\mu\\mathrm{{m}}$ la diferencia hacia adelante (orden 1) da un error relativo de {err_fw:.1e}, contra {err_c:.1e} de la centrada: {err_fw / err_c:.0f} veces peor. Como criterio práctico, el resultado es confiable si al cambiar $h$ por $h/2$ y $2h$ la fuerza cambia mucho menos que la precisión buscada. La tercera ley da una verificación adicional independiente del paso.",
         [rf"\left\|\sum_i \vec F_i\right\| = {sci(res3, 3)}\ \mathrm{{N}}\quad\Rightarrow\quad \dfrac{{\left\|\sum_i \vec F_i\right\|}}{{\max_i\|\vec F_i\|}} = {sci(res3 / scale, 2)}"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== EquilibriumPoint
def _equilibrium() -> dict:
    d = e1.SEP
    xa = e1.zero_force_point_analytic()
    lo0, hi0 = 1e-6 * d, d * (1 - 1e-6)
    same = lambda mid: float(e1.net_force_on_test(mid)) > 0  # noqa: E731
    ks = [5, 10, 20, 30, 40, 50]
    err_s = [abs(_bisect(same, lo0, hi0, k) - xa) for k in ks]
    bound = [(hi0 - lo0) / 2 ** (k + 1) for k in ks]
    k_tol = int(np.ceil(np.log2((hi0 - lo0) / 1e-6) - 1))
    # carga de signo opuesto
    q1, q2 = 4e-6, -1e-6
    oa = e1.opposite_charges_point_analytic()
    f_op = lambda x: q1 / x**2 + q2 / (x - d) ** 2  # noqa: E731
    lo1, hi1 = d * (1 + 1e-9), 100 * d
    opp = lambda mid: not (f_op(mid) > 0)  # noqa: E731  (f>0 => hi=mid)
    kso = [10, 20, 40, 60, 100]
    err_o = [abs(_bisect(opp, lo1, hi1, k) - oa) for k in kso]
    bound_o = [(hi1 - lo1) / 2 ** (k + 1) for k in kso]
    # residuo y condicionamiento (caso de signos iguales, k = 20)
    x20 = _bisect(same, lo0, hi0, 20)
    h = 1e-6
    fp = float((e1.net_force_on_test(xa + h) - e1.net_force_on_test(xa - h)) / (2 * h))
    res20 = abs(float(e1.net_force_on_test(x20)))
    x100 = e1.zero_force_point_numeric()

    intro = [
        "Encontrar el punto donde la fuerza neta se anula es resolver una ecuación $F(x)=0$. Para dos cargas del mismo signo existe una fórmula cerrada, pero el método numérico no la necesita: solo exige poder evaluar la fuerza en un punto y saber en qué lado de la raíz se está. Por eso sirve igual cuando la ecuación no tiene solución algebraica sencilla (varias cargas, geometrías raras).",
        "La bisección parte de un intervalo $[a,b]$ donde $F(a)$ y $F(b)$ tienen signos opuestos. En cada iteración se evalúa el punto medio $m=(a+b)/2$ y se conserva la mitad que sigue conteniendo el cambio de signo. Si $F$ es continua, el teorema del valor intermedio garantiza que hay una raíz en el intervalo y que, tras $k$ iteraciones, $|x_k-x^*|\\le (b-a)/2^{k+1}$. La convergencia es lineal, con orden 1 y razón $1/2$: se gana un bit (unos $0.3$ dígitos decimales) por iteración, sin importar la forma de $F$.",
        f"Las limitaciones son claras: solo encuentra raíces con cambio de signo, necesita un intervalo que las acote y es lento comparado con Newton. En el código el intervalo es $[10^{{-6}}d,\\,(1-10^{{-6}})d]$ porque la fuerza diverge en $x=0$ y $x=d$ (cargas puntuales) y hay que evitar esos extremos. Para tolerancia $10^{{-6}}$ m bastan {k_tol} iteraciones; el código hace 100, de sobra: después de unas 50 el intervalo ya es menor que la separación entre números en coma flotante ($\\sim10^{{-16}}$ m) y no se mejora más.",
        "Es independiente del resultado analítico porque la fórmula $x=d\\sqrt{q_1}/(\\sqrt{q_1}+\\sqrt{q_2})$ nunca aparece en el código: solo se usa el signo de $F_{net}(x)=kq_t\\left[q_1/x^2-q_2/(d-x)^2\\right]$. Para cargas opuestas el equilibrio queda fuera del segmento, así que el intervalo cambia a $[d(1+10^{-9}),\\,100d]$ y el criterio de signo se invierte; el método sigue igual, solo cambia el intervalo inicial.",
    ]
    code = """def biseccion(F, a, b, iteraciones=100):
    # F(a) y F(b) deben tener signos opuestos (la raiz esta dentro)
    for _ in range(iteraciones):
        m = 0.5 * (a + b)             # punto medio del intervalo actual
        if F(m) > 0:                  # F > 0: la fuerza empuja a la derecha, la raiz esta a la derecha
            a = m                     # nos quedamos con la mitad [m, b]
        else:
            b = m                     # nos quedamos con la mitad [a, m]
    return 0.5 * (a + b)              # estimacion final de la raiz

def F_neta(x, q1=4e-6, q2=1e-6, d=0.6, qt=1e-9):
    return K * qt * (q1 / x**2 - q2 / (d - x)**2)   # fuerza sobre la carga de prueba en x

x_eq = biseccion(F_neta, 1e-6 * 0.6, 0.6 * (1 - 1e-6))   # evita los extremos x=0 y x=d
# caso de signos opuestos: f(x) = q1/x**2 + q2/(x-d)**2 con q2 = -1e-6, x en (d(1+1e-9), 100 d)
"""
    conv = [
        ("Error frente al número de iteraciones (cargas del mismo signo)", f"Se repite la bisección con $k$ iteraciones y se compara con $x^*=d\\sqrt{{q_1}}/(\\sqrt{{q_1}}+\\sqrt{{q_2}})={xa:.6f}$ m. La cota teórica $(b-a)/2^{{k+1}}$ se cumple mientras el error esté por encima del redondeo ($\\sim10^{{-16}}$ m; en la última fila el error real ya es del tamaño de la cota y deja de bajar). En promedio el error se reduce a la mitad por iteración, es decir unas $2^{{10}}\\approx1000$ veces cada 10 iteraciones.",
         [_arr("cccc", [r"k", r"|x_k-x^*|\ (\mathrm{m})", r"\text{cota }(b-a)/2^{k+1}", r"\text{real/cota}"],
               [[str(k), sci(e, 3) if e > 0 else "0", sci(b_, 3), fx(e / b_, 3)] for k, e, b_ in zip(ks, err_s, bound)])]),
        ("Condicionamiento: residuo frente a error", f"Una pregunta práctica es si basta mirar el residuo $|F(x_k)|$. Cerca de una raíz simple, $|x_k-x^*|\\approx|F(x_k)|/|F'(x^*)|$. Para $k=20$ el residuo es {res20:.2e} N y la pendiente de $F$ en la raíz es {abs(fp):.3e} N/m, lo que predice un error de {res20 / abs(fp):.2e} m; el error real es {abs(x20 - xa):.2e} m. Con $F'$ grande, la raíz está bien condicionada y un residuo pequeño implica error pequeño.",
         [rf"|x_{{20}}-x^*| = {sci(abs(x20 - xa), 3)}\ \mathrm{{m}},\qquad \dfrac{{|F(x_{{20}})|}}{{|F'(x^*)|}} = \dfrac{{{sci(res20, 3)}}}{{{sci(abs(fp), 3)}}} = {sci(res20 / abs(fp), 3)}\ \mathrm{{m}}"]),
        ("Cargas opuestas e iteraciones necesarias", f"En el caso $q_2=-1\\ \\mu\\mathrm{{C}}$ el intervalo es mucho más largo ($\\approx99d$) y hacen falta más iteraciones para la misma tolerancia. El resultado de 100 iteraciones es {_bisect(opp, lo1, hi1, 100):.9f} m frente al exacto $2d={oa:.9f}$ m. Criterio de confiabilidad: la cota $(b-a)/2^{{k+1}}$ es conocida de antemano, por lo que el error está garantizado sin conocer la respuesta; con 100 iteraciones queda por debajo del redondeo.",
         [_arr("ccc", [r"k", r"|x_k-x^*|\ (\mathrm{m})", r"\text{cota}"],
               [[str(k), sci(e, 3) if e > 0 else "0", sci(b_, 3)] for k, e, b_ in zip(kso, err_o, bound_o)]),
          rf"x_{{100}}^{{(\mathrm{{mismo\ signo}})}} = {x100:.12f}\ \mathrm{{m}}"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== DipoleField
def _dipole() -> dict:
    x0, y0 = 0.3, 0.2
    ev = e2.field_dipole_vector(x0, y0)
    nev = float(np.linalg.norm(ev))
    en = e2.field_from_potential_numeric(x0, y0)
    hs = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    errs = [float(np.linalg.norm(e2.field_from_potential_numeric(x0, y0, h=h) - ev) / nev) for h in hs]
    s1 = _slope(hs[0], hs[1], errs[0], errs[1])
    s2 = _slope(hs[1], hs[2], errs[1], errs[2])

    def line(ds: float, length: float = 0.016, ang: float = 60.0) -> np.ndarray:
        a = np.deg2rad(ang)
        r = np.array([e2.DSEP / 2, 0.0]) + 2e-4 * np.array([np.cos(a), np.sin(a)])
        rhs = lambda _s, rr: (lambda e: e / np.linalg.norm(e))(e2.field_dipole_vector(rr[0], rr[1]))  # noqa: E731
        for _ in range(int(round(length / ds))):
            r = rk4_step(rhs, 0.0, r, ds)
        return r

    L = 0.016
    ref = line(L / 2**14)
    ns = [64, 128, 256, 512, 1024]
    lerr = [float(np.linalg.norm(line(L / n) - ref)) for n in ns]
    sl = [_slope(L / ns[i], L / ns[i + 1], lerr[i], lerr[i + 1]) for i in range(len(ns) - 1)]
    zs = [0.05, 0.10, 0.20, 0.50, 1.00]
    dev = []
    for z in zs:
        ex, ef = float(e2.dipole_field_axis_exact(z)), float(e2.dipole_field_axis_far(z))
        dev.append((ef - ex) / ex)

    intro = [
        "Un dipolo no tiene un campo con forma sencilla fuera del eje, y mucho menos líneas de campo en forma cerrada. Aquí se resuelven dos problemas numéricos distintos. El primero es obtener el campo en un punto cualquiera a partir del potencial escalar, $\\vec E=-\\nabla V$; el segundo es trazar las líneas de campo, que son curvas tangentes a $\\vec E$ en cada punto.",
        "El gradiente se calcula con diferencias centradas, $E_x\\approx-[V(x+h,y)-V(x-h,y)]/(2h)$ y lo mismo en $y$, con error de truncamiento $O(h^2)$ y un error de redondeo que crece como $\\varepsilon V/h$; el código usa $h=1\\ \\mu\\mathrm{m}$. Las líneas se obtienen integrando la ecuación diferencial $d\\vec r/ds=\\vec E/|\\vec E|$ con Runge-Kutta de orden 4: $\\vec r_{n+1}=\\vec r_n+\\tfrac{\\Delta s}{6}(k_1+2k_2+2k_3+k_4)$, donde cada $k_i$ es el campo unitario evaluado en un punto intermedio. Al normalizar el campo, el parámetro $s$ es la longitud de arco.",
        "RK4 tiene error local $O(\\Delta s^5)$ y error global $O(\\Delta s^4)$ si el campo es suficientemente suave. Falla (o exige pasos muy pequeños) cerca de las cargas, donde el campo diverge y la curva se curva fuertemente; por eso la línea arranca a $0.2$ mm de la carga positiva y termina cuando llega a $0.4$ mm de la negativa. El paso del código, $\\Delta s=50\\ \\mu\\mathrm{m}$, es del orden de la escala de la zona crítica, mucho menor que la separación de $2$ cm.",
        "Ninguno de los dos cálculos usa la fórmula analítica del dipolo. El gradiente parte del potencial $V=kq(1/r_1-1/r_2)$ y se compara con la suma vectorial de Coulomb; las líneas se construyen únicamente con el campo coulombiano punto a punto. La aproximación de campo lejano $2kp/z^3$ de la parte analítica es un error de modelo (se truncó la serie en $d/z$), distinto del error numérico que se estudia aquí.",
    ]
    code = """def V(x, y):                              # potencial de las dos cargas +q (en d/2) y -q (en -d/2)
    r1 = hypot(x - d/2, y); r2 = hypot(x + d/2, y)
    return K * q * (1/r1 - 1/r2)

def campo_desde_V(x, y, h=1e-6):
    Ex = -(V(x + h, y) - V(x - h, y)) / (2 * h)    # -dV/dx, diferencia centrada
    Ey = -(V(x, y + h) - V(x, y - h)) / (2 * h)    # -dV/dy
    return array([Ex, Ey])

def rhs(s, r):                            # dr/ds = E/|E|, direccion unitaria del campo
    E = campo_coulomb(r)                  # suma vectorial de las dos cargas
    return E / norm(E)

def linea_de_campo(angulo, ds=5e-5):
    r = [d/2, 0] + 2e-4 * [cos(angulo), sin(angulo)]   # arranca cerca de la carga positiva
    for _ in range(20000):
        r = rk4_step(rhs, 0, r, ds)       # un paso de Runge-Kutta de orden 4
        if norm(r - [-d/2, 0]) < 4e-4:    # llego a la carga negativa: la linea se cierra
            return True
    return False
"""
    conv = [
        ("Error del gradiente frente al paso h", f"Se compara $-\\nabla V$ con la suma vectorial de Coulomb en $(0.3,\\,0.2)$ m para distintos $h$. La pendiente entre $10^{{-2}}$ y $10^{{-3}}$ es {s1:.2f} y entre $10^{{-3}}$ y $10^{{-4}}$ es {s2:.2f}: confirma el orden 2. Por debajo de $10^{{-6}}$ el redondeo domina y el error vuelve a subir.",
         [_arr("cc", [r"h\ (\mathrm{m})", r"\|\vec E_{num}-\vec E\|/\|\vec E\|"], [[sci(h, 1), sci(e, 3)] for h, e in zip(hs, errs)]),
          rf"\vec E(0.3,\,0.2) = ({sci(ev[0], 5)},\ {sci(ev[1], 5)})\ \mathrm{{N/C}},\qquad h=10^{{-6}}\ \text{{da}}\ ({sci(en[0], 5)},\ {sci(en[1], 5)})"]),
        ("Error de RK4 al trazar una línea de campo", f"Se integra una línea que sale a $60^\\circ$ de la carga positiva a lo largo de un arco de {L * 100:.1f} cm con distintos pasos y se mide la distancia del punto final a una solución de referencia con paso $L/2^{{14}}$. Si RK4 es de orden 4 el error debe caer un factor $16$ al dividir $\\Delta s$ entre 2; las pendientes locales observadas son " + ", ".join(f"{s_:.2f}" for s_ in sl) + " (el orden asintótico 4 se alcanza solo cuando $\\Delta s$ es mucho menor que la escala de curvatura cerca de la carga).",
         [_arr("ccc", [r"\Delta s\ (\mathrm{m})", r"\text{error final }(\mathrm{m})", r"\text{pendiente}"],
               [[sci(L / n, 3), sci(e, 3), "-" if i == 0 else fx(sl[i - 1], 2)] for i, (n, e) in enumerate(zip(ns, lerr))])]),
        ("Error numérico frente a error de modelo", "El paso del código, $\\Delta s=5\\times10^{-5}$ m, queda entre las dos últimas filas de la tabla anterior, con errores por debajo de $10^{-9}$ m: la línea se traza con una precisión muchísimo mayor que el grosor de la figura. En cambio, la aproximación dipolar de la parte analítica sí tiene un error apreciable, que no es numérico sino de truncar la serie en $d/z$ y que decae como $(d/z)^2$.",
         [_arr("ccc", [r"z\ (\mathrm{m})", r"E_{lejano}/E_{exacto}-1", r"(d/z)^2"],
               [[fx(z, 2), sci(v, 3), sci((e2.DSEP / z) ** 2, 3)] for z, v in zip(zs, dev)]),
          r"\text{Criterio: el gradiente es confiable si al cambiar }h\text{ por }h/2\text{ el campo cambia menos que la tolerancia, y la línea si al reducir }\Delta s\text{ no se mueve el punto final.}"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== ChargedRing
def _ring_comps(z: float, n: int) -> tuple[float, float, float]:
    """Replica de ring_axis_field_numeric devolviendo (Ez, Ex, Ey) con Simpson en n subintervalos."""
    from physics.ch07_work_energy import simpson

    a, q = e2.RING_R, e2.RING_Q
    phi = np.linspace(0, 2 * np.pi, n + 1)
    dq = q / (2 * np.pi)
    src = np.stack([a * np.cos(phi), a * np.sin(phi), np.zeros_like(phi)], axis=1)
    r = np.array([0.0, 0.0, z]) - src
    dist = np.linalg.norm(r, axis=1)
    c = K_E * dq * r / dist[:, None] ** 3
    return simpson(c[:, 2], phi), simpson(c[:, 0], phi), simpson(c[:, 1], phi)


def _ring() -> dict:
    a = e2.RING_R
    za = e2.ring_max_field_position_analytic()
    z0 = 0.05
    ea = float(e2.ring_axis_field_analytic(z0))
    ns = [2, 4, 8, 16, 400]
    rows = []
    for n in ns:
        ez, ex, ey = _ring_comps(z0, n)
        rows.append([str(n), sci(abs(ez / ea - 1), 2) if ez != ea else "0", sci(abs(ex) / ez, 2), sci(abs(ey) / ez, 2)])
    ez400 = _ring_comps(z0, 400)[0]
    # maximo: error vs separacion de muestreo
    strides = [40, 20, 10, 5]
    zs_all = np.linspace(0.001, 3 * a, 2000)
    res = []
    for st in strides:
        zc = zs_all[::st]
        e = np.array([e2.ring_axis_field_numeric(float(z), n=200) for z in zc])
        i = int(np.argmax(e))
        lo, hi = max(i - 3, 0), min(i + 4, len(zc))
        c = np.polyfit(zc[lo:hi], e[lo:hi], 2)
        zf = float(-c[1] / (2 * c[0]))
        res.append((float(zc[1] - zc[0]), abs(zf - za), abs(float(zc[i]) - za)))
    sl = [_slope(res[i][0], res[i + 1][0], res[i][1], res[i + 1][1]) for i in range(len(res) - 1)]
    zl = [0.3, 0.5, 1.0, 5.0]
    far = []
    for z in zl:
        far.append(e2.ring_axis_field_numeric(z) / (K_E * e2.RING_Q / z**2) - 1)

    intro = [
        "El campo del anillo se calcula sin usar la fórmula $kQz/(z^2+a^2)^{3/2}$: se suma la contribución de cada elemento de carga $dq=Q\\,d\\varphi/2\\pi$ con la ley de Coulomb vectorial $d\\vec E=k\\,dq\\,\\vec r/r^3$ y se integra sobre el ángulo $\\varphi$. En cada punto del anillo $\\vec r$ apunta del elemento al punto de observación, así que se necesitan los tres componentes de $d\\vec E$ y la integral no se reduce a una cuenta de una línea.",
        "La integral se aproxima con la regla de Simpson compuesta con $n$ subintervalos de ancho $h=2\\pi/n$: $\\int f\\,d\\varphi\\approx\\tfrac h3\\left[f_0+4f_1+2f_2+\\dots+4f_{n-1}+f_n\\right]$. Su error es $-\\dfrac{(b-a)h^4}{180}f^{(4)}(\\xi)$, es decir $O(h^4)$ (de ahí que exija $n$ par). Falla si el integrando tiene singularidades o saltos; aquí no, porque el punto de observación está siempre fuera del anillo.",
        "Hay una particularidad: sobre el eje el integrando es muy especial. El componente $z$ es constante en $\\varphi$ y los componentes $x$, $y$ son $\\cos\\varphi$ y $\\sin\\varphi$ (se cancelan al integrar sobre una vuelta completa). Para una función periódica integrada sobre un período, Simpson es esencialmente exacto con muy pocos puntos (a partir de $n=4$ en el estudio de abajo), por lo que $n=400$ es una elección conservadora y el error es de redondeo; la comprobación de código (`assert`) exige que $E_x$ y $E_y$ queden por debajo de $10^{-6}E_z$.",
        "La posición del máximo se halla muestreando $E_z$ cada $10$ puntos de una malla de 2000 entre $1$ mm y $30$ cm (separación $1.5$ mm), tomando el máximo muestreado y ajustando una parábola a 7 puntos alrededor de él: el vértice $-c_1/2c_0$ estima $z_{max}$. Este paso sí tiene error numérico apreciable, de orden $O(\\Delta z^2)$, y es independiente de la derivada analítica $dE/dz=0$ que da $a/\\sqrt2$.",
    ]
    code = """def campo_anillo_eje(z, Q=10e-9, a=0.10, n=400):
    phi = linspace(0, 2*pi, n + 1)               # angulos de los elementos de carga
    dq = Q / (2*pi)                              # carga por radian (anillo uniforme)
    fuente = [a*cos(phi), a*sin(phi), 0]         # posicion de cada elemento dq
    r = [0, 0, z] - fuente                       # vector del elemento al punto de observacion
    dist = norm(r)                               # |r| en cada elemento
    dE = K * dq * r / dist**3                    # Coulomb vectorial por radian
    Ez = simpson(dE[:, 2], phi)                  # integral del componente axial (Simpson, n par)
    Ex = simpson(dE[:, 0], phi)                  # los transversales deben salir ~0
    Ey = simpson(dE[:, 1], phi)
    assert abs(Ex) < 1e-6*abs(Ez) and abs(Ey) < 1e-6*abs(Ez)   # comprobacion de simetria
    return Ez

z = linspace(0.001, 0.30, 2000)[::10]            # muestreo grueso (separacion 1.5 mm)
E = [campo_anillo_eje(zi, n=200) for zi in z]
i = argmax(E)                                    # maximo muestreado
c = polyfit(z[i-3:i+4], E[i-3:i+4], 2)           # parabola por 7 puntos
z_max = -c[1] / (2*c[0])                         # vertice de la parabola
"""
    conv = [
        ("Simpson frente al número de subintervalos", f"Se evalúa $E_z(5\\ \\mathrm{{cm}})$ con $n$ subintervalos y se compara con $kQz/(z^2+a^2)^{{3/2}}={sci(ea, 6)}$ N/C. El componente axial es exacto desde $n=2$ porque su integrando es constante en $\\varphi$. Los transversales ($\\cos\\varphi$, $\\sin\\varphi$) no se cancelan con $n=2$ (un componente queda en $0.67\\,E_z$: la malla es demasiado burda para ver una vuelta completa) pero desde $n=4$ ya están en el redondeo ($\\sim10^{{-16}}$), de modo que la cancelación por simetría se cumple con holgura y $n=400$ es muy conservador.",
         [_arr("cccc", [r"n", r"|E_z/E_{an}-1|", r"|E_x|/E_z", r"|E_y|/E_z"], rows),
          rf"E_z^{{(n=400)}} = {sci(ez400, 7)}\ \mathrm{{N/C}}"]),
        ("Posición del máximo frente al muestreo", "Aquí sí aparece un error de discretización. Se repite la búsqueda del máximo con distinta separación entre puntos de muestreo $\\Delta z$ y se mide $|z_{fit}-a/\\sqrt2|$. El ajuste parabólico mejora al máximo muestreado a secas y su error cae casi como $\\Delta z^2$: pendientes locales " + ", ".join(f"{s_:.2f}" for s_ in sl) + ".",
         [_arr("ccc", [r"\Delta z\ (\mathrm{mm})", r"|z_{fit}-z_{max}|\ (\mathrm{mm})", r"|z_{muestreado}-z_{max}|\ (\mathrm{mm})"],
               [[fx(r_[0] * 1000, 2), fx(r_[1] * 1000, 4), fx(r_[2] * 1000, 4)] for r_ in res]),
          rf"z_{{max}} = a/\sqrt2 = {fx(za * 100, 4)}\ \mathrm{{cm}};\qquad \Delta z=1.5\ \mathrm{{mm}}\ \Rightarrow\ \text{{error }}{sci(res[2][1] * 1000, 2)}\ \mathrm{{mm}}"]),
        ("Verificación cruzada con el límite lejano", "Como control independiente, lejos del anillo ($z\\gg a$) el campo debe tender al de una carga puntual $kQ/z^2$ con corrección relativa $\\approx-\\tfrac32(a/z)^2$. La tabla muestra que el cálculo numérico reproduce esa tendencia sin degradarse con la distancia, lo que descarta errores de signo o de factor. Criterio práctico: el resultado es confiable si los transversales están en el redondeo y el límite lejano se recupera.",
         [_arr("cc", [r"z\ (\mathrm{m})", r"E_{num}/(kQ/z^2)-1"], [[fx(z, 2), sci(v, 3)] for z, v in zip(zl, far)])]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== GaussFlux
def _gauss_flux() -> dict:
    phi = e3.flux_sphere_analytic()
    off = (0.1, 0.05, 0.0)
    nth = [25, 50, 100, 200]
    e_off = [e3.flux_sphere_numeric(charge_pos=off, n_theta=n, n_phi=2 * n) / phi - 1 for n in nth]
    e_cen = [e3.flux_sphere_numeric(n_theta=n, n_phi=2 * n) / phi - 1 for n in nth]
    ncube = [50, 100, 200, 400]
    cube = [e3.flux_cube_numeric(n=n) for n in ncube]
    e_cube = [c / phi - 1 for c in cube]
    s_cube = [_slope(ncube[i + 1], ncube[i], abs(e_cube[i + 1]), abs(e_cube[i])) for i in range(3)]
    s_off = [_slope(nth[i + 1], nth[i], abs(e_off[i + 1]), abs(e_off[i])) for i in range(3)]
    rich = (4 * cube[3] - cube[2]) / 3
    out = [e3.flux_sphere_numeric(radius=0.1, charge_pos=(0.3, 0.0, 0.0), n_theta=n, n_phi=2 * n) for n in nth]
    d_min = 0.3 - float(np.linalg.norm(off))

    intro = [
        "La ley de Gauss afirma que el flujo $\\oint\\vec E\\cdot d\\vec A$ vale $q/\\varepsilon_0$ para cualquier superficie cerrada, pero para verlo hay que hacer la integral de superficie. Con la carga descentrada o en un cubo el campo ya no es perpendicular ni constante sobre la superficie y la integral no se resuelve a mano; el método numérico la calcula sin usar el resultado $q/\\varepsilon_0$ y así lo pone a prueba.",
        "Se usa la regla del punto medio en dos dimensiones. La superficie se divide en celdas, en cada una se evalúa $\\vec E\\cdot\\hat n$ en su centro y se multiplica por el área de la celda: $\\Phi\\approx\\sum_{ij}\\vec E(P_{ij})\\cdot\\hat n_{ij}\\,\\Delta A_{ij}$. En la esfera las celdas son de $(\\theta,\\varphi)$ con $\\Delta A=R^2\\sin\\theta\\,\\Delta\\theta\\,\\Delta\\varphi$; en el cubo, seis mallas cuadradas de $n\\times n$ celdas.",
        "El error del punto medio es $\\dfrac{h^2}{24}(b-a)\\,f''$ en cada dirección, por lo que el método es de orden $O(h^2)$ y duplicar la resolución reduce el error unas 4 veces. Funciona mientras el integrando sea suave: aquí el campo es suave en toda la superficie porque la carga nunca está sobre ella (la distancia mínima en el caso descentrado es $0.188$ m). Se degradaría muchísimo si la carga estuviera casi sobre la superficie, donde $E\\sim1/d^2$ crece sin control.",
        f"Los parámetros del código (esfera de $200\\times400$ celdas y cubo con $6\\times400^2$) producen errores relativos de entre $10^{{-6}}$ y $10^{{-5}}$ (esfera descentrada: {abs(e3.flux_sphere_numeric(charge_pos=off) / phi - 1):.1e}; cubo: {abs(e3.flux_cube_numeric() / phi - 1):.1e}), por debajo de lo que se ve en una tabla de cuatro cifras. La carga fuera de la superficie es un caso especialmente instructivo: el integrando tiene signos opuestos en las dos caras y la integral numérica debe dar $0$ como resultado de una cancelación.",
    ]
    code = """def flujo_esfera(R=0.3, q=2e-9, carga=(0, 0, 0), n_theta=200, n_phi=400):
    th = (arange(n_theta) + 0.5) * pi / n_theta       # angulos polares en el CENTRO de cada celda
    ph = (arange(n_phi) + 0.5) * 2*pi / n_phi         # angulos azimutales en el centro
    T, P = meshgrid(th, ph)
    n = [sin(T)*cos(P), sin(T)*sin(P), cos(T)]        # normal exterior unitaria
    puntos = R * n                                    # puntos de la superficie
    r = puntos - carga
    E = K * q * r / norm(r)**3                        # campo de Coulomb en cada celda
    dA = R**2 * sin(T) * (pi/n_theta) * (2*pi/n_phi)  # area de cada celda en (theta, phi)
    return suma((E * n).sum(eje=-1) * dA)             # sum E.n dA

def flujo_cubo(lado=0.4, q=2e-9, n=400):
    s = (arange(n) + 0.5) / n * lado - lado/2         # centros de celda en una cara
    dA = (lado / n)**2
    total = 0
    for eje in range(3):                              # tres ejes
        for signo in (+1, -1):                        # caras + y -
            E = campo(cara_en(eje, signo*lado/2, s))  # campo en los centros de la cara
            total += suma(signo * E[eje]) * dA        # E.n = +-E_eje
    return total
"""
    conv = [
        ("Error frente a la resolución", f"Se repite la integración duplicando la malla. Para la esfera con la carga descentrada $(0.1,\\,0.05,\\,0)$ m se usa $n_\\theta\\times n_\\varphi=n\\times2n$ y para el cubo $n\\times n$ por cara. Las pendientes log-log observadas son {', '.join(f'{s:.2f}' for s in s_off)} (esfera descentrada) y {', '.join(f'{s:.2f}' for s in s_cube)} (cubo): confirman el orden 2 del punto medio.",
         [_arr("cccc", [r"n_\theta\ (\text{esfera})", r"|\Phi/\Phi_0-1|\ \text{(esfera)}", r"n\ (\text{cubo})", r"|\Phi/\Phi_0-1|\ \text{(cubo)}"],
               [[str(a), sci(abs(b), 3), str(c), sci(abs(d), 3)] for a, b, c, d in zip(nth, e_off, ncube, e_cube)]),
          rf"\Phi_0 = \dfrac{{q}}{{\varepsilon_0}} = {phi:.4f}\ \mathrm{{N\,m^2/C}}"]),
        ("Esfera centrada y carga exterior: cancelaciones", f"Con la carga centrada $\\vec E\\cdot\\hat n$ es constante y el único error es el de integrar $\\sin\\theta$ con el punto medio: error relativo {abs(e_cen[-1]):.1e} con $n_\\theta=200$ (cae como $n_\\theta^{{-2}}$). Cuando la carga está fuera de la superficie el flujo exacto es cero, de modo que la medida natural es el error absoluto frente a la escala $\\Phi_0$; el valor numérico es un residuo de cancelación entre contribuciones positivas y negativas.",
         [_arr("cc", [r"n_\theta", r"\Phi_{num}\ (\mathrm{N\,m^2/C})\ \text{(carga fuera)}"], [[str(n), sci(v, 3)] for n, v in zip(nth, out)]),
          rf"\dfrac{{|\Phi_{{num}}|}}{{\Phi_0}}\Big|_{{n_\theta=200}} = {sci(abs(out[-1]) / phi, 2)}"]),
        ("Extrapolación de Richardson y criterio de confianza", f"Como el error es $\\approx Ch^2$, combinar dos resoluciones elimina el término dominante: $\\Phi_R=(4\\Phi_{{2n}}-\\Phi_n)/3$. Aplicado al cubo con $n=200$ y $400$ da error relativo {abs(rich / phi - 1):.1e}, frente a {abs(e_cube[-1]):.1e} sin extrapolar. Costo: la esfera usa $200\\times400=8\\times10^4$ evaluaciones y el cubo $6\\times400^2\\approx9.6\\times10^5$. Criterio práctico: el flujo es confiable cuando al duplicar $n$ el cambio es unas 4 veces menor que el anterior y el resultado concuerda con $q/\\varepsilon_0$ en la tolerancia deseada.",
         [rf"\Phi_R = \dfrac{{4\Phi_{{400}}-\Phi_{{200}}}}{{3}} = {phi * (rich / phi):.5f}\ \mathrm{{N\,m^2/C}},\qquad \left|\dfrac{{\Phi_R}}{{\Phi_0}}-1\right| = {sci(abs(rich / phi - 1), 3)}",
          rf"d_{{min}} = R - |\vec r_q| = {fx(d_min, 3)}\ \mathrm{{m}}\quad(\text{{distancia mínima de la carga a la superficie}})"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== SphereFieldProfile
def _sphere() -> dict:
    ref = {r: float(e3.sphere_field_analytic(r)) for r in (0.05, 0.10, 0.20)}
    seeds = range(10)

    def stats(r: float, n: int) -> tuple[float, float, float]:
        errs, ses = [], []
        for s in seeds:
            v, se = e3.sphere_field_monte_carlo(r, n=n, seed=s)
            errs.append(v - ref[r])
            ses.append(se)
        rms = float(np.sqrt(np.mean(np.square(errs))))
        cover = float(np.mean(np.abs(errs) < 2 * np.array(ses)))
        return rms, float(np.mean(ses)), cover

    ns = [1000, 4000, 16000, 64000, 256000]
    s_out = [stats(0.20, n) for n in ns]
    sl_out = _slope(ns[0], ns[-1], s_out[0][0], s_out[-1][0])
    sl_se = _slope(ns[0], ns[-1], s_out[0][1], s_out[-1][1])
    n_cmp = [16000, 256000]
    cmp_ = {r: [stats(r, n) for n in n_cmp] for r in (0.05, 0.10)}
    cmp_[0.20] = [s_out[2], s_out[4]]
    sl_cmp = {r: _slope(n_cmp[0], n_cmp[1], cmp_[r][0][0], cmp_[r][1][0]) for r in cmp_}
    v17, se17 = e3.sphere_field_monte_carlo(0.20, n=200_000)
    n_need = ns[-1] * (s_out[-1][1] / (0.01 * ref[0.20])) ** 2

    intro = [
        "Para la esfera uniforme la ley de Gauss da $E(r)$ en un renglón, pero la verificación independiente consiste en no usar la simetría en absoluto: se reparte la carga $Q$ en $N$ cargas puntuales colocadas al azar dentro del volumen y se suma el campo de Coulomb de todas. Esto es un método de Monte Carlo, útil porque funciona igual para formas sin simetría, donde ninguna integral cerrada existe.",
        "El estimador es $\\vec E(\\vec r)\\approx\\sum_{i=1}^{N}k\\,\\dfrac{Q}{N}\\,\\dfrac{\\vec r-\\vec r_i}{|\\vec r-\\vec r_i|^3}$, con $\\vec r_i$ uniformes en la esfera: dirección uniforme (vector gaussiano normalizado) y radio $R\\,u^{1/3}$ con $u\\in[0,1]$ uniforme, que compensa el crecimiento del volumen como $r^3$. Por la ley de los grandes números converge al campo verdadero y, por el teorema central del límite, el error es aleatorio de tamaño $\\sigma/\\sqrt N$, con $\\sigma$ la desviación de una sola contribución. Convergencia lenta, orden $1/\\sqrt N$: para ganar un dígito hacen falta 100 veces más muestras.",
        "Tiene una limitación importante en este problema. Cada término es $\\propto1/d^2$ con $d$ la distancia a una carga y, cuando el punto de observación está dentro o sobre la esfera, hay cargas muy cercanas a él que producen contribuciones enormes. La varianza de una sola contribución diverge ($\\int d^{-4}\\,d^2\\,dd$), y el error ya no sigue la regla $1/\\sqrt N$ con limpieza; la desviación estándar muestral subestima los casos de cola larga. Fuera de la esfera ($r>R$) la distancia mínima es positiva y el comportamiento es el regular.",
        f"El código usa $N=200\\,000$ y una semilla fija (17) para que el resultado sea reproducible; la incertidumbre que reporta es la desviación estándar de la suma, $\\sqrt N\\,s$, con $s$ la desviación muestral de los términos. Con $r=0.20$ m se obtiene {v17:.1f} ± {se17:.1f} N/C frente a {ref[0.20]:.1f} N/C. Es un método independiente del analítico porque el campo interior $kQr/R^3$ (que resulta de Gauss) no se usa en ningún lugar del muestreo.",
    ]
    code = """def campo_esfera_MC(r, Q=5e-9, R=0.10, N=200_000, semilla=17):
    rng = default_rng(semilla)                      # semilla fija: resultado reproducible
    v = rng.normal(size=(N, 3))                     # direcciones: gaussiana normalizada = uniforme en la esfera
    v /= norm(v, axis=1)
    rad = R * rng.uniform(size=(N, 1)) ** (1/3)     # radio con densidad uniforme en volumen (u**(1/3))
    fuente = v * rad                                # N cargas puntuales dentro de la esfera
    obs = [r, 0, 0]                                 # punto de observacion sobre el eje x
    d = obs - fuente
    dist = norm(d, axis=1)
    E = K * (Q / N) * d / dist**3                   # Coulomb de cada carga q_i = Q/N
    Ex = E[:, 0]
    suma = Ex.sum()                                 # estimador del campo
    error = Ex.std(ddof=1) * sqrt(N)                # desviacion estandar de la suma
    return suma, error

for r in (0.05, 0.10, 0.20):                        # dentro, en la superficie, fuera
    print(r, campo_esfera_MC(r))
"""
    conv = [
        ("Error frente al número de muestras N (punto exterior r = 0.20 m)", f"Para eliminar la suerte de una semilla se repite el cálculo con 10 semillas y se toma la raíz del error cuadrático medio. La pendiente log-log observada entre $N=10^3$ y $N=2.56\\times10^5$ es {sl_out:.2f}, cercana al $-1/2$ teórico, y la incertidumbre reportada por el código escala con pendiente {sl_se:.2f}. Además el error real queda dentro de $2\\sigma$ en la fracción de semillas indicada en la última columna (esperado $\\approx0.95$).",
         [_arr("cccc", [r"N", r"\text{RMS del error (N/C)}", r"\text{incertidumbre reportada (N/C)}", r"|e|<2\sigma"],
               [[sci(n, 3) if n >= 1e4 else str(n), fx(s[0], 3), fx(s[1], 3), fx(s[2], 2)] for n, s in zip(ns, s_out)])]),
        ("Dentro, en la superficie y fuera: el mismo N, distinto comportamiento", f"Se compara el error de las tres posiciones de la tabla de la escena al pasar de $N=1.6\\times10^4$ a $N=2.56\\times10^5$ (16 veces más). Con $1/\\sqrt N$ el error debería bajar 4 veces. Pendientes observadas: {sl_cmp[0.05]:.2f} (r = 0.05, dentro), {sl_cmp[0.10]:.2f} (r = 0.10, superficie) y {sl_cmp[0.20]:.2f} (r = 0.20, fuera). Solo fuera de la esfera la pendiente se acerca a $-0.5$. Dentro y en la superficie los valores se alejan de él en cualquiera de los dos sentidos y el RMS de 10 semillas depende de unas pocas muestras extremas (cargas muy cercanas al punto): no es una estimación estable, y el error relativo sigue siendo de varios por ciento con $N=2.56\\times10^5$, mientras que fuera es de décimas de por ciento.",
         [_arr("ccccc", [r"r\ (\mathrm{m})", r"E_{an}\ (\mathrm{N/C})", r"\text{RMS con }N=1.6\times10^4", r"\text{RMS con }N=2.56\times10^5", r"\text{pendiente}"],
               [[fx(r, 2), fx(ref[r], 1), fx(cmp_[r][0][0], 2), fx(cmp_[r][1][0], 2), fx(sl_cmp[r], 2)] for r in cmp_])]),
        ("Fuentes de error y cuántas muestras hacen falta", f"La única fuente relevante es la estadística: no hay truncamiento (la suma es exacta para esas $N$ cargas) y el redondeo en $10^5$ sumas es despreciable. Como el error escala como $\\sigma_1/\\sqrt N$, a partir de la incertidumbre medida con $N=2.56\\times10^5$ en $r=0.20$ m se estima que se necesitarían unas $N\\approx{int(round(n_need, -2))}$ cargas para llegar a $1\\%$ de error relativo; para $0.1\\%$ serían 100 veces más. Criterio práctico: confiar en el resultado cuando la incertidumbre reportada sea menor que la tolerancia y cuando cambiar de semilla produzca cambios del mismo tamaño que ella; si no coinciden, sospechar de colas largas.",
         [rf"N_{{1\%}} \approx N\left(\dfrac{{\sigma_N}}{{0.01\,E_{{an}}}}\right)^2 = {sci(ns[-1], 3)}\left(\dfrac{{{fx(s_out[-1][1], 3)}}}{{0.01\cdot{fx(ref[0.20], 2)}}}\right)^2 = {sci(n_need, 2)}",
          rf"E_{{MC}}(0.20) = {fx(v17, 2)} \pm {fx(se17, 2)}\ \mathrm{{N/C}},\qquad E_{{an}}(0.20) = {fx(ref[0.20], 2)}\ \mathrm{{N/C}}\quad(N=2\times10^5,\ \text{{semilla }}17)"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== DiskPotential
def _trapz(n: int, z: float = 0.05) -> float:
    s = np.linspace(0, e4.R_DISK, n + 1)
    y = K_E * e4.SIGMA * 2 * np.pi * s / np.sqrt(s**2 + z**2)
    h = s[1] - s[0]
    return float(h * (y.sum() - 0.5 * (y[0] + y[-1])))


def _disk() -> dict:
    z0 = 0.05
    va = float(e4.disk_potential_analytic(z0))
    ea = float(e4.disk_field_analytic(z0))
    ns = [4, 8, 16, 32, 64]
    es = [abs(e4.disk_potential_numeric(z0, n=n) / va - 1) for n in ns]
    et = [abs(_trapz(n, z0) / va - 1) for n in ns]
    ss = [_slope(ns[i + 1], ns[i], es[i + 1], es[i]) for i in range(len(ns) - 1)]
    st = [_slope(ns[i + 1], ns[i], et[i + 1], et[i]) for i in range(len(ns) - 1)]
    s_est = abs(e4.disk_potential_numeric(z0, n=64) - e4.disk_potential_numeric(z0, n=32)) / 15
    hs = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    vnum = lambda z: e4.disk_potential_numeric(z, n=400)  # noqa: E731
    e_an = [abs(e4.field_from_potential_numeric(lambda z: float(e4.disk_potential_analytic(z)), z0, h=h) / ea - 1) for h in hs]
    e_nu = [abs(e4.field_from_potential_numeric(vnum, z0, h=h) / ea - 1) for h in hs]
    vn400 = e4.disk_potential_numeric(z0)
    n_eq = 64 * (et[4] / es[3]) ** 0.5

    intro = [
        "El potencial de un disco cargado se obtiene sumando el de anillos concéntricos: cada anillo de radio $s$ y ancho $ds$ tiene carga $dq=\\sigma\\,2\\pi s\\,ds$ y, vista desde un punto del eje a altura $z$, todos sus puntos están a la misma distancia $\\sqrt{s^2+z^2}$. Entonces $V(z)=\\int_0^R \\dfrac{k\\sigma\\,2\\pi s}{\\sqrt{s^2+z^2}}\\,ds$. Esta integral sí tiene primitiva elemental, pero aquí se resuelve numéricamente para comprobarla sin usar el resultado $\\sigma(\\sqrt{z^2+R^2}-z)/2\\varepsilon_0$, y porque el mismo método vale si $\\sigma$ depende de $s$ de manera complicada.",
        "La integral se evalúa con la regla de Simpson compuesta con $n$ subintervalos de ancho $h=R/n$: $\\int f\\,ds\\approx\\tfrac h3\\left[f_0+4f_1+2f_2+\\dots+4f_{n-1}+f_n\\right]$. Se obtiene ajustando una parábola por cada tres puntos, y su error es $-\\tfrac{R\\,h^4}{180}f^{(4)}(\\xi)$, es decir $O(h^4)$. Es exacta para polinomios de grado tres. Falla cuando el integrando tiene singularidades o derivadas grandes: con $z$ muy pequeño el término $\\sqrt{s^2+z^2}$ varía rápido cerca de $s=0$ y haría falta una malla más fina.",
        f"Con $n=400$ (ancho $h=0.25$ mm) el error relativo del potencial es {abs(vn400 / va - 1):.1e}; no hacen falta tantos subintervalos, pero el costo es insignificante. Después, el campo se obtiene derivando: $E_z=-dV/dz\\approx-[V(z+h)-V(z-h)]/(2h)$ con $h=1\\ \\mu\\mathrm{{m}}$, que es de orden $O(h^2)$ en truncamiento con redondeo $\\varepsilon V/h$. Ambas operaciones son independientes de la fórmula cerrada del campo $\\tfrac{{\\sigma}}{{2\\varepsilon_0}}(1-z/\\sqrt{{z^2+R^2}})$.",
        "Un detalle de implementación: en la escena la derivada se aplica a la función potencial analítica y no al $V$ de Simpson. Eso mide el error de la diferencia centrada por separado; en el estudio de convergencia de abajo se muestra también el caso encadenado (derivar el potencial numérico), que es el esquema completamente numérico.",
    ]
    code = """def simpson(y, x):                            # n subintervalos, n par
    n = len(x) - 1; h = (x[-1] - x[0]) / n
    return h/3 * (y[0] + y[-1] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum())   # pesos 1,4,2,4,...,4,1

def V_disco(z, sigma=2e-6, R=0.10, n=400):
    s = linspace(0, R, n + 1)                 # radios de los anillos
    dq_ds = sigma * 2*pi * s                  # carga por unidad de radio: sigma 2 pi s
    integrando = K * dq_ds / sqrt(s**2 + z**2)   # dV = k dq / distancia al punto del eje
    return simpson(integrando, s)             # suma de todos los anillos

def campo_desde_V(Vfun, z, h=1e-6):
    return -(Vfun(z + h) - Vfun(z - h)) / (2*h)  # E_z = -dV/dz con diferencia centrada

z0 = 0.05
V = V_disco(z0)                               # potencial numerico a 5 cm
Ez = campo_desde_V(lambda z: V_disco_analitico(z), z0)   # como en la escena (V analitico)
"""
    conv = [
        ("Simpson frente al número de subintervalos", f"Se calcula $V(5\\ \\mathrm{{cm}})$ con $n=4,8,\\dots,64$ y se compara con $\\dfrac{{\\sigma}}{{2\\varepsilon_0}}(\\sqrt{{z^2+R^2}}-z)={fx(va, 3)}$ V. Cada vez que $n$ se duplica el error se divide por $\\approx16$: pendientes locales {', '.join(f'{s:.2f}' for s in ss)}, que tienden al orden 4. Con $n=400$ el error ya es {abs(vn400 / va - 1):.1e}. Una estimación del error sin conocer la respuesta es $|S_{{2n}}-S_n|/15$; para $n=32$ da {s_est:.2e} V.",
         [_arr("cccc", [r"n", r"|V_{S}/V-1|", r"\text{pendiente}", r"|V_{T}/V-1|\ \text{(trapecio)}"],
               [[str(n), sci(e, 3), "-" if i == 0 else fx(ss[i - 1], 2), sci(t, 3)] for i, (n, e, t) in enumerate(zip(ns, es, et))])]),
        ("Por qué Simpson y no el trapecio", f"La última columna usa la regla del trapecio con los mismos $n$. Su error es $O(h^2)$ (pendientes {', '.join(f'{s:.2f}' for s in st)}) y con $n=64$ llega a {sci(et[4], 2)}, unas {et[4] / es[4]:.0f} veces peor que Simpson con el mismo $n$. Para igualar el error que Simpson logra con $n=32$ ({sci(es[3], 2)}) el trapecio necesitaría $n\\approx{n_eq:.0f}$ subintervalos. A igual costo, Simpson gana varios órdenes de magnitud mientras el integrando sea suave.",
         [rf"\text{{trapecio, }}n=64:\ {sci(et[4], 3)}\qquad\text{{Simpson, }}n=32:\ {sci(es[3], 3)}\qquad\text{{Simpson, }}n=64:\ {sci(es[4], 3)}"]),
        ("Derivada numérica del potencial: truncamiento, redondeo y propagación", f"Se calcula $E_z=-dV/dz$ con distinto $h$ en dos variantes: derivando el potencial analítico (como la escena) y derivando el potencial de Simpson con $n=400$. Para $h\\ge10^{{-4}}$ domina el truncamiento $O(h^2)$ y las dos variantes coinciden (el error de Simpson, $\\sim10^{{-11}}$, no se nota); en $h=10^{{-6}}$ aparece una pequeña diferencia entre ellas y en $h=10^{{-8}}$ domina el redondeo de $V$ ($\\varepsilon V/h$). Referencia: $E_{{an}}={fx(ea / 1000, 4)}$ kV/m.",
         [_arr("ccc", [r"h\ (\mathrm{m})", r"|E/E_{an}-1|\ (V_{an})", r"|E/E_{an}-1|\ (V_{Simpson})"],
               [[sci(h, 1), sci(a, 2), sci(b, 2)] for h, a, b in zip(hs, e_an, e_nu)]),
          r"\text{Criterio: el potencial por Simpson es confiable si }|S_{2n}-S_n|/15\text{ es menor que la tolerancia; la derivada, si cambia poco al variar }h\text{ por un factor 2.}"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


# =========================================================================== EquipotentialPair
def _grid(n: int, mask: float = 0.12, extent: float = 0.5) -> tuple[float, float, float]:
    """Error relativo maximo de -grad V: toda la malla lejana, y solo el interior (sin las 2 filas exteriores)."""
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dx = xs[1] - xs[0]
    gx, gy = np.gradient(e4.potential_pair(X, Y), dx, dx)
    eax, eay = e4.field_pair_analytic(X, Y)
    far = (np.hypot(X + e4.D / 2, Y) > mask) & (np.hypot(X - e4.D / 2, Y) > mask)
    rel = np.hypot(-gx - eax, -gy - eay) / np.hypot(eax, eay)
    inner = np.zeros_like(far)
    inner[2:-2, 2:-2] = True
    return float(rel[far].max()), float(rel[far & inner].max()), float(dx)


def _grid_quantiles(n: int, mask: float = 0.12) -> tuple[float, float]:
    xs = np.linspace(-0.5, 0.5, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dx = xs[1] - xs[0]
    gx, gy = np.gradient(e4.potential_pair(X, Y), dx, dx)
    eax, eay = e4.field_pair_analytic(X, Y)
    far = (np.hypot(X + e4.D / 2, Y) > mask) & (np.hypot(X - e4.D / 2, Y) > mask)
    inner = np.zeros_like(far)
    inner[2:-2, 2:-2] = True
    rel = (np.hypot(-gx - eax, -gy - eay) / np.hypot(eax, eay))[far & inner]
    return float(np.median(rel)), float(np.percentile(rel, 99))


def _pair() -> dict:
    u = e4.interaction_energy()
    ns = [51, 101, 201, 401, 801]
    g = [_grid(n) for n in ns]
    s_all = [_slope(g[i + 1][2], g[i][2], g[i + 1][0], g[i][0]) for i in range(4)]
    s_int = [_slope(g[i + 1][2], g[i][2], g[i + 1][1], g[i][1]) for i in range(4)]
    err_pub = e4.field_grid_error()
    masks = [0.03, 0.06, 0.12, 0.24]
    gm = [_grid(801, m)[1] for m in masks]
    q50, q99 = _grid_quantiles(801)

    intro = [
        "Las líneas equipotenciales se dibujan a partir de $V(x,y)=k\\sum q_i/r_i$, que es un escalar fácil de evaluar en cualquier punto. Lo que se pone a prueba es la relación $\\vec E=-\\nabla V$: si se calcula el gradiente numéricamente sobre una malla y se compara con la suma vectorial de Coulomb, se confirma que el campo es perpendicular a las equipotenciales y que ambos objetos describen el mismo sistema.",
        "El gradiente se obtiene con `np.gradient`, que usa diferencias centradas en el interior de la malla, $\\partial_xV\\approx[V_{i+1,j}-V_{i-1,j}]/(2\\Delta x)$, con error $O(\\Delta x^2)$, y diferencias de primer orden (hacia adelante o atrás) en el borde, con error $O(\\Delta x)$. La malla de $801\\times801$ en $[-0.5,0.5]$ m da $\\Delta x=1.25$ mm. El error local crece mucho cerca de las cargas, donde $V\\sim1/r$ tiene derivadas de alto orden enormes, por lo que el error se mide solo a más de $12$ cm de ellas.",
        "Hay un matiz importante: el error máximo que reporta la escena incluye el borde de la malla. En el borde `np.gradient` baja a orden 1 y ese pequeño anillo de puntos domina el máximo. En el interior de la malla el método sí es de orden 2. El estudio de abajo separa las dos contribuciones con números reales y muestra que el error reportado por la escena es una cota conservadora del que se tiene en la región donde se dibujan las equipotenciales.",
        "El método es independiente del campo analítico porque solo usa valores de $V$ en puntos de la malla y diferencias entre vecinos; la fórmula $kq\\,\\vec r/r^3$ solo se usa después, para juzgar el resultado. La energía del par, $U=kq_aq_b/d$, no requiere método numérico y por eso en la tabla de la escena aparece sin error; la verificación numérica relevante de este capítulo es el gradiente en la malla.",
    ]
    code = """def V(x, y, qa=4e-9, qb=-4e-9, d=0.20):
    ra = hypot(x + d/2, y); rb = hypot(x - d/2, y)    # distancias a las dos cargas
    return K * (qa/ra + qb/rb)                        # superposicion de potenciales

def campo_en_malla(extent=0.5, n=801):
    xs = linspace(-extent, extent, n)                 # puntos de la malla (dx = 1.25 mm)
    X, Y = meshgrid(xs, xs, indexing="ij")
    dx = xs[1] - xs[0]
    gx, gy = gradient(V(X, Y), dx, dx)                # diferencias centradas (primer orden en el borde)
    return xs, -gx, -gy                               # E = -grad V

def error_relativo(extent=0.5, n=801, mascara=0.12):
    xs, Ex, Ey = campo_en_malla(extent, n)
    Eax, Eay = campo_analitico(X, Y)                  # Coulomb vectorial, solo para comparar
    lejos = (dist_a_qa > mascara) & (dist_a_qb > mascara)   # excluye el entorno de las cargas
    err = hypot(Ex - Eax, Ey - Eay) / hypot(Eax, Eay)
    return err[lejos].max()                           # error relativo maximo
"""
    conv = [
        ("Error del gradiente frente al tamaño de malla", f"Se repite el cálculo con distintas resoluciones. La columna «toda la malla» es la métrica de la escena (para $n=801$ da {err_pub:.2e}); la columna «interior» excluye las 2 filas exteriores. Pendientes locales observadas: {', '.join(f'{s:.2f}' for s in s_all)} (toda la malla, orden 1 por el borde) y {', '.join(f'{s:.2f}' for s in s_int)} (interior, orden 2).",
         [_arr("cccc", [r"n", r"\Delta x\ (\mathrm{mm})", r"\text{toda la malla}", r"\text{interior}"],
               [[str(n), fx(x[2] * 1000, 3), sci(x[0], 3), sci(x[1], 3)] for n, x in zip(ns, g)])]),
        ("Sensibilidad a la cercanía de las cargas", "El error depende de cuánto se acerque uno a las cargas, donde el potencial cambia más rápido que la malla. Con $n=801$ y mirando solo el interior de la malla, el error máximo para distintos radios de exclusión $\\rho$ alrededor de cada carga es:",
         [_arr("cc", [r"\rho\ (\mathrm{m})", r"\text{error máximo (interior)}"], [[fx(m, 2), sci(e, 3)] for m, e in zip(masks, gm)]),
          r"\text{El error baja entre 3 y 4 veces al duplicar el radio de exclusión }\rho\text{ (comportamiento }(\Delta x/\rho)^2\text{).}"]),
        ("Error típico frente a error máximo y criterio de confianza", f"El máximo es una medida muy pesimista: está en los puntos más cercanos a las cargas. Con $n=801$ y radio de exclusión $0.12$ m, el error mediano en el interior es {sci(q50, 2)} y el percentil 99 es {sci(q99, 2)}, frente al máximo de {sci(gm[2], 2)}. Sobre el plano mediatriz $x=0$, simétrico, las distancias a las dos cargas son idénticas bit a bit y $V=0$ sale exactamente, de modo que esa fila de la tabla de la escena no pone a prueba nada numérico. Criterio práctico: el campo por gradiente es confiable si el error del interior baja unas 4 veces al duplicar $n$ y se evitan el borde y el entorno inmediato de las cargas.",
         [rf"\varepsilon_{{50\%}} = {sci(q50, 3)},\qquad \varepsilon_{{99\%}} = {sci(q99, 3)},\qquad \varepsilon_{{max}} = {sci(gm[2], 3)}",
          rf"U = k\dfrac{{q_aq_b}}{{d}} = {fx(u * 1e6, 4)}\ \mu\mathrm{{J}}\qquad(V(0,y)=0\ \text{{exacto por simetría}})"]),
    ]
    return dict(numeric_intro=intro, numeric_code=code, numeric_converge=conv)


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
