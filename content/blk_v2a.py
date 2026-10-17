"""Volumen 2, capitulos 1 a 7."""
from __future__ import annotations

import numpy as np

from content.model import Chapter, Row, Scene
from physics import v2ch01_coulomb as e1
from physics import v2ch02_field as e2
from physics import v2ch03_gauss as e3
from physics import v2ch04_potential as e4
from physics import v2ch05_capacitance as e5
from physics import v2ch06_current as e6
from physics import v2ch07_circuits as e7

PART = "Electromagnetismo"


def _ch1() -> Chapter:
    fa, fn = e1.force_analytic(), e1.force_numeric()
    xa, xn = e1.zero_force_point_analytic(), e1.zero_force_point_numeric()
    oa, on = e1.opposite_charges_point_analytic(), e1.opposite_charges_point_numeric()
    s1 = Scene(
        id="CoulombForces", title="Fuerza de Coulomb sobre una carga en el plano",
        statement=("Tres cargas puntuales: q₁ = +3 µC en (0, 0), q₂ = −2 µC en (0.4, 0) m y q₃ = +1 µC en (0.2, 0.3) m. Hallar la fuerza neta sobre q₃ y "
                   "verificar que las fuerzas internas suman cero."),
        foundations=[(r"\vec F_{12}=k\dfrac{q_1q_2}{r^2}\hat r_{12}", "Ley de Coulomb."), (r"\vec F_i=\sum_{j\ne i}\vec F_{ij},\qquad\vec F=-\nabla U", "Superposición y relación con la energía potencial.")],
        analytic=[(rf"\vec F_3=({fa[0]*1000:.3f},\ {fa[1]*1000:.3f})\ \mathrm{{mN}},\quad|\vec F_3|={np.linalg.norm(fa)*1000:.3f}\ \mathrm{{mN}}", "Suma vectorial de las dos fuerzas de Coulomb.")],
        numeric_method="Se calcula F = −∇U derivando numéricamente (diferencias centradas, h = 1 µm) la energía potencial total respecto de la posición de q₃. La suma de las fuerzas sobre las tres cargas debe anularse.",
        rows=[Row("Fx sobre q₃", "F_x", fa[0] * 1000, fn[0] * 1000, "mN", 5), Row("Fy sobre q₃", "F_y", fa[1] * 1000, fn[1] * 1000, "mN", 5),
              Row("Suma de fuerzas internas", r"\sum\vec F", 0.0, e1.newton_third_law_residual(), "N", 3)],
    )
    s2 = Scene(
        id="EquilibriumPoint", title="Punto de equilibrio entre dos cargas",
        statement=("Dos cargas de +4 µC y +1 µC están a 0.60 m. Hallar dónde debe colocarse una carga de prueba para quedar en equilibrio, y repetir el análisis si la "
                   "segunda carga fuera −1 µC."),
        foundations=[(r"F=kq_t\left(\dfrac{q_1}{x^2}-\dfrac{q_2}{(d-x)^2}\right)=0", "Condición de fuerza nula."), (r"\dfrac{x}{d-x}=\sqrt{q_1/q_2}", "Razón de distancias.")],
        analytic=[(rf"x=d\dfrac{{\sqrt{{q_1}}}}{{\sqrt{{q_1}}+\sqrt{{q_2}}}}={xa:.3f}\ \mathrm{{m}}", "Cargas del mismo signo."),
                  (rf"q_2=-1\ \mu\mathrm{{C}}:\quad x=2d={oa:.3f}\ \mathrm{{m}}", "Cargas de signo opuesto: fuera del segmento.")],
        numeric_method="Bisección sobre la fuerza neta sobre la carga de prueba (100 iteraciones) en el intervalo adecuado a cada caso.",
        rows=[Row("Equilibrio con cargas del mismo signo", "x", xa, xn, "m", 6), Row("Equilibrio con cargas de signo opuesto", "x", oa, on, "m", 6)],
    )
    return Chapter(number=1, volume=2, part=PART, slug="v2ch01", module="v2ch01_coulomb", title="Carga eléctrica y ley de Coulomb",
                   summary="Carga, aislantes y conductores, y la fuerza electrostática entre cargas puntuales.",
                   concepts=["Carga eléctrica", "Conductores y aislantes", "Ley de Coulomb", "Conservación y cuantización de la carga"], scenes=[s1, s2])


def _ch2() -> Chapter:
    q, d = e2.QD, e2.DSEP
    exa, exf = float(e2.dipole_field_axis_exact(0.5)), float(e2.dipole_field_axis_far(0.5))
    en, ev = e2.field_from_potential_numeric(0.3, 0.2), e2.field_dipole_vector(0.3, 0.2)
    za, zn = e2.ring_max_field_position_analytic(), e2.ring_max_field_position_numeric()
    s1 = Scene(
        id="DipoleField", title="Campo y líneas de campo de un dipolo",
        statement=("Un dipolo está formado por +5 nC y −5 nC separadas 2 cm. Hallar el campo sobre su eje a 0.50 m del centro con la fórmula exacta y la de "
                   "campo lejano, y trazar sus líneas de campo."),
        foundations=[(r"\vec E=k\dfrac{q}{r^2}\hat r,\qquad\vec E=-\nabla V", "Campo de una carga puntual."), (r"E_{eje}(z\gg d)\approx\dfrac{2kp}{z^3},\quad p=qd", "Campo lejano del dipolo.")],
        analytic=[(rf"E_{{exacto}}=kq\left[\dfrac1{{(z-d/2)^2}}-\dfrac1{{(z+d/2)^2}}\right]={exa:.5f}\ \mathrm{{N/C}}", "A 0.50 m sobre el eje."),
                  (rf"E_{{lejano}}=\dfrac{{2kp}}{{z^3}}={exf:.5f}\ \mathrm{{N/C}}", "Aproximación dipolar.")],
        numeric_method="El campo en (0.3, 0.2) m se calcula como −∇V con diferencias centradas y se compara con la suma vectorial de Coulomb; las líneas de campo se integran con RK4 sobre dr/ds = E/|E| desde la carga positiva.",
        rows=[Row("E sobre el eje (campo lejano vs exacto)", "E", exa, exf, "N/C", 5), Row("Ex en (0.3, 0.2) m", "E_x", ev[0], en[0], "N/C", 4), Row("Ey en (0.3, 0.2) m", "E_y", ev[1], en[1], "N/C", 4)],
        notes=["En la primera fila la columna numérica es la aproximación dipolar: la diferencia es el error de truncar el desarrollo en d/z."],
    )
    s2 = Scene(
        id="ChargedRing", title="Campo de un anillo cargado",
        statement="Un anillo de radio 10 cm tiene una carga total de 10 nC repartida uniformemente. Hallar E sobre su eje y la posición donde es máximo.",
        foundations=[(r"d\vec E=k\dfrac{dq}{r^2}\hat r,\qquad\vec E=\int d\vec E", "Distribución continua."), (r"E_z=\dfrac{kQz}{(z^2+a^2)^{3/2}}", "Campo sobre el eje.")],
        analytic=[(rf"z_{{max}}=\dfrac a{{\sqrt2}}={za*100:.3f}\ \mathrm{{cm}}", "Máximo de E_z."), (rf"E_{{max}}={float(e2.ring_axis_field_analytic(za)):.1f}\ \mathrm{{N/C}}", "Valor en el máximo.")],
        numeric_method="Suma vectorial de 400 elementos dq del anillo con Simpson sobre el ángulo (se comprueba que las componentes perpendiculares se cancelan); el máximo se localiza con un ajuste parabólico.",
        rows=[Row("E_z a 5 cm", "E_z", float(e2.ring_axis_field_analytic(0.05)), e2.ring_axis_field_numeric(0.05), "N/C", 3),
              Row("E_z a 30 cm", "E_z", float(e2.ring_axis_field_analytic(0.30)), e2.ring_axis_field_numeric(0.30), "N/C", 3),
              Row("Posición del máximo", r"z_{max}", za * 100, zn * 100, "cm", 3)],
    )
    return Chapter(number=2, volume=2, part=PART, slug="v2ch02", module="v2ch02_field", title="El campo eléctrico",
                   summary="Líneas de campo y cálculo del campo de cargas puntuales y distribuciones continuas.",
                   concepts=["Campo eléctrico", "Líneas de campo", "Campo de un dipolo", "Campo de distribuciones continuas de carga"], scenes=[s1, s2])


def _ch3() -> Chapter:
    phi = e3.flux_sphere_analytic()
    mc = {r: e3.sphere_field_monte_carlo(r, n=200_000) for r in (0.05, 0.10, 0.20)}
    s1 = Scene(
        id="GaussFlux", title="Flujo de una carga puntual a través de superficies cerradas",
        statement=("Una carga de 2 nC se rodea de una esfera de 0.3 m y de un cubo de 0.4 m de lado. Calcular el flujo en cada caso con la carga centrada, "
                   "descentrada y fuera de la superficie."),
        foundations=[(r"\Phi_E=\oint\vec E\cdot d\vec A=\dfrac{q_{enc}}{\varepsilon_0}", "Ley de Gauss.")],
        analytic=[(rf"\Phi_E=\dfrac q{{\varepsilon_0}}={phi:.2f}\ \mathrm{{N\,m^2/C}}", "Cualquier superficie que encierre la carga."), (r"\Phi_E=0", "Carga fuera de la superficie.")],
        numeric_method="Integración de superficie de E·n dA con la regla del punto medio: 200×400 celdas (θ, φ) en la esfera y seis caras de 400×400 en el cubo.",
        rows=[Row("Esfera, carga centrada", r"\Phi", phi, e3.flux_sphere_numeric(), "N m²/C", 3), Row("Esfera, carga descentrada", r"\Phi", phi, e3.flux_sphere_numeric(charge_pos=(0.1, 0.05, 0.0)), "N m²/C", 3),
              Row("Cubo", r"\Phi", phi, e3.flux_cube_numeric(), "N m²/C", 3), Row("Carga fuera de la esfera", r"\Phi", 0.0, e3.flux_sphere_numeric(radius=0.1, charge_pos=(0.3, 0.0, 0.0)), "N m²/C", 6)],
    )
    s2 = Scene(
        id="SphereFieldProfile", title="Campo de una esfera aislante uniformemente cargada",
        statement=("Una esfera aislante de radio 10 cm tiene una carga uniforme de 5 nC. Hallar E(r) dentro y fuera, y verificarlo sumando el campo de cargas puntuales "
                   "repartidas al azar en el volumen."),
        foundations=[(r"E_{in}=\dfrac{kQ\,r}{R^3},\qquad E_{out}=\dfrac{kQ}{r^2}", "Resultado de la ley de Gauss con simetría esférica.")],
        analytic=[(rf"E(R)=\dfrac{{kQ}}{{R^2}}={float(e3.sphere_field_analytic(0.1)):.1f}\ \mathrm{{N/C}}", "Valor máximo, en la superficie.")],
        numeric_method="Monte Carlo: el campo en cada punto es la suma coulombiana de 200 000 cargas puntuales muestreadas uniformemente en el volumen; el error es la desviación estándar de la media.",
        rows=[Row(f"E a r = {r:.2f} m", "E", float(e3.sphere_field_analytic(r)), mc[r][0], "N/C", 1) for r in (0.05, 0.10, 0.20)],
        notes=[f"Incertidumbres estadísticas del Monte Carlo: " + ", ".join(f"±{mc[r][1]:.1f} N/C a {r:.2f} m" for r in (0.05, 0.10, 0.20)) + "."],
    )
    return Chapter(number=3, volume=2, part=PART, slug="v2ch03", module="v2ch03_gauss", title="Ley de Gauss",
                   summary="Flujo eléctrico, ley de Gauss y su uso en distribuciones con simetría.",
                   concepts=["Flujo del campo eléctrico", "Ley de Gauss", "Aplicaciones de la ley de Gauss", "Conductores en equilibrio"], scenes=[s1, s2])


def _ch4() -> Chapter:
    z0 = 0.05
    v_an, v_nu = float(e4.disk_potential_analytic(z0)), e4.disk_potential_numeric(z0)
    e_an = float(e4.disk_field_analytic(z0))
    e_nu = e4.field_from_potential_numeric(lambda z: float(e4.disk_potential_analytic(z)), z0)
    u = e4.interaction_energy()
    s1 = Scene(
        id="DiskPotential", title="Potencial y campo de un disco cargado",
        statement=("Un disco de radio 10 cm tiene σ = 2 µC/m². Hallar V y E sobre su eje a 5 cm del centro sumando anillos concéntricos y derivando después."),
        foundations=[(r"V=k\int\dfrac{dq}{r},\qquad\vec E=-\nabla V", "Potencial de una distribución continua."), (r"V(z)=\dfrac\sigma{2\varepsilon_0}\left(\sqrt{z^2+R^2}-z\right)", "Disco uniforme.")],
        analytic=[(rf"V(5\ \mathrm{{cm}})={v_an/1000:.3f}\ \mathrm{{kV}}", "Potencial."), (rf"E_z(5\ \mathrm{{cm}})={e_an/1000:.3f}\ \mathrm{{kV/m}}", "Campo (derivada con signo cambiado).")],
        numeric_method="Simpson (400 subintervalos) sobre anillos dq = σ 2π s ds para V; E = −dV/dz con diferencias centradas.",
        rows=[Row("Potencial a 5 cm", "V", v_an, v_nu, "V", 4), Row("Campo a 5 cm", "E_z", e_an, e_nu, "V/m", 3)],
    )
    s2 = Scene(
        id="EquipotentialPair", title="Equipotenciales de un par de cargas opuestas",
        statement=("Dos cargas de +4 nC y −4 nC están a 20 cm. Trazar las equipotenciales, calcular E = −∇V en una malla y compararlo con el campo coulombiano; "
                   "hallar la energía almacenada en el par."),
        foundations=[(r"V=k\sum\dfrac{q_i}{r_i},\qquad U=k\dfrac{q_aq_b}d", "Potencial y energía de un par."), (r"\vec E\perp\text{equipotenciales}", "Geometría del campo.")],
        analytic=[(rf"U=k\dfrac{{q_aq_b}}{{d}}={u*1e6:.3f}\ \mu\mathrm{{J}}", "Energía del par.")],
        numeric_method="V se evalúa en una malla de 801×801 y E se obtiene con np.gradient (diferencias centradas); el error se mide frente al campo exacto a más de 12 cm de las cargas.",
        rows=[Row("Energía de interacción", "U", u * 1e6, e4.interaction_energy() * 1e6, "µJ", 4), Row("Potencial en el plano mediatriz", "V", 0.0, float(e4.potential_pair(0.0, 0.1)), "V", 6),
              Row("Error relativo máximo de −∇V", r"\varepsilon", 0.0, e4.field_grid_error(), "", 4)],
    )
    return Chapter(number=4, volume=2, part=PART, slug="v2ch04", module="v2ch04_potential", title="Potencial eléctrico",
                   summary="Energía potencial eléctrica, diferencia de potencial y su relación con el campo.",
                   concepts=["Energía potencial eléctrica", "Potencial eléctrico", "Cálculo del potencial", "Equipotenciales", "Cálculo del campo a partir del potencial"], scenes=[s1, s2])


def _ch5() -> Chapter:
    c_an = e5.coax_capacitance_per_length_analytic()
    c_nu = e5.coax_capacitance_numeric(n=161, iterations=4000)
    nd = e5.network_nodal_numeric()
    ceq = e5.network_ceq_analytic()
    d = e5.dielectric_energies()
    s1 = Scene(
        id="CoaxialLaplace", title="Capacitancia de un cable coaxial por relajación",
        statement=("Un cable coaxial tiene un conductor interno de radio 0.5 cm y uno externo de 2 cm. Hallar su capacitancia por unidad de longitud resolviendo la "
                   "ecuación de Laplace en una malla y calculándola a partir de la energía almacenada."),
        foundations=[(r"C=\dfrac Q{\Delta V},\qquad U=\tfrac12C\Delta V^2=\tfrac12\varepsilon_0\int E^2\,dV", "Capacitancia y energía."), (r"\nabla^2V=0", "Región sin cargas.")],
        analytic=[(rf"C'=\dfrac{{2\pi\varepsilon_0}}{{\ln(b/a)}}={c_an*1e12:.2f}\ \mathrm{{pF/m}}", "Capacitancia por unidad de longitud.")],
        numeric_method="Relajación sobremuestreada (SOR rojo-negro) de la ecuación de Laplace en una malla de 161×161 con 4000 iteraciones; C' = 2U/V₀² con U = ½ε₀∫|E|² dA calculada con np.gradient.",
        rows=[Row("Capacitancia por unidad de longitud", "C'", c_an * 1e12, c_nu * 1e12, "pF/m", 3)],
        notes=[f"El error relativo ({abs(c_nu - c_an) / c_an * 100:.1f} %) proviene de la discretización de las fronteras circulares en una malla cartesiana."],
    )
    s2 = Scene(
        id="CapacitorNetwork", title="Red de capacitores y dieléctricos",
        statement=("Un capacitor de 2 µF está en serie con dos de 3 µF y 6 µF en paralelo, conectados a 12 V. Hallar C_eq, la carga y la energía. Luego insertar un "
                   "dieléctrico κ = 4 en un capacitor de 2 µF con y sin batería."),
        foundations=[(r"C_{par}=\sum C_i,\qquad\dfrac1{C_{ser}}=\sum\dfrac1{C_i}", "Combinaciones."), (r"C=\kappa C_0,\quad U=\dfrac{Q^2}{2C}=\tfrac12CV^2", "Dieléctrico y energía.")],
        analytic=[(rf"C_{{eq}}=\dfrac{{2\cdot9}}{{2+9}}={ceq*1e6:.4f}\ \mu\mathrm{{F}}", "Serie con la combinación en paralelo."), (rf"Q={ceq*12*1e6:.2f}\ \mu\mathrm{{C}},\quad U={0.5*ceq*144*1e6:.1f}\ \mu\mathrm{{J}}", "Carga y energía.")],
        numeric_method="Conservación de la carga en el nodo intermedio resuelta como sistema lineal; energías con dieléctrico a V constante (U = κU₀) y a Q constante (U = U₀/κ).",
        rows=[Row("Capacitancia equivalente", r"C_{eq}", ceq * 1e6, nd["c_eq"] * 1e6, "µF", 5), Row("Potencial del nodo", r"V_x", 12 * 2 / 11, nd["vx"], "V", 5),
              Row("Energía almacenada", "U", 0.5 * ceq * 144 * 1e6, nd["energy"] * 1e6, "µJ", 3),
              Row("Energía con dieléctrico a V constante", "U", 4 * d["u0"] * 1e6, d["u_const_v"] * 1e6, "µJ", 3), Row("Energía con dieléctrico a Q constante", "U", d["u0"] * 1e6 / 4, d["u_const_q"] * 1e6, "µJ", 3)],
    )
    return Chapter(number=5, volume=2, part=PART, slug="v2ch05", module="v2ch05_capacitance", title="Capacitancia y dieléctricos",
                   summary="Capacitores, combinaciones en serie y en paralelo, energía almacenada y dieléctricos.",
                   concepts=["Capacitancia", "Cálculo de la capacitancia", "Capacitores en serie y en paralelo", "Energía almacenada", "Dieléctricos"], scenes=[s1, s2])


def _ch6() -> Chapter:
    mu_n, sig, _ = e6.drift_mobility_monte_carlo()
    mu_a = e6.drift_mobility_analytic()
    ra, rn, ru = e6.tapered_resistance_analytic(), e6.tapered_resistance_numeric(), e6.uniform_resistance()
    s1 = Scene(
        id="DrudeElectrons", title="Modelo de Drude: velocidad de deriva",
        statement=("En cobre (n = 8.5×10²⁸ m⁻³, τ = 2.5×10⁻¹⁴ s) hay un campo de 100 V/m. Hallar la velocidad de deriva y la conductividad, y verificar la movilidad "
                   "v_d/E con una simulación Monte Carlo de 20 000 electrones."),
        foundations=[(r"J=nqv_d=\sigma E", "Densidad de corriente."), (r"v_d=\dfrac{eE\tau}m,\qquad\sigma=\dfrac{ne^2\tau}m", "Modelo de Drude.")],
        analytic=[(rf"v_d={e6.drift_speed_analytic():.4f}\ \mathrm{{m/s}}", "Con E = 100 V/m."), (rf"\sigma={e6.conductivity_analytic()/1e7:.3f}\times10^7\ \mathrm{{S/m}}", "Conductividad.")],
        numeric_method="Monte Carlo: cada electrón acelera con a = −eE/m y sufre colisiones de Poisson (probabilidad dt/τ por paso, dt = τ/50) que reinician su velocidad térmica. Se usa E = 2×10⁶ V/m para superar el ruido (el modelo es lineal en E) y se compara la movilidad μ = v_d/E.",
        rows=[Row("Movilidad μ = v_d/E", r"\mu", mu_a * 1e3, mu_n * 1e3, "10⁻³ m²/(V s)", 3), Row("Velocidad de deriva a 100 V/m", "v_d", e6.drift_speed_analytic(), mu_n * 100, "m/s", 4)],
        notes=[f"Incertidumbre estadística de la movilidad: ±{sig * 1e3:.3f}×10⁻³ m²/(V s)."],
    )
    s2 = Scene(
        id="TaperedWire", title="Resistencia de un cable de sección variable",
        statement=("Un cable de cobre (ρ = 1.68×10⁻⁸ Ω·m) de 2 m tiene un radio que crece linealmente de 0.5 mm a 1.5 mm. Hallar su resistencia por integración y "
                   "compararla con la de un cable uniforme de 0.5 mm."),
        foundations=[(r"R=\rho\dfrac LA,\qquad R=\int_0^L\dfrac{\rho\,dx}{A(x)}", "Resistencia de un conductor no uniforme.")],
        analytic=[(rf"R=\dfrac{{\rho L}}{{\pi r_1r_2}}={ra*1000:.3f}\ \mathrm{{m\Omega}}", "Cable cónico."), (rf"R_{{uniforme}}={ru*1000:.3f}\ \mathrm{{m\Omega}}", "Cable de 0.5 mm.")],
        numeric_method="Regla de Simpson (400 subintervalos) sobre dR/dx = ρ/(π r(x)²).",
        rows=[Row("Resistencia del cable cónico", "R", ra * 1000, rn * 1000, "mΩ", 6)],
    )
    return Chapter(number=6, volume=2, part=PART, slug="v2ch06", module="v2ch06_current", title="Corriente y resistencia",
                   summary="Corriente eléctrica, densidad de corriente, ley de Ohm y modelos de conducción.",
                   concepts=["Corriente eléctrica", "Densidad de corriente", "Resistencia y resistividad", "Ley de Ohm", "Modelo clásico de la conducción"], scenes=[s1, s2])


def _ch7() -> Chapter:
    i_a, i_n = e7.mesh_currents_analytic(), e7.mesh_currents_numeric()
    res = e7.kirchhoff_residuals(i_n)
    en = e7.rc_energies_numeric()
    eps, C, R = e7.EPS, e7.C, e7.R
    s1 = Scene(
        id="MeshCircuit", title="Circuito de dos mallas: leyes de Kirchhoff",
        statement=("Un circuito de dos mallas tiene ε₁ = 12 V con R₁ = 4 Ω, ε₂ = 6 V con R₂ = 2 Ω y R₃ = 6 Ω compartida. Hallar las corrientes de malla y de rama y "
                   "verificar el balance de potencia."),
        foundations=[(r"\sum I_{nodo}=0,\qquad\sum\Delta V_{malla}=0", "Leyes de Kirchhoff.")],
        analytic=[(r"(R_1+R_3)i_1-R_3i_2=\varepsilon_1,\quad-R_3i_1+(R_2+R_3)i_2=-\varepsilon_2", "Ecuaciones de malla."),
                  (rf"i_1={i_a[0]:.4f}\ \mathrm{{A}},\quad i_2={i_a[1]:.4f}\ \mathrm{{A}}", "Corrientes de malla.")],
        numeric_method="Sistema lineal 2×2 con numpy.linalg.solve; se evalúan los residuos de ambas mallas y el balance de potencia entre las fuentes y las resistencias.",
        rows=[Row("Corriente de malla 1", "i_1", i_a[0], i_n[0], "A", 6), Row("Corriente de malla 2", "i_2", i_a[1], i_n[1], "A", 6),
              Row("Residuo de la malla izquierda", r"\sum\Delta V", 0.0, res["loop_left"], "V", 3), Row("Balance de potencia", r"\Delta P", 0.0, res["power_balance"], "W", 3)],
    )
    s2 = Scene(
        id="RCCharging", title="Carga de un capacitor en un circuito RC",
        statement="Un capacitor de 100 µF se carga con 12 V a través de 2 kΩ. Hallar q(t), la constante de tiempo y el balance de energía.",
        foundations=[(r"R\dfrac{dq}{dt}=\varepsilon-\dfrac qC", "Ecuación del circuito."), (r"q=C\varepsilon(1-e^{-t/RC})", "Solución.")],
        analytic=[(rf"\tau=RC={R*C:.3f}\ \mathrm{{s}},\quad q_\infty={C*eps*1e3:.2f}\ \mathrm{{mC}}", "Constante de tiempo y carga final."),
                  (rf"E_{{bat}}={C*eps**2*1e3:.2f}\ \mathrm{{mJ}},\quad E_{{cap}}=E_{{dis}}={0.5*C*eps**2*1e3:.2f}\ \mathrm{{mJ}}", "Energías.")],
        numeric_method="RK4 de dq/dt = (ε − q/C)/R con 2400 pasos; las energías se calculan integrando I²R y εI con Simpson sobre 3 s.",
        rows=[Row("Tiempo para q = 63.2 % de q_∞", "t", -R * C * np.log(1 - 0.632), e7.time_to_fraction_numeric(), "s", 5),
              Row("Energía disipada en la resistencia", r"E_{dis}", 0.5 * C * eps**2 * 1e3, en["dissipated"] * 1e3, "mJ", 4),
              Row("Energía en el capacitor", r"E_{cap}", 0.5 * C * eps**2 * 1e3, en["capacitor"] * 1e3, "mJ", 4),
              Row("Energía entregada por la batería", r"E_{bat}", C * eps**2 * 1e3, en["battery"] * 1e3, "mJ", 4)],
    )
    return Chapter(number=7, volume=2, part=PART, slug="v2ch07", module="v2ch07_circuits", title="Circuitos de corriente directa",
                   summary="Leyes de Kirchhoff, fuerza electromotriz y circuitos RC.",
                   concepts=["Fuerza electromotriz", "Cálculo de la corriente en un circuito", "Leyes de Kirchhoff", "Circuitos RC"], scenes=[s1, s2])


def chapters() -> list[Chapter]:
    return [_ch1(), _ch2(), _ch3(), _ch4(), _ch5(), _ch6(), _ch7()]
