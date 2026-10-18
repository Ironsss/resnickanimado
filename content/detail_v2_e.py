"""Explicacion y pasos a mano. Volumen 2, capitulos 19 a 22 (fotones, ondas de materia, Schrodinger, atomo).

Todas las cifras se calculan con las funciones de `physics/` o replicando su algoritmo en pasos pequenos.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from content.texfmt import fx, sci
from physics import v2ch19_photons as p19
from physics import v2ch20_matter_waves as p20
from physics import v2ch21_schrodinger as p21
from physics import v2ch22_atom as p22
from physics.constants import C_LIGHT, E_CH, H_PLANCK, HBAR, K_B, M_E, SIGMA_SB


def _cz(z: complex, d: int = 4) -> str:
    """Numero complejo en TeX."""
    re, im = float(z.real), float(z.imag)
    sg = "+" if im >= 0 else "-"
    return f"({sci(re, d)} {sg} {sci(abs(im), d)}\\,i)"


# =============================================================================== BlackbodySpectrum
def _blackbody() -> dict:
    t = 5800.0
    b = p19.WIEN_B
    lam_max = p19.wien_peak_analytic(t)
    xw = H_PLANCK * C_LIGHT / (b * K_B)
    sig = 2 * np.pi**5 * K_B**4 / (15 * H_PLANCK**3 * C_LIGHT**2)
    pa = p19.stefan_analytic(t)
    pn = p19.stefan_numeric(t)
    lam3 = p19.wien_peak_analytic(3000.0)

    # seccion aurea replicada
    lo, hi = 0.2 * b / t, 5 * b / t
    g = (np.sqrt(5) - 1) / 2
    c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    hist = []
    for _ in range(6):
        pc, pe = float(p19.planck(c, t)), float(p19.planck(e, t))
        hist.append((lo, hi, c, e, pc, pe))
        if pc > pe:
            hi = e
        else:
            lo = c
        c, e = hi - g * (hi - lo), lo + g * (hi - lo)
    lo0, hi0 = 0.2 * b / t, 5 * b / t
    n = 20000
    a_l, b_l = 0.02 * lam_max, 200 * lam_max
    hs = (b_l - a_l) / n
    lam = np.linspace(a_l, b_l, n + 1)
    f = p19.planck(lam, t)
    sw = f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum()
    tail = 2 * np.pi * C_LIGHT * K_B * t / (3 * b_l**3)
    rows = "\\\\".join(
        rf"{i + 1} & {h[0] * 1e9:.4f} & {h[1] * 1e9:.4f} & {h[2] * 1e9:.4f} & {h[3] * 1e9:.4f} & {'c' if h[4] > h[5] else 'e'}" for i, h in enumerate(hist)
    )
    h0 = hist[0]
    return dict(
        explanation=[
            "Un cuerpo negro ideal absorbe toda la radiación que le llega y, en equilibrio térmico, emite un espectro que depende únicamente de su temperatura, no de su material. Esa universalidad lo convierte en el modelo natural para el Sol, una estufa o un filamento incandescente. La física clásica predecía que la emisión crecía sin límite hacia las longitudes de onda cortas (la llamada catástrofe ultravioleta).",
            "Planck resolvió el problema suponiendo que cada modo de la cavidad solo intercambia energía en paquetes $E = hf$. Los modos de alta frecuencia quedan entonces suprimidos por el factor $1/(e^{hc/\\lambda kT}-1)$, que se vuelve exponencialmente pequeño cuando $hc/\\lambda \\gg kT$. El resultado es una curva con un único máximo.",
            "Dos consecuencias se obtienen integrando o derivando la curva. La ley de Wien, $\\lambda_{max}T = b$, dice que el máximo se desplaza hacia el azul al calentar el cuerpo. La ley de Stefan-Boltzmann, $P/A = \\sigma T^4$, dice que la potencia emitida por unidad de área crece como la cuarta potencia de la temperatura.",
            "Como orden de magnitud, a 5800 K (la superficie del Sol) el máximo cae cerca de 500 nm, en el centro del visible, y cada metro cuadrado radia decenas de megavatios. A 3000 K, la temperatura de una bombilla, el máximo está en el infrarrojo cercano y por eso gran parte de la energía no se ve.",
        ],
        analytic_steps=[
            ("Escribir la ley de Planck con una variable adimensional", "Se define $x = hc/(\\lambda kT)$ para que la forma de la curva no dependa de las constantes. En términos de $x$ la emitancia espectral por unidad de longitud de onda es proporcional a $x^5/(e^x-1)$.",
             [r"u(\lambda,T) = \dfrac{2\pi hc^2}{\lambda^5}\,\dfrac{1}{e^{hc/\lambda kT}-1},\qquad x = \dfrac{hc}{\lambda kT}\ \Rightarrow\ u = \dfrac{2\pi k^5T^5}{h^4c^3}\,\dfrac{x^5}{e^x-1}"]),
            ("Anular la derivada para hallar el máximo", "Se deriva $u$ respecto de $\\lambda$ (equivale a derivar $x^5/(e^x-1)$ respecto de $x$) y se iguala a cero. Queda una ecuación trascendente que se resuelve una sola vez, con un valor universal de $x$.",
             [r"\dfrac{d}{dx}\left[\dfrac{x^5}{e^x-1}\right]=0\ \Rightarrow\ (5-x)\,e^x = 5",
              rf"x_W = {xw:.6f}\qquad\text{{(comprobación: }}(5-x_W)e^{{x_W}} = {(5 - xw) * np.exp(xw):.6f}\text{{)}}"]),
            ("Obtener la constante de Wien", "De $x_W = hc/(\\lambda_{max}kT)$ se despeja el producto $\\lambda_{max}T$, que solo contiene constantes fundamentales.",
             [r"b = \lambda_{max}T = \dfrac{hc}{x_W\,k}",
              rf"b = \dfrac{{({sci(H_PLANCK, 6)})({sci(C_LIGHT, 6)})}}{{({xw:.6f})({sci(K_B, 6)})}} = {sci(b, 6)}\ \mathrm{{m\,K}}"]),
            ("Longitud de onda del máximo a 5800 K", "Se divide $b$ entre la temperatura. El mismo cálculo a 3000 K muestra cómo el máximo se mueve al infrarrojo.",
             [rf"\lambda_{{max}} = \dfrac{{b}}{{T}} = \dfrac{{{sci(b, 6)}\ \mathrm{{m\,K}}}}{{{t:.0f}\ \mathrm{{K}}}} = {sci(lam_max, 5)}\ \mathrm{{m}} = {lam_max * 1e9:.4f}\ \mathrm{{nm}}",
              rf"\lambda_{{max}}(3000\ \mathrm{{K}}) = \dfrac{{{sci(b, 6)}}}{{3000}} = {lam3 * 1e9:.4f}\ \mathrm{{nm}}"]),
            ("Integrar sobre todas las longitudes de onda", "La potencia por área es el área bajo la curva. Con la sustitución $x = hc/(\\lambda kT)$ la integral se reduce a $\\int_0^\\infty x^3/(e^x-1)\\,dx = \\pi^4/15$, un número exacto.",
             [r"\dfrac{P}{A} = \int_0^\infty u\,d\lambda = \dfrac{2\pi k^4T^4}{h^3c^2}\int_0^\infty \dfrac{x^3}{e^x-1}\,dx = \dfrac{2\pi^5k^4}{15\,h^3c^2}\,T^4"]),
            ("Calcular la constante de Stefan-Boltzmann", "Se sustituyen $h$, $c$ y $k$ en el prefactor de la integral anterior.",
             [rf"\sigma = \dfrac{{2\pi^5 ({sci(K_B, 6)})^4}}{{15\,({sci(H_PLANCK, 6)})^3({sci(C_LIGHT, 6)})^2}} = {sci(sig, 6)}\ \mathrm{{W/(m^2K^4)}}"]),
            ("Potencia por unidad de área a 5800 K", "Se multiplica $\\sigma$ por $T^4$. El valor concuerda con el de la fila de verificación.",
             [rf"\dfrac{{P}}{{A}} = \sigma T^4 = ({sci(SIGMA_SB, 6)})({t:.0f})^4 = {sci(pa, 5)}\ \mathrm{{W/m^2}} = {pa / 1e6:.4f}\ \mathrm{{MW/m^2}}"]),
        ],
        numeric_steps=[
            ("Máximo por sección áurea: intervalo y razón", "Como $u(\\lambda)$ tiene un único máximo (es unimodal), se acorta un intervalo que lo contiene comparando dos puntos interiores. Se parte de $[0.2\\,b/T,\\ 5\\,b/T]$ y se usa $g = (\\sqrt5-1)/2$ para que un punto interior se reutilice.",
             [rf"[\lambda_{{lo}},\lambda_{{hi}}] = [{lo0 * 1e9:.4f},\ {hi0 * 1e9:.4f}]\ \mathrm{{nm}},\qquad g = {g:.6f}",
              r"c = \lambda_{hi} - g(\lambda_{hi}-\lambda_{lo}),\qquad e = \lambda_{lo} + g(\lambda_{hi}-\lambda_{lo})"]),
            ("Primera iteración", "Se calculan los dos puntos interiores y se evalúa Planck en cada uno. Si $u(c) > u(e)$ el máximo no puede estar a la derecha de $e$, y se hace $\\lambda_{hi}=e$; si no, $\\lambda_{lo}=c$.",
             [rf"c = {h0[2] * 1e9:.4f}\ \mathrm{{nm}},\qquad e = {h0[3] * 1e9:.4f}\ \mathrm{{nm}}",
              rf"u(c) = {sci(h0[4], 5)}\ \mathrm{{W/m^3}},\qquad u(e) = {sci(h0[5], 5)}\ \mathrm{{W/m^3}}",
              rf"u(c){'>' if h0[4] > h0[5] else '<'}u(e)\ \Rightarrow\ \text{{{'se descarta [e, hi]' if h0[4] > h0[5] else 'se descarta [lo, c]'}}},\quad [\lambda_{{lo}},\lambda_{{hi}}] = [{hist[1][0] * 1e9:.4f},\ {hist[1][1] * 1e9:.4f}]\ \mathrm{{nm}}"]),
            ("Las primeras seis iteraciones", "El intervalo se reduce por el factor $g \\approx 0.618$ en cada vuelta. La tabla muestra los extremos y los dos puntos interiores (en nm); la última columna indica qué punto tiene mayor $u$.",
             [r"\begin{array}{c|ccccc} k & \lambda_{lo} & \lambda_{hi} & c & e & u\ \text{mayor}\\ \hline " + rows + r"\end{array}"]),
            ("Criterio de parada y resultado", "El código hace 200 iteraciones, de modo que el ancho inicial se multiplica por $g^{200}$, mucho menor que la resolución de un número de doble precisión; en la práctica el intervalo deja de cambiar mucho antes. El resultado es el centro del intervalo final.",
             [rf"g^{{200}}(\lambda_{{hi}}-\lambda_{{lo}})_0 \approx {sci(g**200 * (hi0 - lo0), 2)}\ \mathrm{{m}}",
              rf"\lambda_{{max}}^{{num}} = {p19.wien_peak_numeric(t) * 1e9:.7f}\ \mathrm{{nm}}\ \ (\text{{analítico }}{lam_max * 1e9:.7f}\ \mathrm{{nm}})"]),
            ("Potencia: regla de Simpson, malla", "La integral de Planck se calcula con Simpson en $N = 20000$ subintervalos entre $0.02\\lambda_{max}$ y $200\\lambda_{max}$, que deja fuera solo colas despreciables. El paso es constante.",
             [rf"a = 0.02\lambda_{{max}} = {sci(a_l, 4)}\ \mathrm{{m}},\quad b = 200\lambda_{{max}} = {sci(b_l, 4)}\ \mathrm{{m}},\quad h = \dfrac{{b-a}}{{N}} = {sci(hs, 4)}\ \mathrm{{m}}"]),
            ("Primeros términos de Simpson", "La fórmula pesa los extremos con 1, los nodos impares con 4 y los pares interiores con 2: $\\int \\approx \\tfrac h3\\,[f_0 + 4f_1 + 2f_2 + 4f_3 + \\dots + f_N]$. Los primeros valores de $u$ son diminutos porque la exponencial suprime longitudes de onda muy cortas, y el aporte importante viene del entorno del máximo.",
             [rf"f_0 = u(a) = {sci(f[0], 4)},\quad f_1 = {sci(f[1], 4)},\quad f_2 = {sci(f[2], 4)}\ \mathrm{{W/m^3}}",
              rf"\text{{suma ponderada}} = f_0 + 4f_1 + 2f_2 + \dots + f_N = {sci(sw, 5)}\ \mathrm{{W/m^3}}",
              rf"\dfrac{{P}}{{A}} \approx \dfrac{{h}}{{3}}\,(\text{{suma}}) = \dfrac{{{sci(hs, 4)}}}{{3}}\,({sci(sw, 5)}) = {sci(pn, 6)}\ \mathrm{{W/m^2}}"]),
            ("Comparación con Stefan-Boltzmann", "La diferencia relativa con $\\sigma T^4$ es pequeña y negativa: la integral numérica queda corta porque falta la cola de longitudes de onda mayores que $200\\lambda_{max}$, donde $u \\approx 2\\pi ckT/\\lambda^4$ (régimen de Rayleigh-Jeans).",
             [rf"\dfrac{{\Delta(P/A)}}{{P/A}} = \dfrac{{{sci(pn, 6)} - {sci(pa, 6)}}}{{{sci(pa, 6)}}} = {sci((pn - pa) / pa, 3)}",
              rf"\int_{{200\lambda_{{max}}}}^\infty \dfrac{{2\pi ckT}}{{\lambda^4}}\,d\lambda = \dfrac{{2\pi ckT}}{{3\,(200\lambda_{{max}})^3}} = {sci(tail, 3)}\ \mathrm{{W/m^2}}\ \ \Rightarrow\ \ \dfrac{{\text{{cola}}}}{{P/A}} = {sci(tail / pa, 3)}"]),
        ],
    )


# =============================================================================== ComptonPhotoelectric
def _compton() -> dict:
    lam0 = 71.0e-12
    e0 = H_PLANCK * C_LIGHT / lam0
    mc2 = M_E * C_LIGHT**2
    lc = p19.LAMBDA_C
    d90 = float(p19.compton_shift_analytic(np.pi / 2))
    d180 = float(p19.compton_shift_analytic(np.pi))
    n90 = p19.compton_shift_numeric(np.pi / 2)
    n180 = p19.compton_shift_numeric(np.pi)
    e1_90 = H_PLANCK * C_LIGHT / (lam0 + d90)
    keV = E_CH * 1e3

    def resid(e1, th):
        p0, p1 = e0 / C_LIGHT, e1 / C_LIGHT
        pe2 = p0**2 + p1**2 - 2 * p0 * p1 * np.cos(th)
        return (e0 + mc2 - e1) ** 2 - (pe2 * C_LIGHT**2 + mc2**2)

    th = np.pi / 2
    lo, hi = e0 * 1e-6, e0
    hist = []
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        rl, rm = resid(lo, th), resid(mid, th)
        hist.append((lo, hi, mid, rl, rm))
        if rl * rm <= 0:
            hi = mid
        else:
            lo = mid
    scale = keV**2
    tab = "\\\\".join(
        rf"{i + 1} & {h[0] / keV:.4f} & {h[1] / keV:.4f} & {h[2] / keV:.4f} & {h[3] / scale:.2f} & {h[4] / scale:.2f}" for i, h in enumerate(hist)
    )
    # ajuste lineal replicado
    rng = np.random.default_rng(3)
    fr = np.linspace(6.0e14, 12.0e14, 25)
    v = p19.stopping_potential_analytic(fr) + rng.normal(0, 0.01, fr.size)
    f14 = fr / 1e14
    n = f14.size
    sx, sy, sxx, sxy = f14.sum(), v.sum(), (f14**2).sum(), (f14 * v).sum()
    slope = (n * sxy - sx * sy) / (n * sxx - sx**2)
    icpt = (sy - slope * sx) / n
    h_fit = slope * E_CH / 1e14
    phi_fit = -icpt
    hh, pp, dh, dpp = p19.millikan_fit()
    resid_fit = v - (slope * f14 + icpt)
    dat = "\\\\".join(rf"{i + 1} & {fr[i] / 1e14:.4f} & {p19.stopping_potential_analytic(fr[i]):.4f} & {v[i]:.4f}" for i in (0, 1, 2))
    f0 = p19.F_THRESH
    return dict(
        explanation=[
            "El efecto fotoeléctrico y el efecto Compton son las dos pruebas clásicas de que la luz intercambia energía y momento en cuantos localizados (fotones) de energía $E = hf$ y momento $p = h/\\lambda$. En el primero un fotón arranca un electrón de un metal; en el segundo un fotón de rayos X rebota contra un electrón casi libre.",
            "En el fotoeléctrico, la energía del fotón se reparte entre vencer la función de trabajo $\\phi$ (la energía mínima para sacar un electrón del metal) y dar energía cinética al electrón. Frenar al electrón más rápido requiere un potencial $V_0$ tal que $eV_0 = hf - \\phi$. Graficar $V_0$ contra $f$ da una recta cuya pendiente es $h/e$ sin importar el metal, y cuya ordenada al origen revela $\\phi$. Por debajo de la frecuencia umbral $f_0 = \\phi/h$ no hay emisión, por intensa que sea la luz.",
            "En la dispersión Compton, el fotón entrega parte de su energía y momento al electrón, de modo que sale con mayor longitud de onda. Conservar energía relativista y momento vectorial da $\\Delta\\lambda = (h/m_ec)(1-\\cos\\theta)$, que depende solo del ángulo de dispersión y no de la longitud de onda incidente. La cantidad $h/m_ec \\approx 2.43$ pm es la longitud de onda de Compton del electrón.",
            "El efecto Compton solo es medible con fotones de energía comparable a $m_ec^2$ en proporción apreciable: con luz visible (500 nm) el corrimiento de 2.4 pm es de unas $5\\times10^{-6}$ partes, inobservable, mientras que con rayos X de 71 pm es un 3 % y se ve con claridad. En ambos fenómenos el modelo ondulatorio clásico falla y el cuántico da el valor correcto.",
        ],
        analytic_steps=[
            ("Plantear la conservación en la dispersión Compton", "Un fotón de energía $E_0 = hc/\\lambda$ choca con un electrón en reposo y sale a un ángulo $\\theta$ con energía $E_1$. Se conservan la energía total (relativista) y el momento vectorial.",
             [r"E_0 + m_ec^2 = E_1 + E_e,\qquad \vec p_0 = \vec p_1 + \vec p_e"]),
            ("Eliminar el electrón", "Del momento se obtiene $p_e^2 = p_0^2 + p_1^2 - 2p_0p_1\\cos\\theta$ (ley de cosenos). Se eleva al cuadrado la energía del electrón $E_e = E_0 - E_1 + m_ec^2$ y se iguala a $p_e^2c^2 + m_e^2c^4$.",
             [r"(E_0 - E_1)^2 + 2m_ec^2(E_0 - E_1) = E_0^2 + E_1^2 - 2E_0E_1\cos\theta",
              r"m_ec^2\,(E_0 - E_1) = E_0E_1\,(1-\cos\theta)"]),
            ("Pasar a longitudes de onda", "Se divide entre $E_0E_1$ y se usa $E = hc/\\lambda$, de modo que $1/E_1 - 1/E_0 = (\\lambda_1-\\lambda_0)/hc$.",
             [r"\Delta\lambda = \lambda_1 - \lambda_0 = \dfrac{h}{m_ec}\,(1-\cos\theta)"]),
            ("Longitud de onda de Compton", "Se evalúa el prefactor con las constantes.",
             [rf"\lambda_C = \dfrac{{h}}{{m_ec}} = \dfrac{{{sci(H_PLANCK, 6)}}}{{({sci(M_E, 6)})({sci(C_LIGHT, 6)})}} = {sci(lc, 6)}\ \mathrm{{m}} = {lc * 1e12:.5f}\ \mathrm{{pm}}"]),
            ("Corrimientos a 90° y 180°", "A 90° el coseno vale 0 y a 180° vale $-1$; el máximo corrimiento es el doble de $\\lambda_C$. Para rayos X de 71 pm la longitud de onda dispersada es ligeramente mayor y su energía baja un poco.",
             [rf"\Delta\lambda(90^\circ) = {lc * 1e12:.5f}\,(1-0) = {d90 * 1e12:.5f}\ \mathrm{{pm}}\ \Rightarrow\ \lambda_1 = {(lam0 + d90) * 1e12:.5f}\ \mathrm{{pm}}",
              rf"\Delta\lambda(180^\circ) = {lc * 1e12:.5f}\,(1-(-1)) = {d180 * 1e12:.5f}\ \mathrm{{pm}}\ \Rightarrow\ \lambda_1 = {(lam0 + d180) * 1e12:.5f}\ \mathrm{{pm}}",
              rf"E_0 = \dfrac{{hc}}{{\lambda_0}} = {e0 / keV:.4f}\ \mathrm{{keV}}\ \to\ E_1(90^\circ) = {e1_90 / keV:.4f}\ \mathrm{{keV}}"]),
            ("Ecuación de Einstein para el fotoeléctrico", "La energía cinética máxima del fotoelectrón es lo que sobra tras vencer $\\phi$, y se mide con el potencial de frenado. En forma de recta $V_0 = (h/e)f - \\phi/e$.",
             [r"eV_0 = hf - \phi\ \Rightarrow\ V_0 = \dfrac{h}{e}\,f - \dfrac{\phi}{e}"]),
            ("Valores de entrada del sodio", "Los datos sintéticos se generan con $h$ exacto y $\\phi = 2.30$ eV. La pendiente esperada es $h/e$ y el umbral $f_0 = \\phi/h$. Por ejemplo, a $9\\times10^{14}$ Hz el potencial de frenado esperado es $V_0 = (hf-\\phi)/e$.",
             [rf"\dfrac{{h}}{{e}} = \dfrac{{{sci(H_PLANCK, 6)}}}{{{sci(E_CH, 6)}}} = {sci(H_PLANCK / E_CH, 5)}\ \mathrm{{V\,s}},\qquad f_0 = \dfrac{{\phi}}{{h}} = \dfrac{{2.30\,({sci(E_CH, 6)})}}{{{sci(H_PLANCK, 6)}}} = {sci(f0, 4)}\ \mathrm{{Hz}}",
              rf"V_0(9\times10^{{14}}\ \mathrm{{Hz}}) = \dfrac{{({sci(H_PLANCK, 6)})(9\times10^{{14}})}}{{{sci(E_CH, 6)}}} - 2.30 = {float(p19.stopping_potential_analytic(9e14)):.4f}\ \mathrm{{V}}"]),
        ],
        numeric_steps=[
            ("Compton numérico: planteamiento de la raíz", "En lugar de usar la fórmula cerrada, el código busca la energía $E_1$ del fotón dispersado que cumple conservación. Se define un residuo que vale cero cuando energía y momento son compatibles.",
             [r"F(E_1) = (E_0 + m_ec^2 - E_1)^2 - \left[E_0^2 + E_1^2 - 2E_0E_1\cos\theta + m_e^2c^4\right]",
              rf"E_0 = {e0 / keV:.4f}\ \mathrm{{keV}},\qquad m_ec^2 = {mc2 / keV:.3f}\ \mathrm{{keV}},\qquad \theta = 90^\circ"]),
            ("Primera bisección", "Se parte de $[E_{lo}, E_{hi}] = [10^{-6}E_0,\\ E_0]$. Se evalúa el punto medio; si $F(E_{lo})\\,F(E_{mid}) \\le 0$ la raíz está en la mitad izquierda ($E_{hi}=E_{mid}$), y si no en la derecha ($E_{lo}=E_{mid}$).",
             [rf"E_{{lo}} = {hist[0][0] / keV:.5f},\ E_{{hi}} = {hist[0][1] / keV:.4f},\ E_{{mid}} = {hist[0][2] / keV:.4f}\ \mathrm{{keV}}",
              rf"F(E_{{lo}}) = {hist[0][3] / scale:.2f},\ F(E_{{mid}}) = {hist[0][4] / scale:.2f}\ \mathrm{{keV^2}}\ \Rightarrow\ F(E_{{lo}})F(E_{{mid}}){'>' if hist[0][3] * hist[0][4] > 0 else '<'}0\ \Rightarrow\ {'E_{lo}=E_{mid}' if hist[0][3] * hist[0][4] > 0 else 'E_{hi}=E_{mid}'}"]),
            ("Primeras seis iteraciones", "El intervalo se divide a la mitad en cada paso. Las cifras están en keV (y $F$ en keV$^2$).",
             [r"\begin{array}{c|ccccc} k & E_{lo} & E_{hi} & E_{mid} & F(E_{lo}) & F(E_{mid})\\ \hline " + tab + r"\end{array}"]),
            ("Parada y resultado de Compton", "Tras 200 bisecciones el ancho es $E_0\\,2^{-200}$, muy por debajo de la precisión de máquina, así que $E_1$ queda fijo hasta el último dígito. El corrimiento se obtiene restando longitudes de onda.",
             [rf"E_1 = {H_PLANCK * C_LIGHT / (lam0 + n90) / keV:.5f}\ \mathrm{{keV}}\ \Rightarrow\ \lambda_1 = \dfrac{{hc}}{{E_1}} = {(lam0 + n90) * 1e12:.5f}\ \mathrm{{pm}}",
              rf"\Delta\lambda^{{num}}(90^\circ) = {n90 * 1e12:.6f}\ \mathrm{{pm}}\ \ (\text{{analítico }}{d90 * 1e12:.6f}),\qquad \Delta\lambda^{{num}}(180^\circ) = {n180 * 1e12:.6f}\ \mathrm{{pm}}\ \ (\text{{analítico }}{d180 * 1e12:.6f})"]),
            ("Fotoeléctrico: datos sintéticos", "Se generan 25 frecuencias equiespaciadas entre $6\\times10^{14}$ y $12\\times10^{14}$ Hz y a cada $V_0$ exacto se le suma ruido gaussiano de 10 mV (semilla 3). Para el ajuste se usa $x = f/10^{14}$ y así evitar números del orden de $10^{14}$ en las sumas. Primeros tres puntos:",
             [r"\begin{array}{c|ccc} i & x = f/10^{14} & V_0^{exacto}\ (\mathrm{V}) & V_0^{medido}\ (\mathrm{V})\\ \hline " + dat + r"\end{array}"]),
            ("Mínimos cuadrados: ecuaciones normales", "Se busca la recta $V = a\\,x + c$ que minimiza $\\sum (V_i - ax_i - c)^2$. Anular las derivadas respecto de $a$ y $c$ da dos ecuaciones lineales cuya solución se escribe con las sumas $S_x, S_y, S_{xx}, S_{xy}$ y $N=25$.",
             [r"a = \dfrac{N S_{xy} - S_xS_y}{N S_{xx} - S_x^2},\qquad c = \dfrac{S_y - aS_x}{N}",
              rf"S_x = {sx:.4f},\quad S_y = {sy:.4f},\quad S_{{xx}} = {sxx:.4f},\quad S_{{xy}} = {sxy:.4f}"]),
            ("Resolver y convertir a h y φ", "Con las sumas se evalúan $a$ y $c$. La pendiente está en V por unidad de $10^{14}$ Hz, así que $h = a\\,e/10^{14}$; la ordenada al origen es $-\\phi/e$.",
             [rf"a = \dfrac{{25({sxy:.4f}) - ({sx:.4f})({sy:.4f})}}{{25({sxx:.4f}) - ({sx:.4f})^2}} = {slope:.6f}\ \mathrm{{V}},\qquad c = {icpt:.6f}\ \mathrm{{V}}",
              rf"h = \dfrac{{a\,e}}{{10^{{14}}}} = \dfrac{{({slope:.6f})({sci(E_CH, 6)})}}{{10^{{14}}}} = {sci(h_fit, 5)}\ \mathrm{{J\,s}},\qquad \phi = -c = {phi_fit:.4f}\ \mathrm{{eV}}"]),
            ("Incertidumbre y comparación", "La varianza residual $s^2 = \\sum r_i^2/(N-2)$ y la matriz $(A^TA)^{-1}$ dan los errores estándar. El ajuste recupera los valores de entrada dentro de aproximadamente una incertidumbre (el código usa `lstsq`, que da el mismo resultado que las ecuaciones normales).",
             [rf"s = {np.sqrt((resid_fit**2).sum() / (n - 2)):.5f}\ \mathrm{{V}}\ \ (\text{{ruido impuesto }}0.01\ \mathrm{{V}})",
              rf"h = ({hh * 1e34:.4f} \pm {dh * 1e34:.4f})\times10^{{-34}}\ \mathrm{{J\,s}}\ \ (\text{{real }}{H_PLANCK * 1e34:.4f}),\qquad \phi = ({pp:.4f} \pm {dpp:.4f})\ \mathrm{{eV}}\ \ (\text{{real }}2.30)"]),
        ],
    )


# =============================================================================== ElectronInterference
def _interference() -> dict:
    ke = 100.0
    lam = p20.de_broglie(ke)
    d, big_l = p20.D_SLIT, p20.L_SCR
    dy = p20.fringe_spacing_analytic()
    mc2 = M_E * C_LIGHT**2
    kej = ke * E_CH
    pmom = np.sqrt(kej * (kej + 2 * mc2)) / C_LIGHT
    lam_nr = H_PLANCK / np.sqrt(2 * M_E * kej)
    hits = p20.electron_hits()
    est, _ = p20.fringe_spacing_from_hits(hits)
    span = 12 * lam * big_l / d
    rng = np.random.default_rng(7)
    ys = rng.uniform(-span / 2, span / 2, 80000)
    us = rng.uniform(size=ys.size)
    prob = np.cos(np.pi * d * ys / (lam * big_l)) ** 2
    nb = 600
    hist, edges = np.histogram(hits, bins=nb, range=(-span / 2, span / 2))
    hbin = edges[1] - edges[0]
    hist2 = hist - hist.mean()
    spec = np.abs(np.fft.rfft(hist2, 16 * nb))
    freqs = np.fft.rfftfreq(16 * nb, hbin)
    k = int(np.argmax(spec))
    y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    fpk = freqs[k] + shift * (freqs[1] - freqs[0])
    ba = p20.bragg_angle_analytic()
    bn = p20.bragg_angle_numeric()
    lam54 = p20.de_broglie(54.0)
    dB = 0.215e-9
    lo, hi = 0.0, 90.0
    bt = []
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        val = 2 * dB * np.sin(np.deg2rad(mid))
        bt.append((lo, hi, mid, val))
        if val < lam54:
            lo = mid
        else:
            hi = mid
    btab = "\\\\".join(rf"{i + 1} & {x[0]:.4f} & {x[1]:.4f} & {x[2]:.4f} & {x[3] * 1e12:.2f} & {'<' if x[3] < lam54 else '>'}" for i, x in enumerate(bt))
    smp = "\\\\".join(
        rf"{i + 1} & {ys[i] * 1e6:.3f} & {prob[i]:.4f} & {us[i]:.4f} & \text{{{'acepta' if us[i] < prob[i] else 'rechaza'}}}" for i in range(5)
    )
    return dict(
        explanation=[
            "De Broglie propuso que toda partícula con momento $p$ tiene asociada una onda de longitud $\\lambda = h/p$. Para un electrón de 100 eV esa longitud es del orden de la décima de nanómetro, comparable con el espaciado entre átomos de un cristal, y por eso los electrones se difractan en sólidos cristalinos (experimento de Davisson y Germer) y producen franjas en una doble rendija.",
            "En la doble rendija, cada electrón llega a un único punto de la pantalla, pero la distribución de muchos impactos reproduce la intensidad de interferencia $I(y) \\propto \\cos^2(\\pi d y/\\lambda L)$. Las franjas brillantes aparecen donde la diferencia de camino entre las dos rendijas es un número entero de longitudes de onda, lo que da un espaciado $\\Delta y = \\lambda L/d$ (válido para ángulos pequeños).",
            "Un electrón de 100 eV tiene una energía cinética muy pequeña frente a su energía en reposo (511 keV), por lo que la corrección relativista a $p$ es minúscula; aun así, el código usa la forma relativista $p = \\sqrt{K(K+2m c^2)}/c$ por ser exacta. Con $d = 1\\ \\mu$m y $L = 1$ m las franjas son de unos 0.12 $\\mu$m, finas pero resolubles.",
            "La difracción de Bragg complementa la imagen: los planos atómicos espaciados $d$ refuerzan la onda cuando la diferencia de camino $2d\\sin\\theta$ es múltiplo de $\\lambda$. Para electrones de 54 eV sobre planos de 0.215 nm el primer máximo aparece a un ángulo de unos 50 grados, resultado que confirmó la naturaleza ondulatoria de la materia.",
        ],
        analytic_steps=[
            ("Hallar el momento relativista", "A 100 eV la energía cinética es mucho menor que $mc^2$, pero se usa la relación relativista exacta entre energía y momento: $E^2 = (pc)^2 + (mc^2)^2$ con $E = K + mc^2$.",
             [r"p = \dfrac{\sqrt{K\,(K + 2mc^2)}}{c}",
              rf"K = 100\ \mathrm{{eV}} = {sci(kej, 5)}\ \mathrm{{J}},\quad mc^2 = {sci(mc2, 5)}\ \mathrm{{J}}\ \Rightarrow\ p = {sci(pmom, 5)}\ \mathrm{{kg\,m/s}}"]),
            ("Longitud de onda de De Broglie", "Se divide la constante de Planck entre el momento. Se compara con la fórmula no relativista $\\lambda = h/\\sqrt{2mK}$ para ver que la diferencia es de unas partes por cien mil.",
             [rf"\lambda = \dfrac{{h}}{{p}} = \dfrac{{{sci(H_PLANCK, 6)}}}{{{sci(pmom, 5)}}} = {sci(lam, 5)}\ \mathrm{{m}} = {lam * 1e12:.4f}\ \mathrm{{pm}}",
              rf"\lambda_{{no\,rel}} = \dfrac{{h}}{{\sqrt{{2mK}}}} = {lam_nr * 1e12:.4f}\ \mathrm{{pm}}"]),
            ("Condición de máximos en la pantalla", "Los dos haces recorren caminos que difieren en $d\\sin\\theta \\approx d\\,y/L$ (ángulos pequeños). Hay refuerzo cuando esa diferencia es $m\\lambda$.",
             [r"d\,\dfrac{y_m}{L} = m\lambda\ \Rightarrow\ y_m = \dfrac{m\lambda L}{d}"]),
            ("Espaciado entre franjas", "La distancia entre dos máximos consecutivos es $y_{m+1}-y_m$. Se sustituyen los datos del enunciado.",
             [r"\Delta y = \dfrac{\lambda L}{d}",
              rf"\Delta y = \dfrac{{({sci(lam, 5)}\ \mathrm{{m}})(1.0\ \mathrm{{m}})}}{{{sci(d, 2)}\ \mathrm{{m}}}} = {sci(dy, 5)}\ \mathrm{{m}} = {dy * 1e6:.3f}\ \mu\mathrm{{m}}"]),
            ("Difracción de Bragg: longitud de onda a 54 eV", "En los planos de un cristal el refuerzo exige $2d\\sin\\theta = \\lambda$ (primer orden). Primero se calcula $\\lambda$ con el mismo procedimiento relativista para $K = 54$ eV.",
             [rf"\lambda_{{54}} = \dfrac{{h}}{{\sqrt{{54E_{{ch}}(54E_{{ch}} + 2mc^2)}}/c}} = {lam54 * 1e12:.4f}\ \mathrm{{pm}}"]),
            ("Despejar el ángulo de Bragg", "Con $d = 0.215$ nm se despeja $\\theta$ de la condición de refuerzo.",
             [r"\theta = \arcsin\left(\dfrac{\lambda}{2d}\right)",
              rf"\theta = \arcsin\left(\dfrac{{{lam54 * 1e12:.4f}\ \mathrm{{pm}}}}{{2({dB * 1e12:.1f}\ \mathrm{{pm}})}}\right) = \arcsin({lam54 / (2 * dB):.6f}) = {ba:.5f}^\circ"]),
        ],
        numeric_steps=[
            ("Muestreo por rechazo: idea", "Se quiere generar impactos $y$ con densidad $I(y) = \\cos^2(\\pi dy/\\lambda L)$. Se sortea $y$ uniforme en $[-s/2, s/2]$ con $s = 12\\lambda L/d$ (12 franjas), y se acepta con probabilidad $I(y) \\le 1$ comparando con un número aleatorio $u$ uniforme en $[0,1)$.",
             [rf"s = 12\,\dfrac{{\lambda L}}{{d}} = {s_fmt(span)}\ \mu\mathrm{{m}} \quad\Rightarrow\quad y \in [{-span / 2 * 1e6:.3f},\ {span / 2 * 1e6:.3f}]\ \mu\mathrm{{m}}"]),
            ("Primeras cinco muestras", "Con la semilla 7 se obtienen estos candidatos. Se acepta cuando $u < I(y)$.",
             [r"\begin{array}{c|cccc} j & y\ (\mu\mathrm{m}) & I(y) & u & \text{decisión}\\ \hline " + smp + r"\end{array}"]),
            ("Qué se obtiene al repetir", "Se repite hasta reunir $N = 20000$ aceptados. La fracción esperada de aceptación es el promedio de $\\cos^2$, es decir $1/2$; por eso se sortean cuatro veces más candidatos de los necesarios.",
             [rf"\text{{aceptados en el primer lote de 80000}} = {int((us < prob).sum())}\quad (\approx 40000)"]),
            ("Histograma", "Los impactos se cuentan en 600 intervalos de ancho $\\delta y$ y se resta la media para quitar la componente constante.",
             [rf"\delta y = \dfrac{{s}}{{600}} = {hbin * 1e9:.3f}\ \mathrm{{nm}},\qquad \bar n = \dfrac{{20000}}{{600}} = {hist.mean():.2f}\ \text{{impactos por intervalo}}"]),
            ("Transformada de Fourier: versión reducida a mano", "Para entender qué mide la FFT, un ejemplo de 4 intervalos con cuentas $(3, 1, 3, 1)$: restada la media $(2)$ queda $(1,-1,1,-1)$. La transformada $X_k = \\sum_j n_j e^{-2\\pi i jk/4}$ solo es distinta de cero en $k=2$, es decir una oscilación de un periodo cada 2 intervalos. La frecuencia espacial es $k/(N\\delta y)$.",
             [r"X_0 = 1-1+1-1 = 0,\qquad X_2 = 1\cdot1 + (-1)(-1) + 1\cdot1 + (-1)(-1) = 4,\qquad f = \dfrac{2}{4\,\delta y} = \dfrac{1}{2\,\delta y}"]),
            ("FFT real y refinamiento", "El código aplica lo mismo a los 600 intervalos, con relleno de ceros hasta 9600 puntos para tener una rejilla de frecuencias más fina, y toma el máximo del espectro. Un ajuste parabólico a tres puntos corrige el máximo entre muestras de frecuencia.",
             [rf"k_{{pico}} = {k},\quad f_k = {sci(freqs[k], 5)}\ \mathrm{{m^{{-1}}}},\quad \text{{corrección}} = {shift:.4f}\,\Delta f",
              rf"f = {sci(fpk, 5)}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ \Delta y_{{num}} = \dfrac{{1}}{{f}} = {1 / fpk * 1e6:.4f}\ \mu\mathrm{{m}}"]),
            ("Comparar con el análisis", "La diferencia viene del ruido de conteo de 20000 electrones, no del método.",
             [rf"\Delta y_{{num}} = {est * 1e6:.4f}\ \mu\mathrm{{m}},\quad \Delta y_{{ana}} = {dy * 1e6:.4f}\ \mu\mathrm{{m}},\quad \dfrac{{\Delta y_{{num}} - \Delta y_{{ana}}}}{{\Delta y_{{ana}}}} = {sci((est - dy) / dy, 2)}"]),
            ("Bragg numérico: bisección en el ángulo", "La función $2d\\sin\\theta - \\lambda$ crece con $\\theta$ en $[0^\\circ, 90^\\circ]$. Si $2d\\sin(\\theta_{mid}) < \\lambda$ se sube el extremo inferior; si no, se baja el superior. Primeras seis iteraciones (la columna $2d\\sin\\theta_{mid}$ en pm se compara con $\\lambda_{54}$ = " + f"{lam54 * 1e12:.2f}" + " pm):",
             [r"\begin{array}{c|ccccc} k & \theta_{lo} & \theta_{hi} & \theta_{mid} & 2d\sin\theta_{mid} & \text{vs }\lambda\\ \hline " + btab + r"\end{array}",
              rf"\text{{tras 80 iteraciones: }}\theta_{{num}} = {bn:.8f}^\circ\ \ (\text{{analítico }}{ba:.8f}^\circ)"]),
        ],
    )


def s_fmt(x: float) -> str:
    return f"{x * 1e6:.3f}"


# =============================================================================== WavePacketUncertainty
def _packet() -> dict:
    s0 = p20.SIGMA0
    sx0, sp0 = p20.packet_numeric(0.0)
    tau = 2 * M_E * s0**2 / HBAR
    w05 = float(p20.packet_width_analytic(0.5e-15))
    w1 = float(p20.packet_width_analytic(1e-15))
    n05, p05 = p20.packet_numeric(0.5e-15)
    n1, _ = p20.packet_numeric(1e-15)
    dp_ana = HBAR / (2 * s0)
    n, box = 4096, 400e-10
    dx = box / n
    x = np.linspace(-box / 2, box / 2, n, endpoint=False)
    psi = np.exp(-(x**2) / (4 * s0**2))
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    dk = 2 * np.pi / box
    k1 = 1.0 / s0
    ph1 = HBAR * k1**2 * 0.5e-15 / (2 * M_E)
    ph_dk = HBAR * dk**2 * 0.5e-15 / (2 * M_E)
    t = 0.5e-15
    return dict(
        explanation=[
            "En mecánica cuántica una partícula libre localizada se describe con un paquete de onda: una superposición de ondas planas de distintos números de onda $k$. Cuanto más estrecho es el paquete en el espacio, más ancha debe ser la distribución de momentos que lo forma. Esto no es una limitación del instrumento de medida, sino una propiedad matemática de las transformadas de Fourier.",
            "El principio de incertidumbre afirma $\\Delta x\\,\\Delta p \\ge \\hbar/2$. La igualdad se alcanza solo para un paquete gaussiano, que es el de mínima incertidumbre. Si $|\\psi|^2$ es gaussiana con desviación estándar $\\sigma$, la distribución de momentos también lo es, con $\\Delta p = \\hbar/2\\sigma$.",
            "Un paquete libre también se ensancha con el tiempo: las ondas componentes tienen distinta velocidad de fase ($\\omega = \\hbar k^2/2m$), de modo que se desfasan entre sí. La distribución de momentos no cambia, pero el ancho espacial crece según $\\sigma(t) = \\sigma_0\\sqrt{1 + (\\hbar t/2m\\sigma_0^2)^2}$.",
            "Para un electrón con $\\sigma_0 = 0.1$ nm (tamaño atómico) la escala de tiempo del ensanchamiento es $\\tau = 2m\\sigma_0^2/\\hbar \\approx 0.17$ fs: el paquete se duplica en unos 0.3 fs. Para un objeto macroscópico, $m$ enorme hace $\\tau$ astronómicamente larga y el ensanchamiento es inobservable.",
        ],
        analytic_steps=[
            ("Plantear el paquete gaussiano", "Se toma un paquete cuya densidad de probabilidad es una gaussiana de desviación estándar $\\sigma_0$. La amplitud correspondiente es proporcional a $e^{-x^2/4\\sigma_0^2}$ (el factor 4 aparece porque la densidad es el cuadrado de la amplitud).",
             [r"\psi(x,0) \propto e^{-x^2/4\sigma_0^2},\qquad |\psi|^2 \propto e^{-x^2/2\sigma_0^2}\ \Rightarrow\ \Delta x = \sigma_0"]),
            ("Transformada de Fourier", "La transformada de una gaussiana es otra gaussiana. Integrando $\\phi(k)=\\int\\psi\\,e^{-ikx}dx$ se obtiene $e^{-k^2\\sigma_0^2}$, y su cuadrado tiene desviación estándar $1/2\\sigma_0$ en $k$.",
             [r"\phi(k) \propto e^{-k^2\sigma_0^2},\qquad |\phi|^2 \propto e^{-k^2/(2\,(1/2\sigma_0)^2)}\ \Rightarrow\ \Delta k = \dfrac{1}{2\sigma_0}"]),
            ("Incertidumbre en el momento", "Como $p = \\hbar k$, se multiplica $\\Delta k$ por $\\hbar$.",
             [rf"\Delta p = \hbar\Delta k = \dfrac{{\hbar}}{{2\sigma_0}} = \dfrac{{{sci(HBAR, 6)}}}{{2({sci(s0, 3)}\ \mathrm{{m}})}} = {sci(dp_ana, 5)}\ \mathrm{{kg\,m/s}}"]),
            ("Producto de incertidumbres", "Se multiplica $\\Delta x\\,\\Delta p$; $\\sigma_0$ se cancela y queda el límite inferior del principio.",
             [rf"\Delta x\,\Delta p = \sigma_0\cdot\dfrac{{\hbar}}{{2\sigma_0}} = \dfrac{{\hbar}}{{2}} = {sci(HBAR / 2, 6)}\ \mathrm{{J\,s}}"]),
            ("Evolución libre en el espacio de momentos", "Cada onda plana evoluciona con su propia fase $e^{-i\\omega t}$, con $\\omega = \\hbar k^2/2m$. La integral gaussiana se resuelve completando cuadrados y da un ancho complejo $\\sigma_0^2 + i\\hbar t/2m$.",
             [r"\psi(x,t) \propto \int \phi(k)\,e^{ikx - i\hbar k^2t/2m}\,dk \ \Rightarrow\ \sigma(t)^2 = \sigma_0^2 + \left(\dfrac{\hbar t}{2m\sigma_0}\right)^2"]),
            ("Escala de tiempo del ensanchamiento", "Se factoriza $\\sigma_0$ y se define $\\tau = 2m\\sigma_0^2/\\hbar$: cuando $t = \\tau$ el ancho vale $\\sqrt2\\,\\sigma_0$.",
             [r"\sigma(t) = \sigma_0\sqrt{1 + (t/\tau)^2},\qquad \tau = \dfrac{2m\sigma_0^2}{\hbar}",
              rf"\tau = \dfrac{{2({sci(M_E, 5)})({sci(s0, 3)})^2}}{{{sci(HBAR, 6)}}} = {sci(tau, 4)}\ \mathrm{{s}} = {tau * 1e15:.4f}\ \mathrm{{fs}}"]),
            ("Anchos a 0.5 fs y 1 fs", "Se evalúa la fórmula con las razones $t/\\tau$.",
             [rf"\sigma(0.5\ \mathrm{{fs}}) = (1.000\ \text{{Å}})\sqrt{{1 + ({0.5e-15 / tau:.4f})^2}} = {w05 * 1e10:.5f}\ \text{{Å}}",
              rf"\sigma(1\ \mathrm{{fs}}) = (1.000\ \text{{Å}})\sqrt{{1 + ({1e-15 / tau:.4f})^2}} = {w1 * 1e10:.5f}\ \text{{Å}}"]),
        ],
        numeric_steps=[
            ("Malla y paquete inicial", "Se discretiza una caja de 400 Å en $n=4096$ puntos (paso $\\delta x$) y se evalúa $\\psi(x,0)=e^{-x^2/4\\sigma_0^2}$. Se normaliza para que $\\sum|\\psi|^2\\delta x = 1$.",
             [rf"\delta x = \dfrac{{400\ \text{{Å}}}}{{4096}} = {dx * 1e10:.5f}\ \text{{Å}},\qquad x_j = -200\ \text{{Å}} + j\,\delta x",
              rf"\sum_j e^{{-x_j^2/2\sigma_0^2}}\,\delta x = {norm**2 * 1e10:.5f}\ \text{{Å}}\ \Rightarrow\ \psi_{{2048}}\ (x=0) = \dfrac{{1}}{{\sqrt{{{norm**2 * 1e10:.5f}\ \text{{Å}}}}}} = {1 / norm * 1e-5:.5f}\ \text{{Å}}^{{-1/2}}"]),
            ("Rejilla de números de onda", "La FFT necesita los números de onda $k_m = 2\\pi m/(n\\,\\delta x)$ con $m$ de $-n/2$ a $n/2-1$. El paso en $k$ es $2\\pi/\\text{(caja)}$ y el máximo es $\\pi/\\delta x$.",
             [rf"\delta k = \dfrac{{2\pi}}{{400\ \text{{Å}}}} = {dk * 1e-10:.6f}\ \text{{Å}}^{{-1}},\qquad k_{{max}} = \dfrac{{\pi}}{{\delta x}} = {np.pi / dx * 1e-10:.4f}\ \text{{Å}}^{{-1}}"]),
            ("Transformada directa e inversa", "El método es exacto para la partícula libre: se aplica $\\phi = \\mathrm{FFT}(\\psi)$, se multiplica cada componente por su fase y se regresa con la FFT inversa. No hay error de paso temporal.",
             [r"\phi_m = \sum_{j}\psi_j\,e^{-2\pi i jm/n},\quad \phi_m(t) = \phi_m\,e^{-i\hbar k_m^2t/2m},\quad \psi_j(t) = \dfrac{1}{n}\sum_m \phi_m(t)\,e^{2\pi i jm/n}"]),
            ("Fase de ejemplo a 0.5 fs", "La fase crece como $k^2$, así que las componentes de $k$ grande se retrasan mucho más. Dos ejemplos: la componente de $k=\\delta k$ casi no gira y la de $k=1/\\sigma_0$ (a dos desviaciones estándar del centro de $|\\phi|^2$) gira casi tres radianes.",
             [rf"k=\delta k:\ \ \varphi = \dfrac{{\hbar k^2 t}}{{2m}} = {sci(ph_dk, 3)}\ \mathrm{{rad}}",
              rf"k = 1/\sigma_0 = {sci(k1, 3)}\ \mathrm{{m^{{-1}}}}:\ \ \varphi = \dfrac{{({sci(HBAR, 5)})({sci(k1, 3)})^2({sci(t, 2)})}}{{2({sci(M_E, 5)})}} = {ph1:.4f}\ \mathrm{{rad}}"]),
            ("Momentos de la distribución", "Con $P(x)=|\\psi|^2/\\sum|\\psi|^2\\delta x$ se calcula $\\Delta x = \\sqrt{\\langle x^2\\rangle - \\langle x\\rangle^2}$ por suma sobre la malla; igual con $|\\phi|^2$ para $\\Delta k$, y $\\Delta p = \\hbar\\Delta k$.",
             [r"\langle x^n\rangle = \sum_j x_j^n\,P_j\,\delta x,\qquad \langle k^n\rangle = \sum_m k_m^n\,Q_m,\quad Q_m = \dfrac{|\phi_m|^2}{\sum|\phi|^2}",
              rf"t = 0:\ \ \Delta x = {sx0 * 1e10:.7f}\ \text{{Å}},\ \ \Delta p = {sci(sp0, 6)}\ \mathrm{{kg\,m/s}}"]),
            ("Producto de incertidumbres", "Se compara $\\Delta x\\,\\Delta p$ con $\\hbar/2$; el cociente debe valer 1 en $t=0$ y, como $\\Delta p$ no cambia en la evolución libre, el producto crece con $\\Delta x$.",
             [rf"\dfrac{{\Delta x\,\Delta p}}{{\hbar/2}} = \dfrac{{({sci(sx0, 6)})({sci(sp0, 6)})}}{{{sci(HBAR / 2, 6)}}} = {sx0 * sp0 / (HBAR / 2):.8f}",
              rf"t = 0.5\ \mathrm{{fs}}:\ \ \dfrac{{\Delta x\,\Delta p}}{{\hbar/2}} = {n05 * p05 / (HBAR / 2):.5f}"]),
            ("Anchos evolucionados y error", "El ancho numérico a 0.5 fs y 1 fs coincide con la fórmula analítica a muchas cifras, porque la FFT resuelve exactamente la evolución libre; las pequeñas diferencias vienen del tamaño finito de la caja y de la discretización.",
             [rf"\sigma_{{num}}(0.5\ \mathrm{{fs}}) = {n05 * 1e10:.7f}\ \text{{Å}}\ \ (\text{{analítico }}{w05 * 1e10:.7f}),\qquad \sigma_{{num}}(1\ \mathrm{{fs}}) = {n1 * 1e10:.7f}\ \text{{Å}}\ \ (\text{{analítico }}{w1 * 1e10:.7f})"]),
        ],
    )


# =============================================================================== InfiniteWell
def _well() -> dict:
    ln = p21.L_WELL
    ea = p21.well_energy_analytic([1, 2, 3, 4])
    en, x, v = p21.well_numeric(4)
    npts = 2000
    h = ln / (npts + 1)
    t_ev = HBAR**2 / (2 * M_E * h**2) / E_CH
    h3 = ln / 4
    t3 = HBAR**2 / (2 * M_E * h3**2) / E_CH
    e3 = np.array([t3 * (2 - np.sqrt(2)), 2 * t3, t3 * (2 + np.sqrt(2))])
    th = np.pi / (npts + 1)
    ov = p21.wavefunction_overlap(1)
    c0 = HBAR**2 * np.pi**2 / (2 * M_E * ln**2) / E_CH
    amp = np.sqrt(2 / ln)
    return dict(
        explanation=[
            "La ecuación de Schrödinger independiente del tiempo, $-\\tfrac{\\hbar^2}{2m}\\psi'' + V\\psi = E\\psi$, es la versión cuántica de la conservación de la energía: la energía cinética está asociada a la curvatura de la función de onda y la potencial a $V$. El sistema más simple con partícula confinada es el pozo infinito, en el que $V=0$ dentro de $0<x<L$ y $V\\to\\infty$ fuera.",
            "Como la partícula no puede estar fuera, $\\psi$ debe anularse en las paredes. Dentro del pozo las soluciones son senos; que quepan exactamente un número entero de semiondas entre $0$ y $L$ es la condición que cuantiza la energía. Cuanto más estrecho es el pozo, más corta es la longitud de onda y mayor es la energía: $E_n \\propto n^2/L^2$.",
            "La densidad de probabilidad $|\\psi_n|^2$ tiene $n-1$ nodos interiores, de modo que los estados más excitados oscilan más. El estado fundamental $n=1$ tiene energía distinta de cero (energía de punto cero): por el principio de incertidumbre, confinar una partícula obliga a darle energía cinética.",
            "Para un electrón en $L = 1$ nm la escala de energía es de décimas de eV, comparable con transiciones ópticas de puntos cuánticos. El espaciado crece con $n$ como $(2n+1)E_1$, y en la cercanía del límite clásico (números cuánticos grandes) el espectro se vuelve prácticamente continuo.",
        ],
        analytic_steps=[
            ("Plantear la ecuación dentro del pozo", "Con $V=0$ la ecuación se reduce a $\\psi'' = -k^2\\psi$, con $k^2 = 2mE/\\hbar^2$. La solución general combina senos y cosenos.",
             [r"-\dfrac{\hbar^2}{2m}\psi'' = E\psi\ \Rightarrow\ \psi(x) = A\sin kx + B\cos kx,\qquad k = \dfrac{\sqrt{2mE}}{\hbar}"]),
            ("Aplicar las condiciones de frontera", "Que $\\psi(0)=0$ obliga a $B=0$. Que $\\psi(L)=0$ exige $\\sin kL = 0$, es decir $kL = n\\pi$ con $n=1,2,3,\\dots$ ($n=0$ daría $\\psi\\equiv0$).",
             [r"\psi(0)=0\ \Rightarrow\ B = 0,\qquad \psi(L) = 0\ \Rightarrow\ k_n = \dfrac{n\pi}{L}"]),
            ("Energías permitidas", "Se sustituye $k_n$ en $E = \\hbar^2k^2/2m$.",
             [r"E_n = \dfrac{\hbar^2k_n^2}{2m} = \dfrac{n^2\pi^2\hbar^2}{2mL^2}",
              rf"E_1 = \dfrac{{\pi^2({sci(HBAR, 6)})^2}}{{2({sci(M_E, 5)})({sci(ln, 2)})^2}} = {sci(c0 * E_CH, 5)}\ \mathrm{{J}} = {c0:.5f}\ \mathrm{{eV}}"]),
            ("Los cuatro primeros niveles", "Como $E_n = n^2E_1$, basta multiplicar por 1, 4, 9 y 16.",
             [rf"E_2 = 4({c0:.5f}) = {ea[1]:.5f}\ \mathrm{{eV}},\quad E_3 = 9({c0:.5f}) = {ea[2]:.5f}\ \mathrm{{eV}},\quad E_4 = 16({c0:.5f}) = {ea[3]:.5f}\ \mathrm{{eV}}"]),
            ("Normalizar la función de onda", "La probabilidad total debe ser 1. Se usa $\\int_0^L\\sin^2(n\\pi x/L)\\,dx = L/2$.",
             [r"\int_0^L A^2\sin^2\dfrac{n\pi x}{L}\,dx = A^2\,\dfrac{L}{2} = 1\ \Rightarrow\ A = \sqrt{\dfrac{2}{L}}",
              rf"A = \sqrt{{\dfrac{{2}}{{{sci(ln, 2)}\ \mathrm{{m}}}}}} = {sci(amp, 5)}\ \mathrm{{m^{{-1/2}}}}"]),
            ("Función de onda del tercer estado", "Con $n=3$ la función tiene dos nodos interiores, en $x = L/3$ y $x=2L/3$, y tres lóbulos. Su máximo de amplitud es $\\sqrt{2/L}$ en $x=L/6,\\ L/2,\\ 5L/6$.",
             [rf"\psi_3(x) = {sci(amp, 5)}\,\sin\left(\dfrac{{3\pi x}}{{{sci(ln, 2)}}}\right)\ \mathrm{{m^{{-1/2}}}},\qquad E_3 = {ea[2]:.5f}\ \mathrm{{eV}}"]),
        ],
        numeric_steps=[
            ("Discretizar la ecuación", "Se divide el pozo en $N=2000$ nodos interiores con paso $h = L/(N+1)$ y se reemplaza $\\psi''$ por la diferencia central $(\\psi_{j+1}-2\\psi_j+\\psi_{j-1})/h^2$. Las condiciones $\\psi_0=\\psi_{N+1}=0$ ya están incorporadas al no haber nodos de frontera.",
             [rf"h = \dfrac{{{sci(ln, 2)}}}{{2001}} = {sci(h, 5)}\ \mathrm{{m}},\qquad t = \dfrac{{\hbar^2}}{{2mh^2}} = {t_ev:.3f}\ \mathrm{{eV}}",
              r"-t\,\psi_{j-1} + 2t\,\psi_j - t\,\psi_{j+1} = E\,\psi_j"]),
            ("Matriz del hamiltoniano", "La ecuación es un problema de autovalores $H\\vec\\psi = E\\vec\\psi$ con $H$ tridiagonal: $2t$ en la diagonal y $-t$ en las dos diagonales adyacentes.",
             [r"H = t\begin{pmatrix} 2 & -1 & & \\ -1 & 2 & -1 & \\ & -1 & 2 & \ddots \\ & & \ddots & \ddots \end{pmatrix}_{2000\times2000}"]),
            ("Versión reducida a mano: N = 3", "Con 3 nodos, $h=L/4$ y $H = t_3\\,\\mathrm{tridiag}(-1, 2, -1)$. El polinomio característico en $\\lambda = E/t_3$ es $(2-\\lambda)^3 - 2(2-\\lambda) = 0$, con raíces $2-\\sqrt2,\\ 2,\\ 2+\\sqrt2$.",
             [rf"t_3 = \dfrac{{\hbar^2}}{{2m(L/4)^2}} = {t3:.4f}\ \mathrm{{eV}}",
              rf"E_1 = (2-\sqrt2)\,t_3 = {e3[0]:.4f},\quad E_2 = 2t_3 = {e3[1]:.4f},\quad E_3 = (2+\sqrt2)\,t_3 = {e3[2]:.4f}\ \mathrm{{eV}}",
              rf"\text{{(exactos: }}{ea[0]:.4f},\ {ea[1]:.4f},\ {ea[2]:.4f}\ \mathrm{{eV}}\text{{; con 3 nodos el error es grande)}}"]),
            ("Fórmula de la malla fina", "Para la matriz tridiagonal de $N$ nodos los autovalores se conocen: $E_j = 2t[1-\\cos(j\\pi/(N+1))]$. Para $j\\pi/(N+1)$ pequeño, esto es $E_j\\approx E_j^{exacto}\\,[1 - \\theta_j^2/12]$ con $\\theta_j = j\\pi/(N+1)$: el error relativo decrece como $1/N^2$.",
             [rf"\theta_1 = \dfrac{{\pi}}{{2001}} = {sci(th, 4)},\qquad \dfrac{{E_1^{{num}} - E_1}}{{E_1}} \approx -\dfrac{{\theta_1^2}}{{12}} = {sci(-th**2 / 12, 3)}"]),
            ("Diagonalización", "El programa usa `eigh_tridiagonal`, que calcula solo los 4 autovalores más bajos de la matriz de $2000\\times2000$ (por bisección de Sturm y QR para tridiagonales simétricas). Se dividen entre $e$ para pasar a eV.",
             [rf"E^{{num}} = ({en[0]:.5f},\ {en[1]:.5f},\ {en[2]:.5f},\ {en[3]:.5f})\ \mathrm{{eV}}",
              rf"E^{{ana}} = ({ea[0]:.5f},\ {ea[1]:.5f},\ {ea[2]:.5f},\ {ea[3]:.5f})\ \mathrm{{eV}}"]),
            ("Normalizar los autovectores", "Los autovectores salen con $\\sum\\psi_j^2=1$. Para que cumplan $\\int|\\psi|^2dx\\approx\\sum\\psi_j^2h=1$ se dividen entre $\\sqrt h$.",
             [rf"\psi_j^{{(n)}}\ \to\ \dfrac{{\psi_j^{{(n)}}}}{{\sqrt h}},\qquad \sqrt h = {sci(np.sqrt(h), 4)}\ \mathrm{{m^{{1/2}}}}",
              rf"\text{{nodos de }}\psi_3\ \text{{(cambios de signo)}}:\ {int((np.diff(np.sign(v[:, 2])) != 0).sum())}"]),
            ("Verificar con el traslape y los errores", "Si el método es correcto, el estado numérico se superpone con el analítico: $|\\langle\\psi|\\psi_1\\rangle| = |\\sum_j\\psi_j\\psi_1(x_j)h| \\approx 1$.",
             [rf"|\langle\psi|\psi_1\rangle| = {ov:.8f}",
              rf"\dfrac{{E_n^{{num}} - E_n}}{{E_n}} = ({sci((en[0] - ea[0]) / ea[0], 2)},\ {sci((en[1] - ea[1]) / ea[1], 2)},\ {sci((en[2] - ea[2]) / ea[2], 2)},\ {sci((en[3] - ea[3]) / ea[3], 2)})"]),
        ],
    )


# =============================================================================== BarrierTunneling
def _tunnel() -> dict:
    m = M_E
    e_ev, v0_ev, a = p21.E_BEAM_EV, p21.V0_EV, p21.A_BAR
    e, v0 = e_ev * E_CH, v0_ev * E_CH
    k = np.sqrt(2 * m * e) / HBAR
    kap = np.sqrt(2 * m * (v0 - e)) / HBAR
    sh = np.sinh(kap * a)
    ta = p21.transmission_analytic()
    approx = 16 * e * (v0 - e) / v0**2 * np.exp(-2 * kap * a)
    t12 = p21.transmission_analytic(12.0)
    q12 = np.sqrt(2 * m * (12.0 - 10.0) * E_CH) / HBAR
    t8a, t8n = p21.transmission_analytic(8.0), p21.transmission_numeric(8.0)
    tn = p21.transmission_numeric()
    t12n = p21.transmission_numeric(12.0)
    # RK4 replicado
    n = 4000
    hs = -a / n
    psi = np.array([1.0 + 0j, 1j * k])
    f = lambda y: np.array([y[1], -(2 * m / HBAR**2) * (e - v0) * y[0]])  # noqa: E731
    k1 = f(psi)
    k2 = f(psi + 0.5 * hs * k1)
    k3 = f(psi + 0.5 * hs * k2)
    k4 = f(psi + hs * k3)
    y1 = psi + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ex = np.array([np.exp(1j * k * hs), 1j * k * np.exp(1j * k * hs)])  # no aplica en la barrera; solo referencia
    del ex
    ps = psi.copy()
    traj = []
    for i in range(n):
        a1 = f(ps)
        a2 = f(ps + 0.5 * hs * a1)
        a3 = f(ps + 0.5 * hs * a2)
        a4 = f(ps + hs * a3)
        ps = ps + hs / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        if i in (0, 999, 1999, 2999, 3999):
            traj.append((a + hs * (i + 1), ps.copy()))
    amp = 0.5 * (ps[0] + ps[1] / (1j * k))
    ttab = "\\\\".join(rf"{x * 1e9:.4f} & {abs(y[0]):.5f} & {sci(abs(y[1]), 4)}" for x, y in traj)
    return dict(
        explanation=[
            "En mecánica clásica una partícula con energía $E$ menor que la altura $V_0$ de una barrera rebota siempre. En mecánica cuántica la función de onda no se anula bruscamente: dentro de la barrera decae exponencialmente pero no es cero, y si la barrera es lo bastante delgada queda una amplitud al otro lado. Esa probabilidad no nula de atravesarla se llama efecto túnel.",
            "Se resuelve la ecuación de Schrödinger por zonas. Antes y después de la barrera la solución es una onda plana con número de onda $k=\\sqrt{2mE}/\\hbar$; dentro, con $E<V_0$, es una combinación de exponenciales reales con $\\kappa = \\sqrt{2m(V_0-E)}/\\hbar$. Imponer continuidad de $\\psi$ y $\\psi'$ en las dos paredes determina el coeficiente de transmisión $T$.",
            "La dependencia clave es $T \\sim e^{-2\\kappa a}$: crece de forma exponencial al adelgazar la barrera o bajarla. Con $\\kappa a \\approx 5.7$ el factor es $10^{-5}$, y la transmisión resulta de un uno por mil. Esta sensibilidad hace posible el microscopio de efecto túnel y explica la desintegración alfa y la fusión nuclear en las estrellas.",
            "Si $E>V_0$ la partícula cruza, pero la reflexión clásica no es cero: la discontinuidad del potencial refleja parte de la onda, y $T$ oscila con la anchura de la barrera, con valor 1 cuando $q a$ es múltiplo de $\\pi$ (resonancias de transmisión), donde $q=\\sqrt{2m(E-V_0)}/\\hbar$.",
        ],
        analytic_steps=[
            ("Números de onda en cada zona", "Fuera de la barrera la energía cinética es $E$; dentro, con $E<V_0$, la solución es evanescente con decaimiento $\\kappa$. Se evalúan con $E=5$ eV, $V_0 = 10$ eV, $a = 0.50$ nm.",
             [rf"k = \dfrac{{\sqrt{{2m E}}}}{{\hbar}} = \dfrac{{\sqrt{{2({sci(m, 5)})({sci(e, 5)})}}}}{{{sci(HBAR, 6)}}} = {sci(k, 5)}\ \mathrm{{m^{{-1}}}}",
              rf"\kappa = \dfrac{{\sqrt{{2m (V_0 - E)}}}}{{\hbar}} = {sci(kap, 5)}\ \mathrm{{m^{{-1}}}}"]),
            ("Plantear la solución y las condiciones de empalme", "Se propone $\\psi = e^{ikx} + re^{-ikx}$ para $x<0$, $\\psi = Ce^{\\kappa x} + De^{-\\kappa x}$ en $0<x<a$, y $\\psi = te^{ikx}$ para $x>a$. Continuidad de $\\psi$ y $\\psi'$ en $x=0$ y $x=a$ da cuatro ecuaciones para $r, C, D, t$.",
             [r"T = |t|^2 = \left[1 + \dfrac{V_0^2\,\sinh^2(\kappa a)}{4E\,(V_0 - E)}\right]^{-1}"]),
            ("Calcular el argumento del seno hiperbólico", "La combinación $\\kappa a$ mide cuántas longitudes de decaimiento atraviesa la onda.",
             [rf"\kappa a = ({sci(kap, 5)})({sci(a, 2)}) = {kap * a:.5f},\qquad \sinh(\kappa a) = {sh:.4f},\qquad \sinh^2(\kappa a) = {sh**2:.3f}"]),
            ("Sustituir y obtener T", "El prefactor $V_0^2/4E(V_0-E)$ con $E = V_0/2$ vale 1, por lo que $T = 1/(1+\\sinh^2\\kappa a) = 1/\\cosh^2\\kappa a$ en este caso particular.",
             [rf"\dfrac{{V_0^2}}{{4E(V_0-E)}} = \dfrac{{10^2}}{{4(5)(5)}} = 1\ \Rightarrow\ T = \dfrac{{1}}{{1 + {sh**2:.3f}}} = {sci(ta, 5)}"]),
            ("Aproximación de barrera gruesa", "Cuando $\\kappa a\\gg1$, $\\sinh^2\\kappa a \\approx e^{2\\kappa a}/4$ y $T\\approx 16\\,\\tfrac{E}{V_0}\\left(1-\\tfrac{E}{V_0}\\right)e^{-2\\kappa a}$. Sirve para estimar ordenes de magnitud.",
             [rf"T \approx 16\,(0.5)(0.5)\,e^{{-2({kap * a:.5f})}} = 4\,e^{{-{2 * kap * a:.4f}}} = {sci(approx, 4)}"]),
            ("Energía por encima de la barrera", "Si $E=12$ eV $>V_0$, el seno hiperbólico se convierte en seno: $\\kappa\\to iq$, con $q=\\sqrt{2m(E-V_0)}/\\hbar$, y la transmisión oscila.",
             [r"T = \left[1 + \dfrac{V_0^2\sin^2(qa)}{4E\,(E-V_0)}\right]^{-1}",
              rf"q = {sci(q12, 5)}\ \mathrm{{m^{{-1}}}},\quad qa = {q12 * a:.5f},\quad \sin^2(qa) = {np.sin(q12 * a) ** 2:.5f}",
              rf"T = \left[1 + \dfrac{{10^2\,({np.sin(q12 * a) ** 2:.5f})}}{{4(12)(2)}}\right]^{{-1}} = {t12:.5f}"]),
        ],
        numeric_steps=[
            ("Resolver la ecuación por integración directa", "En lugar de empalmar a mano, se integra $\\psi'' = -\\tfrac{2m}{\\hbar^2}(E-V)\\psi$ numéricamente. Se parte de $x=a$ (a la derecha) con una onda transmitida pura $\\psi = e^{ik(x-a)}$, de modo que $t=1$, y se integra hacia atrás hasta $x=0$. Allí se descompone la solución en onda incidente y reflejada; $T = 1/|A|^2$.",
             [rf"\psi(a) = 1,\qquad \psi'(a) = ik = {_cz(1j * k, 4)}\ \mathrm{{m^{{-1}}}}",
              rf"N = {n},\qquad h = -\dfrac{{a}}{{N}} = {sci(hs, 4)}\ \mathrm{{m}}"]),
            ("Sistema de primer orden y función derivada", "Se escribe $\\vec y = (\\psi, \\psi')$ y $d\\vec y/dx = (\\psi',\\ -\\tfrac{2m}{\\hbar^2}(E-V_0)\\psi)$. En la barrera $E-V_0<0$, de modo que la segunda componente es $+\\kappa^2\\psi$.",
             [rf"\vec f(\vec y) = (y_2,\ \kappa^2 y_1),\qquad \kappa^2 = {sci(kap**2, 5)}\ \mathrm{{m^{{-2}}}}"]),
            ("Primer paso RK4: las cuatro pendientes", "Cada paso evalúa la derivada cuatro veces: al inicio, dos veces en el punto medio (con las pendientes previas) y al final. Se combinan con pesos $1,2,2,1$.",
             [rf"\vec k_1 = \vec f(\vec y_0) = {_cz(k1[0], 4)},\ {_cz(k1[1], 4)}",
              rf"\vec k_2 = \vec f(\vec y_0 + \tfrac h2\vec k_1) = {_cz(k2[0], 4)},\ {_cz(k2[1], 4)}",
              rf"\vec k_3 = {_cz(k3[0], 4)},\ {_cz(k3[1], 4)},\qquad \vec k_4 = {_cz(k4[0], 4)},\ {_cz(k4[1], 4)}"]),
            ("Primer valor actualizado", "La nueva solución en $x = a + h$ es $\\vec y_1 = \\vec y_0 + \\tfrac h6(\\vec k_1 + 2\\vec k_2 + 2\\vec k_3 + \\vec k_4)$. Se observa que $\\psi$ crece ligeramente hacia atrás, como corresponde al crecimiento exponencial hacia la izquierda.",
             [rf"\psi_1 = 1 + \dfrac{{h}}{{6}}(\dots) = {_cz(y1[0], 6)},\qquad \psi'_1 = {_cz(y1[1], 5)}\ \mathrm{{m^{{-1}}}}"]),
            ("Avance hasta x = 0", "Se repite el mismo paso 4000 veces. Algunos valores de la trayectoria muestran cómo $|\\psi|$ aumenta desde 1 hasta el borde izquierdo de la barrera (las columnas dan $x$ en nm, $|\\psi|$ y $|\\psi'|$ en m$^{-1}$).",
             [r"\begin{array}{ccc} x\ (\mathrm{nm}) & |\psi| & |\psi'|\ (\mathrm{m^{-1}})\\ \hline " + ttab + r"\end{array}"]),
            ("Descomponer en incidente y reflejada", "En $x=0$ el potencial es cero a la izquierda, donde $\\psi = Ae^{ikx} + Be^{-ikx}$ y $\\psi' = ik(Ae^{ikx} - Be^{-ikx})$. De ahí se despeja $A$, y como la onda transmitida tiene amplitud 1, $T = |1/A|^2$.",
             [r"A = \dfrac{1}{2}\left(\psi(0) + \dfrac{\psi'(0)}{ik}\right)",
              rf"\psi(0) = {_cz(ps[0], 5)},\quad \psi'(0) = {_cz(ps[1], 5)}\ \Rightarrow\ A = {_cz(amp, 5)},\ \ |A|^2 = {abs(amp) ** 2:.5f}"]),
            ("Resultado y comparación", "La transmisión es el inverso de $|A|^2$; la diferencia con la fórmula cerrada es del orden del error de truncamiento de RK4, $O(h^4)$ con $kh \\approx 10^{-3}$.",
             [rf"T^{{num}}(5\ \mathrm{{eV}}) = \dfrac{{1}}{{{abs(amp) ** 2:.5f}}} = {sci(tn, 6)}\ \ (\text{{analítico }}{sci(ta, 6)})",
              rf"T^{{num}}(8\ \mathrm{{eV}}) = {sci(t8n, 6)}\ \ (\text{{analítico }}{sci(t8a, 6)}),\qquad T^{{num}}(12\ \mathrm{{eV}}) = {t12n:.6f}\ \ (\text{{analítico }}{t12:.6f})"]),
        ],
    )


# =============================================================================== HydrogenSpectrum
def _hydrogen() -> dict:
    le = {n: p22.level_analytic(n) for n in (1, 2, 3, 4)}
    la = {n: p22.balmer_wavelength_analytic(n) for n in (3, 4, 5, 6)}
    e, r, u = p22.radial_numeric(8)
    ln_ = {n: p22.balmer_wavelength_numeric(n, e) for n in (3, 4)}
    npts, rmax = 32000, 160.0
    h = rmax / (npts + 1)
    # versión reducida
    nt, rt = 4, 10.0
    ht = rt / (nt + 1)
    rr = np.arange(1, nt + 1) * ht
    dg = 1 / ht**2 - 1 / rr
    of = np.full(nt - 1, -0.5 / ht**2)
    wt = eigh_tridiagonal(dg, of, eigvals_only=True)
    series_limit = 1e9 / (p22.RINF_M * 0.25)
    ly = 1e9 / (p22.RINF_M * (1 - 0.25))
    de3 = le[3] - le[2]
    return dict(
        explanation=[
            "En el modelo de Bohr (y en la solución exacta de la ecuación de Schrödinger para el hidrógeno) los niveles de energía del electrón ligado están cuantizados: $E_n = -13.6\\ \\mathrm{eV}/n^2$. El signo negativo indica que el electrón está ligado; $n=1$ es el estado fundamental y $E\\to0$ cuando $n\\to\\infty$ (ionización).",
            "Un átomo emite o absorbe luz cuando el electrón salta entre niveles, y el fotón se lleva exactamente la diferencia de energías: $hc/\\lambda = E_i - E_f$. Como los niveles son discretos, el espectro es de líneas, no continuo. Cada serie recibe el nombre del nivel final: Lyman ($n_f=1$, ultravioleta), Balmer ($n_f=2$, visible) y Paschen ($n_f=3$, infrarrojo).",
            "La serie de Balmer es la que se ve a simple vista: H$\\alpha$ ($3\\to2$) es roja, H$\\beta$ ($4\\to2$) azul-verdosa y las siguientes se acumulan hacia un límite de unos 365 nm. El hecho de que una fórmula tan simple con la constante de Rydberg reproduzca las líneas observadas fue la primera gran evidencia de la cuantización atómica.",
            "Como comprobación independiente, la ecuación de Schrödinger radial para $l=0$ se resuelve como problema de autovalores en una malla; sus energías numéricas deben coincidir con la fórmula de Bohr. La coincidencia valida tanto la fórmula como el método numérico.",
        ],
        analytic_steps=[
            ("Niveles de energía de Bohr", "Los niveles son $E_n = -R_yhc/n^2$ con $R_yhc = 13.6057$ eV. Se evalúan los primeros.",
             [rf"E_1 = {le[1]:.4f}\ \mathrm{{eV}},\quad E_2 = \dfrac{{{le[1]:.4f}}}{{4}} = {le[2]:.4f}\ \mathrm{{eV}},\quad E_3 = \dfrac{{{le[1]:.4f}}}{{9}} = {le[3]:.4f}\ \mathrm{{eV}}"]),
            ("Energía del fotón emitido", "Al pasar del nivel $n$ al nivel $2$ el electrón pierde la diferencia de energía, que se lleva el fotón.",
             [r"h\nu = E_n - E_2 = R_yhc\left(\dfrac{1}{2^2} - \dfrac{1}{n^2}\right)",
              rf"H\alpha\ (3\to2):\ \ h\nu = {le[3]:.4f} - ({le[2]:.4f}) = {de3:.4f}\ \mathrm{{eV}}"]),
            ("Fórmula de Balmer en longitud de onda", "Como $\\lambda = hc/h\\nu$, se obtiene la forma con la constante de Rydberg $R_\\infty = 1.0973732\\times10^{7}\\ \\mathrm{m^{-1}}$.",
             [r"\dfrac{1}{\lambda} = R_\infty\left(\dfrac{1}{2^2} - \dfrac{1}{n^2}\right)"]),
            ("Línea H-alfa", "Para $n=3$ la diferencia entre paréntesis es $1/4 - 1/9 = 5/36$.",
             [rf"\dfrac{{1}}{{\lambda}} = ({sci(p22.RINF_M, 8)})\left(\dfrac{{1}}{{4}} - \dfrac{{1}}{{9}}\right) = {sci(p22.RINF_M * (0.25 - 1 / 9), 6)}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ \lambda_{{H\alpha}} = {la[3]:.3f}\ \mathrm{{nm}}"]),
            ("Línea H-beta", "Para $n=4$: $1/4 - 1/16 = 3/16$.",
             [rf"\dfrac{{1}}{{\lambda}} = ({sci(p22.RINF_M, 8)})\left(\dfrac{{3}}{{16}}\right) = {sci(p22.RINF_M * 3 / 16, 6)}\ \mathrm{{m^{{-1}}}}\ \Rightarrow\ \lambda_{{H\beta}} = {la[4]:.3f}\ \mathrm{{nm}}"]),
            ("Otras líneas y el límite de la serie", "Al aumentar $n$ las líneas se aprietan hacia el límite $n\\to\\infty$. Se muestran H$\\gamma$, H$\\delta$ y el límite.",
             [rf"\lambda_{{H\gamma}}\ (5\to2) = {la[5]:.3f}\ \mathrm{{nm}},\qquad \lambda_{{H\delta}}\ (6\to2) = {la[6]:.3f}\ \mathrm{{nm}}",
              rf"\lambda_{{lim}} = \dfrac{{4}}{{R_\infty}} = {series_limit:.3f}\ \mathrm{{nm}},\qquad \text{{(Lyman-}}\alpha\ 2\to1:\ \lambda = {ly:.3f}\ \mathrm{{nm}}\text{{)}}"]),
        ],
        numeric_steps=[
            ("Ecuación radial en unidades atómicas", "Para $l=0$ se escribe $\\psi = u(r)/r\\cdot Y_{00}$ y la ecuación queda $-\\tfrac12u'' - u/r = Eu$ con $u(0)=u(\\infty)=0$, en hartrees y radios de Bohr $a_0$. Se discretiza en $N=32000$ nodos hasta $r_{max}=160\\,a_0$ (suficiente para $n$ pequeños, cuyo tamaño es $\\sim n^2a_0$).",
             [rf"h = \dfrac{{r_{{max}}}}{{N+1}} = \dfrac{{160}}{{32001}} = {sci(h, 5)}\ a_0,\qquad r_i = i\,h,\quad i=1,\dots,N"]),
            ("Elementos de la matriz", "Con $u''\\approx(u_{i+1}-2u_i+u_{i-1})/h^2$ la ecuación es $-\\tfrac{1}{2h^2}u_{i-1} + (\\tfrac{1}{h^2} - \\tfrac{1}{r_i})u_i - \\tfrac{1}{2h^2}u_{i+1} = Eu_i$: una matriz tridiagonal simétrica.",
             [rf"H_{{ii}} = \dfrac{{1}}{{h^2}} - \dfrac{{1}}{{r_i}},\qquad H_{{i,i\pm1}} = -\dfrac{{1}}{{2h^2}} = {sci(-0.5 / h**2, 5)}",
              rf"H_{{11}} = {sci(1 / h**2, 5)} - \dfrac{{1}}{{{h:.6f}}} = {sci(1 / h**2 - 1 / h, 5)}\ \mathrm{{Ha}}"]),
            ("Versión reducida a mano: N = 4", "Con $r_{max}=10$, $N=4$, $h=2$ los nodos son $r = 2, 4, 6, 8$ y la matriz es muy pequeña. Es demasiado gruesa para dar la energía correcta, pero muestra el mecanismo: el autovalor más bajo es la mejor estimación de $E_1$.",
             [rf"H = \begin{{pmatrix}} {dg[0]:.4f} & {of[0]:.4f} & 0 & 0\\ {of[0]:.4f} & {dg[1]:.4f} & {of[0]:.4f} & 0\\ 0 & {of[0]:.4f} & {dg[2]:.4f} & {of[0]:.4f}\\ 0 & 0 & {of[0]:.4f} & {dg[3]:.4f} \end{{pmatrix}}",
              rf"\text{{autovalores: }}{wt[0]:.4f},\ {wt[1]:.4f},\ {wt[2]:.4f},\ {wt[3]:.4f}\ \mathrm{{Ha}}\ \ (\text{{exacto }}E_1=-0.5,\ E_2=-0.125,\ E_3 = -0.0556)"]),
            ("Diagonalización a escala real", "El programa usa el mismo planteamiento con 32000 nodos y extrae los 8 autovalores más bajos con `eigh_tridiagonal`. Cada energía en hartrees se convierte a eV multiplicando por 27.2114.",
             [rf"E_n^{{num}}\ \mathrm{{(Ha)}} = ({e[0] / p22.HARTREE_EV:.6f},\ {e[1] / p22.HARTREE_EV:.6f},\ {e[2] / p22.HARTREE_EV:.6f},\dots)"]),
            ("Energías en eV frente a Bohr", "La diferencia viene de la discretización (error $O(h^2)$) y de la caja finita; es mayor para $n$ grande porque esos orbitales son más extensos.",
             [rf"E_1:\ {e[0]:.5f}\ \text{{ vs }}\ {le[1]:.5f},\qquad E_2:\ {e[1]:.5f}\ \text{{ vs }}\ {le[2]:.5f},\qquad E_3:\ {e[2]:.5f}\ \text{{ vs }}\ {le[3]:.5f}\ \ \mathrm{{eV}}"]),
            ("Longitud de onda desde las energías numéricas", "Se aplica $\\lambda = hc/(E_n - E_2)$ con $hc = 1239.84198$ eV nm. Para H$\\alpha$ ($n=3$):",
             [rf"E_3 - E_2 = {e[2]:.5f} - ({e[1]:.5f}) = {e[2] - e[1]:.5f}\ \mathrm{{eV}}\ \Rightarrow\ \lambda_{{H\alpha}} = \dfrac{{1239.84198}}{{{e[2] - e[1]:.5f}}} = {ln_[3]:.4f}\ \mathrm{{nm}}"]),
            ("Resultado para H-beta y errores", "Lo mismo con $n=4$. Se compara con la fórmula de Rydberg.",
             [rf"\lambda_{{H\beta}}^{{num}} = {ln_[4]:.4f}\ \mathrm{{nm}}\ \ (\text{{analítico }}{la[4]:.4f})",
              rf"\dfrac{{\lambda^{{num}} - \lambda^{{ana}}}}{{\lambda^{{ana}}}}:\ \ H\alpha = {sci((ln_[3] - la[3]) / la[3], 2)},\quad H\beta = {sci((ln_[4] - la[4]) / la[4], 2)}"]),
        ],
    )


# =============================================================================== RadialProbability
def _radial() -> dict:
    e, r, u = p22.radial_numeric(1)
    hh = r[1] - r[0]
    p = u[:, 0] ** 2
    mn = p22.mean_radius_1s_numeric()
    mpn = p22.most_probable_radius_numeric()
    k = int(np.argmax(p))
    y0, y1, y2 = p[k - 1], p[k], p[k + 1]
    sh = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    nodes = len(r) - 1
    m = len(r) - (1 if len(r) % 2 == 0 else 0)
    norm = p22.simpson(p[:m], r[:m])
    pexact = p22.radial_probability_analytic_1s(r[:4])
    # Simpson reducido en [0, 8]
    def simp(nn):
        xx = np.linspace(0, 8, nn + 1)
        ff = 4 * xx**3 * np.exp(-2 * xx)
        return xx, ff, p22.simpson(ff, xx)

    x4, f4, s4 = simp(4)
    x8, f8, s8 = simp(8)
    pr = 4 * np.exp(-2)
    pin = 1 - 5 * np.exp(-2)
    return dict(
        explanation=[
            "En el estado fundamental del hidrógeno la función de onda $\\psi_{1s} \\propto e^{-r/a_0}$ depende solo de la distancia al núcleo. Su valor es máximo en el origen, pero eso no significa que el electrón se encuentre más probablemente allí: lo relevante es la probabilidad de encontrarlo en una capa esférica de radio $r$ y espesor $dr$.",
            "Esa probabilidad es $P(r)\\,dr = |\\psi|^2\\,4\\pi r^2\\,dr$. El factor $4\\pi r^2$ (el volumen de la capa) crece con $r$ mientras que $|\\psi|^2$ decae exponencialmente; el producto tiene un máximo a una distancia finita, el radio más probable, que resulta ser exactamente el radio de Bohr $a_0 = 0.0529$ nm.",
            "El radio medio $\\langle r\\rangle = 1.5\\,a_0$ es mayor que el más probable porque la distribución es asimétrica, con una cola larga hacia afuera. Ambos son magnitudes distintas: una es la moda y la otra el promedio. El electrón se encuentra a más de $a_0$ del núcleo con probabilidad cercana a 0.68.",
            "Numéricamente, la ecuación radial da $u(r) = r R(r)$; su cuadrado $u^2$ es justamente $P(r)$. Esto permite calcular radios característicos de manera independiente de la fórmula analítica y comprobar que el método numérico reproduce la solución exacta.",
        ],
        analytic_steps=[
            ("Función de onda y densidad de probabilidad", "En unidades de $a_0$ la función de onda es $\\psi_{1s}=e^{-r}/\\sqrt\\pi$. La probabilidad por unidad de volumen es su cuadrado.",
             [r"\psi_{1s} = \dfrac{1}{\sqrt{\pi a_0^3}}\,e^{-r/a_0},\qquad |\psi_{1s}|^2 = \dfrac{1}{\pi a_0^3}\,e^{-2r/a_0}"]),
            ("Probabilidad radial", "Se multiplica por el volumen $4\\pi r^2\\,dr$ de la capa esférica.",
             [r"P(r) = 4\pi r^2\,|\psi_{1s}|^2 = \dfrac{4r^2}{a_0^3}\,e^{-2r/a_0}"]),
            ("Comprobar la normalización", "Con $\\int_0^\\infty r^ne^{-2r/a_0}dr = n!\\,(a_0/2)^{n+1}$ y $n=2$ se tiene $2(a_0/2)^3 = a_0^3/4$.",
             [r"\int_0^\infty P(r)\,dr = \dfrac{4}{a_0^3}\cdot\dfrac{a_0^3}{4} = 1"]),
            ("Radio más probable", "Se anula la derivada de $P$: $dP/dr = \\tfrac{4}{a_0^3}\\,(2r - 2r^2/a_0)\\,e^{-2r/a_0} = 0$. La raíz distinta de cero es $r=a_0$.",
             [r"\dfrac{d}{dr}\left(r^2e^{-2r/a_0}\right) = 2r\left(1 - \dfrac{r}{a_0}\right)e^{-2r/a_0} = 0\ \Rightarrow\ r_{mp} = a_0",
              rf"P(a_0) = \dfrac{{4}}{{a_0}}\,e^{{-2}} = {pr:.5f}\,/a_0"]),
            ("Radio medio", "El promedio es $\\langle r\\rangle=\\int_0^\\infty rP(r)\\,dr$, con $n=3$: $3!\\,(a_0/2)^4 = 3a_0^4/8$.",
             [r"\langle r\rangle = \dfrac{4}{a_0^3}\int_0^\infty r^3e^{-2r/a_0}\,dr = \dfrac{4}{a_0^3}\cdot\dfrac{3a_0^4}{8} = \dfrac{3}{2}\,a_0"]),
            ("Valores en metros y probabilidad de estar dentro de a0", "Se multiplica por $a_0 = 0.529177\\times10^{-10}$ m. La probabilidad acumulada hasta $r=a_0$ es $1 - 5e^{-2}$.",
             [rf"r_{{mp}} = a_0 = 0.0529177\ \mathrm{{nm}},\qquad \langle r\rangle = 1.5a_0 = {1.5 * 0.0529177:.5f}\ \mathrm{{nm}}",
              rf"P(r<a_0) = 1 - 5e^{{-2}} = {pin:.4f}\ \Rightarrow\ P(r>a_0) = {1 - pin:.4f}"]),
        ],
        numeric_steps=[
            ("Obtener u(r) numérico", "Se diagonaliza la matriz tridiagonal del problema radial (el mismo de la escena del espectro) con 32000 nodos hasta $160\\,a_0$ y se toma el autovector del menor autovalor, normalizado con $\\sum u_i^2h = 1$. La probabilidad radial es $P=u^2$.",
             [rf"h = {sci(hh, 5)}\ a_0,\qquad E_1 = {e[0] / p22.HARTREE_EV:.7f}\ \mathrm{{Ha}}\ \ (\text{{exacto }}{-0.5:.7f})"]),
            ("Comprobar los primeros nodos", "En los primeros nodos $P_{num}$ debe seguir a $4r^2e^{-2r}$; se muestran los tres primeros.",
             [rf"r_1 = {r[0]:.6f}:\ \ P_{{num}} = {sci(p[0], 5)},\ \ P_{{ana}} = {sci(pexact[0], 5)}",
              rf"r_2 = {r[1]:.6f}:\ \ P_{{num}} = {sci(p[1], 5)},\ \ P_{{ana}} = {sci(pexact[1], 5)};\qquad r_3 = {r[2]:.6f}:\ \ P_{{num}} = {sci(p[2], 5)},\ \ P_{{ana}} = {sci(pexact[2], 5)}"]),
            ("Regla de Simpson: mecánica con N = 4 y 8", "Para ver la mecánica, se integra $rP = 4r^3e^{-2r}$ entre 0 y 8 con $h=2$ ($N=4$) y $h=1$ ($N=8$). La fórmula $\\tfrac h3[f_0 + 4f_1 + 2f_2 + 4f_3 + f_4]$ en el caso $N=4$ es:",
             [rf"f(0,2,4,6,8) = ({f4[0]:.4f},\ {f4[1]:.4f},\ {f4[2]:.4f},\ {f4[3]:.5f},\ {sci(f4[4], 3)})",
              rf"S_4 = \dfrac{{2}}{{3}}\left[{f4[0]:.4f} + 4({f4[1]:.4f}) + 2({f4[2]:.4f}) + 4({f4[3]:.5f}) + {sci(f4[4], 3)}\right] = {s4:.5f}",
              rf"S_8 = \dfrac{{1}}{{3}}\left[f_0 + 4f_1 + 2f_2 + \dots + f_8\right] = {s8:.5f}\qquad (\text{{exacto con límite 8: }}1.5\ \text{{menos cola}})"]),
            ("Radio medio con la malla real", "Con los $r_i$ y $P_i=u_i^2$ se forma $r_iP_i$ y se integra con Simpson. Como Simpson necesita un número par de subintervalos, se usan $m=31999$ nodos (31998 subintervalos).",
             [rf"h = {sci(hh, 5)},\qquad \int P\,dr \approx \dfrac{{h}}{{3}}\,[\dots] = {norm:.7f}\ (\approx 1)",
              rf"\langle r\rangle^{{num}} = \dfrac{{h}}{{3}}\,\left[r_1P_1 + 4r_2P_2 + 2r_3P_3 + \dots\right] = {mn:.6f}\,a_0\ \ (\text{{analítico }}1.500000)"]),
            ("Radio más probable: nodo máximo", "Se busca el índice $k$ con $P_k$ máximo. Con un paso $h=0.005\\,a_0$ el nodo ya está muy cerca de $r=a_0$, pero se afina.",
             [rf"k = {k}\ \to\ r_k = {r[k]:.5f}\,a_0,\qquad P_{{k-1}} = {p[k - 1]:.6f},\ P_k = {p[k]:.6f},\ P_{{k+1}} = {p[k + 1]:.6f}"]),
            ("Ajuste parabólico a tres puntos", "Una parábola que pasa por tres puntos equiespaciados tiene su vértice desplazado del punto central $\\delta = \\tfrac12\\,(P_{k-1}-P_{k+1})/(P_{k-1} - 2P_k + P_{k+1})$ en unidades de $h$.",
             [rf"\delta = \dfrac{{1}}{{2}}\,\dfrac{{{y0:.6f} - {y2:.6f}}}{{{y0:.6f} - 2({y1:.6f}) + {y2:.6f}}} = {sh:.4f}",
              rf"r_{{mp}}^{{num}} = r_k + \delta\,h = {r[k]:.5f} + ({sh:.4f})({sci(hh, 4)}) = {mpn:.6f}\,a_0"]),
            ("Comparación con el resultado exacto", "Los errores son del orden de $h^2$ (discretización de la derivada) y de la caja finita.",
             [rf"\langle r\rangle:\ \ {mn:.6f}\ \text{{ vs }}\ 1.500000\ \ (\Delta = {sci(mn - 1.5, 2)}),\qquad r_{{mp}}:\ \ {mpn:.6f}\ \text{{ vs }}\ 1.000000\ \ (\Delta = {sci(mpn - 1, 2)})"]),
        ],
    )


DETAIL = {
    "BlackbodySpectrum": _blackbody(),
    "ComptonPhotoelectric": _compton(),
    "ElectronInterference": _interference(),
    "WavePacketUncertainty": _packet(),
    "InfiniteWell": _well(),
    "BarrierTunneling": _tunnel(),
    "HydrogenSpectrum": _hydrogen(),
    "RadialProbability": _radial(),
}
