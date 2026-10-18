"""Desarrollo conceptual y analitico general (campo `concept`) de las escenas cuanticas del volumen 2 (cap. 19-22).

Cada escena: A) conceptual, B) deduccion general desde principios, C) resultado general,
D) particularizacion a los datos del enunciado (todo numero se calcula con physics/), E) errores y limites.
"""
from __future__ import annotations

import numpy as np

from content.texfmt import fx, sci
from physics import v2ch19_photons as m19
from physics import v2ch20_matter_waves as m20
from physics import v2ch21_schrodinger as m21
from physics import v2ch22_atom as m22
from physics.constants import C_LIGHT, E_CH, EPS0, H_PLANCK, HBAR, K_B, M_E, M_P, SIGMA_SB

EV = E_CH


# =============================================================================== BlackbodySpectrum
def _blackbody() -> dict:
    t = 5800.0
    t2 = 3000.0
    b = m19.WIEN_B
    xw = 5.0
    for _ in range(60):
        xw -= ((5 - xw) * np.exp(xw) - 5) / ((4 - xw) * np.exp(xw))
    b_calc = H_PLANCK * C_LIGHT / (xw * K_B)
    sig = 2 * np.pi**5 * K_B**4 / (15 * H_PLANCK**3 * C_LIGHT**2)
    lam = b / t
    lam2 = b / t2
    m_tot = SIGMA_SB * t**4
    e_peak = H_PLANCK * C_LIGHT / lam / EV
    x_at = H_PLANCK * C_LIGHT / (lam * K_B * t)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Todo cuerpo a temperatura finita emite radiación electromagnética, y un cuerpo negro ideal (una cavidad con una pequeña abertura) la emite con un espectro que depende únicamente de la temperatura, no del material. La física clásica predecía una emisión que crecía sin límite hacia el ultravioleta, resultado absurdo conocido como catástrofe ultravioleta; resolverlo obligó a introducir la cuantización de la energía de los modos del campo. El mismo espectro describe el color de una estrella, la radiación cósmica de fondo a 2.7 K, el filamento de una lámpara y la pirometría de hornos. Estudiar la ley de Planck significa pasar de contar modos del campo a obtener, con estadística, las leyes de Wien y de Stefan-Boltzmann.",
         []),
        ("Magnitudes y convenciones",
         r"La densidad espectral de energía $u(\lambda,T)$ se mide en $\mathrm{J/m^3}$ por unidad de longitud de onda, y la emitancia espectral $M_\lambda$ (potencia emitida por unidad de área y de longitud de onda) en $\mathrm{W/m^3}$. Las constantes son $h = 6.626\times10^{-34}\ \mathrm{J\,s}$, $k = 1.381\times10^{-23}\ \mathrm{J/K}$ y $c = 2.998\times10^{8}\ \mathrm{m/s}$; la combinación $hc/k \approx 1.44\times10^{-2}\ \mathrm{m\,K}$ fija la escala de longitudes de onda térmicas. La temperatura es siempre absoluta (kelvin). A 5800 K el máximo cae en el visible (centenas de nm) y a 300 K en el infrarrojo medio (unos 10 µm).",
         []),
        ("Intuición y analogías",
         r"Una cavidad se comporta como una colección de osciladores independientes, uno por cada onda estacionaria posible. La equipartición clásica da a cada oscilador la misma energía $kT$, y como hay más modos cuanto mayor es la frecuencia, la suma diverge. En el tratamiento cuántico cada modo solo puede intercambiar energía en paquetes $\hbar\omega$: si $\hbar\omega \gg kT$ el modo casi nunca recibe un cuanto y queda «congelado», y esa supresión exponencial es la que corta la divergencia. Es la misma lógica por la que un escalón de energía muy alto no se puebla a baja temperatura.",
         []),
        ("Modos de la cavidad y densidad de estados",
         r"Se parte de un campo electromagnético en una cavidad cúbica de lado $L$ con paredes conductoras, de modo que el campo eléctrico tangencial se anula en ellas. Cada componente es una onda estacionaria con vector de onda $\vec k = (\pi/L)(n_x,n_y,n_z)$, $n_i = 1,2,\dots$, y frecuencia $\omega = c|\vec k|$. Cada $\vec k$ admite dos polarizaciones transversales independientes. Contar los modos con $|\vec k|<K$ equivale a tomar un octante de esfera en el espacio de los $n_i$, lo cual es legítimo cuando $KL\gg\pi$, es decir, para longitudes de onda mucho menores que la cavidad.",
         [r"N(\omega) = 2\cdot\dfrac18\cdot\dfrac{4\pi}{3}\left(\dfrac{\omega L}{\pi c}\right)^3 = \dfrac{V\omega^3}{3\pi^2c^3}",
          r"g(\omega) = \dfrac{dN}{d\omega} = \dfrac{V\,\omega^2}{\pi^2c^3}\qquad (V = L^3)"]),
        ("Estadística de Bose-Einstein de cada modo",
         r"Cada modo es un oscilador armónico de frecuencia $\omega$, con niveles $E_n = \hbar\omega(n+\tfrac12)$, donde $n$ es el número de fotones del modo. Los fotones son bosones sin número conservado, por lo que el potencial químico es nulo y la probabilidad de que el modo tenga $n$ fotones es proporcional a $e^{-\beta E_n}$ con $\beta = 1/kT$. La función de partición es una serie geométrica (converge porque $\beta\hbar\omega>0$) y la energía media sale de $-\partial\ln Z/\partial\beta$. Se descarta la energía de punto cero $\hbar\omega/2$ porque no depende de $T$ y no se irradia; lo que queda es la ocupación de Bose-Einstein.",
         [r"Z_\omega = \sum_{n=0}^{\infty}e^{-\beta\hbar\omega(n+1/2)} = \dfrac{e^{-\beta\hbar\omega/2}}{1-e^{-\beta\hbar\omega}}",
          r"\langle E_\omega\rangle = -\dfrac{\partial\ln Z_\omega}{\partial\beta} = \dfrac{\hbar\omega}{2} + \dfrac{\hbar\omega}{e^{\hbar\omega/kT}-1}",
          r"\langle n\rangle = \dfrac{1}{e^{\hbar\omega/kT}-1}"]),
        ("Densidad espectral de energía y límites",
         r"La energía por unidad de volumen y de frecuencia es el producto de la densidad de modos por la energía media por modo. El resultado, la ley de Planck en frecuencia, tiene dos límites instructivos: para $\hbar\omega\ll kT$ el denominador se linealiza y reaparece la equipartición clásica (Rayleigh-Jeans), divergente; para $\hbar\omega\gg kT$ domina la exponencial y se obtiene la forma de Wien. La interpolación entre ambos regímenes está gobernada por el único parámetro adimensional $\hbar\omega/kT$.",
         [r"u(\omega,T) = \dfrac{g(\omega)}{V}\,\dfrac{\hbar\omega}{e^{\hbar\omega/kT}-1} = \dfrac{\hbar\omega^3}{\pi^2c^3}\,\dfrac{1}{e^{\hbar\omega/kT}-1}",
          r"\hbar\omega\ll kT:\ u\to\dfrac{\omega^2kT}{\pi^2c^3},\qquad \hbar\omega\gg kT:\ u\to\dfrac{\hbar\omega^3}{\pi^2c^3}e^{-\hbar\omega/kT}"]),
        ("De la densidad de energía a la emitancia por longitud de onda",
         r"Como la conservación de la energía exige $u_\lambda\,|d\lambda| = u_\omega\,d\omega$, se sustituye $\omega = 2\pi c/\lambda$ y $|d\omega| = 2\pi c\,d\lambda/\lambda^2$. Para pasar de la energía almacenada a la potencia que sale por una abertura de área unidad se usa que la radiación es isótropa: el flujo por una superficie vale $c\,u\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\,\tfrac{2\pi}{4\pi} = cu/4$. Así se obtiene la forma de la ley de Planck usada en la escena.",
         [r"u_\lambda = u_\omega\,\dfrac{2\pi c}{\lambda^2} = \dfrac{8\pi hc}{\lambda^5}\,\dfrac{1}{e^{hc/\lambda kT}-1}",
          r"M_\lambda = \dfrac c4\,u_\lambda = \dfrac{2\pi hc^2}{\lambda^5}\,\dfrac{1}{e^{hc/\lambda kT}-1}"]),
        ("Variable adimensional y desplazamiento de Wien",
         r"Con $x = hc/(\lambda kT)$ toda la dependencia de la temperatura se factoriza: $M_\lambda = \dfrac{2\pi k^5T^5}{h^4c^3}\,\dfrac{x^5}{e^x-1}$. El máximo respecto de $\lambda$ coincide con el máximo de $f(x)=x^5/(e^x-1)$, porque $x$ es proporcional a $1/\lambda$ con un factor que no depende de $\lambda$ y el prefactor es constante. Derivar e igualar a cero produce una ecuación trascendente con una única raíz positiva $x_W$, universal. De ella se despeja el producto $\lambda_{max}T$, que solo contiene constantes fundamentales.",
         [r"f'(x)=0\ \Rightarrow\ 5(e^x-1)=xe^x\ \Rightarrow\ (5-x)\,e^{x}=5",
          r"x_W = 4.965114\ldots\quad\Rightarrow\quad b = \lambda_{max}T = \dfrac{hc}{x_W\,k}"]),
        ("Ley de Stefan-Boltzmann por integración",
         r"La emitancia total es la integral de $M_\lambda$ sobre todas las longitudes de onda. Con el cambio $\lambda\to x$ (que invierte los límites) el integrando queda proporcional a $x^3/(e^x-1)$, integral que se resuelve desarrollando $1/(e^x-1)=\sum_{n\ge1}e^{-nx}$, legítimo porque $x>0$, e integrando término a término: $\int_0^\infty x^3e^{-nx}dx = 6/n^4$. La suma $\sum n^{-4}=\pi^4/90$ da el valor final, y el resultado crece como $T^4$ sin que haya que conocer nada más.",
         [r"M = \int_0^\infty M_\lambda\,d\lambda = \dfrac{2\pi k^4T^4}{h^3c^2}\int_0^\infty\dfrac{x^3}{e^x-1}\,dx",
          r"\int_0^\infty\dfrac{x^3}{e^x-1}dx = \sum_{n=1}^\infty\dfrac{6}{n^4} = 6\cdot\dfrac{\pi^4}{90} = \dfrac{\pi^4}{15}",
          r"M = \sigma T^4,\qquad \sigma = \dfrac{2\pi^5k^4}{15\,h^3c^2}"]),
        ("Resultado general",
         r"Todas las propiedades del cuerpo negro dependen solo de $T$ y de $h$, $c$, $k$. El máximo se desplaza como $1/T$, la potencia total como $T^4$ y la forma de la curva es universal en la variable $x$. Dimensionalmente, $[hc/k]=\mathrm{J\,s\cdot m/s\cdot K/J}=\mathrm{m\,K}$, así que $b$ tiene unidades correctas, y $\sigma$ se reduce a $\mathrm{W\,m^{-2}K^{-4}}$ porque $k^4/(h^3c^2)$ tiene unidades $\mathrm{J^4K^{-4}/(J^3s^3\cdot m^2s^{-2})}=\mathrm{J\,s^{-1}m^{-2}K^{-4}}$.",
         [r"\boxed{\;M_\lambda(\lambda,T)=\dfrac{2\pi hc^2}{\lambda^5}\,\dfrac{1}{e^{hc/\lambda kT}-1},\qquad \lambda_{max}=\dfrac{hc}{x_Wk\,T},\qquad M=\dfrac{2\pi^5k^4}{15h^3c^2}\,T^4\;}"]),
        ("Particularización a 5800 K",
         r"Se sustituyen las constantes y $T = 5800$ K. Primero se resuelve $(5-x)e^x=5$ por Newton y se comprueba que $hc/(x_Wk)$ reproduce la constante de Wien; luego se evalúa $\sigma$ con la fórmula de Planck, se obtiene $\lambda_{max}$ y la emitancia total, y se confirma que en el máximo $hc/(\lambda_{max}kT)=x_W$.",
         [rf"x_W = {xw:.6f},\qquad b = \dfrac{{({sci(H_PLANCK, 6)})({sci(C_LIGHT, 6)})}}{{({xw:.6f})({sci(K_B, 6)})}} = {sci(b_calc, 6)}\ \mathrm{{m\,K}}",
          rf"\lambda_{{max}} = \dfrac{{b}}{{T}} = \dfrac{{{sci(b, 6)}}}{{{t:.0f}}} = {lam * 1e9:.4f}\ \mathrm{{nm}},\qquad \lambda_{{max}}(3000\ \mathrm{{K}}) = {lam2 * 1e9:.4f}\ \mathrm{{nm}}",
          rf"\sigma = \dfrac{{2\pi^5({sci(K_B, 6)})^4}}{{15({sci(H_PLANCK, 6)})^3({sci(C_LIGHT, 6)})^2}} = {sci(sig, 6)}\ \mathrm{{W\,m^{{-2}}K^{{-4}}}}",
          rf"\dfrac{{P}}{{A}} = \sigma T^4 = ({sci(sig, 5)})({t:.0f})^4 = {m_tot / 1e6:.4f}\ \mathrm{{MW/m^2}},\qquad \dfrac{{hc}}{{\lambda_{{max}}kT}} = {x_at:.5f}"]),
        ("Errores comunes y límites de validez",
         r"Es frecuente creer que el máximo en longitud de onda y el máximo en frecuencia coinciden: no es así, porque $M_\lambda\,d\lambda = M_\nu\,d\nu$ introduce un jacobiano $1/\lambda^2$ y las constantes de Wien difieren ($x_W\approx4.965$ en $\lambda$ frente a $2.821$ en $\nu$). Tampoco debe confundirse la energía por fotón en el máximo, $hc/\lambda_{max}\approx" + f"{e_peak:.3f}" + r"\ \mathrm{eV}$, con $kT$. Los cuerpos reales emiten $M=\varepsilon\sigma T^4$ con emisividad $\varepsilon<1$ y el modelo del cuerpo negro supone equilibrio térmico y cavidad grande frente a $\lambda$. Para temperaturas muy bajas o cavidades de tamaño comparable a $\lambda$ la densidad de modos continua deja de valer y hay que contar modos discretos.",
         [r"M_\nu\,d\nu = M_\lambda\,|d\lambda|\ \Rightarrow\ \lambda_{max}\nu_{max}\ne c"]),
    ])


# =============================================================================== ComptonPhotoelectric
def _compton() -> dict:
    lam0 = 71.0e-12
    lc = m19.LAMBDA_C
    d90 = float(m19.compton_shift_analytic(np.pi / 2))
    d180 = float(m19.compton_shift_analytic(np.pi))
    lam1 = lam0 + d90
    e0 = H_PLANCK * C_LIGHT / lam0
    e1 = H_PLANCK * C_LIGHT / lam1
    k_el = e0 - e1
    h_fit, phi_fit, dh, dphi = m19.millikan_fit()
    phi = m19.PHI_EV
    f0 = phi * E_CH / H_PLANCK
    slope = H_PLANCK / E_CH
    f_ex = 1.0e15
    v_ex = float(m19.stopping_potential_analytic(f_ex))
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El efecto fotoeléctrico (emisión de electrones por una superficie iluminada) y el efecto Compton (dispersión de rayos X por electrones casi libres) son las dos pruebas experimentales decisivas de que la luz intercambia energía y momento en cuantos localizados, los fotones. Ninguno puede explicarse con una onda clásica continua: en el primero la energía de los electrones no depende de la intensidad y existe una frecuencia umbral; en el segundo la longitud de onda dispersada cambia con el ángulo. Las aplicaciones van desde las celdas solares y los fotomultiplicadores hasta la dosimetría de rayos X y gamma y la astrofísica de altas energías.",
         []),
        ("Magnitudes y convenciones",
         r"La energía de un fotón es $E = hf = hc/\lambda$ y su momento $p = h/\lambda = E/c$; en SI, $h = 6.626\times10^{-34}\ \mathrm{J\,s}$ y es habitual expresar energías en eV ($1\ \mathrm{eV} = 1.602\times10^{-19}\ \mathrm{J}$). La función de trabajo $\phi$ es la energía mínima para extraer un electrón (unos 2 a 5 eV en metales) y el potencial de frenado $V_0$ se mide en voltios. En Compton, $\theta$ es el ángulo de dispersión del fotón respecto de su dirección incidente y $\lambda_C = h/(m_ec)$ es la longitud de onda Compton del electrón, del orden de $10^{-12}$ m.",
         []),
        ("Intuición y analogías",
         r"El efecto fotoeléctrico se parece a sacar agua de un pozo con baldes: cada fotón es un balde de energía fija $hf$, y solo si ese balde supera la profundidad $\phi$ sale algo; muchos baldes pequeños no suman, porque un electrón absorbe un solo fotón a la vez. El efecto Compton es un choque elástico entre dos partículas, como una bola de billar que golpea a otra en reposo: la que golpea pierde energía, y cuanto más la desvía, más pierde. La diferencia es que aquí el proyectil viaja a la velocidad $c$ y hay que usar cinemática relativista.",
         []),
        ("Efecto fotoeléctrico: balance de energía",
         r"Un fotón es absorbido por un electrón de la superficie. Si la energía sobrante tras vencer la barrera $\phi$ se convierte en energía cinética, el electrón más energético sale con $K_{max} = hf-\phi$ (los demás pierden energía adicional en el material). El potencial de frenado es el que anula la corriente: $eV_0 = K_{max}$. De aquí se deduce un umbral $f_0=\phi/h$ por debajo del cual no hay emisión, una pendiente universal $h/e$ en la gráfica $V_0(f)$, y que la intensidad solo controla cuántos electrones salen, no su energía.",
         [r"hf = \phi + K_{max},\qquad K_{max} = eV_0",
          r"V_0(f) = \dfrac{h}{e}\,f - \dfrac{\phi}{e},\qquad f_0 = \dfrac{\phi}{h}"]),
        ("Cinemática relativista del choque fotón-electrón",
         r"El electrón se toma libre y en reposo (hipótesis válida si la energía del fotón es mucho mayor que la ligadura atómica). Las dos leyes de conservación que gobiernan el choque son la de energía total relativista y la del momento lineal, que al ser un vector fija un triángulo de momentos. Los datos de entrada son $E$, $E'$ (energías del fotón antes y después) y el ángulo $\theta$; se elimina el momento final del electrón elevando al cuadrado.",
         [r"E + m_ec^2 = E' + E_e,\qquad \vec p - \vec p' = \vec p_e",
          r"(p_ec)^2 = E^2 + E'^2 - 2EE'\cos\theta,\qquad E_e^2 = (p_ec)^2 + m_e^2c^4"]),
        ("Deducción del corrimiento Compton",
         r"Se iguala $E_e^2 = (E-E'+m_ec^2)^2$ con la expresión anterior. Los términos $E^2$, $E'^2$ y $m_e^2c^4$ se cancelan a ambos lados, y queda una relación lineal entre $EE'(1-\cos\theta)$ y $m_ec^2(E-E')$. Dividir por $EE'm_ec^2$ y usar $E = hc/\lambda$ transforma la relación en una diferencia de longitudes de onda, que depende solo de $\theta$ y de la constante $\lambda_C$, no de la energía incidente.",
         [r"EE'(1-\cos\theta) = m_ec^2\,(E-E')",
          r"\dfrac{1}{E'} - \dfrac{1}{E} = \dfrac{1-\cos\theta}{m_ec^2}",
          r"\Delta\lambda = \lambda'-\lambda = \dfrac{h}{m_ec}\,(1-\cos\theta) = \lambda_C\,(1-\cos\theta)"]),
        ("Casos límite y energía transferida",
         r"En $\theta=0$ no hay corrimiento; en $\theta=\pi/2$ vale $\lambda_C$; en retrodispersión, $\theta=\pi$, alcanza su máximo $2\lambda_C$. El cambio relativo $\Delta\lambda/\lambda$ solo es apreciable cuando $\lambda$ es comparable con $\lambda_C$, es decir, para rayos X duros y gamma; con luz visible es del orden de $10^{-5}$ y no se observa. La energía cinética que recibe el electrón se obtiene de la conservación de energía, y es máxima en retrodispersión.",
         [r"K_e = E - E' = E\,\dfrac{(E/m_ec^2)(1-\cos\theta)}{1+(E/m_ec^2)(1-\cos\theta)}",
          r"\dfrac{\Delta\lambda}{\lambda} = \dfrac{E}{m_ec^2}\,(1-\cos\theta)\quad\text{(con }E=hc/\lambda\text{)}"]),
        ("Estimación de h y φ por mínimos cuadrados",
         r"Con $N$ datos $(f_i,V_i)$ afectados de ruido gaussiano independiente de varianza $s^2$, el modelo lineal $V=af+b$ se ajusta minimizando $\sum(V_i-af_i-b)^2$, lo cual conduce a las ecuaciones normales. La relación con la física es $h = e\,a$ y $\phi = -e\,b$. La matriz de covarianza de los parámetros es $s^2(X^TX)^{-1}$, donde $s^2$ se estima con los residuos y $N-2$ grados de libertad; el estimador es insesgado y de mínima varianza entre los lineales (teorema de Gauss-Markov).",
         [r"\begin{pmatrix}a\\ b\end{pmatrix} = (X^TX)^{-1}X^T\vec V,\qquad X = \begin{pmatrix}f_1 & 1\\ \vdots & \vdots\\ f_N & 1\end{pmatrix}",
          r"\mathrm{Cov}(a,b) = s^2\,(X^TX)^{-1},\qquad s^2 = \dfrac{1}{N-2}\sum_i (V_i-af_i-b)^2",
          r"h = e\,a,\qquad \phi = -e\,b"]),
        ("Resultado general",
         r"Las dos leyes cuantitativas son el corrimiento de longitud de onda, que solo depende del ángulo, y la recta del potencial de frenado, cuya pendiente universal $h/e$ es independiente del material. Comprobación dimensional: $h/(m_ec)$ tiene unidades $\mathrm{J\,s/(kg\,m/s)} = \mathrm{m}$, y $h/e$ tiene $\mathrm{J\,s/C} = \mathrm{V\,s}$, de modo que $(h/e)f$ es un voltaje.",
         [r"\boxed{\;\Delta\lambda(\theta)=\dfrac{h}{m_ec}\,(1-\cos\theta),\qquad eV_0(f)=hf-\phi,\qquad f_0=\dfrac{\phi}{h}\;}"]),
        ("Particularización al problema",
         r"Con $\lambda = 71.0$ pm se calculan la longitud de onda Compton, el corrimiento a 90° y a 180°, y la energía cedida al electrón. Para el sodio se usa $\phi = 2.30$ eV; el ajuste de los 25 puntos sintéticos devuelve $h$ y $\phi$ con sus incertidumbres, que se comparan con los valores de entrada.",
         [rf"\lambda_C = \dfrac{{{sci(H_PLANCK, 6)}}}{{({sci(M_E, 5)})({sci(C_LIGHT, 6)})}} = {lc * 1e12:.5f}\ \mathrm{{pm}}",
          rf"\Delta\lambda(90^\circ) = \lambda_C(1-0) = {d90 * 1e12:.5f}\ \mathrm{{pm}},\quad \Delta\lambda(180^\circ) = 2\lambda_C = {d180 * 1e12:.5f}\ \mathrm{{pm}},\quad \lambda'(90^\circ) = {lam1 * 1e12:.4f}\ \mathrm{{pm}}",
          rf"E = \dfrac{{hc}}{{\lambda}} = {e0 / EV / 1e3:.4f}\ \mathrm{{keV}},\quad E'(90^\circ) = {e1 / EV / 1e3:.4f}\ \mathrm{{keV}},\quad K_e = {k_el / EV:.2f}\ \mathrm{{eV}}",
          rf"f_0 = \dfrac{{\phi}}{{h}} = {sci(f0, 4)}\ \mathrm{{Hz}},\quad \dfrac{{h}}{{e}} = {sci(slope, 5)}\ \mathrm{{V\,s}},\quad V_0({sci(f_ex, 2)}\ \mathrm{{Hz}}) = {v_ex:.4f}\ \mathrm{{V}}",
          rf"h_{{ajuste}} = ({h_fit * 1e34:.4f}\pm{dh * 1e34:.4f})\times10^{{-34}}\ \mathrm{{J\,s}},\qquad \phi_{{ajuste}} = {phi_fit:.4f}\pm{dphi:.4f}\ \mathrm{{eV}}\ \ (\text{{entrada: }}{phi:.2f})"]),
        ("Errores comunes y límites de validez",
         r"No debe aplicarse la fórmula de Compton a electrones ligados con energías de fotón comparables a la ligadura, ni intentar observar un efecto Compton con luz visible; tampoco hay que confundirlo con la dispersión Thomson, que es el límite clásico $E\ll m_ec^2$ sin cambio de longitud de onda. En el fotoeléctrico es un error frecuente pensar que una luz más intensa a frecuencia subumbral acabará arrancando electrones: la absorción multifotónica solo aparece con láseres de intensidades enormes. Además, el electrón no puede absorber un fotón libre y quedar libre sin más (violaría la conservación del momento); en el fotoeléctrico el retroceso lo absorbe la red. Los datos reales tienen potencial de contacto y distribución de energías, por lo que la pendiente es la magnitud fiable, no la ordenada.",
         [r"\text{Libre: }\gamma+e\to e\ \text{imposible};\qquad \text{Thomson: }E\ll m_ec^2\Rightarrow\Delta\lambda/\lambda\to0"]),
    ])


# =============================================================================== ElectronInterference
def _interference() -> dict:
    ke = m20.KE_EV
    d, big_l = m20.D_SLIT, m20.L_SCR
    k = ke * EV
    mc2 = M_E * C_LIGHT**2
    p_rel = np.sqrt(k * (k + 2 * mc2)) / C_LIGHT
    lam = m20.de_broglie()
    lam_nr = H_PLANCK / np.sqrt(2 * M_E * k)
    dy = m20.fringe_spacing_analytic()
    d2l = d**2 / lam
    ang = m20.bragg_angle_analytic()
    lam_b = m20.de_broglie(54.0)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"De Broglie propuso que a toda partícula de momento $p$ le corresponde una onda de longitud $\lambda = h/p$. Los experimentos de difracción de electrones por cristales (Davisson y Germer) y la interferencia de electrones aislados en una doble rendija lo confirmaron: aunque cada electrón llega al detector como un impacto puntual, la distribución de muchos impactos reproduce un patrón de franjas. Esta dualidad onda-partícula fundamenta la microscopía electrónica, la difracción de neutrones en cristalografía y la interferometría atómica.",
         []),
        ("Magnitudes y convenciones",
         r"La energía cinética $K$ se da en eV; el momento $p$ en $\mathrm{kg\,m/s}$ y la longitud de onda en picómetros o ångström. En la doble rendija, $d$ es la separación entre rendijas, $L$ la distancia a la pantalla, $\theta$ el ángulo de observación e $y$ la posición sobre la pantalla medida desde el eje de simetría. Un electrón de 100 eV tiene $\lambda\sim10^{-10}$ m, comparable a los espaciados atómicos, y por eso los cristales funcionan como redes de difracción para electrones.",
         []),
        ("Intuición y analogías",
         r"La amplitud de probabilidad de llegar a un punto de la pantalla es la suma de las amplitudes por cada rendija, igual que las ondas de dos fuentes en una cubeta de agua. Lo que se detecta es el cuadrado del módulo de esa suma, y como las dos amplitudes tienen una fase relativa que depende de la posición, aparecen máximos donde se refuerzan y ceros donde se anulan. Cuando se envía un electrón por vez la figura se construye gota a gota: cada impacto es aleatorio, pero la densidad de impactos sigue la intensidad ondulatoria.",
         []),
        ("Longitud de onda de De Broglie y energía relativista",
         r"La relación $\lambda=h/p$ es válida en todo régimen; lo que cambia es la expresión de $p$ en función de la energía cinética $K$. Partiendo de $E^2=(pc)^2+(mc^2)^2$ y $E=K+mc^2$ se obtiene el momento relativista, cuyo límite no relativista $K\ll mc^2$ da la conocida fórmula de $h/\sqrt{2mK}$. El parámetro que decide si hay que corregir es $K/mc^2$: para 100 eV es $2\times10^{-4}$, de modo que la corrección es del orden de $10^{-4}$.",
         [r"p = \dfrac{1}{c}\sqrt{K\,(K+2mc^2)},\qquad \lambda = \dfrac{hc}{\sqrt{K(K+2mc^2)}}",
          r"K\ll mc^2:\quad \lambda\simeq\dfrac{h}{\sqrt{2mK}}\left(1-\dfrac{K}{4mc^2}\right)"]),
        ("Onda libre de materia y velocidad de grupo",
         r"La ecuación de Schrödinger libre admite ondas planas $\psi = e^{i(kx-\omega t)}$ con $p=\hbar k$ y $E=\hbar\omega=\hbar^2k^2/2m$ (caso no relativista). La velocidad de fase $\omega/k=v/2$ no es observable por sí sola; la partícula viaja a la velocidad de grupo $d\omega/dk=\hbar k/m=v$, que es la del centro del paquete. Esto justifica que se identifique la longitud de onda con $2\pi/k=h/p$ sin que ello contradiga la velocidad $v$ de la partícula.",
         [r"i\hbar\,\partial_t\psi = -\dfrac{\hbar^2}{2m}\partial_x^2\psi\ \Rightarrow\ \omega=\dfrac{\hbar k^2}{2m}",
          r"v_\varphi = \dfrac{\omega}{k} = \dfrac{p}{2m},\qquad v_g = \dfrac{d\omega}{dk} = \dfrac{p}{m} = v"]),
        ("Superposición de dos rendijas",
         r"Para $L\gg d$ las rendijas se ven desde la pantalla casi bajo el mismo ángulo, de modo que las dos amplitudes tienen módulos prácticamente iguales y solo difieren en la fase. La diferencia de camino es $\delta=d\sin\theta$ y la de fase $\Delta\varphi=2\pi\delta/\lambda$. Se suman las amplitudes antes de elevar al cuadrado (regla de Born), lo que produce el término de interferencia; se desprecia la difracción de cada rendija individual, válido si su ancho es menor que $d$.",
         [r"\psi = \psi_0\left(1 + e^{i\Delta\varphi}\right),\qquad \Delta\varphi = \dfrac{2\pi d\sin\theta}{\lambda}",
          r"I(\theta)=|\psi|^2 = 4|\psi_0|^2\cos^2\!\left(\dfrac{\pi d\sin\theta}{\lambda}\right)"]),
        ("Aproximación paraxial y espaciado de franjas",
         r"Para ángulos pequeños, $\sin\theta\simeq\tan\theta = y/L$, y la intensidad queda como función periódica de $y$. Los máximos aparecen cuando el argumento del coseno vale $m\pi$, es decir, en $y_m=m\lambda L/d$, de modo que el espaciado $\Delta y=\lambda L/d$ es uniforme. La condición de campo lejano (Fraunhofer) es $L\gg d^2/\lambda$; sin ella las trayectorias desde cada rendija no son paralelas y el patrón se deforma.",
         [r"I(y)\propto\cos^2\!\left(\dfrac{\pi d\,y}{\lambda L}\right),\qquad y_m = m\,\dfrac{\lambda L}{d}",
          r"\Delta y = \dfrac{\lambda L}{d},\qquad \text{campo lejano: } L\gg\dfrac{d^2}{\lambda}"]),
        ("Difracción por un cristal: condición de Bragg",
         r"Dos planos atómicos paralelos separados una distancia $d_B$ reflejan el haz, y la diferencia de camino entre los rayos reflejados por planos consecutivos es $2d_B\sin\theta$ (con $\theta$ medido desde el plano). El refuerzo ocurre cuando esa diferencia es un múltiplo entero de $\lambda$. Como $\sin\theta\le1$, solo hay difracción si $\lambda\le2d_B/n$. Contar impactos aleatorios introduce además un error estadístico: el espaciado estimado a partir de $N$ detecciones tiene una incertidumbre relativa que escala como $1/\sqrt N$.",
         [r"2d_B\sin\theta = n\lambda\ \Rightarrow\ \theta_n = \arcsin\dfrac{n\lambda}{2d_B}",
          r"\dfrac{\delta(\Delta y)}{\Delta y}\sim\dfrac{1}{\sqrt N}"]),
        ("Resultado general",
         r"El patrón de franjas es periódico en $y$ con un espaciado que crece con $\lambda$ y $L$ y disminuye con $d$. Dimensionalmente, $\lambda L/d$ es $\mathrm{m\cdot m/m}=\mathrm{m}$, y $hc/\sqrt{K(K+2mc^2)}$ es $\mathrm{J\,m/J}=\mathrm{m}$.",
         [r"\boxed{\;\lambda=\dfrac{hc}{\sqrt{K(K+2mc^2)}},\qquad I(y)\propto\cos^2\!\dfrac{\pi dy}{\lambda L},\qquad \Delta y=\dfrac{\lambda L}{d},\qquad 2d_B\sin\theta=\lambda\;}"]),
        ("Particularización al problema",
         r"Con $K=100$ eV, $d=1.0\ \mu\mathrm{m}$ y $L=1.0$ m se calcula el momento relativista, la longitud de onda (comparada con la aproximación no relativista), el espaciado y la comprobación de campo lejano. Para el cristal se usa $K=54$ eV y $d_B=0.215$ nm.",
         [rf"p = \dfrac{{1}}{{c}}\sqrt{{K(K+2mc^2)}} = {sci(p_rel, 5)}\ \mathrm{{kg\,m/s}}",
          rf"\lambda = \dfrac{{h}}{{p}} = {lam * 1e12:.4f}\ \mathrm{{pm}}\quad(\text{{no relativista: }}{lam_nr * 1e12:.4f}\ \mathrm{{pm}})",
          rf"\Delta y = \dfrac{{\lambda L}}{{d}} = \dfrac{{({sci(lam, 5)})({big_l:.1f})}}{{{sci(d, 2)}}} = {dy * 1e6:.3f}\ \mu\mathrm{{m}},\qquad \dfrac{{d^2}}{{\lambda}} = {d2l * 1e3:.2f}\ \mathrm{{mm}}\ll L",
          rf"\lambda(54\ \mathrm{{eV}}) = {lam_b * 1e12:.3f}\ \mathrm{{pm}},\qquad \theta = \arcsin\dfrac{{\lambda}}{{2d_B}} = {ang:.4f}^\circ"]),
        ("Errores comunes y límites de validez",
         r"Un error frecuente es suponer que cada electrón se «divide» y pasa por ambas rendijas en pedazos: lo que se divide es la amplitud de probabilidad, y cada detección es indivisible. Si se mide por cuál rendija pasa, la interferencia desaparece, porque las dos alternativas dejan de ser indistinguibles. Otro error es usar $h/\sqrt{2mK}$ para energías de keV o mayores, donde la corrección relativista ya no es despreciable. El modelo de rendijas puntuales, la onda plana incidente y la coherencia se pierden si la fuente tiene demasiada dispersión de energía o la pantalla está demasiado cerca.",
         []),
    ])


# =============================================================================== WavePacketUncertainty
def _packet() -> dict:
    s0 = m20.SIGMA0
    dp0 = HBAR / (2 * s0)
    ts = 2 * M_E * s0**2 / HBAR
    st = {t_: float(m20.packet_width_analytic(t_)) for t_ in (0.5e-15, 1e-15)}
    vs = HBAR / (2 * M_E * s0)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"En mecánica cuántica una partícula libre localizada se describe con un paquete de ondas, y el principio de incertidumbre establece un límite insuperable a la precisión simultánea de posición y momento. No es un defecto de los instrumentos, sino una propiedad de las ondas: un pulso corto en el espacio necesita muchas longitudes de onda para formarse. El ensanchamiento del paquete en el tiempo explica por qué un electrón confinado a escalas atómicas se desparrama en femtosegundos, y por qué el concepto de trayectoria deja de ser útil.",
         []),
        ("Magnitudes y convenciones",
         r"$\Delta x$ y $\Delta p$ son desviaciones estándar de las distribuciones $|\psi(x)|^2$ y $|\phi(p)|^2$, medidas en m y en $\mathrm{kg\,m/s}$; el producto tiene unidades de acción, $\mathrm{J\,s}$, como $\hbar=1.0546\times10^{-34}\ \mathrm{J\,s}$. Aquí $\sigma_0$ es el ancho inicial en posición, $k=p/\hbar$ el número de onda y $\sigma(t)$ el ancho en el instante $t$. Para un electrón con $\sigma_0=0.1$ nm el tiempo característico es del orden de femtosegundos.",
         []),
        ("Intuición y analogías",
         r"Es el mismo hecho conocido en señales: un pulso breve tiene un ancho de banda grande, y una nota casi pura tarda en sonar. Aquí la «frecuencia» es el número de onda. Cuando el paquete se deja evolucionar libremente, sus componentes de distinto $k$ viajan a velocidades distintas, y como un pelotón de corredores con ritmos diferentes, el grupo se estira con el tiempo aunque cada corredor mantiene su paso.",
         []),
        ("Paquetes de onda y transformada de Fourier",
         r"Un estado libre se escribe como superposición de ondas planas con amplitud $\phi(k)$, relacionada con $\psi(x)$ por una transformada de Fourier unitaria. La unitariedad (teorema de Parseval) garantiza que la normalización se conserva en ambos espacios, de modo que $|\psi|^2$ y $|\phi|^2$ son densidades de probabilidad legítimas. La relación $p=\hbar k$ convierte el ancho en $k$ en ancho en momento.",
         [r"\psi(x)=\dfrac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\phi(k)\,e^{ikx}\,dk,\qquad \phi(k)=\dfrac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\psi(x)\,e^{-ikx}\,dx",
          r"\int|\psi|^2dx=\int|\phi|^2dk=1,\qquad p=\hbar k"]),
        ("Desigualdad de Robertson-Heisenberg",
         r"Para dos observables hermíticos $A$ y $B$ se definen $\delta A=A-\langle A\rangle$ y análogamente $\delta B$. La desigualdad de Cauchy-Schwarz en el espacio de Hilbert, aplicada a los vectores $\delta A\psi$ y $\delta B\psi$, acota el producto de varianzas. El producto $\delta A\delta B$ se descompone en una parte conmutador (imaginaria pura) y otra anticonmutador (real); como ambas contribuyen con signo positivo al cuadrado, basta con conservar la primera. Con $[x,p]=i\hbar$ resulta la desigualdad de Heisenberg, con igualdad si y solo si $\delta p\,\psi=i\gamma\,\delta x\,\psi$ con $\gamma$ real, cuya solución es un gaussiano.",
         [r"(\Delta A)^2(\Delta B)^2 \ge |\langle\delta A\,\delta B\rangle|^2 = \tfrac14|\langle[A,B]\rangle|^2 + \tfrac14\langle\{\delta A,\delta B\}\rangle^2",
          r"\Delta A\,\Delta B\ \ge\ \tfrac12\,|\langle[A,B]\rangle|\ \Rightarrow\ \Delta x\,\Delta p\ \ge\ \dfrac{\hbar}{2}"]),
        ("El paquete gaussiano inicial",
         r"Se toma $\psi(x,0)$ real y gaussiano con centro en el origen y momento medio nulo. Su transformada de Fourier es de nuevo un gaussiano, y los anchos cumplen $\sigma_k=1/(2\sigma_0)$, por lo que el producto es el mínimo permitido. Es un estado de mínima incertidumbre en $t=0$; para $t>0$ solo permanece mínimo el producto si no hay dispersión, lo que no ocurre.",
         [r"\psi(x,0) = (2\pi\sigma_0^2)^{-1/4}\,e^{-x^2/4\sigma_0^2},\qquad \phi(k)\propto e^{-k^2\sigma_0^2}",
          r"\Delta x=\sigma_0,\quad \Delta k=\dfrac{1}{2\sigma_0},\quad \Delta p=\dfrac{\hbar}{2\sigma_0}\ \Rightarrow\ \Delta x\,\Delta p=\dfrac{\hbar}{2}"]),
        ("Evolución libre exacta",
         r"Para $V=0$ las ondas planas son estados estacionarios con $\omega(k)=\hbar k^2/2m$, de modo que cada componente solo recibe una fase. Se multiplica $\phi(k)$ por esa fase y se repite la integral gaussiana, que sigue siendo exacta con coeficiente complejo (se completa el cuadrado). El resultado equivale a sustituir $\sigma_0^2$ por $\sigma_0^2+i\hbar t/2m$ en el exponente, y de $|\psi|^2$ se lee la nueva varianza. Como $|\phi(k,t)|^2=|\phi(k,0)|^2$, el ancho en momento no cambia: toda la variación está en $\Delta x$.",
         [r"\phi(k,t)=\phi(k,0)\,e^{-i\hbar k^2t/2m},\qquad \psi(x,t)\propto\exp\!\left[-\dfrac{x^2}{4\left(\sigma_0^2+i\hbar t/2m\right)}\right]",
          r"|\psi|^2\propto\exp\!\left[-\dfrac{x^2\sigma_0^2}{2(\sigma_0^4+\tau^2)}\right],\quad \tau=\dfrac{\hbar t}{2m}"]),
        ("Ancho en función del tiempo y escalas",
         r"De la forma gaussiana de $|\psi|^2$ se lee la varianza $\sigma^2=(\sigma_0^4+\tau^2)/\sigma_0^2$. Define un tiempo de dispersión $t_s=2m\sigma_0^2/\hbar$: para $t\ll t_s$ el paquete conserva su ancho; para $t\gg t_s$ crece linealmente con velocidad $\Delta p/m$, que es la dispersión de velocidades del estado inicial. El producto $\Delta x\Delta p$ crece en el tiempo, siempre por encima de $\hbar/2$. Como $t_s\propto m\sigma_0^2$, para una masa macroscópica el tiempo es astronómico y el ensanchamiento es inobservable.",
         [r"\sigma(t)=\sigma_0\sqrt{1+\left(\dfrac{t}{t_s}\right)^2},\qquad t_s=\dfrac{2m\sigma_0^2}{\hbar}",
          r"t\gg t_s:\ \sigma\to\dfrac{\hbar t}{2m\sigma_0}=\dfrac{\Delta p}{m}\,t,\qquad \Delta x\,\Delta p=\dfrac\hbar2\sqrt{1+(t/t_s)^2}"]),
        ("Resultado general",
         r"El paquete gaussiano libre minimiza $\Delta x\Delta p$ solo en $t=0$ y luego se ensancha de forma hiperbólica, con un tiempo característico $t_s$ que crece con la masa y con el cuadrado del ancho inicial. Dimensionalmente, $\hbar t/(m\sigma_0^2)$ es $\mathrm{J\,s\cdot s/(kg\,m^2)}=1$, adimensional, y $\hbar/2\sigma_0$ tiene unidades $\mathrm{kg\,m/s}$ de momento.",
         [r"\boxed{\;\Delta x\,\Delta p\ge\dfrac\hbar2,\qquad \sigma(t)=\sigma_0\sqrt{1+\left(\dfrac{\hbar t}{2m\sigma_0^2}\right)^2},\qquad \Delta p=\dfrac{\hbar}{2\sigma_0}\ \text{constante}\;}"]),
        ("Particularización al problema",
         r"Para un electrón con $\sigma_0=0.100$ nm se calcula el mínimo de incertidumbre, el ancho del momento, el tiempo de dispersión y los anchos a 0.5 fs y 1 fs, que son los que reproduce la FFT.",
         [rf"\Delta p = \dfrac{{\hbar}}{{2\sigma_0}} = \dfrac{{{sci(HBAR, 5)}}}{{2({sci(s0, 3)})}} = {sci(dp0, 5)}\ \mathrm{{kg\,m/s}},\qquad \Delta x\,\Delta p = {sci(HBAR / 2, 5)}\ \mathrm{{J\,s}}",
          rf"t_s = \dfrac{{2m\sigma_0^2}}{{\hbar}} = \dfrac{{2({sci(M_E, 5)})({sci(s0, 3)})^2}}{{{sci(HBAR, 5)}}} = {ts * 1e15:.4f}\ \mathrm{{fs}},\qquad \dfrac{{\Delta p}}{{m}} = {sci(vs, 4)}\ \mathrm{{m/s}}",
          rf"\sigma(0.5\ \mathrm{{fs}}) = \sigma_0\sqrt{{1+(0.5/{ts * 1e15:.4f})^2}} = {st[0.5e-15] * 1e10:.5f}\ \text{{Å}},\qquad \sigma(1\ \mathrm{{fs}}) = {st[1e-15] * 1e10:.5f}\ \text{{Å}}"]),
        ("Errores comunes y límites de validez",
         r"La incertidumbre no es el error de un aparato: aun con instrumentos perfectos, las dos distribuciones tienen anchos que no pueden reducirse a la vez. Tampoco dice que la medida de una magnitud perturbe necesariamente a la otra; la desigualdad relaciona las dispersiones de estados preparados de forma idéntica. La cota $\hbar/2$ aplica a posición y momento; para energía y tiempo, que no tienen un operador de tiempo en la teoría estándar, hay que reformular. El modelo libre no contempla potenciales ni efectos relativistas: si $\Delta p/m$ se acerca a $c$ el tratamiento no relativista deja de valer.",
         []),
    ])


# =============================================================================== InfiniteWell
def _well() -> dict:
    length = m21.L_WELL
    e1 = float(m21.well_energy_analytic(1))
    en = [float(m21.well_energy_analytic(n)) for n in (1, 2, 3, 4)]
    e1_j = np.pi**2 * HBAR**2 / (2 * M_E * length**2)
    lam12 = H_PLANCK * C_LIGHT / ((en[1] - en[0]) * EV)
    prod = np.pi * np.sqrt(1 / 12 - 1 / (2 * np.pi**2))
    t_rev = 4 * M_E * length**2 / (np.pi * HBAR)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El pozo infinito es el modelo más simple de confinamiento: una partícula que no puede salir de una región. Muestra, con matemáticas elementales, los rasgos centrales de lo cuántico: energías discretas, energía mínima no nula, funciones de onda con nodos y estados ortogonales. Es la primera aproximación a electrones en puntos cuánticos, a la libre circulación de electrones a lo largo de moléculas conjugadas y a nucleones en un núcleo.",
         []),
        ("Magnitudes y convenciones",
         r"El potencial es $V=0$ para $0<x<L$ e infinito fuera; $\psi$ tiene unidades $\mathrm{m^{-1/2}}$ y $|\psi|^2dx$ es la probabilidad de hallar la partícula en $[x,x+dx]$. La energía se da en eV; para un electrón en $L=1$ nm la escala es $\hbar^2/(2mL^2)\approx0.038$ eV. Los números cuánticos $n=1,2,\dots$ cuentan los semiciclos de la onda dentro del pozo.",
         []),
        ("Intuición y analogías",
         r"La onda de materia atrapada es como una cuerda de guitarra con extremos fijos: solo caben ondas con un número entero de semilongitudes de onda, y por eso la energía, ligada a la longitud de onda, está cuantizada. Cuanto más estrecho es el pozo, más corta es la longitud de onda y mayor la energía, lo cual se relaciona con el principio de incertidumbre: confinar en $L$ obliga a un momento de al menos $\hbar/L$ y a una energía cinética mínima.",
         []),
        ("Separación de variables en la ecuación de Schrödinger",
         r"Cuando $V$ no depende del tiempo se propone $\Psi(x,t)=\psi(x)T(t)$. Al dividir la ecuación por $\psi T$, el miembro izquierdo depende solo de $t$ y el derecho solo de $x$; si dos funciones de variables independientes son iguales, ambas son una constante $E$. De ahí salen una ecuación temporal trivial, que da una fase, y la ecuación de autovalores del hamiltoniano. En estos estados la densidad $|\Psi|^2=|\psi|^2$ no cambia con el tiempo, por eso se llaman estacionarios.",
         [r"i\hbar\,\dfrac{\partial\Psi}{\partial t}=-\dfrac{\hbar^2}{2m}\dfrac{\partial^2\Psi}{\partial x^2}+V\Psi\ \Rightarrow\ \dfrac{i\hbar\,T'}{T}=\dfrac{\hat H\psi}{\psi}=E",
          r"T(t)=e^{-iEt/\hbar},\qquad \hat H\psi=-\dfrac{\hbar^2}{2m}\psi''+V\psi=E\psi"]),
        ("Condiciones de contorno y soluciones",
         r"Dentro del pozo $V=0$ y la ecuación se reduce a $\psi''=-k^2\psi$ con $k=\sqrt{2mE}/\hbar$ (solo hay soluciones oscilantes si $E>0$; para $E\le0$ las soluciones son exponenciales o lineales y no pueden anularse en ambos extremos). Fuera la probabilidad es nula porque el potencial es infinito, y la continuidad de $\psi$ impone $\psi(0)=\psi(L)=0$. La derivada puede ser discontinua en las paredes, resultado del límite de un pozo finito cuya profundidad tiende a infinito.",
         [r"\psi''=-k^2\psi\ \Rightarrow\ \psi(x)=A\sin kx+B\cos kx,\qquad k=\dfrac{\sqrt{2mE}}{\hbar}",
          r"\psi(0)=0\Rightarrow B=0;\qquad \psi(L)=0\Rightarrow A\sin kL=0"]),
        ("Cuantización de la energía y normalización",
         r"Una solución no trivial exige $\sin kL=0$, es decir $k_nL=n\pi$; $n=0$ da $\psi\equiv0$ y los $n<0$ repiten estados con signo cambiado. Elevar al cuadrado $k_n$ da los niveles; el nivel más bajo no es cero (energía de punto cero). La constante $A$ se fija con la condición de normalización, y por hermiticidad del hamiltoniano los estados de energías distintas son ortogonales.",
         [r"E_n=\dfrac{\hbar^2k_n^2}{2m}=\dfrac{n^2\pi^2\hbar^2}{2mL^2}=n^2\dfrac{h^2}{8mL^2},\qquad n=1,2,3,\dots",
          r"\int_0^L|\psi_n|^2dx=A^2\dfrac L2=1\ \Rightarrow\ \psi_n(x)=\sqrt{\dfrac2L}\,\sin\dfrac{n\pi x}{L}",
          r"\int_0^L\psi_m\psi_n\,dx=\delta_{mn}"]),
        ("Estado general y evolución temporal",
         r"Como la ecuación es lineal y los $\psi_n$ forman una base completa en $[0,L]$ (serie de Fourier de senos), cualquier estado con $\psi(0)=\psi(L)=0$ se escribe como combinación lineal. Cada componente evoluciona con su fase, y los coeficientes se obtienen proyectando con la ortonormalidad. Al ser las energías proporcionales a $n^2$, el estado es periódico con periodo de recurrencia $T_{rev}=4mL^2/(\pi\hbar)$, pues todas las fases vuelven a coincidir módulo $2\pi$.",
         [r"\Psi(x,t)=\sum_{n=1}^\infty c_n\,\psi_n(x)\,e^{-iE_nt/\hbar},\qquad c_n=\int_0^L\psi_n\,\Psi(x,0)\,dx",
          r"\sum|c_n|^2=1,\qquad \langle E\rangle=\sum|c_n|^2E_n,\qquad T_{rev}=\dfrac{4mL^2}{\pi\hbar}"]),
        ("Valores esperados, incertidumbre y límite clásico",
         r"Para el estado $\psi_n$ se calculan las integrales con identidades trigonométricas: la posición media es el centro, $\langle p\rangle=0$ por la paridad, y $\langle p^2\rangle=2mE_n$ porque $V=0$. El producto $\Delta x\Delta p$ resulta siempre mayor que $\hbar/2$, cumpliendo Heisenberg, y crece con $n$. Para $n\to\infty$ la densidad $|\psi_n|^2$ oscila tan rápido que promediada es uniforme, que es la distribución de una partícula clásica que rebota; es el principio de correspondencia.",
         [r"\langle x\rangle=\dfrac L2,\quad \langle x^2\rangle=L^2\left(\dfrac13-\dfrac{1}{2n^2\pi^2}\right),\quad \langle p^2\rangle=\left(\dfrac{n\pi\hbar}{L}\right)^2",
          r"\Delta x\,\Delta p=\dfrac{\hbar}{2}\,n\pi\sqrt{\dfrac13-\dfrac{2}{n^2\pi^2}}>\dfrac\hbar2"]),
        ("Resultado general",
         r"Los niveles crecen como $n^2$ y como $1/(mL^2)$: pozos más estrechos o partículas más ligeras dan niveles más separados. Dimensionalmente, $\hbar^2/(mL^2)$ es $\mathrm{J^2s^2/(kg\,m^2)}=\mathrm{J}$ y $\sqrt{2/L}$ tiene $\mathrm{m^{-1/2}}$, como requiere la normalización.",
         [r"\boxed{\;E_n=\dfrac{n^2\pi^2\hbar^2}{2mL^2},\qquad \psi_n(x)=\sqrt{\dfrac2L}\sin\dfrac{n\pi x}{L},\qquad \Psi(x,t)=\sum_nc_n\psi_ne^{-iE_nt/\hbar}\;}"]),
        ("Particularización al problema",
         r"Para un electrón en $L=1.0$ nm se evalúa $E_1=\pi^2\hbar^2/(2mL^2)$ y se escalan los demás con $n^2$. La función de onda del tercer estado y su transición con el segundo siguen de la misma fórmula.",
         [rf"E_1 = \dfrac{{\pi^2({sci(HBAR, 5)})^2}}{{2({sci(M_E, 5)})({sci(length, 2)})^2}} = {sci(e1_j, 5)}\ \mathrm{{J}} = {e1:.5f}\ \mathrm{{eV}}",
          rf"E_2 = 4E_1 = {en[1]:.5f},\quad E_3 = 9E_1 = {en[2]:.5f},\quad E_4 = 16E_1 = {en[3]:.5f}\ \mathrm{{eV}}",
          rf"\psi_3(x) = \sqrt{{\dfrac{{2}}{{{sci(length, 2)}\ \mathrm{{m}}}}}}\sin\dfrac{{3\pi x}}{{L}} = ({sci(np.sqrt(2 / length), 5)}\ \mathrm{{m^{{-1/2}}}})\sin(3\pi x/L),\qquad x_{{nodos}}=\dfrac L3,\ \dfrac{{2L}}{{3}}",
          rf"\lambda_{{1\to2}} = \dfrac{{hc}}{{E_2-E_1}} = \dfrac{{hc}}{{3E_1}} = {lam12 * 1e9:.2f}\ \mathrm{{nm}},\qquad \Delta x\Delta p(n{{=}}1) = {prod:.4f}\,\hbar"]),
        ("Errores comunes y límites de validez",
         r"No debe interpretarse $|\psi_n|^2$ como la posición instantánea de una partícula que oscila: en un estado estacionario la densidad no depende del tiempo y tiene nodos donde la probabilidad es nula, por lo que no hay trayectoria. Tampoco debe tomarse $n=0$ como estado fundamental, que es $n=1$. Un pozo real es de profundidad finita: los niveles altos (energía cercana a $V_0$) quedan por debajo del valor de pozo infinito y la onda penetra en las paredes. A energías relativistas, $E_n\sim mc^2$, hay que usar la ecuación de Dirac.",
         []),
    ])


# =============================================================================== BarrierTunneling
def _tunnel() -> dict:
    e_ev, v_ev, a = m21.E_BEAM_EV, m21.V0_EV, m21.A_BAR
    e, v0 = e_ev * EV, v_ev * EV
    k = np.sqrt(2 * M_E * e) / HBAR
    kap = np.sqrt(2 * M_E * (v0 - e)) / HBAR
    t_exact = m21.transmission_analytic()
    t_asym = 16 * e * (v0 - e) / v0**2 * np.exp(-2 * kap * a)
    sh = np.sinh(kap * a)
    t_hand = 1 / (1 + v0**2 * sh**2 / (4 * e * (v0 - e)))
    t12 = m21.transmission_analytic(12.0)
    t8 = m21.transmission_analytic(8.0)
    q12 = np.sqrt(2 * M_E * (12.0 - v_ev) * EV) / HBAR
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"Una partícula clásica con energía menor que la altura de una barrera no puede atravesarla. En cuántica, la función de onda no se anula bruscamente en una región prohibida: decae exponencialmente, y si la barrera es lo bastante delgada queda una amplitud al otro lado. Este efecto túnel gobierna la desintegración alfa, la fusión en el Sol, el microscopio de efecto túnel, los diodos túnel y la memoria flash. Su probabilidad depende de forma exponencial del ancho y de la raíz de la masa, por lo que pequeños cambios en la geometría dan cambios enormes en la corriente.",
         []),
        ("Magnitudes y convenciones",
         r"Se denota $E$ la energía de la partícula, $V_0$ la altura y $a$ el ancho de la barrera; $k=\sqrt{2mE}/\hbar$ es el número de onda fuera de la barrera y $\kappa=\sqrt{2m(V_0-E)}/\hbar$ la constante de atenuación dentro. Ambos se miden en $\mathrm{m^{-1}}$, y $1/\kappa$ es la longitud de penetración (del orden de 0.1 nm para electrones y barreras de unos eV). La transmisión $T$ es adimensional, la fracción de flujo de probabilidad que atraviesa.",
         []),
        ("Intuición y analogías",
         r"Es parecido a la luz que sufre reflexión total interna: más allá de la interfaz hay una onda evanescente que decae, pero si se acerca un segundo prisma a una distancia de una longitud de onda, la luz «salta» al otro lado. En cuántica el papel del índice de refracción lo hace $\sqrt{E-V}$; cuando $E<V$ ese número se vuelve imaginario y la onda pasa de oscilar a decaer. No hay energía negativa ni «préstamo»: es un fenómeno ondulatorio de la amplitud.",
         []),
        ("Planteamiento y corriente de probabilidad",
         r"El potencial es $V=0$ para $x<0$ y $x>a$, y $V=V_0$ en $0<x<a$. Para estados estacionarios se resuelve la ecuación independiente del tiempo en cada región y se empalma. La conservación de la probabilidad, $\partial_t\rho+\partial_xj=0$, implica que la corriente $j$ es constante en estados estacionarios, de modo que flujo incidente, reflejado y transmitido satisfacen $R+T=1$. La transmisión se define como el cociente de corrientes, que para ondas planas con el mismo $k$ a ambos lados es el módulo cuadrado de las amplitudes.",
         [r"j=\dfrac{\hbar}{2mi}\left(\psi^*\psi'-\psi\,\psi^{*\prime}\right),\qquad T=\dfrac{j_{trans}}{j_{inc}},\quad R=\dfrac{j_{refl}}{j_{inc}}",
          r"-\dfrac{\hbar^2}{2m}\psi''+V(x)\psi=E\psi,\qquad \psi,\ \psi'\ \text{continuas (}V\text{ finito)}"]),
        ("Soluciones por regiones y empalme",
         r"Para $E<V_0$ la solución es oscilatoria fuera y exponencial real dentro. A la derecha solo hay onda saliente (no hay fuente en $+\infty$). La continuidad de $\psi$ y $\psi'$ en $x=a$ permite escribir los coeficientes internos en función de $t$, y la continuidad en $x=0$ da una ecuación para $t$. Se eliminan $C$, $D$ y $r$ en ese orden.",
         [r"\psi_I=e^{ikx}+r\,e^{-ikx},\quad \psi_{II}=Ce^{\kappa x}+De^{-\kappa x},\quad \psi_{III}=t\,e^{ikx}",
          r"x=a:\ \ Ce^{\kappa a}+De^{-\kappa a}=te^{ika},\quad \kappa\left(Ce^{\kappa a}-De^{-\kappa a}\right)=ik\,te^{ika}",
          r"x=0:\ \ 1+r=C+D,\quad ik(1-r)=\kappa(C-D)\ \Rightarrow\ 2ik=(\kappa+ik)C+(ik-\kappa)D"]),
        ("Amplitud de transmisión",
         r"De las condiciones en $x=a$ se despejan $C=\tfrac12te^{ika}e^{-\kappa a}(1+ik/\kappa)$ y $D=\tfrac12te^{ika}e^{\kappa a}(1-ik/\kappa)$. Al sustituir en la ecuación de $x=0$ aparecen $(\kappa\pm ik)^2/\kappa$ multiplicando a las exponenciales. Agrupando términos con $\cosh\kappa a$ y $\sinh\kappa a$ queda una expresión cerrada para $t$, cuyo único factor de fase es $e^{-ika}$; por tanto, la onda transmitida solo se desplaza una distancia $a$ y el módulo de $t$ contiene toda la física.",
         [r"2ik=\dfrac{te^{ika}}{2\kappa}\Big[(\kappa+ik)^2e^{-\kappa a}-(\kappa-ik)^2e^{\kappa a}\Big]=\dfrac{te^{ika}}{2\kappa}\Big[4ik\kappa\cosh\kappa a-2(\kappa^2-k^2)\sinh\kappa a\Big]",
          r"t\,e^{ika}=\dfrac{1}{\cosh\kappa a+i\,\dfrac{\kappa^2-k^2}{2k\kappa}\,\sinh\kappa a}"]),
        ("Transmisión: fórmula exacta y continuación E > V0",
         r"Se calcula $T=|t|^2$ usando $\cosh^2=1+\sinh^2$ y la identidad $(\kappa^2+k^2)^2=(2mV_0/\hbar^2)^2$, junto con $k^2\kappa^2=(2m/\hbar^2)^2E(V_0-E)$. Para $E>V_0$ basta con la continuación analítica $\kappa\to iq$, con $q=\sqrt{2m(E-V_0)}/\hbar$: $\sinh\to i\sin$, y $T=1$ cuando $qa=n\pi$ (resonancias: la barrera es transparente porque se interfieren constructivamente las reflexiones de ambos bordes).",
         [r"T=\dfrac{1}{\cosh^2\kappa a+\dfrac{(\kappa^2-k^2)^2}{4k^2\kappa^2}\sinh^2\kappa a}=\left[1+\dfrac{V_0^2\sinh^2\kappa a}{4E(V_0-E)}\right]^{-1}",
          r"E>V_0:\quad T=\left[1+\dfrac{V_0^2\sin^2qa}{4E(E-V_0)}\right]^{-1},\qquad R=1-T"]),
        ("Barrera opaca y aproximación WKB",
         r"Cuando $\kappa a\gg1$, $\sinh\kappa a\simeq e^{\kappa a}/2$ y la transmisión queda como una exponencial: el factor $e^{-2\kappa a}$ domina sobre el prefactor. Para una barrera de forma arbitraria $V(x)$ lenta frente a la longitud de onda, la aproximación WKB generaliza el resultado integrando la constante de atenuación local sobre la región prohibida. La dependencia es muy sensible: duplicar el ancho eleva el exponente al cuadrado.",
         [r"\kappa a\gg1:\quad T\simeq\dfrac{16E(V_0-E)}{V_0^2}\,e^{-2\kappa a}",
          r"T\simeq\exp\!\left[-\dfrac2\hbar\int_{x_1}^{x_2}\sqrt{2m\left(V(x)-E\right)}\,dx\right]"]),
        ("Resultado general",
         r"La transmisión es función de dos parámetros adimensionales: $E/V_0$ y $\kappa a$. Dimensionalmente, $\kappa a$ es adimensional, y $V_0^2/(E(V_0-E))$ también, de modo que cada término suma con el 1. Tiende a 1 si $\kappa a\to0$ y a 0 si $\kappa a\to\infty$, y cumple $T+R=1$.",
         [r"\boxed{\;T(E)=\left[1+\dfrac{V_0^2\sinh^2\kappa a}{4E(V_0-E)}\right]^{-1},\quad \kappa=\dfrac{\sqrt{2m(V_0-E)}}{\hbar};\qquad E>V_0:\ \kappa\to iq\;}"]),
        ("Particularización al problema",
         r"Con $E=5$ eV, $V_0=10$ eV y $a=0.50$ nm se calculan $k$, $\kappa$ y $\kappa a$, y se evalúa la fórmula exacta junto con la aproximación para barrera opaca. Luego se repite para 8 eV y para 12 eV, por encima de la barrera.",
         [rf"k = \dfrac{{\sqrt{{2mE}}}}{{\hbar}} = {k * 1e-9:.4f}\ \mathrm{{nm^{{-1}}}},\qquad \kappa = \dfrac{{\sqrt{{2m(V_0-E)}}}}{{\hbar}} = {kap * 1e-9:.4f}\ \mathrm{{nm^{{-1}}}},\qquad \kappa a = {kap * a:.4f}",
          rf"T = \left[1+\dfrac{{(10)^2\sinh^2({kap * a:.4f})}}{{4(5)(5)}}\right]^{{-1}} = \left[1+\dfrac{{100({sh ** 2:.4f})}}{{100}}\right]^{{-1}} = {sci(t_hand, 4)}",
          rf"T_{{opaca}} = \dfrac{{16(5)(5)}}{{100}}\,e^{{-2\kappa a}} = {sci(t_asym, 4)}\quad(\text{{exacta }}{sci(t_exact, 4)})",
          rf"T(8\ \mathrm{{eV}}) = {t8:.6f},\qquad q(12\ \mathrm{{eV}})\,a = {q12 * a:.4f},\quad T(12\ \mathrm{{eV}}) = {t12:.5f}"]),
        ("Errores comunes y límites de validez",
         r"Es erróneo decir que la partícula tiene «energía negativa» dentro de la barrera: la energía total es $E$, y lo que se vuelve negativo es la energía cinética en el sentido clásico, que no es un observable que pueda medirse en ese estado. Tampoco hay un tiempo finito de «cruce» que se pueda asignar de manera inequívoca. Para $E>V_0$ la transmisión no es 1 salvo en resonancia: hay reflexión cuántica. El modelo supone barrera rectangular de bordes abruptos y haz monoenergético; barreras reales redondeadas piden WKB, y a alta energía o masa muy pequeña hay que usar la ecuación de Dirac.",
         []),
    ])


# =============================================================================== HydrogenSpectrum
def _hydrogen() -> dict:
    ry = m22.RYDBERG_EV
    alpha = E_CH**2 / (4 * np.pi * EPS0 * HBAR * C_LIGHT)
    ry_calc = 0.5 * alpha**2 * M_E * C_LIGHT**2 / EV
    a0 = 4 * np.pi * EPS0 * HBAR**2 / (M_E * E_CH**2)
    rinf = m22.RINF_M
    rinf_calc = alpha**2 * M_E * C_LIGHT / (2 * H_PLANCK)
    en = [float(m22.level_analytic(n)) for n in (1, 2, 3, 4)]
    la = {n: m22.balmer_wavelength_analytic(n) for n in (3, 4)}
    lim = 4e9 / rinf
    mu_fac = 1 / (1 + M_E / M_P)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"El hidrógeno es el único átomo cuya ecuación de Schrödinger se resuelve de forma exacta, y sus líneas espectrales fueron la primera prueba cuantitativa de la teoría cuántica. Las longitudes de onda emitidas siguen una fórmula empírica simple (Balmer, Rydberg) que la mecánica cuántica explica a partir de niveles discretos de energía. Es la base para entender átomos más complejos, la tabla periódica y los espectros estelares; las líneas de hidrógeno sirven además para medir velocidades radiales y temperaturas en astronomía.",
         []),
        ("Magnitudes y convenciones",
         r"Los niveles se denotan por el número cuántico principal $n=1,2,\dots$ y su energía en eV es negativa (estado ligado, cero en la ionización). La constante de Rydberg $R_\infty\approx1.0974\times10^7\ \mathrm{m^{-1}}$ tiene unidades de inverso de longitud, y $\mathrm{Ry}=13.606$ eV es la energía de ionización. El radio de Bohr $a_0=0.0529$ nm es la escala de longitud atómica. Las líneas de Balmer terminan en $n=2$; las de Lyman, en $n=1$, y las de Paschen, en $n=3$.",
         []),
        ("Intuición y analogías",
         r"El electrón ligado es una onda estacionaria alrededor del núcleo, como una cuerda circular: solo ciertos patrones son compatibles con la condición de que la función de onda sea regular en el origen y decaiga en el infinito. Cada patrón tiene energía definida. El átomo no emite en un continuo sino al saltar entre patrones, con un fotón cuya energía es la diferencia; por eso el espectro es de líneas y no una banda continua.",
         []),
        ("Hamiltoniano y masa reducida",
         r"En el sistema de dos cuerpos (electrón y núcleo) la energía cinética del centro de masa se separa, y queda un problema de una partícula con masa reducida $\mu=m_eM/(m_e+M)$ en el potencial coulombiano. Como el potencial solo depende de $r$, el momento angular se conserva y el hamiltoniano conmuta con $L^2$ y $L_z$. El efecto de $\mu$ es pequeño en el hidrógeno ($\mu\simeq m_e(1-m_e/M_p)$), pero es lo que separa el hidrógeno del deuterio.",
         [r"\hat H=-\dfrac{\hbar^2}{2\mu}\nabla^2-\dfrac{e^2}{4\pi\varepsilon_0\,r},\qquad \mu=\dfrac{m_eM}{m_e+M}",
          r"[\hat H,\hat L^2]=[\hat H,\hat L_z]=0"]),
        ("Separación de variables en coordenadas esféricas",
         r"Para un potencial central se propone $\psi=R(r)\,Y_{\ell m}(\theta,\varphi)$. Al sustituir y dividir, el término angular queda aislado y debe ser una constante, que resulta ser $\ell(\ell+1)$ porque las $Y_{\ell m}$ son autofunciones de $L^2$ con autovalor $\hbar^2\ell(\ell+1)$. La parte radial recibe una barrera centrífuga que se suma al potencial. Con $u=rR$ la ecuación radial toma la forma de una ecuación 1D en $r>0$ con $u(0)=0$, que es lo que diagonaliza la parte numérica de la escena.",
         [r"\hat L^2Y_{\ell m}=\hbar^2\ell(\ell+1)Y_{\ell m},\qquad \hat L_zY_{\ell m}=\hbar m\,Y_{\ell m}",
          r"-\dfrac{\hbar^2}{2\mu}u''+\left[\dfrac{\hbar^2\ell(\ell+1)}{2\mu r^2}-\dfrac{e^2}{4\pi\varepsilon_0r}\right]u=Eu,\qquad u(0)=0"]),
        ("Solución radial y cuantización",
         r"En unidades atómicas ($\hbar=\mu=e^2/4\pi\varepsilon_0=1$) la ecuación es $-\tfrac12u''+[\ell(\ell+1)/2r^2-1/r]u=Eu$. Para $E<0$ el comportamiento asintótico es $u\sim e^{-\kappa r}$ con $E=-\kappa^2/2$, y cerca del origen $u\sim r^{\ell+1}$. Se extrae esa estructura, $u=r^{\ell+1}e^{-\kappa r}\sum_ja_jr^j$, y la sustitución en la ecuación da una relación de recurrencia. Si la serie no termina, crece como $e^{2\kappa r}$ y $u$ no es normalizable; por tanto debe truncarse en un $j_{max}$, lo cual fija $\kappa$ como el inverso de un entero.",
         [r"a_{j+1}=\dfrac{2\left[\kappa(j+\ell+1)-1\right]}{(j+1)(j+2\ell+2)}\,a_j",
          r"\text{Termina en }j_{max}\ \Rightarrow\ \kappa=\dfrac1n,\quad n=j_{max}+\ell+1\ge\ell+1",
          r"E_n=-\dfrac{\kappa^2}{2}=-\dfrac{1}{2n^2}\ \text{hartree},\qquad \ell=0,1,\dots,n-1"]),
        ("Niveles con unidades y degeneración",
         r"Al recuperar las unidades, el hartree es $e^4\mu/[(4\pi\varepsilon_0)^2\hbar^2]$ y la energía de Rydberg es su mitad. Se puede reescribir como $\tfrac12\alpha^2\mu c^2$ con la constante de estructura fina $\alpha=e^2/(4\pi\varepsilon_0\hbar c)$, lo que muestra que la escala atómica es $\alpha^2$ veces la energía en reposo. La energía no depende de $\ell$ ni de $m$ (simetría coulombiana adicional), por lo que el nivel $n$ tiene degeneración $n^2$ (o $2n^2$ con el espín).",
         [r"E_n=-\dfrac{\mu e^4}{2(4\pi\varepsilon_0)^2\hbar^2}\,\dfrac1{n^2}=-\dfrac{\alpha^2\mu c^2}{2}\,\dfrac1{n^2}=-\dfrac{\mathrm{Ry}}{n^2}",
          r"a_0=\dfrac{4\pi\varepsilon_0\hbar^2}{\mu e^2},\qquad g_n=\sum_{\ell=0}^{n-1}(2\ell+1)=n^2"]),
        ("Transiciones y series espectrales",
         r"Un fotón emitido en la transición $n\to m$ ($m<n$) lleva la diferencia de energías (condición de frecuencias de Bohr). Dividir por $hc$ expresa el resultado como número de onda. La regla de selección dipolar $\Delta\ell=\pm1$ restringe qué líneas aparecen, pero no cambia sus longitudes de onda. Las series se nombran por el nivel final: Lyman ($m=1$), Balmer ($m=2$) y Paschen ($m=3$). El límite de serie ($n\to\infty$) es la energía de ionización desde el nivel final. Con masa reducida, la constante efectiva es $R_H=R_\infty\,\mu/m_e$.",
         [r"h\nu=E_n-E_m=\mathrm{Ry}\left(\dfrac1{m^2}-\dfrac1{n^2}\right)",
          r"\dfrac1\lambda=R_\infty\left(\dfrac1{m^2}-\dfrac1{n^2}\right),\qquad R_\infty=\dfrac{\alpha^2m_ec}{2h}",
          r"\lambda_{Balmer}=\dfrac{1}{R_\infty}\,\dfrac{4n^2}{n^2-4},\qquad \lambda_{lim}=\dfrac{4}{R_\infty}"]),
        ("Resultado general",
         r"Los niveles y las líneas de cualquier átomo hidrogenoide de carga $Z$ siguen escalando con $Z^2$. Dimensionalmente, $\alpha$ es adimensional, $\alpha^2m_ec^2$ tiene unidades de energía, y $R_\infty=\alpha^2m_ec/2h$ tiene $\mathrm{kg\,m/s\,/J\,s}=\mathrm{m^{-1}}$.",
         [r"\boxed{\;E_n=-\dfrac{Z^2\,\mathrm{Ry}}{n^2},\qquad \dfrac1{\lambda_{n\to m}}=Z^2R_\infty\left(\dfrac1{m^2}-\dfrac1{n^2}\right)\;}"]),
        ("Particularización al hidrógeno",
         r"Con $Z=1$ se verifica primero que las constantes fundamentales dan $\mathrm{Ry}$ y $R_\infty$ y luego se evalúan los niveles y las dos primeras líneas de Balmer, así como el límite de serie.",
         [rf"\alpha = \dfrac{{e^2}}{{4\pi\varepsilon_0\hbar c}} = \dfrac{{1}}{{{1 / alpha:.3f}}},\qquad \mathrm{{Ry}} = \dfrac12\alpha^2m_ec^2 = {ry_calc:.5f}\ \mathrm{{eV}}\ \ ({ry:.5f}),\qquad a_0 = {sci(a0, 5)}\ \mathrm{{m}}",
          rf"R_\infty = \dfrac{{\alpha^2m_ec}}{{2h}} = {sci(rinf_calc, 7)}\ \mathrm{{m^{{-1}}}}\ \ ({sci(rinf, 7)})",
          rf"E_1 = {en[0]:.4f},\quad E_2 = {en[1]:.4f},\quad E_3 = {en[2]:.4f},\quad E_4 = {en[3]:.4f}\ \mathrm{{eV}}",
          rf"\lambda_{{H\alpha}} = \dfrac{{1}}{{R_\infty(1/4-1/9)}} = {la[3]:.3f}\ \mathrm{{nm}},\quad \lambda_{{H\beta}} = \dfrac{{1}}{{R_\infty(1/4-1/16)}} = {la[4]:.3f}\ \mathrm{{nm}},\quad \lambda_{{lim}} = {lim:.2f}\ \mathrm{{nm}}",
          rf"\dfrac{{\mu}}{{m_e}} = \dfrac{{1}}{{1+m_e/M_p}} = {mu_fac:.6f}\ \Rightarrow\ \lambda_{{H\alpha}}^{{(\mu)}}\approx{la[3] / mu_fac:.3f}\ \mathrm{{nm}}"]),
        ("Errores comunes y límites de validez",
         r"No debe interpretarse $n$ como el radio de una órbita: el electrón no tiene trayectoria y su ubicación es una distribución radial. Tampoco la energía depende solo de $n$ en el átomo real: la estructura fina (corrección relativista y espín-órbita, de relativo $\alpha^2$) y el desplazamiento de Lamb levantan parte de la degeneración, y la estructura hiperfina agrega el espín nuclear. Para átomos con varios electrones, la repulsión mutua rompe la degeneración en $\ell$. Los espectros medidos en aire dan longitudes de onda ligeramente distintas (índice de refracción) de las del vacío, que son las de la fórmula.",
         []),
    ])


# =============================================================================== RadialProbability
def _radial() -> dict:
    a0 = 4 * np.pi * EPS0 * HBAR**2 / (M_E * E_CH**2)
    r_mp, r_mean = 1.0, 1.5
    r2 = 3.0
    dr = np.sqrt(r2 - r_mean**2)
    p_out = 13 * np.exp(-4)
    # radio mediano
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.exp(-2 * mid) * (1 + 2 * mid + 2 * mid**2) > 0.5:
            lo = mid
        else:
            hi = mid
    r_med = 0.5 * (lo + hi)
    pmax = 4 * np.exp(-2)
    return dict(concept=[
        ("Qué se estudia y por qué importa",
         r"En mecánica cuántica el electrón del hidrógeno no está en una órbita, sino que se describe con una función de onda cuyo cuadrado da la densidad de probabilidad de hallarlo. La distribución radial responde a una pregunta concreta y medible: ¿a qué distancia del núcleo es más probable encontrarlo? Su forma determina el tamaño del átomo, la energía de enlace por el valor medio de $1/r$ y las propiedades químicas. Es el punto de partida de los orbitales de la química cuántica.",
         []),
        ("Magnitudes y convenciones",
         r"La función de onda $\psi_{1s}$ tiene unidades $\mathrm{m^{-3/2}}$ y la densidad $|\psi|^2$, $\mathrm{m^{-3}}$. La densidad radial $P(r)$ es la probabilidad por unidad de longitud ($\mathrm{m^{-1}}$), definida de modo que $P(r)\,dr$ es la probabilidad de encontrar al electrón en una cáscara esférica de radio $r$ y espesor $dr$. La unidad natural es el radio de Bohr $a_0=0.0529$ nm. Los valores esperados $\langle r^k\rangle$ se miden en $a_0^k$.",
         []),
        ("Intuición y analogías",
         r"La densidad $|\psi|^2$ es máxima en el núcleo y decae monótonamente, pero la probabilidad radial se anula en el origen: hay muy poco volumen cerca de $r=0$. El máximo de $P(r)$ es el resultado de una competencia entre el volumen de la cáscara, que crece como $r^2$, y la densidad, que decae como $e^{-2r/a_0}$. Es como buscar a una persona en una ciudad: la mayor densidad está en el centro, pero la mayor cantidad de gente está en el anillo intermedio, porque hay más espacio.",
         []),
        ("Densidad de probabilidad en coordenadas esféricas",
         r"La probabilidad de encontrar al electrón en el elemento de volumen $d^3r=r^2\sin\theta\,dr\,d\theta\,d\varphi$ es $|\psi|^2d^3r$. Para la distribución radial se integra sobre los ángulos, que son independientes de $r$ en un estado de variables separadas. Con $\psi=R\,Y_{\ell m}$ y $Y_{\ell m}$ normalizada sobre la esfera, solo sobrevive $r^2R^2$, que coincide con $u^2$ al definir $u=rR$. Esto conecta la distribución con la ecuación radial de la escena precedente.",
         [r"P(r)\,dr=\left[\int_0^\pi\!\!\int_0^{2\pi}|\psi|^2\sin\theta\,d\theta\,d\varphi\right]r^2dr",
          r"P(r)=r^2R(r)^2=u(r)^2,\qquad \int_0^\infty P(r)\,dr=1"]),
        ("Estado fundamental normalizado",
         r"Para $n=1$, $\ell=0$ la serie de la escena anterior se corta en el primer término: $u=Nre^{-r/a}$, con $a=a_0/Z$ para un ion hidrogenoide. La constante $N$ se fija con $\int u^2dr=1$, y la integral $\int_0^\infty r^ke^{-\beta r}dr=k!/\beta^{k+1}$ (se obtiene por inducción con integración por partes) da el resultado. Como $Y_{00}=1/\sqrt{4\pi}$, la función 3D es esféricamente simétrica.",
         [r"\int_0^\infty r^2e^{-2r/a}dr=\dfrac{2!}{(2/a)^3}=\dfrac{a^3}{4}\ \Rightarrow\ N^2=\dfrac4{a^3}",
          r"P(r)=\dfrac{4r^2}{a^3}\,e^{-2r/a},\qquad \psi_{1s}=\dfrac{1}{\sqrt{\pi a^3}}\,e^{-r/a}"]),
        ("Radio más probable",
         r"El máximo de $P(r)$ se halla anulando su derivada: $P'\propto 2r-\tfrac{2}{a}r^2=0$ da $r=0$ (donde $P$ es mínima, igual a cero) y $r=a$. La segunda derivada en $r=a$ es negativa, lo que confirma que es un máximo. Nótese que no coincide con el máximo de $|\psi|^2$, que está en el origen: son distribuciones respecto de medidas distintas ($d^3r$ frente a $dr$). El valor máximo de $P$ es $4/(ea)$.",
         [r"\dfrac{dP}{dr}=\dfrac{4}{a^3}\,e^{-2r/a}\left(2r-\dfrac{2r^2}{a}\right)=0\ \Rightarrow\ r_{mp}=a",
          r"P(r_{mp})=\dfrac{4}{a}\,e^{-2},\qquad \dfrac{d^2P}{dr^2}\Big|_{a}<0"]),
        ("Momentos radiales y teorema del virial",
         r"El valor esperado de cualquier potencia de $r$ sigue de la misma integral gamma. Resulta $\langle r^k\rangle=(k+2)!\,a^k/2^{k+1}$ para $k>-3$. Esto da el radio medio, la dispersión y $\langle 1/r\rangle=1/a$. Con él, $\langle V\rangle=-(e^2/4\pi\varepsilon_0)\langle1/r\rangle=-2\,\mathrm{Ry}$ y $E=-\mathrm{Ry}$ implican $\langle T\rangle=+\mathrm{Ry}$: se cumple el teorema del virial, $2\langle T\rangle=-\langle V\rangle$.",
         [r"\langle r^k\rangle=\dfrac4{a^3}\int_0^\infty r^{k+2}e^{-2r/a}dr=\dfrac{(k+2)!}{2^{k+1}}\,a^k",
          r"\langle r\rangle=\dfrac32a,\quad \langle r^2\rangle=3a^2,\quad \Delta r=\sqrt{3-\tfrac94}\,a=\dfrac{\sqrt3}{2}\,a,\quad \left\langle\dfrac1r\right\rangle=\dfrac1a"]),
        ("Probabilidad fuera de una esfera: región clásicamente prohibida",
         r"La probabilidad de hallar al electrón más allá de $R$ es la cola de $P$, que se integra por partes dos veces. Un electrón clásico con energía $E=-\mathrm{Ry}$ en el potencial coulombiano no podría pasar de $r=2a$ (donde $E=V$, es decir, toda su energía cinética se anula); el electrón cuántico tiene una probabilidad no nula de estar allí, es otra manifestación del efecto túnel. El radio mediano $r_{1/2}$ (que deja la mitad de la probabilidad a cada lado) cumple $e^{-2x}(1+2x+2x^2)=\tfrac12$.",
         [r"P(r>R)=\int_R^\infty P\,dr=e^{-2R/a}\left(1+\dfrac{2R}{a}+\dfrac{2R^2}{a^2}\right)",
          r"P(r>2a)=13\,e^{-4},\qquad e^{-2x}\left(1+2x+2x^2\right)=\tfrac12\Rightarrow r_{1/2}=x\,a"]),
        ("Resultado general",
         r"Para un ion hidrogenoide de carga $Z$ toda la geometría escala como $a_0/Z$. La forma de la distribución es universal, con la moda en $a$, la media en $3a/2$ y una cola exponencial. Dimensionalmente, $4r^2/a^3\,e^{-2r/a}$ tiene $\mathrm{m^{2}/m^{3}}=\mathrm{m^{-1}}$, que es la unidad correcta para una densidad lineal.",
         [r"\boxed{\;P(r)=\dfrac{4r^2}{a^3}e^{-2r/a},\quad r_{mp}=a,\quad \langle r\rangle=\dfrac32a,\quad \Delta r=\dfrac{\sqrt3}2a,\quad a=\dfrac{a_0}{Z}=\dfrac{4\pi\varepsilon_0\hbar^2}{Zm_ee^2}\;}"]),
        ("Particularización al hidrógeno",
         r"Con $Z=1$, $a=a_0$ se obtienen los radios en unidades de $a_0$ y en metros, que son los que reproduce el cálculo numérico de la ecuación radial; también la dispersión, el valor máximo de $P$ y la probabilidad de la región prohibida.",
         [rf"a_0 = \dfrac{{4\pi\varepsilon_0\hbar^2}}{{m_ee^2}} = {sci(a0, 5)}\ \mathrm{{m}} = {a0 * 1e10:.5f}\ \text{{Å}}",
          rf"r_{{mp}} = a_0 = {r_mp:.1f}\,a_0 = {sci(r_mp * a0, 4)}\ \mathrm{{m}},\qquad \langle r\rangle = \dfrac32a_0 = {r_mean:.1f}\,a_0 = {sci(r_mean * a0, 4)}\ \mathrm{{m}}",
          rf"\langle r^2\rangle = {r2:.1f}\,a_0^2,\qquad \Delta r = \dfrac{{\sqrt3}}{{2}}a_0 = {dr:.4f}\,a_0,\qquad P(r_{{mp}}) = \dfrac{{4}}{{e^2a_0}} = {pmax:.4f}/a_0",
          rf"P(r>2a_0) = 13e^{{-4}} = {p_out:.4f},\qquad r_{{1/2}} = {r_med:.4f}\,a_0"]),
        ("Errores comunes y límites de validez",
         r"La confusión más habitual es identificar el radio más probable con el máximo de la densidad $|\psi|^2$ (que está en el núcleo) o con el radio medio: la distribución es asimétrica y $r_{mp}<\langle r\rangle$. Tampoco es correcto llamar «órbita de Bohr» a la distribución: coincide en el valor $a_0$ solo por casualidad para $n=1$. Para estados con $\ell>0$ el momento angular desplaza el máximo y para $n$ grande aparecen nodos radiales. El modelo desprecia la masa reducida, el espín y las correcciones relativistas, y para $Z$ grande el electrón interno se mueve a velocidades relativistas.",
         []),
    ])


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
