# Resnick animado

Visualización con [Manim](https://www.manim.community/) de los temas del volumen 1 de *Física*, de Resnick, Halliday y Krane, con solución analítica y numérica de cada problema. El sitio se publica en GitHub Pages.

Sitio: <https://ironsss.github.io/resnickanimado/>

## Objetivo

Cada tema del índice de conceptos se trata con tres capas:

1. **Fundamentos.** Definiciones y leyes, con su notación.
2. **Solución analítica.** Deducción en forma cerrada, paso a paso.
3. **Solución numérica.** Un método independiente (integración RK4, álgebra lineal, diferencias finitas, Monte Carlo o búsqueda de raíces) cuyo resultado se contrasta con el analítico.

Los números que muestran las animaciones y el sitio se calculan con las mismas funciones que se prueban automáticamente. Si el método numérico y la solución exacta divergen más allá de la tolerancia esperada, las pruebas fallan.

## Estado

| Capítulo | Tema | Animaciones | Métodos numéricos |
| --- | --- | --- | --- |
| 1 | Medición | Análisis dimensional del péndulo; propagación de incertidumbre | Mínimos cuadrados, RK4, Monte Carlo |
| 2 | Movimiento en una dimensión | Velocidad instantánea y área bajo v(t); aceleración constante y caída libre | Secantes, diferencias centradas, trapecio, Euler y RK4 |
| 3 | Vectores | Suma de vectores; producto escalar y vectorial | Componentes, Monte Carlo, invarianza por rotación |
| 4 | Movimiento en dos y tres dimensiones | Proyectiles; movimiento circular uniforme | RK4 con refinamiento del impacto, sección áurea, diferencias finitas |
| 5 | Fuerza y leyes de Newton | Equilibrio con dos cables; plano inclinado con polea | Sistema lineal, RK4, bisección |

La hoja de ruta para completar los volúmenes 1 y 2 está en [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Estructura

```
physics/        Soluciones analíticas y numéricas por capítulo. No depende de Manim.
animations/     Escenas de Manim y sistema visual compartido (common.py).
content/        Texto, ecuaciones y filas de verificación de cada escena.
web/assets/     Hojas de estilo y scripts del sitio.
scripts/        Renderizado de escenas y generación del sitio estático.
tests/          Pruebas que comparan solución analítica y numérica.
docs/           Hoja de ruta.
.github/        Flujo de trabajo de pruebas, renderizado y despliegue.
```

La separación es deliberada: `physics/` es la fuente de verdad; `animations/` y `content/` solo presentan sus resultados.

## Requisitos

- Python 3.10 o superior
- FFmpeg
- Una distribución de LaTeX (`texlive-latex-base`, `texlive-latex-extra`, `texlive-fonts-recommended`) y `dvisvgm`
- Bibliotecas de sistema de Manim: Cairo y Pango (`libcairo2-dev`, `libpango1.0-dev`, `pkg-config`)
- Tipografías Inter y JetBrains Mono (`fonts-inter`, `fonts-jetbrains-mono`). Sin ellas Manim usa una tipografía de reserva.

En Debian y Ubuntu:

```bash
sudo apt-get install ffmpeg libcairo2-dev libpango1.0-dev pkg-config \
  texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm \
  fonts-inter fonts-jetbrains-mono
```

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verificar que cada solución numérica coincide con la analítica
python -m pytest -q

# Renderizar una escena en baja calidad
PYTHONPATH=. manim -ql animations/ch04_motion_2d.py Projectile

# Renderizar todo (1080p, 30 fps) y construir el sitio
python scripts/render_all.py
python scripts/build_site.py
python -m http.server --directory public 8000
```

`scripts/render_all.py --low` renderiza a 480p y 15 fps para iterar con rapidez. Pasar el identificador de capítulo (`ch02`) limita el renderizado a ese capítulo.

## Añadir un capítulo

1. Crear `physics/chNN_<tema>.py` con la solución analítica y una numérica independiente para cada problema.
2. Añadir pruebas en `tests/` con una tolerancia justificada por el método (por ejemplo, orden de convergencia de Euler frente a RK4).
3. Crear `animations/chNN_<tema>.py` con una clase por escena que herede de `PhysicsScene`. Las fases son: Fundamentos, Problema, Solución analítica y Solución numérica.
4. Declarar el capítulo, sus escenas y las filas de verificación en `content/chapters.py`.
5. Renderizar, revisar los fotogramas y ejecutar `python scripts/build_site.py`.

Los criterios completos de aceptación están en [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Despliegue

El flujo [`.github/workflows/pages.yml`](.github/workflows/pages.yml) se ejecuta en cada push a `main` o `master`: ejecuta las pruebas, renderiza las animaciones a 1080p, genera el sitio y lo publica con GitHub Pages.

Requisito único en el repositorio: en *Settings > Pages*, seleccionar **GitHub Actions** como origen.

## Diseño

El sitio y los videos comparten un sistema visual sobrio inspirado en [shadcn/ui](https://ui.shadcn.com/): escala neutra zinc, bordes finos, tipografía Inter, un acento por función semántica y tema claro y oscuro. En las animaciones, el blanco indica la solución analítica, el azul la numérica, el rojo el error y el ámbar el resultado destacado.

## Alcance y atribución

- Este proyecto es independiente. No está afiliado a los autores ni a los editores del libro.
- Se usa el índice de temas del libro como guía de cobertura. No se reproducen textos, figuras ni problemas del libro. Los enunciados son originales.
- La numeración de capítulos y los títulos del volumen 1 siguen la estructura de la quinta edición en inglés. Deben confirmarse contra la edición que se use, sobre todo a partir del capítulo 6.

Referencia: R. Resnick, D. Halliday y K. S. Krane, *Física*, vol. 1 y 2.

## Licencia

MIT. Véase [`LICENSE`](LICENSE).
