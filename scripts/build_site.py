"""Genera el sitio estatico en ./public a partir de content/ y de los videos renderizados.

Uso:
    python scripts/build_site.py [--media-dir media/videos] [--quality 1080p30] [--out public]
"""
from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from content.chapters import ROADMAP, load_chapters  # noqa: E402
from content.model import Chapter, Row, Scene  # noqa: E402
from content.references import REFERENCES  # noqa: E402

REPO_URL = "https://github.com/ironsss/resnickanimado"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist"
FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
         "&family=JetBrains+Mono:wght@400;500&display=swap")
LOGO = ('<svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="currentColor"/>'
        '<path d="M7 22 Q 16 4 25 22" fill="none" stroke="var(--background)" stroke-width="2.4" stroke-linecap="round"/>'
        '<circle cx="16" cy="9.5" r="2.4" fill="var(--background)"/></svg>')

e = html.escape


# ----------------------------------------------------------------- utilidades
def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def fmt_err(err: float) -> str:
    return "&lt; 1e-12" if err < 1e-12 else f"{err:.1e}"


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def make_poster(video: Path, dest: Path, at: float) -> bool:
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video),
                        "-frames:v", "1", "-vf", "scale=1280:-1", "-q:v", "4", str(dest)], check=True)
        return dest.exists()
    except (OSError, subprocess.CalledProcessError):
        return False


def fmt_duration(sec: float | None) -> str:
    if sec is None:
        return ""
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


# ------------------------------------------------------------------- plantilla
def page(title: str, body: str, base: str, description: str, current: str = "") -> str:
    def nav(href: str, label: str, key: str, cls: str = "") -> str:
        cur = ' aria-current="page"' if current == key else ""
        return f'<a href="{base}{href}"{cur} class="{cls}">{label}</a>'

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(description)}">
<link rel="icon" href="{base}assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<link rel="stylesheet" href="{base}assets/style.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"></script>
<script defer src="{base}assets/app.js"></script>
</head>
<body>
<header class="site-header">
  <div class="container">
    <a class="brand" href="{base}index.html">{LOGO}<span>Resnick animado</span></a>
    <nav class="nav" aria-label="Principal">
      {nav("index.html#capitulos", "Capítulos", "chapters")}
      {nav("index.html#metodologia", "Metodología", "method", "hide-sm")}
      {nav("index.html#hoja-de-ruta", "Hoja de ruta", "roadmap", "hide-sm")}
      {nav("index.html#fuentes", "Fuentes", "sources", "hide-sm")}
      <a href="{REPO_URL}">GitHub</a>
      <button class="btn btn-ghost" id="theme-toggle" type="button" aria-label="Cambiar tema">Tema</button>
    </nav>
  </div>
</header>
<main>
{body}
</main>
<footer class="site-footer">
  <div class="container">
    <p>Proyecto abierto bajo licencia MIT. Los enunciados y las soluciones son originales y se basan en los temas del índice de <em>Física</em>, vol. 1, de Resnick, Halliday y Krane. No se reproduce texto, figuras ni problemas del libro.</p>
    <p>Animaciones generadas con <a href="https://www.manim.community/">Manim Community</a>. Código y datos en <a href="{REPO_URL}">el repositorio</a>.</p>
  </div>
</footer>
</body>
</html>
"""


def video_block(scene: Scene, chapter: Chapter, base: str, media: dict[str, dict]) -> str:
    info = media.get(scene.id)
    if not info:
        return ('<div class="video-frame"><div class="video-missing">'
                'La animación se genera durante la construcción del sitio y aún no está disponible en esta copia.</div></div>')
    poster = f' poster="{base}media/{scene.id}.jpg"' if info.get("poster") else ""
    dur = fmt_duration(info.get("duration"))
    return (f'<div class="video-frame"><video controls preload="metadata" playsinline{poster}>'
            f'<source src="{base}media/{scene.id}.mp4" type="video/mp4"></video></div>'
            f'<div class="video-caption"><span>{e(scene.title)}</span><span>{dur}</span></div>')


def steps_html(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="step"><div>$${e(tex)}$$</div><div class="cap">{e(cap)}</div></div>' for tex, cap in items)


def rows_html(rows: list[Row]) -> str:
    body = "".join(
        f"<tr><td>{e(r.label)}</td><td>${e(r.tex)}$</td>"
        f'<td class="num">{fmt(r.analytic, r.digits)}</td><td class="num">{fmt(r.numeric, r.digits)}</td>'
        f'<td class="num err">{fmt_err(r.error)}</td><td>{e(r.unit) if r.unit else "&mdash;"}</td></tr>'
        for r in rows)
    return ('<div class="table-wrap"><table><thead><tr><th>Magnitud</th><th>Símbolo</th>'
            '<th class="num">Analítico</th><th class="num">Numérico</th><th class="num">Error abs.</th><th>Unidad</th>'
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def scene_html(scene: Scene, chapter: Chapter, idx: int, base: str, media: dict) -> str:
    cmd = f"PYTHONPATH=. manim -qh animations/{chapter.module}.py {scene.id}"
    notes = "".join(f'<p class="note">{e(n)}</p>' for n in scene.notes)
    return f"""
<section class="scene" id="{scene.id}">
  <div class="eyebrow">Animación {chapter.number}.{idx}</div>
  <h2>{e(scene.title)}</h2>
  <p class="statement">{e(scene.statement)}</p>
  {video_block(scene, chapter, base, media)}
  <div class="step-block"><h3><span class="eyebrow">01</span> Fundamentos</h3>{steps_html(scene.foundations)}</div>
  <div class="step-block"><h3><span class="eyebrow">02</span> Solución analítica</h3>{steps_html(scene.analytic)}</div>
  <div class="step-block"><h3><span class="eyebrow">03</span> Solución numérica</h3>
    <p class="method">{e(scene.numeric_method)}</p>
  </div>
  <div class="step-block"><h3><span class="eyebrow">04</span> Verificación</h3>
    {rows_html(scene.rows)}{notes}
  </div>
  <div class="step-block"><h3><span class="eyebrow">05</span> Reproducir</h3>
    <pre class="cmd"><code>{e(cmd)}</code><button class="btn btn-outline btn-sm" type="button" data-copy="{e(cmd)}">Copiar</button></pre>
  </div>
</section>"""


# ---------------------------------------------------------------------- paginas
def chapter_page(ch: Chapter, chapters: list[Chapter], media: dict) -> str:
    base = "../"
    toc = "".join(f'<a href="#{s.id}">{e(s.title)}</a>' for s in ch.scenes)
    concepts = "".join(f'<span class="badge badge-secondary">{e(c)}</span>' for c in ch.concepts)
    scenes = "".join(scene_html(s, ch, i + 1, base, media) for i, s in enumerate(ch.scenes))
    i = chapters.index(ch)
    prev_ = chapters[i - 1] if i > 0 else None
    next_ = chapters[i + 1] if i < len(chapters) - 1 else None
    pager = "<div class=\"pager\">"
    pager += (f'<a href="{prev_.slug}.html"><span class="label">Anterior</span>{prev_.number:02d}. {e(prev_.title)}</a>'
              if prev_ else "<span></span>")
    pager += (f'<a class="next" href="{next_.slug}.html"><span class="label">Siguiente</span>{next_.number:02d}. {e(next_.title)}</a>'
              if next_ else "")
    pager += "</div>"
    body = f"""
<div class="container">
  <nav class="crumbs" aria-label="Ruta"><a href="{base}index.html">Inicio</a><span>/</span><a href="{base}index.html#capitulos">Capítulos</a><span>/</span><span>{ch.number:02d}</span></nav>
  <header class="chapter-head prose-col">
    <div class="eyebrow">Volumen 1 &middot; Capítulo {ch.number}</div>
    <h1>{e(ch.title)}</h1>
    <p class="lead" style="margin-bottom:0">{e(ch.summary)}</p>
    <div class="concepts" aria-label="Índice de conceptos">{concepts}</div>
  </header>
  <div class="layout">
    <aside class="toc"><div class="eyebrow">En esta página</div>{toc}</aside>
    <div>{scenes}{pager}</div>
  </div>
</div>"""
    return page(f"{ch.number:02d}. {ch.title} | Resnick animado", body, base,
                f"Animaciones, solución analítica y verificación numérica: {ch.title}.", "chapters")


def index_page(chapters: list[Chapter], media: dict) -> str:
    base = ""
    n_scenes = sum(len(c.scenes) for c in chapters)
    n_rows = sum(len(s.rows) for c in chapters for s in c.scenes)
    n_concepts = sum(len(c.concepts) for c in chapters)
    featured = chapters[3].scenes[0]
    cards = ""
    for c in chapters:
        tags = "".join(f'<span class="badge badge-muted">{e(t)}</span>' for t in c.concepts[:3])
        cards += (f'<a class="card" href="chapters/{c.slug}.html"><span class="num">CAPÍTULO {c.number:02d}</span>'
                  f"<h3>{e(c.title)}</h3><p>{e(c.summary)}</p><div class=\"tags\">{tags}</div>"
                  f'<div class="foot"><span>{len(c.scenes)} animaciones</span><span>Disponible</span></div></a>')
    road = ""
    for vol, subtitle, parts in ROADMAP:
        n_all = sum(len(items) for _, items in parts)
        n_done = sum(1 for _, items in parts for it in items if it[2] == "Publicado")
        blocks = ""
        for part, items in parts:
            rows = "".join(
                f"<tr><td>{e(t)}</td><td style=\"color:var(--muted-foreground)\">{e(c)}</td>"
                f'<td><span class="badge {"badge-ok" if st == "Publicado" else "badge-muted"}">{e(st)}</span></td></tr>'
                for t, c, st in items)
            blocks += (f'<h3 style="margin:1.5rem 0 .7rem">{e(part)}</h3><div class="table-wrap"><table><thead><tr>'
                       f"<th>Tema</th><th>Conceptos</th><th>Estado</th></tr></thead><tbody>{rows}</tbody></table></div>")
        road += (f'<div><div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem">'
                 f"<h2 style=\"font-size:1.25rem\">{e(vol)}</h2><span class=\"eyebrow\">{n_done} de {n_all} temas</span></div>"
                 f'<p style="color:var(--muted-foreground);margin:.2rem 0 0;font-size:14px">{e(subtitle)}</p>{blocks}</div>')
    refs = ""
    counter = 1
    for group, items in REFERENCES:
        lis = "".join(f"<li>{t} <span style=\"color:var(--muted-foreground)\">({u})</span></li>" for t, u in items)
        refs += f'<div><h3 style="margin-bottom:.6rem">{e(group)}</h3><ol class="refs" start="{counter}">{lis}</ol></div>'
        counter += len(items)
    body = f"""
<div class="container">
  <section class="hero">
    <div class="hero-grid">
      <div>
        <span class="badge">Volumen 1 &middot; capítulos 1 a 5</span>
        <h1>Física de Resnick, Halliday y Krane con rigor numérico</h1>
        <p class="lead">Visualizaciones con Manim de cada tema del volumen 1. Cada problema se resuelve de forma analítica y numérica, y ambas soluciones se contrastan en pantalla.</p>
        <div class="actions">
          <a class="btn btn-primary" href="#capitulos">Ver capítulos</a>
          <a class="btn btn-outline" href="{REPO_URL}">Código fuente</a>
        </div>
      </div>
      <div>{video_block(featured, chapters[3], base, media)}</div>
    </div>
  </section>
  <div class="stats" role="list">
    <div role="listitem"><div class="stat-value">{len(chapters)}</div><div class="stat-label">Capítulos publicados</div></div>
    <div role="listitem"><div class="stat-value">{n_scenes}</div><div class="stat-label">Animaciones</div></div>
    <div role="listitem"><div class="stat-value">{n_concepts}</div><div class="stat-label">Conceptos del índice</div></div>
    <div role="listitem"><div class="stat-value">{n_rows}</div><div class="stat-label">Magnitudes verificadas</div></div>
  </div>
  <section class="block" id="capitulos" style="margin-top:3.5rem">
    <div class="section-head"><div class="eyebrow">Contenido</div><h2>Capítulos</h2>
      <p>Los cinco primeros temas del volumen 1, de la medición a las leyes de Newton.</p></div>
    <div class="grid grid-3">{cards}</div>
  </section>
  <section class="block" id="metodologia">
    <div class="section-head"><div class="eyebrow">Método</div><h2>Cada problema, tres capas</h2>
      <p>Las animaciones no ilustran: calculan. Los números en pantalla provienen de las mismas funciones que se prueban automáticamente.</p></div>
    <div class="grid grid-3">
      <div class="card"><span class="num">01</span><h3>Fundamentos</h3><p>Las definiciones y leyes del tema, con su notación, antes de resolver nada.</p></div>
      <div class="card"><span class="num">02</span><h3>Solución analítica</h3><p>Deducción en forma cerrada, paso a paso, con las ecuaciones tal como se escriben en el libro.</p></div>
      <div class="card"><span class="num">03</span><h3>Solución numérica</h3><p>Integración RK4, álgebra lineal, diferencias finitas o Monte Carlo, según el problema, contrastadas con la solución exacta.</p></div>
    </div>
    <p class="note" style="max-width:46rem;margin-top:1.5rem">La capa <code>physics/</code> no depende de Manim. Cada resultado se comprueba con pruebas automáticas que fallan si analítico y numérico divergen más allá de la tolerancia esperada del método.</p>
  </section>
  <section class="block" id="hoja-de-ruta">
    <div class="section-head"><div class="eyebrow">Plan</div><h2>Hoja de ruta</h2>
      <p>El objetivo es completar los volúmenes 1 y 2 siguiendo el índice de conceptos de referencia del proyecto.</p></div>
    <div class="grid grid-2 roadmap">{road}</div>
  </section>
  <section class="block" id="fuentes">
    <div class="section-head"><div class="eyebrow">Bibliografía</div><h2>Fuentes</h2>
      <p>Texto de referencia, métodos numéricos y software utilizados. Los enunciados y resultados son originales.</p></div>
    <div class="grid" style="gap:2rem">{refs}</div>
  </section>
</div>"""
    return page("Resnick animado | Física con Manim y verificación numérica", body, base,
                "Visualizaciones con Manim de la Física de Resnick, Halliday y Krane, con solución analítica y numérica.", "")


# ------------------------------------------------------------------------ main
def collect_media(chapters: list[Chapter], media_root: Path, quality: str, out: Path) -> dict[str, dict]:
    dest = out / "media"
    dest.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    for ch in chapters:
        for sc in ch.scenes:
            src = media_root / ch.module / quality / f"{sc.id}.mp4"
            if not src.exists():
                alts = sorted((media_root / ch.module).glob(f"*/{sc.id}.mp4"))
                src = alts[-1] if alts else src
            if not src.exists():
                print(f"  aviso: sin video para {sc.id}", file=sys.stderr)
                continue
            shutil.copy2(src, dest / f"{sc.id}.mp4")
            dur = ffprobe_duration(src)
            poster = make_poster(src, dest / f"{sc.id}.jpg", (dur or 10) * 0.93)
            found[sc.id] = {"duration": dur, "poster": poster}
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media-dir", default=str(ROOT / "media" / "videos"))
    ap.add_argument("--quality", default="1080p30")
    ap.add_argument("--out", default=str(ROOT / "public"))
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "chapters").mkdir(parents=True)
    shutil.copytree(ROOT / "web" / "assets", out / "assets")
    (out / ".nojekyll").write_text("")

    chapters = load_chapters()
    media = collect_media(chapters, Path(args.media_dir), args.quality, out)
    (out / "index.html").write_text(index_page(chapters, media), encoding="utf-8")
    for ch in chapters:
        (out / "chapters" / f"{ch.slug}.html").write_text(chapter_page(ch, chapters, media), encoding="utf-8")
    print(f"sitio generado en {out} ({len(media)} videos)")


if __name__ == "__main__":
    main()
