"""Regenera docs/ROADMAP.md desde content.chapters.ROADMAP (fuente unica)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from content.chapters import load_chapters  # noqa: E402
from content.index import INDEX, VOLUMES  # noqa: E402

CRITERIA = """## Criterios de entrada para cada tema nuevo

1. Índice de conceptos tomado del índice de referencia (solo títulos, no texto del libro).
2. Al menos dos escenas por capítulo, cada una con un problema original.
3. Solución analítica en forma cerrada en `physics/` y un método numérico independiente.
4. Pruebas que comparen ambas con una tolerancia justificada por el método.
5. Fila de verificación en `content/chapters.py` y revisión visual del video a 1080p.
"""


def main() -> None:
    published = {(c.volume, c.number) for c in load_chapters()}
    out = ["# Hoja de ruta", "",
           "Cobertura de *Física*, vol. 1 y 2, de Resnick, Halliday y Krane, según el índice de referencia del proyecto.",
           "La numeración de capítulos sigue el orden de ese índice y puede diferir de la de cada edición.", ""]
    for vol, title in VOLUMES.items():
        entries = [t for t in INDEX if t[0] == vol]
        done = sum(1 for n in range(1, len(entries) + 1) if (vol, n) in published)
        out += [f"## Volumen {vol}: {title}", "", f"Avance: {done} de {len(entries)} capítulos.", ""]
        part = None
        for n, (_, prt, name, concepts) in enumerate(entries, start=1):
            if prt != part:
                part = prt
                out += ["", f"### {prt}", "", "| Cap. | Tema | Conceptos | Estado |", "| --- | --- | --- | --- |"]
            state = "Publicado" if (vol, n) in published else "Planificado"
            out.append(f"| {n} | {name} | {concepts} | {state} |")
        out.append("")
    out.append(CRITERIA)
    (ROOT / "docs" / "ROADMAP.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
