"""Regenera docs/ROADMAP.md desde content.chapters.ROADMAP (fuente unica)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from content.chapters import ROADMAP  # noqa: E402

CRITERIA = """## Criterios de entrada para cada tema nuevo

1. Índice de conceptos tomado del índice de referencia (solo títulos, no texto del libro).
2. Al menos dos escenas por capítulo, cada una con un problema original.
3. Solución analítica en forma cerrada en `physics/` y un método numérico independiente.
4. Pruebas que comparen ambas con una tolerancia justificada por el método.
5. Fila de verificación en `content/chapters.py` y revisión visual del video a 1080p.
"""


def main() -> None:
    out = ["# Hoja de ruta", "",
           "Cobertura de *Física*, vol. 1 y 2, de Resnick, Halliday y Krane, según el índice de referencia del proyecto.",
           "La numeración exacta de capítulos varía entre ediciones; aquí se identifica cada tema por su título.", ""]
    for vol, subtitle, parts in ROADMAP:
        total = sum(len(i) for _, i in parts)
        done = sum(1 for _, i in parts for it in i if it[2] == "Publicado")
        out += [f"## {vol}: {subtitle}", "", f"Avance: {done} de {total} temas.", ""]
        for part, items in parts:
            out += [f"### {part}", "", "| Tema | Conceptos | Estado |", "| --- | --- | --- |"]
            out += [f"| {t} | {c} | {s} |" for t, c, s in items]
            out.append("")
    out.append(CRITERIA)
    (ROOT / "docs" / "ROADMAP.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
