"""Renderiza todas las escenas declaradas en content/chapters.py.

Uso:
    python scripts/render_all.py            # calidad definida en manim.cfg (1080p30)
    python scripts/render_all.py --low      # 480p15, para iterar rapido
    python scripts/render_all.py ch02       # solo un capitulo
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from content.chapters import load_chapters  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapters", nargs="*", help="slugs a renderizar (por defecto, todos)")
    ap.add_argument("--low", action="store_true", help="480p15")
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    failed = []
    for ch in load_chapters():
        if args.chapters and ch.slug not in args.chapters:
            continue
        for sc in ch.scenes:
            cmd = [sys.executable, "-m", "manim", "render", "--progress_bar", "none"]
            if args.low:
                cmd.append("-ql")
            cmd += [f"animations/{ch.module}.py", sc.id]
            print(f"[{ch.slug}] {sc.id}", flush=True)
            if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
                failed.append(sc.id)
    if failed:
        print("fallaron:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
