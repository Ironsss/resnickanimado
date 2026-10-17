"""Renderiza todas las escenas declaradas en content/chapters.py.

Uso:
    python scripts/render_all.py            # calidad definida en manim.cfg (1080p30)
    python scripts/render_all.py --low      # 480p15, para iterar rapido
    python scripts/render_all.py ch02       # solo un capitulo
    python scripts/render_all.py --shard 0/8   # reparte las escenas en 8 grupos (CI)
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
    ap.add_argument("--shard", default="0/1", help="i/n: renderiza solo las escenas con indice % n == i")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    failed = []
    jobs = [(ch, sc) for ch in load_chapters() for sc in ch.scenes
            if not args.chapters or ch.slug in args.chapters]
    for idx, (ch, sc) in enumerate(jobs):
        if idx % shard_n != shard_i:
            continue
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
