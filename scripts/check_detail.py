"""Valida las explicaciones y soluciones paso a paso. Uso: python scripts/check_detail.py [--render]

Comprueba estructura, que no haya notacion '1.0e+05' dentro de TeX y, con --render, que KaTeX no marque errores."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from content.chapters import load_chapters  # noqa: E402

BAD_E = re.compile(r"\d[eE][+-]?\d")


def tex_ok(t: str) -> bool:
    return t.count("{") == t.count("}") and not BAD_E.search(t.replace("\\mathrm", "").replace("\\text", ""))


def main() -> int:
    bad = 0
    done = 0
    for ch in load_chapters():
        for s in ch.scenes:
            if not (s.explanation or s.analytic_steps or s.numeric_steps):
                print(f"FALTA   {s.id}")
                bad += 1
                continue
            done += 1
            probs = []
            if len(s.explanation) < 2:
                probs.append("explicacion < 2 parrafos")
            if len(s.analytic_steps) < 4:
                probs.append("pasos analiticos < 4")
            if len(s.numeric_steps) < 4:
                probs.append("pasos numericos < 4")
            if len(s.concept) < 8 and "--sin-concepto" not in sys.argv:
                probs.append("concept < 8 secciones")
            if len(s.numeric_intro) < 3:
                probs.append("numeric_intro < 3 parrafos")
            if len([l for l in s.numeric_code.splitlines() if l.strip()]) < 6:
                probs.append("numeric_code < 6 lineas")
            if len(s.numeric_converge) < 2:
                probs.append("numeric_converge < 2 pasos")
            for p_ in s.numeric_intro:
                if p_.count("$") % 2:
                    probs.append("$ desbalanceado en numeric_intro")
            for kind, steps in (("analitico", s.analytic_steps), ("numerico", s.numeric_steps), ("convergencia", s.numeric_converge), ("concepto", s.concept)):
                for i, st in enumerate(steps, 1):
                    if len(st) != 3 or not st[0] or not st[1] or (kind in ('analitico', 'numerico') and not st[2]):
                        probs.append(f"{kind} {i}: paso mal formado")
                        continue
                    for t in st[2]:
                        if not tex_ok(t):
                            probs.append(f"{kind} {i}: TeX sospechoso: {t[:70]}")
                    if st[1].count("$") % 2:
                        probs.append(f"{kind} {i}: $ desbalanceado en la prosa")
            for p in s.explanation:
                if p.count("$") % 2:
                    probs.append("$ desbalanceado en la explicacion")
            if probs:
                bad += 1
                print(f"PROBLEMA {s.id}: " + "; ".join(probs))
    print(f"escenas con detalle: {done}; con problemas o faltantes: {bad}")
    if "--render" in sys.argv:
        return render_check() or (1 if bad else 0)
    return 1 if bad else 0


def render_check() -> int:
    import glob
    import subprocess
    import time

    from playwright.sync_api import sync_playwright

    out = Path("/tmp/pub_check")
    subprocess.run([sys.executable, "scripts/build_site.py", "--out", str(out)], check=True, capture_output=True)
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "8795", "-d", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    errs = 0
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=glob.glob("/opt/pw-browsers/chromium*/chrome-linux*/chrome")[0])
            pg = b.new_page()
            for ch in load_chapters():
                pg.goto(f"http://localhost:8795/chapters/{ch.slug}.html", wait_until="load")
                pg.wait_for_timeout(1500)
                n = pg.evaluate("document.querySelectorAll('.katex-error').length")
                if n:
                    errs += n
                    txt = pg.evaluate("Array.from(document.querySelectorAll('.katex-error')).slice(0,3).map(e=>e.getAttribute('title')||e.textContent)")
                    print(f"KATEX {ch.slug}: {n} errores: {txt}")
            b.close()
    finally:
        srv.terminate()
    print(f"errores KaTeX: {errs}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
