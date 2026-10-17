"""Sistema visual compartido por todas las escenas.

Paleta neutra (escala zinc) con un unico acento por rol semantico:
  analitico -> blanco, numerico -> azul, error/alerta -> rojo, destacado -> ambar.
"""
from __future__ import annotations

from manim import (
    DOWN, LEFT, RIGHT, UP, UL, UR, Axes, FadeIn, FadeOut, Line, MathTex, Mobject,
    RoundedRectangle, Scene, Text, VGroup, config,
)

BG = "#09090b"
FG = "#fafafa"
MUTED = "#a1a1aa"
FAINT = "#52525b"
BORDER = "#27272a"
PANEL = "#18181b"
ANALYTIC = "#fafafa"
NUMERIC = "#38bdf8"
ERROR = "#f87171"
ACCENT = "#fbbf24"

FONT = "Inter"
MONO = "JetBrains Mono"

config.background_color = BG


def T(text: str, size: int = 26, color: str = FG, mono: bool = False, weight: str = "NORMAL") -> Text:
    return Text(text, font=MONO if mono else FONT, font_size=size, color=color, weight=weight)


def M(tex: str, size: int = 40, color: str = FG) -> MathTex:
    return MathTex(tex, font_size=size, color=color)


def make_axes(x_range, y_range, width: float, height: float, x_step=None, y_step=None, numbers: bool = True) -> Axes:
    xr = list(x_range) + ([x_step] if x_step and len(x_range) == 2 else [])
    yr = list(y_range) + ([y_step] if y_step and len(y_range) == 2 else [])
    return Axes(
        x_range=xr, y_range=yr, x_length=width, y_length=height,
        axis_config={"color": FAINT, "stroke_width": 2, "include_tip": False,
                     "include_numbers": numbers, "font_size": 20, "decimal_number_config": {"color": MUTED}},
    )


def panel(width: float, height: float) -> RoundedRectangle:
    return RoundedRectangle(corner_radius=0.12, width=width, height=height,
                            stroke_color=BORDER, stroke_width=2, fill_color=PANEL, fill_opacity=1)


def arrange_lines(*mobs: Mobject, buff: float = 0.28, left: bool = True) -> VGroup:
    g = VGroup(*mobs).arrange(DOWN, buff=buff, aligned_edge=LEFT if left else None)
    return g


class PhysicsScene(Scene):
    """Escena con encabezado de capitulo, indicador de fase y leyenda inferior."""

    chapter_number = 0
    chapter_title = ""
    volume = 1

    def setup(self):
        label = f"CAPÍTULO {self.chapter_number:02d}" if self.volume == 1 else f"VOLUMEN {self.volume} · CAPÍTULO {self.chapter_number:02d}"
        kicker = T(label, 15, MUTED, mono=True)
        title = T(self.chapter_title, 22, FG, weight="BOLD")
        head = VGroup(kicker, title).arrange(DOWN, aligned_edge=LEFT, buff=0.08).to_corner(UL, buff=0.5)
        rule_top = Line(LEFT * 6.65, RIGHT * 6.65, stroke_color=BORDER, stroke_width=2).move_to(UP * 2.85)
        rule_bot = Line(LEFT * 6.65, RIGHT * 6.65, stroke_color=BORDER, stroke_width=2).move_to(DOWN * 3.3)
        self.add(head, rule_top, rule_bot)
        self._chrome_ids = {id(m) for m in (head, rule_top, rule_bot)}
        self._chip: Mobject | None = None
        self._caption: Mobject | None = None

    def _make_chip(self, name: str) -> VGroup:
        txt = T(name, 17, FG)
        box = RoundedRectangle(corner_radius=0.18, width=txt.width + 0.5, height=0.46,
                               stroke_color=FAINT, stroke_width=2, fill_opacity=0)
        chip = VGroup(box, txt)
        chip.to_corner(UR, buff=0.62)
        return chip

    def phase(self, name: str, run_time: float = 0.5):
        new = self._make_chip(name)
        if self._chip is None:
            self.play(FadeIn(new, shift=DOWN * 0.1), run_time=run_time)
        else:
            self.play(FadeOut(self._chip, shift=UP * 0.1), FadeIn(new, shift=DOWN * 0.1), run_time=run_time)
        self._chip = new

    def say(self, text: str, *anims, run_time: float = 1.0, wait: float = 0.0):
        new = T(text, 21, MUTED)
        if new.width > 13.2:
            new.scale(13.2 / new.width)
        new.move_to(DOWN * 3.72).align_to(LEFT * 6.65, LEFT)
        outs = [FadeOut(self._caption)] if self._caption is not None else []
        self.play(*outs, FadeIn(new, shift=UP * 0.08), *anims, run_time=run_time)
        self._caption = new
        if wait:
            self.wait(wait)

    def clear_stage(self, *keep: Mobject, run_time: float = 0.6):
        fixed = set()
        for m in (self._chip, self._caption, *keep):
            if m is not None:
                fixed.add(id(m))
                for s in m.get_family():
                    fixed.add(id(s))
        stage = [m for m in self.mobjects if id(m) not in fixed and not self._is_chrome(m)]
        if stage:
            self.play(*[FadeOut(m) for m in stage], run_time=run_time)

    def _is_chrome(self, m: Mobject) -> bool:
        return id(m) in self._chrome_ids


def mono_table(header, rows, size: int = 19, colors=None, gap: float = 0.8, row_buff: float = 0.3) -> VGroup:
    """Tabla de texto monoespaciado alineada por columnas."""
    ncol = len(header)
    colors = colors or [FG] * ncol
    cols = []
    for j in range(ncol):
        cells = [T(header[j], size - 2, MUTED, mono=True)]
        cells += [T(str(r[j]), size, colors[j], mono=True) for r in rows]
        cols.append(VGroup(*cells).arrange(DOWN, aligned_edge=LEFT, buff=row_buff))
    return VGroup(*cols).arrange(RIGHT, buff=gap, aligned_edge=UP)
