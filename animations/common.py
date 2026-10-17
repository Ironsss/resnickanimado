"""Sistema visual compartido por todas las escenas.

Paleta neutra (escala zinc) con un unico acento por rol semantico:
  analitico -> blanco, numerico -> azul, error/alerta -> rojo, destacado -> ambar.
"""
from __future__ import annotations

import re

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


def _texify_accents(tex: str) -> str:
    """LaTeX no admite letras acentuadas en modo matematico: se envuelven en \\text{} (subindices, \\mathrm)."""
    tex = re.sub(r"\\mathrm\{([^{}]*[^\x00-\x7f][^{}]*)\}", lambda m: "\\text{" + m.group(1) + "}", tex)
    return re.sub(r"([_^])\{([^{}]*[^\x00-\x7f][^{}]*)\}", lambda m: m.group(1) + "{\\text{" + m.group(2) + "}}", tex)


def M(tex: str, size: int = 40, color: str = FG) -> MathTex:
    return MathTex(_texify_accents(tex), font_size=size, color=color)


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


# ----------------------------------------------------------------- helpers de escena
import numpy as np  # noqa: E402
from manim import Dot, VMobject, Write  # noqa: E402


def fit(group: Mobject, max_w: float = 12.6, max_h: float = 5.4) -> Mobject:
    """Reduce el grupo si excede el area util del cuadro."""
    if group.width > max_w:
        group.scale(max_w / group.width)
    if group.height > max_h:
        group.scale(max_h / group.height)
    return group


def _item(x, size, default=FG):
    tex, color = (x if isinstance(x, tuple) else (x, default))
    return M(tex, size, color)


def polyline(axes: Axes, xs, ys, color: str = NUMERIC, width: float = 3.5) -> VMobject:
    m = VMobject(color=color, stroke_width=width)
    m.set_points_as_corners([axes.c2p(x, y) for x, y in zip(xs, ys)])
    return m


def axes_block(xr, yr, w, h, pos, xticks=(), yticks=(), xlabel="", ylabel="", fmt="{:g}"):
    """Ejes con marcas numericas y rotulos. Devuelve (axes, grupo_decoracion)."""
    def _step(rng, ticks):
        st = float(ticks[1] - ticks[0]) if len(ticks) >= 2 else 0.0
        return st if st > 0 else (rng[1] - rng[0]) / 5.0

    ax = make_axes([xr[0], xr[1]], [yr[0], yr[1]], w, h, _step(xr, xticks), _step(yr, yticks), numbers=False).move_to(pos)
    deco = VGroup()
    for v in xticks:
        deco.add(T(fmt.format(v), 14, MUTED, mono=True).next_to(ax.c2p(v, yr[0]), DOWN, buff=0.1))
    for v in yticks:
        deco.add(T(fmt.format(v), 14, MUTED, mono=True).next_to(ax.c2p(xr[0], v), LEFT, buff=0.1))
    if xlabel:
        deco.add(T(xlabel, 16, MUTED).next_to(ax.x_axis.get_end(), DOWN, buff=0.3).shift(LEFT * 0.2))
    if ylabel:
        deco.add(T(ylabel, 16, MUTED).next_to(ax.y_axis.get_end(), UP, buff=0.1).align_to(ax.y_axis, LEFT))
    return ax, deco


def legend_block(items, pos) -> VGroup:
    """items: [(color, texto)]. Linea corta de color + texto monoespaciado."""
    rows = [VGroup(Line(LEFT * 0.15, RIGHT * 0.15, color=c, stroke_width=4), T(t, 17, MUTED, mono=True)).arrange(RIGHT, buff=0.2)
            for c, t in items]
    return VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(pos)


def text_lines(lines, size=22, color=FG, buff=0.22, mono=True) -> VGroup:
    """Lineas de texto alineadas a la izquierda; cada linea es str o (str, color)."""
    mobs = []
    for ln in lines:
        txt, col = ln if isinstance(ln, tuple) else (ln, color)
        mobs.append(T(txt, size, col, mono=mono))
    return VGroup(*mobs).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def _phase_methods(cls):
    def foundations(self, eqs, note=None, caption="", size=42, run_time=3.0):
        self.phase("Fundamentos")
        g = fit(VGroup(*[_item(e, size) for e in eqs]).arrange(DOWN, aligned_edge=LEFT, buff=0.5))
        g.move_to(DOWN * (0.1 if not note else 0.35))
        n = T(note, 22, MUTED) if note else None
        if n is not None:
            fit(n, 12.4)
            n.next_to(g, DOWN, buff=0.55)
            if n.width > g.width:
                n.set_x(0)
            else:
                n.align_to(g, LEFT)
        self.say(caption, *[Write(x) for x in g], run_time=run_time)
        if n is not None:
            self.play(FadeIn(n))
        self.wait(1.0)
        self.clear_stage()

    def problem(self, lines, caption=""):
        self.phase("Problema")
        mobs = [T(l, 26 if i < len(lines) - 1 else 24, FG if i < len(lines) - 1 else MUTED) for i, l in enumerate(lines)]
        g = fit(VGroup(*mobs).arrange(DOWN, aligned_edge=LEFT, buff=0.32), 12.4).move_to(DOWN * 0.1)
        self.say(caption, FadeIn(g), run_time=1.4)
        self.wait(1.6)
        self.clear_stage()

    def analytic(self, eqs, caption="", size=38, keep=False):
        self.phase("Solución analítica")
        g = fit(VGroup(*[_item(e, size) for e in eqs]).arrange(DOWN, aligned_edge=LEFT, buff=0.42)).move_to(DOWN * 0.1)
        self.say(caption, run_time=0.4)
        for x in g:
            self.play(Write(x), run_time=1.1)
        self.wait(1.5)
        if not keep:
            self.clear_stage()
        return g

    def numeric_phase(self, caption=""):
        self.phase("Solución numérica")
        if caption:
            self.say(caption, run_time=0.5)

    cls.foundations, cls.problem, cls.analytic, cls.numeric_phase = foundations, problem, analytic, numeric_phase
    return cls


_phase_methods(PhysicsScene)


# ------------------------------------------------------------ simbolos de circuito
def wire(*pts, color: str = FG, width: float = 3) -> VMobject:
    m = VMobject(color=color, stroke_width=width)
    m.set_points_as_corners([np.array([p[0], p[1], 0.0]) for p in pts])
    return m


def resistor_symbol(a, b, color: str = FG, teeth: int = 6, amp: float = 0.14) -> VMobject:
    a, b = np.array([a[0], a[1], 0.0]), np.array([b[0], b[1], 0.0])
    d = b - a
    ln = np.linalg.norm(d)
    u = d / ln
    n = np.array([-u[1], u[0], 0.0])
    lead = 0.18 * ln
    pts = [a, a + u * lead]
    body = ln - 2 * lead
    for k in range(teeth):
        pts.append(a + u * (lead + body * (k + 0.5) / teeth) + n * amp * (1 if k % 2 == 0 else -1))
    pts += [a + u * (ln - lead), b]
    m = VMobject(color=color, stroke_width=3)
    m.set_points_as_corners(pts)
    return m


def capacitor_symbol(center, vertical: bool = False, size: float = 0.35, color: str = FG, gap: float = 0.11) -> VGroup:
    c = np.array([center[0], center[1], 0.0])
    if vertical:
        p1 = Line(c + np.array([-size, gap, 0]), c + np.array([size, gap, 0]), color=color, stroke_width=4)
        p2 = Line(c + np.array([-size, -gap, 0]), c + np.array([size, -gap, 0]), color=color, stroke_width=4)
    else:
        p1 = Line(c + np.array([-gap, -size, 0]), c + np.array([-gap, size, 0]), color=color, stroke_width=4)
        p2 = Line(c + np.array([gap, -size, 0]), c + np.array([gap, size, 0]), color=color, stroke_width=4)
    return VGroup(p1, p2)


def battery_symbol(center, vertical: bool = True, size: float = 0.3, color: str = FG) -> VGroup:
    c = np.array([center[0], center[1], 0.0])
    if vertical:
        long_ = Line(c + np.array([-size, 0.1, 0]), c + np.array([size, 0.1, 0]), color=color, stroke_width=4)
        short = Line(c + np.array([-size * 0.55, -0.1, 0]), c + np.array([size * 0.55, -0.1, 0]), color=color, stroke_width=6)
    else:
        long_ = Line(c + np.array([-0.1, -size, 0]), c + np.array([-0.1, size, 0]), color=color, stroke_width=4)
        short = Line(c + np.array([0.1, -size * 0.55, 0]), c + np.array([0.1, size * 0.55, 0]), color=color, stroke_width=6)
    return VGroup(long_, short)


# --------------------------------------------------- plantilla de escena con curva y tabla
def curve_scene(scene, *, found, note, fcap, problem, pcap, analytic, acap, numeric, xr, yr, xticks, yticks, xlabel, ylabel,
                curves, dots, legend, table, final, fsize=36, asize=34, dot_every=1, table_pos=None):
    """Fundamentos, problema, solucion analitica y fase numerica con curva analitica, puntos numericos y tabla de verificacion.

    curves: lista de (xs, ys, color); dots: (xs, ys) numericos; legend: [(color, texto)]; table: (encabezado, filas)."""
    from manim import Create, FadeIn, LaggedStart, Dot, VGroup, RIGHT, LEFT, DOWN, UP

    scene.foundations(found, note=note, caption=fcap, size=fsize)
    scene.problem(problem, caption=pcap)
    scene.analytic(analytic, size=asize, caption=acap)
    scene.numeric_phase(numeric)
    ax, deco = axes_block(xr, yr, 6.4, 3.6, LEFT * 3.2 + DOWN * 0.15, xticks, yticks, xlabel, ylabel)
    cs = [polyline(ax, xs, ys, col, 3.2) for xs, ys, col in curves]
    scene.play(Create(ax), FadeIn(deco), *[Create(c) for c in cs], run_time=2.0)
    xs, ys = dots
    d = VGroup(*[Dot(ax.c2p(float(x), float(y)), radius=0.05, color=NUMERIC) for x, y in list(zip(xs, ys))[::dot_every]])
    leg = legend_block(legend, RIGHT * 0.9 + UP * 2.0)
    scene.play(LaggedStart(*[FadeIn(q, scale=1.4) for q in d], lag_ratio=0.04), FadeIn(leg), run_time=2.0)
    tab = mono_table(table[0], table[1], size=15, colors=[MUTED, NUMERIC, ANALYTIC][: len(table[0])], gap=0.3)
    tab.move_to(table_pos if table_pos is not None else RIGHT * 4.4 + DOWN * 0.3)
    scene.say(final, FadeIn(tab), run_time=1.3)
    scene.wait(2.5)
