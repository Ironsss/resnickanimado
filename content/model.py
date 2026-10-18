"""Modelo de datos del contenido: capitulos, escenas y filas de verificacion."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Row:
    """Una magnitud calculada por via analitica y numerica."""

    label: str
    tex: str
    analytic: float
    numeric: float
    unit: str
    digits: int = 4

    @property
    def error(self) -> float:
        return abs(self.analytic - self.numeric)


# Paso de una solucion: (titulo, explicacion en prosa, lista de ecuaciones TeX en display)
Step = tuple[str, str, list[str]]


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    statement: str
    foundations: list[tuple[str, str]]  # (tex, descripcion)
    analytic: list[tuple[str, str]]  # (tex, descripcion)
    numeric_method: str
    rows: list[Row]
    notes: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)  # el tema, en parrafos
    analytic_steps: list[Step] = field(default_factory=list)  # solucion analitica paso a paso
    numeric_steps: list[Step] = field(default_factory=list)  # metodo numerico a mano, con cifras


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    module: str
    title: str
    summary: str
    concepts: list[str]
    scenes: list[Scene]
    volume: int = 1
    part: str = ""
