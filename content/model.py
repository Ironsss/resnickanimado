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


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    module: str
    title: str
    summary: str
    concepts: list[str]
    scenes: list[Scene]
