from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """A continuous vector space: `shape` floats, each within [low, high]."""

    low: float
    high: float
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class Discrete:
    """A single integer choice in `range(n)`."""

    n: int


Space = Box | Discrete
