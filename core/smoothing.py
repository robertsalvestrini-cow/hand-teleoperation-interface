from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


Point = Tuple[float, float]


@dataclass
class EmaSmoother2D:
    """
    Exponential Moving Average smoother for 2D points.
    alpha in (0,1): higher = follows input more closely (less smoothing)
    """
    alpha: float = 0.35
    _prev: Optional[List[Point]] = None

    def reset(self) -> None:
        self._prev = None

    def update(self, pts: List[Point]) -> List[Point]:
        if self._prev is None:
            self._prev = list(pts)
            return list(pts)

        out: List[Point] = []
        a = self.alpha
        for (x, y), (px, py) in zip(pts, self._prev):
            sx = (1 - a) * px + a * x
            sy = (1 - a) * py + a * y
            out.append((sx, sy))

        self._prev = out
        return out