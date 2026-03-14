from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FingerCommand:
    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float


@dataclass(frozen=True)
class RobotHandCommand:
    timestamp: float
    right: Optional[FingerCommand] = None
    left: Optional[FingerCommand] = None