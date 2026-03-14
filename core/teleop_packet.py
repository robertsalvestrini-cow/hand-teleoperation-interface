from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HandCommand:
    state: str
    thumb: float = 0.0
    index: float = 0.0
    middle: float = 0.0
    ring: float = 0.0
    pinky: float = 0.0


@dataclass(frozen=True)
class TeleopPacket:
    timestamp: float
    right: Optional[HandCommand] = None
    left: Optional[HandCommand] = None