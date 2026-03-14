from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


Point = Tuple[float, float]


def _angle(a: Point, b: Point, c: Point) -> float:
    """
    Returns angle ABC in degrees (b is the vertex).
    """
    bax, bay = (a[0] - b[0], a[1] - b[1])
    bcx, bcy = (c[0] - b[0], c[1] - b[1])

    dot = bax * bcx + bay * bcy
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return 180.0

    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosv))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class FingerCurls:
    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "thumb": self.thumb,
            "index": self.index,
            "middle": self.middle,
            "ring": self.ring,
            "pinky": self.pinky,
        }


class FingerCurlEstimator:
    """
    Computes normalized finger curl values (0=open, 1=closed) from 21 hand landmarks.

    Assumes landmarks are MediaPipe Hands indexing:
      0 wrist
      thumb: 1-4
      index: 5-8
      middle: 9-12
      ring: 13-16
      pinky: 17-20

    Uses joint angles at PIP (or IP for thumb) to estimate curl.
    """

    # Heuristic mapping angles to curl:
    # straight finger ~ 170-180 deg -> curl ~ 0
    # bent finger ~ 60-90 deg -> curl ~ 1
    def __init__(self, straight_deg: float = 170.0, bent_deg: float = 70.0):
        self.straight_deg = straight_deg
        self.bent_deg = bent_deg

    def _curl_from_angle(self, deg: float) -> float:
        # Map [straight..bent] to [0..1]
        if self.straight_deg == self.bent_deg:
            return 0.0
        t = (self.straight_deg - deg) / (self.straight_deg - self.bent_deg)
        return _clamp01(t)

    def estimate(self, lm: List[Point]) -> FingerCurls:
        if len(lm) != 21:
            raise ValueError("Expected 21 landmarks")

        # Index/Middle/Ring/Pinky: use PIP joint angle (MCP-PIP-DIP)
        index_deg = _angle(lm[5], lm[6], lm[7])
        middle_deg = _angle(lm[9], lm[10], lm[11])
        ring_deg = _angle(lm[13], lm[14], lm[15])
        pinky_deg = _angle(lm[17], lm[18], lm[19])

        # Thumb: use IP joint angle (MCP-IP-TIP) = (2,3,4)
        thumb_deg = _angle(lm[2], lm[3], lm[4])

        return FingerCurls(
            thumb=self._curl_from_angle(thumb_deg),
            index=self._curl_from_angle(index_deg),
            middle=self._curl_from_angle(middle_deg),
            ring=self._curl_from_angle(ring_deg),
            pinky=self._curl_from_angle(pinky_deg),
        )