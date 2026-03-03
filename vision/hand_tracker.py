from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import mediapipe as mp


@dataclass(frozen=True)
class HandLandmarks:
    """Structured output for one detected hand."""
    handedness: str                  # "Left" or "Right"
    score: float                     # classification confidence
    landmarks_px: List[Tuple[int, int]]  # 21 (x,y) points in pixel coords


class MediaPipeHandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,  # 0 = faster, 1 = better
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._style = mp.solutions.drawing_styles

    def process(self, frame_bgr) -> Tuple[Optional[HandLandmarks], object]:
        """
        Returns:
          (hand, results)
          hand: first detected hand as HandLandmarks (or None)
          results: raw MediaPipe results (useful for debug drawing)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, results

        # Use first hand only (max_num_hands=1 by default)
        hand_lms = results.multi_hand_landmarks[0]
        handed = "Unknown"
        score = 0.0
        if results.multi_handedness:
            handed = results.multi_handedness[0].classification[0].label
            score = float(results.multi_handedness[0].classification[0].score)

        h, w = frame_bgr.shape[:2]
        pts = []
        for lm in hand_lms.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))

        return HandLandmarks(handedness=handed, score=score, landmarks_px=pts), results

    def draw(self, frame_bgr, results) -> None:
        """Draw landmarks + connections in-place."""
        if not results.multi_hand_landmarks:
            return
        for hand_lms in results.multi_hand_landmarks:
            self._drawer.draw_landmarks(
                frame_bgr,
                hand_lms,
                self._mp_hands.HAND_CONNECTIONS,
                self._style.get_default_hand_landmarks_style(),
                self._style.get_default_hand_connections_style(),
            )