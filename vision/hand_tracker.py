from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp


@dataclass(frozen=True)
class HandLandmarks:
    """
    Structured output for one detected hand.

    handedness_raw is MediaPipe's label (often from camera perspective).
    handedness_mirrored is user-centric (mirrored control) for teleop.
    """
    hand_index: int
    handedness_raw: str                 # "Left" or "Right" from MediaPipe
    handedness_mirrored: str            # mirrored mapping for teleop
    score: float                        # handedness confidence
    landmarks_px: List[Tuple[int, int]] # 21 (x,y) points in pixel coords


class MediaPipeHandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        mirrored: bool = True,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._style = mp.solutions.drawing_styles
        self._mirrored = mirrored

    @staticmethod
    def _mirror_label(label: str) -> str:
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return "Unknown"

    def process(self, frame_bgr) -> Tuple[List[HandLandmarks], object]:
        """
        Returns:
          (hands, results)
          hands: list of detected hands (possibly empty)
          results: raw MediaPipe results (useful for debug drawing)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return [], results

        h, w = frame_bgr.shape[:2]

        hands_out: List[HandLandmarks] = []
        n = len(results.multi_hand_landmarks)

        for i in range(n):
            hand_lms = results.multi_hand_landmarks[i]

            handed_raw = "Unknown"
            score = 0.0
            if results.multi_handedness and i < len(results.multi_handedness):
                cls = results.multi_handedness[i].classification[0]
                handed_raw = cls.label
                score = float(cls.score)

            handed_m = self._mirror_label(handed_raw) if self._mirrored else handed_raw

            pts: List[Tuple[int, int]] = []
            for lm in hand_lms.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append((x, y))

            hands_out.append(
                HandLandmarks(
                    hand_index=i,
                    handedness_raw=handed_raw,
                    handedness_mirrored=handed_m,
                    score=score,
                    landmarks_px=pts,
                )
            )

        return hands_out, results

    def draw(self, frame_bgr, results) -> None:
        """
        Draw landmarks + connections in-place (raw MediaPipe results).
        """
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