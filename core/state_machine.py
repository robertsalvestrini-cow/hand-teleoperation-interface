from enum import Enum, auto


class HandState(Enum):
    NO_HAND = auto()
    ENTER = auto()
    TRACK = auto()
    LOST = auto()


class HandStateMachine:
    """
    Robust hand detection state machine.

    Prevents flicker and false activation by requiring a number
    of consecutive frames before transitioning states.
    """

    def __init__(self, enter_frames=5, lost_frames=5):
        self.state = HandState.NO_HAND
        self.enter_frames = enter_frames
        self.lost_frames = lost_frames

        self._enter_counter = 0
        self._lost_counter = 0

    def update(self, hand_detected: bool) -> HandState:
        """
        Update the state machine with the current detection result.
        """

        if self.state == HandState.NO_HAND:
            if hand_detected:
                self._enter_counter += 1
                if self._enter_counter >= self.enter_frames:
                    self.state = HandState.ENTER
                    self._enter_counter = 0
            else:
                self._enter_counter = 0

        elif self.state == HandState.ENTER:
            # Immediately transition to TRACK
            self.state = HandState.TRACK

        elif self.state == HandState.TRACK:
            if not hand_detected:
                self._lost_counter += 1
                if self._lost_counter >= self.lost_frames:
                    self.state = HandState.LOST
                    self._lost_counter = 0
            else:
                self._lost_counter = 0

        elif self.state == HandState.LOST:
            if hand_detected:
                self._enter_counter += 1
                if self._enter_counter >= self.enter_frames:
                    self.state = HandState.ENTER
                    self._enter_counter = 0
            else:
                self.state = HandState.NO_HAND

        return self.state