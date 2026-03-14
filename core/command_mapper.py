from __future__ import annotations

from dataclasses import dataclass
from core.teleop_packet import TeleopPacket
from core.robot_command import RobotHandCommand, FingerCommand
from config.robot_calibration import RIGHT_HAND, LEFT_HAND


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass(frozen=True)
class FingerCalibration:
    input_min: float = 0.0
    input_max: float = 1.0
    deadband: float = 0.0
    invert: bool = False
    gain: float = 1.0


class CommandMapper:
    """
    Maps teleoperation packet values into robot-ready normalized commands.

    Supports:
    - input range remapping
    - deadband
    - gain
    - inversion
    - per-finger calibration
    """

    def __init__(self):
        self.right_cal = {
            name: FingerCalibration(**params)
            for name, params in RIGHT_HAND.items()
        }

        self.left_cal = {
            name: FingerCalibration(**params)
            for name, params in LEFT_HAND.items()
        }

    def map_value(self, value: float, cal: FingerCalibration) -> float:
        value = clamp01(value)

        if value < cal.deadband:
            value = 0.0

        span = cal.input_max - cal.input_min
        if span <= 1e-6:
            mapped = 0.0
        else:
            mapped = (value - cal.input_min) / span

        mapped = clamp01(mapped)
        mapped *= cal.gain
        mapped = clamp01(mapped)

        if cal.invert:
            mapped = 1.0 - mapped

        return clamp01(mapped)

    def map_hand(self, hand_cmd, cal_map) -> FingerCommand:
        return FingerCommand(
            thumb=self.map_value(hand_cmd.thumb, cal_map["thumb"]),
            index=self.map_value(hand_cmd.index, cal_map["index"]),
            middle=self.map_value(hand_cmd.middle, cal_map["middle"]),
            ring=self.map_value(hand_cmd.ring, cal_map["ring"]),
            pinky=self.map_value(hand_cmd.pinky, cal_map["pinky"]),
        )

    def map_packet(self, packet: TeleopPacket) -> RobotHandCommand:
        right_cmd = None
        left_cmd = None

        if packet.right is not None:
            right_cmd = self.map_hand(packet.right, self.right_cal)

        if packet.left is not None:
            left_cmd = self.map_hand(packet.left, self.left_cal)

        return RobotHandCommand(
            timestamp=packet.timestamp,
            right=right_cmd,
            left=left_cmd,
        )