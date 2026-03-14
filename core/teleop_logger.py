import csv
import os
from core.teleop_packet import TeleopPacket


class TeleopLogger:
    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "r_state","r_thumb","r_index","r_middle","r_ring","r_pinky",
            "l_state","l_thumb","l_index","l_middle","l_ring","l_pinky",
        ])

    def log(self, packet: TeleopPacket):

        right_cmd = packet.right
        left_cmd = packet.left

        row = [
            packet.timestamp,

            right_cmd.state if right_cmd else "",
            right_cmd.thumb if right_cmd else "",
            right_cmd.index if right_cmd else "",
            right_cmd.middle if right_cmd else "",
            right_cmd.ring if right_cmd else "",
            right_cmd.pinky if right_cmd else "",

            left_cmd.state if left_cmd else "",
            left_cmd.thumb if left_cmd else "",
            left_cmd.index if left_cmd else "",
            left_cmd.middle if left_cmd else "",
            left_cmd.ring if left_cmd else "",
            left_cmd.pinky if left_cmd else "",
        ]

        self.writer.writerow(row)

    def close(self):
        self.file.close()