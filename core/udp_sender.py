import json
import socket

from core.robot_command import RobotHandCommand


class UdpSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _hand_to_dict(self, hand):
        if hand is None:
            return None

        return {
            "thumb": hand.thumb,
            "index": hand.index,
            "middle": hand.middle,
            "ring": hand.ring,
            "pinky": hand.pinky,
        }

    def send(self, cmd: RobotHandCommand):
        payload = {
            "timestamp": cmd.timestamp,
            "right": self._hand_to_dict(cmd.right),
            "left": self._hand_to_dict(cmd.left),
        }

        message = json.dumps(payload).encode("utf-8")
        self.sock.sendto(message, (self.host, self.port))

    def close(self):
        self.sock.close()