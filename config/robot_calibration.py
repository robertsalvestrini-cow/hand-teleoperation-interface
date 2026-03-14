"""
Robot calibration configuration.

These values define how vision curl values map to robot commands.
All values should remain in the normalized range [0,1].
"""


RIGHT_HAND = {
    "thumb": dict(input_min=0.10, input_max=0.90, deadband=0.05, invert=False, gain=1.0),
    "index": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
    "middle": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
    "ring": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
    "pinky": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
}


LEFT_HAND = {
    "thumb": dict(input_min=0.10, input_max=0.90, deadband=0.05, invert=False, gain=1.0),
    "index": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
    "middle": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
    "ring": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
    "pinky": dict(input_min=0.05, input_max=0.95, deadband=0.02, invert=False, gain=1.0),
}