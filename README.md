# Real-Time Hand Teleoperation Interface

A real-time computer vision system that converts hand motion into robot-ready control commands using MediaPipe hand tracking.

## Features

- Dual-hand tracking
- Finger curl estimation
- State-based tracking stability
- Landmark smoothing
- Command mapping with calibration
- UDP streaming for robot control
- CSV logging for analysis

## System Pipeline

Camera → MediaPipe → State Machine → Landmark Smoothing → Curl Estimation → Teleop Packet → Command Mapping → UDP Output

## Demo

Video demonstration:
https://youtu.be/T_EicSmUV_I

## Requirements

Python 3.10+

Install dependencies:

pip install -r requirements.txt

## Run the Demo

python -m scripts.hand_tracking_demo

## UDP Receiver Example

python -m scripts.udp_receiver

## Future Work

- Integration with robotic hand hardware
- Servo/tendon driven actuation
- Closed-loop teleoperation
