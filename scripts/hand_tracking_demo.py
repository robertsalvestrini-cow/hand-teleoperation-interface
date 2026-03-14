"""
HAND_TELEOP — Hand Tracking Demo (Decoupled)

Pipeline (per frame):
  Capture → Resize(proc) → Detect → StateMachine → Smooth → FeatureExtract(curls) → Draw → Overlay → Display

Keys:
  q  Quit
"""

import time
import cv2
import mediapipe as mp
from dataclasses import replace

from core.state_machine import HandStateMachine
from core.smoothing import EmaSmoother2D
from vision.hand_tracker import MediaPipeHandTracker
from features.finger_curl import FingerCurlEstimator
from core.teleop_packet import TeleopPacket, HandCommand
from core.teleop_logger import TeleopLogger
from core.command_mapper import CommandMapper
from core.udp_sender import UdpSender

# SECTION 1 Camera Helpers
def safe_read(cap: cv2.VideoCapture, retries: int = 30, delay_s: float = 0.02):
    """
    Robustly read frames. MSMF can occasionally produce invalid frames during startup.
    Returns (ok, frame). If it can't get a valid frame after retries, returns (False, None).
    """
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None and hasattr(frame, "shape") and frame.size > 0:
            return True, frame
        time.sleep(delay_s)
    return False, None


def open_camera(prefer_msmf: bool = True) -> cv2.VideoCapture:
    """
    Try MSMF first (fast runtime), fall back to DSHOW if needed (sometimes more reliable).
    """
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if prefer_msmf else [cv2.CAP_DSHOW, cv2.CAP_MSMF]

    last_err = None
    for backend in backends:
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(0, backend)
        t1 = time.perf_counter()
        print(f"VideoCapture({backend}) time: {(t1 - t0)*1000:.1f} ms")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            continue

        try:
            name = cap.getBackendName()
        except Exception:
            name = str(backend)
        print("Backend:", name)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        ok, frame = safe_read(cap, retries=40, delay_s=0.03)
        if ok:
            print(f"First valid frame: {frame.shape[1]}x{frame.shape[0]}")
            return cap

        last_err = f"Backend {name} opened but did not produce valid frames."
        cap.release()

    raise RuntimeError(last_err or "Could not open camera with any backend.")


# SECTION 1B Drawing Helper
def draw_hand_landmarks_px(img, pts):
    """
    Draws 21 hand landmarks and connections on img.
    pts must be a list of 21 (x, y) pixel points.
    """
    if pts is None or len(pts) != 21:
        return

    h, w = img.shape[:2]

    for a, b in mp.solutions.hands.HAND_CONNECTIONS:
        ax, ay = int(pts[a][0]), int(pts[a][1])
        bx, by = int(pts[b][0]), int(pts[b][1])

        ax = 0 if ax < 0 else w - 1 if ax >= w else ax
        ay = 0 if ay < 0 else h - 1 if ay >= h else ay
        bx = 0 if bx < 0 else w - 1 if bx >= w else bx
        by = 0 if by < 0 else h - 1 if by >= h else by

        cv2.line(img, (ax, ay), (bx, by), (0, 255, 0), 2)

    for x, y in pts:
        x, y = int(x), int(y)
        x = 0 if x < 0 else w - 1 if x >= w else x
        y = 0 if y < 0 else h - 1 if y >= h else y
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)


# SECTION 2 Main Demo Entry Point
def main():
    # SECTION 2A System Initialization
    cap = open_camera(prefer_msmf=True)

    tracker = MediaPipeHandTracker(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    sm_right = HandStateMachine(enter_frames=5, lost_frames=5)
    sm_left = HandStateMachine(enter_frames=5, lost_frames=5)

    # Note: this smoother is shared for now (next step we will split per-hand)
    smoother_right = EmaSmoother2D(alpha=0.35)
    smoother_left = EmaSmoother2D(alpha=0.35)
    logger = TeleopLogger("logs/teleop_session.csv")
    mapper = CommandMapper()
    udp_sender = UdpSender(host="127.0.0.1", port=5005)

    curl_est = FingerCurlEstimator()
    proc_w, proc_h = 960, 540
    display_w, display_h = 960, 540
    fps = 0.0
    fps_alpha = 0.1
    t_capture = t_resize = t_infer = t_draw = t_show = 0.0
    beta = 0.1
    window_name = "HAND_TELEOP - Hand Tracking (decoupled)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Hand tracking demo running. Press 'q' to quit.")

    # SECTION 2B Real Time Loop
    while True:
        # SECTION 3 Frame Capture
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()

        if (not ok) or frame is None or frame.size == 0:
            ok2, frame2 = safe_read(cap, retries=10, delay_s=0.01)
            if not ok2:
                print("Frame grab failed repeatedly.")
                break
            frame = frame2
            t1 = time.perf_counter()

        # SECTION 4 Preprocessing
        proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        t2 = time.perf_counter()

        # SECTION 5 Hand Detection
        hands, results = tracker.process(proc)

        # SECTION 6 State Machine
        hand_states = {}
        seen_labels = set()
        for hnd in hands:
            hand_label = hnd.handedness_mirrored
            seen_labels.add(hand_label)
            if hand_label == "Right":
                hand_states[hand_label] = sm_right.update(True)
            elif hand_label == "Left":
                hand_states[hand_label] = sm_left.update(True)
            else:
                hand_states[hand_label] = None
        if "Right" not in seen_labels:
            hand_states["Right"] = sm_right.update(False)
        if "Left" not in seen_labels:
            hand_states["Left"] = sm_left.update(False)
        state_right = hand_states["Right"]
        state_left = hand_states["Left"]
        # SECTION 7 Landmark Smoothing
        processed_hands = []
        for hnd in hands:
            pts = None
            label = hnd.handedness_mirrored
            if label == "Right":
                current_state = state_right
                smoother = smoother_right
            elif label == "Left":
                current_state = state_left
                smoother = smoother_left
            else:
                current_state = None
                smoother = None
            if current_state is not None and current_state.name in ("NO_HAND", "LOST") and smoother is not None:
                smoother.reset()
            if current_state is not None and current_state.name == "TRACK" and smoother is not None:
                pts_raw = [(float(x), float(y)) for (x, y) in hnd.landmarks_px]
                pts = smoother.update(pts_raw)
                hnd = replace(hnd, landmarks_px=[(int(x), int(y)) for (x, y) in pts])
            processed_hands.append((hnd, pts, current_state))

        # SECTION 8 Feature Extraction
        curl_outputs = []
        for hnd, pts, current_state in processed_hands:
            if pts is None:
                continue
            curls = curl_est.estimate(pts)
            curl_outputs.append((hnd, curls, current_state))
        t3 = time.perf_counter()

        # SECTION 8B Teleoperation Packet Generation
        right_cmd = None
        left_cmd = None
        for hnd, curls, current_state in curl_outputs:
            cmd = HandCommand(
                state=current_state.name,
                thumb=curls.thumb,
                index=curls.index,
                middle=curls.middle,
                ring=curls.ring,
                pinky=curls.pinky,
            )
            if hnd.handedness_mirrored == "Right":
                right_cmd = cmd
            elif hnd.handedness_mirrored == "Left":
                left_cmd = cmd
        packet = TeleopPacket(
            timestamp=time.time(),
            right=right_cmd,
            left=left_cmd,
        )
        robot_cmd = mapper.map_packet(packet)
        logger.log(packet)
        udp_sender.send(robot_cmd)

        # SECTION 9 Visualization
        if processed_hands:
            any_drawn = False
            for hnd, pts, current_state in processed_hands:
                if pts is not None:
                    draw_hand_landmarks_px(proc, pts)
                    any_drawn = True
            if not any_drawn:
                tracker.draw(proc, results)
        else:
            tracker.draw(proc, results)
        t4 = time.perf_counter()

        # SECTION 10 Performance Metrics
        dt = t4 - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = (1 - fps_alpha) * fps + fps_alpha * inst_fps

        t_capture = (1 - beta) * t_capture + beta * ((t1 - t0) * 1000)
        t_resize = (1 - beta) * t_resize + beta * ((t2 - t1) * 1000)
        t_infer = (1 - beta) * t_infer + beta * ((t3 - t2) * 1000)
        t_draw = (1 - beta) * t_draw + beta * ((t4 - t3) * 1000)

        # SECTION 11 Operator Overlay
        cv2.putText(
            proc,
            f"FPS: {fps:5.1f}   R:{state_right.name}   L:{state_left.name}", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2,
        )

        y = proc.shape[0] - 10
        for hnd, curls, current_state in curl_outputs:
            text = (
                f"{hnd.handedness_mirrored.upper()} "
                f"T:{curls.thumb:.2f} "
                f"I:{curls.index:.2f} "
                f"M:{curls.middle:.2f} "
                f"R:{curls.ring:.2f} "
                f"P:{curls.pinky:.2f}"
            )
            cv2.putText(
                proc,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            y -= 22

        if robot_cmd.right is not None:
            text = (
                f"R_CMD "
                f"T:{robot_cmd.right.thumb:.2f} "
                f"I:{robot_cmd.right.index:.2f} "
                f"M:{robot_cmd.right.middle:.2f} "
                f"R:{robot_cmd.right.ring:.2f} "
                f"P:{robot_cmd.right.pinky:.2f}"
            )
            cv2.putText(
                proc,
                text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
            )

        if robot_cmd.left is not None:
            text = (
                f"L_CMD "
                f"T:{robot_cmd.left.thumb:.2f} "
                f"I:{robot_cmd.left.index:.2f} "
                f"M:{robot_cmd.left.middle:.2f} "
                f"R:{robot_cmd.left.ring:.2f} "
                f"P:{robot_cmd.left.pinky:.2f}"
            )
            cv2.putText(
                proc,
                text,
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
            )    

        # SECTION 12 Display
        display = cv2.resize(proc, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

        t5 = time.perf_counter()
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        t6 = time.perf_counter()
        t_show = (1 - beta) * t_show + beta * ((t6 - t5) * 1000)

        # SECTION 13 Exit
        if key == ord("q"):
            break

    # SECTION 14 Cleanup
    logger.close()
    udp_sender.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()