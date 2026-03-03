import time
import cv2

from vision.hand_tracker import MediaPipeHandTracker


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 0")

    # Try MJPG (often improves Logitech throughput)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Capture high quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    tracker = MediaPipeHandTracker(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Processing resolution (decoupled)
    proc_w, proc_h = 640, 360

    fps = 0.0
    alpha = 0.1

    # rolling stage timings (ms)
    t_capture = t_resize = t_infer = t_draw = t_show = 0.0
    beta = 0.1

    print("Hand tracking demo w/ timing. Press 'q' to quit.")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            print("Frame grab failed.")
            break

        proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        t2 = time.perf_counter()

        hand, results = tracker.process(proc)
        t3 = time.perf_counter()

        tracker.draw(proc, results)
        t4 = time.perf_counter()

        dt = t4 - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = (1 - alpha) * fps + alpha * inst_fps

        t_capture = (1 - beta) * t_capture + beta * ((t1 - t0) * 1000)
        t_resize = (1 - beta) * t_resize + beta * ((t2 - t1) * 1000)
        t_infer = (1 - beta) * t_infer + beta * ((t3 - t2) * 1000)
        t_draw = (1 - beta) * t_draw + beta * ((t4 - t3) * 1000)

        label = "No hand"
        if hand:
            label = f"{hand.handedness} hand ({hand.score:.2f})"

        cv2.putText(proc, f"CAP 1280x720@60 | PROC {proc_w}x{proc_h}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(proc, f"FPS: {fps:5.1f}  {label}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(proc, f"ms cap:{t_capture:4.1f} resz:{t_resize:4.1f} infer:{t_infer:4.1f} draw:{t_draw:4.1f}",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        t5 = time.perf_counter()
        cv2.imshow("HAND_TELEOP — Hand Tracking (decoupled)", proc)
        key = cv2.waitKey(1) & 0xFF
        t6 = time.perf_counter()
        t_show = (1 - beta) * t_show + beta * ((t6 - t5) * 1000)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()