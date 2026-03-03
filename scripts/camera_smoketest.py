import time
import cv2

def main(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Windows-friendly
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    # Try C922-friendly defaults
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last = time.perf_counter()
    fps = 0.0
    alpha = 0.1  # FPS smoothing

    print("Camera smoke test running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        now = time.perf_counter()
        dt = now - last
        last = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = (1 - alpha) * fps + alpha * inst_fps

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"{w}x{h}  FPS: {fps:5.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("HAND_TELEOP - Camera Smoke Test (C922)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)