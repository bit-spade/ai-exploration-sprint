import time
from collections import deque
import cv2
import numpy as np

from src.pose_detector import PoseDetector
from src.utils import PushUpCounter


def draw_hud(
    frame: np.ndarray,
    reps: int,
    feedback: str,
    elbow_angle: float | None,
    hip_angle: float | None,
    knee_angle: float | None,
    fps: float
) -> None:
    """Draws metrics and form feedback on top of the webcam feed."""
    h, w, _ = frame.shape

    # --- Top Left: FPS & Rep Counter ---
    card_w, card_h = 220, 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), (70, 70, 70), 1)

    # Rep counter
    cv2.putText(frame, "REPS", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, str(reps), (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # FPS tag
    fps_color = (0, 230, 120) if fps >= 25 else (0, 165, 255)
    cv2.putText(frame, f"{fps:.1f} FPS", (135, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

    # --- Top Right: Angles & Form Feedback ---
    info_w, info_h = 320, 145
    x_start = w - info_w - 15
    cv2.rectangle(overlay, (x_start, 15), (x_start + info_w, 15 + info_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x_start, 15), (x_start + info_w, 15 + info_h), (70, 70, 70), 1)

    # Elbow angle + mini progress bar
    elbow_str = f"{int(elbow_angle)} deg" if elbow_angle is not None else "--"
    cv2.putText(frame, f"Arm: {elbow_str}", (x_start + 15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    progress = np.interp(elbow_angle if elbow_angle else 160, [90, 160], [100, 0])
    bar_w = 120
    fill = int((progress / 100.0) * bar_w)
    cv2.rectangle(frame, (x_start + 180, 35), (x_start + 180 + bar_w, 47), (50, 50, 50), -1)
    if fill > 0:
        cv2.rectangle(frame, (x_start + 180, 35), (x_start + 180 + fill, 47), (0, 220, 100), -1)
    cv2.rectangle(frame, (x_start + 180, 35), (x_start + 180 + bar_w, 47), (120, 120, 120), 1)

    # Hip angle (spine alignment check)
    hip_str = f"{int(hip_angle)} deg" if hip_angle is not None else "--"
    hip_color = (0, 220, 100) if (hip_angle and hip_angle >= 150) else (0, 140, 255)
    cv2.putText(frame, f"Hip / Back: {hip_str}", (x_start + 15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hip_color, 1, cv2.LINE_AA)

    # Knee angle (leg alignment check)
    knee_str = f"{int(knee_angle)} deg" if knee_angle is not None else "--"
    knee_color = (0, 220, 100) if (knee_angle and knee_angle >= 150) else (0, 140, 255)
    cv2.putText(frame, f"Knee: {knee_str}", (x_start + 15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, knee_color, 1, cv2.LINE_AA)

    # Feedback message
    cv2.putText(frame, feedback, (x_start + 15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)


def run_pipeline(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    detector = PoseDetector(
        model_name="yolo26n-pose.pt",
        conf_threshold=0.45,
        keypoint_confidence=0.50
    )

    counter = PushUpCounter(
        down_threshold=90.0,
        up_threshold=160.0,
        min_confidence=0.50
    )

    fps_history = deque(maxlen=20)
    prev_time = time.perf_counter()

    print("\n[Running tracker] Press 'q' or Esc to quit.\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from camera.")
                break

            # Mirror for selfie view
            frame = cv2.flip(frame, 1)

            annotated_frame, landmarks = detector.process_frame(frame)

            rep_count = counter.counter
            feedback = "No subject detected"
            elbow_angle, hip_angle, knee_angle = None, None, None

            if landmarks:
                rep_count, feedback, elbow_angle, hip_angle, knee_angle = counter.update(landmarks[0])

            # FPS calculation
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_history.append(1.0 / dt)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            draw_hud(annotated_frame, rep_count, feedback, elbow_angle, hip_angle, knee_angle, avg_fps)

            cv2.imshow("Push-Up Tracker", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exited cleanly.")


if __name__ == "__main__":
    run_pipeline(camera_index=0)