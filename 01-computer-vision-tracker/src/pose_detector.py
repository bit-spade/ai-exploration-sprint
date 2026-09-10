import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PoseDetector:
    """Wrapper around YOLO pose estimation running on Apple Silicon / CPU."""

    # Standard COCO skeleton connections: (start_id, end_id, BGR_color)
    SKELETON_CONNECTIONS = (
        # Head / face
        (0, 1, (255, 180, 0)), (0, 2, (255, 180, 0)),
        (1, 3, (255, 180, 0)), (2, 4, (255, 180, 0)),
        # Torso
        (5, 6, (0, 255, 255)), (5, 11, (0, 255, 255)),
        (6, 12, (0, 255, 255)), (11, 12, (0, 255, 255)),
        # Left arm
        (5, 7, (255, 0, 128)), (7, 9, (255, 0, 128)),
        # Right arm
        (6, 8, (0, 255, 0)), (8, 10, (0, 255, 0)),
        # Left leg
        (11, 13, (255, 0, 128)), (13, 15, (255, 0, 128)),
        # Right leg
        (12, 14, (0, 255, 0)), (14, 16, (0, 255, 0))
    )

    def __init__(
        self,
        model_name: str = "yolo26n-pose.pt",
        conf_threshold: float = 0.5,
        keypoint_confidence: float = 0.5
    ) -> None:
        self.conf_threshold = conf_threshold
        self.kp_threshold = keypoint_confidence

        # Use Apple Silicon GPU (mps) if available, otherwise fallback to cpu
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[PoseDetector] Loading {model_name} on {self.device.upper()}...")
        self.model = YOLO(model_name)

        # Warm up model once so the first webcam frame doesn't freeze
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Runs YOLO pose on a frame and returns (annotated_frame, list_of_landmarks)."""
        results = self.model(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        parsed_persons = []
        if results.keypoints is None or len(results.keypoints) == 0:
            return frame, parsed_persons

        keypoints_xy = results.keypoints.xy.cpu().numpy()
        keypoints_conf = results.keypoints.conf.cpu().numpy()

        for pts, confs in zip(keypoints_xy, keypoints_conf):
            person_landmarks = [
                {"id": i, "x": int(x), "y": int(y), "conf": float(conf)}
                for i, ((x, y), conf) in enumerate(zip(pts, confs))
            ]
            parsed_persons.append(person_landmarks)
            self._draw_skeleton(frame, person_landmarks)

        return frame, parsed_persons

    def _draw_skeleton(self, frame: np.ndarray, landmarks: list[dict]) -> None:
        """Draws bone lines and joint dots on the frame."""
        # Draw bones
        for start_idx, end_idx, color in self.SKELETON_CONNECTIONS:
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]

            if pt1["conf"] > self.kp_threshold and pt2["conf"] > self.kp_threshold:
                cv2.line(
                    frame,
                    (pt1["x"], pt1["y"]),
                    (pt2["x"], pt2["y"]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

        # Draw joints
        for lm in landmarks:
            if lm["conf"] > self.kp_threshold:
                cv2.circle(
                    frame,
                    (lm["x"], lm["y"]),
                    radius=4,
                    color=(0, 255, 0),
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )