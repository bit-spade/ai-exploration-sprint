import numpy as np


def calculate_angle(
    a: tuple[int, int] | list[int],
    b: tuple[int, int] | list[int],
    c: tuple[int, int] | list[int]
) -> float:
    """Returns the angle at joint B formed by points A-B-C (0 to 180 degrees)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    angle_ba = np.arctan2(ba[1], ba[0])
    angle_bc = np.arctan2(bc[1], bc[0])

    angle = np.abs(np.degrees(angle_ba - angle_bc))
    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


class PushUpCounter:
    """Tracks push-up repetitions and checks exercise form."""

    def __init__(
        self,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
        min_confidence: float = 0.5
    ) -> None:
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.min_confidence = min_confidence

        self.counter = 0
        self.stage: str | None = None  # "up" or "down"
        self.feedback = "Get into plank position"

    def update(self, landmarks: list[dict]) -> tuple[int, str, float | None, float | None]:
        """
        Updates rep count and body posture from keypoints.
        Returns: (rep_count, feedback_msg, elbow_angle, hip_angle)
        """
        if not landmarks or len(landmarks) < 17:
            return self.counter, "No subject detected", None, None

        # Pick whichever side faces the camera better
        # Left side: shoulder=5, elbow=7, wrist=9, hip=11, ankle=15
        # Right side: shoulder=6, elbow=8, wrist=10, hip=12, ankle=16
        left_arm_conf = (landmarks[5]['conf'] + landmarks[7]['conf'] + landmarks[9]['conf']) / 3.0
        right_arm_conf = (landmarks[6]['conf'] + landmarks[8]['conf'] + landmarks[10]['conf']) / 3.0

        if left_arm_conf >= right_arm_conf:
            s_idx, e_idx, w_idx, h_idx, a_idx = 5, 7, 9, 11, 15
        else:
            s_idx, e_idx, w_idx, h_idx, a_idx = 6, 8, 10, 12, 16

        # Check visibility for the arm joints
        for idx in (s_idx, e_idx, w_idx):
            if landmarks[idx]['conf'] < self.min_confidence:
                return self.counter, "Keep arm visible", None, None

        shoulder = (landmarks[s_idx]['x'], landmarks[s_idx]['y'])
        elbow = (landmarks[e_idx]['x'], landmarks[e_idx]['y'])
        wrist = (landmarks[w_idx]['x'], landmarks[w_idx]['y'])

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Whole-body check: calculate hip angle if hip & ankle are visible
        hip_angle = None
        is_body_straight = True
        if landmarks[h_idx]['conf'] >= self.min_confidence and landmarks[a_idx]['conf'] >= self.min_confidence:
            hip = (landmarks[h_idx]['x'], landmarks[h_idx]['y'])
            ankle = (landmarks[a_idx]['x'], landmarks[a_idx]['y'])
            hip_angle = calculate_angle(shoulder, hip, ankle)

            # Planks/push-ups usually want hip angle roughly between 150 and 180 deg
            if hip_angle < 150.0:
                is_body_straight = False

        # State machine for reps
        if elbow_angle <= self.down_threshold:
            if not is_body_straight:
                self.feedback = "Straighten your back!"
            else:
                self.feedback = "Push up!"
            self.stage = "down"

        elif elbow_angle >= self.up_threshold:
            if self.stage == "down":
                if is_body_straight:
                    self.counter += 1
                    self.feedback = "Good rep!"
                else:
                    self.feedback = "Form off - keep hips aligned"
                self.stage = "up"
            elif self.stage is None:
                self.stage = "up"
                self.feedback = "Lower your chest"

        return self.counter, self.feedback, elbow_angle, hip_angle