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

    def update(self, landmarks: list[dict]) -> tuple[int, str, float | None, float | None, float | None]:
        """
        Updates rep count and body posture from keypoints.
        Returns: (rep_count, feedback_msg, elbow_angle, hip_angle, knee_angle)
        """
        if not landmarks or len(landmarks) < 17:
            return self.counter, "No subject detected", None, None, None

        # Pick whichever side faces the camera better
        left_arm_conf = (landmarks[5]['conf'] + landmarks[7]['conf'] + landmarks[9]['conf']) / 3.0
        right_arm_conf = (landmarks[6]['conf'] + landmarks[8]['conf'] + landmarks[10]['conf']) / 3.0

        if left_arm_conf >= right_arm_conf:
            s_idx, e_idx, w_idx, h_idx, k_idx, a_idx = 5, 7, 9, 11, 13, 15
        else:
            s_idx, e_idx, w_idx, h_idx, k_idx, a_idx = 6, 8, 10, 12, 14, 16

        # Check visibility for arm joints
        for idx in (s_idx, e_idx, w_idx):
            if landmarks[idx]['conf'] < self.min_confidence:
                return self.counter, "Keep arm visible", None, None, None

        shoulder = (landmarks[s_idx]['x'], landmarks[s_idx]['y'])
        elbow = (landmarks[e_idx]['x'], landmarks[e_idx]['y'])
        wrist = (landmarks[w_idx]['x'], landmarks[w_idx]['y'])

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Hands must be planted down on the floor (lower half of screen and below shoulders)
        hands_on_floor = wrist[1] > 340 and wrist[1] >= shoulder[1] - 10
        if not hands_on_floor:
            return self.counter, "Place hands on the floor", elbow_angle, None, None

        # Check lower body keypoints if in frame
        hip_visible = landmarks[h_idx]['conf'] >= 0.35
        knee_visible = landmarks[k_idx]['conf'] >= 0.35
        ankle_visible = landmarks[a_idx]['conf'] >= 0.35

        hip_angle = None
        knee_angle = None
        knees_on_floor = False

        if knee_visible:
            knee = (landmarks[k_idx]['x'], landmarks[k_idx]['y'])

            # If knees are near wrist level on screen, knees are touching the floor
            if abs(knee[1] - wrist[1]) < 80:
                knees_on_floor = True

            # Measure real hip angle (shoulder - hip - knee)
            if hip_visible:
                hip = (landmarks[h_idx]['x'], landmarks[h_idx]['y'])
                hip_angle = calculate_angle(shoulder, hip, knee)

            # Measure real knee angle (hip - knee - ankle) if ankle is visible
            if ankle_visible and hip_visible:
                ankle = (landmarks[a_idx]['x'], landmarks[a_idx]['y'])
                knee_angle = calculate_angle(hip, knee, ankle)
                # In a knee push-up, feet are lifted in the air above the knees
                if ankle[1] < knee[1] - 20:
                    knees_on_floor = True

        # Rep counting state machine
        if elbow_angle <= self.down_threshold:
            if knees_on_floor:
                self.feedback = "Off your knees - full plank!"
            else:
                self.feedback = "Push up!"
            self.stage = "down"

        elif elbow_angle >= self.up_threshold:
            if self.stage == "down":
                if not knees_on_floor:
                    self.counter += 1
                    self.feedback = "Good rep!"
                else:
                    self.feedback = "Form off - on your knees"
                self.stage = "up"

            elif self.stage == "up":
                if knees_on_floor:
                    self.feedback = "Lift knees off floor"
                else:
                    self.feedback = "Ready - lower chest"

            elif self.stage is None:
                self.stage = "up"
                self.feedback = "Ready - lower chest"

        return self.counter, self.feedback, elbow_angle, hip_angle, knee_angle