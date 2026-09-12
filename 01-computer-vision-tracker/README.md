# Real-Time Push-Up Tracker

A computer vision tracker built with YOLOv8/26 Pose and OpenCV that analyzes push-up repetitions and form in real time using a standard webcam (accelerated with Apple Silicon MPS).

## Quickstart

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the live tracker
python3 main.py
```
> Press **`q`** or **`ESC`** in the video window to quit cleanly.

---

## Project Structure

```text
01-computer-vision-tracker/
├── src/
│   ├── __init__.py
│   ├── pose_detector.py   # YOLO pose wrapper with Apple Silicon (MPS) acceleration
│   └── utils.py           # Geometry, ground-contact checks & rep counter state machine
├── main.py                # OpenCV webcam capture, telemetry HUD & main loop
└── requirements.txt       # Dependencies (PyTorch, Ultralytics, OpenCV)
```

---

## Sprint Notes: Real-World CV Challenges & Lessons Learned

While building this, moving from simple arm tracking to full-body posture tracking with a laptop camera turned out to be much trickier than expected. Here are the main problems I ran into and how I solved them:

### 1. The Floor-Camera Perspective Problem
Most fitness tutorials assume the camera is placed 3 meters away from the side. In reality, I had my MacBook sitting on the floor directly in front of me:
- **The Issue:** When facing the camera head-on, your head, chest, and hips all stack vertically on the screen. Textbook 2D spine angles completely break down because the body's depth axis (Z-axis) gets compressed onto the 2D frame. Trying to measure hip sag from the front caused constant false alarms.
- **How I Fixed It:** Instead of trusting 2D spine angles from the front, I focused on ground-contact geometry: checking that hands are firmly planted in the lower half of the frame and verifying that the person is prone rather than standing upright.

### 2. Ankle Detection & Knee Push-Ups
- **The Issue:** YOLO is great at finding shoulders, elbows, and knees, but confidence on ankles drops significantly during a push-up (feet are tucked away, shoes blend with the floor, or toes are bent). Strictly requiring ankle keypoints caused the tracker to frequently think legs were missing or bent.
- **How I Fixed It:** To catch knee push-ups without relying on flaky ankle keypoints, I added a relative floor-plane check: when someone is on their knees, the knees touch the ground right near wrist level. In a true plank, the knees remain elevated off the carpet.

### 3. Preventing "Air" Push-Ups (False Reps)
- **The Issue:** Initially, just standing or sitting in front of the camera and bending arms would count reps, because the elbow angle still went from 90° to 160°.
- **How I Fixed It:** Added a floor-plane check requiring wrists to be planted low in the frame (lower half of the 720p view and at or below shoulder height). If hands are moving freely in the air, the tracker immediately prompts to place hands on the floor.