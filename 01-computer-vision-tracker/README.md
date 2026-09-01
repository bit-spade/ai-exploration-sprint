# Real-Time Pose Tracker

Computer vision application that processes a video feed to detect pose landmarks and calculate joint angles in real time using MediaPipe and OpenCV.

## Project Architecture

01-computer-vision-tracker/
├── src/
│   ├── init.py
│   ├── pose_detector.py   # MediaPipe pipeline wrapper
│   └── utils.py           # Angle calculations & math helpers
├── app.py                 # Streamlit UI runner
├── main.py                # Standalone OpenCV entrypoint
└── requirements.txt       # safe dependencies