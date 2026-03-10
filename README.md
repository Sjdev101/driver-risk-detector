# Driver Risk Detector
> Real-time driver monitoring system using computer vision

![CI](https://github.com/YOUR_USERNAME/driver-risk-detector/actions/workflows/ci.yml/badge.svg)

## What it does
Monitors a driver through a camera feed and detects two risk events in real time:
- **Drowsiness** — detects eye closure using Eye Aspect Ratio (EAR)
- **Phone use** — detects phone holding using YOLOv8 object detection

All events are logged with a timestamp and confidence score.

## System Architecture
```
Camera Feed
    ↓
Face Mesh (MediaPipe) → 468 landmarks
    ↓
Eye Landmarks → EAR Calculation
    ↓
Decision Engine → DROWSY / ALERT
    ↓
Event Logger → timestamp, type, confidence
```

## Tech Stack
- Python, OpenCV, MediaPipe, YOLOv8, ONNX Runtime

## Benchmark
| Method | FPS |
|---|---|
| PyTorch | 24.95 |
| ONNX Runtime | 27.16 |
| Improvement | +8.9% |

## Known Limitations
- EAR is sensitive to head tilt (see configs/known_limitations.md)
- Landmark accuracy varies by eye shape

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/driver-risk-detector.git
cd driver-risk-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.run_detection
```
```

Replace `YOUR_USERNAME` with your actual GitHub username, save it, then commit:
```
git add .
```
```
git commit -m "docs: add project README with architecture and benchmarks"
```
```
git push origin main