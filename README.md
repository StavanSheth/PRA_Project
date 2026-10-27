# Live Multi-Model Vision Studio (MediaPipe & YOLOv8)

A lightweight, real-time computer vision application supporting both **Google MediaPipe Hand Gesture Recognition** and **Ultralytics YOLOv8 Nano (`yolov8n.pt`) Object Detection**, available as both an interactive web dashboard and a native desktop GUI.

---

## 🚀 Key Features

- **Multi-Model Pipeline Selection**:
  - **Model A: Google MediaPipe Gesture Recognizer** (`models/gesture_recognizer.task`): Tracks 21 3D hand landmarks, finger skeletal bonds, and classifies 7 official hand gestures.
  - **Model B: Ultralytics YOLOv8 Nano** (`yolov8n.pt`): Detects 80 real-world object classes (people, vehicles, animals, accessories, electronics, food, furniture, etc.) with real-time bounding boxes and confidence scores.
- **Interactive Start & Stop Controls**: Pause or resume the camera feed on the fly directly from the web interface without restarting the server.
- **Live Camera Stream**: Real-time webcam streaming with horizontal mirroring and smooth overlay rendering.
- **Static Image Analysis**: Upload any photo from your machine to get immediate predictions from whichever model is currently active.
- **Safe Lifecycle & Resource Management**: Clean camera release and non-blocking inference.

---

## 📂 Project Structure

```text
PRA Project/
├── app/
│   ├── __init__.py           # Package marker
│   ├── camera_service.py     # Threaded OpenCV capture, BGR->RGB, monotonic timestamps, FPS estimation
│   ├── config.py             # Config constants, thresholds, paths, and color tokens
│   ├── controller.py         # Thread-safe queue orchestration, smoothing, and lifecycle
│   ├── gui.py                # Tkinter dark UI, custom video canvas overlays, and cards
│   └── model_service.py      # MediaPipe model downloader, LIVE_STREAM recognizer, and parser
├── models/
│   └── gesture_recognizer.task  # Official MediaPipe model bundle (auto-downloaded on first run)
├── tests/
│   ├── __init__.py
│   ├── test_model_loading.py        # Model download, Tasks initialization, and synthetic inference
│   └── test_prediction_parsing.py   # Unit tests for edge cases, label mappings, and bounding boxes
├── main.py                   # Main desktop application entry point
├── requirements.txt          # Minimal dependency requirements
└── README.md                 # Project documentation
```

---

## 🛠️ Technology Stack

- **Python 3.11**
- **MediaPipe Tasks Vision** (`mediapipe>=1.1.1`): Gesture Recognizer in `LIVE_STREAM` mode
- **OpenCV** (`opencv-python`): Video capture, BGR to RGB conversion, skeletal drawing
- **Tkinter**: Native Python desktop GUI
- **Pillow** (`PIL`): Image array to Tkinter PhotoImage conversion
- **NumPy**: Image array manipulation

---

## 📋 Installation & Running (Windows PowerShell)

### 1. Create and Activate Virtual Environment
Open PowerShell in the project directory:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify Model & Run Automated Tests
```powershell
python -m unittest discover tests
```
*Note: On first execution, the test suite or application will automatically download `models/gesture_recognizer.task` (~8.3 MB) from Google's official Cloud Storage bucket.*

### 4. Launch Application
```powershell
python main.py
```

---

## 🖥️ User Interface Overview

- **Preview Window**: Displays the live 640×480 webcam stream with overlaid 21-point skeletal hand graphs, bounding boxes, and gesture banners.
- **Recognized Gesture Card**: Large high-contrast display of current gesture and confidence score.
- **Detection Details Card**: Handedness (Left/Right), number of detected hands (up to 2), and current inference state.
- **Controls**:
  - `▶ Start Camera`: Starts the background camera capture thread and streaming pipeline.
  - `⏹ Stop Camera`: Stops video streaming and resets display to standby.
  - `✕ Exit Application`: Releases camera hardware, shuts down recognizer, and terminates cleanly.
  - `Enable Prediction Smoothing`: Toggles rolling 5-frame majority voting.
- **Status Bar**: Live status indicator (Loaded / Running / Idle) and current real-time FPS counter.

---

## 🧪 Verification & Test Results

The project includes unit and integration tests:

1. **`tests/test_model_loading.py`**:
   - Auto-downloads model if missing and verifies file size.
   - Instantiates `vision.GestureRecognizer` in `LIVE_STREAM` mode.
   - Submits a synthetic 640×480 RGB frame with a millisecond timestamp and validates asynchronous callback execution.
2. **`tests/test_prediction_parsing.py`**:
   - Validates `None` result handling -> `"No hand detected"`.
   - Validates empty gestures handling -> `"No hand detected"`.
   - Validates unrecognized gesture `"None"` -> `"Unknown gesture"`.
   - Validates recognized gestures (`Open_Palm`, `Victory`, etc.) -> verified display names, confidences, handedness, and bounding box coordinates.
3. **Camera Hardware Smoke Test**:
   - Verified OpenCV camera initialization with Windows DirectShow (`cv2.CAP_DSHOW`) and standard backend fallback.

All automated tests passed:
```text
Ran 6 tests in 1.091s
OK
```

---

## ⚠️ Limitations & Troubleshooting

- **Webcam Access**: Ensure no other application (Zoom, Teams, Discord, browser tabs) is occupying camera index `0`.
- **Low Light / Occlusion**: Extreme low light or partially occluded hands can reduce detection confidence below the `0.5` threshold.
- **Multiple Hands**: MediaPipe is configured for up to 2 hands; the primary gesture displayed reflects the highest scoring hand.
