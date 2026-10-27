# Live Hand Gesture Recognition Desktop Application

A lightweight, real-time desktop application built in Python and native Tkinter that recognizes hand gestures using Google's pretrained MediaPipe Gesture Recognizer model (`gesture_recognizer.task`).

The application runs entirely locally on CPU inference using OpenCV for camera streaming and MediaPipe Tasks Vision in asynchronous `LIVE_STREAM` mode.

---

## 🚀 Key Features

- **Live Camera Feed**: Captures laptop webcam feed with horizontal mirroring for natural interaction.
- **Skeletal Hand Landmarks & Bounding Boxes**: Automatically tracks and overlays 21 joint positions, skeletal connections, and bounding boxes in real time.
- **Official Gesture Recognition**: Supports Google's official MediaPipe gesture labels:
  - `Closed_Fist` (✊)
  - `Open_Palm` (✋)
  - `Pointing_Up` (☝️)
  - `Thumb_Down` (👎)
  - `Thumb_Up` (👍)
  - `Victory` (✌️)
  - `ILoveYou` (🤟)
  - `Unknown gesture` (hand present but gesture unrecognized / "None")
  - `No hand detected` (no hand in frame)
- **Detailed Prediction Panel**: Real-time display of recognized gesture, confidence score with progress bar, handedness (Left/Right), detected hand count, and inference state.
- **Prediction Smoothing**: Optional rolling window majority voting (5 frames) to eliminate visual flicker while preserving raw debugging capability.
- **Safe Lifecycle & Resource Management**: Clean startup and shutdown; camera hardware is cleanly released (`cap.release()`) and MediaPipe recognizer is closed safely.

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

### 4. Launch Options

#### Option A: One-Click Desktop Icon (Recommended)
Double-click the **Hand Gesture Studio** icon directly on your Windows Desktop!
- Silently launches the Python backend in the background (no console popups).
- Opens Google Chrome in standalone app mode (`--app=http://localhost:5000`) without browser tabs or URL bars.
- Automatically stops the backend server and releases your webcam when the window is closed.

*If you need to re-create the desktop shortcut:*
```powershell
python scripts/create_desktop_shortcut.py
```

#### Option B: Standalone Desktop Window (Tkinter)
```powershell
python main.py
```

#### Option C: Manual Web Server
```powershell
python server.py
# Access at http://localhost:5000
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
