"""
Flask Web Server for Live Hand Gesture Recognition and Image Analysis.
Provides real-time browser streaming at http://localhost:5000 and REST API.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.config import (
    MODEL_PATH,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    GESTURE_DISPLAY_NAMES,
)
from app.model_service import ModelService, GestureResultData
from app.gui import HAND_CONNECTIONS

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Global states
model_service = ModelService(model_path=MODEL_PATH)
latest_result = GestureResultData()
latest_fps = 0.0
result_lock = threading.Lock()
camera_lock = threading.Lock()


def on_mediapipe_result(parsed_data: GestureResultData, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = parsed_data


# Initialize model at startup
model_service.load_recognizer(result_callback=on_mediapipe_result)


def generate_video_stream():
    """Captures frames from webcam, passes to MediaPipe, draws overlays, and yields MJPEG stream."""
    global latest_fps

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Server] Camera could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start_time_ns = time.time_ns()
    last_timestamp_ms = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror frame horizontally
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Monotonic timestamp
            now_ms = int((time.time_ns() - start_time_ns) / 1_000_000)
            if now_ms <= last_timestamp_ms:
                now_ms = last_timestamp_ms + 1
            last_timestamp_ms = now_ms

            # Compute FPS
            now = time.time()
            delta = now - prev_time
            if delta > 0:
                latest_fps = 1.0 / delta
            prev_time = now

            # Submit RGB to MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model_service.recognize_async(rgb_frame, now_ms)

            # Draw overlays from latest result
            with result_lock:
                current = latest_result

            canvas = frame.copy()
            if current and current.has_hand:
                # Bounding boxes
                for (bx1, by1, bx2, by2) in current.bounding_boxes:
                    px1 = max(0, int(bx1 * w) - 10)
                    py1 = max(0, int(by1 * h) - 10)
                    px2 = min(w, int(bx2 * w) + 10)
                    py2 = min(h, int(by2 * h) + 10)
                    cv2.rectangle(canvas, (px1, py1), (px2, py2), (16, 185, 129), 2)

                # Landmarks & bones
                for hand_pts in current.landmarks:
                    for (i1, i2) in HAND_CONNECTIONS:
                        if i1 < len(hand_pts) and i2 < len(hand_pts):
                            pt1 = (int(hand_pts[i1][0] * w), int(hand_pts[i1][1] * h))
                            pt2 = (int(hand_pts[i2][0] * w), int(hand_pts[i2][1] * h))
                            cv2.line(canvas, pt1, pt2, (56, 189, 248), 2, cv2.LINE_AA)

                    for pt in hand_pts:
                        cv2.circle(canvas, (int(pt[0] * w), int(pt[1] * h)), 4, (245, 158, 11), -1, cv2.LINE_AA)

                # Gesture Banner
                banner_text = f"{current.display_name} ({current.confidence*100:.1f}%)"
                cv2.rectangle(canvas, (15, 15), (min(w - 15, 380), 55), (15, 23, 42), -1)
                cv2.rectangle(canvas, (15, 15), (min(w - 15, 380), 55), (56, 189, 248), 1)
                cv2.putText(
                    canvas,
                    banner_text,
                    (25, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (248, 250, 252),
                    2,
                    cv2.LINE_AA,
                )

            # Encode as JPEG
            ret, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.01)

    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with result_lock:
        res = latest_result
    return jsonify(
        {
            "has_hand": res.has_hand,
            "gesture_name": res.gesture_name,
            "display_name": res.display_name,
            "confidence": res.confidence,
            "handedness": res.handedness,
            "num_hands": res.num_hands,
            "fps": round(latest_fps, 1),
        }
    )


@app.route("/api/predict_image", methods=["POST"])
def api_predict_image():
    """Predicts hand gesture from an uploaded static image file."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"success": False, "error": "Failed to decode image"}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Use a temporary synchronous recognizer for static IMAGE mode
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        res = recognizer.recognize(mp_image)
        parsed = ModelService.parse_result(res)

    return jsonify(
        {
            "success": True,
            "has_hand": parsed.has_hand,
            "gesture_name": parsed.gesture_name,
            "display_name": parsed.display_name,
            "confidence": parsed.confidence,
            "handedness": parsed.handedness,
            "num_hands": parsed.num_hands,
        }
    )


if __name__ == "__main__":
    print("[Server] Starting Hand Gesture Studio on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
