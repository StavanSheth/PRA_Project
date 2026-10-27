"""
Flask Web Server for Live Multi-Model Vision Studio.
Supports Google MediaPipe Hand Gesture Recognition AND Ultralytics YOLOv8 Nano (yolov8n.pt).
Uses a single background camera capture thread to guarantee thread safety and strictly monotonic timestamps.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
from ultralytics import YOLO

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

# Path to YOLOv8 model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Global locks and configuration
model_lock = threading.Lock()
frame_lock = threading.Lock()
active_model = "mediapipe"  # "mediapipe" or "yolov8"
camera_active = True

# MediaPipe state
mediapipe_service = ModelService(model_path=MODEL_PATH)
latest_gesture_result = GestureResultData()

# YOLOv8 state
yolo_model = None
try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"[Server] Successfully loaded YOLOv8 model from {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"[Server] Warning: Failed to load YOLOv8 model: {e}")

latest_yolo_result = {
    "detected_objects": [],
    "total_objects": 0,
    "top_class": "None",
    "top_confidence": 0.0,
}

latest_fps = 0.0
global_latest_jpeg = b""


def on_mediapipe_result(parsed_data: GestureResultData, timestamp_ms: int):
    global latest_gesture_result
    with model_lock:
        latest_gesture_result = parsed_data


# Initialize MediaPipe recognizer
mediapipe_service.load_recognizer(result_callback=on_mediapipe_result)


def create_standby_frame(message="Camera Paused"):
    """Creates a dark styled standby frame to stream when camera is stopped."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (15, 23, 42)  # Dark slate background #0f172a

    cv2.rectangle(frame, (20, 20), (FRAME_WIDTH - 20, FRAME_HEIGHT - 20), (51, 65, 85), 2)
    cv2.putText(
        frame,
        message,
        (FRAME_WIDTH // 2 - 140, FRAME_HEIGHT // 2 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (148, 163, 184),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Click 'Start Camera' to resume live stream",
        (FRAME_WIDTH // 2 - 180, FRAME_HEIGHT // 2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (56, 189, 248),
        1,
        cv2.LINE_AA,
    )

    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes() if ret else b""


# Pre-generate standby JPEG
STANDBY_JPEG = create_standby_frame("Camera is Paused")


def background_camera_worker():
    """
    Dedicated background thread capturing camera frames, running inference,
    and updating the global JPEG buffer. Ensures single-camera ownership and strictly
    increasing monotonic timestamps for MediaPipe.
    """
    global latest_fps, global_latest_jpeg, latest_yolo_result

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Server] Camera device could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start_time_ns = time.time_ns()
    last_timestamp_ms = 0
    prev_time = time.time()

    print("[Server] Camera capture worker thread started.")

    while True:
        if not camera_active:
            with frame_lock:
                global_latest_jpeg = STANDBY_JPEG
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Mirror horizontally for natural webcam interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Strictly increasing monotonic timestamp in milliseconds
        now_ms = int((time.time_ns() - start_time_ns) / 1_000_000)
        if now_ms <= last_timestamp_ms:
            now_ms = last_timestamp_ms + 1
        last_timestamp_ms = now_ms

        # Compute smooth FPS
        now = time.time()
        delta = now - prev_time
        if delta > 0:
            latest_fps = 1.0 / delta
        prev_time = now

        canvas = frame.copy()

        with model_lock:
            current_mode = active_model

        # -------------------------------------------------------------
        # PIPELINE A: Google MediaPipe Gesture Recognizer
        # -------------------------------------------------------------
        if current_mode == "mediapipe":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                mediapipe_service.recognize_async(rgb_frame, now_ms)
            except Exception as e:
                # Catch any transient timestamp or buffer exception safely
                pass

            with model_lock:
                gesture_data = latest_gesture_result

            if gesture_data and gesture_data.has_hand:
                # Bounding box
                for (bx1, by1, bx2, by2) in gesture_data.bounding_boxes:
                    px1 = max(0, int(bx1 * w) - 10)
                    py1 = max(0, int(by1 * h) - 10)
                    px2 = min(w, int(bx2 * w) + 10)
                    py2 = min(h, int(by2 * h) + 10)
                    cv2.rectangle(canvas, (px1, py1), (px2, py2), (16, 185, 129), 2)

                # Hand landmarks and bones
                for hand_pts in gesture_data.landmarks:
                    for (i1, i2) in HAND_CONNECTIONS:
                        if i1 < len(hand_pts) and i2 < len(hand_pts):
                            pt1 = (int(hand_pts[i1][0] * w), int(hand_pts[i1][1] * h))
                            pt2 = (int(hand_pts[i2][0] * w), int(hand_pts[i2][1] * h))
                            cv2.line(canvas, pt1, pt2, (56, 189, 248), 2, cv2.LINE_AA)

                    for pt in hand_pts:
                        cv2.circle(canvas, (int(pt[0] * w), int(pt[1] * h)), 4, (245, 158, 11), -1, cv2.LINE_AA)

                # Top banner overlay
                banner_text = f"{gesture_data.display_name} ({gesture_data.confidence*100:.1f}%)"
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

        # -------------------------------------------------------------
        # PIPELINE B: Ultralytics YOLOv8 Nano Object Detection
        # -------------------------------------------------------------
        elif current_mode == "yolov8" and yolo_model is not None:
            try:
                results = yolo_model(frame, conf=0.35, verbose=False)[0]
                detected_list = []

                boxes = results.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        cls_name = yolo_model.names.get(cls_id, str(cls_id))

                        detected_list.append({
                            "name": cls_name,
                            "confidence": round(conf, 3),
                            "bbox": [x1, y1, x2, y2]
                        })

                        # Draw bounding rectangle
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), (245, 158, 11), 2)

                        # Label badge
                        label_str = f"{cls_name} {conf*100:.0f}%"
                        (label_w, label_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + label_w + 10, y1), (245, 158, 11), -1)
                        cv2.putText(
                            canvas,
                            label_str,
                            (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (15, 23, 42),
                            1,
                            cv2.LINE_AA
                        )

                # Update global YOLO state
                with model_lock:
                    if detected_list:
                        top_obj = detected_list[0]
                        latest_yolo_result = {
                            "detected_objects": detected_list,
                            "total_objects": len(detected_list),
                            "top_class": top_obj["name"],
                            "top_confidence": top_obj["confidence"],
                        }
                    else:
                        latest_yolo_result = {
                            "detected_objects": [],
                            "total_objects": 0,
                            "top_class": "No object detected",
                            "top_confidence": 0.0,
                        }

                # Top model badge
                badge_text = f"YOLOv8 Nano • {len(detected_list)} detected"
                cv2.rectangle(canvas, (15, 15), (min(w - 15, 300), 50), (15, 23, 42), -1)
                cv2.rectangle(canvas, (15, 15), (min(w - 15, 300), 50), (245, 158, 11), 1)
                cv2.putText(
                    canvas,
                    badge_text,
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (248, 250, 252),
                    2,
                    cv2.LINE_AA,
                )
            except Exception as e:
                pass

        # Compress canvas to JPEG and update shared buffer
        ret, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with frame_lock:
                global_latest_jpeg = buffer.tobytes()

        time.sleep(0.01)


# Start the background capture worker
camera_thread = threading.Thread(target=background_camera_worker, daemon=True)
camera_thread.start()


def generate_video_stream():
    """Generator yielding latest JPEG frame from the shared memory buffer to HTTP clients."""
    while True:
        with frame_lock:
            frame_data = global_latest_jpeg

        if frame_data:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
        time.sleep(0.03)  # ~30 FPS client rate


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/camera/control", methods=["POST"])
def api_camera_control():
    """Start or stop the camera video acquisition loop."""
    global camera_active
    data = request.get_json(silent=True) or {}
    action = data.get("action", "").lower()

    if action == "start":
        camera_active = True
        return jsonify({"success": True, "camera_active": True, "message": "Camera started."})
    elif action == "stop":
        camera_active = False
        return jsonify({"success": True, "camera_active": False, "message": "Camera stopped."})
    else:
        camera_active = not camera_active
        return jsonify({"success": True, "camera_active": camera_active})


@app.route("/api/model/select", methods=["POST"])
def api_select_model():
    """Switches active model between MediaPipe and YOLOv8."""
    global active_model
    data = request.get_json(silent=True) or {}
    requested_model = data.get("model", "").lower()

    if requested_model in ["mediapipe", "yolov8"]:
        with model_lock:
            active_model = requested_model
        print(f"[Server] Active model switched to: {active_model}")
        return jsonify({"success": True, "active_model": active_model})

    return jsonify({"success": False, "error": f"Invalid model: {requested_model}"}), 400


@app.route("/api/status")
def api_status():
    """Returns real-time status and detection outputs for both models."""
    with model_lock:
        curr_mode = active_model
        gest = latest_gesture_result
        yolo_res = dict(latest_yolo_result)

    response_data = {
        "camera_active": camera_active,
        "active_model": curr_mode,
        "fps": round(latest_fps, 1),
    }

    if curr_mode == "mediapipe":
        response_data.update({
            "has_hand": gest.has_hand,
            "gesture_name": gest.gesture_name,
            "display_name": gest.display_name,
            "confidence": gest.confidence,
            "handedness": gest.handedness,
            "num_hands": gest.num_hands,
        })
    elif curr_mode == "yolov8":
        response_data.update({
            "total_objects": yolo_res["total_objects"],
            "top_class": yolo_res["top_class"],
            "top_confidence": yolo_res["top_confidence"],
            "detected_objects": yolo_res["detected_objects"],
        })

    return jsonify(response_data)


@app.route("/api/yolo/labels")
def api_yolo_labels():
    """Returns the complete list of 80 COCO classes supported by YOLOv8."""
    if yolo_model and hasattr(yolo_model, "names"):
        return jsonify({"success": True, "total": len(yolo_model.names), "labels": yolo_model.names})
    return jsonify({"success": False, "error": "YOLOv8 model is not loaded."}), 500


@app.route("/api/predict_image", methods=["POST"])
def api_predict_image():
    """Runs image prediction using currently active model on an uploaded file."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"success": False, "error": "Failed to decode image"}), 400

    with model_lock:
        curr_mode = active_model

    if curr_mode == "mediapipe":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

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

        return jsonify({
            "success": True,
            "model": "mediapipe",
            "has_hand": parsed.has_hand,
            "gesture_name": parsed.gesture_name,
            "display_name": parsed.display_name,
            "confidence": parsed.confidence,
            "handedness": parsed.handedness,
            "num_hands": parsed.num_hands,
        })

    elif curr_mode == "yolov8":
        if yolo_model is None:
            return jsonify({"success": False, "error": "YOLOv8 model unavailable"}), 500

        res = yolo_model(img_bgr, conf=0.3, verbose=False)[0]
        detected = []
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = yolo_model.names.get(cls_id, str(cls_id))
                detected.append({"name": name, "confidence": round(conf, 3)})

        return jsonify({
            "success": True,
            "model": "yolov8",
            "total_objects": len(detected),
            "top_class": detected[0]["name"] if detected else "None detected",
            "top_confidence": detected[0]["confidence"] if detected else 0.0,
            "detected_objects": detected,
        })


if __name__ == "__main__":
    print("[Server] Starting Multi-Model Vision Studio on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
