"""
Controller orchestrating CameraService, MediaPipe ModelService, YOLOv8, and Tkinter GUI via thread-safe queues.
Ensures non-blocking inference, model switching, and main-thread Tkinter updates.
"""

import os
import queue
import sys
import threading
from collections import Counter
from ultralytics import YOLO

from app.config import (
    BASE_DIR,
    SMOOTHING_WINDOW_SIZE,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
)
from app.model_service import ModelService, GestureResultData
from app.camera_service import CameraService

YOLO_PATH = os.path.join(BASE_DIR, "yolov8n.pt")


class GestureAppController:
    """Coordinates lifecycle, multi-model execution (MediaPipe & YOLOv8), and GUI synchronization."""

    def __init__(self, root):
        self.root = root
        self.gui = None

        # Thread-safe queue for delivering frames & predictions to GUI main thread
        self.display_queue = queue.Queue(maxsize=2)

        # Active model mode: "mediapipe" or "yolov8"
        self.active_model = "mediapipe"
        self.model_lock = threading.Lock()

        # Services
        self.model_service = ModelService()
        self.camera_service = CameraService(
            on_frame_callback=self._on_camera_frame,
            on_error_callback=self._on_camera_error,
        )

        # YOLOv8 model initialization
        self.yolo_model = None
        try:
            if os.path.exists(YOLO_PATH):
                self.yolo_model = YOLO(YOLO_PATH)
                print(f"[Controller] Loaded YOLOv8 from {YOLO_PATH}")
        except Exception as e:
            print(f"[Controller] Warning: Could not load YOLOv8: {e}")

        # Latest shared prediction state from MediaPipe callback
        self._latest_gesture_data = GestureResultData()
        self._gesture_lock = threading.Lock()

        # Rolling window for prediction smoothing
        self.smoothing_enabled = True
        self._recent_predictions = []
        self._is_running = False

    def set_gui(self, gui):
        """Sets the GUI reference."""
        self.gui = gui

    def switch_model(self, model_key: str):
        """Switches active model between MediaPipe and YOLOv8."""
        with self.model_lock:
            self.active_model = model_key
            self._recent_predictions.clear()

        if self.gui:
            self.gui.apply_model_ui(model_key)
            if model_key == "mediapipe":
                self.gui.set_status("Active Model: Google MediaPipe Gesture Recognizer", COLOR_SUCCESS, COLOR_SUCCESS)
            else:
                self.gui.set_status("Active Model: Ultralytics YOLOv8 Nano (yolov8n.pt)", COLOR_SUCCESS, COLOR_SUCCESS)

    def set_smoothing_enabled(self, enabled: bool):
        """Toggles rolling window prediction smoothing."""
        self.smoothing_enabled = enabled
        self._recent_predictions.clear()

    def initialize_model_async(self):
        """Loads MediaPipe model bundle on background thread at startup."""
        self.gui.set_status("Loading models (MediaPipe & YOLOv8)...", COLOR_WARNING, COLOR_WARNING)

        def _worker():
            try:
                self.model_service.load_recognizer(result_callback=self._on_mediapipe_result)
                self.root.after(0, self._on_model_loaded_success)
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: self._on_model_loaded_failed(err_msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_loaded_success(self):
        """Called on main thread when model loads successfully."""
        self.gui.set_status("Models loaded successfully. Camera Ready.", COLOR_SUCCESS, COLOR_SUCCESS)
        self.gui.btn_start.config(state="normal")

    def _on_model_loaded_failed(self, error_msg):
        """Called on main thread when model fails to load."""
        self.gui.set_status(f"Model Error: {error_msg}", COLOR_DANGER, COLOR_DANGER)

    def start_recognition(self):
        """Starts video capture and live recognition loop."""
        if self._is_running:
            return

        if not self.model_service.is_loaded:
            self.gui.set_status("Cannot start: Model is not loaded.", COLOR_DANGER, COLOR_DANGER)
            return

        self.gui.set_status("Opening camera...", COLOR_WARNING, COLOR_WARNING)
        success = self.camera_service.start()
        if not success:
            self.gui.set_status("Camera unavailable", COLOR_DANGER, COLOR_DANGER)
            return

        self._is_running = True
        self.gui.set_controls_running(True)
        self.gui.set_status("Recognition running", COLOR_SUCCESS, COLOR_SUCCESS)

        # Start GUI polling loop
        self.root.after(15, self._poll_display_queue)

    def stop_recognition(self):
        """Stops camera capture and clears displays."""
        self._is_running = False
        self.camera_service.stop()
        self.gui.set_controls_running(False)
        self.gui.set_status("Recognition stopped. Camera Ready.", COLOR_WARNING, COLOR_WARNING)
        self.gui.reset_view()

        with self._gesture_lock:
            self._latest_gesture_data = GestureResultData()
        self._recent_predictions.clear()

    def _on_camera_frame(self, frame_bgr, rgb_frame, timestamp_ms, fps):
        """Callback invoked by CameraService when a new frame is captured."""
        if not self._is_running:
            return

        with self.model_lock:
            curr_model = self.active_model

        if curr_model == "mediapipe":
            # 1. Asynchronously submit RGB frame to MediaPipe GestureRecognizer
            try:
                self.model_service.recognize_async(rgb_frame, timestamp_ms)
            except Exception:
                pass

            # 2. Get the latest available prediction
            with self._gesture_lock:
                current_gesture = self._latest_gesture_data

            # 3. Apply smoothing if enabled
            payload = self._apply_smoothing(current_gesture)

        elif curr_model == "yolov8" and self.yolo_model is not None:
            # Run YOLOv8 inference directly on BGR frame
            detected_items = []
            try:
                results = self.yolo_model(frame_bgr, conf=0.35, verbose=False)[0]
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        name = self.yolo_model.names.get(cls_id, str(cls_id))
                        detected_items.append({
                            "name": name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2)
                        })
            except Exception as e:
                print(f"[Controller YOLO Error] {e}")

            payload = {
                "detected_objects": detected_items,
                "total_objects": len(detected_items),
                "top_class": detected_items[0]["name"] if detected_items else "None detected",
                "top_confidence": detected_items[0]["confidence"] if detected_items else 0.0,
            }
        else:
            payload = None

        # 4. Push to display queue (drop oldest if full)
        try:
            self.display_queue.put_nowait((frame_bgr, payload, fps, curr_model))
        except queue.Full:
            try:
                _ = self.display_queue.get_nowait()
                self.display_queue.put_nowait((frame_bgr, payload, fps, curr_model))
            except Exception:
                pass

    def _on_mediapipe_result(self, parsed_data: GestureResultData, timestamp_ms: int):
        """Callback invoked by MediaPipe LIVE_STREAM mode when inference finishes."""
        with self._gesture_lock:
            self._latest_gesture_data = parsed_data

    def _apply_smoothing(self, current_data: GestureResultData) -> GestureResultData:
        """Applies rolling window majority voting to prevent flickering predictions."""
        if not self.smoothing_enabled or not current_data.has_hand:
            self._recent_predictions.clear()
            return current_data

        self._recent_predictions.append(current_data)
        if len(self._recent_predictions) > SMOOTHING_WINDOW_SIZE:
            self._recent_predictions.pop(0)

        # Majority vote on gesture_name
        names = [d.gesture_name for d in self._recent_predictions]
        most_common_name, _ = Counter(names).most_common(1)[0]

        for item in reversed(self._recent_predictions):
            if item.gesture_name == most_common_name:
                return item

        return current_data

    def _poll_display_queue(self):
        """Polls display queue and renders latest frame in Tkinter GUI."""
        if not self._is_running:
            return

        try:
            while True:
                frame_bgr, payload, fps, model_type = self.display_queue.get_nowait()
                self.gui.update_frame(frame_bgr, payload, fps, model_type)
        except queue.Empty:
            pass

        self.root.after(16, self._poll_display_queue)

    def _on_camera_error(self, error_message):
        """Invoked when camera fails or disconnects."""
        self.root.after(0, lambda: self.gui.set_status(f"Camera Error: {error_message}", COLOR_DANGER, COLOR_DANGER))
        self.root.after(0, self.stop_recognition)

    def on_exit(self):
        """Clean shutdown handler releasing all resources."""
        self._is_running = False
        try:
            self.camera_service.stop()
        except Exception:
            pass
        try:
            self.model_service.close()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass
        sys.exit(0)
