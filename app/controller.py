"""
Controller orchestrating CameraService, ModelService, and GUI via thread-safe queues.
Ensures non-blocking inference and main-thread Tkinter updates.
"""

import queue
import sys
import threading
from collections import Counter
from app.config import SMOOTHING_WINDOW_SIZE, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING
from app.model_service import ModelService, GestureResultData
from app.camera_service import CameraService


class GestureAppController:
    """Coordinates lifecycle, thread-safe communication, and prediction smoothing."""

    def __init__(self, root):
        self.root = root
        self.gui = None  # Injected after GUI instantiation

        # Thread-safe queue for delivering frames & predictions to GUI main thread
        self.display_queue = queue.Queue(maxsize=2)

        # Services
        self.model_service = ModelService()
        self.camera_service = CameraService(
            on_frame_callback=self._on_camera_frame,
            on_error_callback=self._on_camera_error,
        )

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

    def set_smoothing_enabled(self, enabled: bool):
        """Toggles rolling window prediction smoothing."""
        self.smoothing_enabled = enabled
        self._recent_predictions.clear()

    def initialize_model_async(self):
        """Loads MediaPipe model bundle on background thread at startup."""
        self.gui.set_status("Loading MediaPipe Gesture Recognizer model...", COLOR_WARNING, COLOR_WARNING)

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
        self.gui.set_status("Model loaded successfully. Camera Ready.", COLOR_SUCCESS, COLOR_SUCCESS)
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

        # 1. Asynchronously submit RGB frame to MediaPipe GestureRecognizer
        self.model_service.recognize_async(rgb_frame, timestamp_ms)

        # 2. Get the latest available prediction
        with self._gesture_lock:
            current_gesture = self._latest_gesture_data

        # 3. Apply smoothing if enabled
        smoothed_gesture = self._apply_smoothing(current_gesture)

        # 4. Push to display queue (drop oldest if queue is full to avoid latency)
        try:
            self.display_queue.put_nowait((frame_bgr, smoothed_gesture, fps))
        except queue.Full:
            try:
                _ = self.display_queue.get_nowait()
                self.display_queue.put_nowait((frame_bgr, smoothed_gesture, fps))
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

        # Use the latest item with that name for display
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
                frame_bgr, gesture_data, fps = self.display_queue.get_nowait()
                self.gui.update_frame(frame_bgr, gesture_data, fps)
        except queue.Empty:
            pass

        # Schedule next refresh cycle (~60 Hz GUI loop)
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
