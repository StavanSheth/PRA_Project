"""
Threaded Camera Service for smooth webcam acquisition, BGR-to-RGB conversion, and FPS calculation.
"""

import threading
import time
import cv2
from app.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraService:
    """Manages thread-safe webcam capture with monotonic timestamps and FPS estimation."""

    def __init__(self, camera_index=CAMERA_INDEX, on_frame_callback=None, on_error_callback=None):
        self.camera_index = camera_index
        self.on_frame_callback = on_frame_callback
        self.on_error_callback = on_error_callback

        self.cap = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

        # FPS Tracking
        self.fps = 0.0
        self._prev_frame_time = 0.0
        self._fps_history = []

        # Monotonic timestamp generator (in milliseconds)
        self._last_timestamp_ms = 0
        self._start_time_ns = 0

    def start(self):
        """Starts video capture in a dedicated background daemon thread."""
        with self.lock:
            if self.is_running:
                return True

            # Try DirectShow first on Windows, then default
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.cap = None
                if self.on_error_callback:
                    self.on_error_callback("Webcam device could not be opened. Check connection and permissions.")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            self.is_running = True
            self._start_time_ns = time.time_ns()
            self._last_timestamp_ms = 0

            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
            return True

    def _capture_worker(self):
        """Continuous camera capture loop running in background thread."""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror horizontally for natural webcam interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe and Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate strictly increasing timestamp in milliseconds
            now_ms = int((time.time_ns() - self._start_time_ns) / 1_000_000)
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = now_ms

            # Update FPS calculation
            self._compute_fps()

            # Forward to callback
            if self.on_frame_callback and self.is_running:
                try:
                    self.on_frame_callback(frame, rgb_frame, now_ms, self.fps)
                except Exception as e:
                    print(f"[CameraService] Callback error: {e}")

    def _compute_fps(self):
        """Computes smooth moving-average FPS."""
        now = time.time()
        if self._prev_frame_time > 0:
            delta = now - self._prev_frame_time
            if delta > 0:
                current_fps = 1.0 / delta
                self._fps_history.append(current_fps)
                if len(self._fps_history) > 15:
                    self._fps_history.pop(0)
                self.fps = sum(self._fps_history) / len(self._fps_history)
        self._prev_frame_time = now

    def stop(self):
        """Safely stops capture loop and releases camera hardware."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.6)
        self.thread = None

        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        self.fps = 0.0
