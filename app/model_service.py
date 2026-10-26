"""
Model service for Google MediaPipe Tasks Gesture Recognizer.
Handles automatic model downloading, lifecycle, async LIVE_STREAM inference, and result parsing.
"""

import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from app.config import (
    MODEL_DIR,
    MODEL_PATH,
    MODEL_URL,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    NUM_HANDS,
    GESTURE_DISPLAY_NAMES,
)


class GestureResultData:
    """Encapsulates parsed gesture recognition output for UI rendering."""

    def __init__(
        self,
        has_hand=False,
        gesture_name="No hand detected",
        display_name="No hand detected",
        confidence=0.0,
        handedness="None",
        num_hands=0,
        landmarks=None,
        bounding_boxes=None,
    ):
        self.has_hand = has_hand
        self.gesture_name = gesture_name
        self.display_name = display_name
        self.confidence = confidence
        self.handedness = handedness
        self.num_hands = num_hands
        self.landmarks = landmarks or []
        self.bounding_boxes = bounding_boxes or []


class ModelService:
    """Manages the lifecycle and asynchronous execution of MediaPipe GestureRecognizer."""

    def __init__(self, model_path=MODEL_PATH, result_callback=None):
        self.model_path = model_path
        self.user_callback = result_callback
        self.recognizer = None
        self.is_loaded = False

    def ensure_model_file(self):
        """Checks if the .task model bundle exists; downloads it from official source if missing."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 100_000:
            return True

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"[ModelService] Model missing at {self.model_path}. Downloading from official source...")
        try:
            # Download model bundle
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 100_000:
                raise RuntimeError("Downloaded model bundle file is invalid or incomplete.")
            print(f"[ModelService] Model successfully downloaded ({os.path.getsize(self.model_path)} bytes).")
            return True
        except Exception as e:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            raise RuntimeError(f"Failed to download MediaPipe gesture model: {e}")

    def load_recognizer(self, result_callback=None):
        """Initializes the MediaPipe GestureRecognizer in LIVE_STREAM mode."""
        if result_callback:
            self.user_callback = result_callback

        self.ensure_model_file()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            result_callback=self._internal_callback,
        )

        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.is_loaded = True
        print("[ModelService] MediaPipe GestureRecognizer loaded successfully.")
        return True

    def _internal_callback(self, result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        """Processes raw MediaPipe result and forwards parsed data to user callback."""
        parsed_data = self.parse_result(result, output_image)
        if self.user_callback:
            self.user_callback(parsed_data, timestamp_ms)

    @staticmethod
    def parse_result(result: vision.GestureRecognizerResult, output_image=None) -> GestureResultData:
        """
        Parses raw GestureRecognizerResult into structured GestureResultData:
        - len(gestures) == 0 -> No hand detected
        - gesture == 'None' -> Unknown gesture
        - otherwise -> recognized gesture name and confidence
        """
        if not result or not result.gestures or len(result.gestures) == 0:
            return GestureResultData(
                has_hand=False,
                gesture_name="No hand detected",
                display_name="No hand detected",
                confidence=0.0,
                handedness="None",
                num_hands=0,
                landmarks=[],
                bounding_boxes=[],
            )

        num_hands = len(result.gestures)
        primary_gestures = result.gestures[0]

        if not primary_gestures or len(primary_gestures) == 0:
            gesture_name = "Unknown gesture"
            display_name = "Unknown gesture"
            confidence = 0.0
        else:
            top_category = primary_gestures[0]
            cat_name = top_category.category_name

            if not cat_name or cat_name == "None":
                gesture_name = "Unknown gesture"
                display_name = "Unknown gesture"
                confidence = float(top_category.score) if top_category.score else 0.0
            else:
                gesture_name = cat_name
                display_name = GESTURE_DISPLAY_NAMES.get(cat_name, cat_name)
                confidence = float(top_category.score)

        # Parse handedness
        handedness_str = "Unknown"
        if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
            handedness_str = result.handedness[0][0].category_name or "Unknown"

        # Parse landmarks and compute normalized bounding boxes
        landmarks_list = []
        bounding_boxes = []

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                pts = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                landmarks_list.append(pts)

                # Compute hand bounding rectangle
                xs = [lm.x for lm in hand_lms]
                ys = [lm.y for lm in hand_lms]
                if xs and ys:
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    bounding_boxes.append(bbox)

        return GestureResultData(
            has_hand=True,
            gesture_name=gesture_name,
            display_name=display_name,
            confidence=confidence,
            handedness=handedness_str,
            num_hands=num_hands,
            landmarks=landmarks_list,
            bounding_boxes=bounding_boxes,
        )

    def recognize_async(self, rgb_frame, timestamp_ms: int):
        """Submits an RGB frame array to MediaPipe asynchronously."""
        if not self.is_loaded or self.recognizer is None:
            return False

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.recognizer.recognize_async(mp_image, timestamp_ms)
        return True

    def close(self):
        """Safely closes MediaPipe GestureRecognizer."""
        if self.recognizer is not None:
            try:
                self.recognizer.close()
            except Exception:
                pass
            self.recognizer = None
        self.is_loaded = False
