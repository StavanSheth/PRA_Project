"""
Integration test for MediaPipe Gesture Recognizer model download, loading, and async inference.
"""

import os
import time
import unittest
import numpy as np
from app.config import MODEL_PATH
from app.model_service import ModelService


class TestModelLoading(unittest.TestCase):

    def setUp(self):
        self.service = ModelService(model_path=MODEL_PATH)

    def tearDown(self):
        if self.service:
            self.service.close()

    def test_01_ensure_model_file(self):
        """Verifies that the model file is downloaded or exists and has valid size."""
        self.assertTrue(self.service.ensure_model_file())
        self.assertTrue(os.path.exists(MODEL_PATH))
        file_size = os.path.getsize(MODEL_PATH)
        # Official gesture_recognizer.task is ~5.6 MB (well over 1MB)
        self.assertGreater(file_size, 1_000_000, f"Model file is unexpectedly small: {file_size} bytes")

    def test_02_load_recognizer(self):
        """Verifies that MediaPipe can instantiate the recognizer in LIVE_STREAM mode."""
        callback_fired = []

        def dummy_callback(result_data, timestamp_ms):
            callback_fired.append((result_data, timestamp_ms))

        loaded = self.service.load_recognizer(result_callback=dummy_callback)
        self.assertTrue(loaded)
        self.assertTrue(self.service.is_loaded)
        self.assertIsNotNone(self.service.recognizer)

        # Run inference on a blank synthetic image
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts_ms = int(time.time() * 1000)
        success = self.service.recognize_async(synthetic_frame, ts_ms)
        self.assertTrue(success)

        # Give async callback a moment to process
        time.sleep(0.3)
        self.assertGreaterEqual(len(callback_fired), 1)
        res_data, returned_ts = callback_fired[0]
        self.assertFalse(res_data.has_hand)
        self.assertEqual(res_data.gesture_name, "No hand detected")
        self.assertEqual(returned_ts, ts_ms)


if __name__ == "__main__":
    unittest.main()
