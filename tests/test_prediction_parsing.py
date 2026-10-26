"""
Unit tests for GestureRecognizer result parsing logic.
Tests edge cases including no hand detected, unknown gestures, and valid gestures.
"""

import unittest
from types import SimpleNamespace
from app.model_service import ModelService, GestureResultData


class MockCategory:
    def __init__(self, category_name, score):
        self.category_name = category_name
        self.score = score


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class TestPredictionParsing(unittest.TestCase):

    def test_none_or_empty_result(self):
        # When result is None
        res_none = ModelService.parse_result(None)
        self.assertFalse(res_none.has_hand)
        self.assertEqual(res_none.gesture_name, "No hand detected")
        self.assertEqual(res_none.display_name, "No hand detected")
        self.assertEqual(res_none.confidence, 0.0)

        # When gestures list is empty
        mock_empty = SimpleNamespace(gestures=[], handedness=[], hand_landmarks=[])
        res_empty = ModelService.parse_result(mock_empty)
        self.assertFalse(res_empty.has_hand)
        self.assertEqual(res_empty.gesture_name, "No hand detected")
        self.assertEqual(res_empty.display_name, "No hand detected")

    def test_unrecognized_gesture(self):
        # When hand is detected, but MediaPipe returns "None" as the category name
        mock_unrecognized = SimpleNamespace(
            gestures=[[MockCategory("None", 0.65)]],
            handedness=[[MockCategory("Right", 0.95)]],
            hand_landmarks=[[MockLandmark(0.5, 0.5), MockLandmark(0.6, 0.6)]],
        )
        res = ModelService.parse_result(mock_unrecognized)
        self.assertTrue(res.has_hand)
        self.assertEqual(res.gesture_name, "Unknown gesture")
        self.assertEqual(res.display_name, "Unknown gesture")
        self.assertEqual(res.handedness, "Right")
        self.assertAlmostEqual(res.confidence, 0.65)
        self.assertEqual(len(res.bounding_boxes), 1)

    def test_recognized_gesture_open_palm(self):
        mock_palm = SimpleNamespace(
            gestures=[[MockCategory("Open_Palm", 0.92)]],
            handedness=[[MockCategory("Left", 0.98)]],
            hand_landmarks=[
                [
                    MockLandmark(0.1, 0.2),
                    MockLandmark(0.3, 0.4),
                    MockLandmark(0.5, 0.6),
                ]
            ],
        )
        res = ModelService.parse_result(mock_palm)
        self.assertTrue(res.has_hand)
        self.assertEqual(res.gesture_name, "Open_Palm")
        self.assertEqual(res.display_name, "Open Palm ✋")
        self.assertAlmostEqual(res.confidence, 0.92)
        self.assertEqual(res.handedness, "Left")
        self.assertEqual(res.num_hands, 1)

        # Verify bounding box calculation: (min_x, min_y, max_x, max_y)
        self.assertEqual(len(res.bounding_boxes), 1)
        xmin, ymin, xmax, ymax = res.bounding_boxes[0]
        self.assertAlmostEqual(xmin, 0.1)
        self.assertAlmostEqual(ymin, 0.2)
        self.assertAlmostEqual(xmax, 0.5)
        self.assertAlmostEqual(ymax, 0.6)

    def test_recognized_gesture_victory(self):
        mock_victory = SimpleNamespace(
            gestures=[[MockCategory("Victory", 0.88)]],
            handedness=[[MockCategory("Right", 0.99)]],
            hand_landmarks=[[MockLandmark(0.4, 0.4)]],
        )
        res = ModelService.parse_result(mock_victory)
        self.assertTrue(res.has_hand)
        self.assertEqual(res.gesture_name, "Victory")
        self.assertEqual(res.display_name, "Victory ✌️")
        self.assertAlmostEqual(res.confidence, 0.88)
        self.assertEqual(res.handedness, "Right")


if __name__ == "__main__":
    unittest.main()
