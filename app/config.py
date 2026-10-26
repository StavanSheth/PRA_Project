"""
Configuration constants, paths, thresholds, and UI styling tokens.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model storage and official download URL
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "gesture_recognizer.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

# MediaPipe recognizer confidence thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_HANDS = 2

# Camera acquisition parameters
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Smoothing window length
SMOOTHING_WINDOW_SIZE = 5

# Supported official gesture labels mapping
GESTURE_DISPLAY_NAMES = {
    "None": "Unknown Gesture",
    "Closed_Fist": "Closed Fist ✊",
    "Open_Palm": "Open Palm ✋",
    "Pointing_Up": "Pointing Up ☝️",
    "Thumb_Down": "Thumb Down 👎",
    "Thumb_Up": "Thumb Up 👍",
    "Victory": "Victory ✌️",
    "ILoveYou": "I Love You 🤟",
}

# UI Theme Colors (Modern Dark Theme)
COLOR_BG_DARK = "#0f172a"
COLOR_PANEL_BG = "#1e293b"
COLOR_CARD_BORDER = "#334155"
COLOR_PRIMARY = "#2563eb"
COLOR_PRIMARY_HOVER = "#1d4ed8"
COLOR_SUCCESS = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#dc2626"
COLOR_TEXT_MAIN = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_ACCENT = "#38bdf8"
