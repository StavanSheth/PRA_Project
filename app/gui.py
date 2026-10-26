"""
Desktop GUI interface implemented in native Tkinter.
Renders live camera feed with hand landmarks, bounding boxes, prediction cards, controls, and status bar.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from app.config import (
    COLOR_BG_DARK,
    COLOR_PANEL_BG,
    COLOR_CARD_BORDER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    COLOR_ACCENT,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

# MediaPipe Hand Skeletal Connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (9, 10), (10, 11), (11, 12),              # Middle finger
    (13, 14), (14, 15), (15, 16),             # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky finger
    (5, 9), (9, 13), (13, 17),                # Palm base
]


class GestureAppGUI:
    """Manages the Tkinter user interface widgets, video rendering, and status updates."""

    def __init__(self, root, controller):
        self.root = root
        self.controller = controller

        self.root.title("MediaPipe Hand Gesture Recognition - Live Studio")
        self.root.geometry("1100x720")
        self.root.minsize(980, 640)
        self.root.configure(bg=COLOR_BG_DARK)

        self.image_tk = None
        self.smoothing_var = tk.BooleanVar(value=True)

        self._setup_styles()
        self._build_ui()

        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.controller.on_exit)

    def _setup_styles(self):
        """Configures modern TTK styles."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 11, "bold"),
            background=COLOR_PRIMARY,
            foreground="#ffffff",
            borderwidth=0,
            padding=9,
        )
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#334155")])

        self.style.configure(
            "Stop.TButton",
            font=("Segoe UI", 11, "bold"),
            background=COLOR_WARNING,
            foreground="#0f172a",
            borderwidth=0,
            padding=9,
        )
        self.style.map("Stop.TButton", background=[("active", "#d97706"), ("disabled", "#334155")])

        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 11, "bold"),
            background=COLOR_DANGER,
            foreground="#ffffff",
            borderwidth=0,
            padding=9,
        )
        self.style.map("Danger.TButton", background=[("active", "#b91c1c")])

    def _build_ui(self):
        """Builds all graphical elements, layout containers, and bindings."""
        # Top Header Bar
        header = tk.Frame(self.root, bg=COLOR_PANEL_BG, height=52, padx=20, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="🤟 Live Hand Gesture Recognition",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL_BG,
        )
        title_lbl.pack(side=tk.LEFT)

        tag_lbl = tk.Label(
            header,
            text="MediaPipe Tasks Vision • Live Stream",
            font=("Segoe UI", 10),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            padx=14,
        )
        tag_lbl.pack(side=tk.LEFT)

        # Main Workspace Container
        content = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=16, pady=16)
        content.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Preview
        video_col = tk.Frame(content, bg=COLOR_PANEL_BG, bd=1, relief=tk.SOLID)
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas = tk.Label(
            video_col,
            text="Camera is Stopped\n\nClick '▶ Start Camera' to begin real-time gesture recognition",
            font=("Segoe UI", 13),
            fg=COLOR_TEXT_MUTED,
            bg="#020617",
            width=640,
            height=480,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Right Column: Prediction & Control Panel
        panel_col = tk.Frame(content, bg=COLOR_PANEL_BG, width=380, padx=16, pady=14)
        panel_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        panel_col.pack_propagate(False)

        # 1. Prediction Card
        pred_box = tk.LabelFrame(
            panel_col,
            text=" RECOGNIZED GESTURE ",
            font=("Segoe UI", 11, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_PANEL_BG,
            padx=14,
            pady=12,
        )
        pred_box.pack(fill=tk.X, pady=(0, 12))

        self.gesture_display = tk.Label(
            pred_box,
            text="No hand detected",
            font=("Segoe UI", 20, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg="#020617",
            height=2,
            relief=tk.FLAT,
            wraplength=320,
        )
        self.gesture_display.pack(fill=tk.X, pady=(4, 10))

        # Confidence Bar
        conf_row = tk.Frame(pred_box, bg=COLOR_PANEL_BG)
        conf_row.pack(fill=tk.X, pady=(2, 4))
        tk.Label(conf_row, text="Confidence:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.conf_lbl = tk.Label(conf_row, text="0.0 %", font=("Segoe UI", 11, "bold"), fg=COLOR_SUCCESS, bg=COLOR_PANEL_BG)
        self.conf_lbl.pack(side=tk.RIGHT)

        self.conf_bar = ttk.Progressbar(pred_box, orient=tk.HORIZONTAL, length=200, mode="determinate")
        self.conf_bar.pack(fill=tk.X, pady=(2, 6))

        # 2. Hand Properties Card (Handedness, count, status)
        props_box = tk.LabelFrame(
            panel_col,
            text=" DETECTION DETAILS ",
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            padx=14,
            pady=10,
        )
        props_box.pack(fill=tk.X, pady=(0, 12))

        # Handedness row
        row_hand = tk.Frame(props_box, bg=COLOR_PANEL_BG)
        row_hand.pack(fill=tk.X, pady=3)
        tk.Label(row_hand, text="Handedness:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.handedness_lbl = tk.Label(row_hand, text="None", font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_MAIN, bg=COLOR_PANEL_BG)
        self.handedness_lbl.pack(side=tk.RIGHT)

        # Hands count row
        row_count = tk.Frame(props_box, bg=COLOR_PANEL_BG)
        row_count.pack(fill=tk.X, pady=3)
        tk.Label(row_count, text="Detected Hands:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.hands_count_lbl = tk.Label(row_count, text="0", font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_MAIN, bg=COLOR_PANEL_BG)
        self.hands_count_lbl.pack(side=tk.RIGHT)

        # Inference State row
        row_state = tk.Frame(props_box, bg=COLOR_PANEL_BG)
        row_state.pack(fill=tk.X, pady=3)
        tk.Label(row_state, text="Inference State:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.state_lbl = tk.Label(row_state, text="Idle", font=("Segoe UI", 10, "bold"), fg=COLOR_WARNING, bg=COLOR_PANEL_BG)
        self.state_lbl.pack(side=tk.RIGHT)

        # Smoothing Toggle
        opt_box = tk.Frame(panel_col, bg=COLOR_PANEL_BG)
        opt_box.pack(fill=tk.X, pady=(0, 10))
        self.chk_smoothing = tk.Checkbutton(
            opt_box,
            text="Enable Prediction Smoothing (5-frame window)",
            variable=self.smoothing_var,
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            activebackground=COLOR_PANEL_BG,
            activeforeground=COLOR_TEXT_MAIN,
            selectcolor="#020617",
            command=self._on_smoothing_toggled,
        )
        self.chk_smoothing.pack(anchor=tk.W)

        # 3. Controls Card
        ctrl_box = tk.Frame(panel_col, bg=COLOR_PANEL_BG)
        ctrl_box.pack(fill=tk.X, pady=(8, 0))

        self.btn_start = ttk.Button(
            ctrl_box,
            text="▶ Start Camera",
            style="Primary.TButton",
            command=self.controller.start_recognition,
        )
        self.btn_start.pack(fill=tk.X, pady=4)

        self.btn_stop = ttk.Button(
            ctrl_box,
            text="⏹ Stop Camera",
            style="Stop.TButton",
            state=tk.DISABLED,
            command=self.controller.stop_recognition,
        )
        self.btn_stop.pack(fill=tk.X, pady=4)

        self.btn_exit = ttk.Button(
            ctrl_box,
            text="✕ Exit Application",
            style="Danger.TButton",
            command=self.controller.on_exit,
        )
        self.btn_exit.pack(fill=tk.X, pady=4)

        # Bottom Status Bar
        self.status_bar = tk.Frame(self.root, bg="#020617", height=28, padx=16, pady=4)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_dot = tk.Label(self.status_bar, text="●", font=("Segoe UI", 11), fg=COLOR_WARNING, bg="#020617")
        self.status_dot.pack(side=tk.LEFT, padx=(0, 6))

        self.status_text = tk.Label(
            self.status_bar,
            text="Status: Initializing...",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_MUTED,
            bg="#020617",
        )
        self.status_text.pack(side=tk.LEFT)

        self.fps_lbl = tk.Label(
            self.status_bar,
            text="FPS: 0.0",
            font=("Segoe UI", 9, "bold"),
            fg=COLOR_ACCENT,
            bg="#020617",
        )
        self.fps_lbl.pack(side=tk.RIGHT)

    def _on_smoothing_toggled(self):
        """Notifies controller when smoothing is toggled."""
        self.controller.set_smoothing_enabled(self.smoothing_var.get())

    def set_status(self, text, color=COLOR_TEXT_MUTED, dot_color=COLOR_SUCCESS):
        """Updates the status bar message and dot color."""
        self.status_text.config(text=f"Status: {text}", fg=color)
        self.status_dot.config(fg=dot_color)

    def set_controls_running(self, is_running):
        """Toggles Start/Stop button states to prevent multiple camera instances."""
        if is_running:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.state_lbl.config(text="Running", fg=COLOR_SUCCESS)
        else:
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.state_lbl.config(text="Stopped", fg=COLOR_WARNING)

    def update_frame(self, frame_bgr, gesture_data, fps=0.0):
        """
        Renders the video frame with overlays (skeletal landmarks, bounding boxes, labels)
        and updates the prediction panel. Must be called on main GUI thread.
        """
        h, w = frame_bgr.shape[:2]
        canvas_img = frame_bgr.copy()

        # 1. Draw hand landmarks and bounding boxes
        if gesture_data and gesture_data.has_hand:
            # Draw bounding boxes
            for (bx1, by1, bx2, by2) in gesture_data.bounding_boxes:
                px1 = max(0, int(bx1 * w) - 10)
                py1 = max(0, int(by1 * h) - 10)
                px2 = min(w, int(bx2 * w) + 10)
                py2 = min(h, int(by2 * h) + 10)
                cv2.rectangle(canvas_img, (px1, py1), (px2, py2), (16, 185, 129), 2)

            # Draw skeletal lines & joint points
            for hand_pts in gesture_data.landmarks:
                # Connections
                for (idx1, idx2) in HAND_CONNECTIONS:
                    if idx1 < len(hand_pts) and idx2 < len(hand_pts):
                        p1 = (int(hand_pts[idx1][0] * w), int(hand_pts[idx1][1] * h))
                        p2 = (int(hand_pts[idx2][0] * w), int(hand_pts[idx2][1] * h))
                        cv2.line(canvas_img, p1, p2, (56, 189, 248), 2, cv2.LINE_AA)

                # Joints
                for pt in hand_pts:
                    px = int(pt[0] * w)
                    py = int(pt[1] * h)
                    cv2.circle(canvas_img, (px, py), 4, (245, 158, 11), -1, cv2.LINE_AA)

            # Overlay gesture banner on top of video frame
            banner_text = f"{gesture_data.display_name} ({gesture_data.confidence*100:.1f}%)"
            cv2.rectangle(canvas_img, (10, 10), (min(w - 10, 360), 50), (15, 23, 42), -1)
            cv2.rectangle(canvas_img, (10, 10), (min(w - 10, 360), 50), (56, 189, 248), 1)
            cv2.putText(
                canvas_img,
                banner_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (248, 250, 252),
                2,
                cv2.LINE_AA,
            )

        # 2. Convert to ImageTk and update canvas
        rgb_disp = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_disp)
        self.image_tk = ImageTk.PhotoImage(image=pil_img)
        self.video_canvas.config(image=self.image_tk, text="")

        # 3. Update prediction panel
        if gesture_data:
            if not gesture_data.has_hand:
                self.gesture_display.config(text="No hand detected", fg=COLOR_TEXT_MUTED)
                self.conf_lbl.config(text="0.0 %", fg=COLOR_TEXT_MUTED)
                self.conf_bar["value"] = 0
                self.handedness_lbl.config(text="None")
                self.hands_count_lbl.config(text="0")
            elif gesture_data.gesture_name == "Unknown gesture":
                self.gesture_display.config(text="Unknown gesture", fg=COLOR_WARNING)
                conf_pct = gesture_data.confidence * 100
                self.conf_lbl.config(text=f"{conf_pct:.1f} %", fg=COLOR_WARNING)
                self.conf_bar["value"] = conf_pct
                self.handedness_lbl.config(text=gesture_data.handedness)
                self.hands_count_lbl.config(text=str(gesture_data.num_hands))
            else:
                self.gesture_display.config(text=gesture_data.display_name, fg=COLOR_ACCENT)
                conf_pct = gesture_data.confidence * 100
                self.conf_lbl.config(text=f"{conf_pct:.1f} %", fg=COLOR_SUCCESS)
                self.conf_bar["value"] = conf_pct
                self.handedness_lbl.config(text=gesture_data.handedness)
                self.hands_count_lbl.config(text=str(gesture_data.num_hands))

        # 4. Update FPS counter
        self.fps_lbl.config(text=f"FPS: {fps:.1f}")

    def reset_view(self):
        """Clears video display and resets prediction panel to idle state."""
        self.video_canvas.config(
            image="",
            text="Camera is Stopped\n\nClick '▶ Start Camera' to resume recognition",
        )
        self.gesture_display.config(text="No hand detected", fg=COLOR_TEXT_MUTED)
        self.conf_lbl.config(text="0.0 %", fg=COLOR_TEXT_MUTED)
        self.conf_bar["value"] = 0
        self.handedness_lbl.config(text="None")
        self.hands_count_lbl.config(text="0")
        self.fps_lbl.config(text="FPS: 0.0")
        self.image_tk = None
