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
        self.current_model_key = "mediapipe"

        self.root.title("Hand Gesture & Object Detection Studio")
        self.root.geometry("1100x740")
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

        # Style Combobox
        self.style.configure(
            "TCombobox",
            fieldbackground="#0f172a",
            background="#1e293b",
            foreground="#ffffff",
            darkcolor="#334155",
            lightcolor="#334155",
        )

    def _build_ui(self):
        """Builds all graphical elements, layout containers, and bindings."""
        # Top Header Bar
        header = tk.Frame(self.root, bg=COLOR_PANEL_BG, height=52, padx=20, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="🤟 AI Vision Studio",
            font=("Segoe UI", 16, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_PANEL_BG,
        )
        title_lbl.pack(side=tk.LEFT)

        self.tag_lbl = tk.Label(
            header,
            text="MediaPipe Tasks Vision • Live Stream",
            font=("Segoe UI", 10),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            padx=14,
        )
        self.tag_lbl.pack(side=tk.LEFT)

        # Main Workspace Container
        content = tk.Frame(self.root, bg=COLOR_BG_DARK, padx=16, pady=16)
        content.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Preview
        video_col = tk.Frame(content, bg=COLOR_PANEL_BG, bd=1, relief=tk.SOLID)
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas = tk.Label(
            video_col,
            text="Camera is Stopped\n\nClick '▶ Start Camera' to begin real-time recognition",
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

        # 1. Model Selector Card
        model_box = tk.LabelFrame(
            panel_col,
            text=" MODEL SELECTION PIPELINE ",
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_PANEL_BG,
            padx=14,
            pady=10,
        )
        model_box.pack(fill=tk.X, pady=(0, 12))

        self.model_combo_var = tk.StringVar(value="MediaPipe Hand Gestures")
        self.model_combo = ttk.Combobox(
            model_box,
            textvariable=self.model_combo_var,
            values=["MediaPipe Hand Gestures", "YOLOv8 Object Detection"],
            state="readonly",
            font=("Segoe UI", 10, "bold"),
        )
        self.model_combo.pack(fill=tk.X, pady=(2, 4))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_combo_selected)

        # 2. Prediction Card
        self.pred_box = tk.LabelFrame(
            panel_col,
            text=" RECOGNIZED GESTURE ",
            font=("Segoe UI", 11, "bold"),
            fg=COLOR_ACCENT,
            bg=COLOR_PANEL_BG,
            padx=14,
            pady=12,
        )
        self.pred_box.pack(fill=tk.X, pady=(0, 12))

        self.gesture_display = tk.Label(
            self.pred_box,
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
        conf_row = tk.Frame(self.pred_box, bg=COLOR_PANEL_BG)
        conf_row.pack(fill=tk.X, pady=(2, 4))
        tk.Label(conf_row, text="Confidence:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.conf_lbl = tk.Label(conf_row, text="0.0 %", font=("Segoe UI", 11, "bold"), fg=COLOR_SUCCESS, bg=COLOR_PANEL_BG)
        self.conf_lbl.pack(side=tk.RIGHT)

        self.conf_bar = ttk.Progressbar(self.pred_box, orient=tk.HORIZONTAL, length=200, mode="determinate")
        self.conf_bar.pack(fill=tk.X, pady=(2, 6))

        # 3. Detection Properties Card
        self.props_box = tk.LabelFrame(
            panel_col,
            text=" DETECTION DETAILS ",
            font=("Segoe UI", 10, "bold"),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL_BG,
            padx=14,
            pady=10,
        )
        self.props_box.pack(fill=tk.X, pady=(0, 12))

        # Detail Row 1 (Handedness or Total Objects)
        row_prop1 = tk.Frame(self.props_box, bg=COLOR_PANEL_BG)
        row_prop1.pack(fill=tk.X, pady=3)
        self.prop1_title = tk.Label(row_prop1, text="Handedness:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG)
        self.prop1_title.pack(side=tk.LEFT)
        self.prop1_val = tk.Label(row_prop1, text="None", font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_MAIN, bg=COLOR_PANEL_BG)
        self.prop1_val.pack(side=tk.RIGHT)

        # Detail Row 2 (Detected Hands or Top Class)
        row_prop2 = tk.Frame(self.props_box, bg=COLOR_PANEL_BG)
        row_prop2.pack(fill=tk.X, pady=3)
        self.prop2_title = tk.Label(row_prop2, text="Detected Hands:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG)
        self.prop2_title.pack(side=tk.LEFT)
        self.prop2_val = tk.Label(row_prop2, text="0", font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_MAIN, bg=COLOR_PANEL_BG)
        self.prop2_val.pack(side=tk.RIGHT)

        # Inference State row
        row_state = tk.Frame(self.props_box, bg=COLOR_PANEL_BG)
        row_state.pack(fill=tk.X, pady=3)
        tk.Label(row_state, text="Inference State:", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_PANEL_BG).pack(side=tk.LEFT)
        self.state_lbl = tk.Label(row_state, text="Idle", font=("Segoe UI", 10, "bold"), fg=COLOR_WARNING, bg=COLOR_PANEL_BG)
        self.state_lbl.pack(side=tk.RIGHT)

        # Smoothing Toggle
        opt_box = tk.Frame(panel_col, bg=COLOR_PANEL_BG)
        opt_box.pack(fill=tk.X, pady=(0, 10))
        self.chk_smoothing = tk.Checkbutton(
            opt_box,
            text="Enable Prediction Smoothing",
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

        # 4. Controls Card
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

    def _on_model_combo_selected(self, event=None):
        """Invoked when user changes the model selection dropdown."""
        val = self.model_combo_var.get()
        if "MediaPipe" in val:
            self.controller.switch_model("mediapipe")
        elif "YOLO" in val:
            self.controller.switch_model("yolov8")

    def apply_model_ui(self, model_key: str):
        """Updates UI layout, card titles, and labels to reflect active model."""
        self.current_model_key = model_key
        if model_key == "mediapipe":
            self.model_combo_var.set("MediaPipe Hand Gestures")
            self.tag_lbl.config(text="MediaPipe Tasks Vision • Live Stream")
            self.pred_box.config(text=" RECOGNIZED GESTURE ")
            self.prop1_title.config(text="Handedness:")
            self.prop2_title.config(text="Detected Hands:")
            self.gesture_display.config(text="No hand detected", fg=COLOR_TEXT_MUTED)
            self.prop1_val.config(text="None")
            self.prop2_val.config(text="0")
            self.chk_smoothing.config(state=tk.NORMAL)
        elif model_key == "yolov8":
            self.model_combo_var.set("YOLOv8 Object Detection")
            self.tag_lbl.config(text="Ultralytics YOLOv8 • Live Stream")
            self.pred_box.config(text=" TOP DETECTED OBJECT ")
            self.prop1_title.config(text="Total Objects:")
            self.prop2_title.config(text="Top Class:")
            self.gesture_display.config(text="No object detected", fg=COLOR_TEXT_MUTED)
            self.prop1_val.config(text="0")
            self.prop2_val.config(text="None")
            self.chk_smoothing.config(state=tk.DISABLED)

        self.conf_lbl.config(text="0.0 %", fg=COLOR_TEXT_MUTED)
        self.conf_bar["value"] = 0

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

    def update_frame(self, frame_bgr, payload, fps=0.0, model_type="mediapipe"):
        """
        Renders the video frame with overlays (landmarks, bounding boxes, labels)
        and updates the prediction panel. Must be called on main GUI thread.
        """
        h, w = frame_bgr.shape[:2]
        canvas_img = frame_bgr.copy()

        if model_type == "mediapipe":
            gesture_data = payload
            if gesture_data and getattr(gesture_data, "has_hand", False):
                # Draw bounding boxes
                for (bx1, by1, bx2, by2) in gesture_data.bounding_boxes:
                    px1 = max(0, int(bx1 * w) - 10)
                    py1 = max(0, int(by1 * h) - 10)
                    px2 = min(w, int(bx2 * w) + 10)
                    py2 = min(h, int(by2 * h) + 10)
                    cv2.rectangle(canvas_img, (px1, py1), (px2, py2), (16, 185, 129), 2)

                # Draw skeletal lines & joint points
                for hand_pts in gesture_data.landmarks:
                    for (idx1, idx2) in HAND_CONNECTIONS:
                        if idx1 < len(hand_pts) and idx2 < len(hand_pts):
                            p1 = (int(hand_pts[idx1][0] * w), int(hand_pts[idx1][1] * h))
                            p2 = (int(hand_pts[idx2][0] * w), int(hand_pts[idx2][1] * h))
                            cv2.line(canvas_img, p1, p2, (56, 189, 248), 2, cv2.LINE_AA)

                    for pt in hand_pts:
                        px = int(pt[0] * w)
                        py = int(pt[1] * h)
                        cv2.circle(canvas_img, (px, py), 4, (245, 158, 11), -1, cv2.LINE_AA)

                # Overlay gesture banner
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

            # Update prediction panel for mediapipe
            if gesture_data:
                if not getattr(gesture_data, "has_hand", False):
                    self.gesture_display.config(text="No hand detected", fg=COLOR_TEXT_MUTED)
                    self.conf_lbl.config(text="0.0 %", fg=COLOR_TEXT_MUTED)
                    self.conf_bar["value"] = 0
                    self.prop1_val.config(text="None")
                    self.prop2_val.config(text="0")
                elif gesture_data.gesture_name == "Unknown gesture":
                    self.gesture_display.config(text="Unknown gesture", fg=COLOR_WARNING)
                    conf_pct = gesture_data.confidence * 100
                    self.conf_lbl.config(text=f"{conf_pct:.1f} %", fg=COLOR_WARNING)
                    self.conf_bar["value"] = conf_pct
                    self.prop1_val.config(text=gesture_data.handedness)
                    self.prop2_val.config(text=str(gesture_data.num_hands))
                else:
                    self.gesture_display.config(text=gesture_data.display_name, fg=COLOR_ACCENT)
                    conf_pct = gesture_data.confidence * 100
                    self.conf_lbl.config(text=f"{conf_pct:.1f} %", fg=COLOR_SUCCESS)
                    self.conf_bar["value"] = conf_pct
                    self.prop1_val.config(text=gesture_data.handedness)
                    self.prop2_val.config(text=str(gesture_data.num_hands))

        elif model_type == "yolov8" and isinstance(payload, dict):
            # Render YOLOv8 bounding boxes & labels
            detected_items = payload.get("detected_objects", [])
            for item in detected_items:
                bx1, by1, bx2, by2 = item["bbox"]
                cls_name = item["name"]
                conf_val = item["confidence"]

                # Box
                cv2.rectangle(canvas_img, (bx1, by1), (bx2, by2), (236, 72, 153), 2)

                # Badge label
                lbl_text = f"{cls_name} {conf_val*100:.0f}%"
                (tw, th), baseline = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                py1 = max(0, by1 - th - 6)
                cv2.rectangle(canvas_img, (bx1, py1), (bx1 + tw + 8, py1 + th + 6), (236, 72, 153), -1)
                cv2.putText(
                    canvas_img,
                    lbl_text,
                    (bx1 + 4, py1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            tot = payload.get("total_objects", 0)
            top_cls = payload.get("top_class", "None detected")
            top_conf = payload.get("top_confidence", 0.0)

            # Top overlay banner
            banner_text = f"YOLOv8: {top_cls} ({top_conf*100:.1f}%) | {tot} detected"
            cv2.rectangle(canvas_img, (10, 10), (min(w - 10, 420), 50), (15, 23, 42), -1)
            cv2.rectangle(canvas_img, (10, 10), (min(w - 10, 420), 50), (236, 72, 153), 1)
            cv2.putText(
                canvas_img,
                banner_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (248, 250, 252),
                2,
                cv2.LINE_AA,
            )

            # Update prediction panel for YOLO
            if tot == 0 or top_cls == "None detected":
                self.gesture_display.config(text="No object detected", fg=COLOR_TEXT_MUTED)
                self.conf_lbl.config(text="0.0 %", fg=COLOR_TEXT_MUTED)
                self.conf_bar["value"] = 0
                self.prop1_val.config(text="0")
                self.prop2_val.config(text="None")
            else:
                self.gesture_display.config(text=top_cls.capitalize(), fg=COLOR_ACCENT)
                conf_pct = top_conf * 100
                self.conf_lbl.config(text=f"{conf_pct:.1f} %", fg=COLOR_SUCCESS)
                self.conf_bar["value"] = conf_pct
                self.prop1_val.config(text=str(tot))
                self.prop2_val.config(text=top_cls)

        # Convert to ImageTk and update canvas
        rgb_disp = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_disp)
        self.image_tk = ImageTk.PhotoImage(image=pil_img)
        self.video_canvas.config(image=self.image_tk, text="")

        # Update FPS counter
        self.fps_lbl.config(text=f"FPS: {fps:.1f}")

    def reset_view(self):
        """Clears video display and resets prediction panel to idle state."""
        self.video_canvas.config(
            image="",
            text="Camera is Stopped\n\nClick '▶ Start Camera' to resume recognition",
        )
        if self.current_model_key == "mediapipe":
            self.gesture_display.config(text="No hand detected", fg=COLOR_TEXT_MUTED)
            self.prop1_val.config(text="None")
            self.prop2_val.config(text="0")
        else:
            self.gesture_display.config(text="No object detected", fg=COLOR_TEXT_MUTED)
            self.prop1_val.config(text="0")
            self.prop2_val.config(text="None")

        self.conf_lbl.config(text="0.0 %", fg=COLOR_TEXT_MUTED)
        self.conf_bar["value"] = 0
        self.fps_lbl.config(text="FPS: 0.0")
        self.image_tk = None
