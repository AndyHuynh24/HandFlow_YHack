#!/usr/bin/env python3
"""Process video through the full HandFlow detection pipeline with visualization."""
import argparse
import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mediapipe as mp
import tensorflow as tf
from handflow.utils import load_config, load_setting, Setting
from handflow.features import FeatureEngineer
from handflow.detector.handedness_tracker import HandTracker
from handflow.detector import ArUcoScreenDetector, MacroPadDetector


class Colors:
    """Color palette for elegant visualization."""
    # Main colors (BGR format)
    PRIMARY = (255, 180, 0)      # Cyan-ish blue
    SECONDARY = (180, 255, 0)   # Lime green
    ACCENT = (0, 165, 255)      # Orange
    WARNING = (0, 100, 255)     # Red-orange

    # Hand colors
    RIGHT_HAND = (255, 200, 100)  # Light blue
    LEFT_HAND = (100, 200, 255)   # Light orange

    # UI colors
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    TEXT_SHADOW = (30, 30, 30)

    # Overlay
    PANEL_BG = (40, 40, 40)
    PANEL_BORDER = (80, 80, 80)

    # Gesture colors
    GESTURE_NONE = (100, 100, 100)
    GESTURE_ACTIVE = (0, 255, 150)
    GESTURE_TOUCH = (0, 200, 255)


def draw_text_with_shadow(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = Colors.TEXT_PRIMARY,
    thickness: int = 1,
    shadow_offset: int = 2
):
    """Draw text with subtle shadow for better readability."""
    x, y = pos
    # Shadow
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, Colors.TEXT_SHADOW, thickness + 1)
    # Main text
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def draw_rounded_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
    alpha: float = 0.7
):
    """Draw a rounded rectangle with optional transparency."""
    overlay = frame.copy()

    # Draw filled rounded rectangle using multiple shapes
    if thickness == -1:
        # Fill
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_elegant_landmark(
    frame: np.ndarray,
    x: int, y: int,
    color: Tuple[int, int, int],
    size: int = 4,
    glow: bool = True
):
    """Draw an elegant landmark point with optional glow effect."""
    if glow:
        # Outer glow
        cv2.circle(frame, (x, y), size + 3, (*color[:3],), 1, cv2.LINE_AA)
    # Inner filled
    cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)
    # Highlight
    cv2.circle(frame, (x - 1, y - 1), max(1, size // 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_elegant_connection(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2
):
    """Draw an elegant connection line with anti-aliasing."""
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def draw_touch_effect(
    frame: np.ndarray,
    x: int, y: int,
    intensity: float = 1.0
):
    """Draw a touch visual effect (ripple/glow) at the given position."""
    # Simplified touch effect - no expensive overlay copies
    # Outer rings (just draw directly)
    cv2.circle(frame, (x, y), int(35 * intensity), (0, 200, 200), 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), int(25 * intensity), (0, 230, 230), 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), int(15 * intensity), (0, 255, 255), 2, cv2.LINE_AA)

    # Inner glow
    cv2.circle(frame, (x, y), int(10 * intensity), (0, 255, 255), -1, cv2.LINE_AA)

    # Bright center
    cv2.circle(frame, (x, y), int(5 * intensity), (255, 255, 255), -1, cv2.LINE_AA)


def draw_elegant_aruco_screen(
    frame: np.ndarray,
    detector: ArUcoScreenDetector,
    finger_pos: Optional[Tuple[int, int]] = None
):
    """Draw elegant ArUco screen boundary visualization."""
    if not detector.is_valid:
        return

    corners = detector.screen_corners
    if corners is None:
        return

    h, w = frame.shape[:2]

    # Determine color based on detection quality
    mode = detector.detection_mode
    mode_colors = {
        "full": (0, 220, 180),      # Teal - best
        "partial_3": (0, 200, 255),  # Orange - good
        "partial_2": (0, 150, 255),  # Deep orange
        "grace": (200, 200, 0),      # Cyan - cached
    }
    boundary_color = mode_colors.get(mode, (0, 220, 180))

    # Draw screen boundary with glow effect
    pts = corners.astype(np.int32).reshape((-1, 1, 2))

    # Outer glow
    overlay = frame.copy()
    cv2.polylines(overlay, [pts], True, boundary_color, 6, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Main boundary
    cv2.polylines(frame, [pts], True, boundary_color, 2, cv2.LINE_AA)

    # Draw corner markers (small, elegant)
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, corner in enumerate(corners):
        cx, cy = int(corner[0]), int(corner[1])
        # Small corner indicator
        cv2.circle(frame, (cx, cy), 4, boundary_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, boundary_color, 1, cv2.LINE_AA)

    # Draw subtle "SCREEN" label at top center
    top_center = ((corners[0] + corners[1]) / 2).astype(int)
    label_pos = (top_center[0] - 30, top_center[1] - 10)
    draw_text_with_shadow(frame, "SCREEN", label_pos, font_scale=0.4,
                         color=boundary_color, thickness=1, shadow_offset=1)

    # If finger is in screen area, show mapping indicator
    if finger_pos is not None:
        if detector.is_point_in_screen(finger_pos):
            screen_pos = detector.transform_point(finger_pos)
            if screen_pos:
                # Draw subtle crosshair at finger position
                fx, fy = int(finger_pos[0]), int(finger_pos[1])
                cv2.line(frame, (fx - 8, fy), (fx + 8, fy), (0, 255, 100), 1, cv2.LINE_AA)
                cv2.line(frame, (fx, fy - 8), (fx, fy + 8), (0, 255, 100), 1, cv2.LINE_AA)


def draw_elegant_macropad(
    frame: np.ndarray,
    detector: MacroPadDetector,
    finger_pos: Optional[Tuple[int, int]] = None,
    hovered_button: Optional[int] = None,
    button_names: Optional[List[str]] = None
):
    """Draw elegant macropad grid visualization."""
    if not detector.is_valid:
        return

    detection = detector.detection
    if detection is None:
        return

    h, w = frame.shape[:2]

    # Draw detection region with subtle glow
    region_pts = detection.detection_region.astype(np.int32)
    overlay = frame.copy()
    cv2.polylines(overlay, [region_pts], True, (180, 120, 0), 4, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [region_pts], True, (180, 120, 0), 1, cv2.LINE_AA)

    # Draw grid cells
    for idx, cell in enumerate(detection.grid_cells):
        cell_pts = cell.astype(np.int32)

        # Determine cell state and color
        if idx == hovered_button:
            color = (0, 255, 200)  # Bright cyan for hovered
            thickness = 2
            alpha = 0.3
        else:
            color = (140, 100, 0)  # Subtle blue
            thickness = 1
            alpha = 0.0

        # Draw cell fill if hovered
        if alpha > 0:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [cell_pts], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw cell border
        cv2.polylines(frame, [cell_pts], True, color, thickness, cv2.LINE_AA)

    # Draw prominent "MACROPAD" title label
    top_center = ((detection.detection_region[0] + detection.detection_region[1]) / 2).astype(int)
    label_text = "MACROPAD"
    font_scale = 1.2
    label_thickness = 3
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
    label_pos = (top_center[0] - tw // 2, top_center[1] - 15)

    # Draw with glow effect for maximum visibility
    # Outer glow
    cv2.putText(frame, label_text, (label_pos[0], label_pos[1]),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), label_thickness + 4, cv2.LINE_AA)
    # Main text - bright cyan/yellow for attention
    cv2.putText(frame, label_text, (label_pos[0], label_pos[1]),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), label_thickness, cv2.LINE_AA)


# MediaPipe hand connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17)
]

# Fingertip indices
FINGERTIPS = [4, 8, 12, 16, 20]


def draw_elegant_hand(
    frame: np.ndarray,
    landmarks: List[Tuple[int, int]],
    handedness: str,
    gesture: str = "none",
    confidence: float = 0.0,
    show_gesture_label: bool = False
):
    """Draw elegant hand visualization with gesture info."""
    h, w = frame.shape[:2]

    # Choose color based on hand
    if handedness == "Right":
        base_color = Colors.RIGHT_HAND
        accent_color = (255, 150, 50)
    else:
        base_color = Colors.LEFT_HAND
        accent_color = (50, 150, 255)

    # Modify color based on gesture
    if gesture in ('touch', 'touch_hold'):
        base_color = Colors.GESTURE_TOUCH
    elif gesture != 'none':
        base_color = Colors.GESTURE_ACTIVE

    # Draw connections first (behind landmarks)
    for conn in HAND_CONNECTIONS:
        if conn[0] < len(landmarks) and conn[1] < len(landmarks):
            pt1 = landmarks[conn[0]]
            pt2 = landmarks[conn[1]]
            # Gradient thickness - thinner at fingertips
            thickness = 2 if conn[1] in FINGERTIPS else 3
            draw_elegant_connection(frame, pt1, pt2, base_color, thickness)

    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        if i == 0:  # Wrist - larger
            draw_elegant_landmark(frame, x, y, base_color, size=6, glow=True)
        elif i in FINGERTIPS:  # Fingertips - medium with glow
            draw_elegant_landmark(frame, x, y, accent_color, size=5, glow=True)
        else:  # Joints - small
            draw_elegant_landmark(frame, x, y, base_color, size=3, glow=False)

    # Draw touch effect at index fingertip when touch gesture detected
    if gesture in ('touch', 'touch_hold') and len(landmarks) > 8:
        index_tip = landmarks[8]
        draw_touch_effect(frame, index_tip[0], index_tip[1])


def draw_skeleton_frame(
    frame_shape: Tuple[int, int, int],
    detections: Dict,
    gesture_counters: Dict[str, int],
    last_gestures: Dict[str, str]
) -> np.ndarray:
    """
    Create a skeleton-only frame with black background.
    Includes hand skeleton and touch effects.
    """
    h, w = frame_shape[:2]
    # Black background
    skeleton_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw each detected hand
    for handedness, data in detections.items():
        landmarks = data.get('landmarks', [])
        gesture = data.get('gesture', 'none')

        if not landmarks:
            continue

        # Choose color based on hand
        if handedness == "Right":
            base_color = Colors.RIGHT_HAND
            accent_color = (255, 150, 50)
        else:
            base_color = Colors.LEFT_HAND
            accent_color = (50, 150, 255)

        # Modify color based on gesture
        if gesture in ('touch', 'touch_hold'):
            base_color = Colors.GESTURE_TOUCH
        elif gesture != 'none':
            base_color = Colors.GESTURE_ACTIVE

        # Draw connections
        for conn in HAND_CONNECTIONS:
            if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                pt1 = landmarks[conn[0]]
                pt2 = landmarks[conn[1]]
                thickness = 2 if conn[1] in FINGERTIPS else 3
                cv2.line(skeleton_frame, pt1, pt2, base_color, thickness, cv2.LINE_AA)

        # Draw landmarks
        for i, (x, y) in enumerate(landmarks):
            if i == 0:  # Wrist
                cv2.circle(skeleton_frame, (x, y), 6, base_color, -1, cv2.LINE_AA)
            elif i in FINGERTIPS:  # Fingertips
                cv2.circle(skeleton_frame, (x, y), 5, accent_color, -1, cv2.LINE_AA)
            else:  # Joints
                cv2.circle(skeleton_frame, (x, y), 3, base_color, -1, cv2.LINE_AA)

        # Draw touch effect at index fingertip when touch gesture detected
        if gesture in ('touch', 'touch_hold') and len(landmarks) > 8:
            index_tip = landmarks[8]
            draw_touch_effect(skeleton_frame, index_tip[0], index_tip[1])

    return skeleton_frame


def draw_gesture_label_top_center(
    frame: np.ndarray,
    gesture: str,
    handedness: str,
    offset_index: int = 0
):
    """Draw gesture label at top center of frame."""
    if frame is None or gesture is None:
        return

    h, w = frame.shape[:2]

    # Choose color based on hand
    if handedness == "Right":
        color = Colors.RIGHT_HAND
    else:
        color = Colors.LEFT_HAND

    # Modify color based on gesture type
    if gesture in ('touch', 'touch_hold'):
        color = Colors.GESTURE_TOUCH
    elif gesture != 'none':
        color = Colors.GESTURE_ACTIVE

    # Draw at top center (offset vertically if multiple gestures)
    label_text = str(gesture)
    font_scale = 1.2
    thickness = 3

    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    label_x = int((w - text_w) // 2)
    label_y = int(150 + (offset_index * 55))  # Stack vertically if multiple gestures

    # Draw background for better visibility (with bounds checking)
    padding = 10
    rect_x1 = max(0, label_x - padding)
    rect_y1 = max(0, label_y - text_h - padding)
    rect_x2 = min(w, label_x + text_w + padding)
    rect_y2 = min(h, label_y + padding)

    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

    # Draw text with shadow for visibility
    cv2.putText(frame, label_text, (label_x + 2, label_y + 2),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, label_text, (label_x, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_info_panel(
    frame: np.ndarray,
    fps: float,
    gestures: Dict[str, Tuple[str, float]],
    frame_num: int,
    total_frames: int
):
    """Draw elegant info panel in corner."""
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 200
    panel_h = 100
    margin = 10

    # Draw panel background
    draw_rounded_rect(
        frame,
        margin, margin,
        margin + panel_w, margin + panel_h,
        Colors.PANEL_BG,
        radius=8,
        alpha=0.75
    )

    # Panel content
    y_offset = margin + 20

    # FPS
    draw_text_with_shadow(
        frame, f"FPS: {fps:.1f}",
        (margin + 10, y_offset),
        font_scale=0.45,
        color=Colors.PRIMARY
    )
    y_offset += 20

    # Progress
    progress = frame_num / max(total_frames, 1)
    draw_text_with_shadow(
        frame, f"Frame: {frame_num}/{total_frames}",
        (margin + 10, y_offset),
        font_scale=0.4,
        color=Colors.TEXT_SECONDARY
    )
    y_offset += 15

    # Progress bar
    bar_x = margin + 10
    bar_w = panel_w - 20
    cv2.rectangle(frame, (bar_x, y_offset), (bar_x + bar_w, y_offset + 4),
                 Colors.PANEL_BORDER, -1)
    cv2.rectangle(frame, (bar_x, y_offset), (bar_x + int(bar_w * progress), y_offset + 4),
                 Colors.PRIMARY, -1)
    y_offset += 15

    # Gesture info
    for hand, (gesture, conf) in gestures.items():
        if gesture != "none":
            color = Colors.RIGHT_HAND if hand == "Right" else Colors.LEFT_HAND
            draw_text_with_shadow(
                frame, f"{hand[0]}: {gesture}",
                (margin + 10, y_offset),
                font_scale=0.4,
                color=color
            )
            y_offset += 15


class VideoProcessor:
    """Full HandFlow video processor with elegant visualization."""

    def __init__(
        self,
        model_path: str = "models/hand_action.tflite",
        config_path: str = "config/config.yaml",
        setting_path: str = "config/handflow_setting.yaml",
        minimal_mode: bool = False,
        enable_aruco: bool = True,
        enable_macropad: bool = True,
        ignored_gestures: List[str] = None
    ):
        self.minimal_mode = minimal_mode
        self.enable_aruco = enable_aruco
        self.enable_macropad = enable_macropad

        # Load config
        self.config = load_config(config_path)
        self.gesture_classes = self.config.model.gestures

        # Load user settings (for macropad button names)
        self.setting = None
        self.button_names = None
        try:
            if os.path.exists(setting_path):
                self.setting = load_setting(setting_path)
                # Get button names from first macropad set
                if self.setting.macropad_sets:
                    first_set = self.setting.macropad_sets[0]
                    self.button_names = []
                    for i in range(12):
                        btn = first_set.buttons.get(i)
                        if btn and btn.label:
                            self.button_names.append(btn.label)
                        else:
                            self.button_names.append("")
        except Exception as e:
            print(f"Note: Could not load settings ({e}), using defaults")

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
            model_complexity=self.config.mediapipe.model_complexity
        )

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Load TFLite model
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model(model_path)

        # Hand tracker
        self.hand_tracker = HandTracker()

        # Sequence buffers
        self.sequence_length = self.config.data.sequence_length
        self.right_sequence = []
        self.left_sequence = []

        # Smoothing
        self.right_predictions = deque(maxlen=5)
        self.left_predictions = deque(maxlen=5)

        # Minimum confidence threshold for gesture recognition
        self.min_confidence_threshold = 0.7

        # Gestures to ignore (won't be displayed even if detected)
        self.ignored_gestures = set(ignored_gestures) if ignored_gestures else set()

        # Current gestures
        self.current_gestures = {"Right": ("none", 0.0), "Left": ("none", 0.0)}

        # Prediction throttling - only predict every N frames for performance
        self.predict_every_n_frames = 2
        self._frame_counter = 0

        # Gesture display timing (show for N frames after detection)
        self.gesture_display_frames = 35  # Show gesture label for ~1.75 seconds at 20fps
        self.gesture_display_counters = {"Right": 0, "Left": 0}
        self.last_displayed_gesture = {"Right": "none", "Left": "none"}

        # ArUco screen detector
        self.aruco_detector = None
        if enable_aruco:
            try:
                self.aruco_detector = ArUcoScreenDetector(
                    screen_width=1920,
                    screen_height=1080
                )
                print("ArUco screen detector: enabled")
            except Exception as e:
                print(f"ArUco detector not available: {e}")

        # MacroPad detector
        self.macropad_detector = None
        if enable_macropad:
            try:
                self.macropad_detector = MacroPadDetector()
                print("MacroPad detector: enabled")
            except Exception as e:
                print(f"MacroPad detector not available: {e}")

        # Current finger position (for ArUco/MacroPad interaction)
        self.current_finger_pos = None
        self.hovered_button = None
        self._finger_in_screen_area = False  # Track if finger is in screen area

        print(f"Loaded model: {model_path}")
        print(f"Gesture classes: {self.gesture_classes}")
        if self.ignored_gestures:
            print(f"Ignored gestures: {list(self.ignored_gestures)}")

    def _load_model(self, model_path: str):
        """Load TFLite model."""
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _predict_gesture(self, sequence: np.ndarray) -> Tuple[str, float]:
        """Run gesture prediction on sequence."""
        if self.interpreter is None:
            return "none", 0.0

        # Prepare input
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        predicted_idx = np.argmax(output)
        confidence = output[predicted_idx]
        gesture = self.gesture_classes[predicted_idx]

        return gesture, float(confidence)

    def _get_smoothed_prediction(self, predictions: deque, gesture: str, conf: float) -> Tuple[str, float]:
        """Get smoothed prediction from recent history."""
        predictions.append((gesture, conf))

        if len(predictions) < 3:
            return gesture, conf

        # Count recent predictions
        recent = list(predictions)[-5:]
        gesture_counts = {}
        for g, c in recent:
            if g not in gesture_counts:
                gesture_counts[g] = []
            gesture_counts[g].append(c)

        # Find most common with minimum confidence
        best_gesture = "none"
        best_conf = 0.0
        for g, confs in gesture_counts.items():
            avg_conf = sum(confs) / len(confs)
            if len(confs) >= 2 and avg_conf > best_conf:
                best_gesture = g
                best_conf = avg_conf

        # Apply confidence threshold - reject low confidence predictions
        if best_conf < self.min_confidence_threshold:
            return "none", best_conf

        return best_gesture, best_conf

    def process_frame(self, frame: np.ndarray, frame_num: int = 0) -> Tuple[np.ndarray, Dict]:
        """Process a single frame and return annotated frame."""
        h, w = frame.shape[:2]
        output = frame.copy()

        # Increment frame counter for prediction throttling
        self._frame_counter += 1

        # Reset finger tracking
        self.current_finger_pos = None
        self.hovered_button = None

        # ArUco/MacroPad detection
        # Run every frame if finger is in screen area (for accurate touch detection)
        # Otherwise run every 3 frames for performance
        run_aruco = (
            self._finger_in_screen_area or
            frame_num % 3 == 0 or
            frame_num <= 1
        )

        if run_aruco:
            # ArUco screen detection
            if self.aruco_detector is not None:
                self.aruco_detector.detect(frame)

            # MacroPad detection (use same frame)
            if self.macropad_detector is not None:
                self.macropad_detector.detect(frame)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        detections = {}

        if results.multi_hand_landmarks:
            # Use hand tracker for reliable handedness
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                hand_data.append({
                    'wrist_x': wrist.x,
                    'landmarks': hand_landmarks
                })

            # Sort by x position for handedness assignment
            hand_data.sort(key=lambda h: h['wrist_x'])

            # Assign handedness (rightmost = Right hand typically with mirrored camera)
            for idx, data in enumerate(hand_data):
                hand_landmarks = data['landmarks']

                # Determine handedness
                if len(hand_data) == 1:
                    handedness = "Right"  # Default single hand
                else:
                    handedness = "Left" if idx == 0 else "Right"

                # Extract landmarks as pixel coordinates
                landmarks = []
                landmarks_flat = []  # Flat array for feature engineering (x, y, z, visibility)
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmarks.append((px, py))
                    # Feature engineer expects (21*4) = 84 values: x, y, z, visibility
                    landmarks_flat.extend([lm.x, lm.y, lm.z, lm.visibility])

                landmarks_np = np.array(landmarks_flat)  # Shape: (84,)

                # Add raw landmarks to sequence (feature engineering done on full sequence)
                if handedness == "Right":
                    self.right_sequence.append(landmarks_np)
                    if len(self.right_sequence) > self.sequence_length:
                        self.right_sequence.pop(0)

                    # Predict if sequence is full (throttled for performance)
                    if len(self.right_sequence) == self.sequence_length:
                        if self._frame_counter % self.predict_every_n_frames == 0:
                            seq_array = np.array(self.right_sequence)  # (seq_len, 84)
                            # Feature engineering on entire sequence
                            features = self.feature_engineer.transform(seq_array)
                            gesture, conf = self._predict_gesture(features)
                            gesture, conf = self._get_smoothed_prediction(
                                self.right_predictions, gesture, conf
                            )
                            self.current_gestures["Right"] = (gesture, conf)
                else:
                    self.left_sequence.append(landmarks_np)
                    if len(self.left_sequence) > self.sequence_length:
                        self.left_sequence.pop(0)

                    if len(self.left_sequence) == self.sequence_length:
                        if self._frame_counter % self.predict_every_n_frames == 0:
                            seq_array = np.array(self.left_sequence)  # (seq_len, 84)
                            # Feature engineering on entire sequence
                            features = self.feature_engineer.transform(seq_array)
                            gesture, conf = self._predict_gesture(features)
                            gesture, conf = self._get_smoothed_prediction(
                                self.left_predictions, gesture, conf
                            )
                            self.current_gestures["Left"] = (gesture, conf)

                # Get current gesture for this hand
                gesture, conf = self.current_gestures[handedness]

                # Track finger position (index fingertip) for ArUco/MacroPad interaction
                finger_pos = None
                if len(landmarks) > 8:
                    finger_pos = landmarks[8]  # Index fingertip
                    if handedness == "Right":
                        self.current_finger_pos = finger_pos
                        # Check if finger is in ArUco screen area (for next frame's detection rate)
                        if self.aruco_detector is not None and self.aruco_detector.is_valid:
                            self._finger_in_screen_area = self.aruco_detector.is_point_in_screen(finger_pos)
                        else:
                            self._finger_in_screen_area = False

                # Suppress touch gestures:
                # 1. Touch only works with RIGHT hand
                # 2. Finger must be in ArUco screen area
                display_gesture = gesture
                if gesture in ('touch_hover', 'touch', 'touch_hold'):
                    # Touch gestures only valid for right hand
                    if handedness != "Right":
                        display_gesture = "none"
                    elif finger_pos is not None:
                        # Use cached finger_in_screen state
                        if not self._finger_in_screen_area:
                            display_gesture = "none"

                # Filter out ignored gestures
                if display_gesture in self.ignored_gestures:
                    display_gesture = "none"

                # Manage gesture display timing
                # Reset counter when a new non-none gesture is detected
                if display_gesture != "none" and display_gesture != self.last_displayed_gesture[handedness]:
                    self.gesture_display_counters[handedness] = self.gesture_display_frames
                    self.last_displayed_gesture[handedness] = display_gesture

                # Decrement counter
                if self.gesture_display_counters[handedness] > 0:
                    self.gesture_display_counters[handedness] -= 1

                # Show label only while counter is active
                show_label = self.gesture_display_counters[handedness] > 0

                # Draw elegant hand
                draw_elegant_hand(
                    output, landmarks, handedness,
                    gesture=display_gesture,
                    confidence=conf,
                    show_gesture_label=show_label
                )

                detections[handedness] = {
                    'gesture': display_gesture,
                    'confidence': conf,
                    'landmarks': landmarks
                }

        # Check macropad hover state
        if self.current_finger_pos is not None and self.macropad_detector is not None:
            if self.macropad_detector.is_valid:
                self.hovered_button = self.macropad_detector.get_button_at_point(self.current_finger_pos)

        # Draw ArUco screen boundary (elegant, subtle)
        if self.aruco_detector is not None and not self.minimal_mode:
            draw_elegant_aruco_screen(output, self.aruco_detector, self.current_finger_pos)

        # Draw MacroPad grid (elegant, subtle)
        if self.macropad_detector is not None and not self.minimal_mode:
            draw_elegant_macropad(
                output,
                self.macropad_detector,
                self.current_finger_pos,
                self.hovered_button,
                self.button_names
            )

        return output, detections

    def close(self):
        """Release resources."""
        self.hands.close()


def process_video(
    input_path: str,
    output_path: str = None,
    model_path: str = "models/hand_action.tflite",
    config_path: str = "config/config.yaml",
    setting_path: str = "config/handflow_setting.yaml",
    minimal_mode: bool = False,
    show_preview: bool = False,
    enable_aruco: bool = True,
    enable_macropad: bool = True,
    export_skeleton: bool = False,
    ignored_gestures: List[str] = None
):
    """
    Process a video file with full HandFlow detection pipeline.
    """
    # Validate input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Generate output path
    if output_path is None:
        input_stem = Path(input_path).stem
        input_dir = Path(input_path).parent
        output_path = str(input_dir / f"{input_stem}_processed.mp4")

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        sys.exit(1)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"{'='*50}")
    print(f"HandFlow Video Processor")
    print(f"{'='*50}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Frames: {total_frames}")
    print(f"{'='*50}")

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video")
        cap.release()
        sys.exit(1)

    # Initialize skeleton video writer if requested
    skeleton_out = None
    skeleton_path = None
    if export_skeleton:
        skeleton_path = str(Path(output_path).with_stem(Path(output_path).stem + "_skeleton"))
        skeleton_out = cv2.VideoWriter(skeleton_path, fourcc, fps, (width, height))
        if skeleton_out.isOpened():
            print(f"Skeleton: {skeleton_path}")
        else:
            print("Warning: Could not create skeleton video")
            skeleton_out = None

    # Initialize processor
    processor = VideoProcessor(
        model_path=model_path,
        config_path=config_path,
        setting_path=setting_path,
        minimal_mode=minimal_mode,
        enable_aruco=enable_aruco,
        enable_macropad=enable_macropad,
        ignored_gestures=ignored_gestures
    )

    # Process frames
    frame_num = 0
    start_time = time.time()
    fps_counter = 0
    fps_time = time.time()
    current_fps = 0.0

    print("\nProcessing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            fps_counter += 1

            # Calculate FPS
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()

            # Process frame
            output, detections = processor.process_frame(frame, frame_num)

            # Write frame
            out.write(output)

            # Write skeleton frame if enabled
            if skeleton_out is not None:
                skeleton_frame = draw_skeleton_frame(
                    frame.shape,
                    detections,
                    processor.gesture_display_counters,
                    processor.last_displayed_gesture
                )
                skeleton_out.write(skeleton_frame)

            # Preview
            if show_preview:
                preview = cv2.resize(output, (960, 540))
                cv2.imshow('Processing (Q to quit)', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            # Progress
            if frame_num % 50 == 0 or frame_num == total_frames:
                progress = (frame_num / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_num) * (total_frames - frame_num) if frame_num > 0 else 0
                print(f"\rProgress: {frame_num}/{total_frames} ({progress:.1f}%) "
                      f"ETA: {eta:.0f}s", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        cap.release()
        out.release()
        if skeleton_out is not None:
            skeleton_out.release()
        processor.close()
        if show_preview:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n\n{'='*50}")
    print(f"Done!")
    print(f"Processed {frame_num} frames in {elapsed:.1f}s ({frame_num/elapsed:.1f} FPS)")
    print(f"Output: {output_path}")
    if skeleton_path:
        print(f"Skeleton: {skeleton_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Process video with HandFlow detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/process_video.py input.mp4
    python scripts/process_video.py input.mp4 -o output.mp4
    python scripts/process_video.py input.mp4 --preview
    python scripts/process_video.py recordings/raw_*.mp4 --minimal
        """
    )

    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument("--model", default="models/hand_action.tflite",
                       help="Path to gesture model (default: models/hand_action.tflite)")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--setting", default="config/handflow_setting.yaml",
                       help="Path to user settings file")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal visualization (no overlays)")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview window while processing")
    parser.add_argument("--no-aruco", action="store_true",
                       help="Disable ArUco screen detection")
    parser.add_argument("--no-macropad", action="store_true",
                       help="Disable MacroPad detection")
    parser.add_argument("--skeleton", action="store_true",
                       help="Export additional skeleton-only video (black background)")
    parser.add_argument("--ignore-gestures", nargs="+", default=[],
                       help="List of gestures to ignore (e.g., --ignore-gestures pinch swipe)")

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        config_path=args.config,
        setting_path=args.setting,
        minimal_mode=args.minimal,
        show_preview=args.preview,
        enable_aruco=not args.no_aruco,
        enable_macropad=not args.no_macropad,
        export_skeleton=args.skeleton,
        ignored_gestures=args.ignore_gestures
    )


if __name__ == "__main__":
    main()
