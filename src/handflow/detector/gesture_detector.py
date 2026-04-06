"""Real-time gesture detection pipeline: MediaPipe landmarks -> feature engineering -> TFLite inference -> action execution."""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from collections import deque, Counter

from handflow.utils import Setting, get_logger
from handflow.utils.smoothing import OneEuroFilter
from handflow.detector.handedness_tracker import HandTracker
import mediapipe as mp



from handflow.actions import ActionExecutor
from handflow.features import FeatureEngineer

if TYPE_CHECKING:
    from handflow.detector import ArUcoScreenDetector


class GestureDetector:
    """Runs the full gesture recognition pipeline per frame: tracking, feature extraction, classification, and action dispatch."""
    def __init__(
        self,
        setting: Setting,
        executor: ActionExecutor,
        aruco_detector: Optional["ArUcoScreenDetector"] = None,
    ):
        self.logger = get_logger("handflow.GestureDetector")
        self.setting = setting
        self.executor = executor

        from handflow.utils import load_config
        self.config = load_config("config/config.yaml")

        self.DEFAULT_MODEL = self.config.model.model_path
        self.gesture_classes = self.config.model.gestures
        self.logger.info(f"[GestureDetector] Hand gestures: {self.gesture_classes}")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
            max_num_hands=self.config.mediapipe.max_num_hands, 
            model_complexity=self.config.mediapipe.model_complexity
        )

        self.feature_engineer = FeatureEngineer()
        self.model_path = self.DEFAULT_MODEL
        self._load_model()

        self.sequence_length = self.config.data.sequence_length
        self.right_sequence = []
        self.left_sequence = []
        self.right_lock = False
        self.left_lock = False

        self.right_predictions = deque(maxlen=10)
        self.left_predictions = deque(maxlen=10)

        self.COOLDOWN_FRAMES = setting.inference.cooldown_frames
        self.right_cooldown = 0
        self.left_cooldown = 0

        self.threshold = self.setting.inference.confidence_threshold
        self.stability_window = self.setting.inference.stability_window

        self.res_right = np.zeros(len(self.gesture_classes))
        self.res_left = np.zeros(len(self.gesture_classes))
        self.gesture_display_history = deque(maxlen=5)
        self._last_activated_gesture = None  # (gesture_name, hand, timestamp)

        # touch_hover/touch_hold need cursor updates every frame, not just on model ticks
        self._last_right_gesture = "none"
        self._last_left_gesture = "none"
        self.actions_disabled = False  # When True, skip all action execution

        # Thumb desktop switch: touch → hold (2 frames) → release
        self._thumb_awaiting_hold = False
        self._thumb_hold_count = 0
        self._thumb_switched = False

        # Live capture: pointyclick starts, spacebar ends
        self._cap_state = "idle"   # "idle" or "holding"
        self._cap_start = None
        self._cap_end = None
        self._cap_complete = False
        self._cap_cooldown_until = 0.0


        self._fps_start_time = time.time()
        self._fps_counter = 0
        self.current_fps = 0.0
        self._data_fps_counter = 0
        self._data_fps = 0.0

        self.aruco_detector = aruco_detector
        self._touch_cursor_enabled = aruco_detector is not None

        self._right_index_tip: Optional[Tuple[float, float]] = None
        self._left_index_tip: Optional[Tuple[float, float]] = None
        self._frame_size: Tuple[int, int] = (1280, 720)

        if self._touch_cursor_enabled:
            self.logger.info("[GestureDetector] Touch-to-cursor enabled via ArUco")

        self.hand_tracker = HandTracker()

        # OneEuro filter params: min_cutoff controls smoothness, beta controls speed reactivity
        self._filter_min_cutoff = 1.4
        self._filter_beta = 0.07
        self._filter_d_cutoff = 1.0

        curr_time = time.time()
        self._right_tip_filter_x = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)
        self._right_tip_filter_y = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)
        self._left_tip_filter_x = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)
        self._left_tip_filter_y = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)

        self._is_dragging = False
        self._last_drag_pos: Optional[Tuple[int, int]] = None
        # Grace period prevents brief gesture fluctuations from interrupting drags
        self._drag_end_grace_frames = 0
        self._drag_grace_max = 2

        self._macropad_active = False

        # Cache finger tip positions to use pre-touch frames (less jittery than touch-moment)
        self.TOUCH_CACHE_SIZE = 8
        self.TOUCH_CACHE_LOOKBACK = 6
        self._right_tip_cache: deque = deque(maxlen=self.TOUCH_CACHE_SIZE)
        self._left_tip_cache: deque = deque(maxlen=self.TOUCH_CACHE_SIZE)

        self._cached_mp_results = None
        self._cached_right_kp = None
        self._cached_left_kp = None

        target_fps = getattr(self.config.data, 'target_fps', 20.0)
        self._last_delta_time = 1.0 / target_fps
        self._target_fps = target_fps

        # Adaptive frame sampling: skip frames when FPS exceeds target, use all when below
        self._data_rate_limit_enabled = True
        self._frame_accumulator = 0.0
        self._actual_fps = target_fps
        self._fps_sample_times: deque = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Interpolate missing frames on slow devices to maintain temporal consistency
        self._last_right_kp: Optional[np.ndarray] = None
        self._last_left_kp: Optional[np.ndarray] = None
        self._interpolation_enabled = True

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                interpreter = tf.lite.Interpreter(model_path=self.model_path)
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]

                self.interpreter = {
                    'interpreter': interpreter,
                    'input': input_details,
                    'output': output_details
                }

                self.logger.info(f"[GestureDetector] Loaded model: {self.model_path}")
                self.logger.info(f"[GestureDetector]   Input shape: {input_details['shape']}")
                self.logger.info(f"[GestureDetector]   Output shape: {output_details['shape']}")
            except Exception as e:
                self.logger.info(f"[GestureDetector] Error loading model: {e}")
        else:
            self.logger.info(f"[GestureDetector] Model not found: {self.model_path}")

    def set_aruco_detector(self, detector: "ArUcoScreenDetector") -> None:
        self.aruco_detector = detector
        self._touch_cursor_enabled = detector is not None
        if self._touch_cursor_enabled:
            self.logger.info(f"[GestureDetector] ArUco detector attached")

    def set_macropad_active(self, active: bool) -> None:
        """Disable cursor movements when macropad is handling touch interaction."""
        self._macropad_active = active

    def set_data_rate_limit(self, enabled: bool) -> None:
        self._data_rate_limit_enabled = enabled

    def set_interpolation(self, enabled: bool) -> None:
        self._interpolation_enabled = enabled

    def _get_cached_tip(self, hand: str) -> Optional[Tuple[float, float]]:
        """Return finger tip position from a few frames ago (less jittery than touch-moment position)."""
        cache = self._right_tip_cache if hand == "Right" else self._left_tip_cache

        if len(cache) < self.TOUCH_CACHE_LOOKBACK:
            # Not enough history, return current position as fallback
            return self._right_index_tip if hand == "Right" else self._left_index_tip

        lookback_idx = -self.TOUCH_CACHE_LOOKBACK
        return cache[lookback_idx]

    def _click_at_touch(self, hand: str) -> bool:
        """Map finger tip to screen coordinates via ArUco homography and click."""
        if self._macropad_active:
            self.logger.debug("[Touch] Skipped - macropad is active")
            return False

        if hand != "Right":
            return False
        if self.aruco_detector is None:
            return False
        if not self.aruco_detector.is_valid:
            return False

        tip = self._get_cached_tip(hand)
        if tip is None:
            return False

        cam_w, cam_h = self._frame_size
        finger_x = tip[0] * cam_w
        finger_y = tip[1] * cam_h

        self.logger.info(f"[Touch] Using cached tip: norm=({tip[0]:.3f}, {tip[1]:.3f}) -> cam_px=({finger_x:.1f}, {finger_y:.1f})")

        # Transform to screen coordinates via homography
        screen_pos = self.aruco_detector.transform_point((finger_x, finger_y))

        if screen_pos is None:
            self.logger.info("[Touch] Homography transform failed")
            return False

        screen_x, screen_y = screen_pos
        self.logger.info(f"[Touch] Screen pos: ({screen_x}, {screen_y}) -> Click!")

        # Move cursor and click
        ActionExecutor.click_at(screen_x, screen_y)

        return True

    def _move_on_hover(self, hand: str) -> bool:
        """Move cursor to finger tip position via ArUco homography (no click)."""
        if self._macropad_active:
            return False
        if hand != "Right":
            return False
        if self.aruco_detector is None or not self.aruco_detector.is_valid:
            return False

        tip = self._right_index_tip
        if tip is None:
            return False

        cam_w, cam_h = self._frame_size
        finger_x = tip[0] * cam_w
        finger_y = tip[1] * cam_h

        screen_pos = self.aruco_detector.transform_point((finger_x, finger_y))
        if screen_pos is None:
            return False

        ActionExecutor.move_cursor(int(screen_pos[0]), int(screen_pos[1]))
        return True

    def _drag_on_hold(self, hand: str) -> bool:
        """Drag via ArUco homography: first call presses mouse down, subsequent calls move."""
        if self._macropad_active:
            return False
        if hand != "Right":
            return False
        if self.aruco_detector is None or not self.aruco_detector.is_valid:
            return False

        tip = self._right_index_tip
        if tip is None:
            return False

        cam_w, cam_h = self._frame_size
        finger_x = tip[0] * cam_w
        finger_y = tip[1] * cam_h

        screen_pos = self.aruco_detector.transform_point((finger_x, finger_y))
        if screen_pos is None:
            return False

        final_x, final_y = int(screen_pos[0]), int(screen_pos[1])

        if not self._is_dragging:
            self.logger.info(f"[Touch_hold] Starting drag at ({final_x}, {final_y})")
            ActionExecutor.mouse_down(final_x, final_y)
            self._is_dragging = True
            self._last_drag_pos = (final_x, final_y)
        else:
            ActionExecutor.drag_move(final_x, final_y)
            self._last_drag_pos = (final_x, final_y)
        
        return True

    def _update_continuous_cursor(self):
        """Called every frame (not just model ticks) to keep touch cursor smooth."""
        self._update_drag_grace()

        gesture = self._last_right_gesture
        if self._right_index_tip is None:
            return

        if gesture == "touch_hover":
            self._move_on_hover("Right")
        elif gesture == "touch_hold":
            self._cancel_end_drag()
            self._drag_on_hold("Right")

    def _request_end_drag(self):
        """
        Request to end drag with grace period.
        The drag won't actually end until grace frames expire.
        This prevents brief gesture fluctuations from interrupting drags.
        """
        if self._is_dragging and self._drag_end_grace_frames == 0:
            self._drag_end_grace_frames = self._drag_grace_max

    def _cancel_end_drag(self):
        """Cancel any pending drag end request (gesture came back to touch_hold)."""
        self._drag_end_grace_frames = 0

    def _update_drag_grace(self):
        """Update drag grace period countdown. Called every frame."""
        if self._drag_end_grace_frames > 0:
            self._drag_end_grace_frames -= 1
            if self._drag_end_grace_frames == 0:
                # Grace period expired, actually end the drag
                self._force_end_drag()

    def _force_end_drag(self):
        """Immediately end drag (called when grace period expires)."""
        if self._is_dragging:
            if self._last_drag_pos:
                x, y = self._last_drag_pos
                self.logger.info(f"[Touch_hold] Ending drag at ({x}, {y})")
                ActionExecutor.mouse_up(x, y)
            else:
                import pyautogui
                pos = pyautogui.position()
                ActionExecutor.mouse_up(pos[0], pos[1])
            self._is_dragging = False
            self._last_drag_pos = None

    def _end_drag(self):
        """Request to end drag with grace period to handle gesture fluctuations."""
        self._request_end_drag()


    def _extract_raw_keypoints(self, results, flip_h: bool, swap_hands: bool):
        """
        Extract keypoints using spatial-based hand tracking.
        The HandTracker maintains stable handedness labels by tracking hands
        via centroid position continuity, ignoring MediaPipe's unreliable labels.
        """
        return self.hand_tracker.update(results, flip_h, swap_hands)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_small: np.ndarray = None,
        run_gesture_model: bool = True,
        run_mediapipe: bool = True,
        delta_time: float = None,
        disable_drawing: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a single video frame.

        Args:
            frame: BGR image from camera (full resolution for display)
            frame_small: Optional smaller BGR image for MediaPipe (faster processing)
            run_gesture_model: If False, skip TCN gesture inference (optimization)
            run_mediapipe: If False, skip MediaPipe and reuse cached results (optimization)
            delta_time: Time since last frame in seconds (for FPS-invariant features)
            disable_drawing: If True, skip all drawing operations (for performance testing)
        """
        # Store frame size for coordinate conversion (use full frame size)
        self._frame_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # Update delta time for velocity normalization
        if delta_time is not None and delta_time > 0:
            self._last_delta_time = delta_time

        # FPS calculation
        self._fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self._fps_start_time
        if elapsed > 1.0:
            self.current_fps = self._fps_counter / elapsed  # Loop FPS
            self._data_fps = self._data_fps_counter / elapsed  # Data collection FPS
            self._fps_counter = 0
            self._data_fps_counter = 0
            self._fps_start_time = current_time

        # Fast path: skip MediaPipe, reuse cached finger positions for smooth cursor
        if not run_mediapipe:
            self._fps_sample_times.append(current_time)
            if len(self._fps_sample_times) >= 2:
                time_span = self._fps_sample_times[-1] - self._fps_sample_times[0]
                if time_span > 0:
                    self._actual_fps = (len(self._fps_sample_times) - 1) / time_span
            self._update_continuous_cursor()
            image = frame
            img_h, img_w = image.shape[:2]
            fps_text = f"FPS: {self._data_fps:.0f}/{self._actual_fps:.0f}/{self._target_fps:.0f}"
            if not self._data_rate_limit_enabled:
                fps_text += " [UNCAPPED]"
            cv2.putText(image, fps_text, (5, img_h - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            return image, {}

        # Reset finger positions (will be updated if hands detected)
        self._right_index_tip = None
        self._left_index_tip = None

        # Get setting
        flip_h = self.setting.camera.flip_horizontal
        swap_hands = self.setting.camera.swap_hands

        # Run MediaPipe detection EVERY frame (for smooth landmarks & data collection)
        # Use small frame for MediaPipe if provided (faster)
        mp_frame = frame_small if frame_small is not None else frame
        # Convert BGR to RGB for MediaPipe (in-place when possible)
        # cv2.cvtColor with dst parameter avoids allocation
        image_rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        # Note: don't need to set writeable back since we don't modify image_rgb

        # Use original frame for drawing (no unnecessary conversion)
        image = frame

        detections = {}

        # Extract keypoints FIRST using spatial hand tracker
        # This populates the tracker with stable hand labels before we draw
        right_kp, left_kp = self._extract_raw_keypoints(results, flip_h, swap_hands)

        # Draw landmarks and labels using tracker's stable assignments
        if results.multi_hand_landmarks and not disable_drawing:
            # Get stable hand labels from tracker
            hand_labels = self.hand_tracker.get_hand_labels(swap_hands)

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Draw stable labels at wrist positions (scaled for small frame)
            # The tracker knows which hand is which by centroid matching
            for _, (label, centroid) in hand_labels.items():
                h, w, _ = image.shape
                wx, wy = int(centroid[0] * w), int(centroid[1] * h)
                color = (255, 100, 100) if label == "Right" else (100, 100, 255)
                cv2.putText(image, label[0], (wx - 8, wy + 15),  # Just first letter "R" or "L"
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Store finger tip positions using tracker's stable assignments
        right_tip, left_tip = self.hand_tracker.get_finger_tips(swap_hands)
        curr_time = time.time()

        if right_tip is not None:
            smoothed_x = self._right_tip_filter_x(curr_time, right_tip[0])
            smoothed_y = self._right_tip_filter_y(curr_time, right_tip[1])
            self._right_index_tip = (smoothed_x, smoothed_y)
            # Add to cache for stable touch detection
            self._right_tip_cache.append((smoothed_x, smoothed_y))

        if left_tip is not None:
            smoothed_x = self._left_tip_filter_x(curr_time, left_tip[0])
            smoothed_y = self._left_tip_filter_y(curr_time, left_tip[1])
            self._left_index_tip = (smoothed_x, smoothed_y)
            # Add to cache for stable touch detection
            self._left_tip_cache.append((smoothed_x, smoothed_y))

        # Update cursor EVERY frame for smooth touch_hover/touch_hold movement
        # This runs regardless of whether gesture model runs this frame
        self._update_continuous_cursor()

        # Adaptive frame-based sampling for sequence collection
        # - Processes every frame through MediaPipe (smooth landmarks)
        # - Samples frames for gesture model at target FPS rate
        # - Never waits - uses all available CPU cycles
        # - Adapts to actual FPS: if slow, use every frame; if fast, skip to hit target
        current_time = time.time()

        # Update actual FPS measurement (rolling average)
        self._fps_sample_times.append(current_time)
        if len(self._fps_sample_times) >= 2:
            time_span = self._fps_sample_times[-1] - self._fps_sample_times[0]
            if time_span > 0:
                self._actual_fps = (len(self._fps_sample_times) - 1) / time_span

        # Determine if we should collect this frame using adaptive sampling
        if self._data_rate_limit_enabled:
            # Calculate skip rate: how many frames to skip to hit target FPS
            # If actual_fps=30 and target=20: skip_rate=1.5 (collect every 1.5 frames)
            # If actual_fps=15 and target=20: skip_rate=0.75 (collect every frame)
            if self._actual_fps > 0:
                skip_rate = self._actual_fps / self._target_fps
            else:
                skip_rate = 1.0

            # Accumulate frames
            self._frame_accumulator += 1.0

            # Collect when accumulator reaches skip rate threshold
            if self._frame_accumulator >= skip_rate:
                should_collect = True
                self._frame_accumulator -= skip_rate
                # Cap accumulator to prevent overflow after pause
                if self._frame_accumulator > skip_rate * 2:
                    self._frame_accumulator = 0.0
            else:
                should_collect = False

            # Always collect if FPS is at or below target (don't lose data)
            if self._actual_fps <= self._target_fps:
                should_collect = True
                self._frame_accumulator = 0.0  # Reset accumulator

            missed_frames = 0  # No interpolation needed with this approach
        else:
            should_collect = True
            missed_frames = 0

        if should_collect:
            collected = False

            # Right hand collection with interpolation
            if right_kp is not None and np.any(right_kp):
                # Interpolate missing frames if we have previous keypoints
                if self._interpolation_enabled and missed_frames > 0 and self._last_right_kp is not None:
                    for i in range(1, missed_frames + 1):
                        # Linear interpolation: lerp from last to current
                        t = i / (missed_frames + 1)
                        interp_kp = self._last_right_kp * (1 - t) + right_kp * t
                        self.right_sequence.append(interp_kp)
                    self.right_sequence = self.right_sequence[-self.sequence_length:]

                # Add current frame
                self.right_sequence.append(right_kp)
                self.right_sequence = self.right_sequence[-self.sequence_length:]
                self._last_right_kp = right_kp.copy()  # Store for next interpolation
                self.right_lock = True
                collected = True

            # Left hand collection with interpolation
            if left_kp is not None and np.any(left_kp):
                # Interpolate missing frames if we have previous keypoints
                if self._interpolation_enabled and missed_frames > 0 and self._last_left_kp is not None:
                    for i in range(1, missed_frames + 1):
                        t = i / (missed_frames + 1)
                        interp_kp = self._last_left_kp * (1 - t) + left_kp * t
                        self.left_sequence.append(interp_kp)
                    self.left_sequence = self.left_sequence[-self.sequence_length:]

                # Add current frame
                self.left_sequence.append(left_kp)
                self.left_sequence = self.left_sequence[-self.sequence_length:]
                self._last_left_kp = left_kp.copy()  # Store for next interpolation
                self.left_lock = True
                collected = True

            if collected:
                self._data_fps_counter += 1  # Count collected frames

        # Right hand prediction - only run TCN model every N frames
        if run_gesture_model and self.right_lock and len(self.right_sequence) == self.sequence_length:
            if self.interpreter:
                seq_array = np.array(self.right_sequence)  # (seq_len, 84)
                # Pass delta_time for FPS-invariant velocity features
                features = self.feature_engineer.transform(seq_array, delta_time=self._last_delta_time)

                gesture, confidence, self.res_right = self._predict('Right', features)
                detections['Right'] = {'gesture': gesture, 'confidence': confidence}

                if not self.actions_disabled:
                    self._handle_gesture('Right', gesture, confidence)

        # Left hand prediction - only run TCN model every N frames
        if run_gesture_model and self.left_lock and len(self.left_sequence) == self.sequence_length:
            if self.interpreter:
                seq_array = np.array(self.left_sequence)
                # Pass delta_time for FPS-invariant velocity features
                features = self.feature_engineer.transform(seq_array, delta_time=self._last_delta_time)

                gesture, confidence, self.res_left = self._predict('Left', features)
                detections['Left'] = {'gesture': gesture, 'confidence': confidence}

                if not self.actions_disabled:
                    self._handle_gesture('Left', gesture, confidence)

        # Draw probability bars (before resetting locks) - skip if drawing disabled
        if not disable_drawing:
            image = self._draw_prob_bars(image, self.right_lock, self.left_lock)

            # Draw gesture history
            image = self._draw_history(image)

        # Draw FPS - always show (for performance testing)
        # Shows: Data FPS (sampled for model) / Actual FPS (camera frames) / Target FPS
        img_h, img_w, _ = image.shape
        fps_text = f"FPS: {self._data_fps:.0f}/{self._actual_fps:.0f}/{self._target_fps:.0f}"
        if not self._data_rate_limit_enabled:
            fps_text += " [UNCAPPED]"
        cv2.putText(image, fps_text, (5, img_h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # Reset locks
        self.right_lock = False
        self.left_lock = False

        # Add index tips to detections dict using tracker's stable assignments
        if 'Right' in detections and right_tip is not None:
            detections['Right']['index_tip'] = (right_tip[0], right_tip[1], 0.0)
        if 'Left' in detections and left_tip is not None:
            detections['Left']['index_tip'] = (left_tip[0], left_tip[1], 0.0)

        # Add raw landmark arrays for knuckle macropad (21 landmarks x,y,z,vis)
        if right_kp is not None:
            detections.setdefault('Right', {})['keypoints'] = right_kp
        if left_kp is not None:
            detections.setdefault('Left', {})['keypoints'] = left_kp

        return image, detections

    def _predict(self, hand: str, features: np.ndarray):
        """Run TFLite inference."""
        interpreter = self.interpreter['interpreter']
        input_details = self.interpreter['input']
        output_details = self.interpreter['output']

        # Add batch dimension: (seq_len, 67) -> (1, seq_len, 67)
        input_data = np.expand_dims(features, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]  # Remove batch dim

        # Get prediction
        class_idx = int(np.argmax(output))
        confidence = float(output[class_idx])

        # Smoothing
        if hand == 'Right':
            self.right_predictions.append(class_idx)
            preds = list(self.right_predictions)
        else:
            self.left_predictions.append(class_idx)
            preds = list(self.left_predictions)

        gesture_name = self.gesture_classes[class_idx]

        # Thumb: after thumb_touch, 2 more frames of thumb_touch/thumb_hold = switch
        if self._thumb_awaiting_hold:
            if gesture_name in ("thumb_touch", "thumb_hold") and confidence > self.threshold:
                self._thumb_hold_count += 1
                if self._thumb_hold_count >= 3:
                    self._thumb_awaiting_hold = False
                    self._thumb_hold_count = 0
                    return "thumb_hold", confidence, output
                return "none", 0.0, output
            else:
                self._thumb_hold_count = 0
                self._thumb_awaiting_hold = False

        # Per-gesture stability window
        window = 4 if gesture_name == "thumb_touch" else self.stability_window
        last_preds = preds[-window:]
        if last_preds:
            most_common = Counter(last_preds).most_common(1)[0][0]
            if most_common == class_idx and confidence > self.threshold:
                return gesture_name, confidence, output

        return "none", 0.0, output

    def _handle_gesture(self, hand: str, gesture: str, confidence: float):
        """Execute action for detected gesture."""
        # Always decrement cooldown first
        if hand == 'Right':
            if self.right_cooldown > 0:
                self.right_cooldown -= 1
        else:
            if self.left_cooldown > 0:
                self.left_cooldown -= 1

        # Track last gesture for continuous cursor movement (every frame)
        if hand == 'Right':
            self._last_right_gesture = gesture if confidence >= self.threshold else "none"
        else:
            self._last_left_gesture = gesture if confidence >= self.threshold else "none"

        # "none" gesture: end any drag and skip
        if gesture == "none":
            self._end_drag()
            return

        # Below threshold: end drag immediately and skip
        if confidence < self.threshold:
            self._end_drag()
            return

        # Capture gestures — no longer used for capture (pointyclick triggers it)
        if gesture in ("capture_hold", "capture_touch", "capture_release"):
            return

        # Thumb desktop switch — temporarily disabled
        if gesture in ("thumb_touch", "thumb_hold", "thumb_release"):
            return

        if gesture == "touch_hover":
            # End any active drag when switching to hover
            self._end_drag()
            if hand == 'Right':
                self.right_cooldown = 0
            else:
                self.left_cooldown = 0
            # Cursor movement now handled in _update_continuous_cursor (every frame)
            return
        elif gesture == "touch_hold":
            # Drag gesture - mouse down and move
            if hand == 'Right':
                self.right_cooldown = 0
            else:
                self.left_cooldown = 0
            # Cursor movement now handled in _update_continuous_cursor (every frame)
            return
        else:
            # Any other gesture ends the drag
            self._end_drag()

        # Check cooldown (applies to all gestures including touch)
        # Only trigger action when cooldown expires (gesture is logged)
        if hand == 'Right':
            if self.right_cooldown > 0:
                return  # Still on cooldown, skip
            self.right_cooldown = self.COOLDOWN_FRAMES
        else:
            if self.left_cooldown > 0:
                return  # Still on cooldown, skip
            self.left_cooldown = self.COOLDOWN_FRAMES

        # Log detection (all gestures logged the same way)
        timestamp = time.strftime("%H:%M:%S")
        entry = f"{timestamp} {hand}: {gesture} ({confidence:.2f})"
        self.gesture_display_history.append(entry)
        self.logger.info(f"[Gesture] {entry}")

        # Special action for touch - move cursor and click
        if gesture == "touch":
            self._click_at_touch(hand)
            return

        # Middleclick on right hand starts capture
        if gesture == "middleclick" and hand == "Right":
            self.start_capture()

        # Signal non-touch gesture activation for visual feedback overlay
        # Skip pointyclick — too sensitive for live demo
        if gesture != "pointyclick":
            self._last_activated_gesture = (gesture, hand, time.time())

        # Execute mapped action for other gestures
        mapping_key = f"{hand}_{gesture}"
        actions = self.setting.get_gesture_actions(mapping_key)
        if actions:
            # Filter out "none" actions
            valid_actions = [a for a in actions if a.type != "none"]
            if valid_actions:
                if len(valid_actions) == 1:
                    # Single action - execute directly
                    self.executor.execute(valid_actions[0].type, valid_actions[0].value)
                else:
                    # Multiple actions - execute as sequence with delays
                    self.executor.execute_sequence(valid_actions)

    def start_capture(self):
        """Start capture — called when pointyclick is detected. Records start position."""
        import time as _t
        if _t.time() < self._cap_cooldown_until:
            return
        if self._cap_state != "idle":
            return
        tip = self._right_index_tip
        if tip:
            self._cap_start = tip
            self._cap_end = tip
            self._cap_state = "holding"
            self._cap_complete = False

    def end_capture(self):
        """End capture — called by spacebar."""
        if self._cap_state == "holding":
            if self._right_index_tip:
                self._cap_end = self._right_index_tip
            self._cap_complete = True
            self._cap_state = "idle"
            import time
            self._cap_cooldown_until = time.time() + 2.0



    def get_capture_state(self):
        """Return (start, end, is_holding, is_complete)."""
        is_holding = self._cap_state in ("counting", "holding")
        return self._cap_start, self._cap_end, is_holding, self._cap_complete

    def clear_capture(self):
        """Reset after crop is processed."""
        self._cap_state = "idle"
        self._cap_start = None
        self._cap_end = None
        self._cap_complete = False

    def _handle_thumb_desktop(self, gesture: str):
        """Thumb desktop switch: touch → hold(2 frames) → switch right, release → switch back."""
        if gesture == "thumb_touch":
            if not self._thumb_switched:
                self._thumb_awaiting_hold = True
                self._thumb_hold_count = 0
        elif gesture == "thumb_hold":
            if not self._thumb_switched:
                self._thumb_switched = True
                self.executor.execute("shortcut", "cmd+tab")
        elif gesture == "thumb_release":
            if self._thumb_switched:
                self._thumb_switched = False
                self.executor.execute("shortcut", "cmd+tab")
            self._thumb_awaiting_hold = False
            self._thumb_hold_count = 0

    def _draw_prob_bars(self, image: np.ndarray, right_active: bool, left_active: bool) -> np.ndarray:
        """Draw probability visualization bars — auto-scales to fit any number of classes."""
        h, w, _ = image.shape
        n_classes = len(self.gesture_classes)

        colors = [
            (245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 200, 16),
            (200, 117, 245), (16, 245, 200), (245, 16, 117), (117, 16, 245),
            (16, 200, 117), (200, 245, 16), (117, 16, 200), (245, 16, 200),
            (16, 245, 117), (200, 16, 117), (117, 200, 16), (16, 117, 200),
            (200, 200, 16),
        ]

        # Auto-scale: fit all bars in available height (leave top/bottom margin)
        margin = 4
        top_margin = 20
        available_h = h - top_margin - margin
        bar_spacing = max(available_h // n_classes, 10)
        bar_height = max(bar_spacing - 3, 7)
        font_scale = 0.32 if n_classes > 14 else 0.38
        font_thickness = 1
        bar_max_len = 55
        start_y = top_margin

        # Right hand bars (right side of screen)
        if right_active:
            for num, prob in enumerate(self.res_right):
                if num >= n_classes:
                    break
                bar_length = int(prob * bar_max_len)

                x2 = w - margin
                x1 = x2 - bar_length
                y1 = start_y + num * bar_spacing
                y2 = y1 + bar_height

                color = colors[num % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

                label = self.gesture_classes[num]
                (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = x1 - text_w - 3
                text_y = y1 + bar_height - 2

                cv2.putText(image, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Left hand bars (left side of screen)
        if left_active:
            for num, prob in enumerate(self.res_left):
                if num >= n_classes:
                    break
                bar_length = int(prob * bar_max_len)

                x1 = margin
                x2 = margin + bar_length
                y1 = start_y + num * bar_spacing
                y2 = y1 + bar_height

                color = colors[num % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

                label = self.gesture_classes[num]
                text_x = x2 + 3
                cv2.putText(image, label, (text_x, y1 + bar_height - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return image

    def _draw_history(self, image: np.ndarray) -> np.ndarray:
        """Draw gesture history (scaled for small frame)."""
        h, w, _ = image.shape
        x = w // 2 - 70
        y = 12

        cv2.putText(image, "History:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        for i, entry in enumerate(reversed(list(self.gesture_display_history))):
            if i >= 4:  # Show fewer entries
                break
            y_pos = y + 12 + (i * 11)
            alpha = 1.0 - (i * 0.2)
            color = (int(150 * alpha), int(255 * alpha), int(150 * alpha))
            cv2.putText(image, entry, (x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

        return image

    def close(self):
        """Cleanup resources."""
        self.hands.close()
