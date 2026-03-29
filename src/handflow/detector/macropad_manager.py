"""Manages macro pad sets, touch-based button activation, and action dispatch."""

import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from handflow.utils import Setting
from handflow.detector.macropad_detector import MacroPadDetector
from handflow.actions import ActionExecutor

from handflow.utils import get_logger


@dataclass
class ButtonActivationLog:
    """Record of a button activation for the visual log."""
    timestamp: float
    button_idx: int
    label: str
    action_type: str
    action_value: str


class MacroPadManager:
    """
    Manages ArUco marker-based paper macro pad with multiple named sets.
    
    Touch gesture activates the button under finger tip.
    """
    
    # Activation parameters
    TOUCH_COOLDOWN = 0.7  # Seconds between activations (global cooldown)
    DETECTION_GRACE_FRAMES = 10  # Keep detection valid for N frames after last successful detection
    HOVER_MEMORY_FRAMES = 5  # Remember hovered button for N frames

    def __init__(
        self,
        setting: Setting,
        executor: ActionExecutor,
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
        detection_mode: str = "balanced"
    ):
        """
        Initialize macro pad manager.

        Args:
            setting: Setting manager for accessing macro pad sets
            executor: Action executor for triggering actions
            aruco_dict: ArUco dictionary (must match printed markers)
            detection_mode: Detection preset for motion blur handling
                - "balanced": Good motion blur tolerance with stable static detection
                - "motion_priority": Maximum motion blur tolerance, may jitter when static
        """
        self.logger = get_logger("handflow.detector.macropad_manager")

        self.setting = setting
        self.executor = executor

        # Detection
        self._detector = MacroPadDetector(aruco_dict, detection_mode=detection_mode)

        # State
        self._last_activation_time: Dict[int, float] = {}
        self._global_last_activation: float = 0.0  # Global cooldown tracker
        self._hovered_button: Optional[int] = None
        self._activated_button: Optional[int] = None

        # Finger state
        self._finger_pos: Optional[Tuple[float, float]] = None
        self._finger_touching = False
        self._activation_this_frame = False  # Prevent double activation in same frame

        # Grace period tracking for robust detection
        self._frames_since_detection = 999  # Frames since last successful detection
        self._last_valid_set_id: Optional[int] = None  # Last detected set ID
        self._last_hovered_button: Optional[int] = None  # Last hovered button (for memory)
        self._hover_memory_frames = 0  # Frames since last hover

        # Button activation log for visual display
        self.ACTIVATION_LOG_SIZE = 5  # Number of recent activations to show
        self._activation_log: deque = deque(maxlen=self.ACTIVATION_LOG_SIZE)
    
    @property
    def active_set(self):
        """Get the currently active macro pad set based on detected marker (with grace period)."""
        # Use current detection or fall back to last valid during grace period
        if self._detector.is_valid:
            detected_id = self._detector.current_set_id
        elif self._frames_since_detection < self.DETECTION_GRACE_FRAMES and self._last_valid_set_id is not None:
            detected_id = self._last_valid_set_id
        else:
            return None

        # Check for screen overlay set (ID 20)
        from handflow.app.screen_overlay_macropad import SCREEN_OVERLAY_SET_ID
        if detected_id == SCREEN_OVERLAY_SET_ID:
            return self.setting.get_screen_overlay_macropad()

        # Check paper macropad sets
        for macro_set in self.setting.macropad_sets:
            if macro_set.set_marker_id == detected_id:
                return macro_set

        return None

    def _get_set_by_id(self, set_id: int):
        """Get a macropad set by its marker ID."""
        from handflow.app.screen_overlay_macropad import SCREEN_OVERLAY_SET_ID
        if set_id == SCREEN_OVERLAY_SET_ID:
            return self.setting.get_screen_overlay_macropad()
        for macro_set in self.setting.macropad_sets:
            if macro_set.set_marker_id == set_id:
                return macro_set
        return None

    def detect_markers(self, frame: np.ndarray, prioritize_screen_overlay: bool = False) -> bool:
        """
        Detect macro pad markers in frame.

        Dynamically reads known set IDs from settings each time,
        allowing detection of any configured set without restart.

        Uses a grace period to maintain detection during brief losses

        Args:
            frame: BGR camera frame
            prioritize_screen_overlay: If True, only look for screen overlay markers (ID 20)
                                       This prevents paper macropad from interfering with screen overlay

        Returns:
            True if macro pad detected (or within grace period)
        """
        # Reset frame-level activation lock
        self._activation_this_frame = False
        from handflow.app.screen_overlay_macropad import SCREEN_OVERLAY_SET_ID

        if prioritize_screen_overlay:
            # Only look for screen overlay when it's visible
            # This prevents paper macropad markers from interfering
            known_set_ids = [SCREEN_OVERLAY_SET_ID]
        else:
            # Dynamically get set IDs from settings (allows runtime changes)
            known_set_ids = [s.set_marker_id for s in self.setting.macropad_sets]
            if not known_set_ids:
                known_set_ids = [12, 13, 14]  # Default fallback

            # Also include screen overlay set ID (20) for screen-based macropad
            if SCREEN_OVERLAY_SET_ID not in known_set_ids:
                known_set_ids.append(SCREEN_OVERLAY_SET_ID)

        detected = self._detector.detect(frame, known_set_ids)

        if detected:
            # Reset grace period counter on successful detection
            self._frames_since_detection = 0
            self._last_valid_set_id = self._detector.current_set_id
        else:
            # Increment counter when detection fails
            self._frames_since_detection += 1

        # Return True if detected OR within grace period
        return detected or self._frames_since_detection < self.DETECTION_GRACE_FRAMES
    
    def update_finger_state(
        self,
        finger_pos: Optional[Tuple[float, float]],
        is_touching: bool,
        skip_activation: bool = False
    ):
        """
        Update finger position and touch state.

        Args:
            finger_pos: (x, y) finger tip position in camera coordinates, or None
            is_touching: True if touch gesture detected
            skip_activation: If True, update hover state but don't trigger button activation
                            (used when screen overlay is handling activation separately)
        """
        self._finger_pos = finger_pos
        was_touching = self._finger_touching
        self._finger_touching = is_touching

        # Clear previous activation state
        self._activated_button = None

        # Update hover state - use detector if valid, or grace period
        current_hover = None
        if finger_pos:
            if self._detector.is_valid:
                current_hover = self._detector.get_button_at_point(finger_pos)
            elif self._frames_since_detection < self.DETECTION_GRACE_FRAMES:
                # During grace period, try to use cached detection region
                if self._detector.detection is not None:
                    current_hover = self._detector.get_button_at_point(finger_pos)

        # Update hover memory
        if current_hover is not None:
            self._hovered_button = current_hover
            self._last_hovered_button = current_hover
            self._hover_memory_frames = 0
        else:
            self._hovered_button = None
            self._hover_memory_frames += 1

        # Determine effective hover for activation (current or recent memory)
        effective_hover = self._hovered_button
        if effective_hover is None and self._hover_memory_frames < self.HOVER_MEMORY_FRAMES:
            effective_hover = self._last_hovered_button

        if is_touching != was_touching:
            self.logger.debug(f"[MacroPad] Touch: {was_touching} -> {is_touching}, hover={self._hovered_button}, effective={effective_hover}")

        # Check for activation
        # Use effective_hover which includes recent memory for robustness
        # Skip activation if screen overlay is handling it separately (prevents double activation)
        if is_touching and not was_touching and effective_hover is not None:
            if skip_activation:
                self.logger.debug(f"[MacroPad] Touch on button {effective_hover}, skipping (screen overlay handling)")
            else:
                self.logger.info(f"[MacroPad] Touch detected on button {effective_hover}!")
                self._activate_button(effective_hover)
    
    def _activate_button(self, button_idx: int, force_set_id: Optional[int] = None):
        """
        Activate a button - execute its bound action(s).

        Args:
            button_idx: Button index (0-11)
            force_set_id: If provided, use this set ID instead of detected set
                          (used when screen overlay needs to ensure correct set)
        """
        if self._activation_this_frame:
            return

        current_time = time.time()

        if current_time - self._global_last_activation < self.TOUCH_COOLDOWN:
            return

        last_activation = self._last_activation_time.get(button_idx, 0)
        if current_time - last_activation < self.TOUCH_COOLDOWN:
            return

        if force_set_id is not None:
            active_set = self._get_set_by_id(force_set_id)
        else:
            active_set = self.active_set

        if not active_set:
            self.logger.warning(f"[MacroPad] No active set for ID {force_set_id or self._detector.current_set_id}")
            return

        button = active_set.buttons.get(button_idx)
        if not button:
            return

        actions = button.get_actions()
        valid_actions = [a for a in actions if a.type != "none"]
        if not valid_actions:
            return

        if len(valid_actions) == 1:
            result = self.executor.execute(valid_actions[0].type, valid_actions[0].value)
        else:
            result = self.executor.execute_sequence(valid_actions)

        if result.success:
            self._activated_button = button_idx
            self._activation_this_frame = True
            self._last_activation_time[button_idx] = current_time
            self._global_last_activation = current_time
            self.logger.info(f"[MacroPad] Activated: {button.label or f'Button {button_idx + 1}'} -> {len(valid_actions)} action(s)")

            action_summary = valid_actions[0].type if len(valid_actions) == 1 else f"{len(valid_actions)} actions"
            self._activation_log.append(ButtonActivationLog(
                timestamp=current_time,
                button_idx=button_idx,
                label=button.label or f"Button {button_idx + 1}",
                action_type=action_summary,
                action_value=valid_actions[0].value if len(valid_actions) == 1 else ""
            ))
        else:
            self.logger.warning(f"[MacroPad] Action failed: {result.message}")
    
    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw debug visualization on frame.

        Shows:
        - Detected markers with ID labels
        - Detection region and 4x3 grid
        - Hover state (yellow highlight)
        - Activation state (green flash)
        - Finger position
        - Grace period indicator
        - Button activation log panel (above macropad)

        Args:
            frame: BGR frame to draw on

        Returns:
            Frame with debug overlay
        """
        active_set = self.active_set
        set_name = active_set.name if active_set else ""

        # Use effective hover for display (includes memory)
        effective_hover = self._hovered_button
        if effective_hover is None and self._hover_memory_frames < self.HOVER_MEMORY_FRAMES:
            effective_hover = self._last_hovered_button

        # Add grace period indicator to set name
        if self._frames_since_detection > 0 and self._frames_since_detection < self.DETECTION_GRACE_FRAMES:
            set_name = f"{set_name} (grace:{self.DETECTION_GRACE_FRAMES - self._frames_since_detection})"

        # Get button names from active set (12 buttons, 0-11)
        button_names = None
        if active_set:
            button_names = [
                active_set.buttons.get(i, {}).label if hasattr(active_set.buttons.get(i), 'label') else None
                for i in range(12)
            ]

        output = self._detector.draw_debug(
            frame,
            finger_pos=self._finger_pos,
            hovered_button=effective_hover,
            activated_button=self._activated_button,
            set_name=set_name,
            button_names=button_names
        )

        # Draw finger position
        if self._finger_pos:
            fx, fy = int(self._finger_pos[0]), int(self._finger_pos[1])
            color = (0, 255, 0) if self._finger_touching else (255, 0, 255)
            cv2.circle(output, (fx, fy), 10, color, -1 if self._finger_touching else 2)

        # Draw activation log panel above the macropad
        output = self._draw_activation_log(output)

        return output

    def _draw_activation_log(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw activation log panel above the detected macropad area.

        Args:
            frame: BGR frame to draw on

        Returns:
            Frame with log panel overlay
        """
        if not self._activation_log:
            return frame

        # Get macropad bounding box to position the log above it
        if self._detector.detection is not None:
            region = self._detector.detection.detection_region
            # Get top-left corner of macropad area
            min_x = int(np.min(region[:, 0]))
            min_y = int(np.min(region[:, 1]))
            max_x = int(np.max(region[:, 0]))
        else:
            # Fallback position if no detection
            min_x = 10
            min_y = 150
            max_x = 300

        # Panel dimensions
        line_height = 22
        padding = 8
        num_entries = len(self._activation_log)
        panel_height = (num_entries * line_height) + (padding * 2) + 20  # +20 for header
        panel_width = max(max_x - min_x, 280)

        # Position panel above macropad (with some margin)
        panel_x = min_x
        panel_y = max(0, min_y - panel_height - 10)

        # Draw solid background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (40, 40, 40),  # Dark gray background
            -1
        )
        # Blend for slight transparency
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),
            1
        )

        # Draw header
        header_y = panel_y + padding + 14
        cv2.putText(
            frame, "Button Log",
            (panel_x + padding, header_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # Draw activation entries (most recent first)
        current_time = time.time()
        for i, entry in enumerate(reversed(list(self._activation_log))):
            entry_y = header_y + 20 + (i * line_height)

            # Calculate time ago
            seconds_ago = current_time - entry.timestamp
            if seconds_ago < 60:
                time_str = f"{int(seconds_ago)}s"
            else:
                time_str = f"{int(seconds_ago / 60)}m"

            # Format: "Label -> action_type (time ago)"
            action_display = entry.action_type
            if entry.action_value:
                # Truncate long values
                val = entry.action_value[:15] + "..." if len(entry.action_value) > 15 else entry.action_value
                action_display = f"{entry.action_type}:{val}"

            text = f"{entry.label} -> {action_display}"

            # Fade older entries
            alpha = max(0.4, 1.0 - (i * 0.15))
            color = (int(180 * alpha), int(255 * alpha), int(180 * alpha))

            cv2.putText(
                frame, text,
                (panel_x + padding, entry_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

            # Time indicator on the right
            cv2.putText(
                frame, time_str,
                (panel_x + panel_width - 35, entry_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1
            )

        return frame
    
    def get_detected_set_id(self) -> Optional[int]:
        """Get the marker ID of the currently detected set."""
        return self._detector.current_set_id
    
    def is_detected(self) -> bool:
        """Check if macro pad is currently detected (including grace period)."""
        return self._detector.is_valid or self._frames_since_detection < self.DETECTION_GRACE_FRAMES
    
    def reset(self):
        """Reset detection state."""
        self._detector.reset()
        self._hovered_button = None
        self._activated_button = None
        self._finger_pos = None
        self._finger_touching = False
        self._frames_since_detection = 999
        self._last_valid_set_id = None
        self._last_hovered_button = None
        self._hover_memory_frames = 0
        self._activation_log.clear()
