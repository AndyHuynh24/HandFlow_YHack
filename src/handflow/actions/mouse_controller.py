"""Smooth relative mouse control from hand landmarks with adaptive speed, deadzones, and depth sensitivity."""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import pyautogui

if TYPE_CHECKING:
    from handflow.utils.config import Config

pyautogui.FAILSAFE = False


class MouseController:
    def __init__(self, config: Config) -> None:
        self.config = config
        mouse_cfg = config.mouse

        self.smoothing = mouse_cfg.smoothing
        self.base_sensitivity = mouse_cfg.base_sensitivity
        self.poll_interval = mouse_cfg.poll_interval
        self.inner_deadzone = mouse_cfg.inner_deadzone
        self.outer_deadzone = mouse_cfg.outer_deadzone
        self.activation_threshold = mouse_cfg.activation_threshold

        # State
        self.is_moving = False
        self._thread: threading.Thread | None = None
        self._latest_landmark: tuple[float, float, float] | None = None
        self._origin_landmark: tuple[float, float, float] | None = None
        self._origin_mouse: tuple[float, float] | None = None
        self._prev_x: float | None = None
        self._prev_y: float | None = None

        # Smoothing buffers
        self._velocity = 0.0
        self._smooth_buffer: deque = deque(maxlen=5)
        self._z_smooth: float | None = None

        # Platform detection
        self._platform = self._detect_platform()
        self._virtual_bounds = self._compute_display_bounds()

    def _detect_platform(self) -> str:
        """Detect the current platform."""
        if sys.platform == "darwin":
            try:
                from Quartz import CGEventCreateMouseEvent

                return "mac"
            except ImportError:
                pass
        elif sys.platform.startswith("win"):
            return "win"
        return "generic"

    def _compute_display_bounds(self) -> tuple[float, float, float, float]:
        """
        Compute virtual display bounds.

        Returns:
            Tuple of (min_x, min_y, total_width, total_height).
        """
        if self._platform == "mac":
            try:
                from Quartz import CGDisplayBounds, CGGetActiveDisplayList

                max_displays = 16
                err, display_ids, count = CGGetActiveDisplayList(max_displays, None, None)
                if err == 0 and count > 0:
                    bounds = [CGDisplayBounds(display_ids[i]) for i in range(count)]
                    min_x = min(b.origin.x for b in bounds)
                    min_y = min(b.origin.y for b in bounds)
                    max_x = max(b.origin.x + b.size.width for b in bounds)
                    max_y = max(b.origin.y + b.size.height for b in bounds)
                    return (min_x, min_y, max_x - min_x, max_y - min_y)
            except Exception:
                pass

        sw, sh = pyautogui.size()
        return (0, 0, sw, sh)

    def update_position(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Update the target position from hand landmark.

        Args:
            x: Normalized x coordinate (0-1).
            y: Normalized y coordinate (0-1).
            z: Normalized z coordinate (depth).
        """
        self._latest_landmark = (x, y, z)

    def start(self) -> None:
        """Start mouse following."""
        if self.is_moving:
            return

        self.is_moving = True
        self._reset_state()
        self._thread = threading.Thread(target=self._follow_loop, daemon=True)
        self._thread.start()
        print("[MouseController] Started")

    def stop(self) -> None:
        """Stop mouse following."""
        self.is_moving = False
        if self._thread:
            self._thread.join(timeout=0.2)
            self._thread = None
        self._reset_state()
        print("[MouseController] Stopped")

    def _reset_state(self) -> None:
        """Reset internal state."""
        self._origin_landmark = None
        self._origin_mouse = None
        self._prev_x = None
        self._prev_y = None
        self._velocity = 0.0
        self._smooth_buffer.clear()
        self._z_smooth = None
        self._latest_landmark = None

    def _follow_loop(self) -> None:
        """Main loop for mouse following."""
        active = False
        velocity_smooth = 0.25
        exp_high, exp_low = 0.92, 0.25

        while self.is_moving:
            if not self._latest_landmark:
                time.sleep(self.poll_interval)
                continue

            lx, ly, lz = self._latest_landmark

            # Initialize origin
            if self._origin_landmark is None:
                self._origin_landmark = (lx, ly, lz)
                cur = pyautogui.position()
                min_x, min_y, tw, th = self._virtual_bounds
                self._origin_mouse = ((cur.x - min_x) / tw, (cur.y - min_y) / th)
                self._prev_x, self._prev_y = self._origin_mouse
                self._z_smooth = lz
                self._smooth_buffer.append(self._origin_mouse)
                continue

            # Smooth depth
            self._z_smooth = 0.9 * self._z_smooth + 0.1 * lz

            # Calculate delta from origin
            dx = lx - self._origin_landmark[0]
            dy = ly - self._origin_landmark[1]
            dist = math.hypot(dx, dy)

            # Activation check
            if not active and dist > self.activation_threshold:
                active = True
            elif active and dist < self.inner_deadzone:
                active = False

            if not active:
                time.sleep(self.poll_interval)
                continue

            # Apply deadzones
            if dist < self.inner_deadzone:
                dx = dy = 0
            elif dist < self.outer_deadzone:
                factor = (dist - self.inner_deadzone) / (
                    self.outer_deadzone - self.inner_deadzone
                )
                dx *= factor * 0.5
                dy *= factor * 0.5

            # Calculate depth factor (closer = more sensitive)
            z = max(2e-7, min(8e-7, self._z_smooth))
            z_norm = (z - 2e-7) / (8e-7 - 2e-7)
            depth_factor = 0.5 + (2.0 - 0.5) * ((1.0 - z_norm) ** 1.2)

            # Scaling
            scale = (dist**0.5) * 1.4 + 0.001
            move_scale = scale * self.base_sensitivity * depth_factor
            target_x = self._origin_mouse[0] + dx * move_scale
            target_y = self._origin_mouse[1] + dy * move_scale

            # Velocity-based smoothing
            diff = math.hypot(target_x - self._prev_x, target_y - self._prev_y)
            self._velocity = (
                1 - velocity_smooth
            ) * self._velocity + velocity_smooth * diff
            speed_norm = min(self._velocity / 0.008, 1.0)
            alpha = exp_high - (exp_high - exp_low) * speed_norm

            # Apply smoothing
            final_x = self._prev_x + (target_x - self._prev_x) * alpha
            final_y = self._prev_y + (target_y - self._prev_y) * alpha

            # Buffer smoothing
            self._smooth_buffer.append((final_x, final_y))
            avg_x = sum(p[0] for p in self._smooth_buffer) / len(self._smooth_buffer)
            avg_y = sum(p[1] for p in self._smooth_buffer) / len(self._smooth_buffer)

            # Micro-hysteresis
            if (
                abs(avg_x - self._prev_x) < 0.00025
                and abs(avg_y - self._prev_y) < 0.00025
            ):
                avg_x, avg_y = self._prev_x, self._prev_y

            # Move mouse
            self._move_mouse(avg_x, avg_y)
            self._prev_x, self._prev_y = avg_x, avg_y

            time.sleep(self.poll_interval)

    def _move_mouse(self, norm_x: float, norm_y: float) -> None:
        """
        Move mouse to normalized position.

        Args:
            norm_x: Normalized x (0-1).
            norm_y: Normalized y (0-1).
        """
        min_x, min_y, tw, th = self._virtual_bounds
        px = int(min_x + max(0, min(1, norm_x)) * (tw - 1))
        py = int(min_y + max(0, min(1, norm_y)) * (th - 1))

        if self._platform == "mac":
            try:
                from Quartz import (
                    CGEventCreateMouseEvent,
                    CGEventPost,
                    kCGEventMouseMoved,
                    kCGHIDEventTap,
                    kCGMouseButtonLeft,
                )

                event = CGEventCreateMouseEvent(
                    None, kCGEventMouseMoved, (px, py), kCGMouseButtonLeft
                )
                CGEventPost(kCGHIDEventTap, event)
                return
            except Exception:
                pass

        if self._platform == "win":
            try:
                import ctypes

                ctypes.windll.user32.SetCursorPos(px, py)
                return
            except Exception:
                pass

        # Fallback to PyAutoGUI
        pyautogui.moveTo(px, py)

    @property
    def is_active(self) -> bool:
        """Check if mouse control is active."""
        return self.is_moving
