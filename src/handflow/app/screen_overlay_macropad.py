"""Displays ArUco markers on screen for camera-based macropad detection (Set ID 20)."""

import cv2
import numpy as np
import tkinter as tk
import sys
import time
import threading
from typing import Optional, Dict, List
from PIL import Image, ImageTk

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from handflow.utils import Setting, get_logger
from handflow.actions import ActionExecutor


# Screen overlay uses Set ID 20 (different from paper sets 12, 13, 14)
SCREEN_OVERLAY_SET_ID = 20

# Screen overlay uses DIFFERENT marker IDs from paper macropad
SCREEN_OVERLAY_MARKER_IDS = {
    'TL': 20, 'TR': 21, 'ML': 22, 'MR': 23,
    'BL': 24, 'BR': 25, 'BL2': 26, 'BR2': 27,
}


class ScreenOverlayMacroPad:
    """
    Displays ArUco markers on screen for camera-based detection.
    """

    MARKER_DICT = cv2.aruco.DICT_4X4_50

    # Overlay sizing - large for better marker visibility
    OVERLAY_WIDTH_RATIO = 0.85   # 85% of screen width
    OVERLAY_HEIGHT_RATIO = 0.80  # 80% of screen height

    # Margin around content for better marker detection
    MARGIN = 40

    # Colors - matte gray reduces glare, good contrast with black markers
    BG_COLOR = "#D0D0D0"  # Matte gray background
    BUTTON_COLOR = "#C8C8C8"
    BUTTON_HOVER_COLOR = "#4a90d9"
    BUTTON_BORDER_COLOR = "#888888"
    TEXT_COLOR = "#333333"
    HOVER_TEXT_COLOR = "#ffffff"

    # Grid: 4 columns x 3 rows = 12 buttons
    GRID_COLS = 4
    GRID_ROWS = 3

    def __init__(self, setting: Setting, executor: ActionExecutor):
        self.logger = get_logger("handflow.screen_overlay")
        self.setting = setting
        self.executor = executor

        # Screen dimensions
        if PYAUTOGUI_AVAILABLE:
            self._screen_width, self._screen_height = pyautogui.size()
        else:
            self._screen_width, self._screen_height = 1920, 1080

        # Overlay dimensions
        self._overlay_width = int(self._screen_width * self.OVERLAY_WIDTH_RATIO)
        self._overlay_height = int(self._screen_height * self.OVERLAY_HEIGHT_RATIO)
        self._overlay_x = (self._screen_width - self._overlay_width) // 2
        self._overlay_y = (self._screen_height - self._overlay_height) // 2

        # Content area (inside margin)
        self._content_x = self.MARGIN
        self._content_y = self.MARGIN
        self._content_width = self._overlay_width - 2 * self.MARGIN
        self._content_height = self._overlay_height - 2 * self.MARGIN

        # State
        self._visible = False
        self._cooldown_until = 0.0
        self._hide_grace_counter = 0
        self._hide_grace_max = 5
        self._hovered_button: Optional[int] = None

        # Button labels
        self._button_labels: List[str] = [f"Button {i+1}" for i in range(12)]

        # Tkinter
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._window_created = False

        # Marker images
        self._marker_pil_images: Dict[int, Image.Image] = {}
        self._marker_tk_images: Dict[int, ImageTk.PhotoImage] = {}

        # Thread lock
        self._lock = threading.Lock()

        # Off-screen position for hiding (avoids withdraw/deiconify space switching)
        self._offscreen_x = self._screen_width + 50
        self._offscreen_y = self._overlay_y

        # Calculate layout
        self._calculate_layout()
        self._load_button_labels()

        self.logger.info(f"[ScreenOverlay] Init: {self._overlay_width}x{self._overlay_height}, "
                        f"marker_size={self._marker_size}px")

    def _load_button_labels(self):
        """Load button labels from screen overlay macropad set."""
        overlay_set = self.setting.get_screen_overlay_macropad()
        if overlay_set:
            for i in range(12):
                btn = overlay_set.buttons.get(i)
                if btn and btn.label:
                    self._button_labels[i] = btn.label
                else:
                    self._button_labels[i] = f"Button {i+1}"

    def _calculate_layout(self):
        """
        Calculate marker positions ensuring all fit within the window.
        Uses width-based marker sizing for square buttons in 4x3 grid.

        Layout (same as paper macropad):
            [TL]              [TR]
               |              |
            [ML]              [MR]
               |              |
            [BL2]            [BR2]
               \\            /
               [BL]      [BR]

        BL's top-left corner touches BL2's bottom-right corner.
        BR's top-right corner touches BR2's bottom-left corner.
        """
        w = self._content_width
        h = self._content_height

        # Calculate marker size for square buttons in 4x3 grid
        # For square buttons: grid_width/4 = grid_height/3
        # With grid_width = w - 2*ms and grid_height = 4*ms (detection region)
        # Detection region height = 3*ms + 2*gap, needs to equal 4*ms
        # So gap = 0.5*ms
        # Solving: (w - 2*ms)/4 = 4*ms/3 => ms = 3*w/22
        self._marker_size = int((3 * w) / 22)
        ms = self._marker_size

        # Ensure minimum marker size for detection
        min_marker = 100
        if ms < min_marker:
            ms = min_marker
            self._marker_size = ms

        # Gap between marker rows (TL-ML and ML-BL2)
        # gap = 0.5*ms ensures detection region height = 4*ms for square buttons
        gap = int(ms * 0.5)

        # X positions
        left_x = self._content_x
        right_x = self._content_x + w - ms

        # Y positions
        top_y = self._content_y
        mid_y = top_y + ms + gap
        bottom2_y = mid_y + ms + gap

        # BL/BR Y position: directly below BL2/BR2 (corners touching)
        bottom_y = bottom2_y + ms  # BL/BR top edge at BL2/BR2 bottom edge

        # Ensure everything fits in height
        if bottom_y + ms > self._content_y + h:
            # Scale down to fit height if needed
            available = h
            # Need: 4*ms + 2*gap, where gap = 0.5*ms -> 4*ms + ms = 5*ms
            ms = int(available / 5)
            self._marker_size = ms
            gap = int(ms * 0.5)
            mid_y = top_y + ms + gap
            bottom2_y = mid_y + ms + gap
            bottom_y = bottom2_y + ms

        # BL position: top-left corner at BL2's bottom-right corner
        # BL2 is at (left_x, bottom2_y), its bottom-right is (left_x + ms, bottom2_y + ms)
        # So BL top-left should be at (left_x + ms, bottom2_y + ms)
        bl_x = left_x + ms
        bl_y = bottom2_y + ms  # = bottom_y

        # BR position: top-right corner at BR2's bottom-left corner
        # BR2 is at (right_x, bottom2_y), its bottom-left is (right_x, bottom2_y + ms)
        # BR's top-right is at (br_x + ms, br_y)
        # So br_x + ms = right_x -> br_x = right_x - ms
        br_x = right_x - ms
        br_y = bottom2_y + ms  # = bottom_y

        # Marker positions: (marker_id, x, y)
        self._marker_positions = [
            (SCREEN_OVERLAY_MARKER_IDS['TL'], left_x, top_y),
            (SCREEN_OVERLAY_MARKER_IDS['TR'], right_x, top_y),
            (SCREEN_OVERLAY_MARKER_IDS['ML'], left_x, mid_y),
            (SCREEN_OVERLAY_MARKER_IDS['MR'], right_x, mid_y),
            (SCREEN_OVERLAY_MARKER_IDS['BL2'], left_x, bottom2_y),
            (SCREEN_OVERLAY_MARKER_IDS['BR2'], right_x, bottom2_y),
            (SCREEN_OVERLAY_MARKER_IDS['BL'], bl_x, bl_y),
            (SCREEN_OVERLAY_MARKER_IDS['BR'], br_x, br_y),
        ]

        # Button grid region - matches detection region (from TL's inner corner to BL2/BR2's outer corner)
        # In screen coords (Y down): top_y is TL top, bottom2_y + ms is BL2 bottom
        self._grid_x = left_x + ms
        self._grid_y = top_y
        self._grid_width = right_x - self._grid_x
        self._grid_height = (bottom2_y + ms) - top_y  # From TL top to BL2 bottom

        # Cell size
        self._cell_width = self._grid_width / self.GRID_COLS
        self._cell_height = self._grid_height / self.GRID_ROWS

        self.logger.info(f"[ScreenOverlay] Layout: marker={ms}px, grid={self._grid_width:.0f}x{self._grid_height:.0f}")

    def _generate_markers(self):
        """Generate ArUco marker images."""
        if self._marker_pil_images:
            return

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.MARKER_DICT)
        for name, mid in SCREEN_OVERLAY_MARKER_IDS.items():
            img = cv2.aruco.generateImageMarker(aruco_dict, mid, self._marker_size)
            self._marker_pil_images[mid] = Image.fromarray(img)

    def _create_window(self):
        """Create overlay window - starts off-screen to avoid space switching."""
        if self._window_created:
            return

        try:
            self._root = tk.Tk()
            self._root.overrideredirect(True)

            # Start OFF-SCREEN (not withdraw - avoids space switching on macOS)
            self._root.geometry(f"{self._overlay_width}x{self._overlay_height}+"
                               f"{self._offscreen_x}+{self._offscreen_y}")

            # Background
            self._root.config(bg=self.BG_COLOR)
            self._canvas = tk.Canvas(
                self._root,
                width=self._overlay_width,
                height=self._overlay_height,
                highlightthickness=0,
                bg=self.BG_COLOR
            )
            self._canvas.pack(fill='both', expand=True)

            # Generate markers
            self._generate_markers()
            self._marker_tk_images.clear()
            for mid, pil_img in self._marker_pil_images.items():
                self._marker_tk_images[mid] = ImageTk.PhotoImage(pil_img, master=self._root)

            # CRITICAL: Set topmost so window stays on top of all apps
            self._root.attributes('-topmost', True)

            # Show window (but it's off-screen so invisible)
            self._root.deiconify()
            self._root.lift()

            self._window_created = True
            self.logger.info(f"[ScreenOverlay] Window created off-screen at ({self._offscreen_x}, {self._offscreen_y})")

        except Exception as e:
            self.logger.error(f"[ScreenOverlay] Window creation failed: {e}")
            self._window_created = False

    def _draw(self):
        """Draw the overlay content."""
        if not self._canvas:
            return

        self._canvas.delete('all')

        # Background
        self._canvas.create_rectangle(
            0, 0, self._overlay_width, self._overlay_height,
            fill=self.BG_COLOR, outline=""
        )

        # Draw button grid
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                idx = row * self.GRID_COLS + col

                x1 = self._grid_x + col * self._cell_width
                y1 = self._grid_y + row * self._cell_height
                x2 = x1 + self._cell_width
                y2 = y1 + self._cell_height

                is_hovered = (idx == self._hovered_button)
                fill_color = self.BUTTON_HOVER_COLOR if is_hovered else self.BUTTON_COLOR
                text_color = self.HOVER_TEXT_COLOR if is_hovered else self.TEXT_COLOR
                border_width = 3 if is_hovered else 1

                # Button background
                self._canvas.create_rectangle(
                    x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                    fill=fill_color,
                    outline=self.BUTTON_BORDER_COLOR,
                    width=border_width
                )

                # Button label - larger font with text wrapping
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                label = self._button_labels[idx] if idx < len(self._button_labels) else f"Btn {idx+1}"

                # Larger font sizes for better readability
                font_size = 24 if is_hovered else 20
                # Allow text to wrap within button width (with padding)
                button_width = self._cell_width - 16
                self._canvas.create_text(
                    cx, cy, text=label, fill=text_color,
                    font=('Helvetica', font_size, 'bold'),  # Always bold for better visibility
                    width=button_width,  # Text wraps to new line if too long
                    justify='center'
                )

                # Button number
                self._canvas.create_text(
                    x1 + 12, y1 + 12, text=str(idx + 1),
                    fill="#666666" if not is_hovered else "#aaaaaa",
                    font=('Helvetica', 9)
                )

        # Draw markers
        for mid, mx, my in self._marker_positions:
            if mid in self._marker_tk_images:
                self._canvas.create_image(mx, my, image=self._marker_tk_images[mid], anchor='nw')

        # Info text
        self._canvas.create_text(
            self._overlay_width // 2, self._overlay_height - 15,
            text=f"Screen MacroPad (Set {SCREEN_OVERLAY_SET_ID}) - Touch to activate",
            fill="#666666", font=('Helvetica', 10)
        )

    def set_hovered_button(self, button_idx: Optional[int]):
        """Set the currently hovered button for visual feedback."""
        with self._lock:
            if button_idx != self._hovered_button:
                self._hovered_button = button_idx
                if self._visible and self._canvas:
                    self._draw()
                    if self._root:
                        self._root.update_idletasks()

    def show(self):
        """Show the overlay by moving it on-screen (avoids space switching)."""
        with self._lock:
            if self._visible:
                return

            self._load_button_labels()

            if not self._window_created:
                self._create_window()

            if self._root:
                self._hovered_button = None
                self._draw()

                # Ensure topmost is set
                self._root.attributes('-topmost', True)

                # Move window ON-SCREEN (instead of deiconify to avoid space switch)
                self._root.geometry(f"{self._overlay_width}x{self._overlay_height}+"
                                   f"{self._overlay_x}+{self._overlay_y}")
                self._root.update_idletasks()
                self._root.lift()

                self._visible = True
                self._hide_grace_counter = 0
                self.logger.debug(f"Overlay shown at ({self._overlay_x}, {self._overlay_y})")

    def request_hide(self) -> bool:
        """Request hide with grace period."""
        with self._lock:
            if not self._visible:
                return True

            self._hide_grace_counter += 1
            if self._hide_grace_counter >= self._hide_grace_max:
                self._do_hide()
                return True
            return False

    def _do_hide(self):
        """Hide by moving window off-screen (avoids space switching)."""
        if self._root:
            self._root.geometry(f"{self._overlay_width}x{self._overlay_height}+"
                               f"{self._offscreen_x}+{self._offscreen_y}")
            self._root.update_idletasks()
        self._visible = False
        self._hide_grace_counter = 0
        self._hovered_button = None
        self.logger.debug("Overlay hidden")

    def hide(self):
        """Force hide."""
        with self._lock:
            if self._visible:
                self._do_hide()

    def reset_hide_grace(self):
        """Reset grace counter."""
        with self._lock:
            self._hide_grace_counter = 0

    def is_visible(self) -> bool:
        return self._visible

    def is_in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def set_cooldown(self, duration: float = 0.7):
        self._cooldown_until = time.time() + duration

    def update(self):
        """Process Tk events."""
        if self._root:
            try:
                self._root.update()
            except tk.TclError:
                pass

    def destroy(self):
        """Cleanup."""
        try:
            if self._root:
                self._root.destroy()
                self._root = None
            self._window_created = False
            self._marker_pil_images.clear()
            self._marker_tk_images.clear()
        except:
            pass


def get_screen_overlay_set_id() -> int:
    """Get the set ID used by screen overlay."""
    return SCREEN_OVERLAY_SET_ID
