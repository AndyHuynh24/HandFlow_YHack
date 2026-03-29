"""Floating overlay that shows gesture name on activation."""

import sys
import tkinter as tk
from typing import Optional

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from handflow.utils import get_logger


class GestureFeedback:
    """
    Displays visual feedback overlay for gesture activations.
    Shows gesture name in macropad-button click style (green flash).
    """

    def __init__(self):
        self.logger = get_logger("handflow.gesture_feedback")

        # Screen dimensions
        if PYAUTOGUI_AVAILABLE:
            self._screen_width, self._screen_height = pyautogui.size()
        else:
            self._screen_width, self._screen_height = 1920, 1080

        # Overlay dimensions - same size as macropad feedback
        self._size = 360
        self._margin = 40
        # Position on the RIGHT edge (same as macropad feedback)
        self._x = self._screen_width - self._size - self._margin
        self._y = 80

        # State
        self._visible = False

        # Tkinter
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._window_created = False

        # Auto-hide timer
        self._hide_timer_id: Optional[str] = None

        self.logger.info("[GestureFeedback] Initialized")

    def _create_window(self):
        """Create the feedback overlay window."""
        if self._window_created:
            return

        try:
            self._root = tk.Tk()
            self._root.overrideredirect(True)

            # Start off-screen
            self._root.geometry(
                f"{self._size}x{self._size}+"
                f"{self._screen_width + 100}+{self._y}"
            )

            self._root.config(bg='#2d2d2d')
            self._root.attributes('-topmost', True)

            # macOS transparency
            if sys.platform == 'darwin':
                self._root.attributes('-alpha', 0.95)

            self._canvas = tk.Canvas(
                self._root,
                width=self._size,
                height=self._size,
                highlightthickness=0,
                bg='#2d2d2d'
            )
            self._canvas.pack(fill='both', expand=True)

            self._root.deiconify()
            self._window_created = True
            self.logger.info("[GestureFeedback] Window created")

        except Exception as e:
            self.logger.error(f"[GestureFeedback] Window creation failed: {e}")
            self._window_created = False

    def _draw_activation(self, gesture_name: str, hand: str):
        """Draw gesture activation feedback with green flash (same as macropad click)."""
        if not self._canvas:
            return

        canvas = self._canvas
        size = self._size
        canvas.delete('all')

        # Green background
        canvas.create_rectangle(0, 0, size, size, fill='#1a3d1a', outline='')

        # Main background with green tint
        padding = 4
        canvas.create_rectangle(
            padding, padding, size - padding, size - padding,
            fill='#224422', outline='#33aa33', width=2
        )

        # Inner area
        inner_pad = 10
        canvas.create_rectangle(
            inner_pad, inner_pad, size - inner_pad, size - inner_pad,
            fill='#2a4a2a', outline='#44bb44', width=1
        )

        # Green accent line
        canvas.create_rectangle(
            inner_pad + 25, inner_pad + 3,
            size - inner_pad - 25, inner_pad + 8,
            fill='#44dd44', outline=''
        )

        # Checkmark
        canvas.create_text(
            size // 2, 55,
            text="\u2713",
            fill='#66ff66',
            font=('Helvetica', 36, 'bold')
        )

        # Gesture name - adaptive font size
        label = gesture_name.replace('_', ' ').title()
        label_len = len(label)
        if label_len <= 8:
            font_size = 50
        elif label_len <= 12:
            font_size = 42
        elif label_len <= 18:
            font_size = 34
        else:
            font_size = 26

        display_label = label if len(label) <= 25 else label[:24] + "..."

        canvas.create_text(
            size // 2, size // 2 + 20,
            text=display_label,
            fill='#ffffff',
            font=('Helvetica', font_size, 'bold'),
            width=size - 50,
            justify='center'
        )

        # "activated" text with hand indicator
        canvas.create_text(
            size // 2, size - 38,
            text=f"activated ({hand})",
            fill='#88ff88',
            font=('Helvetica', 15, 'bold')
        )

    def show_gesture(self, gesture_name: str, hand: str):
        """Show gesture activation feedback."""
        if not self._window_created:
            self._create_window()

        if not self._root:
            return

        # Cancel any pending hide timer
        if self._hide_timer_id:
            self._root.after_cancel(self._hide_timer_id)
            self._hide_timer_id = None

        self._draw_activation(gesture_name, hand)
        self._show()

        # Auto-hide after 400ms (same as macropad click feedback)
        self._hide_timer_id = self._root.after(400, self._hide)

    def _show(self):
        """Move overlay on-screen."""
        if not self._root:
            return

        self._root.geometry(
            f"{self._size}x{self._size}+"
            f"{self._x}+{self._y}"
        )
        self._root.attributes('-topmost', True)
        self._root.lift()
        self._root.update_idletasks()
        self._visible = True

    def _hide(self):
        """Move overlay off-screen."""
        if self._root and self._visible:
            self._root.geometry(
                f"{self._size}x{self._size}+"
                f"{self._screen_width + 100}+{self._y}"
            )
            self._root.update_idletasks()
            self._visible = False
            self._hide_timer_id = None

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
        except:
            pass
