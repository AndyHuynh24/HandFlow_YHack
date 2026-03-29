"""Hover and click feedback overlay for the paper macropad."""

import sys
import tkinter as tk
from typing import Optional, List

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from handflow.utils import get_logger


class PaperMacroPadFeedback:
    """
    Displays visual feedback overlay for paper macropad interactions.
    Shows button label when hovering and click confirmation when activated.
    """

    def __init__(self):
        self.logger = get_logger("handflow.paper_macropad_feedback")

        # Screen dimensions
        if PYAUTOGUI_AVAILABLE:
            self._screen_width, self._screen_height = pyautogui.size()
        else:
            self._screen_width, self._screen_height = 1920, 1080

        # Overlay dimensions - larger size, positioned at right edge
        self._size = 360
        self._margin = 40  # Margin from screen edge
        self._x = self._screen_width - self._size - self._margin  # Right edge
        self._y = 80  # Near top of screen

        # State
        self._current_hover: Optional[int] = None
        self._visible = False

        # Tkinter
        self._root: Optional[tk.Tk] = None
        self._canvas: Optional[tk.Canvas] = None
        self._window_created = False

        # Button labels (will be updated from settings)
        self._button_labels: List[str] = [f"Button {i+1}" for i in range(12)]

        # Auto-hide timer ID
        self._hide_timer_id: Optional[str] = None

        # Click feedback state - prevents hover from interrupting click feedback
        self._showing_click_feedback: bool = False
        self._click_feedback_timer_id: Optional[str] = None

        self.logger.info("[PaperFeedback] Initialized")

    def set_button_labels(self, labels: List[str]):
        """Update button labels from macropad settings."""
        self._button_labels = labels

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
            self.logger.info("[PaperFeedback] Window created")

        except Exception as e:
            self.logger.error(f"[PaperFeedback] Window creation failed: {e}")
            self._window_created = False

    def _draw_hover(self, label: str, button_num: int):
        """Draw hover feedback with elegant styling."""
        if not self._canvas:
            return

        canvas = self._canvas
        size = self._size
        canvas.delete('all')

        # Outer dark border
        canvas.create_rectangle(0, 0, size, size, fill='#1a1a1a', outline='')

        # Main background
        padding = 4
        canvas.create_rectangle(
            padding, padding, size - padding, size - padding,
            fill='#2d2d2d', outline='#3d3d3d', width=1
        )

        # Inner highlight
        inner_pad = 10
        canvas.create_rectangle(
            inner_pad, inner_pad, size - inner_pad, size - inner_pad,
            fill='#363636', outline='#454545', width=1
        )

        # Blue accent line at top
        canvas.create_rectangle(
            inner_pad + 25, inner_pad + 3,
            size - inner_pad - 25, inner_pad + 7,
            fill='#4a90d9', outline=''
        )

        # Button number
        canvas.create_text(
            size // 2, 50,
            text=f"Button {button_num}",
            fill='#888888',
            font=('Helvetica', 18, 'normal')
        )

        # Main label - adaptive font size (larger)
        label_len = len(label)
        if label_len <= 8:
            font_size = 54
        elif label_len <= 12:
            font_size = 46
        elif label_len <= 18:
            font_size = 36
        else:
            font_size = 28

        display_label = label if len(label) <= 25 else label[:24] + "..."

        canvas.create_text(
            size // 2, size // 2 + 15,
            text=display_label,
            fill='#ffffff',
            font=('Helvetica', font_size, 'bold'),
            width=size - 50,
            justify='center'
        )

        # "hovering" hint
        canvas.create_text(
            size // 2, size - 38,
            text="hovering",
            fill='#666666',
            font=('Helvetica', 14, 'italic')
        )

    def _draw_click(self, label: str, button_num: int):
        """Draw click confirmation with green flash."""
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

        # Main label (larger)
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

        # "activated" text
        canvas.create_text(
            size // 2, size - 38,
            text="activated",
            fill='#88ff88',
            font=('Helvetica', 15, 'bold')
        )

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
            self._current_hover = None

    def set_hovered_button(self, button_idx: Optional[int]):
        """Update hover state and show/hide feedback."""
        if not self._window_created:
            self._create_window()

        # Don't interrupt click feedback - it takes priority
        if self._showing_click_feedback:
            return

        if button_idx == self._current_hover:
            return

        # Cancel any pending hide timer
        if self._hide_timer_id and self._root:
            self._root.after_cancel(self._hide_timer_id)
            self._hide_timer_id = None

        self._current_hover = button_idx

        if button_idx is not None:
            label = self._button_labels[button_idx] if button_idx < len(self._button_labels) else f"Button {button_idx + 1}"
            self._draw_hover(label, button_idx + 1)
            self._show()
        else:
            self._hide()

    def show_click_feedback(self, button_idx: int):
        """Show brief click confirmation that persists even after touch ends."""
        if not self._window_created:
            self._create_window()

        if not self._root:
            return

        # Cancel any pending timers
        if self._hide_timer_id:
            self._root.after_cancel(self._hide_timer_id)
            self._hide_timer_id = None
        if self._click_feedback_timer_id:
            self._root.after_cancel(self._click_feedback_timer_id)
            self._click_feedback_timer_id = None

        # Set flag to prevent hover from interrupting
        self._showing_click_feedback = True

        label = self._button_labels[button_idx] if button_idx < len(self._button_labels) else f"Button {button_idx + 1}"
        self._draw_click(label, button_idx + 1)
        self._show()

        # Auto-hide after 400ms (longer duration for visibility)
        self._click_feedback_timer_id = self._root.after(400, self._end_click_feedback)

    def _end_click_feedback(self):
        """End click feedback and hide overlay."""
        self._showing_click_feedback = False
        self._click_feedback_timer_id = None
        self._hide()

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
