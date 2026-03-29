"""Action execution layer: keyboard shortcuts, mouse control, media keys, and file launching."""

import os
import sys
import time
import subprocess
import threading
from typing import Optional, Callable
from dataclasses import dataclass

import pyautogui

# Platform-specific imports
if sys.platform == "darwin":
    try:
        from Quartz import (
            CGEventCreateMouseEvent, kCGEventMouseMoved, kCGMouseButtonLeft,
            CGEventPost, kCGHIDEventTap,
            kCGEventLeftMouseDown, kCGEventLeftMouseUp,
            kCGEventRightMouseDown, kCGEventRightMouseUp, kCGMouseButtonRight
        )
        QUARTZ_AVAILABLE = True
    except ImportError:
        QUARTZ_AVAILABLE = False
else:
    QUARTZ_AVAILABLE = False

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str = ""


class ActionExecutor:
    """Dispatches OS-level actions (keyboard, mouse, media, files) with platform-native APIs where available."""

    SCROLL_INCREMENT = 40
    ZOOM_KEYS = {
        'in': ('command', '='),
        'out': ('command', '-'),
    }

    MEDIA_KEYS = {
        'play_pause': 'playpause',
        'next': 'nexttrack', 
        'previous': 'prevtrack',
        'volume_up': 'volumeup',
        'volume_down': 'volumedown',
        'mute': 'volumemute',
    }
    
    def __init__(self):
        self._last_action_time = 0
        self._action_cooldown = 0.1  # 100ms minimum between repeated actions
        self._action_map = self._build_action_map()
    
    def _build_action_map(self) -> dict:
        """Build mapping of action types to handler methods."""
        return {
            'none': self._action_none,
            'leftclick': self._action_left_click,
            'rightclick': self._action_right_click,
            'doubleclick': self._action_double_click,
            'shortcut': self._action_shortcut,
            'text': self._action_text,
            'file': self._action_open_file,
            'scroll_up': self._action_scroll_up,
            'scroll_down': self._action_scroll_down,
            'zoom_in': self._action_zoom_in,
            'zoom_out': self._action_zoom_out,
            'media_play': lambda v: self._action_media('play_pause'),
            'media_next': lambda v: self._action_media('next'),
            'media_prev': lambda v: self._action_media('previous'),
            'volume_up': lambda v: self._action_volume('up'),
            'volume_down': lambda v: self._action_volume('down'),
            'volume_mute': lambda v: self._action_volume('mute'),
            'screenshot': self._action_screenshot,
            'minimize': self._action_minimize_window,
            'maximize': self._action_maximize_window,
            'desktop_left': lambda v: self._action_switch_desktop('left'),
            'desktop_right': lambda v: self._action_switch_desktop('right'),
        }
    
    def execute(self, action_type: str, value: str = "") -> ActionResult:
        """
        Execute an action by type.
        
        Args:
            action_type: Type of action (shortcut, text, scroll_up, etc.)
            value: Optional value for the action (shortcut keys, text, file path)
        
        Returns:
            ActionResult with success status and message
        """
        # Cooldown check
        current_time = time.time()
        if current_time - self._last_action_time < self._action_cooldown:
            return ActionResult(False, "Cooldown active")
        
        handler = self._action_map.get(action_type.lower())
        if handler is None:
            return ActionResult(False, f"Unknown action type: {action_type}")
        
        try:
            result = handler(value)
            self._last_action_time = current_time
            return result if result else ActionResult(True)
        except Exception as e:
            return ActionResult(False, str(e))
    
    def execute_sequence(self, actions: list, skip_first_delay: bool = True) -> ActionResult:
        """
        Execute a sequence of actions with configurable delays.
        
        Args:
            actions: List of ActionBinding objects with type, value, and delay
            skip_first_delay: If True, skip the delay for the first action
        
        Returns:
            ActionResult summarizing the sequence execution
        """
        if not actions:
            return ActionResult(False, "No actions to execute")
        
        def _run_sequence():
            for i, action in enumerate(actions):
                # Apply delay (skip for first action if configured)
                if i > 0 or not skip_first_delay:
                    delay = getattr(action, 'delay', 0.17)
                    if delay > 0:
                        time.sleep(delay)
                
                # Execute the action
                action_type = getattr(action, 'type', 'none')
                action_value = getattr(action, 'value', '')
                self.execute(action_type, action_value)
        
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=_run_sequence, daemon=True)
        thread.start()
        
        return ActionResult(True, f"Executing {len(actions)} action(s)")
    
    # -------------------------------------------------
    #Click Actions (Platform-optimized)
    # -------------------------------------------------
    def _action_none(self, value: str) -> ActionResult:
        """No action."""
        return ActionResult(True, "No action")
    
    def _action_left_click(self, value: str) -> ActionResult:
        """Left mouse click using native API for lowest latency."""
        if QUARTZ_AVAILABLE:
            pos = pyautogui.position()
            down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, pos, kCGMouseButtonLeft)
            up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, pos, kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, down)
            CGEventPost(kCGHIDEventTap, up)
        else:
            pyautogui.click()
        return ActionResult(True, "Left click")
    
    def _action_right_click(self, value: str) -> ActionResult:
        """Right mouse click using native API."""
        if QUARTZ_AVAILABLE:
            pos = pyautogui.position()
            down = CGEventCreateMouseEvent(None, kCGEventRightMouseDown, pos, kCGMouseButtonRight)
            up = CGEventCreateMouseEvent(None, kCGEventRightMouseUp, pos, kCGMouseButtonRight)
            CGEventPost(kCGHIDEventTap, down)
            CGEventPost(kCGHIDEventTap, up)
        else:
            pyautogui.rightClick()
        return ActionResult(True, "Right click")
    
    def _action_double_click(self, value: str) -> ActionResult:
        """Double left click."""
        pyautogui.doubleClick()
        return ActionResult(True, "Double click")
    
    # -------------------------------------------------
    # Keyboard Actions
    # -------------------------------------------------
    def _action_shortcut(self, value: str) -> ActionResult:
        """
        Execute keyboard shortcut.

        Value format: "cmd+shift+s" or "ctrl+c" etc.
        Handles modifier key translation for cross-platform.
        """

        KEY_TRANSLATE = {
            "meta_l": "command",
            "meta_r": "command",
            "control_l": "ctrl",
            "control_r": "ctrl",
            "shift_l": "shift",
            "shift_r": "shift",
            "alt_l": "alt",
            "alt_r": "alt",
            "cmd": "command",
            "command": "command",
            "control": "ctrl",
            "grave": "`",
        }

        if not value:
            return ActionResult(False, "No shortcut specified")

        # Parse keys
        raw_keys = value.lower().split("+")
        keys = []

        for k in raw_keys:
            k = k.strip()
            if not k:
                continue

            # Translate key if needed
            translated = KEY_TRANSLATE.get(k, k)
            keys.append(translated)

        if not keys:
            return ActionResult(False, "Invalid shortcut format")

        try:
            pyautogui.hotkey(*keys)
            return ActionResult(True, f"Shortcut: {value}")
        except Exception as e:
            return ActionResult(False, f"Shortcut error: {e}")
            

    def _action_text(self, value: str) -> ActionResult:
        """
        Type text instantly using clipboard paste.

        Always uses clipboard for consistent, instant text input.
        """
        if not value:
            return ActionResult(False, "No text specified")

        try:
            import pyperclip

            # Save current clipboard content
            try:
                old_clipboard = pyperclip.paste()
            except:
                old_clipboard = None

            # Copy new text to clipboard
            pyperclip.copy(value)

            # Small delay to ensure clipboard is ready
            time.sleep(0.05)

            # Paste using system shortcut
            if sys.platform == "darwin":
                pyautogui.hotkey("command", "v")
            else:
                pyautogui.hotkey("ctrl", "v")

            # Small delay after paste
            time.sleep(0.05)

            # Restore old clipboard
            if old_clipboard is not None:
                time.sleep(0.1)  # Wait for paste to complete
                try:
                    pyperclip.copy(old_clipboard)
                except:
                    pass

            preview = value[:20].replace('\n', '↵')
            return ActionResult(True, f"Pasted: {preview}...")

        except ImportError:
            return ActionResult(False, "pyperclip not installed")
        except Exception as e:
            return ActionResult(False, f"Text error: {e}")


    
    # -------------------------------------------------
    # File/App Actions
    # -------------------------------------------------
    def _action_open_file(self, value: str) -> ActionResult:
        """
        Open file or application.
        
        Platform-aware with proper error handling.
        """
        if not value:
            return ActionResult(False, "No file path specified")
        
        if not os.path.exists(value):
            return ActionResult(False, f"File not found: {value}")
        
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", value], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform == "win32":
                os.startfile(value)
            else:
                subprocess.Popen(["xdg-open", value], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ActionResult(True, f"Opened: {os.path.basename(value)}")
        except Exception as e:
            return ActionResult(False, f"Open error: {e}")
    
    # -------------------------------------------------
    # Scroll Actions (Single Increment)
    # -------------------------------------------------
    def _action_scroll_up(self, value: str) -> ActionResult:
        """Single increment scroll up."""
        amount = int(value) if value.isdigit() else self.SCROLL_INCREMENT
        pyautogui.scroll(amount)  # Positive = scroll up
        return ActionResult(True, f"Scroll up {amount}px")
    
    def _action_scroll_down(self, value: str) -> ActionResult:
        """Single increment scroll down."""
        amount = int(value) if value.isdigit() else self.SCROLL_INCREMENT
        pyautogui.scroll(-amount)  # Negative = scroll down
        return ActionResult(True, f"Scroll down {amount}px")
    
    # -------------------------------------------------
    # Zoom Actions (Single Keystroke)
    # -------------------------------------------------
    def _action_zoom_in(self, value: str) -> ActionResult:
        """Single zoom in keystroke (Cmd+=)."""
        pyautogui.hotkey(*self.ZOOM_KEYS['in'])
        return ActionResult(True, "Zoom in")
    
    def _action_zoom_out(self, value: str) -> ActionResult:
        """Single zoom out keystroke (Cmd+-)."""
        pyautogui.hotkey(*self.ZOOM_KEYS['out'])
        return ActionResult(True, "Zoom out")
    
    # -------------------------------------------------
    # Media Controls
    # -------------------------------------------------
    def _action_media(self, media_type: str) -> ActionResult:
        """Media control actions."""
        key = self.MEDIA_KEYS.get(media_type)
        if key:
            pyautogui.press(key)
            return ActionResult(True, f"Media: {media_type}")
        return ActionResult(False, f"Unknown media type: {media_type}")
    
    def _action_volume(self, direction: str) -> ActionResult:
        """Volume control."""
        if direction == 'up':
            pyautogui.press('volumeup')
        elif direction == 'down':
            pyautogui.press('volumedown')
        elif direction == 'mute':
            pyautogui.press('volumemute')
        else:
            return ActionResult(False, f"Unknown volume direction: {direction}")
        return ActionResult(True, f"Volume {direction}")
    
    # -------------------------------------------------
    # Window/Desktop Actions
    # -------------------------------------------------
    def _action_screenshot(self, value: str) -> ActionResult:
        """Take screenshot (macOS: Cmd+Shift+4 for region)."""
        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'shift', '4')
        else:
            pyautogui.press('printscreen')
        return ActionResult(True, "Screenshot")
    
    def _action_minimize_window(self, value: str) -> ActionResult:
        """Minimize current window."""
        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'm')
        else:
            pyautogui.hotkey('win', 'down')
        return ActionResult(True, "Minimize")
    
    def _action_maximize_window(self, value: str) -> ActionResult:
        """Maximize/fullscreen current window."""
        if sys.platform == "darwin":
            pyautogui.hotkey('command', 'control', 'f')
        else:
            pyautogui.hotkey('win', 'up')
        return ActionResult(True, "Maximize")
    
    def _action_switch_desktop(self, direction: str) -> ActionResult:
        """Switch virtual desktop."""
        if sys.platform == "darwin":
            if direction == 'left':
                pyautogui.hotkey('control', 'left')
            else:
                pyautogui.hotkey('control', 'right')
        else:
            if direction == 'left':
                pyautogui.hotkey('win', 'ctrl', 'left')
            else:
                pyautogui.hotkey('win', 'ctrl', 'right')
        return ActionResult(True, f"Desktop {direction}")
    
    # -------------------------------------------------
    # Utility Methods
    # -------------------------------------------------
    @staticmethod
    def get_available_actions() -> list:
        """Get list of all available action types for UI dropdowns."""
        return [
            ("None", "none"),
            ("Left Click", "leftclick"),
            ("Right Click", "rightclick"),
            ("Double Click", "doubleclick"),
            ("Custom Shortcut", "shortcut"),
            ("Type Text", "text"),
            ("Open File/App", "file"),
            ("Scroll Up", "scroll_up"),
            ("Scroll Down", "scroll_down"),
            ("Zoom In", "zoom_in"),
            ("Zoom Out", "zoom_out"),
            ("Media Play/Pause", "media_play"),
            ("Media Next", "media_next"),
            ("Media Previous", "media_prev"),
            ("Volume Up", "volume_up"),
            ("Volume Down", "volume_down"),
            ("Mute", "volume_mute"),
            ("Screenshot", "screenshot"),
            ("Minimize Window", "minimize"),
            ("Maximize Window", "maximize"),
            ("Desktop Left", "desktop_left"),
            ("Desktop Right", "desktop_right"),
        ]
    
    @staticmethod
    def move_cursor(x: int, y: int, native: bool = True):
        """
        Move cursor to position using native API if available.
        
        Args:
            x, y: Screen coordinates
            native: Use native API for smoother movement
        """
        if native and QUARTZ_AVAILABLE:
            event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event)
        else:
            pyautogui.moveTo(x, y)
    
    @staticmethod
    def click_at(x: int, y: int):
        """Move to position and click."""
        ActionExecutor.move_cursor(x, y)
        time.sleep(0.01)  # Small delay for cursor to settle
        if QUARTZ_AVAILABLE:
            down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
            up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, down)
            CGEventPost(kCGHIDEventTap, up)
        else:
            pyautogui.click(x, y)

    @staticmethod
    def mouse_down(x: int, y: int):
        """Press and hold mouse button at position (for drag start)."""
        ActionExecutor.move_cursor(x, y)
        time.sleep(0.01)  # Small delay for cursor to settle
        if QUARTZ_AVAILABLE:
            down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, down)
        else:
            pyautogui.mouseDown(x, y)

    @staticmethod
    def mouse_up(x: int, y: int):
        """Release mouse button at position (for drag end)."""
        if QUARTZ_AVAILABLE:
            up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, up)
        else:
            pyautogui.mouseUp(x, y)

    @staticmethod
    def drag_move(x: int, y: int):
        """Move cursor while mouse button is held (for dragging)."""
        if QUARTZ_AVAILABLE:
            # Use left mouse dragged event for proper drag behavior
            from Quartz import kCGEventLeftMouseDragged
            event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDragged, (x, y), kCGMouseButtonLeft)
            CGEventPost(kCGHIDEventTap, event)
        else:
            pyautogui.moveTo(x, y)
