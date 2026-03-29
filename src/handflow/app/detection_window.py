"""Real-time detection loop and camera preview. UI runs on main thread to avoid macOS crashes."""

import cv2
import time
import threading
import datetime
import os
import sys
import queue
import numpy as np
import customtkinter as ctk
from PIL import Image
from typing import Optional, Tuple

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

import numpy as np

from handflow.utils import get_logger


# ============================================================
# macOS App Nap Prevention
# ============================================================
# When the app window loses focus, macOS throttles background
# processes ("App Nap"), causing delayed gesture detection.
# This uses multiple approaches to prevent throttling.

_activity_token = None
_caffeinate_process = None

def disable_app_nap():
    """Disable macOS App Nap and background throttling using multiple methods."""
    global _activity_token, _caffeinate_process
    if sys.platform != 'darwin':
        return

    # Method 1: PyObjC with NSActivityLatencyCritical (strongest flag)
    try:
        from Foundation import NSProcessInfo
        NSActivityLatencyCritical = 0xFF00000000
        NSActivityUserInitiated = 0x00FFFFFF
        activity_options = NSActivityLatencyCritical | NSActivityUserInitiated

        process_info = NSProcessInfo.processInfo()
        _activity_token = process_info.beginActivityWithOptions_reason_(
            activity_options,
            "HandFlow real-time gesture detection"
        )
        # Prevent macOS from auto-terminating or sudden-terminating the process
        process_info.disableAutomaticTermination_("HandFlow detection active")
        process_info.disableSuddenTermination()
    except ImportError:
        pass
    except Exception:
        pass

    # Method 2: ALWAYS run caffeinate alongside PyObjC for extra insurance
    # -d: prevent display sleep, -i: prevent idle sleep,
    # -m: prevent disk sleep, -s: prevent system sleep
    try:
        import subprocess
        _caffeinate_process = subprocess.Popen(
            ['caffeinate', '-dims', '-w', str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def enable_app_nap():
    """Re-enable App Nap (cleanup)."""
    global _activity_token, _caffeinate_process

    if sys.platform != 'darwin':
        return

    # Clean up PyObjC activity token
    if _activity_token is not None:
        try:
            from Foundation import NSProcessInfo
            process_info = NSProcessInfo.processInfo()
            process_info.endActivity_(_activity_token)
            process_info.enableAutomaticTermination_("HandFlow detection active")
            process_info.enableSuddenTermination()
            _activity_token = None
        except:
            pass

    # Clean up caffeinate process
    if _caffeinate_process is not None:
        try:
            _caffeinate_process.terminate()
            _caffeinate_process.wait(timeout=1.0)
            _caffeinate_process = None
        except:
            pass

from handflow.utils import Setting

from handflow.detector import GestureDetector, ArUcoScreenDetector, MacroPadManager
from handflow.actions import ActionExecutor
# Import constant directly to avoid per-frame import overhead
SCREEN_OVERLAY_SET_ID = 20  # From screen_overlay_macropad module


class ESP32Camera:
    """Camera capture from ESP32 over serial, mimicking cv2.VideoCapture interface."""

    FRAME_START = b'\xff\xaa\xbb\xcc'
    FRAME_END = b'\xcc\xbb\xaa\xff'

    def __init__(self, serial_port: str, baud_rate: int = 2000000):
        self._port = serial_port
        self._baud = baud_rate
        self._ser = None
        self._opened = False
        self._buf = b''

    def open(self):
        """Open the serial connection."""
        try:
            import serial
            self._ser = serial.Serial(self._port, self._baud, timeout=2)
            import time
            time.sleep(2)  # Wait for ESP32 to initialize
            self._ser.reset_input_buffer()
            self._opened = True
            self._buf = b''
        except Exception:
            self._opened = False

    def isOpened(self) -> bool:
        return self._opened and self._ser is not None

    def read(self):
        """Read one frame from ESP32 serial. Returns (success, frame) like cv2.VideoCapture."""
        if not self.isOpened():
            return False, None

        try:
            # Find frame start marker
            buf = self._buf if self._buf else self._ser.read(1024)
            attempts = 0
            while attempts < 50:
                pos = buf.find(self.FRAME_START)
                if pos >= 0:
                    buf = buf[pos + 4:]
                    break
                new = self._ser.read(1024)
                if not new:
                    attempts += 1
                    continue
                buf = buf[-3:] + new
                attempts += 1
            else:
                self._buf = b''
                return False, None

            # Read frame size (4 bytes, little-endian)
            while len(buf) < 4:
                chunk = self._ser.read(4 - len(buf))
                if not chunk:
                    self._buf = b''
                    return False, None
                buf += chunk

            frame_size = int.from_bytes(buf[:4], 'little')
            buf = buf[4:]

            if frame_size <= 0 or frame_size > 80000:
                self._ser.reset_input_buffer()
                self._buf = b''
                return False, None

            # Read JPEG data + end marker
            needed = frame_size + 4
            while len(buf) < needed:
                remaining = needed - len(buf)
                chunk = self._ser.read(remaining)
                if not chunk:
                    self._buf = b''
                    return False, None
                buf += chunk

            jpeg_data = buf[:frame_size]
            end_marker = buf[frame_size:frame_size + 4]
            self._buf = buf[frame_size + 4:]

            if end_marker != self.FRAME_END:
                self._ser.reset_input_buffer()
                self._buf = b''
                return False, None

            # Decode JPEG
            img = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if frame is None:
                return False, None

            return True, frame

        except Exception:
            return False, None

    def set(self, prop, value):
        """No-op for compatibility with cv2.VideoCapture.set()."""
        pass

    def release(self):
        """Close the serial connection."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._opened = False


class DetectionWindow(ctk.CTkToplevel):
    """
    Detection window showing camera preview and debug info.
    
    Architecture:
    - Main Thread: Handles UI updates (customtkinter)
    - Background Thread: Handles Camera capture & Computer Vision processing
    """
    
    def __init__(self, setting: Setting, executor: ActionExecutor):
        super().__init__()

        self.logger = get_logger("handflow.detection_window")

        self.setting = setting
        self.executor = executor

        # Debug toggles (can be toggled with keyboard) - must be defined before _update_status_label
        self._disable_drawing = True  # D key to toggle
        self._fps_cap_enabled = True  # C key to toggle
        self._screen_overlay_debug = False  # O key to toggle - shows macropad detection info
        self._actions_disabled = False  # A key to toggle - disables all actions, shows only prob bars

        # Recording state (R key to toggle) - simple queue-based
        self._recording = False
        self._recording_thread: Optional[threading.Thread] = None
        self._recording_stop_event: Optional[threading.Event] = None
        self._recording_start_time: Optional[float] = None
        self._recording_frame_count = 0
        self._recording_filename: Optional[str] = None
        self._recording_queue: Optional[queue.Queue] = None  # Simple frame queue
        self._recording_writer = None  # VideoWriter in main thread

        # Track last gesture for touch detection optimization
        self._last_gesture = "none"
        self._last_detections = {}
        self._finger_in_detected_area = False  # Track if finger is in ArUco/macropad area
        
        # Screen overlay macropad (just displays markers - detection via macropad_manager)
        self._screen_overlay = None  # Optional[ScreenOverlayMacroPad]
        self._overlay_cmd_lock = threading.Lock()
        self._overlay_cmd_show: Optional[bool] = None  # True=show, False=hide
        self._overlay_cmd_force_hide: bool = False
        self._overlay_cmd_hovered_button: Optional[int] = None  # For hover visual feedback
        self._overlay_cmd_activate: bool = False  # Activate hovered button on touch/touch_hold
        self._overlay_touch_processed: bool = False  # Track if current touch has been processed
        self._overlay_last_activation_time: float = 0.0  # Prevent double activation
        # Auto-hide when finger not in overlay region (after markers detected)
        self._overlay_no_hover_frames: int = 0  # Frames with no finger in region
        self._overlay_no_hover_threshold: int = 5  # Hide after this many frames

        # Window setup (16:9 aspect ratio to match training data)
        self.title("HandFlow v2.0 - Detection Preview [H/V/S/D/C/O/R/A/Q]")
        self.geometry("1280x750")  # 720 + status bar

        # UI Elements
        self.video_label = ctk.CTkLabel(self, text="Starting camera...")
        self.video_label.pack(fill="both", expand=True)

        # Status bar for flip info
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.pack(fill="x", side="bottom")
        self.status_label = ctk.CTkLabel(self.status_frame, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=10)
        self._update_status_label()

        # CV Components
        # Get screen size for ArUco detector
        sw, sh = 1920, 1080
        try:
            import pyautogui
            sw, sh = pyautogui.size()
        except:
            pass

        # Create ArUco detector first, then pass to GestureDetector
        self.aruco_detector = ArUcoScreenDetector(screen_width=sw, screen_height=sh)

        # Pass ArUco detector to GestureDetector for touch-to-cursor mapping
        self.gesture_Detector = GestureDetector(setting, executor, aruco_detector=self.aruco_detector)

        # detection_mode: "balanced" or "motion_priority"
        self.macropad_manager = MacroPadManager(setting, executor, detection_mode="balanced")

        # Knuckle macropad (palm-up hand = 6 knuckle buttons)
        self._knuckle_macropad = None
        if getattr(self.setting, 'knuckle_macropad_enabled', False):
            from handflow.detector.knuckle_macropad import KnuckleMacroPad
            self._knuckle_macropad = KnuckleMacroPad(setting, executor)
            # Set knuckle button labels for visual feedback
            knuckle_btns = getattr(setting, 'knuckle_macropad_buttons', {})
            knuckle_labels = [knuckle_btns.get(i, type('', (), {'label': ''})()).label or f"Knuckle {i+1}" for i in range(7)]
            self.logger.info("[Detection] Knuckle macropad initialized")

        # Initialize screen overlay macropad if enabled
        if self.setting.screen_overlay_macropad_enabled:
            from handflow.app.screen_overlay_macropad import ScreenOverlayMacroPad
            self._screen_overlay = ScreenOverlayMacroPad(setting, executor)
            self.logger.info("[Detection] Screen overlay macropad initialized")

        # Paper macropad visual feedback overlay
        from handflow.app.paper_macropad_feedback import PaperMacroPadFeedback
        self._paper_feedback = PaperMacroPadFeedback()

        self._spotify_overlay = None
        self._paper_feedback_last_hover: Optional[int] = None
        self._paper_feedback_last_activated: Optional[int] = None
        # Track last activation time to detect new activations reliably
        self._paper_feedback_last_activation_time: float = 0.0

        # Gesture visual feedback overlay (non-touch gestures)
        from handflow.app.gesture_feedback import GestureFeedback
        self._gesture_feedback = GestureFeedback()
        self._gesture_feedback_last_time: float = 0.0

        # State
        self._running = False
        self._thread = None
        self._cap = None
        self._latest_frame = None
        self._lock = threading.Lock()

        # FPS control (read from config for consistency with data collection)
        from handflow.utils import load_config
        config = load_config("config/config.yaml")
        self._target_fps = getattr(config.data, 'target_fps', 20.0)
        self._frame_duration = 1.0 / self._target_fps
        self._mediapipe_interval = 1  # Run MediaPipe every frame (needed for full 20fps data rate)
        self._gesture_model_interval = 2  # Run gesture TCN model every 2 frames
        self._aruco_interval = 3  # Run ArUco/MacroPad every 3 frames
        self._frame_count = 0
        self._last_detection_results = None  # Cache detection results

        # Resolution for display - 16:9 to match training data
        self._mp_width = 640
        self._mp_height = 360

        # Smaller resolution for MediaPipe processing (faster detection)
        self._mp_process_width = 320
        self._mp_process_height = 180

        # Pre-allocated buffer for RGB conversion (avoid allocation each frame)
        self._rgb_buffer = None

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self.stop)

        # Keyboard shortcuts
        self.bind("<Key-h>", self._toggle_flip_horizontal)
        self.bind("<Key-H>", self._toggle_flip_horizontal)
        self.bind("<Key-v>", self._toggle_flip_vertical)
        self.bind("<Key-V>", self._toggle_flip_vertical)
        self.bind("<Key-s>", self._toggle_swap_hands)
        self.bind("<Key-S>", self._toggle_swap_hands)
        self.bind("<Key-d>", self._toggle_drawing)
        self.bind("<Key-D>", self._toggle_drawing)
        self.bind("<Key-c>", self._toggle_fps_cap)
        self.bind("<Key-C>", self._toggle_fps_cap)
        self.bind("<Key-o>", self._toggle_screen_overlay_debug)
        self.bind("<Key-O>", self._toggle_screen_overlay_debug)
        self.bind("<Key-r>", self._toggle_recording)
        self.bind("<Key-R>", self._toggle_recording)
        self.bind("<Key-a>", self._toggle_actions)
        self.bind("<Key-A>", self._toggle_actions)
        self.bind("<space>", lambda e: self.gesture_Detector.end_capture())
        self.bind("<Key-q>", lambda e: self.stop())
        self.bind("<Key-Q>", lambda e: self.stop())
        self.bind("<Escape>", lambda e: self.stop())

        # Focus the window to receive key events
        self.focus_force()
        
    def _toggle_flip_horizontal(self, event=None):
        """Toggle horizontal flip."""
        self.setting.camera.flip_horizontal = not self.setting.camera.flip_horizontal
        self._update_status_label()
        self.logger.debug(f"Horizontal flip: {'ON' if self.setting.camera.flip_horizontal else 'OFF'}")

    def _toggle_flip_vertical(self, event=None):
        """Toggle vertical flip."""
        self.setting.camera.flip_vertical = not self.setting.camera.flip_vertical
        self._update_status_label()
        self.logger.debug(f"Vertical flip: {'ON' if self.setting.camera.flip_vertical else 'OFF'}")

    def _toggle_swap_hands(self, event=None):
        """Toggle swap hands (L/R labels)."""
        self.setting.camera.swap_hands = not self.setting.camera.swap_hands
        self._update_status_label()
        self.logger.debug(f"Swap hands: {'ON' if self.setting.camera.swap_hands else 'OFF'}")

    def _toggle_drawing(self, event=None):
        """Toggle debug drawing."""
        self._disable_drawing = not self._disable_drawing
        self._update_status_label()
        self.logger.debug(f"Drawing: {'OFF' if self._disable_drawing else 'ON'}")

    def _toggle_fps_cap(self, event=None):
        """Toggle data collection rate limiting (20 FPS)."""
        self._fps_cap_enabled = not self._fps_cap_enabled
        self.gesture_Detector.set_data_rate_limit(self._fps_cap_enabled)
        self._update_status_label()
        self.logger.debug(f"Data Rate Limit: {'ON (20 FPS)' if self._fps_cap_enabled else 'OFF (unlimited)'}")

    def _toggle_actions(self, event=None):
        """Toggle all actions on/off. When off, only prob bars show."""
        self._actions_disabled = not self._actions_disabled
        self.gesture_Detector.actions_disabled = self._actions_disabled
        self._update_status_label()
        self.logger.debug(f"Actions: {'DISABLED' if self._actions_disabled else 'ENABLED'}")

    def _toggle_screen_overlay_debug(self, event=None):
        """Toggle screen overlay macropad debug info."""
        self._screen_overlay_debug = not self._screen_overlay_debug
        self._update_status_label()
        self.logger.debug(f"Screen Overlay Debug: {'ON' if self._screen_overlay_debug else 'OFF'}")

    def _toggle_recording(self, event=None):
        """Toggle video recording (raw frames at full resolution)."""
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()
        self._update_status_label()

    def _start_recording(self):
        """Start recording - simple queue-based approach."""
        recordings_dir = "recordings"
        os.makedirs(recordings_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recording_filename = os.path.join(recordings_dir, f"camera_{timestamp}.mp4")

        # Simple bounded queue - drops old frames if writer can't keep up
        self._recording_queue = queue.Queue(maxsize=60)  # ~2 sec buffer at 30fps

        self._recording_stop_event = threading.Event()
        self._recording_start_time = time.time()
        self._recording_frame_count = 0
        self._recording = True
        self._recording_fps = 30.0  # Fixed FPS for constant-speed playback

        # Start recording thread
        self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._recording_thread.start()
        self.logger.info(f"Recording started: {self._recording_filename} ({self._recording_fps} FPS)")

    def _recording_loop(self):
        """Recording thread - constant FPS using wall-clock timing for real-time playback."""
        # Set lower priority for recording thread (UTILITY class = background I/O)
        if sys.platform == 'darwin':
            try:
                import ctypes
                import ctypes.util
                libpthread = ctypes.CDLL(ctypes.util.find_library('pthread'))
                QOS_CLASS_UTILITY = 0x11  # Lower priority for background I/O
                pthread_set_qos = libpthread.pthread_set_qos_class_self_np
                pthread_set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
                pthread_set_qos.restype = ctypes.c_int
                pthread_set_qos(QOS_CLASS_UTILITY, 0)
            except:
                pass

        RECORDING_FPS = self._recording_fps  # Fixed 30 FPS
        frame_interval = 1.0 / RECORDING_FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = None
        frames_written = 0
        last_frame = None
        next_frame_time = None  # Wall-clock deadline for next video frame

        while not self._recording_stop_event.is_set():
            try:
                frame = self._recording_queue.get(timeout=0.05)
                last_frame = frame
                now = time.time()

                # Initialize writer on first frame
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(self._recording_filename, fourcc, RECORDING_FPS, (w, h))
                    if not writer.isOpened():
                        self.logger.error(f"[Recording] Failed to open video writer at {w}x{h}")
                        return
                    self.logger.info(f"[Recording] Writer initialized at {w}x{h}, fixed FPS: {RECORDING_FPS}")
                    next_frame_time = now

                # Write frames to fill up to current wall-clock time
                # - Processing faster than 30fps: only latest frame written per interval
                # - Processing slower than 30fps: last frame duplicated to fill gaps
                # Result: video always plays back at real-time speed
                while next_frame_time <= now:
                    writer.write(last_frame)
                    frames_written += 1
                    next_frame_time += frame_interval

                self._recording_frame_count = frames_written

            except queue.Empty:
                # No new frame - duplicate last frame to fill time gaps
                if last_frame is not None and writer is not None and next_frame_time is not None:
                    now = time.time()
                    while next_frame_time <= now:
                        writer.write(last_frame)
                        frames_written += 1
                        next_frame_time += frame_interval
                    self._recording_frame_count = frames_written
                continue

        # Fill final gap on stop
        if last_frame is not None and writer is not None and next_frame_time is not None:
            now = time.time()
            while next_frame_time <= now:
                writer.write(last_frame)
                frames_written += 1
                next_frame_time += frame_interval

        self._recording_frame_count = frames_written

        if writer:
            writer.release()
        self.logger.info(f"[Recording] Thread done, wrote {frames_written} frames at {RECORDING_FPS} FPS")

    def _stop_recording(self):
        """Stop recording thread."""
        self._recording = False

        if self._recording_stop_event:
            self._recording_stop_event.set()

        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=3.0)

        self._recording_thread = None
        self._recording_stop_event = None
        self._recording_queue = None

        duration = time.time() - self._recording_start_time if self._recording_start_time else 0

        fps = self._recording_frame_count / duration if duration > 0 else 0
        self.logger.info(f"Recording saved: {self._recording_filename} "
                        f"({self._recording_frame_count} frames, {duration:.1f}s, {fps:.1f} FPS)")

        self._recording_start_time = None
        self._recording_frame_count = 0

    def _update_status_label(self):
        """Update status bar with current setting state."""
        h_flip = "ON" if self.setting.camera.flip_horizontal else "OFF"
        v_flip = "ON" if self.setting.camera.flip_vertical else "OFF"
        swap = "ON" if self.setting.camera.swap_hands else "OFF"
        draw = "OFF" if self._disable_drawing else "ON"
        cap = "ON" if self._fps_cap_enabled else "OFF"
        overlay_dbg = "ON" if self._screen_overlay_debug else "OFF"
        rec = "REC" if self._recording else "OFF"
        act = "OFF" if self._actions_disabled else "ON"

        status_text = f"H:{h_flip} V:{v_flip} Swap:{swap} Draw:{draw} Cap:{cap} Act:{act} Rec:{rec}"
        self.status_label.configure(text=status_text)

        # Visual indicator for recording
        if self._recording:
            self.status_label.configure(text_color="red")
        else:
            self.status_label.configure(text_color=("gray10", "gray90"))

    def start(self):
        """Start detection."""
        if self._running:
            return

        # Disable macOS App Nap for consistent background performance
        disable_app_nap()

        self.logger.info("Starting detection window.")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Start UI update loop
        self._update_ui()
        
    def stop(self):
        """Stop detection and close window."""
        self._running = False

        # Stop recording if active
        if self._recording:
            self._stop_recording()

        if self._thread:
            self._thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()

        # Clean up screen overlay
        if self._screen_overlay:
            self._screen_overlay.hide()
            self._screen_overlay.destroy()

        # Clean up paper macropad feedback
        if self._paper_feedback:
            self._paper_feedback.destroy()

        # Clean up gesture feedback
        if self._gesture_feedback:
            self._gesture_feedback.destroy()


        # Re-enable macOS App Nap
        enable_app_nap()

        self.gesture_Detector.close()
        self.destroy()
        
    def _capture_loop(self):
        """Background thread for CV processing with optimized detection."""
        # Boost thread priority for consistent timing even when backgrounded
        if sys.platform == 'darwin':
            try:
                import ctypes
                import ctypes.util
                # Load libpthread for QoS setting
                libpthread = ctypes.CDLL(ctypes.util.find_library('pthread'))
                # pthread_set_qos_class_self_np(qos_class, relative_priority)
                # QOS_CLASS_USER_INTERACTIVE = 0x21 (highest priority for UI responsiveness)
                QOS_CLASS_USER_INTERACTIVE = 0x21
                pthread_set_qos = libpthread.pthread_set_qos_class_self_np
                pthread_set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
                pthread_set_qos.restype = ctypes.c_int
                result = pthread_set_qos(QOS_CLASS_USER_INTERACTIVE, 0)
            except Exception:
                pass

        cam_idx = self.setting.camera.index
        camera_source = getattr(self.setting.camera, 'source', 'webcam')
        consecutive_failures = 0
        max_failures = 30  # Stop after 30 consecutive frame read failures

        try:
            if camera_source == "esp32":
                esp32_port = getattr(self.setting.camera, 'esp32_serial_port', '/dev/cu.usbmodem101')
                esp32_baud = getattr(self.setting.camera, 'esp32_baud_rate', 2000000)
                self.logger.info(f"[Detection] Opening ESP32 camera on {esp32_port} @ {esp32_baud}")
                self._cap = ESP32Camera(esp32_port, esp32_baud)
                self._cap.open()
            else:
                self._cap = cv2.VideoCapture(cam_idx)
                # Use 1920x1080 for high quality recording
                # Resized to smaller resolution internally for processing
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self._cap.set(cv2.CAP_PROP_FPS, 30)
                # Minimize buffer for lowest latency (1 frame = most recent)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._cap.isOpened():
                source_desc = f"ESP32 on {esp32_port}" if camera_source == "esp32" else f"camera {cam_idx}"
                self.logger.error(f"[Detection] Error: Could not open {source_desc}")
                self._running = False
                return

            self.logger.info(f"[Detection] Processing started. Target: {self._target_fps} FPS, "
                            f"Gesture model every {self._gesture_model_interval} frames, "
                            f"ArUco every {self._aruco_interval} frames")

            last_frame_time = time.time()

            while self._running:
                try:
                    ret, frame = self._cap.read()
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            self.logger.error(f"[Detection] Camera read failed {max_failures} times consecutively. Stopping.")
                            self._running = False
                            break
                        time.sleep(0.01)
                        continue

                    # Reset failure counter on successful read
                    consecutive_failures = 0

                    # Calculate delta time for FPS-invariant features
                    current_time = time.time()
                    delta_time = current_time - last_frame_time
                    last_frame_time = current_time

                    self._frame_count += 1

                    # 1. Preprocessing - flip and resize
                    # When recording: flip full-res first, queue it, then resize
                    # When not recording: resize first, then flip small frame (much faster)
                    is_recording = self._recording and self._recording_queue is not None
                    flip_h = self.setting.camera.flip_horizontal
                    flip_v = self.setting.camera.flip_vertical

                    if is_recording:
                        if flip_h:
                            frame = cv2.flip(frame, 1)
                        if flip_v:
                            frame = cv2.flip(frame, 0)
                        try:
                            h, w = frame.shape[:2]
                            if w != 1920 or h != 1080:
                                rec_frame = cv2.resize(frame, (1920, 1080))
                            else:
                                rec_frame = frame.copy()
                            self._recording_queue.put_nowait(rec_frame)
                        except queue.Full:
                            pass

                    # Resize to display size (640x360)
                    frame_small = cv2.resize(frame, (self._mp_width, self._mp_height), interpolation=cv2.INTER_NEAREST)

                    # Flip on small frame when not recording (skip redundant full-res flip)
                    if not is_recording:
                        if flip_h:
                            frame_small = cv2.flip(frame_small, 1)
                        if flip_v:
                            frame_small = cv2.flip(frame_small, 0)

                    h_small, w_small = frame_small.shape[:2]

                    # Even smaller frame for MediaPipe processing (320x180)
                    frame_mp = cv2.resize(frame_small, (self._mp_process_width, self._mp_process_height), interpolation=cv2.INTER_NEAREST)

                    # 2. ArUco/MacroPad Detection - every 2 frames
                    run_aruco = (
                        (self._frame_count % self._aruco_interval == 0) or
                        (self._frame_count <= 1)
                    )

                    if run_aruco:
                        # Detect ArUco markers on small frame (same as display)
                        self.aruco_detector.detect(frame_small)

                        # MacroPad uses same markers - pass detected markers directly
                        if self.setting.macropad_enabled:
                            # When screen overlay is visible, prioritize its detection
                            # This prevents paper macropad markers from interfering
                            screen_overlay_visible = (
                                self._screen_overlay is not None and
                                self._screen_overlay.is_visible()
                            )
                            self.macropad_manager.detect_markers(
                                frame_small,
                                prioritize_screen_overlay=screen_overlay_visible
                            )

                    # 3. Check if finger is in detected area (ArUco screen or macropad)
                    # Used to run ArUco detection every frame when interacting
                    self._finger_in_detected_area = False
                    macropad_active = False
                    last_tip = self.gesture_Detector._right_index_tip
                    if last_tip is not None:
                        pixel_tip = (last_tip[0] * w_small, last_tip[1] * h_small)

                        # Check ArUco screen area
                        if self.aruco_detector.is_point_in_screen(pixel_tip):
                            self._finger_in_detected_area = True

                        # Check macropad area (paper or screen overlay)
                        if self.setting.macropad_enabled and self.macropad_manager.is_detected():
                            if self.macropad_manager._detector.is_point_in_region(pixel_tip):
                                self._finger_in_detected_area = True
                                macropad_active = True  # Finger is in macropad region
                            hovered = self.macropad_manager._detector.get_button_at_point(pixel_tip)
                            if hovered is not None:
                                macropad_active = True

                        # Also check if screen overlay is visible and active
                        if self._screen_overlay and self._screen_overlay.is_visible():
                            macropad_active = True  # Screen overlay handles touch

                    # 4. Tell gesture detector whether macropad is handling interaction
                    self.gesture_Detector.set_macropad_active(macropad_active)

                    # 5. Gesture Recognition
                    # - MediaPipe runs every N frames (biggest CPU cost ~15-30ms)
                    # - Gesture TCN model runs only when MediaPipe ran (needs fresh keypoints)
                    # - On skip frames, cached finger positions keep cursor smooth
                    run_mediapipe = (self._frame_count % self._mediapipe_interval == 0) or (self._frame_count <= 1)
                    run_gesture_model = run_mediapipe  # TCN only when we have new keypoints

                    output, detections = self.gesture_Detector.process_frame(
                        frame_small,  # Display resolution (640x360)
                        frame_small=frame_mp,  # Smaller frame for MediaPipe (320x180)
                        run_gesture_model=run_gesture_model,
                        run_mediapipe=run_mediapipe,
                        delta_time=delta_time,
                        disable_drawing=self._disable_drawing
                    )

                    # When MediaPipe was skipped, reuse last known detections
                    # to prevent overlay flickering and gesture state loss
                    if not run_mediapipe:
                        detections = self._last_detections

                    # Track last gesture for touch optimization
                    self._last_detections = detections
                    current_gesture = "none"
                    for hand in ['Right', 'Left']:
                        if hand in detections and 'gesture' in detections[hand]:
                            current_gesture = detections[hand]['gesture']
                            break
                    self._last_gesture = current_gesture

                    # Get right hand gesture specifically for screen overlay (only right hand triggers overlay)
                    right_hand_gesture = "none"
                    if 'Right' in detections and 'gesture' in detections['Right']:
                        right_hand_gesture = detections['Right']['gesture']

                    # Skip all actions when disabled (prob bars still show)
                    if self._actions_disabled:
                        with self._lock:
                            self._latest_frame = output
                        continue

                    # 6. Screen Overlay MacroPad handling (if enabled)
                    # Hide overlay when knuckle macropad is active (palm up)
                    knuckle_active_now = (self._knuckle_macropad is not None and
                                         self._knuckle_macropad.is_active())
                    if self._screen_overlay is not None and self.setting.screen_overlay_macropad_enabled:
                        if knuckle_active_now:
                            if self._screen_overlay.is_visible():
                                self._screen_overlay.hide()
                        else:
                            self._handle_screen_overlay(right_hand_gesture)
                    
                    # 7. Draw debug overlays (ArUco and MacroPad) - skip if drawing disabled
                    if not self._disable_drawing:
                        output = self.aruco_detector.draw_debug(output)

                    if self.setting.macropad_enabled:
                        # Update finger state for macropad interaction
                        # Only right hand can interact with macropad (left hand gestures ignored)
                        if 'Right' in detections:
                            info = detections['Right']
                            if 'index_tip' in info:
                                idx_norm = info['index_tip']
                                pixel_tip = (idx_norm[0] * w_small, idx_norm[1] * h_small)
                                gesture = info.get('gesture', 'none')
                                is_touching = gesture == 'touch'
                                # Skip activation when screen overlay is visible - it handles activation separately
                                # This prevents double activation (one from update_finger_state, one from _process_overlay_commands)
                                screen_overlay_active = (
                                    self._screen_overlay is not None and
                                    self._screen_overlay.is_visible()
                                )
                                self.macropad_manager.update_finger_state(
                                    pixel_tip,
                                    is_touching,
                                    skip_activation=screen_overlay_active
                                )

                        if not self._disable_drawing:
                            output = self.macropad_manager.draw_debug(output)

                    # 8. Knuckle MacroPad (palm-up hand = 6 buttons per hand)
                    if self._knuckle_macropad is not None:
                        left_kp = detections.get('Left', {}).get('keypoints', None)
                        right_kp = detections.get('Right', {}).get('keypoints', None)

                        left_tip = None
                        right_tip = None
                        l_gesture = detections.get('Left', {}).get('gesture', 'none')
                        r_gesture = detections.get('Right', {}).get('gesture', 'none')

                        if 'Left' in detections and 'index_tip' in detections['Left']:
                            lt = detections['Left']['index_tip']
                            left_tip = (lt[0], lt[1])
                        if 'Right' in detections and 'index_tip' in detections['Right']:
                            rt = detections['Right']['index_tip']
                            right_tip = (rt[0], rt[1])

                        act_left, act_right = self._knuckle_macropad.update(
                            left_kp, right_kp, left_tip, right_tip, l_gesture, r_gesture
                        )

                        # Feedback handled on main thread in _process_paper_feedback()

                        if not self._disable_drawing and self._knuckle_macropad.is_active():
                            output = self._knuckle_macropad.draw_debug(output, w_small, h_small)

                    # 9. Screen overlay debug info (press 'O' to toggle)
                    if self._screen_overlay_debug and self._screen_overlay:
                        overlay_visible = self._screen_overlay.is_visible()
                        detected_set = self.macropad_manager._detector.current_set_id
                        is_screen_set = detected_set == SCREEN_OVERLAY_SET_ID
                        hovered_btn = self.macropad_manager._hovered_button

                        # Draw debug text
                        debug_y = 50
                        color = (0, 255, 0) if is_screen_set else (0, 165, 255)
                        cv2.putText(output, f"[ScreenOverlay Debug]", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        debug_y += 20
                        cv2.putText(output, f"Overlay visible: {overlay_visible}", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        debug_y += 20
                        cv2.putText(output, f"Detected set: {detected_set} (screen={SCREEN_OVERLAY_SET_ID})", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        debug_y += 20
                        cv2.putText(output, f"MacroPad valid: {self.macropad_manager.is_detected()}", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        debug_y += 20
                        hover_color = (0, 255, 0) if hovered_btn is not None else (128, 128, 128)
                        cv2.putText(output, f"Hovered button: {hovered_btn}", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hover_color, 1)


                    # 10. Live capture overlay
                    # Update end point + check pinch release every frame
                    if self.gesture_Detector._cap_state == "holding":
                        tip = self.gesture_Detector._right_index_tip
                        if tip:
                            self.gesture_Detector._cap_end = tip

                    cap_start, cap_end, cap_holding, cap_complete = \
                        self.gesture_Detector.get_capture_state()

                    if cap_start and cap_end and (cap_holding or cap_complete):
                        x1 = int(cap_start[0] * w_small)
                        y1 = int(cap_start[1] * h_small)
                        x2 = int(cap_end[0] * w_small)
                        y2 = int(cap_end[1] * h_small)

                        color = (0, 255, 0) if cap_complete else (0, 200, 255)
                        # Dashed rectangle
                        for ex1, ey1, ex2, ey2 in [
                            (x1, y1, x2, y1), (x1, y2, x2, y2),
                            (x1, y1, x1, y2), (x2, y1, x2, y2),
                        ]:
                            dl = max(abs(ex2 - ex1), abs(ey2 - ey1))
                            if dl == 0: continue
                            for i in range(0, int(dl), 16):
                                t1, t2 = i / dl, min((i + 8) / dl, 1.0)
                                cv2.line(output,
                                    (int(ex1 + (ex2-ex1)*t1), int(ey1 + (ey2-ey1)*t1)),
                                    (int(ex1 + (ex2-ex1)*t2), int(ey1 + (ey2-ey1)*t2)),
                                    color, 2)

                        cv2.putText(output,
                            "CAPTURED" if cap_complete else "CAPTURING...",
                            (min(x1, x2), min(y1, y2) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                    if cap_complete and cap_start and cap_end:
                        ret_cap, cap_frame = self._cap.read()
                        if ret_cap and cap_frame is not None:
                            if flip_h: cap_frame = cv2.flip(cap_frame, 1)
                            if flip_v: cap_frame = cv2.flip(cap_frame, 0)

                            # Save full-res original
                            os.makedirs("captures", exist_ok=True)
                            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            orig_fname = f"captures/original_{ts}.png"
                            cv2.imwrite(orig_fname, cap_frame)

                            # Crop coordinates (normalized 0-1)
                            import json as _json
                            crop_coords = _json.dumps({
                                "x1": min(cap_start[0], cap_end[0]),
                                "y1": min(cap_start[1], cap_end[1]),
                                "x2": max(cap_start[0], cap_end[0]),
                                "y2": max(cap_start[1], cap_end[1]),
                            })

                            fh, fw = cap_frame.shape[:2]
                            self.logger.info(f"[Capture] Saved {orig_fname} ({fw}x{fh})")

                            # Launch vision popup with full image + crop coords
                            try:
                                import subprocess as sp
                                vision_script = os.path.join(os.path.dirname(__file__), "vision_popup.py")
                                sp.Popen([sys.executable, vision_script, os.path.abspath(orig_fname), crop_coords])
                                self.logger.info("[Capture] Vision popup launched")
                            except Exception as ve:
                                self.logger.error(f"[Capture] Vision popup failed: {ve}")
                        self.gesture_Detector.clear_capture()

                    # Update latest frame safely (small frame, will be scaled up by CTkImage)
                    with self._lock:
                        self._latest_frame = output

                except Exception as e:
                    self.logger.error(f"[Detection] Error in frame processing: {e}", exc_info=True)
                    # Continue processing - don't crash on single frame errors
                    continue

        except Exception as e:
            self.logger.error(f"[Detection] Fatal error in capture loop: {e}", exc_info=True)
        finally:
            self.logger.info("[Detection] Capture loop ended.")
            self._running = False

                
    def _update_ui(self):
        """Main thread UI update loop."""
        if not self._running:
            self.logger.info("[Detection] UI update stopped - capture not running.")
            return

        try:
            image = None
            with self._lock:
                if self._latest_frame is not None:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)

            if image:
                # Create/Update CTkImage - scale up small frame (640x360) to display size (1280x720)
                display_size = (1280, 720)
                ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=display_size)
                self.video_label.configure(image=ctk_img, text="")

        except Exception as e:
            self.logger.error(f"[Detection] Error in UI update: {e}")

        # Process screen overlay commands (must run on main thread)
        if self._screen_overlay:
            if self.setting.screen_overlay_macropad_enabled:
                self._process_overlay_commands()
            else:
                # Hide overlay if setting was disabled
                if self._screen_overlay.is_visible():
                    self._screen_overlay.hide()

        # Process paper macropad feedback (only when screen overlay is NOT visible)
        self._process_paper_feedback()

        # Process gesture feedback (non-touch gestures)
        self._process_gesture_feedback()

        # Schedule next update (~20 FPS - sufficient for preview, saves CPU)
        if self._running:
            self.after(50, self._update_ui)

    def _process_overlay_commands(self):
        """
        Process overlay commands from background thread (runs on main thread).

        The screen overlay ONLY displays markers - all detection is handled by
        macropad_manager which sees the markers through the camera.
        """
        with self._overlay_cmd_lock:
            if self._overlay_cmd_show is True:
                if not self._screen_overlay.is_visible():
                    if not self._screen_overlay.is_in_cooldown():
                        self._screen_overlay.show()
                else:
                    self._screen_overlay.reset_hide_grace()

                # Update hover visual on overlay (feedback from macropad_manager detection)
                self._screen_overlay.set_hovered_button(self._overlay_cmd_hovered_button)

                # Activate button on touch/touch_hold gesture
                if self._overlay_cmd_activate and self._overlay_cmd_hovered_button is not None:
                    # Check time-based cooldown to prevent double activation
                    current_time = time.time()
                    if current_time - self._overlay_last_activation_time < 0.5:
                        self._overlay_cmd_activate = False
                    else:
                        # Execute button action via macropad_manager
                        # IMPORTANT: Force set ID to screen overlay (20) to ensure correct button set is used
                        self.macropad_manager._activate_button(
                            self._overlay_cmd_hovered_button,
                            force_set_id=SCREEN_OVERLAY_SET_ID
                        )
                        self._overlay_last_activation_time = current_time
                        self.logger.info(f"[ScreenOverlay] Activated button {self._overlay_cmd_hovered_button} from set {SCREEN_OVERLAY_SET_ID}")
                        self._overlay_cmd_activate = False

                    # Hide overlay after activation and set cooldown
                    if self._overlay_cmd_force_hide:
                        self._screen_overlay.hide()
                        self._screen_overlay.set_cooldown(0.7)  # 0.7 second cooldown before touch_hover can show again
                        self._overlay_cmd_force_hide = False
                        self.logger.debug("Screen overlay hidden with cooldown")

                elif self._overlay_cmd_activate and self._overlay_cmd_hovered_button is None:
                    # Still hide and cooldown even if no button was hovered
                    if self._overlay_cmd_force_hide:
                        self._screen_overlay.hide()
                        self._screen_overlay.set_cooldown(0.5)
                        self._overlay_cmd_force_hide = False

            elif self._overlay_cmd_show is False:
                if self._screen_overlay.is_visible():
                    if self._overlay_cmd_force_hide:
                        self._screen_overlay.hide()
                        self._screen_overlay.set_cooldown(0.7)
                        self._overlay_cmd_force_hide = False
                    else:
                        self._screen_overlay.request_hide()

            self._overlay_cmd_show = None
            self._overlay_cmd_activate = False

        self._screen_overlay.update()

    def _process_paper_feedback(self):
        """
        Process paper macropad visual feedback (runs on main thread).

        Shows hover/click feedback overlay for paper macropad interactions.
        Only active when screen overlay is NOT visible.
        """
        # Knuckle macropad feedback (takes priority when active)
        if self._knuckle_macropad is not None and self._knuckle_macropad.is_active():
            # Hover
            h_left = self._knuckle_macropad._hovered_left
            h_right = self._knuckle_macropad._hovered_right
            hover = h_left if h_left is not None else h_right

            # Set labels from knuckle config
            knuckle_btns = getattr(self.setting, 'knuckle_macropad_buttons', {})
            labels = []
            for i in range(7):
                btn = knuckle_btns.get(i)
                if btn and hasattr(btn, 'label') and btn.label:
                    labels.append(btn.label)
                else:
                    labels.append(f"Knuckle {i+1}")
            self._paper_feedback.set_button_labels(labels)

            # Check for activation (click feedback)
            act_time = self._knuckle_macropad._last_activated_time
            if act_time > self._paper_feedback_last_activation_time:
                act_idx = self._knuckle_macropad._last_activated_idx
                if act_idx is not None:
                    self._paper_feedback.show_click_feedback(act_idx)
                    self._paper_feedback_last_activation_time = act_time

            # Hover feedback
            elif hover != self._paper_feedback_last_hover:
                self._paper_feedback.set_hovered_button(hover)
                self._paper_feedback_last_hover = hover


            self._paper_feedback.update()
            return

        # Skip if screen overlay is visible (it has its own feedback)
        if self._screen_overlay and self._screen_overlay.is_visible():
            # Hide paper feedback if it was showing
            if self._paper_feedback_last_hover is not None:
                self._paper_feedback.set_hovered_button(None)
                self._paper_feedback_last_hover = None
            return

        # Skip if macropad not enabled or not detected
        if not self.setting.macropad_enabled or not self.macropad_manager.is_detected():
            if self._paper_feedback_last_hover is not None:
                self._paper_feedback.set_hovered_button(None)
                self._paper_feedback_last_hover = None
            return

        # Skip if screen overlay set is detected (ID 20)
        detected_set = self.macropad_manager._detector.current_set_id
        if detected_set == SCREEN_OVERLAY_SET_ID:
            return

        # Update button labels from active set
        active_set = self.macropad_manager.active_set
        if active_set:
            labels = [
                active_set.buttons.get(i).label if active_set.buttons.get(i) and active_set.buttons.get(i).label else f"Button {i+1}"
                for i in range(12)
            ]
            self._paper_feedback.set_button_labels(labels)

        # Check current hover state
        current_hover = self.macropad_manager._hovered_button

        # Update hover feedback if changed
        if current_hover != self._paper_feedback_last_hover:
            self._paper_feedback.set_hovered_button(current_hover)
            self._paper_feedback_last_hover = current_hover

        # Check for activation using activation log (more reliable than _activated_button)
        # The log has timestamps so we can detect new activations even if _activated_button was cleared
        activation_log = self.macropad_manager._activation_log
        if activation_log:
            latest_activation = activation_log[-1]
            # Check if this is a new activation (timestamp changed)
            if latest_activation.timestamp > self._paper_feedback_last_activation_time:
                self._paper_feedback.show_click_feedback(latest_activation.button_idx)
                self._paper_feedback_last_activation_time = latest_activation.timestamp

        # Process Tk events
        self._paper_feedback.update()

    def _process_gesture_feedback(self):
        """
        Process gesture visual feedback (runs on main thread).

        Shows macropad-button-style activation overlay when a non-touch gesture
        is detected (after cooldown).
        """
        activated = self.gesture_Detector._last_activated_gesture
        if activated is not None:
            gesture_name, hand, timestamp = activated
            # Only show if this is a new activation (not already shown)
            if timestamp > self._gesture_feedback_last_time:
                self._gesture_feedback.show_gesture(gesture_name, hand)
                self._gesture_feedback_last_time = timestamp

        # Process Tk events
        self._gesture_feedback.update()

    def _is_point_in_expanded_region(self, point: Tuple[float, float], region: np.ndarray, margin_ratio: float = 0.25) -> bool:
        """
        Check if a point is within an expanded version of the detection region.

        Args:
            point: (x, y) point to check
            region: 4 corners of the detection region [TL, TR, BR, BL]
            margin_ratio: How much to expand (0.25 = expand by 25% of region size on each side)

        Returns:
            True if point is within the expanded region
        """
        if region is None or len(region) != 4:
            return False

        # Calculate region center
        center = np.mean(region, axis=0)

        # Expand each corner outward from center
        expanded = []
        for corner in region:
            direction = corner - center
            expanded_corner = corner + direction * margin_ratio
            expanded.append(expanded_corner)

        expanded_region = np.array(expanded, dtype=np.float32)

        # Check if point is inside expanded polygon
        result = cv2.pointPolygonTest(expanded_region, point, False)
        return result >= 0

    def _handle_screen_overlay(self, current_gesture: str):
        """
        Handle screen overlay show/hide based on gesture.

        The overlay ONLY displays ArUco markers - same as paper macropad.
        Detection is handled by macropad_manager which sees the markers via camera.

        Gestures:
        - touch_hover: Show overlay (only if finger NOT hovering over paper macropad)
        - touch / touch_hold: Activate if screen overlay detected, then hide
        - other: Hide overlay
        """
        # During cooldown, completely ignore touch_hover and touch gestures
        # This prevents any overlay interaction until cooldown expires
        if self._screen_overlay.is_in_cooldown():
            return

        # Check what's currently detected
        is_detected = self.macropad_manager.is_detected()
        detected_set = self.macropad_manager._detector.current_set_id if is_detected else None
        mp_hovered = self.macropad_manager._hovered_button

        # Check if finger is near paper macropad region (set IDs 12, 13, 14)
        # Use expanded region (with margin) to suppress screen overlay even when
        # finger is near but not exactly over a button (e.g., due to projection angle)
        finger_over_paper_macropad = False
        if is_detected and detected_set in (12, 13, 14):
            finger_pos = self.macropad_manager._finger_pos
            detection = self.macropad_manager._detector.detection
            if finger_pos and detection is not None:
                # Check if finger is within expanded region (original + ~1 marker size margin)
                # margin_ratio=0.25 expands by 25% on each side, roughly 1 marker size
                finger_over_paper_macropad = self._is_point_in_expanded_region(
                    finger_pos, detection.detection_region, margin_ratio=0.25
                )

        # Get hovered button only if screen overlay is detected
        hovered_btn = None
        if is_detected and detected_set == SCREEN_OVERLAY_SET_ID:
            hovered_btn = mp_hovered

        with self._overlay_cmd_lock:
            self._overlay_cmd_hovered_button = hovered_btn

            if current_gesture == 'touch_hover':
                # Reset touch processed flag when not touching
                self._overlay_touch_processed = False
                # Only show overlay if finger is NOT over paper macropad
                if not finger_over_paper_macropad:
                    self._overlay_cmd_show = True
                    self._overlay_cmd_activate = False
                    self._overlay_cmd_force_hide = False

                    # Auto-hide check: if overlay is visible AND screen overlay markers detected
                    # but finger is not over any button, count frames and hide after threshold
                    if (self._screen_overlay and self._screen_overlay.is_visible() and
                        is_detected and detected_set == SCREEN_OVERLAY_SET_ID):
                        # Screen overlay markers are detected by camera
                        if hovered_btn is None:
                            # Finger not over any button - increment counter
                            self._overlay_no_hover_frames += 1
                            if self._overlay_no_hover_frames >= self._overlay_no_hover_threshold:
                                # Hide overlay - finger not in region for too long
                                self._overlay_cmd_show = False
                                self._overlay_cmd_force_hide = True
                                self._overlay_no_hover_frames = 0
                        else:
                            # Finger is over a button - reset counter
                            self._overlay_no_hover_frames = 0
                    elif not (is_detected and detected_set == SCREEN_OVERLAY_SET_ID):
                        # Markers not detected yet - don't count (camera still acquiring)
                        pass
                else:
                    # Finger is over paper macropad - don't show screen overlay
                    self._overlay_cmd_show = False
                    self._overlay_cmd_activate = False
                    self._overlay_cmd_force_hide = False
                    self._overlay_no_hover_frames = 0  # Reset counter

            elif current_gesture in ('touch', 'touch_hold'):
                # Reset no-hover counter on touch
                self._overlay_no_hover_frames = 0
                # Only activate screen overlay if finger is NOT over paper macropad
                # AND we haven't already processed this touch
                if not finger_over_paper_macropad and not self._overlay_touch_processed:
                    self._overlay_cmd_show = True
                    self._overlay_cmd_activate = True
                    self._overlay_cmd_force_hide = True
                    self._overlay_touch_processed = True  # Mark as processed
                elif finger_over_paper_macropad:
                    # Finger is over paper macropad - don't show/activate screen overlay
                    self._overlay_cmd_show = False
                    self._overlay_cmd_activate = False
                    self._overlay_cmd_force_hide = False

            else:
                # No relevant gesture - hide overlay and reset touch flag
                self._overlay_touch_processed = False
                self._overlay_cmd_show = False
                self._overlay_cmd_activate = False
                self._overlay_cmd_force_hide = False
                self._overlay_no_hover_frames = 0  # Reset counter
