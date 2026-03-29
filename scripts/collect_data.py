"""GUI tool for collecting gesture training data. Left hand is flipped to match right hand orientation."""

import sys
import os
import time
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from handflow.utils import load_config, load_setting

class GUIDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("HandFlow Data Collector")
        self.root.geometry("400x500")
        
        self.config = load_config("config/config.yaml")
        self.setting = load_setting("config/handflow_setting.yaml")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # UI Variables
        self.hand_var = tk.StringVar(value="right")
        self.gesture_var = tk.StringVar()
        self.camera_var = tk.StringVar(value=str(self.setting.camera.index))
        self.batch_size_var = tk.IntVar(value=30)
        self.flip_h_var = tk.BooleanVar(value=self.setting.camera.flip_horizontal)
        self.flip_v_var = tk.BooleanVar(value=self.setting.camera.flip_vertical)
        self.swap_hands_var = tk.BooleanVar(value=False)
        
        self.continuous_var = tk.BooleanVar(value=False)
        self.trigger_record = False # Flag to bridge UI events to loop
        self.is_running = False  # Track if collection is active
        self.stop_flag = False

        self.setup_ui() 
        
    def setup_ui(self):
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Hand Selection
        ttk.Label(settings_frame, text="Hand:").grid(row=0, column=0, sticky="w", pady=5)
        frame_hand = ttk.Frame(settings_frame)
        frame_hand.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frame_hand, text="Right", variable=self.hand_var, value="right", command=self.update_gestures).pack(side="left", padx=5)
        ttk.Radiobutton(frame_hand, text="Left", variable=self.hand_var, value="left", command=self.update_gestures).pack(side="left", padx=5)
        
        # Gesture Selection
        ttk.Label(settings_frame, text="Gesture:").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_gesture = ttk.Combobox(settings_frame, textvariable=self.gesture_var, state="readonly")
        self.combo_gesture.grid(row=1, column=1, sticky="ew", padx=5)
        
        # Camera Selection
        ttk.Label(settings_frame, text="Camera:").grid(row=2, column=0, sticky="w", pady=5)
        self.combo_camera = ttk.Combobox(settings_frame, textvariable=self.camera_var, 
                                       values=[str(i) for i in range(5)], state="readonly", width=5)
        self.combo_camera.grid(row=2, column=1, sticky="w", padx=5)

        # Flip Options
        ttk.Label(settings_frame, text="Flip:").grid(row=3, column=0, sticky="w", pady=5)
        frame_flip = ttk.Frame(settings_frame)
        frame_flip.grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(frame_flip, text="Horiz", variable=self.flip_h_var).pack(side="left", padx=2)
        ttk.Checkbutton(frame_flip, text="Vert", variable=self.flip_v_var).pack(side="left", padx=2)

        # Swap Handedness
        ttk.Label(settings_frame, text="Swap:").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Checkbutton(settings_frame, text="Swap L/R Labels", variable=self.swap_hands_var).grid(row=4, column=1, sticky="w", padx=5)

        # Batch Size
        ttk.Label(settings_frame, text="Batch:").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Spinbox(settings_frame, from_=1, to=200, textvariable=self.batch_size_var, width=5).grid(row=5, column=1, sticky="w", padx=5)

        # Continuous collecting
        ttk.Label(settings_frame, text="Mode:").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Checkbutton(settings_frame, text="Continuous (auto-record)", variable=self.continuous_var).grid(row=6, column=1, sticky="w", padx=5)

        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=20)
        
        self.btn_launch = ttk.Button(control_frame, text="LAUNCH COLLECTOR", command=self.start_collection)
        self.btn_launch.pack(fill="x", pady=5)
        
        # Add a runtime Record button
        self.btn_record = ttk.Button(control_frame, text="START RECORDING (Enter/Space)", command=self.on_trigger)
        self.btn_record.pack(fill="x", pady=5)
        self.btn_record.state(['disabled'])

        # Initialize
        self.update_gestures()

        # Bind keys to root window (works when tkinter window is focused)
        # Return "break" to prevent default button activation
        self.root.bind('<Return>', self._on_key_trigger)
        self.root.bind('<space>', self._on_key_trigger)
        self.root.bind('<Key-f>', self.toggle_flip)
        self.root.bind('<Key-F>', self.toggle_flip)
        self.root.bind('<Key-s>', self.toggle_swap)
        self.root.bind('<Key-S>', self.toggle_swap)
        self.root.bind('<Key-q>', self._on_quit_key)
        self.root.bind('<Key-Q>', self._on_quit_key)
        self.root.bind('<Escape>', self._on_quit_key)

    def _on_key_trigger(self, event=None):
        """Handle Enter/Space key - trigger recording and prevent default button behavior."""
        if self.is_running:
            self.trigger_record = True
            print("[Trigger] Recording triggered!")
        return "break"  # Prevent default tkinter behavior (button activation)

    def _on_quit_key(self, event=None):
        """Handle Q/ESC key - only quit if collector is running."""
        if self.is_running:
            self.stop_flag = True

    def update_gestures(self):
        hand = self.hand_var.get()
       
        gestures = self.config.model.gestures
     
        self.combo_gesture['values'] = gestures
        if gestures:
            self.combo_gesture.current(0)
            
    def set_trigger(self, val):
        """Set trigger flag - checked in process loop."""
        self.trigger_record = val

    def get_next_sequence_id(self, action_path):
        if not os.path.exists(action_path):
            return 0
        dirs = [int(d) for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d)) and d.isdigit()]
        if not dirs:
            return 0
        return max(dirs) + 1

    def extract_keypoints(self, results, apply_x_flip_canon=False):
        """Extract keypoints from the detected hand.

        Forces the recorded hand to match self.hand_selected regardless
        of what MediaPipe's handedness classifier reports.
        """
        if not results.multi_hand_landmarks:
            return np.zeros(21 * 4)

        # If multiple hands detected, pick the one matching our selected label
        # (accounting for flip/swap). If only one hand, just use it.
        hand_landmarks = results.multi_hand_landmarks[0]

        if results.multi_handedness and len(results.multi_hand_landmarks) > 1:
            target = self.hand_selected  # "right" or "left"
            flip_h = self.flip_h_var.get()
            swap_hands = self.swap_hands_var.get()

            for idx, handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label.lower()
                if flip_h:
                    label = "right" if label == "left" else "left"
                if swap_hands:
                    label = "right" if label == "left" else "left"
                if label == target:
                    hand_landmarks = results.multi_hand_landmarks[idx]
                    break

        kp = np.array([[res.x, res.y, res.z, 0.0] for res in hand_landmarks.landmark])

        # Apply x-flip canonicalization for left hand to match right hand orientation
        if apply_x_flip_canon:
            kp[:, 0] = 1.0 - kp[:, 0]

        return kp.flatten()

    def on_trigger(self, event=None):
        """Trigger recording - works from both Tkinter and OpenCV key events."""
        if self.is_running:
            # Always set trigger, even if already recording (will start new after current finishes)
            self.trigger_record = True
            print("[Trigger] Recording triggered!")

    def toggle_flip(self, event=None):
        if self.is_running:
            self.flip_h_var.set(not self.flip_h_var.get())

    def toggle_swap(self, event=None):
        if self.is_running:
            self.swap_hands_var.set(not self.swap_hands_var.get())

    def start_collection(self):
        """Initialize camera and start the Tkinter-driven loop"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            return # Already running

        gesture = self.gesture_var.get()
        if not gesture:
            messagebox.showwarning("Warning", "Please select a gesture")
            return
            
        try:
            cam_idx = int(self.camera_var.get())
        except ValueError:
            cam_idx = 0
            
        # Reset flags at start
        self.stop_flag = False
        self.trigger_record = False

        self.target_batch = self.batch_size_var.get()
        self.hand_selected = self.hand_var.get()
        
        self.DATA_PATH = Path("data/raw")
        self.base_path = self.DATA_PATH / f"{self.hand_selected}_mp_data" / gesture
        self.current_sequence_id = self.get_next_sequence_id(self.base_path)
        
        self.cap = cv2.VideoCapture(cam_idx)
        # Match real-time detection: capture at 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Match real-time detection resolutions
        self._display_width = 640
        self._display_height = 360
        self._mp_process_width = 320
        self._mp_process_height = 180

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {cam_idx}")
            return

        self.btn_record.state(['!disabled'])
        self.btn_launch.state(['disabled']) # Disable launch button while running

        self.sequence_length = self.config.data.sequence_length
        self.batch_count = 0
        self.frame_num = 0
        self.is_recording = False
        self.is_running = True
        self.zero_frame_count = 0
        self.trigger_record = False  # Reset trigger flag

        # FPS tracking and capping (use config value for consistency)
        self.target_fps = getattr(self.config.data, 'target_fps', 20.0)  # From config
        self.frame_interval = 1.0 / self.target_fps  # Time between frames (50ms for 20 FPS)
        self.next_frame_time = time.time()  # When next frame should be captured
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        print(f"[Collector] Target FPS: {self.target_fps}")
        
        # Initialize MediaPipe Hands
        self.hands_module = self.mp_hands.Hands(
            min_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
            max_num_hands=1,
            model_complexity=self.config.mediapipe.model_complexity
        )

        print("[Collector] Keys: Enter/Space=Record, F=Flip, S=Swap, Q=Quit (focus OpenCV window)")

        # Start the loop
        self.process_frame()

    def stop_collection(self, event=None):
        """Cleanup and close camera"""
        if hasattr(self, 'is_running') and self.is_running:
            self.is_running = False

            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hands_module') and self.hands_module:
                self.hands_module.close()
                self.hands_module = None

            cv2.destroyAllWindows()

            self.btn_record.state(['disabled'])
            self.btn_launch.state(['!disabled'])  # Re-enable launch
            self.stop_flag = False
            self.trigger_record = False
            print("[Collection] Stopped. Ready to start again.")

    def _check_opencv_keys(self):
        """Check for OpenCV key presses - called frequently."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
            self.stop_flag = True
        elif key == 13 or key == 32:  # Enter or Space
            self.trigger_record = True
            print("[Trigger] Recording triggered!")
        elif key == ord('f') or key == ord('F'):
            self.flip_h_var.set(not self.flip_h_var.get())
        elif key == ord('s') or key == ord('S'):
            self.swap_hands_var.set(not self.swap_hands_var.get())

    def process_frame(self):
        """Single frame processing scheduled by root.after"""
        if not self.is_running or not self.cap.isOpened() or self.stop_flag:
            self.stop_collection()
            return

        current_time = time.time()

        # Always check for key presses (even during rate limiting wait)
        self._check_opencv_keys()

        # Check if it's time for next frame (rate limiting)
        if current_time < self.next_frame_time:
            # Not time yet, but keep polling keys frequently
            self.root.after(5, self.process_frame)  # Poll every 5ms for responsive keys
            return

        # Update next frame time (increment to maintain exact interval)
        self.next_frame_time += self.frame_interval
        # Prevent drift if we fell behind
        if current_time - self.next_frame_time > self.frame_interval:
            self.next_frame_time = current_time + self.frame_interval

        ret, frame = self.cap.read()
        if not ret:
            # Don't stop on a single failed read — camera may need a moment
            self.root.after(10, self.process_frame)
            return

        # Calculate actual FPS
        self.fps_counter += 1
        elapsed = current_time - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time

        # Get current settings
        flip_h = self.flip_h_var.get()
        flip_v = self.flip_v_var.get()
        swap_hands = self.swap_hands_var.get()

        # Match real-time detection pipeline:
        # 1. Resize to display resolution (640x360)
        frame = cv2.resize(frame, (self._display_width, self._display_height), interpolation=cv2.INTER_NEAREST)

        # 2. Flip on small frame
        if flip_h:
            frame = cv2.flip(frame, 1)
        if flip_v:
            frame = cv2.flip(frame, 0)

        # 3. Create even smaller frame for MediaPipe (320x180) — same as detection
        frame_mp = cv2.resize(frame, (self._mp_process_width, self._mp_process_height), interpolation=cv2.INTER_NEAREST)

        # 4. Run MediaPipe on small frame (matching detection pipeline)
        mp_image = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
        mp_image.flags.writeable = False
        results = self.hands_module.process(mp_image)
        mp_image.flags.writeable = True

        # Use display frame for drawing
        image = frame

        # Draw and Extract
        self.draw_debug(image, results, flip_h, swap_hands)

        # Collection Logic
        self.handle_collection(image, results, flip_h, swap_hands)

        # Draw FPS (top-right corner)
        h, w = image.shape[:2]
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(image, fps_text, (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('HandFlow Collector [Enter/Space=Record | F=Flip | S=Swap | Q=Quit]', image)

        # Schedule next iteration
        self.root.after(1, self.process_frame)

    def draw_debug(self, image, results, flip_h, swap_hands):
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                h, w, c = image.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                if results.multi_handedness:
                    label = results.multi_handedness[idx].classification[0].label
                    if flip_h: label = "Right" if label == "Left" else "Left"
                    if swap_hands: label = "Right" if label == "Left" else "Left"
                    cv2.putText(image, f"{label} Hand", (cx, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def handle_collection(self, image, results, flip_h, swap_hands):
        gesture = self.gesture_var.get()

        # Check trigger flag — only start if not already recording
        if self.trigger_record:
            if not self.is_recording:
                self.is_recording = True
            self.trigger_record = False  # Always consume the flag

        # Continuous mode or noise gesture = auto-collect
        continuous = self.continuous_var.get()
        is_noise_gesture = gesture.lower() in ['none', 'nonezoom', 'touch_hover', 'touch_hold']

        # Check if batch is done
        if self.batch_count >= self.target_batch:
            cv2.putText(image, "BATCH COMPLETE! (Enter/Space to restart)", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # Stop recording if it was somehow active
            self.is_recording = False
            return

        if not self.is_recording:
            # Auto-trigger for noise gestures or continuous mode
            if is_noise_gesture or continuous:
                self.is_recording = True
                
            # Standby UI
            cv2.putText(image, f"COLLECTING: {gesture}" + (" [AUTO]" if is_noise_gesture else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Batch: {self.batch_count}/{self.target_batch}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Next ID: {self.current_sequence_id}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # Recording UI
            cv2.circle(image, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(image, f"RECORDING {self.frame_num}/{self.sequence_length}", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save Logic
            save_path = self.base_path / str(self.current_sequence_id)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Apply x-flip canonicalization for left hand to match right hand orientation
            is_left_hand = self.hand_selected == "left"
            kp = self.extract_keypoints(results, apply_x_flip_canon=is_left_hand)
            
            if np.all(kp == 0):
                self.zero_frame_count += 1
                
            np.save(save_path / f"{self.frame_num}.npy", kp)
            self.frame_num += 1
            
            if self.frame_num >= self.sequence_length:
                if self.zero_frame_count > 12:
                    print(f"❌ Discarding {self.current_sequence_id}")
                    import shutil
                    if save_path.exists(): shutil.rmtree(save_path)
                else:
                    self.current_sequence_id += 1
                    self.batch_count += 1
                    print(f"Saved {self.current_sequence_id-1}")
                
                # Reset
                self.frame_num = 0
                self.zero_frame_count = 0
                self.is_recording = False

def main():
    root = tk.Tk()
    app = GUIDataCollector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
