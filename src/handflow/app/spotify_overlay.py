"""Spotify overlay — shows current track with playback controls."""

import subprocess
import sys
import threading
import time
import tkinter as tk

from handflow.utils import get_logger


class SpotifyOverlay:
    """Floating overlay with Spotify track info and playback controls."""

    WIDTH = 480
    HEIGHT = 160

    def __init__(self):
        self.logger = get_logger("handflow.spotify_overlay")
        self._root = None
        self._window_created = False
        self._visible = False

        self._track = ""
        self._artist = ""
        self._playing = False
        self._info_dirty = False

        self._fetch_thread = None
        self._last_fetch_time = 0
        self._fetch_interval = 2.0

    def _create_window(self):
        if self._window_created:
            return
        try:
            self._root = tk.Toplevel()
            self._root.withdraw()
            self._root.overrideredirect(True)
            self._root.attributes("-topmost", True)
            self._root.config(bg="#1a1a1a")

            screen_w = self._root.winfo_screenwidth()
            screen_h = self._root.winfo_screenheight()
            x = (screen_w - self.WIDTH) // 2
            y = (screen_h - self.HEIGHT) // 2
            self._root.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

            self._frame = tk.Frame(self._root, bg="#1a1a1a", highlightthickness=0)
            self._frame.pack(fill="both", expand=True)

            self._build_ui()
            self._window_created = True
        except Exception as e:
            self.logger.error(f"[SpotifyOverlay] Create failed: {e}")

    def _build_ui(self):
        f = self._frame
        W, H = self.WIDTH, self.HEIGHT

        # Main container with rounded appearance
        bg = "#1a1a1a"
        accent = "#1DB954"

        # Top accent bar
        tk.Frame(f, bg=accent, height=3).pack(fill="x")

        # Content area
        content = tk.Frame(f, bg=bg, padx=20, pady=12)
        content.pack(fill="both", expand=True)

        # Top row: icon + track info
        top = tk.Frame(content, bg=bg)
        top.pack(fill="x")

        # Spotify icon
        icon_frame = tk.Frame(top, bg=accent, width=44, height=44)
        icon_frame.pack(side="left", padx=(0, 14))
        icon_frame.pack_propagate(False)
        tk.Label(icon_frame, text="♫", font=("Helvetica", 20), fg="white", bg=accent).place(relx=0.5, rely=0.5, anchor="center")

        # Track info
        info = tk.Frame(top, bg=bg)
        info.pack(side="left", fill="x", expand=True)

        self._track_label = tk.Label(info, text="Loading...", font=("Helvetica", 16, "bold"),
                                     fg="white", bg=bg, anchor="w")
        self._track_label.pack(fill="x")

        self._artist_label = tk.Label(info, text="", font=("Helvetica", 12),
                                      fg="#a0a0a0", bg=bg, anchor="w")
        self._artist_label.pack(fill="x")

        # Status
        self._status_label = tk.Label(top, text="", font=("Helvetica", 10),
                                      fg=accent, bg=bg)
        self._status_label.pack(side="right", padx=(10, 0))

        # Bottom row: playback controls
        controls = tk.Frame(content, bg=bg)
        controls.pack(fill="x", pady=(12, 0))

        btn_style = {"font": ("Helvetica", 14), "bg": "#2a2a2a", "fg": "white",
                     "activebackground": "#3a3a3a", "activeforeground": "white",
                     "bd": 0, "highlightthickness": 0, "cursor": "hand2",
                     "padx": 16, "pady": 4}

        self._prev_btn = tk.Button(controls, text="⏮  Prev", command=self._prev, **btn_style)
        self._prev_btn.pack(side="left", padx=(0, 6))

        self._play_btn = tk.Button(controls, text="▶  Play", command=self._play_pause,
                                   font=("Helvetica", 14, "bold"), bg=accent, fg="white",
                                   activebackground="#1ed760", activeforeground="white",
                                   bd=0, highlightthickness=0, cursor="hand2", padx=20, pady=4)
        self._play_btn.pack(side="left", padx=6)

        self._next_btn = tk.Button(controls, text="Next  ⏭", command=self._next, **btn_style)
        self._next_btn.pack(side="left", padx=(6, 0))

    def _update_display(self):
        if not self._track_label:
            return

        track = self._track[:35] + "..." if len(self._track) > 35 else self._track
        artist = self._artist[:45] + "..." if len(self._artist) > 45 else self._artist

        self._track_label.config(text=track or "No track")
        self._artist_label.config(text=artist or "")
        self._status_label.config(text="Playing" if self._playing else "Paused")

        if self._playing:
            self._play_btn.config(text="⏸  Pause", bg="#2a2a2a", activebackground="#3a3a3a")
        else:
            self._play_btn.config(text="▶  Play", bg="#1DB954", activebackground="#1ed760")

    def _spotify_cmd(self, cmd: str):
        """Send command to Spotify via AppleScript in background."""
        if sys.platform != "darwin":
            return
        def run():
            try:
                subprocess.run(
                    ["osascript", "-e", f'tell application "Spotify" to {cmd}'],
                    capture_output=True, timeout=3
                )
                # Refresh info after command
                self._fetch_info_bg()
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    def _play_pause(self):
        self._spotify_cmd("playpause")

    def _next(self):
        self._spotify_cmd("next track")

    def _prev(self):
        self._spotify_cmd("previous track")

    def _fetch_info_bg(self):
        if sys.platform != "darwin":
            self._track = "macOS only"
            self._info_dirty = True
            return
        try:
            script = '''
            tell application "System Events"
                if not (exists process "Spotify") then return "NOT_RUNNING|||"
            end tell
            tell application "Spotify"
                set t to name of current track
                set a to artist of current track
                set s to player state as string
                return t & "|||" & a & "|||" & s
            end tell
            '''
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=3
            )
            parts = result.stdout.strip().split("|||")
            if len(parts) >= 3:
                self._track = parts[0]
                self._artist = parts[1]
                self._playing = parts[2] == "playing"
            elif "NOT_RUNNING" in result.stdout:
                self._track = "Spotify not running"
                self._artist = ""
                self._playing = False
            else:
                self._track = "Unknown"
                self._artist = ""
                self._playing = False
        except Exception:
            self._track = "Could not connect"
            self._artist = ""
            self._playing = False
        self._info_dirty = True

    def show(self):
        if not self._window_created:
            self._create_window()
        if not self._root:
            return

        now = time.time()
        if now - self._last_fetch_time > self._fetch_interval:
            if self._fetch_thread is None or not self._fetch_thread.is_alive():
                self._last_fetch_time = now
                self._fetch_thread = threading.Thread(target=self._fetch_info_bg, daemon=True)
                self._fetch_thread.start()

        if self._info_dirty:
            self._update_display()
            self._info_dirty = False

        if not self._visible:
            self._root.deiconify()
            self._visible = True

    def hide(self):
        if self._root and self._visible:
            self._root.withdraw()
            self._visible = False

    def is_visible(self):
        return self._visible

    def update(self):
        if self._root:
            try:
                self._root.update_idletasks()
            except Exception:
                pass

    def destroy(self):
        if self._root:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
            self._window_created = False
