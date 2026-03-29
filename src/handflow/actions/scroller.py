"""Threaded continuous scroller with configurable speed and start delay."""

from __future__ import annotations

import threading
import time

import pyautogui


class ContinuousScroller:
    def __init__(self, speed: int = -6, interval: float = 0.02, start_delay: float = 0.5) -> None:
        self.speed = speed
        self.interval = interval
        self.start_delay = start_delay

        self._scrolling = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._first_start = True

    def start(self) -> None:
        """Start continuous scrolling."""
        with self._lock:
            if self._scrolling:
                return
            self._scrolling = True
            self._thread = threading.Thread(target=self._scroll_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop scrolling."""
        with self._lock:
            self._scrolling = False

        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

    def reset_first_start(self) -> None:
        """Reset first start flag (no delay on next start)."""
        self._first_start = True

    def _scroll_loop(self) -> None:
        """Main scroll loop."""
        # Apply delay on non-first starts
        if not self._first_start:
            start_time = time.time()
            while time.time() - start_time < self.start_delay:
                with self._lock:
                    if not self._scrolling:
                        return
                time.sleep(0.01)

        self._first_start = False

        while True:
            with self._lock:
                if not self._scrolling:
                    break

            pyautogui.scroll(self.speed)
            time.sleep(self.interval)

    @property
    def is_scrolling(self) -> bool:
        """Check if currently scrolling."""
        return self._scrolling


# Pre-configured scrollers
scroll_down = ContinuousScroller(speed=-8, interval=0.01, start_delay=0.5)
scroll_up = ContinuousScroller(speed=8, interval=0.01, start_delay=0.5)
