#!/usr/bin/env python3
"""Standalone detection window launcher for HandFlow."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(str(ROOT))

import customtkinter as ctk
from handflow.utils import load_setting
from handflow.actions import ActionExecutor
from handflow.app.detection_window import DetectionWindow


def main():
    from handflow.utils.logging import setup_logging
    setup_logging(level="INFO", log_file="logs/detection.log")

    ctk.set_appearance_mode("dark")

    setting = load_setting("config/handflow_setting.yaml")
    executor = ActionExecutor()

    root = ctk.CTk()
    root.withdraw()

    window = DetectionWindow(setting, executor)
    window.start()

    def check():
        try:
            if not window.winfo_exists() or not window._running:
                root.quit()
                return
        except Exception:
            root.quit()
            return
        root.after(500, check)

    root.after(1000, check)
    root.mainloop()


if __name__ == "__main__":
    main()
