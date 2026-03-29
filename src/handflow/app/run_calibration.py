#!/usr/bin/env python3
"""Standalone calibration launcher for HandFlow."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(str(ROOT))

from handflow.utils import load_setting


def main():
    setting = load_setting("config/handflow_setting.yaml")

    import pyautogui
    from handflow.detector import ArUcoScreenDetector, ArUcoCalibrationUI

    sw, sh = pyautogui.size()
    detector = ArUcoScreenDetector(screen_width=sw, screen_height=sh)

    cam_idx = setting.camera.index
    ui = ArUcoCalibrationUI(detector, camera_id=cam_idx, settings=setting)
    ui.run()

    print("[Calibration] Done.")


if __name__ == "__main__":
    main()
