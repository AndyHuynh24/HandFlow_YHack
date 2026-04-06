"""Entry point for the HandFlow application."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from handflow.utils.logging import setup_logging, get_logger


def main():
    setup_logging(level="INFO", log_file="logs/app.log")
    logger = get_logger("handflow.main")

    # Use NiceGUI (modern web UI) by default, fall back to CustomTkinter
    try:
        from handflow.app.web_app import main as nicegui_main
        logger.info("Starting HandFlow with NiceGUI interface...")
        nicegui_main()
    except ImportError:
        logger.info("NiceGUI not installed, falling back to CustomTkinter...")
        try:
            import customtkinter as ctk
            from handflow.app import HandFlowApp

            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            HandFlowApp().mainloop()
        except ImportError as e:
            print(f"Missing dependency: {e}")
            print("Install with: pip install nicegui pywebview")
            sys.exit(1)


if __name__ == "__main__":
    main()



