"""HandFlow - gesture-controlled computer interaction via hand tracking and ArUco markers."""

__version__ = "1.0.0"
__author__ = "Huynh Huy"

from handflow.utils.config import Config, load_config
from handflow.utils.setting import Setting, load_setting, save_setting
from handflow.utils.logging import setup_logging

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "Setting", 
    "load_setting", 
    "save_setting",
    "setup_logging",
]
