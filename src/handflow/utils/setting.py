"""User-facing settings: gesture mappings, macro pad sets, camera config, and inference params."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

class ActionBinding(BaseModel): 
    type: str
    value: str = ""
    label: str = ""
    delay: float = 0.17  # Delay before this action executes (seconds)

class GestureMapping(BaseModel): 
    actions: list[ActionBinding] = []

class MacroPadButton(BaseModel):
    # Single action fields (for backwards compatibility)
    type: str = "none"
    value: str = "" 
    label: str = ""
    # Multi-action sequence (preferred for macros)
    actions: list[ActionBinding] = []
    
    def get_actions(self) -> list:
        """Get actions list, falling back to single action if no sequence defined."""
        if self.actions:
            return self.actions
        elif self.type != "none":
            return [ActionBinding(type=self.type, value=self.value, label=self.label)]
        return []

class MacroPadSet(BaseModel): 
    name: str = "Untitled"
    set_marker_id: int = 12  # Top-left marker ID that identifies this set
    buttons: dict[int, MacroPadButton] = {}

    def model_post_init(self, __context):
        # Ensure all 12 buttons exist (index 0-11)
        for button_idx in range(12):
            if button_idx not in self.buttons:
                self.buttons[button_idx] = MacroPadButton()

class CameraSettings(BaseModel):
    """Camera settinguration."""
    index: int = 0
    flip_horizontal: bool = True
    flip_vertical: bool = False
    swap_hands: bool = False
    source: str = "webcam"  # "webcam" or "esp32"
    esp32_serial_port: str = "/dev/cu.usbmodem101"
    esp32_baud_rate: int = 2000000

class MouseSettings(BaseModel):
    """Mouse control settings."""
    smoothing: float = 0.25
    base_sensitivity: float = 1.0
    poll_interval: float = 0.002
    inner_deadzone: float = 0.005
    outer_deadzone: float = 0.014
    activation_threshold: float = 0.06

class InferenceSettings(BaseModel):
    """Model inference settings."""
    confidence_threshold: float = 0.6
    stability_window: int = 5
    cooldown_frames: int = 12
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.2
    model_complexity: int = 1

class Setting(BaseModel):
    """Complete HandFlow settings."""
    # Gesture mappings: gesture_name -> GestureMapping
    gestures: dict[str, GestureMapping] = {}
    
    # Macro pad: up to 5 named sets
    macropad_sets: list[MacroPadSet] = []
    active_macropad_set: int
    macropad_enabled: bool = False
    
    # Component settings
    camera: CameraSettings = CameraSettings()
    mouse: MouseSettings = MouseSettings()
    inference: InferenceSettings = InferenceSettings()
    
    # UI preferences
    theme: str = "dark"
    window_geometry: str = "1000x700"

    # Screen overlay macropad mode (replaces cursor control when enabled)
    screen_overlay_macropad_enabled: bool = False

    # Dedicated screen overlay macropad set (uses marker IDs 20-27)
    # This is separate from paper macropad sets and has its own button configuration
    screen_overlay_macropad: Optional[MacroPadSet] = None

    # Knuckle macropad — palm-up hand turns knuckles into 6 buttons
    knuckle_macropad_enabled: bool = False
    knuckle_macropad_buttons: dict[int, MacroPadButton] = {}

    # Convenience accessors
    def get_gesture_actions(self, gesture: str) -> list[ActionBinding]:
        """Get actions for a gesture."""
        mapping = self.gestures.get(gesture)
        return mapping.actions if mapping else []
    
    def set_gesture_actions(self, gesture: str, actions: list[ActionBinding]):
        """Set actions for a gesture."""
        self.gestures[gesture] = GestureMapping(actions=actions)
    
    def get_active_macropad(self) -> Optional[MacroPadSet]:
        """Get the currently active macro pad set."""
        if self.macropad_sets:
            idx = min(self.active_macropad_set, len(self.macropad_sets) - 1)
            return self.macropad_sets[idx]
        return None
    
    def create_macropad_set(self, name: str, marker_id: Optional[int] = None) -> MacroPadSet:
        """Create a new macro pad set with unique marker ID.

        Args:
            name: Name for the new set
            marker_id: Optional specific marker ID. If None, auto-increments from 12.

        Returns:
            The newly created MacroPadSet
        """
        if len(self.macropad_sets) >= 12:  # Allow up to 12 sets
            raise ValueError("Maximum 12 macro pad sets allowed")

        if marker_id is None:
            # Auto-increment: find the next available ID starting from 12
            used_ids = {s.set_marker_id for s in self.macropad_sets}
            marker_id = 12  # Base marker ID for sets
            while marker_id in used_ids:
                marker_id += 1

        new_set = MacroPadSet(name=name, set_marker_id=marker_id)
        self.macropad_sets.append(new_set)
        return new_set
    
    def delete_macropad_set(self, index: int):
        """Delete a macro pad set."""
        if 0 <= index < len(self.macropad_sets):
            del self.macropad_sets[index]
            if self.active_macropad_set >= len(self.macropad_sets):
                self.active_macropad_set = max(0, len(self.macropad_sets) - 1)

    def get_macropad_set_ids(self) -> list[int]:
        """Get all unique marker IDs from configured macropad sets."""
        return [s.set_marker_id for s in self.macropad_sets]

    def get_screen_overlay_macropad(self) -> MacroPadSet:
        """Get the screen overlay macropad set, creating it if it doesn't exist."""
        if self.screen_overlay_macropad is None:
            self.screen_overlay_macropad = MacroPadSet(
                name="Screen Overlay",
                set_marker_id=20  # Screen overlay uses ID 20
            )
        return self.screen_overlay_macropad


def load_setting(setting_path: str | Path) -> Setting:
    """Load and validate settinguration from YAML file.

    Args:
        setting_path: Path to the YAML settinguration file.

    Returns:
        Validated setting object.
    """
    setting_path = Path(setting_path)

    if not setting_path.exists():
        raise FileNotFoundError(f"settinguration file not found: {setting_path}")

    with open(setting_path, "r") as f:
        raw_setting = yaml.safe_load(f)

    return Setting(**raw_setting)


def save_setting(setting: Setting, output_path: str | Path) -> None:
    """Save settinguration to YAML file.

    Args:
        setting: settinguration object to save.
        output_path: Path to save the YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(setting.model_dump(), f, default_flow_style=False, sort_keys=False)


