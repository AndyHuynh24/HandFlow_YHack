"""Main application window with tabbed interface for gesture mapping, macro pad config, and calibration."""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional, Callable, Dict, List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from handflow.utils import Setting, load_setting, save_setting

from handflow.detector import ActionBinding, MacroPadButton
from handflow.actions import ActionExecutor


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DEFAULT_SETTING_PATH = "config/handflow_setting.yaml"


class GestureMappingTab(ctk.CTkFrame):
    """Tab for mapping gestures to actions with multi-action macro support."""
    
    MAX_ACTIONS_PER_GESTURE = 10

    def __init__(self, parent, setting: Setting, executor: ActionExecutor):
        super().__init__(parent, fg_color="transparent")
        self.setting = setting
        self.executor = executor
        self.gesture_widgets: Dict[str, dict] = {}

        # Get available actions
        actions = ActionExecutor.get_available_actions()
        self.action_names = [a[0] for a in actions]
        self.action_types = {a[0]: a[1] for a in actions}
        self.type_to_name = {a[1]: a[0] for a in actions}

        self._build_ui()

    def _build_ui(self):
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            header_frame,
            text="Gesture to Action Mapping",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left")

        # Save button in header
        ctk.CTkButton(
            header_frame,
            text="Save Mappings",
            command=self._save_mappings,
            width=130,
            height=32
        ).pack(side="right")

        ctk.CTkLabel(
            self,
            text="Assign multiple actions to gestures. Each action executes in sequence with configurable delays.",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(0, 10))

        # Scrollable frame for gestures
        scroll_frame = ctk.CTkScrollableFrame(self, label_text="")
        scroll_frame.pack(fill="both", expand=True, padx=15, pady=5)

        # Separate left and right gestures
        left_gestures = [g for g in self.setting.gestures if g.startswith("Left")]
        right_gestures = [g for g in self.setting.gestures if g.startswith("Right")]

        # Left hand section
        self._create_gesture_section(scroll_frame, "Left Hand", left_gestures, "#ff6b6b")

        # Separator
        ctk.CTkLabel(scroll_frame, text="", height=10).pack()

        # Right hand section
        self._create_gesture_section(scroll_frame, "Right Hand", right_gestures, "#4a9eff")

    def _create_gesture_section(self, parent, title: str, gestures: List[str], color: str):
        """Create a section for a hand's gestures."""
        section_frame = ctk.CTkFrame(parent)
        section_frame.pack(fill="x", pady=5)

        # Section header
        header = ctk.CTkFrame(section_frame, fg_color=color, corner_radius=5)
        header.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            header,
            text=f"  {title}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        ).pack(side="left", pady=5)

        # Gesture rows
        for gesture in gestures:
            self._create_gesture_widget(section_frame, gesture)

    def _create_gesture_widget(self, parent, gesture: str):
        """Create a multi-action widget for a single gesture."""
        # Main container for this gesture
        gesture_frame = ctk.CTkFrame(parent)
        gesture_frame.pack(fill="x", pady=3, padx=10)

        # Header row with gesture name and add button
        header_row = ctk.CTkFrame(gesture_frame, fg_color="transparent")
        header_row.pack(fill="x", pady=(5, 2))

        gesture_display = gesture.split("_", 1)[1].replace("_", " ").title()

        ctk.CTkLabel(
            header_row,
            text=gesture_display,
            width=130,
            anchor="w",
            font=ctk.CTkFont(weight="bold", size=13)
        ).pack(side="left", padx=5)

        # Add action button
        add_btn = ctk.CTkButton(
            header_row,
            text="+ Add Action",
            width=100,
            height=26,
            font=ctk.CTkFont(size=11),
            command=lambda g=gesture: self._add_action_row(g)
        )
        add_btn.pack(side="right", padx=5)

        # Container for action rows
        actions_container = ctk.CTkFrame(gesture_frame, fg_color="transparent")
        actions_container.pack(fill="x", padx=10, pady=(0, 5))

        # Initialize widget storage
        self.gesture_widgets[gesture] = {
            'container': actions_container,
            'action_rows': [],  # List of dicts for each action row
            'add_btn': add_btn
        }

        # Load existing actions
        current_actions = self.setting.get_gesture_actions(gesture)
        if current_actions:
            for action in current_actions:
                self._add_action_row(gesture, action)
        else:
            # Add one empty row by default
            self._add_action_row(gesture)

    def _add_action_row(self, gesture: str, action=None):
        """Add an action row to a gesture."""
        widgets = self.gesture_widgets[gesture]
        
        if len(widgets['action_rows']) >= self.MAX_ACTIONS_PER_GESTURE:
            messagebox.showwarning("Limit Reached", f"Maximum {self.MAX_ACTIONS_PER_GESTURE} actions per gesture.")
            return

        container = widgets['container']
        row_index = len(widgets['action_rows'])

        # Create row frame
        row_frame = ctk.CTkFrame(container, fg_color="transparent")
        row_frame.pack(fill="x", pady=2)

        # Row number label
        ctk.CTkLabel(
            row_frame,
            text=f"{row_index + 1}.",
            width=20,
            font=ctk.CTkFont(size=11)
        ).pack(side="left")

        # Action dropdown
        current_type = action.type if action else "none"
        current_name = self.type_to_name.get(current_type, "None")
        
        action_var = ctk.StringVar(value=current_name)
        dropdown = ctk.CTkComboBox(
            row_frame,
            values=self.action_names,
            variable=action_var,
            width=140,
            height=28,
            font=ctk.CTkFont(size=11)
        )
        dropdown.pack(side="left", padx=3)

        # Value entry
        current_value = action.value if action else ""
        value_var = ctk.StringVar(value=current_value)
        value_entry = ctk.CTkEntry(
            row_frame,
            textvariable=value_var,
            placeholder_text="Value...",
            width=140,
            height=28,
            font=ctk.CTkFont(size=11)
        )
        value_entry.pack(side="left", padx=3)

        # Delay entry
        current_delay = action.delay if action else 0.17
        delay_var = ctk.StringVar(value=str(current_delay))
        
        ctk.CTkLabel(row_frame, text="Delay:", font=ctk.CTkFont(size=10)).pack(side="left", padx=(5, 2))
        delay_entry = ctk.CTkEntry(
            row_frame,
            textvariable=delay_var,
            width=50,
            height=28,
            font=ctk.CTkFont(size=11)
        )
        delay_entry.pack(side="left")
        ctk.CTkLabel(row_frame, text="s", font=ctk.CTkFont(size=10)).pack(side="left")

        # Configure button
        config_btn = ctk.CTkButton(
            row_frame,
            text="...",
            width=30,
            height=28,
            command=lambda: self._configure_action(action_var, value_var)
        )
        config_btn.pack(side="left", padx=3)

        # Delete button
        delete_btn = ctk.CTkButton(
            row_frame,
            text="✕",
            width=30,
            height=28,
            fg_color="#a51d2d",
            hover_color="#8a1829",
            command=lambda g=gesture, rf=row_frame: self._remove_action_row(g, rf)
        )
        delete_btn.pack(side="left", padx=2)

        # Store row data
        row_data = {
            'frame': row_frame,
            'action_var': action_var,
            'value_var': value_var,
            'delay_var': delay_var,
            'dropdown': dropdown,
            'value_entry': value_entry
        }
        widgets['action_rows'].append(row_data)

        # Bind dropdown change
        dropdown.configure(command=lambda choice, rd=row_data: self._on_action_change(rd, choice))
        self._on_action_change(row_data, current_name)

    def _remove_action_row(self, gesture: str, row_frame):
        """Remove an action row from a gesture."""
        widgets = self.gesture_widgets[gesture]
        
        # Find and remove the row
        for i, row_data in enumerate(widgets['action_rows']):
            if row_data['frame'] == row_frame:
                row_frame.destroy()
                widgets['action_rows'].pop(i)
                break
        
        # Renumber remaining rows
        for i, row_data in enumerate(widgets['action_rows']):
            # Find and update the number label
            for child in row_data['frame'].winfo_children():
                if isinstance(child, ctk.CTkLabel):
                    child.configure(text=f"{i + 1}.")
                    break

    def _on_action_change(self, row_data: dict, choice: str):
        """Handle action dropdown change."""
        action_type = self.action_types.get(choice, "none")

        # Configure value entry based on action type
        if action_type in ['none', 'leftclick', 'rightclick', 'doubleclick',
                          'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out',
                          'media_play', 'media_next', 'media_prev',
                          'volume_up', 'volume_down', 'volume_mute',
                          'screenshot', 'minimize', 'maximize',
                          'desktop_left', 'desktop_right']:
            row_data['value_var'].set("")
            row_data['value_entry'].configure(placeholder_text="N/A", state="disabled")
        else:
            row_data['value_entry'].configure(state="normal")
            if action_type == 'shortcut':
                row_data['value_entry'].configure(placeholder_text="e.g. cmd+c")
            elif action_type == 'text':
                row_data['value_entry'].configure(placeholder_text="Text to type...")
            elif action_type == 'file':
                row_data['value_entry'].configure(placeholder_text="File path...")

    def _configure_action(self, action_var: ctk.StringVar, value_var: ctk.StringVar):
        """Open configuration dialog based on action type."""
        action_name = action_var.get()
        action_type = self.action_types.get(action_name, 'none')

        if action_type == 'shortcut':
            self._capture_shortcut(value_var)
        elif action_type == 'file':
            self._select_file(value_var)
        elif action_type == 'text':
            self._enter_text(value_var)
        else:
            messagebox.showinfo("Info", f"No configuration needed for {action_name}")

    def _capture_shortcut(self, value_var: ctk.StringVar):
        """Capture keyboard shortcut."""
        popup = ctk.CTkToplevel(self)
        popup.title("Capture Shortcut")
        popup.geometry("380x160")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()
        popup.focus()

        ctk.CTkLabel(
            popup,
            text="Press your shortcut keys:",
            font=ctk.CTkFont(size=14)
        ).pack(pady=15)

        shortcut_var = ctk.StringVar()
        display = ctk.CTkEntry(popup, textvariable=shortcut_var, width=280, state="readonly")
        display.pack(pady=5)
        display.focus_set()

        pressed = []
        pressed_set = set()

        KEY_TRANSLATE = {
            "Meta_L": "cmd", "Meta_R": "cmd",
            "Control_L": "ctrl", "Control_R": "ctrl",
            "Shift_L": "shift", "Shift_R": "shift",
            "Alt_L": "alt", "Alt_R": "alt",
            "Return": "enter", "BackSpace": "backspace",
            "Tab": "tab", "Escape": "escape",
            "space": "space"
        }

        def on_key(event):
            key = KEY_TRANSLATE.get(event.keysym, event.keysym.lower())
            if key not in pressed_set:
                pressed_set.add(key)
                pressed.append(key)
                shortcut_var.set("+".join(pressed))

        display.bind("<KeyPress>", on_key)

        def save():
            if pressed:
                value_var.set("+".join(pressed))
            popup.destroy()

        def clear():
            pressed.clear()
            pressed_set.clear()
            shortcut_var.set("")

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(btn_frame, text="Clear", width=80, command=clear).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Save", width=100, command=save).pack(side="left", padx=5)

    def _select_file(self, value_var: ctk.StringVar):
        """Open file selection dialog."""
        path = filedialog.askopenfilename(
            title="Select File or Application",
            filetypes=[
                ("Applications", "*.app"),
                ("All files", "*.*")
            ]
        )
        if path:
            value_var.set(path)

    def _enter_text(self, value_var: ctk.StringVar):
        """Open multi-line text entry dialog."""
        popup = ctk.CTkToplevel(self)
        popup.title("Enter Text")
        popup.geometry("450x280")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        ctk.CTkLabel(
            popup, 
            text="Enter text to type (supports multiple lines):", 
            font=ctk.CTkFont(size=14)
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            popup,
            text="Each line will be typed followed by Enter key",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(0, 10))

        # Multi-line text box
        textbox = ctk.CTkTextbox(popup, width=400, height=120)
        textbox.pack(pady=5, padx=20)

        # Allow Tab key to insert actual tab character instead of changing focus
        def insert_tab(event):
            textbox.insert("insert", "\t")
            return "break"  # Prevent default focus-change behavior
        textbox.bind("<Tab>", insert_tab)

        # Load existing text (convert \n and \t back to actual chars for display)
        current_text = value_var.get().replace("\\n", "\n").replace("\\t", "\t")
        if current_text:
            textbox.insert("1.0", current_text)
        textbox.focus_set()

        def save():
            # Get text and preserve newlines as \n for storage
            text = textbox.get("1.0", "end-1c")  # end-1c removes trailing newline
            value_var.set(text)
            popup.destroy()

        ctk.CTkButton(popup, text="Save", width=100, command=save).pack(pady=15)

    def _save_mappings(self):
        """Save all gesture mappings."""
        for gesture, widgets in self.gesture_widgets.items():
            actions = []
            for row_data in widgets['action_rows']:
                action_name = row_data['action_var'].get()
                action_type = self.action_types.get(action_name, 'none')
                custom_value = row_data['value_var'].get()
                
                # Parse delay
                try:
                    delay = float(row_data['delay_var'].get())
                except ValueError:
                    delay = 0.17
                
                actions.append(ActionBinding(
                    type=action_type,
                    value=custom_value,
                    delay=delay
                ))
            
            self.setting.set_gesture_actions(gesture, actions)

        save_setting(self.setting, DEFAULT_SETTING_PATH)
        messagebox.showinfo("Saved", "Gesture mappings saved successfully!")


class MacroPadTab(ctk.CTkFrame):
    """Tab for ArUco macro pad configuration."""

    def __init__(self, parent, setting: Setting, executor: ActionExecutor):
        super().__init__(parent, fg_color="transparent")
        self.setting = setting
        self.executor = executor
        self.button_widgets: Dict[int, dict] = {}

        # Get available actions
        actions = ActionExecutor.get_available_actions()
        self.action_names = [a[0] for a in actions]
        self.action_types = {a[0]: a[1] for a in actions}

        self._build_ui()

    def _build_ui(self):
        # Track if editing screen overlay vs paper macropad
        self._editing_screen_overlay = False

        # Header with set selector
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(15, 5))

        self.tab_title_label = ctk.CTkLabel(
            header_frame,
            text="Macro Pad",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.tab_title_label.pack(side="left")

        # Set selector - include both paper sets AND screen overlay (ID 20)
        set_display_names = self._get_all_set_display_names()

        self.set_var = ctk.StringVar(value=set_display_names[0] if set_display_names else "Default")
        self.set_dropdown = ctk.CTkComboBox(
            header_frame,
            values=set_display_names,
            variable=self.set_var,
            width=220,
            command=self._on_set_changed
        )
        self.set_dropdown.pack(side="right", padx=5)

        ctk.CTkLabel(header_frame, text="Select Set:").pack(side="right", padx=5)

        # Description
        ctk.CTkLabel(
            self,
            text="Configure ArUco markers as touch-activated buttons. Print the PDF and use touch gestures to activate.",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(0, 10))

        # Main content area
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=5)

        # Left: Button grid configuration
        left_panel = ctk.CTkFrame(content)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Button grid header
        grid_header = ctk.CTkFrame(left_panel, fg_color="#2b5797", corner_radius=5)
        grid_header.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            grid_header,
            text="  Button Configuration (4x2 Grid)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        ).pack(side="left", pady=8)

        # Show current set ID in header
        current_set = self.setting.get_active_macropad()
        set_id = current_set.set_marker_id if current_set else "?"
        self.set_id_label = ctk.CTkLabel(
            grid_header,
            text=f"TL Marker ID: {set_id}  ",
            font=ctk.CTkFont(size=11),
            text_color="#aaccff"
        )
        self.set_id_label.pack(side="right", pady=8)

        # Scrollable button config area
        buttons_scroll = ctk.CTkScrollableFrame(left_panel, label_text="")
        buttons_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        # Create button configs in 4 columns (3 rows x 4 cols = 12 buttons)
        for row in range(3):
            row_frame = ctk.CTkFrame(buttons_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=3)

            for col in range(4):
                # Button index 0-11 (matches settings and detector)
                button_idx = row * 4 + col
                self._create_button_config(row_frame, button_idx)

        # Right: Actions panel
        right_panel = ctk.CTkFrame(content, width=220)
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        right_panel.pack_propagate(False)

        # Actions header
        ctk.CTkLabel(
            right_panel,
            text="Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(15, 10))

        # Action buttons
        ctk.CTkButton(
            right_panel,
            text="Save Configuration",
            command=self._save_config,
            width=180,
            height=35
        ).pack(pady=5)

        ctk.CTkButton(
            right_panel,
            text="Export Macro Pad PDF",
            command=self._export_macropad_pdf,
            width=180,
            height=35,
            fg_color="#2b7a0b",
            hover_color="#1e5a08"
        ).pack(pady=5)

        ctk.CTkButton(
            right_panel,
            text="Export Calibration PDF",
            command=self._export_calibration_pdf,
            width=180,
            height=35,
            fg_color="#5a5a5a",
            hover_color="#4a4a4a"
        ).pack(pady=5)

        ctk.CTkButton(
            right_panel,
            text="Export All (Combined)",
            command=self._export_combined_pdf,
            width=180,
            height=35,
            fg_color="#1a5fb4",
            hover_color="#144a8f"
        ).pack(pady=5)

        # Separator
        ctk.CTkLabel(right_panel, text="", height=10).pack()

        # Set management
        ctk.CTkLabel(
            right_panel,
            text="Set Management",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(10, 5))

        ctk.CTkButton(
            right_panel,
            text="+ New Set",
            command=self._create_new_set,
            width=180,
            height=30
        ).pack(pady=3)

        ctk.CTkButton(
            right_panel,
            text="Rename Set",
            command=self._rename_set,
            width=180,
            height=30
        ).pack(pady=3)

        ctk.CTkButton(
            right_panel,
            text="Delete Set",
            command=self._delete_set,
            width=180,
            height=30,
            fg_color="#a51d2d",
            hover_color="#8a1829"
        ).pack(pady=3)

        # Enable/Disable toggle
        self.macropad_enabled_var = ctk.BooleanVar(value=self.setting.macropad_enabled)
        ctk.CTkSwitch(
            right_panel,
            text="Enable Macro Pad",
            variable=self.macropad_enabled_var,
            command=self._toggle_macropad
        ).pack(pady=(20, 10))

        # Separator
        ctk.CTkLabel(right_panel, text="", height=5).pack()

        # Screen Overlay Section
        overlay_header = ctk.CTkFrame(right_panel, fg_color="#1a5fb4", corner_radius=5)
        overlay_header.pack(fill="x", padx=5, pady=(10, 5))

        ctk.CTkLabel(
            overlay_header,
            text="  Screen Overlay",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white"
        ).pack(side="left", pady=5)

        # Screen Overlay Enable toggle
        self.screen_overlay_var = ctk.BooleanVar(value=self.setting.screen_overlay_macropad_enabled)
        screen_overlay_switch = ctk.CTkSwitch(
            right_panel,
            text="Enable Overlay",
            variable=self.screen_overlay_var,
            command=self._toggle_screen_overlay
        )
        screen_overlay_switch.pack(pady=5)

        # Screen overlay info
        ctk.CTkLabel(
            right_panel,
            text="Marker IDs: 20-27",
            text_color="#4a9eff",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(pady=(5, 0))

        ctk.CTkLabel(
            right_panel,
            text="TL:20 TR:21 ML:22 MR:23\nBL:24 BR:25 BL2:26 BR2:27",
            text_color="#888888",
            font=ctk.CTkFont(size=9),
            justify="center"
        ).pack(pady=(2, 0))

        ctk.CTkLabel(
            right_panel,
            text="Select '⬛ Screen Overlay'\nfrom dropdown to configure.\nCan use with paper macropad.",
            text_color="gray",
            font=ctk.CTkFont(size=10),
            justify="center"
        ).pack(pady=(5, 10))

    def _create_button_config(self, parent, button_idx: int):
        """Create configuration widget for a single button with multi-action support."""
        frame = ctk.CTkFrame(parent, width=280, height=100)
        frame.pack(side="left", padx=5, pady=3)
        frame.pack_propagate(False)

        # Header with button number
        header = ctk.CTkFrame(frame, height=25)
        header.pack(fill="x", padx=5, pady=(5, 2))

        ctk.CTkLabel(
            header,
            text=f"Button {button_idx + 1}",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text=f"idx {button_idx}",
            font=ctk.CTkFont(size=10),
            text_color="#4a9eff"
        ).pack(side="right")

        # Get current binding
        current_set = self.setting.get_active_macropad()
        current_btn = current_set.buttons.get(button_idx) if current_set else None
        current_label = current_btn.label if current_btn else ""
        
        # Get actions count for display
        actions_count = len(current_btn.get_actions()) if current_btn else 0

        # Label entry
        label_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Button Label",
            width=160,
            height=25,
            font=ctk.CTkFont(size=10)
        )
        label_entry.pack(padx=5, pady=2)
        if current_label:
            label_entry.insert(0, current_label)

        # Actions info and edit button row
        action_row = ctk.CTkFrame(frame, fg_color="transparent")
        action_row.pack(fill="x", padx=5, pady=2)

        actions_label = ctk.CTkLabel(
            action_row,
            text=f"{actions_count} action(s)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        actions_label.pack(side="left")

        # Edit actions button
        edit_btn = ctk.CTkButton(
            action_row,
            text="Edit Actions",
            width=100,
            height=25,
            font=ctk.CTkFont(size=10),
            command=lambda idx=button_idx: self._open_button_actions_dialog(idx)
        )
        edit_btn.pack(side="right")

        self.button_widgets[button_idx] = {
            'label': label_entry,
            'actions_label': actions_label,
            'actions': current_btn.get_actions() if current_btn else []  # Store actions list
        }

    def _open_button_actions_dialog(self, button_idx: int):
        """Open a dialog to configure multiple actions for a button."""
        popup = ctk.CTkToplevel(self)
        popup.title(f"Configure Button {button_idx + 1} Actions")
        popup.geometry("650x520")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        # Header
        ctk.CTkLabel(
            popup,
            text=f"Actions for Button {button_idx + 1}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            popup,
            text="Add multiple actions that execute in sequence with configurable delays.",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(0, 10))

        # Scrollable actions area
        actions_scroll = ctk.CTkScrollableFrame(popup, label_text="", height=280)
        actions_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        # Track action rows in this dialog
        dialog_action_rows = []
        
        # Type to name mapping
        type_to_name = {a[1]: a[0] for a in ActionExecutor.get_available_actions()}

        def add_action_row(action=None):
            """Add an action row to the dialog."""
            if len(dialog_action_rows) >= 10:
                messagebox.showwarning("Limit", "Maximum 10 actions per button.")
                return

            row_index = len(dialog_action_rows)
            row_frame = ctk.CTkFrame(actions_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=3)

            # Row number
            ctk.CTkLabel(
                row_frame,
                text=f"{row_index + 1}.",
                width=25,
                font=ctk.CTkFont(size=11)
            ).pack(side="left")

            # Action dropdown
            current_type = action.type if action else "none"
            current_name = type_to_name.get(current_type, "None")
            
            action_var = ctk.StringVar(value=current_name)
            dropdown = ctk.CTkComboBox(
                row_frame,
                values=self.action_names,
                variable=action_var,
                width=130,
                height=28,
                font=ctk.CTkFont(size=10)
            )
            dropdown.pack(side="left", padx=3)

            # Value entry
            current_value = action.value if action else ""
            value_var = ctk.StringVar(value=current_value)
            value_entry = ctk.CTkEntry(
                row_frame,
                textvariable=value_var,
                placeholder_text="Value...",
                width=130,
                height=28,
                font=ctk.CTkFont(size=10)
            )
            value_entry.pack(side="left", padx=3)

            # Delay
            current_delay = action.delay if action else 0.17
            delay_var = ctk.StringVar(value=str(current_delay))
            
            ctk.CTkLabel(row_frame, text="Delay:", font=ctk.CTkFont(size=9)).pack(side="left", padx=(5, 2))
            delay_entry = ctk.CTkEntry(
                row_frame,
                textvariable=delay_var,
                width=45,
                height=28,
                font=ctk.CTkFont(size=10)
            )
            delay_entry.pack(side="left")
            ctk.CTkLabel(row_frame, text="s", font=ctk.CTkFont(size=9)).pack(side="left")

            # Configure button
            cfg_btn = ctk.CTkButton(
                row_frame,
                text="...",
                width=28,
                height=28,
                command=lambda: self._configure_dialog_action(action_var, value_var)
            )
            cfg_btn.pack(side="left", padx=3)

            # Delete button
            def remove_row(rf=row_frame):
                for i, rd in enumerate(dialog_action_rows):
                    if rd['frame'] == rf:
                        rf.destroy()
                        dialog_action_rows.pop(i)
                        break
                # Renumber
                for i, rd in enumerate(dialog_action_rows):
                    for child in rd['frame'].winfo_children():
                        if isinstance(child, ctk.CTkLabel) and child.cget("text").endswith("."):
                            child.configure(text=f"{i + 1}.")
                            break

            delete_btn = ctk.CTkButton(
                row_frame,
                text="✕",
                width=28,
                height=28,
                fg_color="#a51d2d",
                hover_color="#8a1829",
                command=remove_row
            )
            delete_btn.pack(side="left", padx=2)

            row_data = {
                'frame': row_frame,
                'action_var': action_var,
                'value_var': value_var,
                'delay_var': delay_var
            }
            dialog_action_rows.append(row_data)

        # Load existing actions
        existing_actions = self.button_widgets[button_idx]['actions']
        if existing_actions:
            for action in existing_actions:
                add_action_row(action)
        else:
            add_action_row()  # Add one empty row

        # Add action button
        add_btn = ctk.CTkButton(
            popup,
            text="+ Add Action",
            width=120,
            height=30,
            command=lambda: add_action_row()
        )
        add_btn.pack(pady=10)

        # Save/Cancel buttons
        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=15)

        def save_actions():
            """Save actions and close dialog."""
            actions = []
            for row_data in dialog_action_rows:
                action_name = row_data['action_var'].get()
                action_type = self.action_types.get(action_name, 'none')
                value = row_data['value_var'].get()
                
                try:
                    delay = float(row_data['delay_var'].get())
                except ValueError:
                    delay = 0.17
                
                actions.append(ActionBinding(
                    type=action_type,
                    value=value,
                    delay=delay
                ))
            
            # Update stored actions
            self.button_widgets[button_idx]['actions'] = actions
            self.button_widgets[button_idx]['actions_label'].configure(
                text=f"{len(actions)} action(s)"
            )
            popup.destroy()

        ctk.CTkButton(btn_frame, text="Cancel", width=80, command=popup.destroy).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Save", width=100, command=save_actions).pack(side="left", padx=5)

    def _configure_dialog_action(self, action_var: ctk.StringVar, value_var: ctk.StringVar):
        """Configure action in dialog (shortcut capture, file select, etc.)."""
        action_name = action_var.get()
        action_type = self.action_types.get(action_name, 'none')

        if action_type == 'shortcut':
            self._capture_shortcut_for_button(value_var)
        elif action_type == 'file':
            path = filedialog.askopenfilename(title="Select File or Application")
            if path:
                value_var.set(path)
        elif action_type == 'text':
            self._enter_text_multiline(value_var)
        else:
            messagebox.showinfo("Info", f"No configuration needed for {action_name}")

    def _configure_button_action(self, button_idx: int, action_var: ctk.StringVar, value_var: ctk.StringVar):
        """Configure action for a button (legacy - redirects to dialog)."""
        self._open_button_actions_dialog(button_idx)

    def _capture_shortcut_for_button(self, value_var: ctk.StringVar):
        """Capture keyboard shortcut for button."""
        popup = ctk.CTkToplevel(self)
        popup.title("Capture Shortcut")
        popup.geometry("350x140")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        ctk.CTkLabel(popup, text="Press shortcut keys:", font=ctk.CTkFont(size=13)).pack(pady=12)

        shortcut_var = ctk.StringVar()
        display = ctk.CTkEntry(popup, textvariable=shortcut_var, width=250, state="readonly")
        display.pack(pady=5)
        display.focus_set()

        pressed = []

        KEY_TRANSLATE = {
            "Meta_L": "cmd", "Meta_R": "cmd",
            "Control_L": "ctrl", "Control_R": "ctrl",
            "Shift_L": "shift", "Shift_R": "shift",
            "Alt_L": "alt", "Alt_R": "alt"
        }

        def on_key(event):
            key = KEY_TRANSLATE.get(event.keysym, event.keysym.lower())
            if key not in pressed:
                pressed.append(key)
                shortcut_var.set("+".join(pressed))

        display.bind("<KeyPress>", on_key)

        def save():
            if pressed:
                value_var.set("+".join(pressed))
            popup.destroy()

        ctk.CTkButton(popup, text="Save", width=100, command=save).pack(pady=12)

    def _enter_text_multiline(self, value_var: ctk.StringVar):
        """Open multi-line text entry dialog for MacroPad buttons."""
        popup = ctk.CTkToplevel(self)
        popup.title("Enter Text")
        popup.geometry("450x280")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        ctk.CTkLabel(
            popup, 
            text="Enter text to type (supports multiple lines):", 
            font=ctk.CTkFont(size=14)
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            popup,
            text="Each line will be typed followed by Enter key",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(0, 10))

        # Multi-line text box
        textbox = ctk.CTkTextbox(popup, width=400, height=120)
        textbox.pack(pady=5, padx=20)

        # Allow Tab key to insert actual tab character instead of changing focus
        def insert_tab(event):
            textbox.insert("insert", "\t")
            return "break"  # Prevent default focus-change behavior
        textbox.bind("<Tab>", insert_tab)

        # Load existing text (convert \n and \t back to actual chars for display)
        current_text = value_var.get().replace("\\n", "\n").replace("\\t", "\t")
        if current_text:
            textbox.insert("1.0", current_text)
        textbox.focus_set()

        def save():
            # Get text and preserve newlines
            text = textbox.get("1.0", "end-1c")
            value_var.set(text)
            popup.destroy()

        ctk.CTkButton(popup, text="Save", width=100, command=save).pack(pady=15)

    def _get_all_set_display_names(self) -> List[str]:
        """Get display names for all sets including screen overlay."""
        names = []
        # Paper macropad sets
        for s in self.setting.macropad_sets:
            names.append(f"{s.name} (ID:{s.set_marker_id})")
        # Screen overlay set (always at the end)
        overlay_set = self.setting.get_screen_overlay_macropad()
        names.append(f"⬛ {overlay_set.name} (ID:{overlay_set.set_marker_id})")
        return names

    def _get_current_editing_set(self):
        """Get the currently selected set for editing (paper or screen overlay)."""
        if self._editing_screen_overlay:
            return self.setting.get_screen_overlay_macropad()
        else:
            return self.setting.get_active_macropad()

    def _on_set_changed(self, choice):
        """Handle set selection change."""
        # Check if screen overlay was selected
        overlay_set = self.setting.get_screen_overlay_macropad()
        overlay_display = f"⬛ {overlay_set.name} (ID:{overlay_set.set_marker_id})"

        if choice == overlay_display:
            self._editing_screen_overlay = True
            self._refresh_buttons()
            return

        # Otherwise, check paper macropad sets
        self._editing_screen_overlay = False
        sets = self.setting.macropad_sets
        for i, s in enumerate(sets):
            display_name = f"{s.name} (ID:{s.set_marker_id})"
            if display_name == choice:
                self.setting.active_macropad_set = i
                self._refresh_buttons()
                break

    def _refresh_buttons(self):
        """Refresh button widgets with current set data."""
        current_set = self._get_current_editing_set()
        if not current_set:
            return

        # Update set ID label in header
        if hasattr(self, 'set_id_label'):
            marker_info = f"TL Marker ID: {current_set.set_marker_id}"
            if self._editing_screen_overlay:
                marker_info += " (Screen)"
            self.set_id_label.configure(text=f"{marker_info}  ")

        # Update title to indicate what we're editing
        if hasattr(self, 'tab_title_label'):
            if self._editing_screen_overlay:
                self.tab_title_label.configure(text="Screen Overlay MacroPad")
            else:
                self.tab_title_label.configure(text="Paper Macro Pad")

        for button_idx, widgets in self.button_widgets.items():
            btn = current_set.buttons.get(button_idx)
            # Clear existing
            widgets['label'].delete(0, "end")
            if btn:
                if btn.label:
                    widgets['label'].insert(0, btn.label)
                actions = btn.get_actions()
                widgets['actions'] = actions
                widgets['actions_label'].configure(text=f"{len(actions)} action(s)")
            else:
                widgets['actions'] = []
                widgets['actions_label'].configure(text="0 action(s)")

    def _save_config(self):
        """Save current macro pad configuration."""
        current_set = self._get_current_editing_set()
        if not current_set:
            return

        for button_idx, widgets in self.button_widgets.items():
            label = widgets['label'].get()
            actions = widgets['actions']

            # Create MacroPadButton with actions list
            current_set.buttons[button_idx] = MacroPadButton(
                label=label,
                actions=actions
            )

        self.setting.macropad_enabled = self.macropad_enabled_var.get()
        save_setting(self.setting, DEFAULT_SETTING_PATH)

        messagebox.showinfo("Saved", "Macro pad configuration saved!")

    def _export_macropad_pdf(self):
        """Export macro pad as PDF."""
        current_set = self.setting.get_active_macropad()
        if not current_set:
            messagebox.showwarning("No Set", "No macro pad set selected.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"macropad_{current_set.name.lower().replace(' ', '_')}.pdf"
        )

        if filepath:
            try:
                from handflow.utils.pdf_generator import MacroPadPDFGenerator
                generator = MacroPadPDFGenerator()
                generator.generate(current_set, filepath)
                messagebox.showinfo("Exported", f"Macro pad PDF saved to:\n{filepath}")
            except ImportError as e:
                messagebox.showerror("Error", f"PDF generation requires reportlab.\n\nInstall with: pip install reportlab")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export PDF:\n{e}")

    def _export_calibration_pdf(self):
        """Export screen calibration markers PDF."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile="screen_calibration_markers.pdf"
        )

        if filepath:
            try:
                from handflow.utils.pdf_generator import ScreenCalibrationPDFGenerator
                generator = ScreenCalibrationPDFGenerator()
                generator.generate(filepath)
                messagebox.showinfo("Exported", f"Calibration markers PDF saved to:\n{filepath}")
            except ImportError:
                messagebox.showerror("Error", "PDF generation requires reportlab.\n\nInstall with: pip install reportlab")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export PDF:\n{e}")

    def _export_combined_pdf(self):
        """Export origami PDF with all configured macropad sets (up to 3)."""
        if not self.setting.macropad_sets:
            messagebox.showwarning("No Sets", "No macro pad sets configured.")
            return

        # Get number of sets to export
        num_sets = min(len(self.setting.macropad_sets), 3)
        set_names = [s.name for s in self.setting.macropad_sets[:3]]

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialdir=os.path.expanduser("~/Desktop"),
            initialfile="macropad_origami.pdf"
        )

        if filepath:
            try:
                from handflow.utils.macropad_pdf_generator import (
                    OrigamiMacroPadPDFGenerator,
                    create_sets_from_settings
                )
                generator = OrigamiMacroPadPDFGenerator()
                sets = create_sets_from_settings(self.setting)
                generator.generate(sets, filepath)

                sets_info = "\n".join([f"  - {name}" for name in set_names])
                messagebox.showinfo(
                    "Exported",
                    f"Origami MacroPad PDF saved to:\n{filepath}\n\n"
                    f"Includes {num_sets} set(s):\n{sets_info}\n\n"
                    f"Print on A4, fold along dotted lines."
                )
            except ImportError:
                messagebox.showerror("Error", "PDF generation requires reportlab.\n\nInstall with: pip install reportlab")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export PDF:\n{e}")

    def _create_new_set(self):
        """Create a new macro pad set."""
        if len(self.setting.macropad_sets) >= 12:
            messagebox.showwarning("Limit Reached", "Maximum 12 macro pad sets allowed.")
            return

        dialog = ctk.CTkInputDialog(text="Enter name for new set:", title="New Macro Pad Set")
        name = dialog.get_input()
        if name:
            self.setting.create_macropad_set(name)
            self._update_set_dropdown()
            self.set_var.set(name)
            self._refresh_buttons()

    def _rename_set(self):
        """Rename current set."""
        current_set = self.setting.get_active_macropad()
        if current_set:
            dialog = ctk.CTkInputDialog(text="Enter new name:", title="Rename Set")
            name = dialog.get_input()
            if name:
                current_set.name = name
                self._update_set_dropdown()
                self.set_var.set(name)

    def _delete_set(self):
        """Delete current set."""
        if len(self.setting.macropad_sets) <= 1:
            messagebox.showwarning("Cannot Delete", "At least one macro pad set is required.")
            return

        if messagebox.askyesno("Confirm Delete", f"Delete set '{self.set_var.get()}'?"):
            idx = self.setting.active_macropad_set
            self.setting.delete_macropad_set(idx)
            self._update_set_dropdown()
            self._refresh_buttons()

    def _update_set_dropdown(self):
        """Update set dropdown options including screen overlay."""
        set_display_names = self._get_all_set_display_names()
        self.set_dropdown.configure(values=set_display_names)

        if self._editing_screen_overlay:
            # Keep screen overlay selected
            overlay_set = self.setting.get_screen_overlay_macropad()
            self.set_var.set(f"⬛ {overlay_set.name} (ID:{overlay_set.set_marker_id})")
        elif self.setting.macropad_sets:
            active_idx = min(self.setting.active_macropad_set, len(self.setting.macropad_sets) - 1)
            active_set = self.setting.macropad_sets[active_idx]
            self.set_var.set(f"{active_set.name} (ID:{active_set.set_marker_id})")

    def _toggle_macropad(self):
        """Toggle macro pad enabled state."""
        self.setting.macropad_enabled = self.macropad_enabled_var.get()
    
    def _toggle_screen_overlay(self):
        """Toggle screen overlay macropad mode."""
        self.setting.screen_overlay_macropad_enabled = self.screen_overlay_var.get()

    def _open_screen_overlay_config(self):
        """Open dialog to configure screen overlay macropad buttons."""
        overlay_set = self.setting.get_screen_overlay_macropad()

        popup = ctk.CTkToplevel(self)
        popup.title("Screen Overlay MacroPad Configuration")
        popup.geometry("750x600")
        popup.transient(self.winfo_toplevel())
        popup.grab_set()

        # Header
        header_frame = ctk.CTkFrame(popup, fg_color="#1a5fb4", corner_radius=5)
        header_frame.pack(fill="x", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            header_frame,
            text="  Screen Overlay MacroPad (Set ID: 20)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        ).pack(side="left", pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Markers: 20-27  ",
            font=ctk.CTkFont(size=11),
            text_color="#aaccff"
        ).pack(side="right", pady=10)

        ctk.CTkLabel(
            popup,
            text="Configure the 12 buttons for the screen overlay macropad. This is separate from paper macropad sets.",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(5, 10))

        # Scrollable button config area
        buttons_scroll = ctk.CTkScrollableFrame(popup, label_text="")
        buttons_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        # Store widgets for saving
        overlay_button_widgets = {}

        # Create button configs in 4 columns (3 rows x 4 cols = 12 buttons)
        for row in range(3):
            row_frame = ctk.CTkFrame(buttons_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=5)

            for col in range(4):
                button_idx = row * 4 + col
                btn_frame = ctk.CTkFrame(row_frame, width=320)
                btn_frame.pack(side="left", padx=10, pady=3)
                btn_frame.pack_propagate(False)

                # Get current button config
                current_btn = overlay_set.buttons.get(button_idx)
                current_label = current_btn.label if current_btn else ""
                actions = current_btn.get_actions() if current_btn else []

                # Header with button number
                btn_header = ctk.CTkFrame(btn_frame, height=28)
                btn_header.pack(fill="x", padx=5, pady=(5, 2))

                ctk.CTkLabel(
                    btn_header,
                    text=f"Button {button_idx + 1}",
                    font=ctk.CTkFont(size=12, weight="bold")
                ).pack(side="left")

                ctk.CTkLabel(
                    btn_header,
                    text=f"idx {button_idx}",
                    font=ctk.CTkFont(size=10),
                    text_color="#4a9eff"
                ).pack(side="right")

                # Label entry
                label_entry = ctk.CTkEntry(
                    btn_frame,
                    placeholder_text="Button Label",
                    width=200,
                    height=28,
                    font=ctk.CTkFont(size=11)
                )
                label_entry.pack(padx=5, pady=3)
                if current_label:
                    label_entry.insert(0, current_label)

                # Actions info and edit button
                action_row = ctk.CTkFrame(btn_frame, fg_color="transparent")
                action_row.pack(fill="x", padx=5, pady=3)

                actions_label = ctk.CTkLabel(
                    action_row,
                    text=f"{len(actions)} action(s)",
                    font=ctk.CTkFont(size=10),
                    text_color="gray"
                )
                actions_label.pack(side="left")

                # Store actions list
                overlay_button_widgets[button_idx] = {
                    'label': label_entry,
                    'actions_label': actions_label,
                    'actions': list(actions)
                }

                # Edit actions button
                edit_btn = ctk.CTkButton(
                    action_row,
                    text="Edit Actions",
                    width=100,
                    height=26,
                    font=ctk.CTkFont(size=10),
                    command=lambda idx=button_idx: self._edit_overlay_button_actions(
                        idx, overlay_button_widgets, popup
                    )
                )
                edit_btn.pack(side="right")

        # Save/Cancel buttons
        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=15)

        def save_overlay_config():
            """Save screen overlay macropad configuration."""
            for button_idx, widgets in overlay_button_widgets.items():
                label = widgets['label'].get()
                actions = widgets['actions']

                overlay_set.buttons[button_idx] = MacroPadButton(
                    label=label,
                    actions=actions
                )

            save_setting(self.setting, DEFAULT_SETTING_PATH)
            messagebox.showinfo("Saved", "Screen overlay macropad configuration saved!")
            popup.destroy()

        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=popup.destroy).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Save", width=120, command=save_overlay_config).pack(side="left", padx=10)

    def _edit_overlay_button_actions(self, button_idx: int, widgets_dict: dict, parent_popup):
        """Edit actions for a screen overlay button."""
        popup = ctk.CTkToplevel(parent_popup)
        popup.title(f"Screen Overlay Button {button_idx + 1} Actions")
        popup.geometry("650x480")
        popup.transient(parent_popup)
        popup.grab_set()

        # Header
        ctk.CTkLabel(
            popup,
            text=f"Actions for Button {button_idx + 1}",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            popup,
            text="Add multiple actions that execute in sequence.",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        ).pack(pady=(0, 10))

        # Scrollable actions area
        actions_scroll = ctk.CTkScrollableFrame(popup, label_text="", height=250)
        actions_scroll.pack(fill="both", expand=True, padx=15, pady=5)

        dialog_action_rows = []
        type_to_name = {a[1]: a[0] for a in ActionExecutor.get_available_actions()}

        def add_action_row(action=None):
            if len(dialog_action_rows) >= 10:
                messagebox.showwarning("Limit", "Maximum 10 actions per button.")
                return

            row_index = len(dialog_action_rows)
            row_frame = ctk.CTkFrame(actions_scroll, fg_color="transparent")
            row_frame.pack(fill="x", pady=3)

            ctk.CTkLabel(row_frame, text=f"{row_index + 1}.", width=25, font=ctk.CTkFont(size=11)).pack(side="left")

            current_type = action.type if action else "none"
            current_name = type_to_name.get(current_type, "None")

            action_var = ctk.StringVar(value=current_name)
            dropdown = ctk.CTkComboBox(
                row_frame, values=self.action_names, variable=action_var,
                width=130, height=28, font=ctk.CTkFont(size=10)
            )
            dropdown.pack(side="left", padx=3)

            current_value = action.value if action else ""
            value_var = ctk.StringVar(value=current_value)
            value_entry = ctk.CTkEntry(
                row_frame, textvariable=value_var, placeholder_text="Value...",
                width=130, height=28, font=ctk.CTkFont(size=10)
            )
            value_entry.pack(side="left", padx=3)

            current_delay = action.delay if action else 0.17
            delay_var = ctk.StringVar(value=str(current_delay))
            ctk.CTkLabel(row_frame, text="Delay:", font=ctk.CTkFont(size=9)).pack(side="left", padx=(5, 2))
            delay_entry = ctk.CTkEntry(row_frame, textvariable=delay_var, width=45, height=28, font=ctk.CTkFont(size=10))
            delay_entry.pack(side="left")
            ctk.CTkLabel(row_frame, text="s", font=ctk.CTkFont(size=9)).pack(side="left")

            cfg_btn = ctk.CTkButton(
                row_frame, text="...", width=28, height=28,
                command=lambda: self._configure_dialog_action(action_var, value_var)
            )
            cfg_btn.pack(side="left", padx=3)

            def remove_row(rf=row_frame):
                for i, rd in enumerate(dialog_action_rows):
                    if rd['frame'] == rf:
                        rf.destroy()
                        dialog_action_rows.pop(i)
                        break
                for i, rd in enumerate(dialog_action_rows):
                    for child in rd['frame'].winfo_children():
                        if isinstance(child, ctk.CTkLabel) and child.cget("text").endswith("."):
                            child.configure(text=f"{i + 1}.")
                            break

            delete_btn = ctk.CTkButton(
                row_frame, text="✕", width=28, height=28,
                fg_color="#a51d2d", hover_color="#8a1829", command=remove_row
            )
            delete_btn.pack(side="left", padx=2)

            dialog_action_rows.append({
                'frame': row_frame, 'action_var': action_var,
                'value_var': value_var, 'delay_var': delay_var
            })

        # Load existing actions
        existing_actions = widgets_dict[button_idx]['actions']
        if existing_actions:
            for action in existing_actions:
                add_action_row(action)
        else:
            add_action_row()

        ctk.CTkButton(popup, text="+ Add Action", width=120, height=30, command=lambda: add_action_row()).pack(pady=10)

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=15)

        def save_actions():
            actions = []
            for row_data in dialog_action_rows:
                action_name = row_data['action_var'].get()
                action_type = self.action_types.get(action_name, 'none')
                value = row_data['value_var'].get()
                try:
                    delay = float(row_data['delay_var'].get())
                except ValueError:
                    delay = 0.17
                actions.append(ActionBinding(type=action_type, value=value, delay=delay))

            widgets_dict[button_idx]['actions'] = actions
            widgets_dict[button_idx]['actions_label'].configure(text=f"{len(actions)} action(s)")
            popup.destroy()

        ctk.CTkButton(btn_frame, text="Cancel", width=80, command=popup.destroy).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Save", width=100, command=save_actions).pack(side="left", padx=5)


class CalibrationTab(ctk.CTkFrame):
    """Tab for ArUco screen calibration."""

    def __init__(self, parent, setting: Setting):
        super().__init__(parent, fg_color="transparent")
        self.setting = setting
        self._build_ui()

    def _build_ui(self):
        # Header
        ctk.CTkLabel(
            self,
            text="ArUco Screen Calibration",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            self,
            text="Position 4 ArUco markers (ID 0-3) at your screen corners\nand calibrate the touch detection area.",
            text_color="gray",
            justify="center",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(0, 20))

        # Instructions card
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=40, pady=10)

        ctk.CTkLabel(
            info_frame,
            text="Setup Instructions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        instructions = """
1. Print the Screen Calibration PDF (export from Macro Pad tab)
2. Cut out the 4 markers (ID 0-3)
3. Attach markers to your physical screen corners:
     ID 0 - Top-Left       ID 1 - Top-Right
     ID 3 - Bottom-Left    ID 2 - Bottom-Right
4. Click 'Start Calibration' to open camera preview
5. Adjust sliders to align the green box with your screen edges
6. Save calibration when satisfied
        """

        ctk.CTkLabel(
            info_frame,
            text=instructions,
            justify="left",
            font=ctk.CTkFont(size=12, family="monospace")
        ).pack(padx=20, pady=(0, 15))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=40, pady=20)

        ctk.CTkButton(
            btn_frame,
            text="Start Calibration",
            command=self._start_calibration,
            width=180,
            height=40,
            fg_color="#2b7a0b",
            hover_color="#1e5a08"
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame,
            text="Reset Offsets",
            command=self._reset_calibration,
            width=150,
            height=40
        ).pack(side="left", padx=10)

        # Status
        self.status_label = ctk.CTkLabel(
            self,
            text="Calibration status: Not calibrated",
            text_color="gray"
        )
        self.status_label.pack(pady=20)

    def _start_calibration(self):
        """Open ArUco calibration window with auto camera preview."""
        try:
            import pyautogui
            from handflow.detector import ArUcoScreenDetector, ArUcoCalibrationUI

            sw, sh = pyautogui.size()
            detector = ArUcoScreenDetector(screen_width=sw, screen_height=sh)

            # Directly open calibration UI - camera preview opens automatically
            # No dialog needed - instructions are shown in the preview window
            ui = ArUcoCalibrationUI(detector, camera_id=self.setting.camera.index, settings=self.setting)
            ui.run()

            self.status_label.configure(text="Calibration status: Calibrated", text_color="green")

        except ImportError as e:
            messagebox.showerror("Error", f"Missing dependency: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Calibration error: {e}")

    def _reset_calibration(self):
        """Reset calibration to defaults."""
        if messagebox.askyesno("Reset", "Reset all calibration offsets to zero?"):
            try:
                import json
                config_file = "config/aruco_calibration.json"
                reset_data = {
                    "top_left": {"horizontal": 0, "vertical": 0},
                    "top_right": {"horizontal": 0, "vertical": 0},
                    "bottom_right": {"horizontal": 0, "vertical": 0},
                    "bottom_left": {"horizontal": 0, "vertical": 0}
                }
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(reset_data, f, indent=2)
                self.status_label.configure(text="Calibration status: Reset to defaults", text_color="orange")
                messagebox.showinfo("Reset", "Calibration reset to defaults.")
            except Exception as e:
                messagebox.showerror("Error", f"Reset failed: {e}")


class settingTab(ctk.CTkFrame):
    """Tab for general application setting."""

    def __init__(self, parent, setting: Setting):
        super().__init__(parent, fg_color="transparent")
        self.setting = setting
        self._build_ui()

    def _build_ui(self):
        # Header
        ctk.CTkLabel(
            self,
            text="setting",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(20, 15))

        # Scrollable setting
        scroll = ctk.CTkScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=20, pady=5)

        # Camera setting
        self._create_section(scroll, "Camera", [
            ("Camera Source", "camera_source", getattr(self.setting.camera, 'source', 'webcam'), "dropdown", ["webcam", "esp32"]),
            ("Camera Index", "camera_index", self.setting.camera.index, "entry"),
            ("ESP32 Serial Port", "esp32_port", getattr(self.setting.camera, 'esp32_serial_port', '/dev/cu.usbmodem101'), "entry"),
            ("ESP32 Baud Rate", "esp32_baud", getattr(self.setting.camera, 'esp32_baud_rate', 2000000), "entry"),
            ("Flip Horizontal", "flip_h", self.setting.camera.flip_horizontal, "switch"),
            ("Flip Vertical", "flip_v", self.setting.camera.flip_vertical, "switch"),
            ("Swap Hands", "swap_hands", self.setting.camera.swap_hands, "switch"),
        ])

        # Mouse setting
        self._create_section(scroll, "Mouse Control", [
            ("Smoothing", "smoothing", self.setting.mouse.smoothing, "slider", 0.1, 1.0),
            ("Sensitivity", "sensitivity", self.setting.mouse.base_sensitivity, "slider", 0.5, 2.0),
        ])

        # Inference setting
        self._create_section(scroll, "Detection", [
            ("Confidence Threshold", "conf_threshold", self.setting.inference.confidence_threshold, "slider", 0.3, 0.9),
            ("Cooldown Frames", "cooldown", self.setting.inference.cooldown_frames, "entry"),
            ("Stability Window", "stability", self.setting.inference.stability_window, "entry"),
        ])

        # Save button
        ctk.CTkButton(
            self,
            text="Save setting",
            command=self._save_setting,
            width=150,
            height=35
        ).pack(pady=20)

    def _create_section(self, parent, title: str, items: list):
        """Create a setting section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", pady=10)

        # Section header
        header = ctk.CTkFrame(section, fg_color="#2b5797", corner_radius=5)
        header.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            header,
            text=f"  {title}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="white"
        ).pack(side="left", pady=6)

        for item in items:
            self._create_setting_row(section, *item)

    def _create_setting_row(self, parent, label: str, key: str, value, widget_type: str, *args):
        """Create a single setting row."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=4)

        ctk.CTkLabel(row, text=label, width=160, anchor="w").pack(side="left")

        if widget_type == "dropdown":
            options = args[0] if args else []
            var = ctk.StringVar(value=str(value))
            dropdown = ctk.CTkComboBox(row, values=options, variable=var, width=150, state="readonly")
            dropdown.pack(side="left")
            setattr(self, f"_{key}_var", var)

        elif widget_type == "entry":
            entry = ctk.CTkEntry(row, width=100)
            entry.insert(0, str(value))
            entry.pack(side="left")
            setattr(self, f"_{key}_widget", entry)

        elif widget_type == "switch":
            var = ctk.BooleanVar(value=value)
            switch = ctk.CTkSwitch(row, text="", variable=var)
            switch.pack(side="left")
            setattr(self, f"_{key}_var", var)

        elif widget_type == "slider":
            min_val, max_val = args[0], args[1]
            var = ctk.DoubleVar(value=value)

            slider = ctk.CTkSlider(row, from_=min_val, to=max_val, variable=var, width=150)
            slider.pack(side="left")

            label = ctk.CTkLabel(row, textvariable=var, width=50)
            label.pack(side="left", padx=5)

            # Update label formatting
            def update_label(val, lbl=label):
                lbl.configure(text=f"{float(val):.2f}")
            slider.configure(command=update_label)

            setattr(self, f"_{key}_var", var)

    def _save_setting(self):
        """Save all setting."""
        # Camera
        if hasattr(self, '_camera_source_var'):
            self.setting.camera.source = self._camera_source_var.get()
        if hasattr(self, '_camera_index_widget'):
            try:
                self.setting.camera.index = int(self._camera_index_widget.get())
            except ValueError:
                pass
        if hasattr(self, '_esp32_port_widget'):
            self.setting.camera.esp32_serial_port = self._esp32_port_widget.get()
        if hasattr(self, '_esp32_baud_widget'):
            try:
                self.setting.camera.esp32_baud_rate = int(self._esp32_baud_widget.get())
            except ValueError:
                pass
        if hasattr(self, '_flip_h_var'):
            self.setting.camera.flip_horizontal = self._flip_h_var.get()
        if hasattr(self, '_flip_v_var'):
            self.setting.camera.flip_vertical = self._flip_v_var.get()
        if hasattr(self, '_swap_hands_var'):
            self.setting.camera.swap_hands = self._swap_hands_var.get()

        # Mouse
        if hasattr(self, '_smoothing_var'):
            self.setting.mouse.smoothing = self._smoothing_var.get()
        if hasattr(self, '_sensitivity_var'):
            self.setting.mouse.base_sensitivity = self._sensitivity_var.get()

        # Inference
        if hasattr(self, '_conf_threshold_var'):
            self.setting.inference.confidence_threshold = self._conf_threshold_var.get()
        if hasattr(self, '_cooldown_widget'):
            try:
                self.setting.inference.cooldown_frames = int(self._cooldown_widget.get())
            except ValueError:
                pass
        if hasattr(self, '_stability_widget'):
            try:
                self.setting.inference.stability_window = int(self._stability_widget.get())
            except ValueError:
                pass

        save_setting(self.setting, DEFAULT_SETTING_PATH)
        messagebox.showinfo("Saved", "setting saved successfully!")


class HandFlowApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Window config
        self.title("HandFlow - Gesture & Macro Control")
        self.geometry("1100x750")
        self.minsize(900, 650)

        # Initialize core components
        self.setting = load_setting(DEFAULT_SETTING_PATH)
        self.executor = ActionExecutor()

        # Detection state
        self.detection_running = False

        # Build UI
        self._build_ui()

        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """Build the main UI."""
        # Title bar
        title_frame = ctk.CTkFrame(self, height=65)
        title_frame.pack(fill="x", padx=10, pady=(10, 0))
        title_frame.pack_propagate(False)

        # Logo/Title
        ctk.CTkLabel(
            title_frame,
            text="HandFlow",
            font=ctk.CTkFont(size=26, weight="bold")
        ).pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(
            title_frame,
            text="Gesture & Macro Control",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(side="left", pady=10)

        # Run detection button
        self.run_btn = ctk.CTkButton(
            title_frame,
            text="Run Detection",
            command=self._toggle_detection,
            width=140,
            height=38,
            fg_color="#2b7a0b",
            hover_color="#1e5a08"
        )
        self.run_btn.pack(side="right", padx=20, pady=12)

        # Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        gestures_tab = self.tabview.add("Gestures")
        macropad_tab = self.tabview.add("Macro Pad")
        calibration_tab = self.tabview.add("Calibration")
        setting_tab = self.tabview.add("setting")

        # Initialize tab contents
        self.gesture_mapping = GestureMappingTab(gestures_tab, self.setting, self.executor)
        self.gesture_mapping.pack(fill="both", expand=True)

        self.macropad_config = MacroPadTab(macropad_tab, self.setting, self.executor)
        self.macropad_config.pack(fill="both", expand=True)

        self.calibration = CalibrationTab(calibration_tab, self.setting)
        self.calibration.pack(fill="both", expand=True)

        self.setting_panel = settingTab(setting_tab, self.setting)
        self.setting_panel.pack(fill="both", expand=True)

    def _toggle_detection(self):
        """Toggle detection on/off."""
        if self.detection_running:
            self._stop_detection()
        else:
            self._start_detection()

    def _start_detection(self):
        """Start gesture/macro pad detection."""
        try:
            from handflow.app.detection_window import DetectionWindow

            self.detection_window = DetectionWindow(self.setting, self.executor)
            self.detection_window.start()

            self.detection_running = True
            self.run_btn.configure(text="Stop Detection", fg_color="#a51d2d", hover_color="#8a1829")

            # Monitor detection window
            self._check_detection_window()

        except ImportError as e:
            messagebox.showerror("Error", f"Detection module error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")

    def _stop_detection(self):
        """Stop detection."""
        if hasattr(self, 'detection_window'):
            try:
                self.detection_window.stop()
            except:
                pass

        self.detection_running = False
        self.run_btn.configure(text="Run Detection", fg_color="#2b7a0b", hover_color="#1e5a08")

    def _check_detection_window(self):
        """Check if detection window is still running."""
        if self.detection_running:
            if hasattr(self, 'detection_window'):
                try:
                    if not self.detection_window._running:
                        self._stop_detection()
                        return
                except:
                    pass
            self.after(500, self._check_detection_window)

    def _on_close(self):
        """Handle window close."""
        self._stop_detection()
        
        save_setting(self.setting, DEFAULT_SETTING_PATH)
        self.destroy()


def main():
    """Entry point for the HandFlow application."""
    app = HandFlowApp()
    app.mainloop()


if __name__ == "__main__":
    main()
