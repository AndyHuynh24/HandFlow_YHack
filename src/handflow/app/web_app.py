"""HandFlow Web UI — NiceGUI-powered configuration interface."""

import sys
import os
import subprocess
from pathlib import Path
from typing import Optional, List

from nicegui import ui, app

# ── Path Setup ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
_src = str(ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
os.chdir(str(ROOT))

from handflow.utils import Setting, load_setting, save_setting
from handflow.utils.setting import ActionBinding, MacroPadButton
from handflow.actions import ActionExecutor

SETTING_PATH = "config/handflow_setting.yaml"

# ── Action Types ────────────────────────────────────────
try:
    _raw = ActionExecutor.get_available_actions()
    ACTION_NAMES: List[str] = [a[0] for a in _raw]
    ACTION_TYPE_MAP: dict = {a[0]: a[1] for a in _raw}
    TYPE_NAME_MAP: dict = {a[1]: a[0] for a in _raw}
except Exception:
    ACTION_NAMES = ["None"]
    ACTION_TYPE_MAP = {"None": "none"}
    TYPE_NAME_MAP = {"none": "None"}

# ── Global State ────────────────────────────────────────
setting: Optional[Setting] = None
detection_proc: Optional[subprocess.Popen] = None
detection_running: bool = False

# ── Theme ───────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #F5F3F0;
    --surface: #FFFFFF;
    --border: #E0DDD9;
    --border-strong: #C8C4BF;
    --text-primary: #1A1815;
    --text-secondary: #4A4540;
    --text-tertiary: #8A857E;
    --accent: #2563EB;
    --accent-soft: #EBF2FF;
    --orange: #EA580C;
    --orange-soft: #FFF7ED;
    --red: #DC2626;
    --green: #16A34A;
    --left-hand: #E11D48;
    --right-hand: #2563EB;
}

body {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Header ──────────────────────────── */
.app-header {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    height: 60px !important;
    min-height: 60px !important;
    padding: 0 28px !important;
}

/* ── Tabs ────────────────────────────── */
.app-tabs {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
}
.app-tabs .q-tab {
    text-transform: none !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    font-size: 16px !important;
}
.app-tabs .q-tab__label { font-size: 16px !important; }

/* ── Cards ───────────────────────────── */
.s-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ── Hand sections ───────────────────── */
.hand-left  { border-left: 4px solid var(--left-hand) !important; border-radius: 0 12px 12px 0 !important; }
.hand-right { border-left: 4px solid var(--right-hand) !important; border-radius: 0 12px 12px 0 !important; }

/* ── Action rows ─────────────────────── */
.action-row {
    border-left: 2px solid var(--orange);
    padding-left: 14px;
    margin-left: 6px;
}

/* ── Macropad grid ───────────────────── */
.pad-wrapper {
    max-width: 520px;
    border: 2px solid #3A3632;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    background: #FAFAF8;
}
.pad-grid-inner {
    display: grid;
    width: 100%;
    grid-template-columns: repeat(4, 1fr);
}
/* Paper cells: 8:9 (w:h), overlay cells: 9:8 (w:h) */
.pad-cell {
    aspect-ratio: 8 / 9;
    border: 1px solid var(--border-strong);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    cursor: pointer;
    transition: background 0.1s;
}
.pad-overlay .pad-cell { aspect-ratio: 9 / 8; }
.pad-cell:hover { background: var(--orange-soft); }

/* Circle: sized relative to cell, always centered */
.pad-circle {
    width: 70%;
    aspect-ratio: 1 / 1;
    border: 2px solid #9A9590;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.pad-cell:hover .pad-circle { border-color: var(--orange); }

/* Label: fills entire cell, text wraps to new lines */
.pad-label {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    color: #9333EA;
    text-align: center;
    line-height: 1.3;
    padding: 4px 3px;
    overflow-wrap: anywhere;
    word-break: break-word;
    z-index: 1;
}
.pad-cell:hover .pad-label { color: var(--orange); }

.pad-cell-num {
    position: absolute; top: 4px; left: 6px;
    font-size: 10px; font-weight: 600;
    color: var(--text-tertiary);
    font-family: 'JetBrains Mono', monospace;
    z-index: 2;
}
.pad-corner {
    position: absolute;
    width: 30px; height: 30px;
    background: #2A2724;
    display: flex; align-items: center; justify-content: center;
    font-size: 7px; font-weight: 700;
    color: #FFF;
    font-family: 'JetBrains Mono', monospace;
    z-index: 3;
}
.pad-corner-tl { top: 0; left: 0; border-radius: 6px 0 4px 0; }
.pad-corner-tr { top: 0; right: 0; border-radius: 0 6px 0 4px; }
.pad-corner-bl { bottom: 0; left: 0; border-radius: 0 4px 0 6px; }
.pad-corner-br { bottom: 0; right: 0; border-radius: 4px 0 6px 0; }

/* ── Expansion panels ────────────────── */
.q-expansion-item .q-item__label { font-size: 16px !important; font-weight: 600 !important; }

/* ── Typography ──────────────────────── */
.t-page    { font-size: 26px !important; font-weight: 700 !important; color: var(--text-primary) !important; letter-spacing: -0.3px !important; }
.t-section { font-size: 18px !important; font-weight: 700 !important; color: var(--text-primary) !important; }
.t-desc    { font-size: 15px !important; font-weight: 500 !important; color: var(--text-secondary) !important; line-height: 1.5 !important; }
.t-label   { font-size: 15px !important; font-weight: 500 !important; color: var(--text-secondary) !important; }
.t-mono    { font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; color: var(--text-tertiary) !important; }
.t-small   { font-size: 14px !important; color: var(--text-tertiary) !important; }
.t-hint    { font-size: 13px !important; font-weight: 500 !important; color: var(--text-tertiary) !important; font-style: italic !important; }

/* ── Knuckle hand diagram ────────────── */
.hand-diagram {
    position: relative;
    width: 340px;
    height: 480px;
    margin: 0 auto;
}
.hand-bg {
    position: absolute;
    inset: 0;
    background: var(--bg);
    border: 2px solid var(--border);
    border-radius: 16px;
}
.knuckle-btn {
    position: absolute;
    border: 2px solid #9A9590;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--surface);
    transition: all 0.12s;
    text-align: center;
    line-height: 1.2;
    padding: 2px;
}
.knuckle-btn:hover {
    border-color: var(--orange);
    color: var(--orange);
    background: var(--accent-soft);
    transform: scale(1.05);
}
.finger-label {
    position: absolute;
    font-size: 10px;
    font-weight: 700;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Quasar overrides ────────────────── */
.q-field__label { font-size: 14px !important; }
.q-field input, .q-field .q-field__native { font-size: 15px !important; }
.q-btn .q-btn__content { font-size: 14px !important; }
"""


# ════════════════════════════════════════════════════════
#  Shared: Action Row Builder
# ════════════════════════════════════════════════════════

def _build_action_rows(container, actions: list, sync_fn=None):
    container.clear()
    with container:
        if not actions:
            ui.label("No actions assigned").classes("t-label italic ml-6 my-2")
            return
        for i, action in enumerate(actions):
            _one_action_row(container, actions, i, action, sync_fn)


def _one_action_row(container, actions, idx, action, sync_fn):
    with ui.row().classes("items-center gap-3 w-full action-row py-2"):
        ui.label(f"{idx + 1}").classes("t-mono w-5 shrink-0 text-center")

        def _on_type(e, _i=idx):
            actions[_i] = ActionBinding(type=ACTION_TYPE_MAP.get(e.value, "none"),
                                        value=actions[_i].value, delay=actions[_i].delay)
            if sync_fn: sync_fn()
            # Rebuild rows to show correct widget for new type
            _build_action_rows(container, actions, sync_fn)

        ui.select(options=ACTION_NAMES, value=TYPE_NAME_MAP.get(action.type, "None"),
                  on_change=_on_type).props("dense outlined").classes("w-40")

        def _on_val(e, _i=idx):
            actions[_i] = ActionBinding(type=actions[_i].type, value=e.value or "",
                                        delay=actions[_i].delay)
            if sync_fn: sync_fn()

        # Text action type: button that opens multi-line editor
        # Other types: single-line input
        action_type = ACTION_TYPE_MAP.get(
            TYPE_NAME_MAP.get(action.type, "None"), "none"
        )
        if action.type == "text":
            # Show preview + edit button
            preview = action.value[:20] + "..." if len(action.value) > 20 else (action.value or "(empty)")
            val_label = ui.label(preview).classes("w-36 text-sm truncate").style(
                "color: var(--text-secondary); cursor: pointer; border: 1px solid var(--border); "
                "border-radius: 4px; padding: 4px 8px;"
            )

            def _open_text_editor(_i=idx, _lbl=val_label):
                with ui.dialog() as dlg, ui.card().classes("w-[550px] s-card"):
                    ui.label("Edit Text").classes("t-section")
                    ui.label("Use Tab for indentation. Supports multi-line text.").classes("t-small mb-2")
                    textarea = ui.textarea(
                        label="Text content",
                        value=actions[_i].value,
                    ).props("outlined autogrow").classes("w-full").style(
                        "font-family: 'JetBrains Mono', monospace; font-size: 13px; min-height: 200px;"
                    )
                    # Allow Tab key to insert tab character
                    textarea.on("keydown.tab.prevent",
                                js_handler="(e) => { "
                                "const ta = e.target; "
                                "const start = ta.selectionStart; "
                                "const end = ta.selectionEnd; "
                                "ta.value = ta.value.substring(0, start) + '\\t' + ta.value.substring(end); "
                                "ta.selectionStart = ta.selectionEnd = start + 1; "
                                "ta.dispatchEvent(new Event('input')); }")

                    with ui.row().classes("justify-end gap-2 mt-3"):
                        ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                        def _save_text(_i=_i, _ta=textarea, _dlg=dlg, _lbl=_lbl):
                            val = _ta.value or ""
                            actions[_i] = ActionBinding(
                                type=actions[_i].type, value=val, delay=actions[_i].delay)
                            if sync_fn: sync_fn()
                            p = val[:20] + "..." if len(val) > 20 else (val or "(empty)")
                            _lbl.set_text(p)
                            _dlg.close()
                        ui.button("Save", on_click=_save_text).props("unelevated no-caps color=primary")
                dlg.open()

            val_label.on("click", _open_text_editor)
            ui.button(icon="edit", on_click=_open_text_editor).props("flat dense round").classes("text-stone-500")

        elif action.type == "shortcut":
            # Shortcut: input with hint
            ui.input(label="e.g. cmd+shift+p", value=action.value,
                     on_change=_on_val).props("dense outlined").classes("w-36")

        elif action.type == "file":
            # File: input + browse button
            file_input = ui.input(label="Path", value=action.value,
                                  on_change=_on_val).props("dense outlined").classes("w-32")

            async def _browse_file(_i=idx, _inp=file_input):
                result = await app.native.main_window.create_file_dialog(
                    allow_multiple=False
                ) if hasattr(app, 'native') else None

                # Fallback: use a simple text prompt since native file dialog
                # doesn't work in browser mode
                if result and len(result) > 0:
                    path = result[0]
                    actions[_i] = ActionBinding(
                        type=actions[_i].type, value=path, delay=actions[_i].delay)
                    _inp.value = path
                    if sync_fn: sync_fn()

            ui.button(icon="folder_open", on_click=_browse_file).props(
                "flat dense round").classes("text-stone-500").tooltip("Browse files")

        else:
            # All other types: plain input
            ui.input(label="Value", value=action.value, on_change=_on_val).props("dense outlined").classes("w-36")

        def _on_delay(e, _i=idx):
            actions[_i] = ActionBinding(type=actions[_i].type, value=actions[_i].value,
                                        delay=float(e.value) if e.value is not None else 0.17)
            if sync_fn: sync_fn()

        ui.number(label="Delay", value=action.delay, min=0, max=5, step=0.01,
                  suffix="s", on_change=_on_delay).props("dense outlined").classes("w-24")

        def _del(_i=idx):
            del actions[_i]
            _build_action_rows(container, actions, sync_fn)

        ui.button(icon="close", on_click=_del).props("flat dense round").classes("text-stone-400 hover:text-red-600")


# ════════════════════════════════════════════════════════
#  Page 1 — Gestures
# ════════════════════════════════════════════════════════

def build_gestures_page():
    s = setting

    def save():
        save_setting(s, SETTING_PATH)
        ui.notify("Mappings saved", type="positive")

    with ui.row().classes("items-center justify-between w-full mb-2"):
        with ui.column().classes("gap-1"):
            ui.label("Gesture Mapping").classes("t-page")
            ui.label("Assign actions to each hand gesture. They run in sequence with configurable delays.").classes("t-desc")
        ui.button("Save", icon="save", on_click=save).props("unelevated no-caps color=primary size=md")

    ui.separator().classes("my-4")

    right = sorted(g for g in s.gestures if g.startswith("Right"))
    left = sorted(g for g in s.gestures if g.startswith("Left"))

    if right:
        _gesture_section("Right Hand", right, "hand-right")
    if left:
        _gesture_section("Left Hand", left, "hand-left")


def _gesture_section(title, gestures, css):
    s = setting
    with ui.card().classes(f"w-full {css} pl-5 s-card mb-5"):
        ui.label(title).classes("t-section mb-2")
        for g in gestures:
            _gesture_card(s, g)


def _gesture_card(s, gesture_name):
    display = gesture_name.split("_", 1)[1].replace("_", " ").title()
    actions = list(s.get_gesture_actions(gesture_name))

    def sync():
        s.set_gesture_actions(gesture_name, actions)
    s.set_gesture_actions(gesture_name, actions)

    n = len(actions)
    header = f"{display}  ({n})" if n else display

    with ui.expansion(header, icon="touch_app").classes("w-full"):
        container = ui.column().classes("w-full gap-1")

        def refresh():
            _build_action_rows(container, actions, sync)

        def add():
            if len(actions) >= 10:
                ui.notify("Maximum 10 actions", type="warning")
                return
            actions.append(ActionBinding(type="none", value="", delay=0.17))
            sync()
            refresh()

        with ui.row().classes("justify-end mb-2"):
            ui.button("Add Action", icon="add", on_click=add).props("flat dense no-caps color=primary")

        refresh()


# ════════════════════════════════════════════════════════
#  Page 2 — Macro Pad
# ════════════════════════════════════════════════════════

def build_macropad_page():
    s = setting

    with ui.row().classes("items-center justify-between w-full mb-2"):
        with ui.column().classes("gap-1"):
            ui.label("Macro Pad").classes("t-page")
            ui.label("Configure paper and screen overlay macro pads separately.").classes("t-desc")
        ui.button("Save All", icon="save", on_click=_save_macropad).props("unelevated no-caps color=primary size=md")

    ui.separator().classes("my-4")

    _build_paper_macropad_section(s)
    ui.space().classes("h-8")
    _build_screen_overlay_section(s)


# ── Paper Macro Pad ─────────────────────────────────────

def _build_paper_macropad_section(s):
    with ui.card().classes("w-full s-card"):
        # Header
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-center gap-3"):
                ui.element("div").style(
                    "width: 10px; height: 10px; border-radius: 50%; background: var(--orange);"
                )
                ui.label("Paper Macro Pad").classes("t-section")
                ui.label("4 x 2  ·  8 buttons per set").classes("t-small")
            ui.switch("Enable", value=s.macropad_enabled,
                      on_change=lambda e: setattr(s, "macropad_enabled", e.value))

        ui.separator().classes("my-3")

        # Set management
        set_names = _macropad_set_names()
        active_idx = min(s.active_macropad_set, max(len(set_names) - 1, 0))

        with ui.row().classes("items-center gap-3 flex-wrap"):
            set_select = ui.select(
                options=set_names or ["No sets"],
                value=set_names[active_idx] if set_names else None,
                label="Active Set",
            ).props("outlined dense").classes("w-60")

            ui.button("New", icon="add",
                      on_click=lambda: _new_set(set_select, paper_grid)).props("flat no-caps")
            ui.button("Rename", icon="edit",
                      on_click=lambda: _rename_set(set_select)).props("flat no-caps")
            ui.button("Delete", icon="delete",
                      on_click=lambda: _delete_set(set_select, paper_grid)).props("flat no-caps text-color=red-8")

        # Instruction
        ui.label("Select a set from the dropdown, then click any button below to assign actions.").classes("t-hint mt-3")

        # Grid
        paper_grid = ui.element("div").classes("w-full mt-4")

        def on_set_changed(e):
            for i, st in enumerate(s.macropad_sets):
                if f"{st.name} (ID:{st.set_marker_id})" == e.value:
                    s.active_macropad_set = i
                    break
            _refresh_paper_grid(paper_grid)

        set_select.on_value_change(on_set_changed)
        _refresh_paper_grid(paper_grid)


def _refresh_paper_grid(grid_box):
    grid_box.clear()
    active = setting.get_active_macropad()
    if not active:
        with grid_box:
            ui.label("No sets. Create one to get started.").classes("t-label italic")
        return

    with grid_box:
        with ui.element("div").classes("pad-wrapper"):
            # Corner markers
            ui.html(f'<div class="pad-corner pad-corner-tl">ID {active.set_marker_id}</div>')
            ui.html('<div class="pad-corner pad-corner-tr">ID 4</div>')
            ui.html('<div class="pad-corner pad-corner-bl">ID 7</div>')
            ui.html('<div class="pad-corner pad-corner-br">ID 8</div>')

            with ui.element("div").classes("pad-grid-inner pad-grid-4x2"):
                for idx in range(8):
                    btn = active.buttons.get(idx, MacroPadButton())
                    _pad_cell(active, idx, btn, grid_box, "paper")


# ── Screen Overlay Macro Pad ────────────────────────────

def _build_screen_overlay_section(s):
    with ui.card().classes("w-full s-card"):
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-center gap-3"):
                ui.element("div").style(
                    "width: 10px; height: 10px; border-radius: 50%; background: var(--accent);"
                )
                ui.label("Screen Overlay Macro Pad").classes("t-section")
                ui.label("4 x 3  ·  12 buttons").classes("t-small")
            ui.switch("Enable", value=s.screen_overlay_macropad_enabled,
                      on_change=lambda e: setattr(s, "screen_overlay_macropad_enabled", e.value))

        ui.separator().classes("my-3")
        ui.label(
            "Virtual buttons displayed on screen, detected via ArUco markers 20–27."
        ).classes("t-desc")
        ui.label("Click any button below to customize its label and actions.").classes("t-hint mb-4")

        overlay_grid = ui.element("div").classes("w-full")
        _refresh_overlay_grid(overlay_grid)


def _refresh_overlay_grid(grid_box):
    grid_box.clear()
    overlay_set = setting.get_screen_overlay_macropad()

    with grid_box:
        with ui.element("div").classes("pad-wrapper pad-overlay"):
            ui.html('<div class="pad-corner pad-corner-tl">ID 20</div>')
            ui.html('<div class="pad-corner pad-corner-tr">ID 21</div>')
            ui.html('<div class="pad-corner pad-corner-bl">ID 24</div>')
            ui.html('<div class="pad-corner pad-corner-br">ID 25</div>')

            with ui.element("div").classes("pad-grid-inner pad-grid-4x3"):
                for idx in range(12):
                    btn = overlay_set.buttons.get(idx, MacroPadButton())
                    _pad_cell(overlay_set, idx, btn, grid_box, "overlay")


def _pad_cell(macropad_set, idx, button, grid_box, mode):
    display_label = button.label or f"Btn {idx + 1}"

    def open_editor():
        _edit_button_dialog(macropad_set, idx, button, grid_box, mode)

    with ui.element("div").classes("pad-cell").on("click", open_editor):
        ui.html(
            f'<span class="pad-cell-num">{idx + 1}</span>'
            f'<div class="pad-circle"></div>'
            f'<div class="pad-label">{display_label}</div>'
        )


def _edit_button_dialog(macropad_set, idx, button, grid_box, mode):
    working = list(button.get_actions())

    with ui.dialog() as dlg, ui.card().classes("w-[660px] s-card"):
        ui.label(f"Button {idx + 1}").classes("t-section")
        label_input = ui.input(label="Button Label", value=button.label or "",
                               placeholder=f"Button {idx + 1}").props("outlined dense").classes("w-full mt-3")
        ui.separator().classes("my-3")

        ui.label("Actions").classes("t-label mb-2")
        actions_container = ui.column().classes("w-full gap-1 max-h-72 overflow-auto")

        def refresh():
            _build_action_rows(actions_container, working, None)
        def add():
            if len(working) >= 10:
                ui.notify("Max 10 actions", type="warning")
                return
            working.append(ActionBinding(type="none", value="", delay=0.17))
            refresh()
        refresh()

        ui.separator().classes("my-3")
        with ui.row().classes("justify-between w-full"):
            ui.button("Add Action", icon="add", on_click=add).props("flat no-caps color=primary")
            with ui.row().classes("gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                def save():
                    button.label = label_input.value or ""
                    button.actions = working
                    macropad_set.buttons[idx] = button
                    dlg.close()
                    if mode == "paper":
                        _refresh_paper_grid(grid_box)
                    else:
                        _refresh_overlay_grid(grid_box)
                    ui.notify("Saved", type="positive")
                ui.button("Save", icon="save", on_click=save).props("unelevated no-caps color=primary")
    dlg.open()


def _save_macropad():
    save_setting(setting, SETTING_PATH)
    ui.notify("Macro pad saved", type="positive")

def _macropad_set_names():
    return [f"{st.name} (ID:{st.set_marker_id})" for st in setting.macropad_sets]

def _new_set(set_select, grid_box):
    with ui.dialog() as dlg, ui.card().classes("w-96 s-card"):
        ui.label("New Macro Pad Set").classes("t-section")
        name_in = ui.input(label="Set Name", value="New Set").props("outlined dense").classes("w-full mt-3")
        with ui.row().classes("justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
            def create():
                try:
                    setting.create_macropad_set(name_in.value)
                    names = _macropad_set_names()
                    set_select.options = names
                    set_select.value = names[-1]
                    setting.active_macropad_set = len(setting.macropad_sets) - 1
                    _refresh_paper_grid(grid_box)
                    dlg.close()
                    ui.notify("Created", type="positive")
                except ValueError as e:
                    ui.notify(str(e), type="negative")
            ui.button("Create", on_click=create).props("unelevated no-caps color=primary")
    dlg.open()

def _rename_set(set_select):
    active = setting.get_active_macropad()
    if not active: return
    with ui.dialog() as dlg, ui.card().classes("w-96 s-card"):
        ui.label("Rename Set").classes("t-section")
        name_in = ui.input(label="Name", value=active.name).props("outlined dense").classes("w-full mt-3")
        with ui.row().classes("justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
            def do_rename():
                active.name = name_in.value
                names = _macropad_set_names()
                set_select.options = names
                set_select.value = names[setting.active_macropad_set]
                dlg.close()
                ui.notify("Renamed", type="positive")
            ui.button("Rename", on_click=do_rename).props("unelevated no-caps color=primary")
    dlg.open()

def _delete_set(set_select, grid_box):
    active = setting.get_active_macropad()
    if not active: return
    if len(setting.macropad_sets) <= 1:
        ui.notify("Cannot delete the last set", type="warning")
        return
    with ui.dialog() as dlg, ui.card().classes("w-96 s-card"):
        ui.label(f'Delete "{active.name}"?').classes("t-section")
        ui.label("This cannot be undone.").classes("t-label mt-1").style("color: var(--red);")
        with ui.row().classes("justify-end gap-2 mt-4"):
            ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
            def do_delete():
                setting.delete_macropad_set(setting.active_macropad_set)
                names = _macropad_set_names()
                set_select.options = names if names else ["No sets"]
                idx = min(setting.active_macropad_set, max(len(names) - 1, 0))
                set_select.value = names[idx] if names else None
                _refresh_paper_grid(grid_box)
                dlg.close()
                ui.notify("Deleted", type="warning")
            ui.button("Delete", on_click=do_delete).props("unelevated no-caps color=negative")
    dlg.open()


# ════════════════════════════════════════════════════════
#  Page 3 — Knuckle Pad
# ════════════════════════════════════════════════════════

def build_knuckle_page():
    s = setting

    with ui.row().classes("items-center justify-between w-full mb-2"):
        with ui.column().classes("gap-1"):
            ui.label("Knuckle Pad").classes("t-page")
            ui.label("Flip your palm up and your finger joints become buttons. Click each button below to customize.").classes("t-desc")
        ui.button("Save", icon="save", on_click=_save_knuckle).props("unelevated no-caps color=primary size=md")

    ui.separator().classes("my-4")

    with ui.row().classes("w-full gap-8 flex-wrap"):
        # Left: hand diagram
        with ui.column().classes("items-center gap-4"):
            ui.label("Palm Up View").classes("t-section")
            _build_hand_diagram()

        # Right: button list
        with ui.column().classes("flex-1 min-w-[300px] gap-3"):
            ui.label("Button Actions").classes("t-section mb-2")
            knuckle_btns = getattr(s, 'knuckle_macropad_buttons', {})

            btn_labels = [
                ("1", "Index MCP", "Base of index finger"),
                ("2", "Index PIP", "Middle of index finger"),
                ("3", "Index Tip", "Tip of index finger"),
                ("4", "Middle MCP", "Base of middle finger"),
                ("5", "Middle PIP", "Middle of middle finger"),
                ("6", "Middle Tip", "Tip of middle finger"),
                ("7", "Palm", "Center of palm"),
            ]

            for i, (num, name, desc) in enumerate(btn_labels):
                btn = knuckle_btns.get(i, MacroPadButton())
                n_actions = len(btn.get_actions())
                label = btn.label or name

                with ui.card().classes("w-full s-card p-3 cursor-pointer").on(
                    "click", lambda _i=i, _btn=btn: _edit_knuckle_button(_i, _btn)
                ):
                    with ui.row().classes("items-center justify-between w-full"):
                        with ui.row().classes("items-center gap-3"):
                            ui.element("div").style(
                                f"width: 28px; height: 28px; border-radius: 50%; "
                                f"background: var(--orange); display: flex; align-items: center; "
                                f"justify-content: center; font-size: 13px; font-weight: 700; color: white;"
                            ).text = num
                            with ui.column().classes("gap-0"):
                                ui.label(label).classes("text-sm font-semibold").style("color: var(--text);")
                                ui.label(desc).classes("text-xs").style("color: var(--text-tertiary);")
                        with ui.row().classes("items-center gap-2"):
                            ui.label(f"{n_actions} action{'s' if n_actions != 1 else ''}").classes("t-small")
                            ui.icon("chevron_right", size="xs").classes("text-stone-400")


def _build_hand_diagram():
    """Draw a visual hand diagram with clickable knuckle zones."""
    # Positions: (left%, top%, width%, height%) relative to container
    # Two fingers (index + middle) with 3 segments each + palm
    buttons = [
        # Index finger (left column)
        (0, "10%", "5%",  "28%", "18%"),   # MCP - base
        (1, "10%", "25%", "28%", "18%"),   # PIP - middle
        (2, "10%", "45%", "28%", "22%"),   # TIP - tip (taller)
        # Middle finger (right column)
        (3, "55%", "5%",  "28%", "18%"),   # MCP - base
        (4, "55%", "25%", "28%", "18%"),   # PIP - middle
        (5, "55%", "45%", "28%", "22%"),   # TIP - tip (taller)
        # Palm (bottom, full width)
        (6, "15%", "72%", "65%", "22%"),   # Palm
    ]

    knuckle_btns = getattr(setting, 'knuckle_macropad_buttons', {})

    with ui.element("div").classes("hand-diagram"):
        ui.element("div").classes("hand-bg")

        # Finger labels
        ui.html('<div class="finger-label" style="left: 15%; top: -20px;">Index</div>')
        ui.html('<div class="finger-label" style="left: 60%; top: -20px;">Middle</div>')

        for idx, left, top, width, height in buttons:
            btn = knuckle_btns.get(idx, MacroPadButton())
            label = btn.label or ["Idx MCP", "Idx PIP", "Idx Tip", "Mid MCP", "Mid PIP", "Mid Tip", "Palm"][idx]

            ui.html(
                f'<div class="knuckle-btn" '
                f'style="left:{left}; top:{top}; width:{width}; height:{height};" '
                f'onclick="document.getElementById(\'kb-{idx}\').click()">'
                f'{label}</div>'
            )
            # Hidden button for click handler
            btn_el = ui.button("", on_click=lambda _i=idx, _b=btn: _edit_knuckle_button(_i, _b)).props(
                "flat dense"
            ).style("display:none;")
            btn_el.props(f"id=kb-{idx}")


def _edit_knuckle_button(idx: int, button):
    """Edit knuckle button actions — same dialog as macropad buttons."""
    knuckle_btns = getattr(setting, 'knuckle_macropad_buttons', {})
    working = list(button.get_actions())

    names = ["Index MCP", "Index PIP", "Index Tip", "Middle MCP", "Middle PIP", "Middle Tip", "Palm"]
    btn_name = names[idx] if idx < len(names) else f"Button {idx + 1}"

    with ui.dialog() as dlg, ui.card().classes("w-[660px] s-card"):
        ui.label(f"Knuckle {idx + 1} — {btn_name}").classes("t-section")

        label_input = ui.input(label="Button Label", value=button.label or "",
                               placeholder=btn_name).props("outlined dense").classes("w-full mt-3")
        ui.separator().classes("my-3")

        ui.label("Actions").classes("t-label mb-2")
        actions_container = ui.column().classes("w-full gap-1 max-h-72 overflow-auto")

        def refresh():
            _build_action_rows(actions_container, working, None)

        def add():
            if len(working) >= 10:
                ui.notify("Max 10 actions", type="warning")
                return
            working.append(ActionBinding(type="none", value="", delay=0.17))
            refresh()

        refresh()

        ui.separator().classes("my-3")
        with ui.row().classes("justify-between w-full"):
            ui.button("Add Action", icon="add", on_click=add).props("flat no-caps color=primary")
            with ui.row().classes("gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat no-caps")

                def save():
                    button.label = label_input.value or ""
                    button.actions = working
                    knuckle_btns[idx] = button
                    setting.knuckle_macropad_buttons = knuckle_btns
                    dlg.close()
                    ui.notify("Saved", type="positive")

                ui.button("Save", icon="save", on_click=save).props("unelevated no-caps color=primary")
    dlg.open()


def _save_knuckle():
    save_setting(setting, SETTING_PATH)
    ui.notify("Knuckle pad saved", type="positive")


# ════════════════════════════════════════════════════════
#  Page 4 — Calibration
# ════════════════════════════════════════════════════════

def build_calibration_page():
    with ui.row().classes("items-center justify-between w-full mb-2"):
        with ui.column().classes("gap-1"):
            ui.label("ArUco Screen Calibration").classes("t-page")
            ui.label("Align 4 ArUco markers at screen corners for touch detection.").classes("t-desc")
        with ui.row().classes("gap-3"):
            ui.button("Start Calibration", icon="videocam",
                      on_click=lambda: _start_calibration()).props("unelevated no-caps color=primary")
            ui.button("Reset Offsets", icon="restart_alt",
                      on_click=_reset_calibration).props("outlined no-caps")

    ui.separator().classes("my-3")

    with ui.card().classes("w-full s-card"):
        with ui.row().classes("gap-8 w-full"):
            # Steps — left side
            with ui.column().classes("gap-1 flex-1"):
                ui.label("Setup").classes("t-section mb-1")
                for i, step in enumerate([
                    "Print calibration PDF from the Macro Pad tab.",
                    "Cut out markers ID 0–3 and place at screen corners.",
                    "Click Start Calibration to open camera preview.",
                    "Adjust sliders to align green box with screen edges, then save.",
                ], 1):
                    with ui.row().classes("items-baseline gap-2"):
                        ui.label(f"{i}.").classes("t-mono shrink-0 w-4 text-right")
                        ui.label(step).classes("t-label")

            # Marker layout — right side
            with ui.column().classes("shrink-0"):
                ui.label("Marker positions").classes("t-small mb-1")
                ui.html(
                    '<pre style="font-family: JetBrains Mono, monospace; font-size: 12px; '
                    'color: var(--text-secondary); margin: 0; line-height: 1.6; '
                    'background: var(--bg); padding: 8px 12px; border-radius: 6px; '
                    'border: 1px solid var(--border);">'
                    "ID 0  Top-Left      ID 1  Top-Right\n"
                    "ID 3  Bottom-Left   ID 2  Bottom-Right"
                    "</pre>"
                )


def _start_calibration():
    """Launch calibration UI as subprocess."""
    save_setting(setting, SETTING_PATH)
    try:
        script = str(Path(__file__).parent / "run_calibration.py")
        subprocess.Popen([sys.executable, script], cwd=str(ROOT))
        ui.notify("Calibration window opening...", type="info")
    except Exception as e:
        ui.notify(f"Failed: {e}", type="negative")


def _reset_calibration():
    import json
    cal_path = ROOT / "config" / "aruco_calibration.json"
    default = {c: {"horizontal": 0, "vertical": 0}
               for c in ["top_left", "top_right", "bottom_right", "bottom_left"]}
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(default, indent=2))
    ui.notify("Offsets reset", type="info")


# ════════════════════════════════════════════════════════
#  Page 4 — Settings
# ════════════════════════════════════════════════════════

def build_settings_page():
    s = setting

    with ui.row().classes("items-center justify-between w-full mb-2"):
        with ui.column().classes("gap-1"):
            ui.label("Settings").classes("t-page")
            ui.label("Camera, mouse control, and detection tuning.").classes("t-desc")
        ui.button("Save", icon="save", on_click=_save_settings).props("unelevated no-caps color=primary size=md")

    ui.separator().classes("my-4")

    # Camera
    with ui.card().classes("w-full s-card mb-5"):
        with ui.row().classes("items-center gap-2 mb-3"):
            ui.element("div").style("width: 8px; height: 8px; border-radius: 50%; background: var(--accent);")
            ui.label("Camera").classes("t-section")
        ui.separator()

        with ui.grid(columns=2).classes("gap-5 w-full mt-3"):
            ui.select(options=["webcam", "esp32"], value=getattr(s.camera, "source", "webcam"),
                      label="Source", on_change=lambda e: setattr(s.camera, "source", e.value)).props("outlined dense")
            ui.number(label="Camera Index", value=s.camera.index, min=0, max=10,
                      on_change=lambda e: setattr(s.camera, "index", int(e.value) if e.value is not None else 0)).props("outlined dense")
            ui.input(label="ESP32 Serial Port", value=getattr(s.camera, "esp32_serial_port", "/dev/cu.usbmodem101"),
                     on_change=lambda e: setattr(s.camera, "esp32_serial_port", e.value or "")).props("outlined dense")
            ui.number(label="ESP32 Baud Rate", value=getattr(s.camera, "esp32_baud_rate", 2000000),
                      on_change=lambda e: setattr(s.camera, "esp32_baud_rate", int(e.value) if e.value is not None else 2000000)).props("outlined dense")

        with ui.row().classes("gap-8 mt-4"):
            ui.switch("Flip Horizontal", value=s.camera.flip_horizontal,
                      on_change=lambda e: setattr(s.camera, "flip_horizontal", e.value))
            ui.switch("Flip Vertical", value=s.camera.flip_vertical,
                      on_change=lambda e: setattr(s.camera, "flip_vertical", e.value))
            ui.switch("Swap Hands", value=s.camera.swap_hands,
                      on_change=lambda e: setattr(s.camera, "swap_hands", e.value))

    # Mouse
    with ui.card().classes("w-full s-card mb-5"):
        with ui.row().classes("items-center gap-2 mb-3"):
            ui.element("div").style("width: 8px; height: 8px; border-radius: 50%; background: var(--green);")
            ui.label("Mouse Control").classes("t-section")
        ui.separator()
        _slider_row("Smoothing", s.mouse.smoothing, 0.1, 1.0, 0.01,
                     lambda v: setattr(s.mouse, "smoothing", v))
        _slider_row("Sensitivity", s.mouse.base_sensitivity, 0.5, 2.0, 0.01,
                     lambda v: setattr(s.mouse, "base_sensitivity", v))

    # Detection
    with ui.card().classes("w-full s-card"):
        with ui.row().classes("items-center gap-2 mb-3"):
            ui.element("div").style("width: 8px; height: 8px; border-radius: 50%; background: var(--orange);")
            ui.label("Detection").classes("t-section")
        ui.separator()
        _slider_row("Confidence", s.inference.confidence_threshold, 0.3, 0.9, 0.01,
                     lambda v: setattr(s.inference, "confidence_threshold", v))
        with ui.grid(columns=2).classes("gap-5 w-full mt-3"):
            ui.number(label="Cooldown Frames", value=s.inference.cooldown_frames, min=0, max=60,
                      on_change=lambda e: setattr(s.inference, "cooldown_frames", int(e.value) if e.value is not None else 12)).props("outlined dense")
            ui.number(label="Stability Window", value=s.inference.stability_window, min=1, max=20,
                      on_change=lambda e: setattr(s.inference, "stability_window", int(e.value) if e.value is not None else 5)).props("outlined dense")


def _slider_row(label_text, value, vmin, vmax, step, setter):
    with ui.row().classes("items-center gap-4 w-full mt-3"):
        ui.label(label_text).classes("w-28 shrink-0 t-label")
        slider = ui.slider(min=vmin, max=vmax, step=step, value=value,
                           on_change=lambda e: setter(e.value)).props("color=orange").classes("flex-grow")
        val_label = ui.label(f"{value:.2f}").classes("w-14 text-right t-mono")
        val_label.bind_text_from(slider, "value", backward=lambda v: f"{v:.2f}")


def _save_settings():
    save_setting(setting, SETTING_PATH)
    ui.notify("Settings saved", type="positive")


# ════════════════════════════════════════════════════════
#  Detection Control
# ════════════════════════════════════════════════════════

def _start_detection_subprocess():
    global detection_proc, detection_running
    if detection_running: return
    save_setting(setting, SETTING_PATH)
    try:
        script = str(Path(__file__).parent / "run_detection.py")
        detection_proc = subprocess.Popen([sys.executable, script], cwd=str(ROOT))
        detection_running = True
        ui.notify("Detection started", type="positive")
    except Exception as e:
        ui.notify(f"Failed: {e}", type="negative")
        detection_running = False

def _stop_detection():
    global detection_proc, detection_running
    if detection_proc:
        try:
            detection_proc.terminate()
            detection_proc.wait(timeout=3)
        except Exception:
            try: detection_proc.kill()
            except Exception: pass
        detection_proc = None
    detection_running = False
    ui.notify("Detection stopped", type="info")

def _toggle_detection():
    if detection_running: _stop_detection()
    else: _start_detection_subprocess()


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════

def main():
    global setting
    setting = load_setting(SETTING_PATH)

    ui.add_css(CUSTOM_CSS)
    ui.colors(primary="#2563EB")

    # Header
    with ui.header().classes("items-center justify-between app-header"):
        with ui.row().classes("items-center gap-2"):
            ui.label("HandFlow").style(
                "font-size: 18px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.4px;"
            )
            ui.element("span").style(
                "width: 1px; height: 20px; background: var(--border-strong); margin: 0 8px;"
            )
            ui.label("Gesture & Macro Control").style(
                "font-size: 13px; font-weight: 500; color: var(--text-tertiary);"
            )

        start_btn = ui.button("Run Detection", icon="play_arrow",
                              on_click=_toggle_detection).props("unelevated no-caps color=primary")
        stop_btn = ui.button("Stop Detection", icon="stop",
                             on_click=_toggle_detection).props("unelevated no-caps color=negative")
        stop_btn.set_visibility(False)

        def _poll():
            global detection_running
            if detection_proc and detection_proc.poll() is not None:
                detection_running = False
            start_btn.set_visibility(not detection_running)
            stop_btn.set_visibility(detection_running)
        ui.timer(1.0, _poll)

    # Tabs
    with ui.tabs().classes("w-full app-tabs") as tabs:
        ui.tab("Gestures", icon="pan_tool")
        ui.tab("Macro Pad", icon="grid_view")
        ui.tab("Knuckle Pad", icon="back_hand")
        ui.tab("Calibration", icon="tune")
        ui.tab("Settings", icon="settings")

    # Content
    with ui.tab_panels(tabs).classes("w-full flex-grow").props("animated").style("background: var(--bg);"):
        with ui.tab_panel("Gestures").classes("p-6"):
            build_gestures_page()
        with ui.tab_panel("Macro Pad").classes("p-6"):
            build_macropad_page()
        with ui.tab_panel("Knuckle Pad").classes("p-6"):
            build_knuckle_page()
        with ui.tab_panel("Calibration").classes("p-6"):
            build_calibration_page()
        with ui.tab_panel("Settings").classes("p-6"):
            build_settings_page()

    def _cleanup():
        global detection_proc
        if detection_proc:
            try:
                detection_proc.terminate()
                detection_proc.wait(timeout=2)
            except Exception: pass
    app.on_shutdown(_cleanup)

    ui.run(native=True, title="HandFlow", window_size=(1200, 800), reload=False)


if __name__ == "__main__":
    main()
