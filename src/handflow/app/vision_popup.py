"""Vision popup — image + draggable crop + Gemini chat. Entire UI is raw HTML."""

import asyncio
import json
import os
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from dotenv import load_dotenv
from nicegui import ui, app
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

load_dotenv(ROOT / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PUBLIC_DIR = ROOT / "captures" / "pub"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
app.add_static_files("/cap", str(PUBLIC_DIR))

MODEL_ID = "gemini-2.5-flash"

PROMPT = """What is in this image? Be brief (2-3 sentences max). Then suggest 2 quick actions.

---SUGGESTIONS---
- First action
- Second action
"""


def split_reply(raw):
    text = raw.strip()
    if "---SUGGESTIONS---" in text:
        main, rest = text.split("---SUGGESTIONS---", 1)
        sug = [m.group(1).strip() for l in rest.strip().splitlines()
               if (m := re.match(r"^\s*[-*•]\s+(.+)$", l.strip()))]
        return main.strip(), sug[:5]
    return text, []


@dataclass
class Msg:
    role: str
    text: str
    suggestions: list = field(default_factory=list)


def create_popup(image_path: str, crop_coords: dict = None):
    img = Image.open(image_path)
    W, H = img.size
    shutil.copy2(image_path, PUBLIC_DIR / "display.png")

    cx1 = crop_coords.get("x1", 0.1) if crop_coords else 0.1
    cy1 = crop_coords.get("y1", 0.1) if crop_coords else 0.1
    cx2 = crop_coords.get("x2", 0.9) if crop_coords else 0.9
    cy2 = crop_coords.get("y2", 0.9) if crop_coords else 0.9

    messages: list[Msg] = []
    gemini_contents = []
    busy = [False]

    # Chat container ref
    chat_container = [None]

    @ui.refreshable
    def chat_area():
        if not messages:
            if busy[0]:
                # Loading state
                with ui.column().classes("items-center justify-center h-full gap-4"):
                    ui.spinner("dots", size="lg").style("color:#E8590C;")
                    ui.label("Gemini is analyzing...").style(
                        "color:#E8590C;font-size:16px;font-weight:600;")
            else:
                with ui.column().classes("items-center justify-center h-full gap-4"):
                    ui.html('''
                        <div style="width:56px;height:56px;border-radius:16px;background:#FFF4ED;
                             display:flex;align-items:center;justify-content:center;">
                            <span style="font-size:24px;color:#E8590C;">✦</span>
                        </div>
                    ''')
                    ui.label(
                        "Click Analyze to start" if GEMINI_API_KEY else "Set GEMINI_API_KEY in .env"
                    ).style("color:#A8A5A0;font-size:15px;font-weight:500;")
        else:
            for msg in messages:
                if msg.role == "user":
                    with ui.row().classes("w-full justify-end my-2"):
                        ui.label(msg.text).style(
                            "background:#18171B;color:white;padding:12px 18px;"
                            "border-radius:18px 18px 4px 18px;font-size:16px;"
                            "max-width:85%;line-height:1.5;")
                elif msg.text == "_loading_":
                    # Loading indicator
                    with ui.row().classes("w-full my-3 gap-3 items-center"):
                        ui.spinner("dots", size="sm").style("color:#E8590C;")
                        ui.label("Thinking...").style("color:#E8590C;font-size:15px;font-weight:600;")
                else:
                    with ui.column().classes("w-full gap-3 my-3"):
                        ui.markdown(msg.text).style(
                            "background:#FFF7ED;border:1px solid #FED7AA;padding:20px;"
                            "border-radius:4px 18px 18px 18px;font-size:16px;font-weight:500;"
                            "line-height:1.8;color:#1a1a1a;")
                        if msg.suggestions:
                            with ui.row().classes("flex-wrap gap-2 mt-1"):
                                for s in msg.suggestions:
                                    ui.button(s, on_click=lambda t=s: asyncio.create_task(followup(t))).props(
                                        "flat dense no-caps").style(
                                        "border:1px solid #E8E6E1;border-radius:100px;padding:8px 18px;"
                                        "font-size:14px;font-weight:500;color:#5C5650;background:white;"
                                        "transition:all 0.12s;")

    async def get_crop(client):
        try:
            return await client.run_javascript("""
                var b=document.getElementById('crop-box'),i=document.getElementById('vc-img');
                if(!b||!i||!i.clientWidth)return null;
                return{x1:b.offsetLeft/i.clientWidth,y1:b.offsetTop/i.clientHeight,
                       x2:(b.offsetLeft+b.offsetWidth)/i.clientWidth,
                       y2:(b.offsetTop+b.offsetHeight)/i.clientHeight};
            """, timeout=5.0)
        except Exception:
            return None

    async def analyze(client=None, silent=False):
        """Analyze the cropped region only."""
        if busy[0] or not GEMINI_API_KEY:
            return
        busy[0] = True
        try:
            pil = Image.open(image_path)
            # Try to get crop from browser, fallback to initial coords
            c = None
            if client:
                c = await get_crop(client)
            if not c:
                c = {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2}
            x1, y1 = max(0, int(c["x1"]*W)), max(0, int(c["y1"]*H))
            x2, y2 = min(W, int(c["x2"]*W)), min(H, int(c["y2"]*H))
            if x2 > x1+10 and y2 > y1+10:
                pil = pil.crop((x1, y1, x2, y2))
            from google import genai
            from google.genai import types
            gc = genai.Client(api_key=GEMINI_API_KEY)
            gemini_contents.clear()
            gemini_contents.append(types.Content(role="user", parts=[
                types.Part(pil), types.Part.from_text(text=PROMPT.strip())]))
            messages.clear()
            if not silent:
                messages.append(Msg(role="user", text="What is in this image?"))
            messages.append(Msg(role="assistant", text="_loading_"))
            chat_area.refresh()
            r = await gc.aio.models.generate_content(model=MODEL_ID, contents=gemini_contents)
            raw = (r.text or "").strip() or "(no response)"
            main, sug = split_reply(raw)
            gemini_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=raw)]))
            # Replace loading with actual response
            messages.pop()  # Remove loading
            messages.append(Msg(role="assistant", text=main, suggestions=sug))
            chat_area.refresh()
        except Exception as e:
            messages.append(Msg(role="assistant", text=f"Error: {e}"))
            chat_area.refresh()
        finally:
            busy[0] = False

    async def followup(text):
        if busy[0] or not text.strip() or not gemini_contents:
            return
        busy[0] = True
        try:
            from google import genai
            from google.genai import types
            gc = genai.Client(api_key=GEMINI_API_KEY)
            gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text.strip())]))
            messages.append(Msg(role="user", text=text.strip()))
            messages.append(Msg(role="assistant", text="_loading_"))
            chat_area.refresh()
            r = await gc.aio.models.generate_content(model=MODEL_ID, contents=gemini_contents)
            raw = (r.text or "").strip() or "(no response)"
            main, sug = split_reply(raw)
            gemini_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=raw)]))
            messages.pop()  # Remove loading
            messages.append(Msg(role="assistant", text=main, suggestions=sug))
            chat_area.refresh()
        except Exception as e:
            messages.append(Msg(role="assistant", text=f"Error: {e}"))
            chat_area.refresh()
        finally:
            busy[0] = False

    async def save_and_copy(client=None):
        """Save cropped image to file and copy to clipboard."""
        pil = Image.open(image_path)
        if client:
            c = await get_crop_norm(client)
            if c:
                x1, y1 = max(0, int(c["x1"]*W)), max(0, int(c["y1"]*H))
                x2, y2 = min(W, int(c["x2"]*W)), min(H, int(c["y2"]*H))
                if x2 > x1+10 and y2 > y1+10:
                    pil = pil.crop((x1, y1, x2, y2))

        # Save to file
        fname = f"crop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out_path = ROOT / "captures" / fname
        pil.save(out_path, "PNG")

        # Copy to clipboard (macOS)
        import subprocess as _sp
        try:
            _sp.run(["osascript", "-e",
                     f'set the clipboard to (read (POSIX file "{out_path}") as «class PNGf»)'],
                    timeout=3)
            ui.notify(f"Saved & copied to clipboard: {fname}", type="positive")
        except Exception:
            ui.notify(f"Saved: {fname} (clipboard copy failed)", type="warning")

    # CSS
    ui.add_css("""
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        html,body{overflow:hidden!important;height:100vh;
            font-family:'DM Sans',-apple-system,sans-serif!important;background:#18171B!important;}
        /* Clean NiceGUI defaults */
        .nicegui-content{padding:0!important;gap:0!important;max-width:none!important;}
        .q-drawer{display:none!important;}
        .q-header{display:none!important;}
        .q-footer{display:none!important;}
        /* Scrollbar styling */
        ::-webkit-scrollbar{width:6px;}
        ::-webkit-scrollbar-track{background:transparent;}
        ::-webkit-scrollbar-thumb{background:rgba(0,0,0,0.15);border-radius:3px;}
        ::-webkit-scrollbar-thumb:hover{background:rgba(0,0,0,0.25);}
    """)

    # Layout: raw HTML left panel + NiceGUI right panel
    with ui.element("div").style("display:flex;width:100vw;height:100vh;overflow:hidden;"):

        # LEFT — raw HTML image + crop box
        with ui.element("div").style("width:62%;height:100vh;flex-shrink:0;background:#18171B;"):
            ui.html(f'''
                <div style="height:100vh;display:flex;flex-direction:column;background:#18171B;">
                    <div style="height:52px;display:flex;align-items:center;padding:0 24px;
                                background:#222126;border-bottom:1px solid rgba(255,255,255,0.08);flex-shrink:0;">
                        <div style="width:8px;height:8px;border-radius:50%;background:#E8590C;margin-right:10px;"></div>
                        <span style="font-size:15px;font-weight:600;color:rgba(255,255,255,0.9);
                                     font-family:'Instrument Sans',sans-serif;">Live Capture</span>
                        <span style="font-size:12px;color:rgba(255,255,255,0.3);margin-left:12px;
                                     font-family:monospace;">{W}x{H}</span>
                    </div>
                    <div style="flex:1;display:flex;align-items:center;justify-content:center;padding:24px;
                                min-height:0;overflow:hidden;">
                        <div id="img-wrap" style="position:relative;display:inline-block;line-height:0;">
                            <img id="vc-img" src="/cap/display.png"
                                 style="display:block;max-width:100%;max-height:calc(100vh - 100px);
                                        border-radius:6px;box-shadow:0 12px 48px rgba(0,0,0,0.6);">
                            <div id="crop-box" style="position:absolute;left:10%;top:10%;width:80%;height:80%;
                                 border:2px solid #E8590C;background:rgba(232,89,12,0.07);cursor:move;
                                 box-sizing:border-box;z-index:10;border-radius:3px;">
                                <div data-h="tl" style="position:absolute;width:14px;height:14px;background:#E8590C;
                                     border:2px solid #fff;border-radius:3px;top:-7px;left:-7px;cursor:nw-resize;
                                     z-index:11;box-shadow:0 2px 6px rgba(0,0,0,0.4);"></div>
                                <div data-h="tr" style="position:absolute;width:14px;height:14px;background:#E8590C;
                                     border:2px solid #fff;border-radius:3px;top:-7px;right:-7px;cursor:ne-resize;
                                     z-index:11;box-shadow:0 2px 6px rgba(0,0,0,0.4);"></div>
                                <div data-h="bl" style="position:absolute;width:14px;height:14px;background:#E8590C;
                                     border:2px solid #fff;border-radius:3px;bottom:-7px;left:-7px;cursor:sw-resize;
                                     z-index:11;box-shadow:0 2px 6px rgba(0,0,0,0.4);"></div>
                                <div data-h="br" style="position:absolute;width:14px;height:14px;background:#E8590C;
                                     border:2px solid #fff;border-radius:3px;bottom:-7px;right:-7px;cursor:se-resize;
                                     z-index:11;box-shadow:0 2px 6px rgba(0,0,0,0.4);"></div>
                            </div>
                        </div>
                    </div>
                    <div style="padding:10px 24px;font-size:13px;color:rgba(255,255,255,0.3);
                                border-top:1px solid rgba(255,255,255,0.06);flex-shrink:0;">
                        Drag box to move · drag corners to resize
                    </div>
                </div>
            ''')

        # RIGHT — Chat panel
        with ui.element("div").style(
            "width:38%;height:100vh;background:#FAFAF9;border-left:1px solid #E8E6E1;"
            "display:flex;flex-direction:column;overflow:hidden;"
        ):
            # Header
            with ui.element("div").style(
                "padding:18px 24px;border-bottom:1px solid #E8E6E1;background:white;"
                "display:flex;align-items:center;gap:14px;flex-shrink:0;"
            ):
                ui.html(
                    '<div style="width:42px;height:42px;border-radius:12px;background:#E8590C;'
                    'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
                    '<span style="color:white;font-size:16px;font-weight:700;">AI</span></div>'
                    '<div style="flex:1;">'
                    '<div style="font-weight:700;font-size:17px;color:#1a1a1a;letter-spacing:-0.3px;">Vision Assistant</div>'
                    '<div style="font-size:13px;font-weight:600;color:#E8590C;">Powered by Gemini ✦</div>'
                    '</div>'
                )
                ui.button("Analyze", icon="auto_awesome",
                          on_click=lambda: asyncio.create_task(analyze(ui.context.client))).props(
                    "unelevated no-caps").style(
                    "background:#E8590C;color:white;font-size:14px;font-weight:600;"
                    "padding:8px 22px;border-radius:10px;")
                ui.button("Save & Copy", icon="content_copy",
                          on_click=lambda: asyncio.create_task(save_and_copy(ui.context.client))).props(
                    "outline no-caps dense").style("font-size:13px;")

            # Chat messages
            with ui.element("div").style(
                "flex:1;overflow-y:auto;padding:24px;min-height:0;"
            ):
                chat_area()

            # Input bar
            with ui.element("div").style(
                "padding:18px 24px;border-top:1px solid #E8E6E1;background:white;"
                "display:flex;align-items:center;gap:12px;flex-shrink:0;"
            ):
                text_in = ui.input(placeholder="Ask something about the image...").props(
                    "outlined dense rounded").classes("flex-grow").style(
                    "font-size:15px;margin-left:4px;")

                def send():
                    t = (text_in.value or "").strip()
                    if not t:
                        return
                    text_in.value = ""
                    asyncio.create_task(followup(t))

                text_in.on("keydown.enter", send)
                ui.button(icon="send", on_click=send).props("round unelevated").style(
                    "background:#E8590C;color:white;width:40px;height:40px;")

    # Auto-analyze on load (silent — no user message shown)
    if GEMINI_API_KEY:
        async def _auto():
            await asyncio.sleep(2.0)
            await analyze(silent=True)
        ui.timer(0.1, lambda: asyncio.create_task(_auto()), once=True)

    # Crop box JS — uses pointer events + setPointerCapture (guaranteed to work)
    ui.add_body_html(f'''<script>
    (function() {{
        var _retries = 0;
        function setup() {{
            var box = document.getElementById('crop-box');
            var img = document.getElementById('vc-img');
            if (!box || !img) {{
                if (_retries++ < 50) setTimeout(setup, 200);
                return;
            }}

            // Wait for image to load
            function onReady() {{
                if (!img.clientWidth) {{ setTimeout(onReady, 100); return; }}

                // Position crop box
                box.style.left = ({cx1} * img.clientWidth) + 'px';
                box.style.top = ({cy1} * img.clientHeight) + 'px';
                box.style.width = ({cx2 - cx1} * img.clientWidth) + 'px';
                box.style.height = ({cy2 - cy1} * img.clientHeight) + 'px';

                console.log('Crop box initialized:', box.style.left, box.style.top, box.style.width, box.style.height);

                var mode = null, sx, sy, sl, st, sw, sh;
                function cl(v,a,b) {{ return Math.max(a, Math.min(b, v)); }}

                // Use pointerdown on box AND all children
                function onDown(e) {{
                    e.preventDefault();
                    e.stopPropagation();
                    e.stopImmediatePropagation();

                    var h = e.target.getAttribute('data-h');
                    mode = h || 'move';
                    sx = e.clientX; sy = e.clientY;
                    sl = box.offsetLeft; st = box.offsetTop;
                    sw = box.offsetWidth; sh = box.offsetHeight;

                    // Capture pointer — all future events go to this element
                    box.setPointerCapture(e.pointerId);
                    document.body.style.userSelect = 'none';
                    console.log('Crop drag start:', mode);
                }}

                function onMove(e) {{
                    if (!mode) return;
                    e.preventDefault();
                    e.stopPropagation();
                    var dx = e.clientX - sx, dy = e.clientY - sy;
                    var mw = img.clientWidth, mh = img.clientHeight;

                    if (mode === 'move') {{
                        box.style.left = cl(sl+dx, 0, mw-sw) + 'px';
                        box.style.top = cl(st+dy, 0, mh-sh) + 'px';
                    }} else if (mode === 'br') {{
                        box.style.width = cl(sw+dx, 40, mw-sl) + 'px';
                        box.style.height = cl(sh+dy, 40, mh-st) + 'px';
                    }} else if (mode === 'tl') {{
                        var nl = cl(sl+dx, 0, sl+sw-40), nt = cl(st+dy, 0, st+sh-40);
                        box.style.width = (sw+sl-nl)+'px'; box.style.height = (sh+st-nt)+'px';
                        box.style.left = nl+'px'; box.style.top = nt+'px';
                    }} else if (mode === 'tr') {{
                        box.style.width = cl(sw+dx, 40, mw-sl)+'px';
                        var nt2 = cl(st+dy, 0, st+sh-40);
                        box.style.height = (sh+st-nt2)+'px'; box.style.top = nt2+'px';
                    }} else if (mode === 'bl') {{
                        var nl2 = cl(sl+dx, 0, sl+sw-40);
                        box.style.width = (sw+sl-nl2)+'px'; box.style.left = nl2+'px';
                        box.style.height = cl(sh+dy, 40, mh-st)+'px';
                    }}
                }}

                function onUp(e) {{
                    if (mode) {{
                        console.log('Crop drag end');
                        mode = null;
                        document.body.style.userSelect = '';
                        try {{ box.releasePointerCapture(e.pointerId); }} catch(_) {{}}
                    }}
                }}

                // Attach to box element directly
                box.addEventListener('pointerdown', onDown);
                box.addEventListener('pointermove', onMove);
                box.addEventListener('pointerup', onUp);
                box.addEventListener('lostpointercapture', function() {{ mode = null; }});

                // Also attach to handles
                var handles = box.querySelectorAll('[data-h]');
                for (var i = 0; i < handles.length; i++) {{
                    handles[i].addEventListener('pointerdown', onDown);
                }}

                // Prevent any touch/pointer defaults on the image wrapper
                var wrap = document.getElementById('img-wrap');
                wrap.style.touchAction = 'none';
                box.style.touchAction = 'none';
            }}

            if (img.complete && img.naturalWidth) onReady();
            else img.addEventListener('load', onReady);
        }}

        // Try multiple times with different strategies
        setTimeout(setup, 500);
        document.addEventListener('DOMContentLoaded', function() {{ setTimeout(setup, 300); }});
        window.addEventListener('load', function() {{ setTimeout(setup, 500); }});
    }})();
    </script>''')


def main():
    if len(sys.argv) < 2:
        print("Usage: python vision_popup.py <image_path> [crop_json]")
        print("API key: set GEMINI_API_KEY in .env")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).is_file():
        print(f"Not found: {image_path}")
        sys.exit(1)

    crop_coords = None
    if len(sys.argv) >= 3:
        try:
            crop_coords = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            pass

    @ui.page("/")
    def index():
        create_popup(image_path, crop_coords)

    # Open as standalone app window (no browser tabs/menu)
    import webbrowser
    import subprocess
    import threading

    def open_app_window():
        import time
        time.sleep(1.5)
        # Try Chrome/Chromium app mode first (cleanest — no tabs, no URL bar)
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
        for path in chrome_paths:
            if os.path.exists(path):
                subprocess.Popen([path, f"--app=http://127.0.0.1:{port}", "--window-size=1400,850"])
                return
        # Fallback: regular browser
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_app_window, daemon=True).start()
    import random
    port = random.randint(8090, 8199)
    ui.run(title="HandFlow Vision", reload=False, port=port, show=False)


if __name__ in ("__main__", "__mp_main__"):
    main()
