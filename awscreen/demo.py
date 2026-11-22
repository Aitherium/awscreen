"""awscreen demo — a recorded, narrated proof that every context surface is explorable.

One run walks the six surfaces the owner named — **desktop, browser, terminal, file
explorer, code, APIs (HAR + system telemetry)** — and for each one it:

1. captures a frame (the full screen, or the window it opened and will close),
2. runs the real ``build_screen_state`` pipeline (propose -> crop -> neurons -> vision),
3. writes the ScreenState JSON, an annotated PNG (boxes in one colour, labels in a
   second), the bounding boxes and the model labels,
4. records the whole run with ``ffmpeg -f gdigrab`` and renders the deliverable MP4
   through media-forge's Op lane (``reassemble`` + ``film_export``; never a raw ffmpeg
   encode),
5. writes ``demo.json`` (the timeline) and ``demo-transcript.md`` (one line per surface).

Safety: it opens only its own windows, closes only those, never clicks inside an app and
never kills a process it did not start. All external effects are injectable so the
offline tests run with drawn screens, no model and no windows.

Invocation note: this environment may carry editable installs of ``awvision`` from an
older deploy tree. ``_prefer_worktree_siblings`` puts this worktree's sibling packages
first BEFORE the pipeline imports ``awvision``; without it the inspect step imports a
copy that predates ``get_vision_response_b64``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCHEMA = "awscreen.demo/1"
DEFAULT_FPS = 10
DEFAULT_HOLD_S = 2.5
DEFAULT_PRESET = "yt_169"
DEFAULT_MEDIAFORGE = "http://127.0.0.1:8200"

SURFACE_ORDER = ("desktop", "browser", "terminal", "file_explorer", "code", "apis")


# ── import resolution ───────────────────────────────────────────────────────


def _prefer_worktree_siblings() -> None:
    """Make this worktree's sibling packages (``awvision``) win over older editable installs.

    Some hosts carry a PEP 660 editable install whose meta-path finder maps ``awvision``
    to an older deploy tree. The finder runs before ``sys.path``, so merely inserting the
    sibling path is not enough: the finder has to be removed, then any module already
    imported from the wrong tree is dropped so the next import uses this worktree.
    """
    # demo.py -> awscreen/ (module dir) -> awscreen/ (package root) -> packages/
    packages = Path(__file__).resolve().parents[2]
    sibling = packages / "awvision"
    if not sibling.is_dir():
        return
    want = str(sibling.resolve())
    if want not in sys.path:
        sys.path.insert(0, want)
    sys.meta_path[:] = [f for f in sys.meta_path
                        if "awvision" not in getattr(f, "__module__", "")]
    mod = sys.modules.get("awvision")
    if mod is not None:
        loc = str(getattr(mod, "__file__", "") or "")
        if loc and not loc.lower().startswith(want.lower()):
            for name in [k for k in sys.modules
                         if k == "awvision" or k.startswith("awvision.")]:
                del sys.modules[name]


_prefer_worktree_siblings()


# ── records ─────────────────────────────────────────────────────────────────


@dataclass
class SurfaceSpec:
    """One context surface the demo proves."""

    name: str
    title: str
    kind: str = "window"          # 'screen' | 'window'
    opener: str = "none"          # browser | browser_context | terminal | explorer | code
    hint: str = ""                # window-title substring used to find what we opened


@dataclass
class Capture:
    """A frame plus the window handle that produced it (``window`` is ours to close)."""

    image: str = ""
    origin: Tuple[int, int] = (0, 0)
    window: Any = None
    status: str = "ok"            # ok | dead
    detail: str = ""


@dataclass
class SurfaceResult:
    name: str
    title: str
    status: str = "ok"            # ok | dead | skipped
    image: str = ""
    annotated: str = ""
    state: dict = field(default_factory=dict)
    elements: int = 0
    model_elements: int = 0
    proposer: str = ""
    timing: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    labels: List[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    capture_detail: str = ""
    wall_ms: float = 0.0
    error: str = ""


@dataclass
class DemoDeps:
    """Every external effect, injectable for the offline tests."""

    capture: Optional[Callable[[SurfaceSpec, Path, dict], Capture]] = None
    close: Optional[Callable[[Any], None]] = None
    state: Optional[Callable[[str, dict], dict]] = None
    context: Optional[Callable[[Path], dict]] = None
    annotate: Optional[Callable[[str, dict, Path], str]] = None
    video: Optional[Callable[[List[str], str, Path, dict], dict]] = None
    recorder_start: Optional[Callable[[Path, int], Any]] = None
    recorder_stop: Optional[Callable[[Any], dict]] = None
    clock: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep


# ── annotation ──────────────────────────────────────────────────────────────


BOX_COLOUR = (0, 196, 255)          # cv / ocr / grid elements
MODEL_BOX = (0, 235, 130)           # elements the vision model labelled
LABEL_COLOUR = (240, 240, 240)      # labels of non-model elements
MODEL_LABEL = (255, 214, 0)         # labels of model elements
HEADER_BG = (12, 14, 20)


def _is_model_element(el: dict) -> bool:
    return el.get("source") == "vlm" or "vlm" in (el.get("sources") or [])


def _wrap(text: str, width: int = 34) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def annotate(image_path: str, state: dict, out_path: Path,
             title: str = "") -> str:
    """Draw every element box, and its number/role/label/confidence in a second colour.

    ``state`` is a ScreenState dict; boxes are ABSOLUTE screen px and are re-based by
    ``frame.origin`` so a cropped window annotates at its own pixel space.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    ox, oy = state.get("frame", {}).get("origin") or (0, 0)
    elements = state.get("elements") or []
    size = max(12, min(38, int(w / 95)))
    try:
        font = ImageFont.truetype("arial.ttf", size)
        small = ImageFont.truetype("arial.ttf", max(10, size - 3))
    except (OSError, ValueError):
        font = ImageFont.load_default()
        small = font

    for i, el in enumerate(elements):
        box = el.get("bbox_image") or el.get("bbox") or [0, 0, 0, 0]
        if el.get("bbox_image"):
            x1, y1, x2, y2 = (int(v) for v in box)
        else:
            x1 = int(box[0]) - ox
            y1 = int(box[1]) - oy
            x2 = int(box[2]) - ox
            y2 = int(box[3]) - oy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        model = _is_model_element(el)
        colour = MODEL_BOX if model else BOX_COLOUR
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3 if model else 1)
        label = _wrap(el.get("label") or el.get("text") or el.get("role") or "element")
        conf = float(el.get("confidence") or 0.0)
        tag = f"{i}. {el.get('role', 'other')} {label} {conf:.2f}"
        tw = draw.textlength(tag, font=small)
        ty = y1 - size - 4 if y1 - size - 4 > 0 else y1 + 2
        ty = min(ty, h - size - 2)
        draw.rectangle([x1, ty, min(w - 1, x1 + tw + 6), ty + size + 2],
                       fill=(0, 0, 0))
        draw.text((x1 + 3, ty + 1), tag, fill=MODEL_LABEL if model else LABEL_COLOUR,
                  font=small)

    header = title or "surface"
    m = state.get("model") or {}
    counts = (f"{len(elements)} elements · "
              f"{sum(1 for e in elements if _is_model_element(e))} from the model · "
              f"proposer={state.get('proposer', {}).get('ran') or 'none'} · "
              f"model {m.get('model') or '—'} {m.get('ok', 0)}/{m.get('calls', 0)} ok")
    band = max(34, size * 2 + 12)
    draw.rectangle([0, 0, w, band], fill=HEADER_BG)
    draw.text((10, 4), header, fill=(255, 255, 255), font=font)
    draw.text((10, band - size - 4), counts, fill=(170, 220, 255), font=small)
    draw.rectangle([0, h - 26, w, h], fill=HEADER_BG)
    draw.text((10, h - 22),
              f"awscreen state · frame {w}x{h} · origin {ox},{oy}",
              fill=(150, 150, 160), font=small)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)


def render_text_card(title: str, lines: Sequence[str], size: Tuple[int, int],
                     out_path: Path) -> str:
    """A title/info card rendered in PIL, used for the video's opening frame."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    img = Image.new("RGB", (w, h), (8, 10, 16))
    draw = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("arial.ttf", max(28, w // 28))
        small = ImageFont.truetype("arial.ttf", max(16, w // 70))
    except (OSError, ValueError):
        big = ImageFont.load_default()
        small = big
    big_h = getattr(big, "size", max(28, w // 28))
    small_h = getattr(small, "size", max(16, w // 70))
    draw.text((w // 16, h // 8), title, fill=(255, 255, 255), font=big)
    y = h // 8 + big_h + 18
    for line in lines:
        draw.text((w // 16, y), line, fill=(170, 220, 255), font=small)
        y += small_h + 8
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return str(out_path)


# ── Windows windows ─────────────────────────────────────────────────────────

_IS_WINDOWS = sys.platform == "win32"
_WM_CLOSE = 0x0010
_SW_RESTORE = 9
_SW_MAXIMIZE = 3
_DWMWA_EXTENDED_FRAME_BOUNDS = 9
_CREATE_NO_WINDOW = 0x08000000 if _IS_WINDOWS else 0
_CREATE_NEW_CONSOLE = 0x00000010 if _IS_WINDOWS else 0


@dataclass
class Window:
    hwnd: int = 0
    proc: Any = None
    pid: int = 0
    label: str = ""
    may_terminate: bool = False


def _user32():
    import ctypes  # noqa: PLC0415 - Windows-only path
    return ctypes.windll.user32


def _list_windows() -> Dict[int, str]:
    if not _IS_WINDOWS:
        return {}
    import ctypes  # noqa: PLC0415

    user32 = _user32()
    out: Dict[int, str] = {}

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lparam):
        handle = ctypes.c_void_p(hwnd)
        if not user32.IsWindowVisible(handle):
            return True
        n = user32.GetWindowTextLengthW(handle)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(handle, buf, n + 1)
            title = buf.value.strip()
            if title:
                out[int(hwnd or 0)] = title
        return True

    user32.EnumWindows(_cb, 0)
    return out


def _window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    if not _IS_WINDOWS:
        return None
    import ctypes  # noqa: PLC0415

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    r = RECT()
    try:
        dwm = ctypes.windll.dwmapi
        hr = dwm.DwmGetWindowAttribute(ctypes.c_void_p(hwnd),
                                       ctypes.c_uint(_DWMWA_EXTENDED_FRAME_BOUNDS),
                                       ctypes.byref(r), ctypes.sizeof(r))
        if hr == 0 and r.right > r.left and r.bottom > r.top:
            return (int(r.left), int(r.top), int(r.right), int(r.bottom))
    except (AttributeError, OSError):
        r = RECT()   # DWM unavailable: fall through to GetWindowRect
    if _user32().GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(r)):
        return (int(r.left), int(r.top), int(r.right), int(r.bottom))
    return None


def _pid_of_window(hwnd: int) -> int:
    if not _IS_WINDOWS:
        return 0
    import ctypes  # noqa: PLC0415
    pid = ctypes.c_ulong(0)
    _user32().GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(pid))
    return int(pid.value)


def _focus(hwnd: int) -> None:
    if not _IS_WINDOWS or not hwnd:
        return
    import ctypes  # noqa: PLC0415
    user32 = _user32()
    user32.ShowWindow(ctypes.c_void_p(hwnd), _SW_MAXIMIZE)
    user32.BringWindowToTop(ctypes.c_void_p(hwnd))
    user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
    # Force above siblings without leaving the window topmost.
    hwnd_topmost, hwnd_notopmost = -1, -2
    flags = 0x0001 | 0x0002   # SWP_NOSIZE | SWP_NOMOVE
    user32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(hwnd_topmost),
                        0, 0, 0, 0, flags)
    user32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(hwnd_notopmost),
                        0, 0, 0, 0, flags)


def _close_window(window: Window) -> None:
    """Ask the window to close (WM_CLOSE). Never kills a process we did not start."""
    if window is None:
        return
    if window.hwnd and _IS_WINDOWS:
        import ctypes  # noqa: PLC0415
        _user32().PostMessageW(ctypes.c_void_p(window.hwnd), _WM_CLOSE, 0, 0)
    proc = window.proc
    if proc is not None and window.may_terminate:
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.terminate()   # our own child only
        except (OSError, ValueError):
            proc = None        # already gone: WM_CLOSE was enough


def _wait_new_window(before: Dict[int, str], hint: str = "", pid: int = 0,
                     timeout: float = 15.0) -> Optional[int]:
    """The first visible window that did not exist before, preferring ``hint``/``pid``."""
    deadline = time.time() + timeout
    fallback: Optional[int] = None
    while time.time() < deadline:
        windows = _list_windows()
        for hwnd, title in windows.items():
            if hwnd in before:
                continue
            rect = _window_rect(hwnd)
            if not rect or (rect[2] - rect[0]) < 180 or (rect[3] - rect[1]) < 120:
                continue
            if hint and hint.lower() in title.lower():
                return hwnd
            if pid and _pid_of_window(hwnd) == pid:
                return hwnd
            if fallback is None:
                fallback = hwnd
        time.sleep(0.2)
    return fallback


# ── surface openers ─────────────────────────────────────────────────────────


_BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _browser_exe() -> Optional[str]:
    for path in _BROWSER_CANDIDATES:
        if Path(path).is_file():
            return path
    return shutil.which("msedge") or shutil.which("chrome")


def _browser_html(path: Path, title: str, body: str) -> str:
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#0b1020;color:#e8edf7;margin:0;padding:40px}}
h1{{font-size:42px;margin:0 0 8px}}p.lead{{color:#9fb4d8;font-size:18px}}
.card{{background:#141c31;border:1px solid #2a3a5c;border-radius:12px;
padding:20px;margin:18px 0;max-width:1100px}}
.row{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}}
button{{font-size:19px;padding:12px 22px;border-radius:9px;border:0;background:#2f7cf6;color:#fff}}
input{{font-size:18px;padding:11px 14px;border-radius:9px;border:1px solid #3d5c8f;
width:340px;background:#0f1728;color:#fff}}
a{{color:#61d6ff;font-size:18px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #24324f;padding:9px 8px;text-align:left;font-size:16px}}
th{{color:#9fb4d8}}.pill{{display:inline-block;background:#1c3b2a;color:#53e39a;
border-radius:99px;padding:5px 12px;font-size:14px}}
</style></head><body>{body}</body></html>"""
    path.write_text(html, encoding="utf-8")
    return str(path)


def _open_surface(spec: SurfaceSpec, out_dir: Path, env: dict) -> Optional[Window]:
    """Open ONE window for this surface and return its handle (never touches the owner's)."""
    before = _list_windows()
    proc = None
    may_terminate = False
    label = spec.name
    if spec.opener == "browser" or spec.opener == "browser_context":
        exe = _browser_exe()
        if not exe:
            return None
        title = ("Vision Swarm API Dashboard" if spec.opener == "browser_context"
                 else "Vision Swarm Demo")
        html = env.get("dashboard_html") if spec.opener == "browser_context" else None
        path = out_dir / (spec.opener + ".html")
        if html is None:
            html = _browser_html(path, title, _browser_body())
        else:
            _browser_html(path, title, html)
        profile = out_dir / ("profile-" + spec.opener)
        cmd = [exe, f"--user-data-dir={profile}", "--no-first-run",
               "--no-default-browser-check",
               "--disable-features=msEdgeFirstRunExperience,msEdgeSidebarV2",
               "--new-window", "--window-size=1600,1000", "file:///" + str(path).replace("\\", "/")]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=_CREATE_NO_WINDOW)
        may_terminate = True
    elif spec.opener == "terminal":
        script = ("$Host.UI.RawUI.WindowTitle='Vision Swarm Demo Terminal'; "
                  "Write-Output 'vision-swarm demo — live context'; "
                  "Write-Output ('now: ' + (Get-Date).ToString('s')); "
                  "Get-Process | Sort-Object CPU -Descending | "
                  "Select-Object -First 8 Name,Id,CPU | Format-Table -AutoSize")
        enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        # Windows Terminal's default-terminal hand-off opens a new TAB in an existing
        # window, which is not "a window we opened". ``wt -w new`` forces a new window;
        # conhost is the classic fallback when Windows Terminal is absent.
        wt = shutil.which("wt")
        if wt:
            cmd = [wt, "-w", "new", "--title", "Vision Swarm Demo Terminal",
                   shell, "-NoLogo", "-NoProfile", "-NoExit", "-EncodedCommand", enc]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    creationflags=_CREATE_NO_WINDOW)
        else:
            conhost = shutil.which("conhost")
            cmd = ([conhost] if conhost else []) + [
                shell, "-NoLogo", "-NoProfile", "-NoExit", "-EncodedCommand", enc]
            proc = subprocess.Popen(cmd, creationflags=_CREATE_NEW_CONSOLE)
        may_terminate = True
    elif spec.opener == "explorer":
        target = env.get("explorer_path") or str(Path.home())
        proc = subprocess.Popen(["explorer.exe", target],
                                creationflags=_CREATE_NO_WINDOW)
        # explorer.exe hands the window to the running shell process and exits.
        may_terminate = False
    elif spec.opener == "code":
        readme = env.get("readme") or ""
        code = shutil.which("code")
        if code:
            proc = subprocess.Popen([code, "--new-window", readme],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=_CREATE_NO_WINDOW)
            may_terminate = False
        else:
            proc = subprocess.Popen(["notepad.exe", readme],
                                    creationflags=_CREATE_NO_WINDOW)
            may_terminate = True
    else:
        return None

    pid = int(getattr(proc, "pid", 0) or 0)
    hwnd = _wait_new_window(before, spec.hint, pid, timeout=18.0)
    if not hwnd:
        return None
    return Window(hwnd=hwnd, proc=proc, pid=pid, label=label, may_terminate=may_terminate)


def _browser_body() -> str:
    return """
<h1>Vision Swarm Demo</h1>
<p class="lead">Every element below is a real target for the neuron pipeline.</p>
<div class="card"><div class="row">
  <input placeholder="Search the platform..." aria-label="Search">
  <button>Sign in</button><button>Documentation</button>
  <a href="#">API reference</a>
</div></div>
<div class="card"><table>
<tr><th>Surface</th><th>Status</th><th>Elements</th></tr>
<tr><td>Desktop</td><td><span class="pill">live</span></td><td>capturing</td></tr>
<tr><td>Browser</td><td><span class="pill">live</span></td><td>this window</td></tr>
<tr><td>Terminal</td><td><span class="pill">live</span></td><td>pwsh</td></tr>
<tr><td>File explorer</td><td><span class="pill">live</span></td><td>folder</td></tr>
<tr><td>Code</td><td><span class="pill">live</span></td><td>README</td></tr>
</table></div>
<div class="card"><h2>Neurons + tools</h2>
<p>propose &rarr; crop &rarr; neurons &rarr; bounded parallel vision &rarr; merge &rarr;
DOM join, all on this host, routed through MicroScheduler.</p>
<button>Run inspection</button><button>Open HAR</button></div>
"""


# ── default capture / close ─────────────────────────────────────────────────


def default_capture(spec: SurfaceSpec, out_path: Path, env: dict,
                    sleep: Callable[[float], None] = time.sleep) -> Capture:
    if spec.kind == "screen":
        from awvision.sight import capture_screen  # noqa: PLC0415
        return Capture(image=capture_screen(str(out_path)), origin=(0, 0))
    window = _open_surface(spec, Path(env["out_dir"]), env)
    if window is None:
        return Capture(status="dead", detail=f"no {spec.opener} window appeared")
    _focus(window.hwnd)
    sleep(float(env.get("settle_s", 1.6)))
    rect = _window_rect(window.hwnd)
    if not rect:
        return Capture(status="dead", window=window, detail="window has no rect")
    left, top, right, bottom = rect
    try:
        from PIL import ImageGrab  # noqa: PLC0415
        img = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out_path)
    except (OSError, ValueError) as exc:
        return Capture(status="dead", window=window, detail=f"grab failed: {exc}")
    return Capture(image=str(out_path), origin=(left, top), window=window)


def default_close(window: Any) -> None:
    if isinstance(window, Window):
        _close_window(window)


# ── default state pipeline ──────────────────────────────────────────────────


def default_state(image: str, params: dict) -> dict:
    from awscreen.state import build_screen_state  # noqa: PLC0415

    proposer = str(params.get("proposer") or "auto")
    explicit = proposer != "auto"
    chain = (proposer,) if explicit else ("omniparser", "ocr", "blobs", "grid")
    state = build_screen_state(
        image=image,
        proposers=chain,
        explicit_proposer=explicit,
        inspect=bool(params.get("inspect", True)),
        max_regions=int(params.get("max_regions", 10)),
        max_concurrency=int(params.get("concurrency", 2)),
        timeout_s=float(params.get("timeout", 45)),
        origin=tuple(params.get("origin") or (0, 0)),
    )
    return state.to_dict()


# ── media-forge render lane ─────────────────────────────────────────────────


class MediaForge:
    """The curated Op surface only (``/op/{name}``), never a raw ffmpeg encode."""

    def __init__(self, base_url: Optional[str] = None, key: Optional[str] = None,
                 timeout: float = 600.0) -> None:
        self.base = (base_url or os.environ.get("AITHER_MEDIAFORGE_URL")
                     or DEFAULT_MEDIAFORGE).rstrip("/")
        self.key = key or os.environ.get("AITHER_INTERNAL_KEY") or _host_key()
        self.timeout = timeout

    def _headers(self, json_body: bool = False) -> dict:
        headers = {"X-Caller-Type": "platform"}
        if self.key:
            headers["X-Internal-Key"] = self.key
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 data: Optional[bytes] = None, content_type: str = "") -> bytes:
        headers = self._headers(json_body=body is not None)
        if content_type:
            headers["Content-Type"] = content_type
        payload = json.dumps(body).encode() if body is not None else data
        req = urllib.request.Request(self.base + path, data=payload, headers=headers,
                                     method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def op(self, name: str, params: dict) -> dict:
        raw = self._request("POST", f"/op/{name}", body=params)
        return json.loads(raw.decode("utf-8"))

    def upload(self, path: str) -> int:
        import mimetypes  # noqa: PLC0415
        import uuid  # noqa: PLC0415
        data = Path(path).read_bytes()
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        boundary = "----awscreen" + uuid.uuid4().hex
        head = (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="file"; filename="{Path(path).name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n").encode()
        tail = f"\r\n--{boundary}--\r\n".encode()
        raw = self._request("POST", "/api/upload", data=head + data + tail,
                            content_type=f"multipart/form-data; boundary={boundary}")
        doc = json.loads(raw.decode("utf-8"))
        if not doc.get("ok") or not doc.get("id"):
            raise RuntimeError(f"upload refused: {doc.get('error') or doc}")
        return int(doc["id"])

    def gallery(self, limit: int = 20) -> List[dict]:
        raw = self._request("GET", f"/api/gallery?limit={int(limit)}")
        doc = json.loads(raw.decode("utf-8"))
        return doc if isinstance(doc, list) else doc.get("items") or doc.get("media") or []

    def gallery_latest(self) -> Tuple[int, str]:
        items = self.gallery(1)
        if not items:
            return (0, "")
        return (int(items[0].get("id") or 0), str(items[0].get("local_path") or ""))

    def latest_child(self, parent_id: int) -> int:
        """film_export returns no ids; the artifact is the newest row derived from it."""
        children = [int(i.get("id") or 0) for i in self.gallery(30)
                    if int(i.get("parent_id") or 0) == int(parent_id)]
        return max(children) if children else 0

    def download(self, media_id: int, out_path: Path) -> str:
        raw = self._request("GET", f"/api/media/{media_id}")
        doc = json.loads(raw.decode("utf-8"))
        local = str(doc.get("local_path") or "")
        if not local:
            raise RuntimeError(f"media {media_id} has no local_path")
        data = self._request("GET", f"/media/{local}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return str(out_path)


def _host_key() -> str:
    path = Path.home() / ".aither" / "internal_key.txt"
    try:
        key = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    return key if len(key) >= 16 else ""


def _fit_frame(src: str, out_path: Path, size: Tuple[int, int]) -> str:
    """Letterbox one frame onto the deliverable canvas before it enters the Op lane.

    A 4K joined clip is four times the pixels of the deliverable and took minutes to
    encode; fitting first keeps the same content at 1920x1080 so the join is quick.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.open(src).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    canvas.save(out_path)
    return str(out_path)


def build_video(annotated: List[str], capture_video: str, out_path: Path,
                params: dict) -> dict:
    """Annotated slides (each held ``hold`` seconds) + the continuous capture, one MP4.

    media-forge owns the encode: frames are fitted to 1920x1080, ``reassemble`` builds
    the slides and joins the two 1080p clips (it preserves duration when ``fps`` equals
    the clips' native rate, which is why both are 10 fps), and ``film_export`` applies
    the final preset with faststart. A refusal is reported; it never falls back to a
    hand-run ffmpeg.
    """
    client = MediaForge(params.get("mediaforge_url"), params.get("mediaforge_key"))
    fps = int(params.get("fps", DEFAULT_FPS))
    hold = float(params.get("hold", DEFAULT_HOLD_S))
    preset = str(params.get("preset", DEFAULT_PRESET))
    canvas = tuple(params.get("render_size") or (1920, 1080))
    out_path = Path(out_path)
    fitted_dir = out_path.parent / "frames"
    fitted_dir.mkdir(parents=True, exist_ok=True)
    fitted = [_fit_frame(p, fitted_dir / f"slide_{i:02d}.png", canvas)
              for i, p in enumerate(annotated)]
    frame_ids = [client.upload(p) for p in fitted]
    copies = max(1, int(round(hold * fps)))
    slides_ids: List[int] = []
    for fid in frame_ids:
        slides_ids.extend([fid] * copies)
    slides = client.op("reassemble", {"frames": slides_ids, "fps": fps, "fmt": "mp4"})
    if not slides.get("ok") or not slides.get("head_ids"):
        return {"ok": False, "error": slides.get("error") or "reassemble refused"}
    slides_id = int(slides["head_ids"][0])
    final_id = slides_id
    joined = False
    warnings: List[str] = []
    if capture_video and Path(capture_video).is_file() and Path(capture_video).stat().st_size:
        try:
            cap_id = client.upload(capture_video)
            # Scale the recording to the deliverable BEFORE joining: a 4K join re-encoded
            # every frame twice and dominated the run.
            cap_export = client.op("film_export", {"video_id": cap_id, "preset": preset,
                                                    "timeout": 1800})
            cap_out = int((cap_export.get("head_ids") or [0])[0]) or client.latest_child(cap_id)
            if cap_export.get("ok") and cap_out:
                merged = client.op("reassemble", {"clips": [slides_id, cap_out],
                                                  "fps": fps, "fmt": "mp4"})
                if merged.get("ok") and merged.get("head_ids"):
                    final_id = int(merged["head_ids"][0])
                    joined = True
                else:
                    warnings.append(f"clip join refused: {merged.get('error')}")
            else:
                warnings.append(f"capture export refused: {cap_export.get('error')}")
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"capture join failed: {exc}")
    exported = client.op("film_export", {"video_id": final_id, "preset": preset,
                                          "timeout": 1800})
    if not exported.get("ok"):
        return {"ok": False, "error": exported.get("error") or "film_export refused",
                "slides_id": slides_id, "warnings": warnings}
    out_id = int((exported.get("head_ids") or [0])[0])
    if not out_id:
        out_id = client.latest_child(final_id)
    if not out_id:
        out_id, _ = client.gallery_latest()
    path = client.download(out_id, out_path)
    return {"ok": True, "path": path, "media_id": out_id, "slides_id": slides_id,
            "capture_joined": joined, "warnings": warnings}


# ── continuous screen recording (capture is allowed to use gdigrab) ─────────


class ScreenRecorder:
    def __init__(self, out_path: Path, fps: int = DEFAULT_FPS) -> None:
        self.out_path = Path(out_path)
        self.fps = fps
        self.proc: Optional[subprocess.Popen] = None
        self.started = 0.0

    def start(self) -> "ScreenRecorder":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is not on PATH (capture lane)")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "gdigrab", "-framerate", str(self.fps), "-i", "desktop",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
               str(self.out_path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     creationflags=_CREATE_NO_WINDOW)
        self.started = time.time()
        return self

    def stop(self) -> dict:
        if self.proc is None:
            return {"status": "skipped"}
        try:
            if self.proc.stdin:
                self.proc.stdin.write(b"q")
                self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.SubprocessError):
                self.proc = None   # recorder already exited; the file is what it is
        size = self.out_path.stat().st_size if self.out_path.is_file() else 0
        return {"status": "ok" if size else "dead", "path": str(self.out_path),
                "bytes": size, "seconds": round(time.time() - self.started, 1),
                "fps": self.fps}


def default_recorder_start(out_path: Path, fps: int) -> ScreenRecorder:
    return ScreenRecorder(out_path, fps).start()


def default_recorder_stop(recorder: Any) -> dict:
    if recorder is None:
        return {"status": "skipped"}
    return recorder.stop()


# ── API / telemetry context (gateway MCP tools + a real HAR) ───────────────


class Gateway:
    """Minimal MCP client: initialize, then tools/call over the local gateway."""

    def __init__(self, url: str = "http://127.0.0.1:8182/mcp") -> None:
        self.url = url
        self.token = _session_bearer()
        self.session = ""

    def _post(self, body: dict) -> Tuple[bytes, dict]:
        headers = {"Authorization": f"Bearer {self.token}",
                   "Accept": "application/json, text/event-stream",
                   "Content-Type": "application/json"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.read(), {k.lower(): v for k, v in resp.headers.items()}

    def initialize(self) -> bool:
        if not self.token:
            return False
        raw, headers = self._post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "awscreen-demo", "version": "1"}}})
        self.session = headers.get("mcp-session-id", "")
        if self.session:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return True

    def call(self, name: str, arguments: Optional[dict] = None) -> Tuple[Any, float]:
        started = time.time()
        raw, _ = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                             "params": {"name": name, "arguments": arguments or {}}})
        elapsed_ms = round((time.time() - started) * 1000, 1)
        for obj in _sse_json(raw):
            result = obj.get("result") or {}
            content = result.get("content") or []
            text = content[0].get("text", "") if content else ""
            if result.get("isError"):
                return {"error": text[:300]}, elapsed_ms
            try:
                return json.loads(text), elapsed_ms
            except (ValueError, TypeError):
                return text, elapsed_ms
        return {"error": "no JSON-RPC result"}, elapsed_ms


def _session_bearer() -> str:
    env = os.environ.get("AITHER_SESSION_BEARER", "").strip()
    if env:
        return env
    try:
        return (Path.home() / ".aither" / "session-bearer").read_text(
            encoding="utf-8").strip()
    except OSError:
        return ""


def _sse_json(raw: bytes) -> List[dict]:
    out: List[dict] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _har_entry(url: str, method: str, status: int, started_ms: float, time_ms: float,
               mime: str, body: Any, size: int = 0) -> dict:
    text = body if isinstance(body, str) else json.dumps(body)[:12000]
    return {
        "startedDateTime": datetime.fromtimestamp(started_ms / 1000, timezone.utc)
        .isoformat().replace("+00:00", "Z"),
        "time": time_ms,
        "request": {"method": method, "url": url,
                    "httpVersion": "HTTP/1.1", "headers": [],
                    "queryString": [], "cookies": [], "headersSize": -1,
                    "bodySize": 0},
        "response": {"status": status, "statusText": "OK",
                     "httpVersion": "HTTP/1.1", "headers": [],
                     "cookies": [],
                     "content": {"size": size or len(text), "mimeType": mime,
                                 "text": text},
                     "redirectURL": "", "headersSize": -1,
                     "bodySize": size or len(text)},
        "cache": {}, "timings": {"send": 0, "wait": time_ms, "receive": 0},
    }


def _har_page(entries: List[dict]) -> dict:
    return {"log": {"version": "1.2",
                    "creator": {"name": "awscreen demo", "version": "1"},
                    "entries": entries}}


def _har_summary(entries: List[dict]) -> dict:
    from collections import Counter  # noqa: PLC0415
    from urllib.parse import urlparse  # noqa: PLC0415

    hosts = Counter(urlparse(e["request"]["url"]).netloc or "local"
                    for e in entries)
    methods = Counter(e["request"]["method"] for e in entries)
    return {"requests": len(entries),
            "hosts": [{"host": h, "requests": n} for h, n in hosts.most_common(6)],
            "methods": dict(methods)}


def _summarise(value: Any, depth: int = 0) -> Any:
    """Small, JSON-safe shape of a tool answer: counts and a bounded sample."""
    if depth > 4:
        return "<nested>"
    if isinstance(value, dict):
        result = value.get("result") if isinstance(value.get("result"), dict) else value
        rows = result.get("rows") if isinstance(result, dict) else None
        if isinstance(rows, list):
            return {"count": len(rows), "sample": rows[:8]}
        return {k: _summarise(v, depth + 1) for k, v in list(value.items())[:12]}
    if isinstance(value, list):
        return {"count": len(value), "sample": [_summarise(v, depth + 1)
                                                for v in value[:6]]}
    if isinstance(value, str):
        return value[:400]
    return value


def collect_context(out_dir: Path, gateway: Optional[Gateway] = None) -> dict:
    """System telemetry + a real HAR of this run's own requests, through the gateway.

    The gateway's ``har_*`` tools run inside a container that cannot see a Windows host
    path. When ``AWSCREEN_HAR_BRIDGE_HOST_DIR`` (a host directory mounted into that
    container) and ``AWSCREEN_HAR_BRIDGE_CONTAINER_DIR`` (its container path) are set,
    the HAR is briefly copied there, indexed through the gateway, and deleted. Without
    the bridge the local artifact is still written and parsed here; the gateway refusal
    is recorded verbatim rather than hidden.
    """
    out_dir = Path(out_dir)
    gw = gateway or Gateway()
    context: Dict[str, Any] = {"captured_at": datetime.now(timezone.utc).isoformat(),
                               "tools": {}, "har": {}}
    entries: List[dict] = []
    gateway_ok = False
    try:
        gateway_ok = gw.initialize()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        context["gateway_error"] = f"{type(exc).__name__}: {exc}"
    calls = ("get_system_status", "desktop_processes", "desktop_ports",
             "gpu_lease_status", "browser_context")
    for name in calls:
        started = time.time() * 1000
        if not gateway_ok:
            context["tools"][name] = {"error": context.get("gateway_error", "gateway down")}
            continue
        try:
            answer, elapsed = gw.call(name)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            answer, elapsed = {"error": f"{type(exc).__name__}: {exc}"}, 0.0
        context["tools"][name] = _summarise(answer)
        status = 200 if not (isinstance(answer, dict) and answer.get("error")) else 500
        entries.append(_har_entry(f"mcp://gateway/tools/call?name={name}", "POST", status,
                                  started, elapsed, "application/json", answer))
    try:
        started = time.time() * 1000
        with urllib.request.urlopen(f"{DEFAULT_MEDIAFORGE}/ops", timeout=8) as resp:
            ops = json.loads(resp.read().decode("utf-8"))
        elapsed = round(time.time() * 1000 - started, 1)
        context["tools"]["mediaforge_ops"] = {"count": len(ops.get("ops") or [])}
        entries.append(_har_entry(f"{DEFAULT_MEDIAFORGE}/ops", "GET", 200, started,
                                  elapsed, "application/json", ops))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        context["tools"]["mediaforge_ops"] = {"error": f"{type(exc).__name__}: {exc}"}
    har_path = out_dir / "demo.har"
    har_path.parent.mkdir(parents=True, exist_ok=True)
    har_path.write_text(json.dumps(_har_page(entries), indent=1), encoding="utf-8")
    context["har"] = {"path": str(har_path), "requests": len(entries),
                      "local": _har_summary(entries)}
    if gateway_ok and entries:
        gateway_har_path = str(har_path)
        bridged: Optional[Path] = None
        bridge_host = os.environ.get("AWSCREEN_HAR_BRIDGE_HOST_DIR", "")
        bridge_ctr = os.environ.get("AWSCREEN_HAR_BRIDGE_CONTAINER_DIR", "")
        if bridge_host and bridge_ctr and Path(bridge_host).is_dir():
            try:
                bridged = Path(bridge_host) / f".awscreen-demo-{os.getpid()}.har"
                shutil.copyfile(har_path, bridged)
                gateway_har_path = bridge_ctr.rstrip("/") + "/" + bridged.name
            except OSError as exc:
                context["har"]["bridge_error"] = f"{type(exc).__name__}: {exc}"
                bridged = None
        context["har"]["gateway_path"] = gateway_har_path
        context["har"]["bridged"] = bridged is not None
        try:
            opened, _ = gw.call("har_open", {"har_path": gateway_har_path})
            context["har"]["gateway"] = _summarise(opened)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            context["har"]["gateway_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if bridged is not None:
                try:
                    bridged.unlink()
                except OSError:
                    bridged = None   # already removed; nothing to clean up
    else:
        context["har"]["gateway"] = {"skipped": "gateway unavailable"}
    context["processes"] = _process_summary(context["tools"].get("desktop_processes"))
    context["ports"] = _port_summary(context["tools"].get("desktop_ports"))
    return context


def _process_summary(tool: Any) -> dict:
    rows = ((tool or {}).get("sample") or []) if isinstance(tool, dict) else []
    count = int(tool.get("count") or 0) if isinstance(tool, dict) else 0
    top = sorted(rows, key=lambda r: _num(r.get("priv_kb")), reverse=True)[:5]
    return {"count": count,
            "top_by_memory": [{"name": r.get("name"), "pid": r.get("pid"),
                               "priv_kb": r.get("priv_kb")} for r in top]}


def _port_summary(tool: Any) -> dict:
    rows = ((tool or {}).get("sample") or []) if isinstance(tool, dict) else []
    count = int(tool.get("count") or 0) if isinstance(tool, dict) else 0
    listening = sum(1 for r in rows if str(r.get("state") or "").upper() == "LISTENING")
    return {"count": count, "listening_in_sample": listening,
            "sample": [{"process": r.get("process"), "pid": r.get("pid"),
                        "local": r.get("local"), "state": r.get("state")}
                       for r in rows[:5]]}


def _num(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def dashboard_html(context: dict) -> str:
    procs = context.get("processes") or {}
    ports = context.get("ports") or {}
    har = context.get("har") or {}
    tools = context.get("tools") or {}
    rows = "".join(
        f"<tr><td>{r.get('name')}</td><td>{r.get('pid')}</td>"
        f"<td>{r.get('priv_kb')} KB</td></tr>"
        for r in procs.get("top_by_memory") or [])
    status = tools.get("get_system_status") or {}
    return f"""
<h1>Vision Swarm API Dashboard</h1>
<p class="lead">Live system telemetry and a real HAR, captured by the demo itself.</p>
<div class="card row">
  <span class="pill">gateway: {status.get('health', 'unknown')}</span>
  <span class="pill">queue depth {status.get('queue_depth', '?')}</span>
  <span class="pill">processes {procs.get('count', 0)}</span>
  <span class="pill">ports {ports.get('count', 0)}</span>
  <span class="pill">HAR {har.get('requests', 0)} requests</span>
</div>
<div class="card"><h2>Top processes by memory</h2>
<table><tr><th>Process</th><th>PID</th><th>Private</th></tr>{rows}</table></div>
<div class="card"><h2>HAR artifact</h2>
<p>{har.get('path', '')}</p>
<p>har_open: {json.dumps(_summarise(har.get('gateway')))[:300]}</p></div>
<div class="card row"><button>Refresh telemetry</button><button>Open HAR</button>
<a href="#">Ports</a><input placeholder="Search processes..."></div>
"""


# ── the run ─────────────────────────────────────────────────────────────────


def default_surfaces() -> List[SurfaceSpec]:
    return [
        SurfaceSpec("desktop", "DESKTOP — full screen", kind="screen"),
        SurfaceSpec("browser", "BROWSER — Edge window", opener="browser",
                    hint="Vision Swarm Demo"),
        SurfaceSpec("terminal", "TERMINAL — pwsh", opener="terminal",
                    hint="Vision Swarm"),
        SurfaceSpec("file_explorer", "FILE EXPLORER — folder", opener="explorer",
                    hint="AitherOS-Fresh"),
        SurfaceSpec("code", "CODE — editor on a README", opener="code",
                    hint="README"),
        SurfaceSpec("apis", "APIs — HAR + system telemetry", opener="browser_context",
                    hint="Vision Swarm API"),
    ]


def _element_labels(state: dict) -> List[dict]:
    out = []
    for el in state.get("elements") or []:
        if not _is_model_element(el):
            continue
        out.append({"id": el.get("id"), "role": el.get("role"),
                    "label": el.get("label"), "text": el.get("text"),
                    "confidence": el.get("confidence"), "bbox": el.get("bbox"),
                    "status": el.get("status")})
    return out


def run_demo(surfaces: Optional[Sequence[SurfaceSpec]] = None,
             deps: Optional[DemoDeps] = None, params: Optional[dict] = None) -> dict:
    """Walk every surface once and return the timeline record (``demo.json``)."""
    params = dict(params or {})
    deps = deps or DemoDeps()
    capture = deps.capture or default_capture
    close = deps.close or default_close
    state_fn = deps.state or default_state
    annotate_fn = deps.annotate or annotate
    video_fn = deps.video or build_video
    context_fn = deps.context or collect_context
    recorder_start = deps.recorder_start or default_recorder_start
    recorder_stop = deps.recorder_stop or default_recorder_stop

    out_dir = Path(params.get("out_dir") or Path.home() / ".aither" / "awscreen" / "demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = list(surfaces or default_surfaces())
    started_at = datetime.now(timezone.utc).isoformat()

    env = {"out_dir": str(out_dir), "settle_s": params.get("settle", 1.6),
           "explorer_path": params.get("explorer_path") or str(Path.home()),
           "readme": params.get("readme") or _default_readme()}

    context: Dict[str, Any] = {}
    if any(s.name == "apis" for s in specs):
        try:
            context = context_fn(out_dir) or {}
        except (OSError, ValueError, RuntimeError) as exc:
            context = {"error": f"{type(exc).__name__}: {exc}"}
        env["dashboard_html"] = dashboard_html(context)
        (out_dir / "context.json").write_text(json.dumps(context, indent=1),
                                              encoding="utf-8")

    recorder = None
    capture_info: Dict[str, Any] = {"status": "skipped"}
    if params.get("record", True):
        try:
            recorder = recorder_start(out_dir / "vision-swarm-capture.mp4",
                                      int(params.get("fps", DEFAULT_FPS)))
        except (OSError, RuntimeError) as exc:
            capture_info = {"status": "dead", "error": str(exc)}

    results: List[SurfaceResult] = []
    try:
        for spec in specs:
            wall0 = deps.clock()
            result = SurfaceResult(name=spec.name, title=spec.title)
            try:
                cap = capture(spec, out_dir / f"{spec.name}.png", env)
            except (OSError, ValueError, RuntimeError) as exc:
                cap = Capture(status="dead", detail=f"{type(exc).__name__}: {exc}")
            result.capture_detail = cap.detail
            if cap.status != "ok" or not cap.image:
                result.status = "dead"
                result.error = cap.detail or "capture produced no frame"
                close(cap.window)
                result.wall_ms = round((deps.clock() - wall0) * 1000, 1)
                results.append(result)
                continue
            try:
                state = state_fn(cap.image, {"origin": cap.origin, **params})
            except (OSError, ValueError, RuntimeError) as exc:
                state = {"frame": {"path": cap.image, "origin": cap.origin},
                         "elements": [], "model": {},
                         "model_error": f"{type(exc).__name__}: {exc}"}
                result.status = "dead"
                result.error = f"state failed: {exc}"
            close(cap.window)
            if spec.name == "apis":
                state["context"] = context
            result.state = state
            result.image = cap.image
            result.elements = len(state.get("elements") or [])
            result.model_elements = sum(1 for e in state.get("elements") or []
                                        if _is_model_element(e))
            result.proposer = (state.get("proposer") or {}).get("ran") or ""
            result.timing = state.get("timing") or {}
            result.model = state.get("model") or {}
            result.labels = _element_labels(state)
            if spec.name == "apis":
                result.context = context
            try:
                result.annotated = annotate_fn(cap.image, state,
                                               out_dir / f"{spec.name}.annotated.png",
                                               spec.title)
            except (OSError, ValueError) as exc:
                result.annotated = ""
                result.error = result.error or f"annotate failed: {exc}"
            result.wall_ms = round((deps.clock() - wall0) * 1000, 1)
            results.append(result)
    finally:
        try:
            capture_info = recorder_stop(recorder) if params.get("record", True) \
                else {"status": "skipped"}
        except (OSError, subprocess.SubprocessError) as exc:
            capture_info = {"status": "dead", "error": str(exc)}

    annotated = [r.annotated for r in results if r.annotated]
    title_card = ""
    if params.get("record", True):
        try:
            w, h = params.get("render_size") or _screen_size()
            title_card = render_text_card(
                "Vision Swarm — recorded demo", [
                    "Every context surface: desktop, browser, terminal,",
                    "file explorer, code, APIs (HAR + system telemetry).",
                    f"Surfaces: {len(results)} · started {started_at}",
                ], (w, h), out_dir / "title.png")
        except (OSError, ValueError):
            title_card = ""
        if title_card:
            annotated.insert(0, title_card)
    video_info: Dict[str, Any] = {"status": "skipped"}
    if params.get("render_video", True) and annotated:
        try:
            video_info = video_fn(annotated, capture_info.get("path", ""),
                                  out_dir / "vision-swarm-demo.mp4", params)
            video_info = {"status": "ok" if video_info.get("ok") else "dead", **video_info}
        except (OSError, ValueError, RuntimeError) as exc:
            video_info = {"status": "dead", "error": f"{type(exc).__name__}: {exc}"}
    record = {
        "schema": SCHEMA,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "host": os.environ.get("COMPUTERNAME") or "",
        "params": {k: params[k] for k in ("fps", "hold", "max_regions", "concurrency",
                                          "timeout", "proposer", "inspect", "preset")
                   if k in params},
        "capture": capture_info,
        "video": video_info,
        "surfaces": [asdict(r) for r in results],
        "totals": {
            "surfaces": len(results),
            "ok": sum(1 for r in results if r.status == "ok"),
            "elements": sum(r.elements for r in results),
            "model_elements": sum(r.model_elements for r in results),
        },
    }
    record["context"] = context
    (out_dir / "demo.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (out_dir / "demo-transcript.md").write_text(transcript(record), encoding="utf-8")
    return record


def _default_readme() -> str:
    """A real README on disk: the package's own if the worktree has one, else the host's."""
    package_readme = Path(__file__).resolve().parent.parent / "README.md"
    for cand in (Path.cwd() / "README.md", package_readme,
                 Path.cwd() / "AGENTS.md",
                 Path.cwd() / "CLAUDE.md"):
        if cand.is_file():
            return str(cand)
    return str(Path.home() / "README.md")


def _screen_size() -> Tuple[int, int]:
    size = (1920, 1080)
    if _IS_WINDOWS:
        try:
            from PIL import ImageGrab  # noqa: PLC0415
            img = ImageGrab.grab()
            size = (img.width, img.height)
        except (OSError, ValueError):
            size = (1920, 1080)   # headless: the deliverable's default canvas
    return size


# ── transcript ──────────────────────────────────────────────────────────────


def transcript(record: dict) -> str:
    lines = ["# Vision swarm demo — recorded transcript", "",
             f"Started {record.get('started_at')} · host {record.get('host')}"
             f" · schema {record.get('schema')}", ""]
    for surface in record.get("surfaces") or []:
        name = surface.get("name")
        if surface.get("status") != "ok":
            lines.append(f"- **{name}** — DEAD: {surface.get('error') or 'no frame'}")
            continue
        model = surface.get("model") or {}
        timing = surface.get("timing") or {}
        labels = [lab.get("label") or lab.get("text") for lab in
                  (surface.get("labels") or []) if lab.get("label") or lab.get("text")]
        shown = ", ".join(f'"{lab}"' for lab in labels[:5]) or "none"
        more = f" (+{len(labels) - 5})" if len(labels) > 5 else ""
        lines.append(
            f"- **{name}** — {surface.get('elements', 0)} elements, "
            f"{surface.get('model_elements', 0)} model-labelled · "
            f"proposer={surface.get('proposer') or 'none'} · "
            f"total {timing.get('total_ms', 0)} ms (inspect {timing.get('inspect_ms', 0)} ms)"
            f" · model {model.get('model') or '—'} "
            f"{model.get('ok', 0)}/{model.get('calls', 0)} ok"
            f"{' ERROR ' + str(model.get('error')) if model.get('error') else ''}"
            f" · model labels: {shown}{more}")
    capture = record.get("capture") or {}
    video = record.get("video") or {}
    lines += ["", f"Capture: {capture.get('status')} {capture.get('path') or ''}"
                  f" ({capture.get('bytes', 0)} bytes, {capture.get('seconds', 0)} s)",
              f"Video: {video.get('status')} {video.get('path') or ''}"
              f" ({video.get('error') or 'media-forge ' + str(video.get('media_id') or '')})",
              ""]
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="awscreen demo",
        description="Recorded proof that every context surface is explorable")
    ap.add_argument("--out-dir", default=str(Path.home() / ".aither" / "awscreen" / "demo"))
    ap.add_argument("--surfaces", default=",".join(SURFACE_ORDER),
                    help="comma-separated subset of " + ",".join(SURFACE_ORDER))
    ap.add_argument("--max-regions", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--proposer", default="auto",
                    choices=["auto", "omniparser", "ocr", "blobs", "grid"])
    ap.add_argument("--no-inspect", action="store_true")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--hold", type=float, default=DEFAULT_HOLD_S)
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--settle", type=float, default=1.6)
    ap.add_argument("--no-record", action="store_true", help="skip gdigrab")
    ap.add_argument("--no-video", action="store_true", help="skip the media-forge render")
    ap.add_argument("--explorer-path", default=str(Path.home()))
    ap.add_argument("--readme", default="")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true", help="print demo.json on completion")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    specs = [s for s in default_surfaces() if s.name in set(args.surfaces.split(","))]
    params = {"out_dir": args.out_dir, "max_regions": args.max_regions,
              "concurrency": args.concurrency, "timeout": args.timeout,
              "proposer": args.proposer, "inspect": not args.no_inspect,
              "fps": args.fps, "hold": args.hold, "preset": args.preset,
              "settle": args.settle, "record": not args.no_record,
              "render_video": not args.no_video,
              "explorer_path": args.explorer_path, "readme": args.readme}
    record = run_demo(specs, params=params)
    if args.json:
        print(json.dumps(record, indent=1))
    else:
        print(transcript(record))
        print(f"demo.json: {Path(args.out_dir) / 'demo.json'}")
    return 0 if record["totals"]["ok"] else 1


# ── offline self-test ───────────────────────────────────────────────────────


def self_test() -> int:
    """Drawn screens, a fake surface list, no model and no windows."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        truth = {}

        def draw(name: str) -> str:
            img = Image.new("RGB", (900, 500), (18, 22, 32))
            d = ImageDraw.Draw(img)
            d.rectangle([40, 40, 300, 100], fill=(40, 90, 200))
            d.text((60, 60), "Sign in", fill=(255, 255, 255))
            d.rectangle([340, 40, 700, 100], outline=(200, 200, 200), width=3)
            d.text((360, 60), "Search the platform", fill=(120, 220, 255))
            d.rectangle([40, 150, 260, 220], fill=(40, 120, 70))
            d.text((60, 175), "Documentation", fill=(255, 255, 255))
            path = root / f"{name}.png"
            img.save(path)
            truth[name] = path
            return str(path)

        def fake_capture(spec, out_path, env):
            path = draw(spec.name)
            return Capture(image=path, origin=(0, 0))

        def fake_context(_out_dir):
            return {"tools": {"get_system_status": {"health": "healthy"}},
                    "processes": {"count": 2, "top_by_memory": []},
                    "ports": {"count": 3}, "har": {"requests": 1}}

        specs = [SurfaceSpec("desktop", "DESKTOP", kind="screen"),
                 SurfaceSpec("terminal", "TERMINAL", opener="terminal")]
        deps = DemoDeps(capture=fake_capture, close=lambda _w: None,
                        state=default_state, context=fake_context,
                        video=lambda *a, **k: {"ok": True, "path": str(root / "v.mp4")},
                        recorder_start=lambda p, f: {"stub": True},
                        recorder_stop=lambda r: {"status": "skipped"})
        params = {"out_dir": str(root), "proposer": "blobs", "inspect": False,
                  "max_regions": 4, "concurrency": 2, "render_video": False}
        record = run_demo(specs, deps=deps, params=params)
        problems = []
        if record["schema"] != SCHEMA:
            problems.append("schema")
        if record["totals"]["surfaces"] != 2:
            problems.append("surface count")
        for surface in record["surfaces"]:
            if surface["status"] != "ok" or surface["elements"] <= 0:
                problems.append(f"{surface['name']}: no elements")
            if not Path(surface["annotated"]).is_file():
                problems.append(f"{surface['name']}: no annotated image")
            if "total_ms" not in (surface["timing"] or {}):
                problems.append(f"{surface['name']}: no timing")
        transcript_text = (root / "demo-transcript.md").read_text(encoding="utf-8")
        if "**desktop**" not in transcript_text:
            problems.append("transcript")
        for problem in problems:
            print(f"FAIL: {problem}")
        if problems:
            return 1
        print(f"OK: {record['totals']['elements']} elements across "
              f"{record['totals']['surfaces']} surfaces, offline")
        return 0


if __name__ == "__main__":
    sys.exit(main())
