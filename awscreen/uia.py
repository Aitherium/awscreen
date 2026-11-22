"""Join Windows UI Automation (UIA) semantics onto pixel elements.

A Windows window gives us pixels only. UIA gives real names, control types and enabled
state for the focused window's controls, in the same desktop pixels as the screenshot,
so a menu item stays a menu item instead of being guessed as a label.

Snapshot shape (``awscreen.uia/1``)::

    {"schema": "awscreen.uia/1",
     "source": "uiautomation" | "powershell" | "fixture",
     "image_scale": 1.0,
     "window": {"title": "Untitled - Notepad", "process_id": 1234, "class": "Notepad"},
     "nodes": [{"name": "Save", "control_type": "MenuItem",
                "rect": [100, 120, 160, 48], "is_enabled": true,
                "automation_id": "", "depth": 2}]}

``rect`` is ``[x, y, w, h]`` in DESKTOP screen pixels (UIA's ``BoundingRectangle``), the
same coordinate space as ``Element.bbox``; ``image_scale`` maps screen px to image px
when the screenshot was resized (default 1.0).

Live tree, in order: the ``uiautomation`` package when importable, else PowerShell's
``UIAutomationClient`` through a documented subprocess created with ``CREATE_NO_WINDOW``
(never flash a console). Off Windows, or when neither bridge is present,
``walk_focused_window`` raises :class:`UiaUnavailableError` with the reason. Nothing is
invented and no element is added.

CI never needs Windows: a recorded tree JSON goes through ``UiaSnapshot.from_dict`` and
the same join path the live bridge feeds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .merge import iou
from .state import INTERACTIVE_ROLES, Element, ScreenState

SCHEMA = "awscreen.uia/1"
MAX_NODES = 2000
MAX_DEPTH = 30

#: Container chrome carries no element semantics of its own; a full-window rect added as
#: an "element" would be noise. These control types are skipped by the join (honestly
#: counted as skipped), never invented into a target.
STRUCTURAL_CONTROL_TYPES = frozenset({
    "window", "pane", "group", "custom", "titlebar", "scrollbar", "separator",
    "toolbar", "menubar", "statusbar", "list", "tree", "table", "datagrid",
})

#: UIA ControlType (normalised: no ``ControlType.`` prefix, lower case, no separators)
#: -> the ScreenState role. A control type missing here is ``other``, never a guess.
UIA_ROLE = {
    "button": "button", "splitbutton": "button",
    "edit": "input", "document": "input", "spinner": "input",
    "hyperlink": "link",
    "text": "text",
    "image": "image",
    "icon": "icon",
    "menuitem": "menuitem",
    "checkbox": "checkbox",
    "radiobutton": "radio",
    "combobox": "combobox",
    "tabitem": "tab",
    "slider": "slider",
    "listitem": "listitem", "treeitem": "listitem", "dataitem": "listitem",
}


def _clean_name(value: str) -> str:
    """Collapse control characters and runs of whitespace in a UIA name.

    Measured on a real Windows tree: PowerShell 5.1's ``ConvertTo-Json`` leaves some
    raw control characters inside strings, which strict JSON rejects and which would
    carry a newline into a label.
    """
    s = "".join(ch if ch >= " " else " " for ch in str(value or ""))
    return " ".join(s.split())


def _norm_control_type(control_type: str) -> str:
    return "".join(ch for ch in str(control_type or "").lower()
                   if ch not in " ._-").replace("controltype", "", 1)


def role_for_control_type(control_type: str) -> str:
    """Map a UIA control type to a ScreenState role; unknown -> ``other``."""
    return UIA_ROLE.get(_norm_control_type(control_type), "other")


class UiaUnavailableError(RuntimeError):
    """No UIA bridge is usable here; the message is the printed skip reason."""


@dataclass
class UiaSnapshot:
    """A recorded or live UIA tree: the focused window plus its control nodes."""

    nodes: List[dict] = field(default_factory=list)
    window: dict = field(default_factory=dict)
    source: str = "fixture"
    image_scale: float = 1.0
    schema: str = SCHEMA

    @classmethod
    def from_dict(cls, d: dict) -> "UiaSnapshot":
        if isinstance(d, (list, tuple)):
            d = {"nodes": list(d)}
        if not isinstance(d, dict):
            raise ValueError("UIA snapshot must be an object or a node list")
        raw_nodes = d.get("nodes") or []
        if not isinstance(raw_nodes, (list, tuple)):
            raise ValueError("UIA snapshot 'nodes' must be a list")
        try:
            scale = float(d.get("image_scale") or 1.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"UIA 'image_scale' must be a number: {exc}") from exc
        if scale <= 0:
            raise ValueError("UIA 'image_scale' must be positive")
        nodes: List[dict] = []
        for n in raw_nodes:
            if not isinstance(n, dict):
                continue
            rect = n.get("rect")
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                continue
            try:
                rect = [int(round(float(v))) for v in rect]
            except (TypeError, ValueError):
                continue
            nodes.append({
                "name": _clean_name(n.get("name")),
                "control_type": str(n.get("control_type") or "").strip(),
                "rect": rect,
                "is_enabled": bool(n.get("is_enabled", True)),
                "automation_id": str(n.get("automation_id") or "").strip(),
                "depth": int(n.get("depth") or 0),
            })
        window = dict(d.get("window") or {})
        if "title" in window:
            window["title"] = _clean_name(window["title"])
        return cls(nodes=nodes, window=window,
                   source=str(d.get("source") or "fixture"), image_scale=scale)

    @classmethod
    def load(cls, path: str) -> "UiaSnapshot":
        with open(path, "r", encoding="utf-8-sig") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "source": self.source, "image_scale": self.image_scale,
                "window": dict(self.window), "nodes": [dict(n) for n in self.nodes]}


def _box_image(rect: Sequence[float], origin: Tuple[int, int], image_scale: float) -> List[int]:
    """UIA screen px ``[x, y, w, h]`` -> image px ``[x1, y1, x2, y2]``."""
    x, y, w, h = (float(v) for v in rect)
    ox, oy = origin
    s = float(image_scale or 1.0)
    return [int(round((x - ox) * s)), int(round((y - oy) * s)),
            int(round((x + w - ox) * s)), int(round((y + h - oy) * s))]


def _contains(b: Sequence[float], pt: Sequence[float]) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def join_uia(state: ScreenState, snap: UiaSnapshot, iou_thresh: float = 0.3) -> ScreenState:
    """Join UIA nodes onto ``state.elements`` the way :func:`awscreen.dom.join_dom` does.

    Geometry: when no browser DOM rect already matched, the UIA rect wins over the pixel
    box (it is the control's real bounds). Semantics: a non-``other`` UIA role always
    wins, so a ``MenuItem`` is reported as ``menuitem`` rather than a label. A disabled
    control is never ``interactable``. Unmatched on-screen nodes are ADDED as elements
    with ``source='uia'``; an off-screen or zero-size node is skipped, never guessed.
    """
    img_w, img_h = state.frame.width, state.frame.height
    ox, oy = state.frame.origin
    matched = added = skipped = 0
    for node in snap.nodes:
        rect = node.get("rect") or []
        if _norm_control_type(node.get("control_type")) in STRUCTURAL_CONTROL_TYPES:
            skipped += 1
            continue
        if len(rect) != 4:
            skipped += 1
            continue
        box = _box_image(rect, (ox, oy), snap.image_scale)
        box = [max(0, box[0]), max(0, box[1]), min(img_w, box[2]), min(img_h, box[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            skipped += 1                                     # off-screen / zero-size node
            continue
        role = role_for_control_type(node.get("control_type"))
        enabled = bool(node.get("is_enabled", True))
        ref = {"name": str(node.get("name") or ""),
               "control_type": str(node.get("control_type") or ""),
               "automation_id": str(node.get("automation_id") or ""),
               "is_enabled": enabled, "bbox_image": list(box)}
        best, best_v = None, 0.0
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        for e in state.elements:
            if e.status != "ok":
                continue
            v = iou(e.bbox_image, box)
            if v < iou_thresh and _contains(e.bbox_image, centre) \
                    and _contains(box, ((e.bbox_image[0] + e.bbox_image[2]) / 2.0,
                                        (e.bbox_image[1] + e.bbox_image[3]) / 2.0)):
                v = max(v, iou_thresh)                      # mutual centre containment
            if v > best_v:
                best, best_v = e, v
        ref["iou"] = round(best_v, 3)
        if best is not None and best_v >= iou_thresh:
            matched += 1
            best.uia = ref
            if "uia" not in best.sources:
                best.sources.append("uia")
            if best.dom is None:                            # DOM CSS geometry already exact
                best.evidence["bbox_pixels"] = list(best.bbox_image)
                best.bbox_image = list(box)
                best.bbox = [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]
                best.source = "uia"
            if role != "other":
                best.role = role                            # UIA semantics outrank guesses
            best.label = best.label or ref["name"]
            if not enabled:
                best.interactable = False
            else:
                best.interactable = best.interactable or best.role in INTERACTIVE_ROLES
            best.confidence = max(best.confidence, 0.9)
        else:
            added += 1
            state.elements.append(Element(
                id="", region_id=None,
                bbox=[box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy],
                bbox_image=list(box), label=ref["name"], text="", role=role,
                confidence=0.9, source="uia", sources=["uia"],
                interactable=enabled and role in INTERACTIVE_ROLES, uia=ref))
    state.uia = {"attempted": True, "ok": True, "source": snap.source,
                 "window": dict(snap.window), "nodes": len(snap.nodes),
                 "matched": matched, "added": added, "skipped": skipped}
    return state


#: PowerShell 5.1 (preferred: UIAutomationClient is guaranteed there) UIA walker. It
#: prints ONE JSON object to stdout; diagnostics go to stderr and a non-zero exit. The
#: process is started with CREATE_NO_WINDOW so no console flashes.
POWERSHELL_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
  Add-Type -AssemblyName UIAutomationClient
  Add-Type -AssemblyName UIAutomationTypes
} catch {
  Write-Error "UIAutomationClient unavailable: $($_.Exception.Message)"
  exit 3
}
try {
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AitherDpi {
  [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr ctx);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}
"@
  [void][AitherDpi]::SetProcessDpiAwarenessContext([IntPtr](-4))
} catch { }
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
# The FOREGROUND window is "the focused window". FocusedElement alone can resolve to the
# desktop when the last focused control is an overlay that never takes focus, which makes
# the walk the whole desktop instead of one app.
$window = $null
try {
  $hwnd = [AitherDpi]::GetForegroundWindow()
  if ($hwnd -ne [IntPtr]::Zero) {
    $window = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
  }
} catch { $window = $null }
if ($window -eq $null) {
  $focused = [System.Windows.Automation.AutomationElement]::FocusedElement
  if ($focused -eq $null) { Write-Error "no focused window"; exit 4 }
  $window = $focused
  while ($true) {
    $parent = $walker.GetParent($window)
    if ($parent -eq $null) { break }
    $parentType = [string]$parent.Current.ControlType.ProgrammaticName
    $parentType = $parentType -replace '^ControlType\.', ''
    if ($parentType -eq 'Desktop') { break }
    $window = $parent
  }
}
$nodes = [System.Collections.ArrayList]::new()
function Walk-Uia($element, [int]$depth) {
  if ($depth -gt 30) { return }
  if ($nodes.Count -ge 2000) { return }
  try {
    $cur = $element.Current
    $r = $cur.BoundingRectangle
    $ct = ([string]$cur.ControlType.ProgrammaticName) -replace '^ControlType\.', ''
    $null = $nodes.Add([pscustomobject]@{
      name = [string]$cur.Name
      control_type = $ct
      automation_id = [string]$cur.AutomationId
      rect = @([int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height)
      is_enabled = [bool]$cur.IsEnabled
      depth = $depth
    })
  } catch { }
  try { $child = $walker.GetFirstChild($element) } catch { $child = $null }
  while ($child -ne $null) {
    Walk-Uia $child ($depth + 1)
    try { $child = $walker.GetNextSibling($child) } catch { $child = $null }
  }
}
Walk-Uia $window 0
$winCur = $window.Current
[pscustomobject]@{
  schema = 'awscreen.uia/1'
  source = 'powershell'
  image_scale = 1.0
  window = [pscustomobject]@{
    title = [string]$winCur.Name
    process_id = [int]$winCur.ProcessId
    class = [string]$winCur.ClassName
  }
  nodes = @($nodes)
} | ConvertTo-Json -Depth 6 -Compress
'''


def parse_bridge_stdout(text: str) -> UiaSnapshot:
    """Parse the PowerShell bridge's stdout into a snapshot; never returns an empty tree."""
    text = (text or "").strip().lstrip("﻿")
    if not text:
        raise ValueError("empty stdout")
    try:
        # strict=False tolerates raw control characters inside string values: measured
        # Windows PowerShell 5.1 ConvertTo-Json output contains them in element names.
        doc = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stdout is not JSON: {exc}") from exc
    snap = UiaSnapshot.from_dict(doc if isinstance(doc, dict) else {"nodes": doc})
    if not snap.nodes:
        raise ValueError("the bridge returned no nodes")
    if snap.source in ("", "fixture"):
        snap.source = "powershell"
    return snap


def _powershell_exe() -> Optional[str]:
    for name in ("powershell", "pwsh"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _walk_with_uiautomation(auto) -> UiaSnapshot:
    root = auto.GetForegroundControl()
    if root is None:
        raise UiaUnavailableError("uiautomation: no focused window (GetForegroundControl None)")
    try:
        win = root.GetTopLevelControl() or root
    except Exception:  # noqa: BLE001 - a control without a top level still has itself
        win = root
    nodes: List[dict] = []
    stack = [(win, 0)]
    while stack and len(nodes) < MAX_NODES:
        ctrl, depth = stack.pop()
        try:
            br = ctrl.BoundingRectangle
            nodes.append({
                "name": str(getattr(ctrl, "Name", "") or ""),
                "control_type": str(getattr(ctrl, "ControlTypeName", "") or ""),
                "automation_id": str(getattr(ctrl, "AutomationId", "") or ""),
                "rect": [int(br.left), int(br.top), int(br.right - br.left),
                         int(br.bottom - br.top)],
                "is_enabled": bool(getattr(ctrl, "IsEnabled", True)),
                "depth": depth,
            })
        except Exception:  # noqa: BLE001 - a stale control is skipped, not fatal
            continue
        if depth >= MAX_DEPTH:
            continue
        try:
            children = list(ctrl.GetChildren())
        except Exception:  # noqa: BLE001
            children = []
        for child in reversed(children):
            stack.append((child, depth + 1))
    if not nodes:
        raise UiaUnavailableError("uiautomation walked the focused window and found no nodes")
    try:
        win_cur = win
        window = {"title": str(getattr(win_cur, "Name", "") or ""),
                  "process_id": int(getattr(win_cur, "ProcessId", 0) or 0),
                  "class": str(getattr(win_cur, "ClassName", "") or "")}
    except Exception:  # noqa: BLE001
        window = {}
    return UiaSnapshot(nodes=nodes, window=window, source="uiautomation")


def _walk_with_powershell(timeout_s: float = 20.0) -> UiaSnapshot:
    exe = _powershell_exe()
    if not exe:
        raise UiaUnavailableError(
            "PowerShell UIAutomationClient bridge: no powershell/pwsh on PATH")
    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", POWERSHELL_SCRIPT],
            capture_output=True, timeout=timeout_s, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UiaUnavailableError(f"PowerShell UIAutomationClient bridge failed to run: {exc}") \
            from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[:400]
        raise UiaUnavailableError(f"PowerShell UIAutomationClient exited {proc.returncode}: {err}")
    try:
        return parse_bridge_stdout(proc.stdout.decode("utf-8-sig", "replace"))
    except ValueError as exc:
        raise UiaUnavailableError(f"PowerShell UIAutomationClient returned no usable tree: {exc}") \
            from exc


def bridge_available(platform: Optional[str] = None) -> Tuple[bool, str]:
    """Can a live UIA bridge run here? ``(ok, reason)``; reason is printable either way."""
    plat = platform or sys.platform
    if not plat.startswith("win"):
        return False, (f"UIA is Windows-only (platform={plat!r}); "
                       "a recorded tree still joins with --uia <file>")
    try:
        import uiautomation  # noqa: F401
        return True, "uiautomation package importable"
    except Exception as exc:  # noqa: BLE001 - any import failure falls to PowerShell
        uia_reason = f"uiautomation not importable ({type(exc).__name__}: {exc})"
    exe = _powershell_exe()
    if exe:
        return True, f"PowerShell UIAutomationClient via {exe} ({uia_reason})"
    return False, f"no UIA bridge: {uia_reason}; no powershell/pwsh on PATH"


def walk_focused_window(platform: Optional[str] = None,
                        timeout_s: float = 20.0) -> UiaSnapshot:
    """Read the focused window's UIA tree, or raise :class:`UiaUnavailableError` with why.

    ``uiautomation`` is tried first; PowerShell's ``UIAutomationClient`` is the
    documented fallback. Nothing is fabricated when both fail.
    """
    plat = platform or sys.platform
    if not plat.startswith("win"):
        raise UiaUnavailableError(
            f"UIA is Windows-only (platform={plat!r}); pass a recorded tree with --uia <file>")
    reasons: List[str] = []
    try:
        import uiautomation as auto
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"uiautomation not importable ({type(exc).__name__}: {exc})")
    else:
        try:
            return _walk_with_uiautomation(auto)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"uiautomation walk failed ({type(exc).__name__}: {exc})")
    try:
        return _walk_with_powershell(timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"PowerShell UIAutomationClient failed ({type(exc).__name__}: {exc})")
    raise UiaUnavailableError("no UIA bridge: " + "; ".join(reasons))
