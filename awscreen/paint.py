"""Deterministic pixel painting: a JSON paint manifest -> a PIL image / PNG.

This is the deterministic counterpart to the *inspection* half of awscreen. Where a
ScreenState reads pixels into elements, a **paint manifest** writes pixels back from a
declarative, replayable recipe. The same manifest always renders the same image, so an
agent, a test, or a reviewer can prove exactly which pixels were touched -- the
building block for "paint pixel by pixel, deterministically".

Manifest shape::

    {
      "width": 640, "height": 480,
      "background": "#101820",          # optional; defaults to transparent black
      "ops": [
        {"op": "clear",   "color": "#ffffff"},
        {"op": "rect",    "bbox": [10, 10, 110, 50], "fill": "#2563eb",
                          "outline": "#0b1220", "width": 2},
        {"op": "ellipse", "bbox": [200, 10, 300, 60], "fill": [255, 0, 0]},
        {"op": "line",    "points": [[0, 0], [639, 479]], "fill": "#00d4ff",
                          "width": 3},
        {"op": "polygon", "points": [[400, 200], [480, 240], [420, 300]],
                          "fill": "#00ff88"},
        {"op": "pixels",  "pixels": [[0, 0, "#ff0000"], [1, 0, [0, 255, 0]]]},
        {"op": "text",    "xy": [20, 400], "text": "SAVE", "fill": "#ffffff",
                          "size": 18}
      ]
    }

Coordinates are IMAGE pixels. ``bbox`` is ``[x1, y1, x2, y2]`` with ``x2``/``y2``
EXCLUSIVE, matching ``ScreenState`` element bboxes: a rect ``[10, 10, 110, 50]`` fills
columns 10..109 and rows 10..49, exactly 100x40 pixels.

Determinism caveat: ``rect``/``ellipse``/``line``/``polygon``/``pixels`` are byte-stable
across Pillow versions. ``text`` depends on the Pillow default bitmap font, so its exact
ink is stable for a given Pillow but not guaranteed across major font changes; prefer
geometry ops when pixel-exactness matters.

No file is written here: :func:`paint_b64` returns the PNG bytes in memory.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont

SCHEMA = "awscreen.paint/1"

#: A frame is a screen, not a planet: 8192 px per side is the ceiling.
MAX_SIDE = 8192
_DEFAULT_SIZE = (640, 480)

Color = Tuple[int, int, int, int]


class PaintError(ValueError):
    """The manifest is malformed or asks for an impossible frame."""


def _color(value: Any, default: Optional[Color] = None) -> Optional[Color]:
    """Parse ``#rgb``/``#rrggbb``/``#rrggbbaa``/name/[r,g,b(,a)] into RGBA."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if len(vals) == 3:
            vals.append(255)
        if len(vals) != 4:
            raise PaintError(f"colour must have 3 or 4 channels, got {value!r}")
        try:
            return tuple(max(0, min(255, int(v))) for v in vals)  # type: ignore[return-value]
        except (TypeError, ValueError) as exc:
            raise PaintError(f"invalid colour {value!r}: {exc}") from exc
    if isinstance(value, str):
        try:
            rgb = ImageColor.getrgb(value.strip())
        except ValueError as exc:
            raise PaintError(f"invalid colour {value!r}: {exc}") from exc
        if len(rgb) == 3:
            return (rgb[0], rgb[1], rgb[2], 255)
        return (rgb[0], rgb[1], rgb[2], rgb[3])
    raise PaintError(f"unsupported colour type {type(value).__name__}")


def _int(value: Any, name: str) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise PaintError(f"{name} must be a number, got {value!r}") from exc


def _bbox(value: Any, name: str = "bbox") -> Tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise PaintError(f"{name} must be [x1, y1, x2, y2], got {value!r}")
    x1, y1, x2, y2 = (_int(v, name) for v in value)
    if x2 < x1 or y2 < y1:
        raise PaintError(f"{name} has an inverted corner: {value!r}")
    return (x1, y1, x2, y2)


def _points(value: Any, name: str = "points") -> List[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise PaintError(f"{name} needs at least two [x, y] points, got {value!r}")
    out: List[Tuple[int, int]] = []
    for pt in value:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise PaintError(f"{name} entry must be [x, y], got {pt!r}")
        out.append((_int(pt[0], name), _int(pt[1], name)))
    return out


def _draw_rect(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int],
               fill: Optional[Color], outline: Optional[Color], width: int) -> None:
    # x2/y2 are EXCLUSIVE: inset by one px so the fill ends before the corner.
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=fill, outline=outline, width=width)


def _apply(image: Image.Image, draw: ImageDraw.ImageDraw, op: Dict[str, Any]) -> None:
    kind = str(op.get("op") or op.get("op_type") or "").strip().lower()
    if not kind:
        raise PaintError("every op needs an 'op' kind")

    if kind == "clear":
        fill = _color(op.get("color", op.get("fill")))
        if fill is not None:
            draw.rectangle((0, 0, image.width - 1, image.height - 1), fill=fill)
        return

    if kind in ("rect", "rectangle"):
        _draw_rect(draw, _bbox(op.get("bbox")), _color(op.get("fill")),
                   _color(op.get("outline")),
                   max(1, _int(op.get("width", 1), "width")))
        return

    if kind in ("ellipse", "circle"):
        x1, y1, x2, y2 = _bbox(op.get("bbox"))
        if x2 > x1 and y2 > y1:
            draw.ellipse((x1, y1, x2 - 1, y2 - 1), fill=_color(op.get("fill")),
                         outline=_color(op.get("outline")),
                         width=max(1, _int(op.get("width", 1), "width")))
        return

    if kind == "line":
        draw.line(_points(op.get("points")), fill=_color(op.get("fill", op.get("color"))),
                  width=max(1, _int(op.get("width", 1), "width")))
        return

    if kind == "polygon":
        draw.polygon(_points(op.get("points")), fill=_color(op.get("fill")),
                     outline=_color(op.get("outline")))
        return

    if kind in ("pixel", "pixels"):
        _apply_pixels(draw, op)
        return

    if kind == "text":
        xy = op.get("xy") or op.get("position")
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            raise PaintError("text op needs 'xy': [x, y]")
        text = str(op.get("text") or "")
        if not text:
            return
        fill = _color(op.get("fill", op.get("color")))
        size = max(1, _int(op.get("size", 12), "size"))
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1: fixed-size bitmap font
            font = ImageFont.load_default()
        draw.text((_int(xy[0], "xy"), _int(xy[1], "xy")), text, fill=fill, font=font)
        return

    raise PaintError(f"unknown op {kind!r}")


def _apply_pixels(draw: ImageDraw.ImageDraw, op: Dict[str, Any]) -> None:
    """Set exact pixels. Accepts [[x, y, color], ...] or points + one fill."""
    raw = op.get("pixels")
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                raise PaintError(f"pixel entry must be [x, y, colour], got {entry!r}")
            channel = entry[2]
            color_value = channel if isinstance(channel, (str, list, tuple)) \
                else list(entry[2:])
            draw.point((_int(entry[0], "pixels"), _int(entry[1], "pixels")),
                       fill=_color(color_value))
        return
    points = op.get("points")
    fill = _color(op.get("fill", op.get("color")))
    if not isinstance(points, (list, tuple)) or not points or fill is None:
        raise PaintError("pixels op needs 'pixels' or 'points' plus 'fill'")
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise PaintError(f"points entry must be [x, y], got {pt!r}")
        draw.point((_int(pt[0], "points"), _int(pt[1], "points")), fill=fill)


def render_paint(manifest: Dict[str, Any]) -> Image.Image:
    """Render a paint manifest to an RGBA PIL image. Raises :class:`PaintError`."""
    if not isinstance(manifest, dict):
        raise PaintError("a paint manifest must be a JSON object")
    width = _int(manifest.get("width", _DEFAULT_SIZE[0]), "width")
    height = _int(manifest.get("height", _DEFAULT_SIZE[1]), "height")
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise PaintError(f"frame {width}x{height} outside 1..{MAX_SIDE}")
    background = _color(manifest.get("background"), default=(0, 0, 0, 0))
    image = Image.new("RGBA", (width, height), background)
    draw = ImageDraw.Draw(image)

    ops = manifest.get("ops")
    if ops is None:
        ops = []
    if not isinstance(ops, (list, tuple)):
        raise PaintError("'ops' must be a list")
    for index, op in enumerate(ops):
        if not isinstance(op, dict):
            raise PaintError(f"op {index} is not an object")
        try:
            _apply(image, draw, op)
        except PaintError as exc:
            raise PaintError(f"op {index}: {exc}") from exc
    return image


def paint_png_bytes(manifest: Dict[str, Any]) -> bytes:
    """Render then encode PNG. No temp file is ever written."""
    image = render_paint(manifest)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def paint_b64(manifest: Dict[str, Any]) -> str:
    """The rendered PNG as base64 (no data: prefix)."""
    return base64.b64encode(paint_png_bytes(manifest)).decode("ascii")


def manifest_sha256(manifest: Dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON, so equal manifests hash equal regardless of order."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def describe(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """A cheap summary (no render): size, op count and kinds, canonical hash."""
    if not isinstance(manifest, dict):
        raise PaintError("a paint manifest must be a JSON object")
    ops = manifest.get("ops") or []
    kinds: List[str] = [str(op.get("op") or "?") for op in ops if isinstance(op, dict)]
    return {"schema": SCHEMA,
            "width": _int(manifest.get("width", _DEFAULT_SIZE[0]), "width"),
            "height": _int(manifest.get("height", _DEFAULT_SIZE[1]), "height"),
            "ops": len(ops), "kinds": kinds,
            "manifest_sha256": manifest_sha256(manifest)}
