"""A synthetic screen at known coordinates, for the self-test and the tests.

``draw_screen(scale=1)`` -> (PIL image, truth). 1280x800 at scale 1, background #f3f4f6;
truth boxes are image px [x1, y1, x2, y2] with x2/y2 exclusive. ``scale=2`` doubles
everything, as a DPR-2 screenshot of the same CSS layout would.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

BG = (243, 244, 246)


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:          # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def draw_screen(scale: int = 1) -> Tuple[Image.Image, Dict[str, List[int]]]:
    s = int(scale)
    img = Image.new("RGB", (1280 * s, 800 * s), BG)
    d = ImageDraw.Draw(img)
    f = _font(20 * s)

    def box(x1, y1, x2, y2):
        return [x1 * s, y1 * s, x2 * s, y2 * s]

    def rect(b, **kw):
        d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], **kw)

    def centred(b, text, fill):
        tb = d.textbbox((0, 0), text, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.text((b[0] + (b[2] - b[0] - tw) // 2 - tb[0], b[1] + (b[3] - b[1] - th) // 2 - tb[1]),
               text, font=f, fill=fill)

    save = box(100, 120, 260, 168)
    rect(save, fill=(37, 99, 235))
    centred(save, "SAVE", (255, 255, 255))
    cancel = box(900, 600, 1100, 650)
    rect(cancel, fill=(107, 114, 128))
    centred(cancel, "CANCEL", (255, 255, 255))
    field_ = box(300, 300, 700, 340)
    rect(field_, fill=(255, 255, 255), outline=(156, 163, 175), width=2 * s)
    org = (100 * s, 500 * s)
    d.text(org, "Hello world", font=f, fill=(17, 24, 39))
    tb = d.textbbox(org, "Hello world", font=f)
    # The measured ink: the exact pixels that differ from the background.
    ink = Image.new("L", img.size, 0)
    ImageDraw.Draw(ink).text(org, "Hello world", font=f, fill=255)
    bb = ink.getbbox() or tb
    truth = {"save": save, "cancel": cancel, "input": field_, "hello": list(bb)}
    return img, truth


def draw_blank(size=(800, 600)) -> Image.Image:
    return Image.new("RGB", size, BG)


# ── two monitors, in memory ─────────────────────────────────────────────────

#: The right-hand monitor's desktop origin in the fixture. It sits immediately right of
#: the 1280-wide primary, so a frame-local x of 0 is desktop x 1280.
RIGHT_MONITOR_ORIGIN = (1280, 0)


def draw_screen_right(size=(1600, 900)) -> Tuple[Image.Image, Dict[str, List[int]]]:
    """A second window on a right-hand monitor, at known image coordinates.

    Independent of :func:`draw_screen` so the two frames can be told apart: this one has
    no SAVE/CANCEL and a single wide ``place`` button, whose truth box is image px.
    """
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    f = _font(22)
    # Window chrome offset from the monitor edge, so the frame is visibly not the primary
    # one and the blobs proposer has background ink to reject.
    d.rectangle([40, 40, w - 40, h - 40], fill=(255, 255, 255),
                outline=(120, 128, 140), width=2)
    d.rectangle([40, 40, w - 40, 96], fill=(45, 55, 72))
    d.text((60, 54), "Checkout - second monitor", font=f, fill=(255, 255, 255))
    place = [300, 400, 520, 456]
    d.rectangle([place[0], place[1], place[2] - 1, place[3] - 1], fill=(22, 163, 74))
    d.text((place[0] + 34, place[1] + 14), "PLACE ORDER", font=f, fill=(255, 255, 255))
    total = [300, 500, 470, 540]
    d.rectangle([total[0], total[1], total[2] - 1, total[3] - 1], fill=(229, 231, 235))
    d.text((total[0] + 12, total[1] + 8), "Total: $42.00", font=f, fill=(17, 24, 39))
    return img, {"place": place, "total": total}


class _FakeShot:
    """The bits of an ``mss.screenshot.ScreenShot`` awscreen reads."""

    def __init__(self, img: Image.Image) -> None:
        self.size = (img.width, img.height)
        self.bgra = img.convert("RGB").tobytes("raw", "BGRX")


class FakeMss:
    """An in-memory ``mss`` whose monitors are synthetic frames.

    ``monitors[0]`` is the union (as real mss reports it); ``1`` and ``2`` are the
    physical monitors. ``grab(rect)`` returns the frame whose desktop rectangle matches,
    so a capture of monitor 2 really is monitor 2's pixels.
    """

    def __init__(self, frames) -> None:
        self._frames = {tuple(m.origin) + m.size: img for m, img, _t in frames}
        lefts = [m.left for m, _i, _t in frames]
        tops = [m.top for m, _i, _t in frames]
        rights = [m.left + m.width for m, _i, _t in frames]
        bottoms = [m.top + m.height for m, _i, _t in frames]
        union = {"left": min(lefts), "top": min(tops),
                 "width": max(rights) - min(lefts), "height": max(bottoms) - min(tops)}
        self.monitors = [union] + [
            {"left": m.left, "top": m.top, "width": m.width, "height": m.height}
            for m, _i, _t in frames]

    def __enter__(self) -> "FakeMss":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def grab(self, rect) -> _FakeShot:
        key = (int(rect["left"]), int(rect["top"]),
               int(rect["width"]), int(rect["height"]))
        if key not in self._frames:
            raise ValueError(f"no synthetic monitor at {key}")
        return _FakeShot(self._frames[key])


def draw_two_monitors():
    """Two in-memory monitors and a fake ``mss`` over them.

    Returns ``(frames, factory)`` where ``frames`` is
    ``[(Monitor, image, truth), ...]``: monitor 1 is 1280x800 at desktop (0,0) and
    monitor 2 is 1600x900 at ``RIGHT_MONITOR_ORIGIN`` (1280,0). ``factory()`` returns a
    fresh :class:`FakeMss`, matching the ``open_mss`` seam in ``awscreen.capture``.
    """
    from .capture import Monitor

    m1_img, m1_truth = draw_screen()
    m2_img, m2_truth = draw_screen_right()
    frames = [
        (Monitor(1, 0, 0, m1_img.width, m1_img.height), m1_img, m1_truth),
        (Monitor(2, RIGHT_MONITOR_ORIGIN[0], RIGHT_MONITOR_ORIGIN[1],
                 m2_img.width, m2_img.height), m2_img, m2_truth),
    ]
    return frames, (lambda: FakeMss(frames))
#: The ten strings drawn by :func:`draw_mixed_screen`, in drawing order.
KNOWN_TEXT = (
    "SAVE",
    "CANCEL",
    "Invoice #1042",
    "Continue",
    "Search",
    "Username",
    "Password",
    "Total: $42.00",
    "Delete",
    "Settings",
)


def _text_ink_bbox(size, org, text, font):
    """The exact ink box of ``text`` drawn at ``org`` (the pixels OCR would box)."""
    ink = Image.new("L", size, 0)
    ImageDraw.Draw(ink).text(org, text, font=font, fill=255)
    return ink.getbbox()


def draw_mixed_screen() -> Tuple[Image.Image, Dict[str, Dict[str, List[int]]]]:
    """1280x800 with ten labelled controls; truth has control and text ink boxes.

    ``truth[name]`` = ``{"bbox": control box, "text": exact string, "text_bbox": ink}``.
    All boxes are image px with x2/y2 exclusive, so an OCR line inside a control can be
    matched back to it exactly.
    """
    img = Image.new("RGB", (1280, 800), BG)
    d = ImageDraw.Draw(img)
    f = _font(20)
    controls = [
        ("save", "SAVE", (40, 40, 200, 90), (37, 99, 235)),
        ("cancel", "CANCEL", (230, 40, 400, 90), (107, 114, 128)),
        ("invoice", "Invoice #1042", (40, 120, 300, 170), (16, 122, 87)),
        ("continue", "Continue", (40, 200, 240, 250), (124, 58, 237)),
        ("search", "Search", (40, 280, 240, 330), (2, 132, 199)),
        ("username", "Username", (40, 360, 280, 410), (180, 83, 9)),
        ("password", "Password", (40, 440, 280, 490), (190, 24, 93)),
        ("total", "Total: $42.00", (40, 520, 320, 570), (55, 65, 81)),
        ("delete", "Delete", (620, 40, 800, 90), (220, 38, 38)),
        ("settings", "Settings", (620, 120, 820, 170), (21, 128, 61)),
    ]
    truth: Dict[str, Dict[str, List[int]]] = {}
    for name, text, box, fill in controls:
        b = list(box)
        d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], fill=fill)
        tb = d.textbbox((0, 0), text, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        org = (b[0] + (b[2] - b[0] - tw) // 2 - tb[0],
               b[1] + (b[3] - b[1] - th) // 2 - tb[1])
        d.text(org, text, font=f, fill=(255, 255, 255))
        ink = _text_ink_bbox(img.size, org, text, f) or tb
        truth[name] = {"bbox": b, "text": text, "text_bbox": list(ink)}
    return img, truth
