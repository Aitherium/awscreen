"""Crop a region with a margin, downscale it for a small model, keep the exact way back."""

from __future__ import annotations

import base64
import io
from typing import Sequence, Tuple

from PIL import Image

from .state import CropTransform, Region


def make_crop(img: Image.Image, region, margin: int = 8, max_side: int = 448,
              origin: Sequence[int] = (0, 0)) -> Tuple[Image.Image, CropTransform]:
    """Crop ``region`` (a Region or an [x1,y1,x2,y2] box in image px) plus ``margin``,
    clamped to the image, downscaled so the longest side is at most ``max_side``."""
    box = region.bbox if isinstance(region, Region) else tuple(region)
    img_w, img_h = img.size
    x1 = max(0, min(img_w - 1, int(box[0]) - margin))
    y1 = max(0, min(img_h - 1, int(box[1]) - margin))
    x2 = max(x1 + 1, min(img_w, int(box[2]) + margin))
    y2 = max(y1 + 1, min(img_h, int(box[3]) + margin))
    crop = img.crop((x1, y1, x2, y2))
    w, h = crop.size
    scale = 1.0
    longest = max(w, h)
    if max_side and longest > max_side:
        scale = max_side / float(longest)
        crop = crop.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.LANCZOS)
    return crop, CropTransform(crop_box=(x1, y1, x2, y2), scale=scale,
                               origin=(int(origin[0]), int(origin[1])),
                               crop_size=(crop.width, crop.height))


def crop_b64(crop: Image.Image) -> str:
    """In-memory PNG as base64. No temp file ever holds a piece of the screen."""
    buf = io.BytesIO()
    (crop if crop.mode in ("RGB", "L") else crop.convert("RGB")).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
