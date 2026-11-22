"""Cheap pixel 'neurons': PIL arithmetic run on every crop before any model call.

They decide which crops are worth a vision call at all (a blank panel is not), and give
every element a heuristic role when no model looks at it. ``template_match`` uses numpy
when it is importable and says ``available: False`` otherwise; it never fakes a match.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image, ImageFilter, ImageStat


def dominant_rgb(img: Image.Image) -> Tuple[int, int, int]:
    small = img.convert("RGB")
    small.thumbnail((64, 64), Image.BILINEAR)
    q = small.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
    counts = q.getcolors() or [(1, 0)]
    idx = max(counts, key=lambda c: c[0])[1]
    pal = q.getpalette() or [0, 0, 0]
    return tuple(pal[idx * 3: idx * 3 + 3])  # type: ignore[return-value]


def run_neurons(crop: Image.Image, template: Optional[Image.Image] = None) -> dict:
    rgb = crop.convert("RGB")
    w, h = rgb.size
    grey = rgb.convert("L")
    st = ImageStat.Stat(grey)
    edges = grey.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 32 else 0)
    if w > 2 and h > 2:
        edges = edges.crop((1, 1, w - 1, h - 1))   # the filter's border is not content
    dom = dominant_rgb(rgb)
    dom_l = int(0.299 * dom[0] + 0.587 * dom[1] + 0.114 * dom[2])
    ink = grey.point(lambda v: 255 if abs(v - dom_l) > 40 else 0)
    feats = {
        "w": w, "h": h,
        "dominant_rgb": list(dom),
        "edge_density": round(ImageStat.Stat(edges).mean[0] / 255.0, 4),
        "ink_ratio": round(ImageStat.Stat(ink).mean[0] / 255.0, 4),
        "lum_mean": round(st.mean[0], 2),
        "lum_std": round(st.stddev[0], 2),
        "aspect": round(w / float(h or 1), 3),
    }
    if template is not None:
        feats["template"] = template_match(rgb, template)
    return feats


def template_match(haystack: Image.Image, needle: Image.Image, coarse: int = 4) -> dict:
    """Locate ``needle`` inside ``haystack`` by grey SSD, coarse-to-fine.

    Returns ``{available, x, y, score}`` (score = mean abs grey error per pixel, lower is
    better) or ``{available: False, reason}`` when numpy is not importable.
    """
    try:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError:
        return {"available": False, "reason": "unavailable: numpy not importable"}
    hay = np.asarray(haystack.convert("L"), dtype=np.float32)
    ndl = np.asarray(needle.convert("L"), dtype=np.float32)
    nh, nw = ndl.shape
    if nh > hay.shape[0] or nw > hay.shape[1]:
        return {"available": True, "x": None, "y": None, "score": None,
                "reason": "needle larger than haystack"}

    def best(hay, nd, xs=None, ys=None):
        win = sliding_window_view(hay, nd.shape)
        if ys is not None:
            win = win[ys[0]:ys[1], xs[0]:xs[1]]
        err = np.abs(win - nd).mean(axis=(2, 3))
        iy, ix = np.unravel_index(int(np.argmin(err)), err.shape)
        return int(ix), int(iy), float(err[iy, ix])

    c = max(1, int(coarse))
    if c > 1 and nh // c >= 4 and nw // c >= 4:
        hs = np.asarray(haystack.convert("L").resize(
            (hay.shape[1] // c, hay.shape[0] // c), Image.BOX), dtype=np.float32)
        ns = np.asarray(needle.convert("L").resize((nw // c, nh // c), Image.BOX),
                        dtype=np.float32)
        cx, cy, _ = best(hs, ns)
        x0, y0 = max(0, cx * c - 2 * c), max(0, cy * c - 2 * c)
        x1 = min(hay.shape[1] - nw + 1, cx * c + 2 * c + 1)
        y1 = min(hay.shape[0] - nh + 1, cy * c + 2 * c + 1)
        fx, fy, score = best(hay, ndl, xs=(x0, x1), ys=(y0, y1))
        return {"available": True, "x": x0 + fx, "y": y0 + fy, "score": round(score, 3)}
    x, y, score = best(hay, ndl)
    return {"available": True, "x": x, "y": y, "score": round(score, 3)}


def worth_asking(features: dict) -> Tuple[bool, str]:
    """Should this crop cost a model call? Blank and already-decided crops should not."""
    if features.get("edge_density", 0.0) < 0.003 and features.get("lum_std", 0.0) < 4.0:
        return False, "blank: no edges, flat luminance"
    if features.get("ink_ratio", 0.0) < 0.002 and features.get("edge_density", 0.0) < 0.01:
        return False, "no ink"
    role, conf = guess_role(features)
    if conf >= 0.8:
        return False, f"heuristic decided role={role} ({conf})"
    return True, "inked crop"


def guess_role(features: dict) -> Tuple[str, float]:
    """A heuristic role from the neurons alone. Below 0.8 means 'ask a model'."""
    ed = features.get("edge_density", 0.0)
    std = features.get("lum_std", 0.0)
    if ed < 0.003 and std < 4.0:
        return "none", 0.9
    w, h = features.get("w", 0), features.get("h", 0)
    aspect = features.get("aspect", 1.0)
    if w <= 64 and h <= 64 and 0.6 <= aspect <= 1.6:
        return "icon", 0.45
    if 18 <= h <= 90 and 1.5 <= aspect <= 12 and features.get("ink_ratio", 0) > 0.25:
        return "button", 0.5
    if aspect >= 3 and h <= 60:
        return "text", 0.5
    return "other", 0.3
