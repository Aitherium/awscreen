"""Deterministic element-level diff between two ScreenStates (or two frames).

The diff answers the question an agent actually asks after acting on a screen:
"what changed?" It matches elements one-to-one, then reports four buckets:

* **added**   -- in B, matched to nothing in A
* **removed** -- in A, matched to nothing in B
* **moved**   -- matched, but the bbox centre shifted more than ``move_px`` on an axis
* **unchanged** -- matched and (nearly) still

Matching is deterministic and has two passes:

1. **IoU** -- every A/B pair scored by intersection-over-union; greedy one-to-one
   descending by IoU, ties broken by equal label first, then element order.
2. **identity** -- leftovers that share a non-empty, case-insensitive label AND role
   are paired in bbox order. This is what turns a *far* jump into ``moved`` instead of
   one ``removed`` plus one ``added``: a button that hops across the screen has IoU 0.

When both source images are supplied, each entry also carries a ``pixel_delta``:
how many pixels (and what fraction) differ between the two images over that entry's
image-px box. Pixels are compared per channel with a tolerance, so a sub-threshold
anti-aliasing shimmer does not read as a change.

Pure arithmetic over PIL images and the ScreenState dataclasses -- no network, no
model call, no host capture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "awscreen.compare/1"

BBox = Tuple[float, float, float, float]


class CompareError(ValueError):
    """The two inputs cannot be compared (unreadable state, mismatched mode...)."""


def _get(el: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an Element dataclass or a plain dict."""
    if isinstance(el, dict):
        return el.get(key, default)
    return getattr(el, key, default)


def _bbox(el: Any) -> List[float]:
    box = _get(el, "bbox") or []
    return [float(v) for v in box[:4]] if len(box) >= 4 else [0.0, 0.0, 0.0, 0.0]


def _bbox_image(el: Any) -> List[float]:
    box = _get(el, "bbox_image") or []
    if len(box) >= 4:
        return [float(v) for v in box[:4]]
    return _bbox(el)


def _union(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _centre(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _label(el: Any) -> str:
    return str(_get(el, "label") or "").strip().lower()


def _role(el: Any) -> str:
    return str(_get(el, "role") or "").strip().lower()


def match_elements(a_elements: Sequence[Any], b_elements: Sequence[Any],
                   iou_threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    """Deterministic one-to-one matches as ``(a_index, b_index, iou)``.

    Pass 1 is greedy IoU. Pass 2 pairs leftover elements that share a non-empty
    label AND role, in bbox order (``iou`` is 0.0 for those far-identity pairs).
    """
    a_boxes = [_bbox(e) for e in a_elements]
    b_boxes = [_bbox(e) for e in b_elements]
    used_a: set = set()
    used_b: set = set()
    matches: List[Tuple[int, int, float]] = []

    candidates: List[Tuple[float, int, int, float]] = []
    for i, ab in enumerate(a_boxes):
        for j, bb in enumerate(b_boxes):
            iou = _iou(ab, bb)
            if iou < iou_threshold:
                continue
            same_label = 0 if (_label(a_elements[i]) and
                               _label(a_elements[i]) == _label(b_elements[j])) else 1
            candidates.append((-iou, same_label, i, j))
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    for neg_iou, _same, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, -neg_iou))

    # Pass 2: identity by (label, role), leftovers only, deterministic order.
    left_a = [i for i in range(len(a_elements)) if i not in used_a]
    left_b = [j for j in range(len(b_elements)) if j not in used_b]
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for j in left_b:
        label, role = _label(b_elements[j]), _role(b_elements[j])
        if not label:
            continue
        buckets.setdefault((label, role), []).append(j)
    for bucket in buckets.values():
        bucket.sort(key=lambda j: (b_boxes[j][1], b_boxes[j][0], j))
    for i in sorted(left_a, key=lambda i: (a_boxes[i][1], a_boxes[i][0], i)):
        label, role = _label(a_elements[i]), _role(a_elements[i])
        if not label:
            continue
        bucket = buckets.get((label, role)) or []
        if not bucket:
            continue
        j = bucket.pop(0)
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, 0.0))

    matches.sort(key=lambda m: (a_boxes[m[0]][1], a_boxes[m[0]][0], m[0], m[1]))
    return matches


def pixel_delta(image_a, image_b, box: Sequence[float], tolerance: int = 8) -> Dict[str, Any]:
    """Changed pixels between two PIL images over ``box`` (image px, exclusive).

    A pixel counts as changed when ANY channel differs by more than ``tolerance``.
    Returns a dict; a box that falls outside both images yields a zero total, never
    an error, so a diff of partial captures still reports.
    """
    from PIL import ImageChops

    clamped = _clamp_box(box, image_a.size, image_b.size)
    if clamped is None:
        return {"pixels_changed": 0, "pixels_total": 0, "fraction": 0.0}
    ca = image_a.crop(clamped).convert("RGB")
    cb = image_b.crop(clamped).convert("RGB")
    diff = ImageChops.difference(ca, cb)
    r, g, b = diff.split()
    mx = ImageChops.lighter(ImageChops.lighter(r, g), b)
    hist = mx.histogram()
    changed = sum(hist[tolerance + 1:])
    total = (clamped[2] - clamped[0]) * (clamped[3] - clamped[1])
    return {"pixels_changed": int(changed), "pixels_total": int(total),
            "fraction": round(changed / float(total), 6) if total else 0.0}


def _clamp_box(box: Sequence[float], size_a, size_b) -> Optional[Tuple[int, int, int, int]]:
    max_w = min(size_a[0], size_b[0])
    max_h = min(size_a[1], size_b[1])
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(max_w, int(round(box[2])))
    y2 = min(max_h, int(round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _el_dict(el: Any) -> Dict[str, Any]:
    if isinstance(el, dict):
        keep = ("id", "bbox", "bbox_image", "label", "text", "role", "confidence",
                "source", "sources", "interactable", "status")
        return {k: el.get(k) for k in keep if k in el}
    from dataclasses import asdict
    return asdict(el)


def _pair(a: Any, b: Any, iou: float, move_px: float, image_a, image_b) -> Dict[str, Any]:
    abox, bbox = _bbox(a), _bbox(b)
    ac, bc = _centre(abox), _centre(bbox)
    dx, dy = round(bc[0] - ac[0], 2), round(bc[1] - ac[1], 2)
    moved = abs(dx) > move_px or abs(dy) > move_px
    union_box = _union(_bbox_image(a), _bbox_image(b))
    delta = (pixel_delta(image_a, image_b, union_box)
             if image_a is not None and image_b is not None else None)
    return {
        "kind": "moved" if moved else "unchanged",
        "a_id": _get(a, "id"), "b_id": _get(b, "id"),
        "from_bbox": [int(round(v)) for v in abox],
        "to_bbox": [int(round(v)) for v in bbox],
        "dx": dx, "dy": dy, "iou": round(iou, 4),
        "label": _get(b, "label") or _get(a, "label") or "",
        "role": _get(b, "role") or _get(a, "role") or "",
        "pixel_delta": delta,
    }


def compare_states(state_a: Any, state_b: Any, *, image_a=None, image_b=None,
                   move_px: float = 3.0, iou_threshold: float = 0.3,
                   tolerance: int = 8) -> Dict[str, Any]:
    """Element-level diff of two ScreenStates (dataclasses or dicts of elements)."""
    a_els = list(_elements(state_a))
    b_els = list(_elements(state_b))
    matches = match_elements(a_els, b_els, iou_threshold=iou_threshold)

    matched_a = {i for i, _j, _iou_v in matches}
    matched_b = {j for _i, j, _iou_v in matches}

    moved: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []
    for i, j, iou in matches:
        pair = _pair(a_els[i], b_els[j], iou, move_px, image_a, image_b)
        (moved if pair["kind"] == "moved" else unchanged).append(pair)

    removed = [_entry_side(a_els[i], "removed", image_a, image_b, tolerance)
               for i in sorted(set(range(len(a_els))) - matched_a,
                               key=lambda i: (_bbox(a_els[i])[1], _bbox(a_els[i])[0], i))]
    added = [_entry_side(b_els[j], "added", image_a, image_b, tolerance)
             for j in sorted(set(range(len(b_els))) - matched_b,
                             key=lambda j: (_bbox(b_els[j])[1], _bbox(b_els[j])[0], j))]

    for bucket in (moved, unchanged):
        bucket.sort(key=lambda p: (p["to_bbox"][1], p["to_bbox"][0], p["a_id"] or ""))
    return {
        "ok": True,
        "schema": SCHEMA,
        "summary": {"added": len(added), "removed": len(removed),
                    "moved": len(moved), "unchanged": len(unchanged),
                    "pairs": len(matches)},
        "added": added, "removed": removed, "moved": moved, "unchanged": unchanged,
    }


def _elements(state: Any) -> Sequence[Any]:
    if isinstance(state, dict):
        els = state.get("elements")
    else:
        els = getattr(state, "elements", None)
    if not isinstance(els, (list, tuple)):
        raise CompareError("state carries no element list (expected 'elements': [...])")
    return els


def _entry_side(el: Any, kind: str, image_a, image_b, tolerance: int) -> Dict[str, Any]:
    box = _bbox_image(el)
    delta = (pixel_delta(image_a, image_b, box, tolerance=tolerance)
             if image_a is not None and image_b is not None else None)
    out = _el_dict(el)
    out["kind"] = kind
    out["pixel_delta"] = delta
    return out


def compare_images(path_a: str, path_b: str, *, proposer: str = "blobs",
                   inspect: bool = False, move_px: float = 3.0,
                   iou_threshold: float = 0.3, tolerance: int = 8,
                   max_regions: int = 48) -> Dict[str, Any]:
    """Build a (deterministic, neurons-only) ScreenState for each frame, then diff.

    ``proposer='blobs'`` is the pure-PIL proposer: no network, no model call. An
    explicit ``proposer='grid'`` is also offline. ``omniparser``/``ocr`` may reach a
    local service and are honoured when named.
    """
    from PIL import Image

    from .state import build_screen_state

    name = str(proposer or "blobs").strip().lower()
    explicit = name not in ("auto", "default", "")
    chain = (name,) if explicit else ("blobs", "grid")
    try:
        state_a = build_screen_state(
            path_a, proposers=chain, explicit_proposer=explicit,
            inspect=bool(inspect), max_regions=max_regions)
        state_b = build_screen_state(
            path_b, proposers=chain, explicit_proposer=explicit,
            inspect=bool(inspect), max_regions=max_regions)
    except Exception as exc:  # noqa: BLE001 - surfaced as a loud CompareError
        raise CompareError(f"could not read a frame: {type(exc).__name__}: {exc}") from exc

    with Image.open(path_a) as raw_a, Image.open(path_b) as raw_b:
        img_a = raw_a.convert("RGB")
        img_b = raw_b.convert("RGB")
        result = compare_states(state_a, state_b, image_a=img_a, image_b=img_b,
                                move_px=move_px, iou_threshold=iou_threshold,
                                tolerance=tolerance)
    result["frame_a"] = {"path": path_a, "width": img_a.width, "height": img_a.height}
    result["frame_b"] = {"path": path_b, "width": img_b.width, "height": img_b.height}
    return result
