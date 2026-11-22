"""Merge overlapping elements from several sources into one per thing on screen.

Geometry goes to the higher-priority source (dom > omniparser > vlm > ocr > cv > grid);
text and labels are unioned, confidence is the max, and every contributing source and
region id is kept. An element whose status is not ``ok`` (a timed-out or failed crop) is
never merged away: it stays an explicit element so a reader can see what was not seen.
"""

from __future__ import annotations

from typing import List, Sequence

from .state import INTERACTIVE_ROLES, Element

PRIORITY = {"uia": 7, "dom": 6, "omniparser": 5, "vlm": 4, "ocr": 3, "cv": 2,
            "neuron": 1, "grid": 0}


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _union_text(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a or b.lower() in a.lower():
        return a or b
    if a.lower() in b.lower():
        return b
    return f"{a} | {b}"


def absorb(keep: Element, other: Element) -> None:
    """Fold ``other`` into ``keep`` (``keep`` already has the geometry that wins)."""
    for s in other.sources or [other.source]:
        if s not in keep.sources:
            keep.sources.append(s)
    before = keep.text
    keep.text = _union_text(keep.text, other.text)
    if keep.text != before:
        # Record WHO read the text so an exact reading is distinguishable from a hint.
        # A union of two different readings keeps both readers, in priority order.
        if not before:
            keep.text_source = other.text_source or other.source
        elif other.text.strip():
            readers = [r for r in (keep.text_source, other.text_source or other.source) if r]
            keep.text_source = "+".join(dict.fromkeys(readers))
    if not keep.label or (PRIORITY.get(other.source, 0) > PRIORITY.get(keep.source, 0)
                          and other.label):
        keep.label = keep.label or other.label
    if keep.role in ("other", "none", "") and other.role not in ("other", "none", ""):
        keep.role = other.role
    keep.confidence = max(keep.confidence, other.confidence)
    keep.interactable = keep.interactable or other.interactable \
        or keep.role in INTERACTIVE_ROLES
    rids = keep.evidence.setdefault("region_ids", [keep.region_id] if keep.region_id else [])
    if other.region_id and other.region_id not in rids:
        rids.append(other.region_id)
    if other.source == "vlm" and other.evidence.get("vlm_raw"):
        keep.evidence["vlm_raw"] = other.evidence["vlm_raw"]
    if other.dom and not keep.dom:
        keep.dom = other.dom


def merge_elements(els: List[Element], iou_thresh: float = 0.5) -> List[Element]:
    ok = [e for e in els if e.status == "ok"]
    rest = [e for e in els if e.status != "ok"]
    ok.sort(key=lambda e: (PRIORITY.get(e.source, 0), e.confidence), reverse=True)
    kept: List[Element] = []
    for e in ok:
        best, best_iou = None, 0.0
        for k in kept:
            v = iou(k.bbox, e.bbox)
            if v > best_iou:
                best, best_iou = k, v
        if best is not None and best_iou >= iou_thresh:
            absorb(best, e)
        else:
            if e.source not in e.sources:
                e.sources.insert(0, e.source)
            kept.append(e)
    return kept + rest
