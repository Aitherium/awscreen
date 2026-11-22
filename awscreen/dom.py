"""Join DOM rects (from a browser) onto pixel elements.

A snapshot is ``{viewport: {dpr, vw, vh, sx, sy}, origin_px: [ox, oy],
image_scale?: float | "fit", elements: [{ref|selector, tag, role, label,
rect: [x, y, w, h] in CSS viewport px, page?: bool}]}``.

CSS -> image px: ``origin + (rect - scroll if page coords) * scale`` where scale is
``image_scale`` when numeric, ``image_w / vw`` when ``"fit"`` (a resized screenshot), else
``dpr``. ``AITHERBROWSER_RECT_JS`` returns exactly this shape from an ``eval`` in the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from .merge import iou
from .state import INTERACTIVE_ROLES, ROLES, Element, ScreenState

_TAG_ROLE = {"button": "button", "a": "link", "input": "input", "textarea": "input",
             "select": "input", "img": "image", "svg": "icon"}


@dataclass
class DomSnapshot:
    dpr: float = 1.0
    vw: float = 0.0
    vh: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    origin_px: Sequence[float] = (0, 0)
    image_scale: Optional[Union[float, str]] = None
    elements: List[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "DomSnapshot":
        if not isinstance(d, dict):
            raise ValueError("DOM snapshot must be an object")
        vp = d.get("viewport") or {}
        scale = d.get("image_scale")
        if scale is not None and scale != "fit":
            scale = float(scale)
        els = [e for e in d.get("elements") or [] if isinstance(e, dict)
               and isinstance(e.get("rect"), (list, tuple)) and len(e["rect"]) == 4]
        return cls(dpr=float(vp.get("dpr") or 1.0), vw=float(vp.get("vw") or 0),
                   vh=float(vp.get("vh") or 0), sx=float(vp.get("sx") or 0),
                   sy=float(vp.get("sy") or 0),
                   origin_px=tuple(d.get("origin_px") or (0, 0)), image_scale=scale,
                   elements=els)

    def scale_for(self, image_w: int) -> float:
        if isinstance(self.image_scale, (int, float)):
            return float(self.image_scale)
        if self.image_scale == "fit":
            if not self.vw:
                raise ValueError("image_scale 'fit' needs viewport.vw")
            return image_w / self.vw
        return self.dpr

    def css_to_image(self, rect: Sequence[float], image_w: int, page: bool = False) -> List[int]:
        x, y, w, h = (float(v) for v in rect)
        if page:
            x, y = x - self.sx, y - self.sy
        s = self.scale_for(image_w)
        ox, oy = float(self.origin_px[0]), float(self.origin_px[1])
        return [int(round(ox + x * s)), int(round(oy + y * s)),
                int(round(ox + (x + w) * s)), int(round(oy + (y + h) * s))]


def _contains(b: Sequence[float], pt) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def join_dom(state: ScreenState, snap: DomSnapshot, iou_thresh: float = 0.3) -> ScreenState:
    img_w, img_h = state.frame.width, state.frame.height
    ox, oy = state.frame.origin
    for node in snap.elements:
        box = snap.css_to_image(node["rect"], img_w, page=bool(node.get("page")))
        box = [max(0, box[0]), max(0, box[1]), min(img_w, box[2]), min(img_h, box[3])]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue                                      # off-screen node
        ref = {"ref": node.get("ref") or node.get("selector") or "", "tag": node.get("tag"),
               "role": node.get("role"), "label": node.get("label"), "bbox_image": box}
        role = str(node.get("role") or _TAG_ROLE.get(str(node.get("tag") or "").lower(),
                                                      "other")).lower()
        role = role if role in ROLES else "other"
        best, best_v = None, 0.0
        centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        for e in state.elements:
            if e.status != "ok":
                continue
            v = iou(e.bbox_image, box)
            if v < iou_thresh and _contains(e.bbox_image, centre) \
                    and _contains(box, ((e.bbox_image[0] + e.bbox_image[2]) / 2.0,
                                        (e.bbox_image[1] + e.bbox_image[3]) / 2.0)):
                v = max(v, iou_thresh)                  # mutual centre containment
            if v > best_v:
                best, best_v = e, v
        ref["iou"] = round(best_v, 3)
        if best is not None and best_v >= iou_thresh:
            best.evidence["bbox_pixels"] = list(best.bbox_image)
            best.bbox_image = list(box)                 # dom outranks pixel geometry
            best.bbox = [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]
            best.dom = ref
            if "dom" not in best.sources:
                best.sources.append("dom")
            best.source = "dom"
            best.label = best.label or str(node.get("label") or "")
            if best.role in ("other", "none", "") or role in INTERACTIVE_ROLES:
                best.role = role if role != "other" else best.role
            best.interactable = best.interactable or best.role in INTERACTIVE_ROLES
            best.confidence = max(best.confidence, 0.9)
        else:
            state.elements.append(Element(
                id="", region_id=None, bbox=[box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy],
                bbox_image=list(box), label=str(node.get("label") or ""), text="",
                role=role, confidence=0.9, source="dom", sources=["dom"],
                interactable=role in INTERACTIVE_ROLES, dom=ref))
    return state


#: Paste into AitherBrowser ``act: eval`` (or any page eval): returns a DomSnapshot dict.
AITHERBROWSER_RECT_JS = r"""(() => {
  const sel = 'a,button,input,textarea,select,[role],[onclick],[tabindex],img,svg,label,h1,h2,h3';
  const out = [];
  const nodes = document.querySelectorAll(sel);
  for (let i = 0; i < nodes.length && out.length < 400; i++) {
    const n = nodes[i];
    const r = n.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    if (r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
    const label = (n.getAttribute('aria-label') || n.innerText || n.value ||
                   n.getAttribute('alt') || n.getAttribute('title') || '').trim().slice(0, 80);
    out.push({ref: n.id ? '#' + n.id : n.tagName.toLowerCase() + ':nth(' + i + ')',
              tag: n.tagName.toLowerCase(), role: n.getAttribute('role') || '',
              label, rect: [r.left, r.top, r.width, r.height], page: false});
  }
  return {viewport: {dpr: devicePixelRatio, vw: innerWidth, vh: innerHeight,
                     sx: scrollX, sy: scrollY},
          origin_px: [0, 0], image_scale: 'fit', elements: out};
})()"""
