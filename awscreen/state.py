"""ScreenState: what is on screen and where, as one structured record.

The pipeline is ``frame -> propose -> crop -> neurons -> inspect -> merge -> dom_join ->
uia_join``:

* **propose** cheap region proposals (OmniParser when it is up, else tesseract line boxes,
  else pure-PIL blobs, else an adaptive grid); every attempt is reported.
* **crop** each region with a margin and an exact transform back to screen pixels.
* **neurons** score every crop with PIL arithmetic; blank or already-decided crops never
  reach a model.
* **inspect** the rest IN PARALLEL with a small vision model, capped at 4 in flight with
  a per-crop deadline; a crop that times out becomes an explicit element, never dropped.
  An optional wall-clock **budget** stops the calls while a whole crop still fits and
  marks every element it left uninspected ``status='budget'``.
* **merge** overlapping elements by IoU with source priority, then join DOM rects when a
  browser snapshot exists, then UI Automation nodes on Windows so a menu item keeps its
  real control type instead of being read as a label.

Crops never leave this host except to the vision endpoint the awvision resolver pins.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .budget import BUDGET_STATUS, DEFAULT_CALL_ESTIMATE_MS, Budget

SCHEMA = "awscreen.screen_state/1"

#: Region proposer name -> the element `source` it yields.
PROPOSER_SOURCE = {"omniparser": "omniparser", "ocr": "ocr", "blobs": "cv", "grid": "grid"}

ROLES = ("button", "input", "text", "icon", "link", "image", "menuitem", "checkbox",
         "radio", "combobox", "tab", "slider", "listitem", "other", "none")
INTERACTIVE_ROLES = frozenset({"button", "input", "link", "icon", "menuitem", "checkbox",
                               "radio", "combobox", "tab", "slider"})


@dataclass
class Frame:
    path: str
    width: int
    height: int
    origin: Tuple[int, int] = (0, 0)   # where image px (0,0) sits on the desktop
    capture: str = "file"              # 'file' | 'primary' | 'monitor:<N>'
    signature: Optional[list] = None   # awvision.sight.frame_signature (32x32 grey)
    sha256: str = ""


@dataclass
class Region:
    id: str
    bbox: Tuple[int, int, int, int]    # IMAGE px [x1, y1, x2, y2], x2/y2 exclusive
    proposer: str
    hint: dict = field(default_factory=dict)
    #: ``{hash, thumb}`` captured from the frame this region came from: a cheap
    #: native-resolution hash plus a 16x16 grey thumbnail. The NEXT frame diffs a crop
    #: of the same box against it to decide whether the region changed at all.
    sig: Optional[dict] = None


@dataclass
class CropTransform:
    """Maps crop pixels back to image and screen pixels, exactly.

    ``scale`` is the nominal downscale of the LONGEST side. The real resize rounds each
    axis independently (``round(w*scale)`` x ``round(h*scale)``), so on a flat crop the
    short axis has a different factor; ``crop_size`` records the actual output size and
    ``scale_xy`` uses it, so ``to_image`` lands on the true pixel instead of drifting
    by up to several px on a 3840-wide, 20-px-tall region.
    """

    crop_box: Tuple[int, int, int, int]  # image px, margin applied, clamped
    scale: float                         # nominal crop px / image px (longest side)
    origin: Tuple[int, int] = (0, 0)
    crop_size: Tuple[int, int] = (0, 0)  # actual crop w,h AFTER resize (0,0 = unknown)

    def scale_xy(self) -> Tuple[float, float]:
        w = self.crop_box[2] - self.crop_box[0]
        h = self.crop_box[3] - self.crop_box[1]
        if self.crop_size and self.crop_size[0] and self.crop_size[1] and w > 0 and h > 0:
            return (self.crop_size[0] / float(w), self.crop_size[1] / float(h))
        return (self.scale, self.scale)

    def to_image(self, x: float, y: float) -> Tuple[float, float]:
        sx, sy = self.scale_xy()
        return (self.crop_box[0] + x / sx, self.crop_box[1] + y / sy)

    def to_screen(self, x: float, y: float) -> Tuple[float, float]:
        ix, iy = self.to_image(x, y)
        return (ix + self.origin[0], iy + self.origin[1])

    def bbox_to_screen(self, b: Sequence[float]) -> List[int]:
        x1, y1 = self.to_screen(b[0], b[1])
        x2, y2 = self.to_screen(b[2], b[3])
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    def from_screen(self, x: float, y: float) -> Tuple[float, float]:
        sx, sy = self.scale_xy()
        return ((x - self.origin[0] - self.crop_box[0]) * sx,
                (y - self.origin[1] - self.crop_box[1]) * sy)


@dataclass
class Element:
    id: str
    region_id: Optional[str]
    bbox: List[int]                      # ABSOLUTE screen px [x1, y1, x2, y2]
    bbox_image: List[int]
    label: str = ""
    text: str = ""
    text_source: str = ""                # who read `text`: tesseract-tsv|vlm-ocr|...
    role: str = "other"
    confidence: float = 0.0
    source: str = "cv"                   # omniparser|ocr|cv|grid|vlm|dom|neuron
    sources: List[str] = field(default_factory=list)
    interactable: bool = False
    status: str = "ok"                   # ok|timeout|error|skipped|budget
    evidence: dict = field(default_factory=dict)
    dom: Optional[dict] = None
    uia: Optional[dict] = None

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)


@dataclass
class ScreenState:
    frame: Frame
    proposer: dict = field(default_factory=dict)
    regions: List[Region] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    uia: dict = field(default_factory=dict)   # UI Automation join report (or skip reason)
    reused: bool = False                 # True only when the WHOLE frame was reused
    reused_regions: int = 0              # unchanged regions carried over from --prev
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScreenState":
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} record")
        fr = dict(d["frame"])
        fr["origin"] = tuple(fr.get("origin") or (0, 0))
        regions = [Region(id=r["id"], bbox=tuple(r["bbox"]), proposer=r["proposer"],
                          hint=dict(r.get("hint") or {}),
                          sig=dict(r["sig"]) if r.get("sig") else None)
                   for r in d.get("regions") or []]
        known = set(Element.__dataclass_fields__)
        elements = [Element(**{k: v for k, v in e.items() if k in known})
                    for e in d.get("elements") or []]
        return cls(frame=Frame(**fr), proposer=dict(d.get("proposer") or {}),
                   regions=regions, elements=elements, timing=dict(d.get("timing") or {}),
                   model=dict(d.get("model") or {}), uia=dict(d.get("uia") or {}),
                   reused=bool(d.get("reused")),
                   reused_regions=int(d.get("reused_regions") or 0), schema=SCHEMA)

    def find(self, label: str) -> List[Element]:
        """Elements whose label or text contains ``label`` (case-insensitive)."""
        needle = (label or "").strip().lower()
        return [e for e in self.elements
                if needle and (needle in e.label.lower() or needle in e.text.lower())]


class ProposerUnavailable(RuntimeError):  # noqa: N818 - public name, mirrors VisionUnavailable
    """The explicitly requested proposer cannot run here; the message says why."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(img) -> list:
    return list(img.convert("L").resize((32, 32)).tobytes())


def _signature_diff(a: Optional[list], b: Optional[list]) -> float:
    """Mean absolute grey difference as a FRACTION (0..1); unknown = 1."""
    try:
        from awvision.sight import frame_diff
        return frame_diff(a, b) / 255.0
    except ImportError:
        if not a or not b or len(a) != len(b):
            return 1.0
        return sum(abs(x - y) for x, y in zip(a, b)) / (255.0 * len(a))


#: A region is UNCHANGED when its cheap hash matches, or its 16x16 grey thumbnail
#: differs from the previous one by at most this many grey levels on average (0..255).
#: The hash catches every exact match; the MAD forgives sub-pixel anti-aliasing drift
#: that would otherwise force a needless model call. A real content change (a button
#: turning blue, a label changing) moves the mean by far more than this.
REUSE_MAD = 3.0

_SIG_THUMB = (16, 16)


def _clamp_box(box: Sequence[float], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    w, h = size
    x1 = max(0, min(w - 1, int(box[0])))
    y1 = max(0, min(h - 1, int(box[1])))
    x2 = max(x1 + 1, min(w, int(box[2])))
    y2 = max(y1 + 1, min(h, int(box[3])))
    return x1, y1, x2, y2


def _region_signature(img, bbox: Sequence[float]) -> dict:
    """``{hash, thumb}`` for the pixels under ``bbox`` (image px).

    The hash is over the native-resolution grey crop, so ANY pixel change flips it; the
    16x16 thumbnail carries the mean absolute difference that lets a one-pixel drift
    still read as unchanged. Both are pure PIL, no model, no network.
    """
    from PIL import Image
    crop = img.crop(_clamp_box(bbox, img.size)).convert("L")
    return {"hash": hashlib.blake2b(crop.tobytes(), digest_size=8).hexdigest(),
            "thumb": list(crop.resize(_SIG_THUMB, Image.BOX).tobytes())}


def _mean_abs_diff(a: Optional[list], b: Optional[list]) -> Optional[float]:
    """Mean absolute grey difference over two thumbnails; ``None`` when not comparable."""
    if not a or not b or len(a) != len(b):
        return None
    return round(sum(abs(int(x) - int(y)) for x, y in zip(a, b)) / len(a), 3)


def _overlap_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Overlap / the SMALLER area, 0..1; 1.0 when either box sits inside the other."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    inter = ix * iy
    ai = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    ao = max(0.0, outer[2] - outer[0]) * max(0.0, outer[3] - outer[1])
    small = min(ai, ao)
    return inter / small if small > 0 else 0.0


def _center_in(box: Sequence[float], outer: Sequence[float]) -> bool:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _covered_unchanged(box: Sequence[float], unchanged: dict) -> Optional[str]:
    """The unchanged prev region a proposed region sits on, or ``None``.

    Needs most of the overlap AND the proposed region's centre inside the prev box, so a
    union region that merely touches an unchanged control is still sent to the changed
    pipeline rather than being skipped whole.
    """
    best, best_ratio = None, 0.0
    for pid, d in unchanged.items():
        ratio = _overlap_ratio(box, d["bbox"])
        if ratio >= 0.6 and _center_in(box, d["bbox"]) and ratio > best_ratio:
            best, best_ratio = pid, ratio
    return best


def _center_inside(box: Sequence[float], unchanged: dict) -> Optional[str]:
    for pid, d in unchanged.items():
        if _center_in(box, d["bbox"]):
            return pid
    return None


def _prev_element_ms(el: Element) -> float:
    ms = (el.evidence or {}).get("ms")
    return float(ms) if isinstance(ms, (int, float)) else 0.0


def _prev_region_ms(prev: "ScreenState", region_id: str) -> float:
    """The model time this region's element(s) cost last frame: the honest 'saved' estimate."""
    total = 0.0
    for e in prev.elements:
        rids = (e.evidence or {}).get("region_ids") or ([e.region_id] if e.region_id else [])
        if region_id in rids:
            total += _prev_element_ms(e)
    return round(total, 1)


def _region_reuse_decisions(img, prev: "ScreenState") -> List[dict]:
    """One decision per previous region: did its pixels change in ``img``?

    Each entry is ``{region_id, bbox, reused, hash_equal, mad, saved_ms}``. ``reused``
    is true when the cheap hash matches OR the 16x16 thumbnail's mean absolute difference
    is at most :data:`REUSE_MAD`. A region with no stored signature (a state saved before
    signatures existed) is treated as changed: never reuse what could not be compared.
    """
    out: List[dict] = []
    for r in prev.regions:
        entry = {"region_id": r.id, "bbox": list(r.bbox), "reused": False,
                 "hash_equal": False, "mad": None, "saved_ms": 0.0}
        if not r.sig:
            entry["reason"] = "no signature in the previous state"
            out.append(entry)
            continue
        new = _region_signature(img, r.bbox)
        entry["hash_equal"] = new["hash"] == r.sig.get("hash")
        mad = _mean_abs_diff(new.get("thumb"), r.sig.get("thumb"))
        entry["mad"] = mad
        entry["reused"] = bool(entry["hash_equal"]
                               or (mad is not None and mad <= REUSE_MAD))
        if entry["reused"]:
            entry["saved_ms"] = _prev_region_ms(prev, r.id)
        out.append(entry)
    return out


def _full_reuse_state(frame: Frame, prev: "ScreenState", timing: Dict[str, Any],
                      t_all: float, decisions: Optional[List[dict]] = None,
                      img=None) -> "ScreenState":
    """A ScreenState that reuses EVERY previous element, rebased to ``frame``.

    The cache keys on the PIXELS only, and an element's bbox is ABSOLUTE screen px.
    Reusing across a different origin must re-base those bboxes by the origin delta;
    otherwise a caller on a second monitor gets ``frame.origin`` saying the new monitor
    and element bboxes still in the old monitor's coordinates. When ``img`` is given the
    region signatures are recomputed from it, so near-miss (MAD-tolerated) drift cannot
    accumulate frame after frame.
    """
    saved = round(sum(_prev_element_ms(e) for e in prev.elements), 1)
    reuse_ms = float(timing.get("reuse_ms", 0.0))
    timing.update({"propose_ms": 0.0, "crop_ms": 0.0, "neurons_ms": 0.0,
                   "ocr_ms": 0.0, "inspect_ms": 0.0, "merge_ms": 0.0,
                   "dom_join_ms": 0.0, "uia_join_ms": 0.0,
                   "sig_ms": 0.0, "reuse_ms": reuse_ms,
                   "reused_regions": len(prev.regions),
                   "reused_ms_saved": saved,
                   "reuse": {"count": len(prev.regions), "ms": reuse_ms, "saved_ms": saved,
                             "regions": decisions or [], "whole_frame": True},
                   "total_ms": _ms(t_all)})
    dx = frame.origin[0] - prev.frame.origin[0]
    dy = frame.origin[1] - prev.frame.origin[1]
    elements = prev.elements
    if dx or dy:
        elements = [replace(e, bbox=[e.bbox[0] + dx, e.bbox[1] + dy,
                                     e.bbox[2] + dx, e.bbox[3] + dy])
                    for e in prev.elements]
    regions = prev.regions
    if img is not None:
        regions = [replace(r, sig=_region_signature(img, r.bbox)) for r in prev.regions]
    return ScreenState(frame=frame, proposer=dict(prev.proposer), regions=regions,
                       elements=elements, timing=timing, model=dict(prev.model, calls=0),
                       uia=dict(prev.uia), reused=True,
                       reused_regions=len(prev.regions))


def _rank_for_inspection(region: Region, features: dict) -> Tuple[int, float]:
    """Model calls are capped; spend them on control-sized, inked regions first."""
    x1, y1, x2, y2 = region.bbox
    w, h = x2 - x1, y2 - y1
    control_sized = 1 if (10 <= h <= 120 and 10 <= w <= 900) else 0
    return (control_sized, float(features.get("edge_density", 0.0)))


def _element_from_region(region: Region, origin: Tuple[int, int], features: dict,
                         guess: Tuple[str, float]) -> Element:
    hint = region.hint or {}
    role = hint.get("role") or guess[0]
    conf = float(hint.get("confidence") or (0.5 if hint else guess[1]))
    b = list(region.bbox)
    ox, oy = origin
    hint_text = str(hint.get("text") or "")
    # A region proposer that already read the text (tesseract TSV / OmniParser OCR)
    # names its own reader; every other hint is geometry only.
    text_source = ""
    if hint_text:
        text_source = ("tesseract-tsv" if region.proposer == "ocr"
                       else PROPOSER_SOURCE.get(region.proposer, region.proposer))
    return Element(
        id="", region_id=region.id, bbox=[b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy],
        bbox_image=b, label=str(hint.get("label") or ""), text=hint_text,
        text_source=text_source,
        role=role if role in ROLES else "other", confidence=round(conf, 3),
        source=PROPOSER_SOURCE.get(region.proposer, region.proposer),
        sources=[PROPOSER_SOURCE.get(region.proposer, region.proposer)],
        interactable=bool(hint.get("interactable", role in INTERACTIVE_ROLES)),
        evidence={"neurons": features})


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _run_in_budget(executor: Any, tasks: List[Any], inspector: Any, budget: Budget,
                   call_ms: float, concurrency: int,
                   timeout_s: float) -> Tuple[List[Any], List[str]]:
    """Run model calls wave by wave while a WHOLE crop still fits.

    Returns ``(results, unrun_region_ids)``. A wave is at most ``concurrency`` crops and
    costs about ``call_ms``; the clock is re-read before every wave, so an optimistic
    estimate can overshoot by one wave but never leave results stranded past the check.
    """
    results: List[Any] = []
    i = 0
    while i < len(tasks) and budget.fits(call_ms):
        wave = tasks[i:i + concurrency]
        results.extend(executor.run(wave, inspector, max_concurrency=concurrency,
                                    timeout_s=timeout_s))
        i += len(wave)
    return results, [t.region_id for t in tasks[i:]]


def build_screen_state(image: Optional[str] = None, *, screen: bool = False,
                       proposers: Sequence[str] = ("omniparser", "ocr", "blobs", "grid"),
                       explicit_proposer: bool = False,
                       inspect: bool = True, max_regions: int = 24, margin: int = 8,
                       executor: Any = None, inspector: Any = None,
                       max_concurrency: int = 4, timeout_s: float = 45.0,
                       dom: Any = None, uia: Any = None,
                       prev: Optional[ScreenState] = None,
                       origin: Optional[Tuple[int, int]] = None,
                       capture_path: Optional[str] = None,
                       vlm_model: Optional[str] = None,
                       monitor: Optional[int] = None,
                       ocr: Any = None,
                       budget_ms: Optional[float] = None,
                       clock: Optional[Callable[[], float]] = None,
                       call_estimate_ms: float = DEFAULT_CALL_ESTIMATE_MS) -> ScreenState:
    """Run the whole pipeline on one frame and return its ScreenState.

    ``monitor`` (1-based, mss order) means "this is monitor N": with ``screen`` it is
    captured through mss and verified, with an image file its pixel size is verified
    against that monitor. Either mismatch raises :class:`awscreen.capture.CaptureError`
    (never a frame silently treated as a different monitor's).

    ``budget_ms`` is a wall-clock deadline for the WHOLE pipeline. Neurons always run
    (they are local and near-free); a vision call is made only while the remaining time
    covers a whole crop, and the call cap is shrunk to fit. Every candidate the clock
    left uninspected stays in the state with ``status='budget'`` -- never dropped, never
    given a model's answer. ``clock`` (seconds, monotonic by default) exists so tests can
    drive the deadline deterministically.

    Raises FileNotFoundError / OSError for an unreadable image and ProposerUnavailable
    when ``explicit_proposer`` is set and that proposer cannot run. A vision endpoint that
    cannot be resolved does NOT raise: the state carries ``model.error`` and no vlm
    elements, so a caller can tell "nothing seen" from "could not look".
    """
    from PIL import Image

    from . import crops as _crops
    from . import merge as _merge
    from . import neurons as _neurons
    from . import regions as _regions
    from .executors import LocalExecutor, clamp_concurrency
    from .inspect import CropTask

    timing: Dict[str, Any] = {}
    t_all = time.perf_counter()
    budget = Budget(budget_ms, clock) if budget_ms is not None else None

    t0 = time.perf_counter()
    capture = "file"
    if screen:
        out = capture_path or str(Path.home() / ".aither" / "awscreen" / "last_capture.jpg")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if monitor is None:
            from awvision.sight import capture_screen_origin
            image, captured_origin = capture_screen_origin(out)
            capture = "primary"
        else:
            # Monitor-aware capture: mss grabs that monitor's own pixels and verifies the
            # frame's size against it. Asking for a non-primary monitor without mss
            # raises instead of handing back a primary frame labelled as it.
            from .capture import capture as capture_monitor
            image, captured_monitor = capture_monitor(out, monitor)
            captured_origin = captured_monitor.origin
            capture = f"monitor:{captured_monitor.index}"
        if origin is None:
            # The captured monitor's real origin, not a hardcoded (0,0): a second
            # monitor's elements are reported in desktop px, not that monitor's px.
            origin = captured_origin
    if not image:
        raise FileNotFoundError("no image given (pass a path or screen=True)")
    if not Path(image).is_file():
        raise FileNotFoundError(f"image not found: {image}")
    if not screen and monitor is not None:
        # A FILE claimed to be a specific monitor: verify its bytes ARE that monitor and
        # take its desktop origin. A 1280x800 frame labelled monitor 2 would put every
        # click 1280 px off, so unequal dimensions raise instead of guessing.
        from .capture import verify_frame
        claimed = verify_frame(str(image), monitor)
        capture = f"monitor:{claimed.index}"
        if origin is None:
            origin = claimed.origin
    if origin is None:
        origin = (0, 0)
    with Image.open(image) as raw:
        img = raw.convert("RGB")
    frame = Frame(path=str(image), width=img.width, height=img.height,
                  origin=(int(origin[0]), int(origin[1])), capture=capture,
                  signature=_signature(img), sha256=_sha256(str(image)))
    timing["capture_ms"] = _ms(t0)

    dims_match = (prev is not None and prev.frame.width == frame.width
                  and prev.frame.height == frame.height)
    prev_has_sigs = bool(dims_match and prev.regions and any(r.sig for r in prev.regions))

    # Legacy whole-frame shortcut: only when the previous state carries no per-region
    # signatures, so nothing finer can be compared. When signatures ARE present they are
    # the authority — the frame-level grey diff (mean over the whole frame) misses a
    # localised change such as four buttons changing colour on a 1280x800 desktop.
    if dims_match and not prev_has_sigs \
            and _signature_diff(frame.signature, prev.frame.signature) < 0.01:
        return _full_reuse_state(frame, prev, timing, t_all, img=img)

    # Per-region change detection is done BEFORE proposing: a frame whose every region is
    # untouched reuses whole with zero model calls and no proposer run.
    reuse_detail: List[dict] = []
    unchanged: Dict[str, dict] = {}
    if dims_match and prev.regions:
        t_dec = time.perf_counter()
        reuse_detail = _region_reuse_decisions(img, prev)
        unchanged = {d["region_id"]: d for d in reuse_detail if d["reused"]}
        timing["reuse_ms"] = _ms(t_dec)
        if prev_has_sigs and len(unchanged) == len(prev.regions):
            return _full_reuse_state(frame, prev, timing, t_all, reuse_detail, img=img)

    t0 = time.perf_counter()
    chain = [_regions.make_proposer(n) for n in proposers]
    if explicit_proposer and len(chain) == 1:
        ok, why = chain[0].available()
        if not ok:
            raise ProposerUnavailable(f"proposer {chain[0].name!r} unavailable: {why}")
    regions, report = _regions.propose(img, chain)
    timing["propose_ms"] = _ms(t0)

    # Per-region pixel signatures: what the NEXT frame diffs against. Cheap (a native
    # hash + a 16x16 grey thumbnail), and computed from the ORIGINAL frame, never a
    # masked copy, so the saved state is always comparable to a real next frame.
    t0 = time.perf_counter()
    for r in regions:
        r.sig = _region_signature(img, r.bbox)
    timing["sig_ms"] = _ms(t0)

    # -- region-level reuse: unchanged regions from --prev cost zero model calls --------
    reused_elements: List[Element] = []
    reused_regions = len(unchanged)
    reused_ms_saved = round(sum(d["saved_ms"] for d in unchanged.values()), 1)
    processing = regions
    if unchanged:
        prev_by_id = {r.id: r for r in prev.regions}
        matched: Dict[str, str] = {}    # prev region id -> this frame's region id
        covered: set = set()             # region ids the changed pipeline must skip
        for nr in regions:
            pid = _covered_unchanged(nr.bbox, unchanged)
            if pid is not None and pid not in matched:
                matched[pid] = nr.id
                covered.add(nr.id)
        # A proposer can miss a flat unchanged region (no ink at all); adopt it so the
        # output still carries every region the next frame will diff.
        for pid in unchanged:
            if pid not in matched and pid in prev_by_id:
                pr = prev_by_id[pid]
                nid = f"r{len(regions)}"
                regions.append(Region(id=nid, bbox=tuple(pr.bbox),
                                      proposer=pr.proposer, hint=dict(pr.hint),
                                      sig=_region_signature(img, pr.bbox)))
                matched[pid] = nid
                covered.add(nid)
        # Reuse the previous element(s) for every unchanged region. The ABSOLUTE bbox
        # moves with an origin change; region ids are re-pointed at this frame's
        # regions. An element spanning a changed region too is left to the changed
        # pipeline rather than half-reused.
        dxo = frame.origin[0] - prev.frame.origin[0]
        dyo = frame.origin[1] - prev.frame.origin[1]
        seen: set = set()
        for e in prev.elements:
            rids = list((e.evidence or {}).get("region_ids")
                        or ([e.region_id] if e.region_id else []))
            if not rids:
                pid = _center_inside(e.bbox_image, unchanged)
                if pid is None:
                    continue
                rids = [pid]
            if not all(rid in unchanged for rid in rids):
                continue
            key = (e.source, tuple(e.bbox_image), e.label, e.text)
            if key in seen:
                continue
            seen.add(key)
            new_rids = [matched[rid] for rid in rids if rid in matched]
            if not new_rids:
                continue
            ne = replace(e)
            ne.bbox = [e.bbox[0] + dxo, e.bbox[1] + dyo,
                       e.bbox[2] + dxo, e.bbox[3] + dyo]
            ne.region_id = new_rids[0]
            ne.evidence = dict(ne.evidence or {}, region_ids=new_rids)
            reused_elements.append(ne)
        processing = [nr for nr in regions if nr.id not in covered]
    timing.setdefault("reuse_ms", 0.0)
    timing["reused_regions"] = reused_regions
    timing["reused_ms_saved"] = reused_ms_saved
    timing["reuse"] = {"count": reused_regions, "ms": timing["reuse_ms"],
                       "saved_ms": reused_ms_saved, "regions": reuse_detail,
                       "whole_frame": False}

    t0 = time.perf_counter()
    crops = [_crops.make_crop(img, r, margin=margin, origin=frame.origin) for r in processing]
    timing["crop_ms"] = _ms(t0)

    t0 = time.perf_counter()
    feats = [_neurons.run_neurons(c) for c, _t in crops]
    timing["neurons_ms"] = _ms(t0)

    elements: List[Element] = []
    candidates = []
    for r, (crop, tf), f in zip(processing, crops, feats):
        guess = _neurons.guess_role(f)
        el = _element_from_region(r, frame.origin, f, guess)
        ask, why = _neurons.worth_asking(f)
        el.evidence["ask"] = why
        elements.append(el)
        if ask:
            candidates.append((r, crop, tf, f))

    model: Dict[str, Any] = {"endpoint_host": "", "model": "", "calls": 0}

    # Exact text, one local OCR pass over the whole frame. It runs for whatever regions
    # the winning proposer yielded, so blobs/grid boxes still carry readable strings.
    # `ocr=False` disables it; `ocr=<provider>` injects one (tests, other engines).
    t0 = time.perf_counter()
    ocr_report: Dict[str, Any] = {"provider": "", "ok": False, "reason": "", "lines": 0}
    if ocr is not False and elements:
        if report.get("ran") == "ocr":
            # The proposer already read the frame with the same engine; its hints ARE
            # the exact text. Re-running tesseract here would be a second identical call.
            ocr_report.update(provider="tesseract-tsv", ok=True, lines=len(regions),
                              reason="regions already came from the tesseract TSV proposer")
        else:
            from .ocr import TesseractTsvProvider, attach_text
            provider = ocr if ocr is not None else TesseractTsvProvider()
            ocr_report["provider"] = getattr(provider, "name", type(provider).__name__)
            ok, why = provider.available()
            ocr_report.update(ok=bool(ok), reason=why)
            if ok:
                try:
                    lines = provider.read(img)
                    ocr_report["lines"] = len(lines)
                    attach_text(elements, lines)
                except Exception as exc:  # noqa: BLE001 - a broken engine is reported
                    ocr_report.update(ok=False,
                                      reason=f"{type(exc).__name__}: {str(exc)[:200]}")
    model["ocr"] = ocr_report
    timing["ocr_ms"] = _ms(t0) if ocr is not False else 0.0

    t0 = time.perf_counter()
    if inspect and candidates:
        candidates.sort(key=lambda c: _rank_for_inspection(c[0], c[3]), reverse=True)
        cap = max(0, int(max_regions))
        conc = clamp_concurrency(max_concurrency)
        call_ms = max(1.0, float(call_estimate_ms))
        effective = cap
        budget_skipped = []
        if budget is not None:
            # Shrink the caps to what the clock can pay for: whole waves of `conc`
            # crops at `call_ms` each, never more than the caller's --max-regions.
            effective = min(cap, budget.cap_for(call_ms, conc))
            model["max_regions"] = cap
            model["cap"] = effective
            model["budget"] = dict(budget.as_dict(), estimate_ms=call_ms,
                                   concurrency=conc)
            budget_skipped = [c[0] for c in candidates[effective:cap]]
        chosen = candidates[:effective]
        model["skipped_by_cap"] = len(candidates) - min(len(candidates), cap)
        insp = inspector
        # Do not even resolve (probe) a vision endpoint when no call can fit.
        if insp is None and not (budget is not None and effective <= 0):
            from .inspect import vision_inspector
            try:
                insp = vision_inspector(model=vlm_model, timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001 - VisionUnavailable and kin
                model["error"] = f"{type(exc).__name__}: {exc}"
                insp = None
        unrun_ids: List[str] = []
        if insp is not None:
            model["endpoint_host"] = getattr(insp, "endpoint_host", "") or "injected"
            model["model"] = getattr(insp, "model", "") or ""
            tasks = [CropTask(region_id=r.id, b64=_crops.crop_b64(crop),
                              media_type="image/png", transform=tf, features=f)
                     for r, crop, tf, f in chosen]
            ex = executor or LocalExecutor()
            model["executor"] = getattr(ex, "name", type(ex).__name__)
            try:
                if budget is None:
                    results = ex.run(tasks, insp, max_concurrency=max_concurrency,
                                     timeout_s=timeout_s)
                else:
                    results, unrun_ids = _run_in_budget(
                        ex, tasks, insp, budget, call_ms, conc, timeout_s)
            except Exception as exc:  # noqa: BLE001 - an executor refusal is reported
                model["error"] = f"{type(exc).__name__}: {exc}"
                results = []
            model["calls"] = len(results)
            model["ok"] = sum(1 for x in results if x.status == "ok")
            by_id = {r.id: r for r in processing}
            for res in results:
                reg = by_id.get(res.region_id)
                if reg is None:
                    continue
                b = list(reg.bbox)
                ox, oy = frame.origin
                ok = res.status == "ok"
                role = res.role if ok and res.role in ROLES else ("other" if ok else "none")
                # The crop the model already inspects also answers its exact text. That
                # reading is part of THIS call (no second request): it is recorded on the
                # element as text_source='vlm-ocr' and tagged in `sources`.
                text = (res.text if ok else "") or ""
                sources = ["vlm"] + (["vlm-ocr"] if text else [])
                elements.append(Element(
                    id="", region_id=reg.id, bbox=[b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy],
                    bbox_image=b, label=res.label if ok else "", text=text,
                    text_source="vlm-ocr" if text else "",
                    role=role, confidence=res.confidence if ok else 0.0, source="vlm",
                    sources=sources, interactable=ok and role in INTERACTIVE_ROLES,
                    status=res.status,
                    evidence={"vlm_raw": (res.raw or "")[:200], "ms": res.ms,
                              "error": res.error}))
        if budget is not None:
            # The clock left these crops no whole call. The neuron-only element stays,
            # explicitly named 'budget': dropped would hide it, a model's guess would
            # invent it.
            model["budget"].update(budget.as_dict())   # refresh with post-call remaining
            skipped_ids = {r.id for r in budget_skipped} | set(unrun_ids)
            for el in elements:
                if el.region_id in skipped_ids:
                    el.status = BUDGET_STATUS
                    el.evidence["budget"] = "not inspected: no whole crop fit in the budget"
            model["skipped_by_budget"] = len(skipped_ids)
    timing["inspect_ms"] = _ms(t0) if model["calls"] else 0.0

    t0 = time.perf_counter()
    merged = _merge.merge_elements(elements)
    timing["merge_ms"] = _ms(t0)

    # Unchanged regions come from --prev and stay exactly as they were (no crop, no
    # neuron, no model). They are disjoint from the processed regions, so they are
    # appended rather than re-merged.
    state = ScreenState(frame=frame, proposer=report, regions=regions,
                        elements=merged + reused_elements, timing=timing, model=model,
                        reused=False, reused_regions=reused_regions)
    t0 = time.perf_counter()
    if dom is not None:
        from .dom import DomSnapshot, join_dom
        snap = dom if isinstance(dom, DomSnapshot) else DomSnapshot.from_dict(dom)
        state = join_dom(state, snap)
    timing["dom_join_ms"] = _ms(t0) if dom is not None else 0.0

    t0 = time.perf_counter()
    if uia is not None and uia is not False:
        from . import uia as _uia
        try:
            if isinstance(uia, _uia.UiaSnapshot):
                snap = uia
            elif isinstance(uia, dict):
                snap = _uia.UiaSnapshot.from_dict(uia)
            elif uia is True or (isinstance(uia, str) and uia.strip().lower() == "auto"):
                snap = _uia.walk_focused_window()
            else:
                raise ValueError("uia must be a UiaSnapshot, a snapshot dict, or 'auto'")
        except _uia.UiaUnavailableError as exc:
            # Skipped, with the reason carried so the CLI (or a caller) can print it.
            state.uia = {"attempted": True, "ok": False, "reason": str(exc)}
        else:
            state = _uia.join_uia(state, snap)
    timing["uia_join_ms"] = _ms(t0) if uia is not None and uia is not False else 0.0

    state.elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    for i, e in enumerate(state.elements):
        e.id = f"e{i}"
    timing["total_ms"] = _ms(t_all)
    return state
