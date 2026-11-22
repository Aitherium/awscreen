"""Exact text from pixels: an OcrProvider seam, a tesseract lane, and text-only reads.

A region proposer answers WHERE; an OCR provider answers WHAT THE STRINGS ARE. The
provider runs once on the whole frame and its line boxes are matched onto whatever
regions the winning proposer yielded, so a :class:`ScreenState` carries exact text even
when the boxes came from pure-PIL blobs. When no OCR engine is installed,
``available()`` says so and ``read()`` returns nothing: a skip, never a guess.

The same seam backs ``awscreen state --text-only``: it takes a region list (from a file
or the proposer chain) and answers one exact string per region, with the local tesseract
lane first and a vision text reader only for the regions OCR could not read. The vision
lane reuses the same short-answer machinery as crop inspection, so a text-only answer
costs one call per unread region and nothing more.

Text provenance travels with the element: ``Element.text_source`` names the reader
(``tesseract-tsv`` or ``vlm-ocr``) and ``Element.sources`` carries the matching source
tag, so a consumer can tell an exact reading from a region hint.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from PIL import Image

from .state import Element, Region

#: Asked of the model for a crop whose strings are all we want. It must return the text
#: exactly, not a summary: the answer becomes element.text.
TEXT_ONLY_PROMPT = (
    "This is a small crop of a computer screen. Reply with ONLY a JSON object: "
    '{"text": the exact visible text in the crop, copied character for character '
    '(keep case, punctuation and spacing), or "" if there is none}.'
)


@dataclass
class OcrLine:
    """One line of text found in an image, in image px (x2/y2 exclusive)."""

    text: str
    bbox: Tuple[int, int, int, int]
    conf: float = 0.0
    provider: str = ""


class OcrProvider(Protocol):
    """Anything that answers exact strings for an image."""

    name: str

    def available(self) -> Tuple[bool, str]: ...

    def read(self, img: Image.Image) -> List[OcrLine]: ...


class NullOcrProvider:
    """An explicitly absent engine: ``available()`` explains why, ``read()`` skips."""

    name = "none"

    def __init__(self, reason: str = "unavailable: no OCR engine configured") -> None:
        self.reason = reason

    def available(self) -> Tuple[bool, str]:
        return False, self.reason

    def read(self, img: Image.Image) -> List[OcrLine]:
        return []


class TesseractTsvProvider:
    """Local tesseract TSV through the SAME lane the region chain already uses.

    Wrapping :class:`awscreen.regions.TesseractTsvProposer` keeps one implementation of
    the TSV call (including the shared vision-service lane and the console-less Windows
    spawn); this class only re-shapes line boxes into :class:`OcrLine` records.
    """

    name = "tesseract-tsv"

    def __init__(self, binary: Optional[str] = None, timeout_s: float = 60.0) -> None:
        from .regions import TesseractTsvProposer

        self._proposer = TesseractTsvProposer(binary=binary, timeout_s=timeout_s)

    def available(self) -> Tuple[bool, str]:
        return self._proposer.available()

    def read(self, img: Image.Image) -> List[OcrLine]:
        ok, _why = self.available()
        if not ok:
            return []
        out: List[OcrLine] = []
        for region in self._proposer.propose(img):
            hint = region.hint or {}
            text = str(hint.get("text") or "").strip()
            if not text:
                continue
            out.append(OcrLine(text=text, bbox=tuple(region.bbox),
                               conf=float(hint.get("confidence") or 0.0),
                               provider=self.name))
        return out


def make_ocr_provider(name: str = "tesseract", **kwargs: Any) -> OcrProvider:
    """Provider by name; ``auto``/``tesseract`` degrade to a skip when the binary is gone."""
    key = (name or "").strip().lower()
    if key in ("tesseract", "tesseract-tsv", "auto", ""):
        return TesseractTsvProvider(**kwargs)
    if key in ("none", "null", "off"):
        return NullOcrProvider()
    raise ValueError(f"unknown OCR provider {name!r} (tesseract|none)")


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _covers(big: Sequence[float], small: Sequence[float], min_frac: float = 0.5) -> bool:
    """True when ``small`` is mostly inside ``big`` (intersection over small's area)."""
    area = _area(small)
    return area > 0 and _intersection(big, small) / area >= min_frac


def assign_lines(regions: Sequence[Tuple[str, Sequence[float]]],
                 lines: Sequence[OcrLine],
                 min_frac: float = 0.5) -> Dict[str, Tuple[str, float]]:
    """Attach each OCR line to the TIGHTEST region that contains it.

    Returns ``{region_id: (text, confidence)}``. A line is awarded to the smallest
    region covering it (either box mostly inside the other), so a parent panel never
    steals the string of a child control that was also proposed. A region's lines are
    joined in reading order.
    """
    items = [(str(rid), tuple(float(v) for v in b)) for rid, b in regions
             if b is not None and len(b) == 4]
    picked: Dict[str, List[OcrLine]] = {}
    for line in lines:
        if _area(line.bbox) <= 0:
            continue
        cands = [(rid, box) for rid, box in items
                 if _covers(box, line.bbox, min_frac) or _covers(line.bbox, box, min_frac)]
        if not cands:
            continue
        rid, _box = min(cands, key=lambda rb: _area(rb[1]))
        picked.setdefault(rid, []).append(line)

    out: Dict[str, Tuple[str, float]] = {}
    for rid, found in picked.items():
        found.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        words: List[str] = []
        confs: List[float] = []
        for line in found:
            text = line.text.strip()
            if not text:
                continue
            if not words or words[-1] != text:
                words.append(text)
            confs.append(line.conf)
        if words:
            out[rid] = (" ".join(words), round(sum(confs) / len(confs), 3))
    return out


def attach_text(elements: Sequence[Element], lines: Sequence[OcrLine], *,
                overwrite: bool = False, source: str = "tesseract-tsv") -> int:
    """Write exact OCR text onto elements that do not already carry text.

    ``source`` is the provider name recorded in ``Element.text_source``; the element's
    ``sources`` list gains ``ocr`` so source-priority merging keeps the reading.
    Returns the number of elements given text.
    """
    if not lines or not elements:
        return 0
    pairs = [(str(i), tuple(e.bbox_image)) for i, e in enumerate(elements) if e.bbox_image]
    assigned = assign_lines(pairs, lines)
    count = 0
    for i, element in enumerate(elements):
        text, conf = assigned.get(str(i), ("", 0.0))
        if not text or (element.text and not overwrite):
            continue
        element.text = text
        element.text_source = source
        if "ocr" not in element.sources:
            element.sources.append("ocr")
        element.evidence.setdefault("ocr_conf", conf)
        count += 1
    return count


@dataclass
class TextResult:
    region_id: str
    bbox: List[int]
    text: str = ""
    source: str = ""              # tesseract-tsv | vlm-ocr
    confidence: float = 0.0
    status: str = "skipped"       # ok | skipped | timeout | error
    error: str = ""


@dataclass
class TextOnlyResult:
    texts: List[TextResult] = field(default_factory=list)
    provider: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _region_pair(region: Any, index: int) -> Tuple[str, Tuple[int, int, int, int]]:
    if isinstance(region, Region):
        return (region.id or f"r{index}"), tuple(int(v) for v in region.bbox)
    if isinstance(region, Mapping):
        bbox = region.get("bbox") or region.get("bbox_image") or []
        return str(region.get("id") or f"r{index}"), tuple(int(v) for v in bbox)
    if (isinstance(region, (tuple, list)) and len(region) == 2
            and not isinstance(region[0], (int, float))
            and isinstance(region[1], (tuple, list)) and len(region[1]) == 4):
        return str(region[0]), tuple(int(v) for v in region[1])   # (id, bbox)
    return f"r{index}", tuple(int(v) for v in region)


def read_region_text(img: Image.Image, regions: Sequence[Any], *,
                     ocr_provider: Optional[OcrProvider] = None,
                     inspector: Optional[Callable] = None,
                     inspector_factory: Optional[Callable[[], Callable]] = None,
                     executor: Any = None, max_concurrency: int = 4,
                     timeout_s: float = 45.0, use_vlm: bool = True,
                     provider_name: str = "tesseract") -> TextOnlyResult:
    """Answer the exact strings for a region list, OCR first and vision second.

    ``regions`` accepts :class:`awscreen.state.Region` objects, dicts with ``id`` and
    ``bbox``, or bare ``[x1, y1, x2, y2]`` boxes. The OCR provider is asked once for the
    whole frame; only regions it could not read are sent to the vision inspector (one
    call each). With neither lane runnable every text is ``status='skipped'`` and the
    provider report says why -- callers turn that into "could not judge", never a pass.
    """
    from . import crops as _crops
    from .executors import LocalExecutor
    from .inspect import CropTask

    t_all = time.perf_counter()
    pairs = [_region_pair(region, i) for i, region in enumerate(regions)]
    texts = [TextResult(region_id=rid, bbox=list(box)) for rid, box in pairs]

    provider_report: Dict[str, Any] = {"provider": "", "ok": False, "reason": "", "lines": 0}
    t0 = time.perf_counter()
    provider = ocr_provider if ocr_provider is not None \
        else make_ocr_provider(provider_name)
    name = getattr(provider, "name", type(provider).__name__)
    ok, why = provider.available()
    provider_report.update(provider=name, ok=bool(ok), reason=why)
    if ok and pairs:
        try:
            lines = provider.read(img)
            provider_report["lines"] = len(lines)
            assigned = assign_lines(pairs, lines)
            for i, (rid, box) in enumerate(pairs):
                text, conf = assigned.get(rid, ("", 0.0))
                texts[i] = TextResult(region_id=rid, bbox=list(box), text=text,
                                      source=name if text else "", confidence=conf,
                                      status="ok")
        except Exception as exc:  # noqa: BLE001 - a broken engine is reported, not fatal
            provider_report.update(ok=False,
                                   reason=f"{type(exc).__name__}: {str(exc)[:200]}")
    timing: Dict[str, float] = {"ocr_ms": round((time.perf_counter() - t0) * 1000, 1)}

    model: Dict[str, Any] = {"calls": 0, "ok": 0}
    missing = [i for i, result in enumerate(texts) if not result.text]
    if use_vlm and missing:
        if inspector is None and inspector_factory is not None:
            try:
                inspector = inspector_factory()
            except Exception as exc:  # noqa: BLE001 - resolution failure is reported
                model["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                inspector = None
        if inspector is not None:
            t0 = time.perf_counter()
            tasks, indexes = [], []
            for i in missing:
                rid, box = pairs[i]
                crop, transform = _crops.make_crop(img, box)
                tasks.append(CropTask(region_id=rid, b64=_crops.crop_b64(crop),
                                      media_type="image/png", transform=transform,
                                      question=TEXT_ONLY_PROMPT))
                indexes.append(i)
            run = executor or LocalExecutor()
            try:
                results = run.run(tasks, inspector, max_concurrency=max_concurrency,
                                  timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001 - an executor refusal is reported
                model["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                results = []
            model["calls"] = len(results)
            model["ok"] = sum(1 for r in results if r.status == "ok")
            for i, result in zip(indexes, results):
                text = (result.text or "").strip()
                if result.status == "ok":
                    texts[i] = replace(texts[i], text=text,
                                       source="vlm-ocr" if text else "",
                                       confidence=result.confidence, status="ok",
                                       error="")
                else:
                    texts[i] = replace(texts[i], status=result.status,
                                       error=result.error)
            timing["vlm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    timing["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
    return TextOnlyResult(texts=texts, provider=provider_report, model=model,
                          timing=timing)
