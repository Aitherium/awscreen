"""Region proposers: cheap guesses at WHERE the things on screen are.

A chain is tried in order and the first proposer that yields regions wins; every attempt
(including the ones that could not run) is reported as ``{name, ok, reason, ms, n}`` so a
reader can see why the grid ran instead of OmniParser.

* ``omniparser`` -- the OmniParser service, only when ``/health`` answers within 2 s AND
  reports its models already loaded. It never asks a cold unit to load (that wakes a GPU).
* ``ocr``        -- local ``tesseract`` TSV, words grouped into line boxes.
* ``blobs``      -- pure PIL: background colour, difference mask, dilate, connected
  components, refined at full resolution.
* ``grid``       -- an adaptive grid; empty cells dropped, the rest shrunk to their ink.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import List, Optional, Protocol, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

from .state import Region

DEFAULT_OMNIPARSER_URL = "https://127.0.0.1:8191"

_OMNI_ROLE = {"button": "button", "input": "input", "textbox": "input", "link": "link",
              "icon": "icon", "text": "text", "image": "image"}


class Proposer(Protocol):
    name: str

    def available(self) -> Tuple[bool, str]: ...

    def propose(self, img: Image.Image) -> List[Region]: ...


def _rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def dominant_background(img: Image.Image, sample: int = 160) -> Tuple[int, int, int]:
    """The most frequent colour of a downsampled copy: the screen's background."""
    small = _rgb(img).copy()
    small.thumbnail((sample, sample), Image.NEAREST)
    colors = small.getcolors(maxcolors=small.width * small.height + 1) or [(1, (0, 0, 0))]
    return max(colors, key=lambda c: c[0])[1]


def ink_mask(img: Image.Image, bg: Tuple[int, int, int], threshold: int = 24) -> Image.Image:
    """L-mode mask, 255 where ANY channel differs from ``bg`` by more than ``threshold``."""
    rgb = _rgb(img)
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg))
    r, g, b = diff.split()
    mx = ImageChops.lighter(ImageChops.lighter(r, g), b)
    return mx.point(lambda v: 255 if v > threshold else 0)


def _components(mask: Image.Image) -> List[Tuple[int, int, int, int]]:
    """4-connected components of a binary L mask, as inclusive (x1, y1, x2, y2)."""
    w, h = mask.size
    data = bytearray(mask.convert("L").tobytes())   # nonzero = ink; getdata() is deprecated
    out = []
    for start in range(w * h):
        if not data[start]:
            continue
        data[start] = 0
        q = deque([start])
        x1 = x2 = start % w
        y1 = y2 = start // w
        while q:
            i = q.popleft()
            x, y = i % w, i // w
            if x < x1:
                x1 = x
            elif x > x2:
                x2 = x
            if y < y1:
                y1 = y
            elif y > y2:
                y2 = y
            if x > 0 and data[i - 1]:
                data[i - 1] = 0
                q.append(i - 1)
            if x < w - 1 and data[i + 1]:
                data[i + 1] = 0
                q.append(i + 1)
            if y > 0 and data[i - w]:
                data[i - w] = 0
                q.append(i - w)
            if y < h - 1 and data[i + w]:
                data[i + w] = 0
                q.append(i + w)
        out.append((x1, y1, x2, y2))
    return out


class BlobProposer:
    """Connected ink components against the dominant background. No model, no numpy."""

    name = "blobs"

    def __init__(self, downsample: int = 4, dilate: int = 3, min_area: int = 64,
                 threshold: int = 16) -> None:
        self.downsample = max(1, int(downsample))
        self.dilate = max(0, int(dilate))
        self.min_area = int(min_area)
        self.threshold = int(threshold)

    def available(self) -> Tuple[bool, str]:
        return True, "pure PIL"

    #: A blob covering more than this fraction of the frame is a WINDOW or panel, not a
    #: control. Its frame lines connect everything inside it, so it is re-split by an
    #: XY-cut (blank row / column gaps) against its OWN background: a dark app on a black
    #: desktop was one blob at the top level, measured on a real 2175x1360 capture.
    split_frac = 0.08
    row_gap = 4
    col_gap = 12

    #: How many times a still-large leaf is re-cut against its own local background
    #: (overlapping windows nest: desktop -> window -> panel).
    max_depth = 3

    def propose(self, img: Image.Image) -> List[Region]:
        img = _rgb(img)
        frame_area = img.width * img.height
        out: List[Region] = []
        for r in self._components_of(img):
            if self._big(r.bbox, frame_area, 0):
                leaves = self._split(img, r.bbox, frame_area, 1)
                if len(leaves) >= 2:
                    out.extend(Region(id="", proposer=self.name, bbox=b) for b in leaves)
                    continue
            out.append(r)
        return out

    def _big(self, b, frame_area: int, depth: int) -> bool:
        # Each level halves the bar, so a panel inside a window is still re-cut.
        frac = self.split_frac / (2 ** depth)
        return (b[2] - b[0]) * (b[3] - b[1]) > frac * frame_area

    def _split(self, img: Image.Image, box, frame_area: int, depth: int):
        """XY-cut ``box`` against its own background; recurse into still-large leaves."""
        x1, y1 = box[0], box[1]
        sub = img.crop(box)
        bg = dominant_background(sub)
        mask = ink_mask(sub, bg, self.threshold)
        raw: List[Tuple[int, int, int, int]] = []
        self._xycut(mask, (0, 0, sub.width, sub.height), 0, 0, raw)
        leaves = [(b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1) for b in raw
                  if (b[2] - b[0]) * (b[3] - b[1]) >= self.min_area]
        if len(leaves) < 2 and depth > 1:
            return leaves or [tuple(box)]
        out = []
        for b in leaves:
            if self._big(b, frame_area, depth):
                inner = (self._split(img, b, frame_area, depth + 1)
                         if depth < self.max_depth and tuple(b) != tuple(box) else [b])
                if len(inner) >= 2:
                    out.extend(inner)
                    continue
                if inner and tuple(inner[0]) != tuple(b):
                    # _split located real content inside b; that IS the element, never
                    # a tile grid over it.
                    out.append(inner[0])
                    continue
                tiles = self._tile(img, b, bg)
                if len(tiles) >= 2:
                    out.extend(tiles)
                    continue
                out.append(b)
            else:
                out.extend(self._recut_local(img, b, bg))
        return out

    def _recut_local(self, img: Image.Image, box, parent_bg) -> List[Tuple[int, int, int, int]]:
        """Re-cut a flat leaf (a banner strip) against its OWN background.

        A banner shares the desktop's ink mask, so the pane-level XY-cut cannot see the
        controls sitting on it (the button fill can differ from the banner by only the
        threshold). When the leaf's background is materially different from the pane's,
        an XY-cut against the leaf's own background finds them. A single compact control
        (its visible text is one leaf) is returned whole, never carved into glyphs.
        """
        sub = img.crop(box)
        lbg = dominant_background(sub)
        if max(abs(int(a) - int(b)) for a, b in zip(lbg, parent_bg)) <= self.threshold:
            return [tuple(box)]
        mask = ink_mask(sub, lbg, self.threshold)
        raw: List[Tuple[int, int, int, int]] = []
        self._xycut(mask, (0, 0, sub.width, sub.height), 0, 0, raw)
        leaves = [(b[0] + box[0], b[1] + box[1], b[2] + box[0], b[3] + box[1]) for b in raw
                  if (b[2] - b[0]) * (b[3] - b[1]) >= self.min_area]
        return leaves if len(leaves) >= 2 else [tuple(box)]

    #: Side of the tiles a panel that no gap can split is cut into (overlapping filled
    #: windows leave no blank row or column). Each tile shrinks to its own ink.
    tile_px = 160

    def _tile(self, img: Image.Image, box, bg) -> List[Tuple[int, int, int, int]]:
        """Cut a gap-less panel into tiles, each shrunk to its ink against the PANEL's
        background. The background is the pane's, never the cell's own dominant colour:
        a control filling most of a tile would otherwise BE that tile's background and
        the tile would report the blank around it as the element. A near-uniform block
        is one thing, not a grid: tiling it would only emit 160-px-aligned boxes that
        are not tight under the module's own ink mask."""
        sub = img.crop(box)
        colors = sub.copy()
        colors.thumbnail((128, 128), Image.NEAREST)
        counts = colors.getcolors(maxcolors=colors.width * colors.height + 1) or []
        total = max(1, colors.width * colors.height)
        if counts and max(c for c, _ in counts) / float(total) > 0.85:
            return [tuple(box)]
        out = []
        for y in range(box[1], box[3], self.tile_px):
            for x in range(box[0], box[2], self.tile_px):
                c = (x, y, min(box[2], x + self.tile_px), min(box[3], y + self.tile_px))
                cell = img.crop(c)
                bb = ink_mask(cell, bg, self.threshold).getbbox()
                if bb and (bb[2] - bb[0]) * (bb[3] - bb[1]) >= self.min_area:
                    out.append((c[0] + bb[0], c[1] + bb[1], c[0] + bb[2], c[1] + bb[3]))
        return out

    @staticmethod
    def _runs(profile, gap: int) -> List[Tuple[int, int]]:
        """Inked runs of a projection profile, split where ``gap`` blank cells occur."""
        runs: List[Tuple[int, int]] = []
        first, blank = -1, 0
        for i, v in enumerate(profile):
            if v:
                if first < 0:
                    first = i
                blank = 0
            elif first >= 0:
                blank += 1
                if blank >= gap:
                    runs.append((first, i - blank + 1))
                    first, blank = -1, 0
        if first >= 0:
            runs.append((first, len(profile) - blank))
        return runs

    @staticmethod
    def _deruled(sub: Image.Image, cover: float = 0.6) -> Image.Image:
        """The mask with frame rules erased: a column (row) inked over more than
        ``cover`` of its length is a border line, and one border line on each side
        makes every row (column) look inked, so an XY-cut finds no gap at all."""
        w, h = sub.size
        cols = sub.resize((w, 1), Image.BOX).tobytes()
        rows = sub.resize((1, h), Image.BOX).tobytes()
        lim = int(255 * cover)

        def thin(profile, max_run=3):
            # Only THIN runs are rules; a thick covered band is a filled panel, and
            # erasing it would erase whatever sits beside it on the same rows.
            hits = [i for i, v in enumerate(profile) if v > lim]
            out_, run = [], []
            for i in hits + [-2]:
                if run and i != run[-1] + 1:
                    if len(run) <= max_run:
                        out_.extend(run)
                    run = []
                if i >= 0:
                    run.append(i)
            return out_

        rule_cols, rule_rows = thin(cols), thin(rows)
        if not rule_cols and not rule_rows:
            return sub
        out = sub.copy()
        d = ImageDraw.Draw(out)
        for i in rule_cols:
            d.line([(i, 0), (i, h - 1)], fill=0)
        for j in rule_rows:
            d.line([(0, j), (w - 1, j)], fill=0)
        return out

    def _xycut(self, mask: Image.Image, box, axis: int, depth: int, out: list) -> None:
        bb = mask.crop(box).getbbox()
        if not bb:
            return
        box = (box[0] + bb[0], box[1] + bb[1], box[0] + bb[2], box[1] + bb[3])
        if depth > 16:
            out.append(box)
            return
        deruled = self._deruled(mask.crop(box))
        xp, yp = deruled.getprojection()
        for ax in (axis, 1 - axis):
            runs = self._runs(yp if ax == 0 else xp, self.row_gap if ax == 0 else self.col_gap)
            if len(runs) > 1:
                for a, b in runs:
                    child = ((box[0], box[1] + a, box[2], box[1] + b) if ax == 0
                             else (box[0] + a, box[1], box[0] + b, box[3]))
                    self._xycut(mask, child, 1 - ax, depth + 1, out)
                return
        # No gap: return the CONTENT extent (frame rules erased), never the raw-mask
        # extent -- a border line shared with the parent pane would otherwise widen a
        # leaf to the whole pane (a flat control returned as its full row band).
        cb = deruled.getbbox()
        if cb:
            box = (box[0] + cb[0], box[1] + cb[1], box[0] + cb[2], box[1] + cb[3])
        out.append(box)

    def _components_of(self, img: Image.Image) -> List[Region]:
        img_w, img_h = img.size
        d = self.downsample
        full = ink_mask(img, dominant_background(img), self.threshold)
        sw, sh = max(1, math.ceil(img_w / d)), max(1, math.ceil(img_h / d))
        small = full.resize((sw, sh), Image.BOX).point(lambda v: 255 if v > 0 else 0)
        if self.dilate:
            small = small.filter(ImageFilter.MaxFilter(2 * self.dilate + 1))
        out: List[Region] = []
        for (x1, y1, x2, y2) in _components(small):
            box = (max(0, (x1 - 1) * d), max(0, (y1 - 1) * d),
                   min(img_w, (x2 + 2) * d), min(img_h, (y2 + 2) * d))
            bb = full.crop(box).getbbox()
            if not bb:
                continue
            rb = (box[0] + bb[0], box[1] + bb[1], box[0] + bb[2], box[1] + bb[3])
            if (rb[2] - rb[0]) * (rb[3] - rb[1]) < self.min_area:
                continue
            out.append(Region(id="", bbox=rb, proposer=self.name))
        return out


class AdaptiveGridProposer:
    """Tile the frame into about ``target_cells`` cells; keep the inked ones, shrunk."""

    name = "grid"

    def __init__(self, target_cells: int = 24, min_edge_density: float = 0.004) -> None:
        self.target_cells = max(1, int(target_cells))
        self.min_edge_density = float(min_edge_density)

    def available(self) -> Tuple[bool, str]:
        return True, "pure PIL"

    def cells(self, size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """An exact tiling: every pixel in exactly one cell."""
        img_w, img_h = size
        s = math.sqrt(img_w * img_h / float(self.target_cells))
        cols, rows = max(1, round(img_w / s)), max(1, round(img_h / s))
        xs = [round(i * img_w / cols) for i in range(cols + 1)]
        ys = [round(j * img_h / rows) for j in range(rows + 1)]
        return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(rows) for i in range(cols)]

    def propose(self, img: Image.Image) -> List[Region]:
        img = _rgb(img)
        bg = dominant_background(img)
        mask = ink_mask(img, bg)
        edges = img.convert("L").filter(ImageFilter.FIND_EDGES).point(
            lambda v: 255 if v > 32 else 0)
        out = []
        for c in self.cells(img.size):
            density = ImageStat.Stat(edges.crop(c)).mean[0] / 255.0
            if density < self.min_edge_density:
                continue
            bb = mask.crop(c).getbbox()
            if not bb:
                continue
            out.append(Region(id="", bbox=(c[0] + bb[0], c[1] + bb[1], c[0] + bb[2],
                                           c[1] + bb[3]), proposer=self.name))
        return out


class TesseractTsvProposer:
    """Local tesseract, TSV output, words grouped into line boxes. Text is the hint."""

    name = "ocr"

    def __init__(self, binary: Optional[str] = None, timeout_s: float = 60.0) -> None:
        self.binary = binary or shutil.which("tesseract")
        self.timeout_s = timeout_s

    def available(self) -> Tuple[bool, str]:
        if not self.binary:
            return False, "unavailable: no tesseract on PATH"
        return True, self.binary

    @staticmethod
    def parse_tsv(tsv: str) -> List[Region]:
        lines: dict = {}
        rows = tsv.splitlines()
        if not rows:
            return []
        head = rows[0].split("\t")
        for row in rows[1:]:
            cols = row.split("\t")
            if len(cols) != len(head):
                continue
            rec = dict(zip(head, cols))
            text = (rec.get("text") or "").strip()
            try:
                conf = float(rec.get("conf") or -1)
                x, y = int(rec["left"]), int(rec["top"])
                w, h = int(rec["width"]), int(rec["height"])
            except (KeyError, ValueError):
                continue
            if not text or conf < 0:
                continue
            key = (rec.get("block_num"), rec.get("par_num"), rec.get("line_num"))
            ent = lines.setdefault(key, {"box": [x, y, x + w, y + h], "words": [], "conf": []})
            b = ent["box"]
            ent["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
            ent["words"].append(text)
            ent["conf"].append(conf)
        out = []
        for ent in lines.values():
            txt = " ".join(ent["words"])
            out.append(Region(id="", bbox=tuple(ent["box"]), proposer="ocr",
                              hint={"text": txt, "label": txt, "role": "text",
                                    "confidence": round(sum(ent["conf"]) / len(ent["conf"])
                                                        / 100.0, 3)}))
        return out

    def _tsv_via_aithervision(self, img: Image.Image) -> Optional[str]:
        """TSV from ``services.perception.AitherVision`` when this tree has it.

        The fleet keeps ONE tesseract-TSV implementation (AitherVision); a published
        brick may be installed without the monorepo, so a missing module is not an
        error -- it falls through to the local call below. Returning None means "no
        shared implementation here", never "the screen has no text".
        """
        try:
            from services.perception.AitherVision import ocr_with_tesseract
        except Exception:  # noqa: BLE001 - absence is expected outside the monorepo
            return None
        import tempfile
        fd, name = tempfile.mkstemp(prefix="awscreen-ocr-", suffix=".png")
        os.close(fd)
        try:
            _rgb(img).save(name, "PNG")
            res = ocr_with_tesseract(name, tsv=True)
        except Exception:  # noqa: BLE001 - an unusable shared lane falls through
            return None
        finally:
            with contextlib.suppress(OSError):   # best-effort temp cleanup
                os.unlink(name)
        if not getattr(res, "success", False):
            return None
        text = getattr(res, "analysis", "") or ""
        return text if text.strip() else None

    def propose(self, img: Image.Image) -> List[Region]:
        shared = self._tsv_via_aithervision(img)
        if shared is not None:
            return self.parse_tsv(shared)
        buf = io.BytesIO()
        _rgb(img).save(buf, "PNG")
        # This brick also runs from a detached, console-less Windows parent; a
        # tesseract child without CREATE_NO_WINDOW opens a terminal tab that steals
        # focus. Same documented pattern as AitherOS/dev/tools.
        proc = subprocess.run([self.binary, "stdin", "stdout", "tsv"], input=buf.getvalue(),
                              capture_output=True, timeout=self.timeout_s, check=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode != 0:
            raise RuntimeError("tesseract exit %d: %s" % (
                proc.returncode, proc.stderr.decode("utf-8", "replace")[:200]))
        return self.parse_tsv(proc.stdout.decode("utf-8", "replace"))


def normalize_omniparser(resp: dict, width: int, height: int) -> List[Region]:
    """Both shapes (the service's normalized ``bbox`` + ``bbox_px``, and the py312
    server's ``bbox_px`` only) become image-pixel Regions with label/role hints."""
    out = []
    for el in (resp or {}).get("elements") or []:
        px = el.get("bbox_px") or []
        if len(px) == 4:
            box = tuple(int(round(float(v))) for v in px)
        else:
            nb = el.get("bbox") or []
            if len(nb) != 4:
                continue
            box = (int(round(nb[0] * width)), int(round(nb[1] * height)),
                   int(round(nb[2] * width)), int(round(nb[3] * height)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        etype = str(el.get("element_type") or "").lower()
        text = str(el.get("ocr_text") or (el.get("label") if etype == "text" else "") or "")
        label = str(el.get("label") or el.get("caption") or text)
        out.append(Region(id="", bbox=box, proposer="omniparser", hint={
            "label": label, "text": text, "role": _OMNI_ROLE.get(etype, "other"),
            "confidence": float(el.get("confidence") or 0.5),
            "interactable": bool(el.get("interactable", etype != "text"))}))
    return out


class OmniParserProposer:
    """OmniParser over HTTPS, only when it is already warm. Never wakes the unit."""

    name = "omniparser"

    def __init__(self, url: Optional[str] = None, probe_timeout: float = 2.0,
                 parse_timeout: float = 30.0) -> None:
        self.url = (url or os.getenv("AWSCREEN_OMNIPARSER_URL")
                    or DEFAULT_OMNIPARSER_URL).rstrip("/")
        self.probe_timeout = probe_timeout
        self.parse_timeout = parse_timeout

    def _open(self, req, timeout):
        from urllib.request import urlopen
        if self.url.startswith("https://"):
            try:
                from awvision.vision import ssl_context
                ctx = ssl_context()
            except ImportError:
                import ssl
                ctx = ssl.create_default_context()
            return urlopen(req, timeout=timeout, context=ctx)
        return urlopen(req, timeout=timeout)

    def _health(self) -> dict:
        with self._open(self.url + "/health", self.probe_timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")

    def available(self) -> Tuple[bool, str]:
        # A wall-clock bound, not just a socket one: connect + TLS + read each get the
        # socket timeout, which let a filtered port hold the chain for ~4 s.
        box: dict = {}

        def _probe():
            try:
                box["body"] = self._health()
            except Exception as exc:  # noqa: BLE001 - any failure means "not here"
                box["exc"] = exc

        th = threading.Thread(target=_probe, daemon=True, name="awscreen-omni-probe")
        th.start()
        th.join(self.probe_timeout)
        if th.is_alive():
            return False, (f"health probe {self.url}/health: no answer within "
                           f"{self.probe_timeout:g}s")
        if "exc" in box:
            exc = box["exc"]
            return False, f"health probe {self.url}/health failed: {type(exc).__name__}: {exc}"
        body = box.get("body") or {}
        loaded = body.get("model_loaded", body.get("models_loaded"))
        if loaded is False:
            return False, "health ok but models not loaded; parsing would wake the GPU unit"
        return True, "health ok"

    def propose(self, img: Image.Image) -> List[Region]:
        from urllib.request import Request
        buf = io.BytesIO()
        _rgb(img).save(buf, "PNG")
        payload = {"image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                   "include_captions": True, "include_ocr": True}
        req = Request(self.url + "/parse", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        with self._open(req, self.parse_timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        if not body.get("success", True):
            raise RuntimeError(f"omniparser: {body.get('error') or 'parse failed'}")
        return normalize_omniparser(body, img.width, img.height)


_FACTORY = {"omniparser": OmniParserProposer, "ocr": TesseractTsvProposer,
            "blobs": BlobProposer, "grid": AdaptiveGridProposer}


def make_proposer(name):
    """A proposer by name, or the object itself when one is passed."""
    if not isinstance(name, str):
        return name
    if name not in _FACTORY:
        raise ValueError(f"unknown proposer {name!r} (one of {sorted(_FACTORY)})")
    return _FACTORY[name]()


def propose(img: Image.Image, chain: Sequence) -> Tuple[List[Region], dict]:
    """Run the chain until one proposer yields regions. Returns (regions, report)."""
    tried = []
    for p in [make_proposer(c) for c in chain]:
        t0 = time.perf_counter()
        ok, why = p.available()
        if not ok:
            tried.append({"name": p.name, "ok": False, "reason": why,
                          "ms": round((time.perf_counter() - t0) * 1000, 1), "n": 0})
            continue
        try:
            regs = p.propose(img)
        except Exception as exc:  # noqa: BLE001 - one broken proposer falls through
            tried.append({"name": p.name, "ok": False,
                          "reason": f"{type(exc).__name__}: {exc}"[:300],
                          "ms": round((time.perf_counter() - t0) * 1000, 1), "n": 0})
            continue
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if not regs:
            tried.append({"name": p.name, "ok": False, "reason": "no regions", "ms": ms,
                          "n": 0})
            continue
        regs.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        for i, r in enumerate(regs):
            r.id = f"r{i}"
        tried.append({"name": p.name, "ok": True, "reason": why, "ms": ms, "n": len(regs)})
        return regs, {"ran": p.name, "tried": tried}
    return [], {"ran": "", "tried": tried}
