"""Region proposers: blobs, grid, the fall-through chain, tesseract, OmniParser shapes."""
import sys
from pathlib import Path

# Import the worktree copies (not an editable install elsewhere), even under --noconftest.
_PKGS = Path(__file__).resolve().parents[2]
for _n in ("awrun", "awgym", "awpredict", "awvision", "awscreen"):
    _p = _PKGS / _n
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import types  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from _synthetic import draw_screen  # noqa: E402
from awscreen import regions  # noqa: E402

_TSV = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\t"
        "height\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t90\tHello\n"
        "5\t1\t1\t1\t1\t2\t45\t21\t40\t12\t80\tworld\n")


def _close(a, b, tol=2):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def test_blobs_find_every_truth_box():
    img, truth = draw_screen()
    boxes = [r.bbox for r in regions.BlobProposer().propose(img)]
    for name, t in truth.items():
        assert any(_close(b, t) for b in boxes), (name, t, boxes)


def test_grid_tiles_without_gaps_or_overlaps():
    img, _ = draw_screen()
    g = regions.AdaptiveGridProposer(target_cells=24)
    cells = g.cells(img.size)
    seen = {}
    for c in cells:
        for x in range(c[0], c[2], 37):
            for y in range(c[1], c[3], 37):
                assert (x, y) not in seen
                seen[(x, y)] = c
    assert sum((c[2] - c[0]) * (c[3] - c[1]) for c in cells) == img.width * img.height
    assert min(c[0] for c in cells) == 0 and max(c[2] for c in cells) == img.width
    regs = g.propose(img)
    assert regs and all(r.proposer == "grid" for r in regs)


def test_chain_falls_through_when_omniparser_down(monkeypatch):
    img, _ = draw_screen()
    monkeypatch.setattr(regions.OmniParserProposer, "_health",
                        lambda self: (_ for _ in ()).throw(OSError("refused")))
    regs, rep = regions.propose(img, ["omniparser", "ocr", "blobs", "grid"])
    assert rep["ran"] == "blobs" and regs
    assert rep["tried"][0]["name"] == "omniparser" and not rep["tried"][0]["ok"]
    assert "health probe" in rep["tried"][0]["reason"]
    assert [t["name"] for t in rep["tried"]][-1] == "blobs"


def test_omniparser_cold_is_not_woken(monkeypatch):
    monkeypatch.setattr(regions.OmniParserProposer, "_health",
                        lambda self: {"status": "healthy", "model_loaded": False})
    ok, why = regions.OmniParserProposer().available()
    assert not ok and "wake" in why


def test_tesseract_absent_is_reported(monkeypatch):
    monkeypatch.setattr(regions.shutil, "which", lambda name: None)
    ok, why = regions.TesseractTsvProposer().available()
    assert not ok and "tesseract" in why


def test_tesseract_tsv_groups_words_into_lines():
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\t"
           "height\tconf\ttext\n"
           "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t90\tHello\n"
           "5\t1\t1\t1\t1\t2\t45\t21\t40\t12\t80\tworld\n"
           "5\t1\t1\t1\t2\t1\t10\t50\t20\t12\t-1\t\n")
    regs = regions.TesseractTsvProposer.parse_tsv(tsv)
    assert len(regs) == 1
    assert regs[0].bbox == (10, 20, 85, 33) and regs[0].hint["text"] == "Hello world"


@pytest.mark.skipif(not regions.shutil.which("tesseract"), reason="no tesseract on PATH")
def test_tesseract_live_reads_synthetic():
    img, _ = draw_screen()
    regs = regions.TesseractTsvProposer().propose(img)
    assert any("Hello" in r.hint.get("text", "") for r in regs)


def test_tesseract_uses_the_shared_aithervision_tsv(monkeypatch):
    """With the monorepo present, the brick asks AitherVision for TSV instead of
    spawning its own tesseract -- one OCR implementation, not two."""
    fake_av = types.ModuleType("services.perception.AitherVision")
    fake_av.ocr_with_tesseract = lambda path, tsv=False: SimpleNamespace(
        success=True, analysis=_TSV)
    services = types.ModuleType("services")
    perception = types.ModuleType("services.perception")
    services.perception = perception
    perception.AitherVision = fake_av
    monkeypatch.setitem(sys.modules, "services", services)
    monkeypatch.setitem(sys.modules, "services.perception", perception)
    monkeypatch.setitem(sys.modules, "services.perception.AitherVision", fake_av)

    def _boom(*a, **k):
        raise AssertionError("spawned tesseract instead of using the shared TSV lane")

    monkeypatch.setattr(regions.subprocess, "run", _boom)
    img, _ = draw_screen()
    regs = regions.TesseractTsvProposer(binary="tesseract-not-real").propose(img)
    assert [r.hint["text"] for r in regs] == ["Hello world"]


def test_tesseract_spawn_is_consoleless(monkeypatch):
    """A detached parent must not get a terminal tab per OCR call."""
    monkeypatch.setattr(regions.TesseractTsvProposer, "_tsv_via_aithervision",
                        lambda self, img: None, raising=False)
    seen = {}

    class Proc:
        returncode = 0
        stdout = _TSV.encode("utf-8")
        stderr = b""

    def fake_run(argv, **kw):
        seen.update(kw)
        return Proc()

    monkeypatch.setattr(regions.subprocess, "run", fake_run)
    img, _ = draw_screen()
    regions.TesseractTsvProposer(binary="tesseract").propose(img)
    import subprocess as _sp
    assert seen.get("creationflags") == getattr(_sp, "CREATE_NO_WINDOW", 0)


def test_omniparser_normalizer_handles_both_shapes():
    service = {"elements": [{"element_type": "button", "label": "Save",
                             "bbox": [0.1, 0.2, 0.3, 0.4], "bbox_px": [], "confidence": 0.8}]}
    py312 = {"elements": [{"element_type": "text", "label": "Hello",
                           "bbox_px": [5, 6, 50, 20], "interactable": False}]}
    a = regions.normalize_omniparser(service, 1000, 500)
    b = regions.normalize_omniparser(py312, 1000, 500)
    assert a[0].bbox == (100, 100, 300, 200) and a[0].hint["role"] == "button"
    assert b[0].bbox == (5, 6, 50, 20) and b[0].hint["role"] == "text"
    assert b[0].hint["text"] == "Hello" and b[0].hint["interactable"] is False


def test_dark_window_on_black_desktop_is_re_split():
    # One window frame connects everything inside it; the top-level blob is the whole
    # window and must be XY-cut against the window's own background.
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1600, 1000), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 1299, 899], fill=(12, 12, 12), outline=(90, 90, 90))
    d.rectangle([100, 100, 1299, 140], fill=(40, 40, 40))           # title bar
    d.rectangle([200, 300, 359, 347], fill=(63, 63, 63))           # a button
    d.text((240, 315), "ignite", fill=(255, 255, 255))
    d.rectangle([700, 600, 899, 640], fill=(181, 181, 186))        # a light pill
    regs = regions.BlobProposer().propose(img)
    boxes = [r.bbox for r in regs]
    assert any(_close(b, (200, 300, 360, 348)) for b in boxes), boxes
    assert any(_close(b, (700, 600, 900, 641)) for b in boxes), boxes
    assert not any(b[2] - b[0] > 1100 and b[3] - b[1] > 700 for b in boxes)


def test_flat_control_in_a_framed_pane_is_not_the_whole_row_band():
    """A control inside a framed window shares the frame's ink mask, which spans the
    pane. The returned leaf must be trimmed to the control: a box spanning the full pane
    width has its centre on window background, not on the control (the verifier's 700x260
    repro returned (20,200,700,240) and never the button at [200,200,320,240])."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (700, 260), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 698, 258], fill=(12, 12, 12), outline=(90, 90, 90))
    d.rectangle([20, 20, 698, 60], fill=(40, 40, 40))
    d.rectangle([200, 200, 319, 239], fill=(63, 63, 63))
    boxes = [r.bbox for r in regions.BlobProposer().propose(img)]
    assert any(_close(b, (200, 200, 320, 240)) for b in boxes), boxes
    for b in boxes:
        if b[1] >= 200:
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            assert img.getpixel((cx, cy)) != (12, 12, 12), (b, img.getpixel((cx, cy)))


def test_tile_uses_the_pane_background_not_the_cell():
    """A control filling most of a 160-px tile must not BECOME that tile's background, or
    the tile grid reports the blank around it as a 160-px-aligned element (the verifier's
    (100,100,260,260) background strip on a control at [140,140,540,540])."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1000, 1000), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 899, 899], fill=(12, 12, 12), outline=(90, 90, 90))
    d.rectangle([140, 140, 539, 539], fill=(63, 63, 63))
    boxes = [r.bbox for r in regions.BlobProposer().propose(img)]
    assert any(_close(b, (140, 140, 540, 540)) for b in boxes), boxes
    assert (100, 100, 260, 260) not in boxes, boxes
    for b in boxes:
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        assert img.getpixel((cx, cy)) != (12, 12, 12), (b, img.getpixel((cx, cy)))


def test_banner_control_whose_fill_differs_by_the_threshold_is_localized():
    """Live 090619.png: a 'Take over' button on a coloured banner whose fill differs from
    the banner by exactly the ink threshold (channel diff 24). The whole 717-px banner
    used to be returned (IoU 0.13, click centre 225-285 px off); the button must now be
    its own region with its centre on the control."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.load_default(size=18)
    except TypeError:          # Pillow < 10.1
        f = ImageFont.load_default()
    img = Image.new("RGB", (1600, 400), (5, 5, 7))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 1299, 149], fill=(40, 37, 59))               # the banner
    d.text((150, 118), "Notifications are on. Take over any time.", font=f,
           fill=(225, 227, 234))
    button = (1000, 107, 1140, 143)
    d.rectangle([button[0], button[1], button[2] - 1, button[3] - 1], fill=(38, 34, 35))
    d.text((1020, 118), "Take over", font=f, fill=(237, 239, 245))
    boxes = [r.bbox for r in regions.BlobProposer().propose(img)]
    best = max(boxes, key=lambda b: _iou(b, button))
    assert _iou(best, button) >= 0.6, (best, boxes)
    cx, cy = (best[0] + best[2]) // 2, (best[1] + best[3]) // 2
    assert button[0] <= cx <= button[2] and button[1] <= cy <= button[3], (best, (cx, cy))
