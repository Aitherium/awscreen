"""IoU merge with source priority, and the DOM join (dpr, fit, scroll, unmatched nodes)."""
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

from _synthetic import draw_screen  # noqa: E402
from awscreen.dom import AITHERBROWSER_RECT_JS, DomSnapshot  # noqa: E402
from awscreen.merge import iou, merge_elements  # noqa: E402
from awscreen.state import Element, build_screen_state  # noqa: E402


def _el(bbox, source, **kw):
    return Element(id="", region_id=kw.pop("rid", "r0"), bbox=list(bbox),
                   bbox_image=list(bbox), source=source, sources=[source], **kw)


def test_iou_basics():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_dedupe_follows_source_priority():
    grid = _el([0, 0, 110, 45], "grid", confidence=0.2)
    cv = _el([1, 1, 101, 41], "cv", role="button", confidence=0.5)
    vlm = _el([0, 0, 100, 40], "vlm", label="Save", text="SAVE", role="button",
              confidence=0.7, rid="r1")
    out = merge_elements([grid, cv, vlm])
    assert len(out) == 1
    m = out[0]
    assert m.source == "vlm" and m.bbox == [0, 0, 100, 40]
    assert set(m.sources) == {"vlm", "cv", "grid"} and m.confidence == 0.7
    assert m.label == "Save" and set(m.evidence["region_ids"]) == {"r0", "r1"}


def test_timeout_elements_are_kept_explicitly():
    ok = _el([0, 0, 100, 40], "cv")
    late = _el([0, 0, 100, 40], "vlm", status="timeout")
    out = merge_elements([ok, late])
    assert len(out) == 2 and {e.status for e in out} == {"ok", "timeout"}


def _state(tmp_path, scale=1, dom=None):
    img, truth = draw_screen(scale=scale)
    p = tmp_path / f"s{scale}.png"
    img.save(p)
    return build_screen_state(str(p), proposers=("blobs",), inspect=False, dom=dom), truth


def test_dom_join_dpr2(tmp_path):
    snap = {"viewport": {"dpr": 2, "vw": 1280, "vh": 800},
            "elements": [{"ref": "#save", "tag": "button", "label": "Save",
                          "rect": [100, 120, 160, 48]}]}
    s, truth = _state(tmp_path, 2, snap)
    hit = [e for e in s.elements if e.dom and e.dom["ref"] == "#save"]
    assert hit and hit[0].dom["iou"] > 0.9
    assert "dom" in hit[0].sources and "cv" in hit[0].sources
    assert hit[0].role == "button" and hit[0].label == "Save"
    assert iou(hit[0].evidence["bbox_pixels"], truth["save"]) > 0.9


def test_fit_and_scroll_mapping():
    fit = DomSnapshot.from_dict({"viewport": {"dpr": 1.5, "vw": 1920, "vh": 1200},
                                 "image_scale": "fit", "elements": []})
    assert fit.css_to_image([150, 180, 240, 72], 1280) == [100, 120, 260, 168]
    scrolled = DomSnapshot.from_dict({"viewport": {"dpr": 1, "vw": 1280, "sx": 50, "sy": 400},
                                      "origin_px": [0, 30], "elements": []})
    assert scrolled.css_to_image([150, 520, 160, 48], 1280, page=True) == [100, 150, 260, 198]
    assert scrolled.css_to_image([150, 520, 160, 48], 1280, page=False) == [150, 550, 310, 598]
    explicit = DomSnapshot.from_dict({"viewport": {"dpr": 2}, "image_scale": 0.5,
                                      "elements": []})
    assert explicit.css_to_image([100, 100, 10, 10], 999) == [50, 50, 55, 55]


def test_fit_mode_join(tmp_path):
    snap = {"viewport": {"dpr": 1, "vw": 1920, "vh": 1200}, "image_scale": "fit",
            "elements": [{"ref": "#cancel", "tag": "button", "label": "Cancel",
                          "rect": [1350, 900, 300, 75]}]}
    s, _ = _state(tmp_path, 1, snap)
    hit = [e for e in s.elements if e.dom and e.dom["ref"] == "#cancel"]
    assert hit and hit[0].dom["iou"] > 0.9


def test_unmatched_dom_node_is_added(tmp_path):
    snap = {"viewport": {"dpr": 1, "vw": 1280},
            "elements": [{"ref": "#menu", "tag": "a", "label": "Menu",
                          "rect": [1000, 20, 80, 20]},
                         {"ref": "#off", "tag": "a", "label": "Off",
                          "rect": [5000, 20, 80, 20]}]}
    s, _ = _state(tmp_path, 1, snap)
    added = [e for e in s.elements if e.source == "dom"]
    assert [e.dom["ref"] for e in added] == ["#menu"]
    assert added[0].role == "link" and added[0].bbox == [1000, 20, 1080, 40]


def test_rect_js_returns_the_snapshot_shape():
    for key in ("viewport", "dpr", "vw", "sx", "sy", "origin_px", "image_scale", "rect"):
        assert key in AITHERBROWSER_RECT_JS
