"""GAP 5: region-level reuse across frames.

A frame with ten regions where six are untouched must reuse exactly those six (zero
model calls) and send only the four changed regions to the (stubbed) model. Offline: the
proposer and the inspector are injected, no network, no real model.
"""
import sys
from pathlib import Path

# Import the worktree copies (not an editable install elsewhere), even under --noconftest.
_PKGS = Path(__file__).resolve().parents[2]
for _n in ("awrun", "awgym", "awpredict", "awvision", "awscreen"):
    _p = _PKGS / _n
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from awscreen.inspect import InspectResult  # noqa: E402
from awscreen.state import Region, build_screen_state  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

BG = (243, 244, 246)
GREY = (107, 114, 128)
BLUE = (37, 99, 235)


def _boxes():
    """Ten well-spaced 160x100 controls on a 1280x800 frame."""
    out = []
    for i in range(10):
        col, row = i % 5, i // 5
        x1, y1 = 40 + col * 240, 60 + row * 340
        out.append((x1, y1, x1 + 160, y1 + 100))
    return out


def _frame(blue: set) -> Image.Image:
    img = Image.new("RGB", (1280, 800), BG)
    d = ImageDraw.Draw(img)
    for i, b in enumerate(_boxes()):
        d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1],
                    fill=(BLUE if i in blue else GREY), outline=(17, 24, 39), width=2)
        # Ink that survives the fill change, so neurons see a real control.
        d.line([(b[0] + 8, b[1] + 8), (b[2] - 9, b[3] - 9)], fill=(255, 255, 255), width=2)
        d.text((b[0] + 12, b[1] + 34), f"B{i}", fill=(255, 255, 255))
    return img


class _StubProposer:
    """Returns the same ten boxes every frame: geometry is not what this test varies."""

    name = "stub"

    def __init__(self, boxes):
        self.boxes = boxes

    def available(self):
        return True, "stub"

    def propose(self, img):
        return [Region(id="", bbox=tuple(b), proposer=self.name) for b in self.boxes]


def _counter_inspector():
    calls = []

    def insp(task):
        calls.append(task.region_id)
        return InspectResult(region_id=task.region_id, status="ok", role="button",
                             label="b", text="", confidence=0.7, ms=50.0,
                             raw='{"role":"button","label":"b","text":""}')

    insp.endpoint_host = "stub"
    insp.model = "stub"
    return insp, calls


def _build(path, blue, inspector, prev=None):
    return build_screen_state(
        str(path), proposers=(_StubProposer(_boxes()),), inspector=inspector, prev=prev)


def test_six_of_ten_untouched_reuses_six_and_calls_model_for_four(tmp_path):
    p1, p2 = tmp_path / "f1.png", tmp_path / "f2.png"
    _frame(set()).save(p1)
    _frame({2, 4, 7, 9}).save(p2)             # exactly four regions changed

    insp1, _ = _counter_inspector()
    st1 = _build(p1, set(), insp1)
    assert len(st1.regions) == 10
    assert st1.reused_regions == 0

    insp2, calls2 = _counter_inspector()
    st2 = build_screen_state(
        str(p2), proposers=(_StubProposer(_boxes()),), inspector=insp2, prev=st1)

    # Six unchanged regions reused, invisible to the model.
    assert st2.reused_regions == 6
    assert st2.timing["reused_regions"] == 6
    reused = [d for d in st2.timing["reuse"]["regions"] if d["reused"]]
    assert len(reused) == 6
    # Only the four changed regions reached the (stubbed) model.
    assert len(calls2) == 4, calls2
    assert st2.model["calls"] == 4
    # Per-region time saved is carried in the timing block (50 ms x 6 from the stub).
    assert st2.timing["reused_ms_saved"] == 300.0
    # Every region and element survives: nothing the model never saw was dropped.
    assert len(st2.regions) == 10
    assert len(st2.elements) == 10
    # The whole frame was NOT reused: `reused` stays the full-frame signal.
    assert st2.reused is False


def test_whole_frame_identical_still_reports_every_reused_region(tmp_path):
    p = tmp_path / "f.png"
    _frame(set()).save(p)
    insp1, _ = _counter_inspector()
    st1 = _build(p, set(), insp1)
    insp2, calls2 = _counter_inspector()
    st2 = build_screen_state(
        str(p), proposers=(_StubProposer(_boxes()),), inspector=insp2, prev=st1)
    assert st2.reused is True
    assert st2.reused_regions == 10
    assert calls2 == [] and st2.model["calls"] == 0
    assert len(st2.elements) == len(st1.elements)


def test_partial_reuse_state_is_complete_for_the_next_frame(tmp_path):
    """After a partial reuse the state still carries all ten regions (with signatures),
    so a third, identical frame reuses the whole thing with zero calls."""
    p1, p2, p3 = tmp_path / "f1.png", tmp_path / "f2.png", tmp_path / "f3.png"
    _frame(set()).save(p1)
    _frame({2, 4, 7, 9}).save(p2)
    _frame({2, 4, 7, 9}).save(p3)             # p3 == p2

    insp1, _ = _counter_inspector()
    st1 = _build(p1, set(), insp1)
    insp2, _ = _counter_inspector()
    st2 = build_screen_state(
        str(p2), proposers=(_StubProposer(_boxes()),), inspector=insp2, prev=st1)
    assert st2.reused_regions == 6 and len(st2.regions) == 10
    assert all(r.sig for r in st2.regions)

    insp3, calls3 = _counter_inspector()
    st3 = build_screen_state(
        str(p3), proposers=(_StubProposer(_boxes()),), inspector=insp3, prev=st2)
    assert st3.reused is True and st3.reused_regions == 10
    assert calls3 == [] and st3.model["calls"] == 0


def test_reused_element_bboxes_rebase_with_a_new_origin(tmp_path):
    """Reused elements are ABSOLUTE screen px; a different --origin shifts them."""
    p1, p2 = tmp_path / "f1.png", tmp_path / "f2.png"
    _frame(set()).save(p1)
    _frame({2, 4, 7, 9}).save(p2)

    insp1, _ = _counter_inspector()
    st1 = build_screen_state(str(p1), proposers=(_StubProposer(_boxes()),), inspector=insp1)
    insp2, _ = _counter_inspector()
    st2 = build_screen_state(str(p2), proposers=(_StubProposer(_boxes()),), inspector=insp2,
                             prev=st1, origin=(1920, 0))
    assert st2.reused_regions == 6
    assert st2.frame.origin == (1920, 0)
    # Every element — reused or freshly inspected — is reported in ABSOLUTE screen px.
    assert st2.elements
    for e in st2.elements:
        assert e.bbox[0] == e.bbox_image[0] + 1920
        assert e.bbox[1] == e.bbox_image[1]
