"""A latency budget: spend a deadline in order, and name what did not fit. Offline.

The clock is FAKE: every model call advances it by a known amount, so "a 5 s budget on a
20-region screen" is exact arithmetic instead of a flaky race against a real timer.
"""
import sys
import threading
from pathlib import Path

# Import the worktree copies (not an editable install elsewhere), even under --noconftest.
_PKGS = Path(__file__).resolve().parents[2]
for _n in ("awrun", "awgym", "awpredict", "awvision", "awscreen"):
    _p = _PKGS / _n
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from awscreen import state as state_mod  # noqa: E402
from awscreen.budget import Budget, resolve_budget_ms  # noqa: E402
from awscreen.cli import main  # noqa: E402
from awscreen.inspect import InspectResult  # noqa: E402
from awscreen.state import Region, build_screen_state  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


class FakeClock:
    """Seconds since 0, advanced explicitly; thread-safe because waves use a pool."""

    def __init__(self) -> None:
        self.t = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.t

    def advance(self, ms: float) -> None:
        with self._lock:
            self.t += ms / 1000.0


def _screen20():
    """20 distinct coloured controls on a light background: every crop is inked."""
    img = Image.new("RGB", (1200, 900), (243, 244, 246))
    draw = ImageDraw.Draw(img)
    boxes = []
    for i in range(20):
        x = 20 + (i % 4) * 290
        y = 20 + (i // 4) * 170
        b = (x, y, x + 140, y + 48)
        draw.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], fill=(37, 99, 235))
        boxes.append(b)
    return img, boxes


class FixedProposer:
    """Hands back the 20 boxes, so the count is the test's, not a proposer's."""

    name = "fixed20"

    def __init__(self, boxes):
        self.boxes = boxes

    def available(self):
        return True, "fixed"

    def propose(self, img):
        return [Region(id=f"r{i}", bbox=tuple(b), proposer="blobs")
                for i, b in enumerate(self.boxes)]


def _inspector(clock, ms_per_call):
    def insp(task):
        clock.advance(ms_per_call)
        return InspectResult(region_id=task.region_id, status="ok", role="button",
                             label="seen", text="", confidence=0.7, raw='{"role":"button"}')

    insp.endpoint_host = "fake"
    insp.model = "fake"
    return insp


@pytest.fixture
def syn20(tmp_path):
    img, boxes = _screen20()
    p = tmp_path / "screen20.png"
    img.save(p)
    return p, boxes


def _statuses(state):
    return [e.status for e in state.elements]


def test_five_second_budget_on_twenty_regions_inspects_only_what_fits(syn20):
    """5 s, 2 s per crop, one in flight -> exactly 2 model calls; the other 18 are named
    'budget', still present, with no invented label."""
    p, boxes = syn20
    clock = FakeClock()
    st = build_screen_state(
        str(p), proposers=(FixedProposer(boxes),), inspector=_inspector(clock, 2000),
        max_regions=24, max_concurrency=1, budget_ms=5000, call_estimate_ms=2000,
        clock=clock)

    assert st.model["calls"] == 2
    assert st.model["max_regions"] == 24 and st.model["cap"] == 2
    assert st.model["skipped_by_budget"] == 18
    assert st.model["budget"]["budget_ms"] == 5000
    assert 0 <= st.model["budget"]["remaining_ms"] < 2000
    # Never dropped: the neuron element for every region survives.
    assert len(st.elements) == 20
    assert _statuses(st).count("ok") == 2
    assert _statuses(st).count("budget") == 18
    for el in st.elements:
        if el.status == "budget":
            assert el.source != "vlm" and el.label == ""      # never invented
            assert el.evidence["budget"].startswith("not inspected")
    # The state round-trips with the new status.
    again = state_mod.ScreenState.from_dict(st.to_dict())
    assert _statuses(again).count("budget") == 18


def test_deadline_holds_even_when_the_estimate_is_wrong(syn20):
    """Estimate 1 s but each crop really costs 3 s: the wave runner re-reads the clock,
    so only the two that fit run and the rest are 'budget', not silently overrun."""
    p, boxes = syn20
    clock = FakeClock()
    st = build_screen_state(
        str(p), proposers=(FixedProposer(boxes),), inspector=_inspector(clock, 3000),
        max_regions=24, max_concurrency=1, budget_ms=5000, call_estimate_ms=1000,
        clock=clock)

    assert st.model["cap"] == 5                       # the optimistic estimate
    assert st.model["calls"] == 2                     # what the clock actually allowed
    assert st.model["skipped_by_budget"] == 18
    assert len(st.elements) == 20
    assert _statuses(st).count("budget") == 18


def test_zero_budget_makes_no_model_call_and_drops_nothing(syn20):
    p, boxes = syn20
    clock = FakeClock()
    st = build_screen_state(
        str(p), proposers=(FixedProposer(boxes),), inspector=_inspector(clock, 2000),
        max_concurrency=1, budget_ms=0, call_estimate_ms=2000, clock=clock)

    assert st.model["calls"] == 0 and st.model["cap"] == 0
    assert st.model["skipped_by_budget"] == 20
    assert len(st.elements) == 20 and set(_statuses(st)) == {"budget"}


def test_budget_cap_never_exceeds_the_caller_cap(syn20):
    p, boxes = syn20
    clock = FakeClock()
    st = build_screen_state(
        str(p), proposers=(FixedProposer(boxes),), inspector=_inspector(clock, 100),
        max_regions=3, max_concurrency=1, budget_ms=60000, call_estimate_ms=100,
        clock=clock)

    assert st.model["cap"] == 3 and st.model["calls"] == 3
    assert st.model["skipped_by_cap"] == 17
    assert "skipped_by_budget" not in st.model or st.model["skipped_by_budget"] == 0
    assert _statuses(st).count("budget") == 0


def test_budget_arithmetic():
    clock = FakeClock()
    b = Budget(5000, clock)
    assert b.remaining_ms() == 5000 and not b.expired() and b.fits(5000)
    assert b.cap_for(2000, 4) == 8          # two whole waves of four
    clock.advance(3000)
    assert b.remaining_ms() == 2000 and b.fits(2000) and not b.fits(2001)
    assert b.cap_for(2000, 4) == 4
    clock.advance(3000)
    assert b.remaining_ms() == 0 and b.expired() and b.cap_for(2000, 4) == 0


def test_resolve_budget_precedence_and_bad_values(monkeypatch):
    assert resolve_budget_ms(None, {}) is None
    assert resolve_budget_ms(5000, {"AWSCREEN_BUDGET_MS": "7000"}) == 5000.0
    assert resolve_budget_ms(None, {"AWSCREEN_BUDGET_MS": "7000"}) == 7000.0
    assert resolve_budget_ms(0, {"AWSCREEN_BUDGET_MS": "7000"}) == 0.0
    assert resolve_budget_ms(None, {"AWSCREEN_BUDGET_MS": "soon"}) is None
    monkeypatch.setenv("AWSCREEN_BUDGET_MS", "9000")
    assert resolve_budget_ms(None) == 9000.0


def test_cli_wires_the_flag_and_the_env(syn20, monkeypatch, capsys):
    p, _ = syn20
    seen = {}
    real = state_mod.build_screen_state

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(state_mod, "build_screen_state", spy)
    monkeypatch.delenv("AWSCREEN_BUDGET_MS", raising=False)

    assert main(["state", str(p), "--no-inspect", "--proposer", "blobs",
                 "--budget-ms", "5000", "--json"]) == 0
    assert seen["budget_ms"] == 5000.0

    monkeypatch.setenv("AWSCREEN_BUDGET_MS", "7000")
    assert main(["state", str(p), "--no-inspect", "--proposer", "blobs", "--json"]) == 0
    assert seen["budget_ms"] == 7000.0
    capsys.readouterr()
