"""GAP4: monitor-aware capture. Two monitors in memory; the right-hand one clicks true.

The measured failure this file pins: a frame captured by Pillow is the PRIMARY display,
so labelling it monitor 2 puts every element 1280 px off and nothing reports it. Here a
frame is either that monitor's own pixels, verified, or an error.
"""
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

import json  # noqa: E402

import pytest  # noqa: E402
from _synthetic import (  # noqa: E402
    RIGHT_MONITOR_ORIGIN,
    FakeMss,
    _FakeShot,
    draw_screen_right,
    draw_two_monitors,
)
from awscreen import capture  # noqa: E402
from awscreen.crops import make_crop  # noqa: E402
from awscreen.env import ScreenEnv  # noqa: E402
from awscreen.state import Region, build_screen_state  # noqa: E402
from PIL import Image  # noqa: E402

#: Monitor 2's desktop origin and its place-button truth box, in image px.
M2 = RIGHT_MONITOR_ORIGIN
PLACE_IMAGE = [300, 400, 520, 456]


def _close(a, b, tol=3):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


@pytest.fixture
def two_monitors(monkeypatch):
    frames, factory = draw_two_monitors()
    monkeypatch.setattr(capture, "_open_mss", factory)
    return frames, factory


def test_the_fixture_reports_two_monitors_at_known_origins(two_monitors):
    mons = capture.list_monitors()
    assert [(m.index, m.origin, m.size) for m in mons] == [
        (1, (0, 0), (1280, 800)),
        (2, RIGHT_MONITOR_ORIGIN, (1600, 900)),
    ]
    assert mons[1].as_dict() == {"index": 2, "left": 1280, "top": 0,
                                 "width": 1600, "height": 900}


def test_capture_of_monitor_2_is_its_own_frame(tmp_path, two_monitors):
    out = tmp_path / "right.png"
    path, mon = capture.capture(str(out), monitor=2)
    assert Path(path).is_file() and mon.origin == M2 and mon.size == (1600, 900)
    with Image.open(out) as im:
        assert im.size == (1600, 900)


def test_element_on_the_right_monitor_clicks_back_to_its_desktop_point(tmp_path,
                                                                      two_monitors):
    """The whole point: an element found in monitor 2's pixels carries DESKTOP px."""
    out = tmp_path / "right.png"
    capture.capture(str(out), monitor=2)
    st = build_screen_state(str(out), monitor=2, proposers=("blobs",), inspect=False)
    assert st.frame.origin == M2 and st.frame.capture == "monitor:2"
    found = [e for e in st.elements if _close(e.bbox_image, PLACE_IMAGE)]
    assert found, [e.bbox_image for e in st.elements]
    el = found[0]
    true_desktop = [PLACE_IMAGE[0] + M2[0], PLACE_IMAGE[1] + M2[1],
                    PLACE_IMAGE[2] + M2[0], PLACE_IMAGE[3] + M2[1]]
    assert _close(el.bbox, true_desktop)
    # ScreenEnv's click is the element centre, already in desktop px.
    env = ScreenEnv(state=st)
    click = [a for a in env.actions()
             if a["kind"] == "click" and a["element_id"] == el.id]
    assert click
    true_centre = [(true_desktop[0] + true_desktop[2]) // 2,
                   (true_desktop[1] + true_desktop[3]) // 2]
    assert _close(click[0]["xy"], true_centre)


def test_crop_transform_adds_the_monitors_origin_exactly():
    img, truth = draw_screen_right()
    _crop, tf = make_crop(img, Region("r0", tuple(truth["place"]), "blobs"),
                          margin=0, origin=M2)
    assert tf.to_screen(10, 10) == (M2[0] + 310, M2[1] + 410)
    assert tf.to_screen(0, 0) == (M2[0] + 300, M2[1] + 400)


def test_a_file_verified_as_monitor_2_carries_its_desktop_origin(tmp_path, two_monitors,
                                                                capsys):
    frames, _ = two_monitors
    p = tmp_path / "right.png"
    frames[1][1].save(p)
    rc = main_state(["state", str(p), "--monitor", "2", "--no-inspect",
                     "--proposer", "blobs", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    doc = json.loads(captured.out)
    assert doc["frame"]["origin"] == list(M2)
    assert doc["frame"]["capture"] == "monitor:2"


def test_a_screenshot_that_is_not_the_claimed_monitor_exits_2(tmp_path, two_monitors,
                                                              capsys):
    """1280x800 bytes claimed as the 1600x900 right-hand monitor: exit 2, never an offset."""
    frames, _ = two_monitors
    p = tmp_path / "actually_primary.png"
    frames[0][1].save(p)                       # monitor 1's pixels
    rc = main_state(["state", str(p), "--monitor", "2", "--no-inspect",
                     "--proposer", "blobs"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "1280x800" in captured.err and "monitor 2" in captured.err
    assert "does not belong" in captured.err


def test_monitor_index_the_host_does_not_have_exits_2(tmp_path, two_monitors, capsys):
    frames, _ = two_monitors
    p = tmp_path / "right.png"
    frames[1][1].save(p)
    rc = main_state(["state", str(p), "--monitor", "3", "--no-inspect"])
    assert rc == 2
    assert "monitor 3" in capsys.readouterr().err


def test_mss_returning_a_frame_of_the_wrong_size_is_refused(tmp_path, two_monitors,
                                                            monkeypatch):
    frames, factory = two_monitors
    primary = frames[0][1]

    def lie(self, rect):
        return _FakeShot(primary)              # always the primary display

    monkeypatch.setattr(FakeMss, "grab", lie)
    monkeypatch.setattr(capture, "_open_mss", factory)
    with pytest.raises(capture.CaptureMonitorMismatchError):
        capture.capture(str(tmp_path / "bad.png"), monitor=2)


def test_without_mss_only_monitor_1_may_be_captured(tmp_path, monkeypatch):
    def no_mss():
        raise ImportError("mss not installed")

    monkeypatch.setattr(capture, "_open_mss", no_mss)
    with pytest.raises(capture.MonitorUnavailableError):
        capture.capture(str(tmp_path / "x.png"), monitor=2)


def test_without_mss_a_claimed_monitor_file_is_not_guessed(tmp_path, monkeypatch):
    p = tmp_path / "right.png"
    draw_screen_right()[0].save(p)

    def no_mss():
        raise ImportError("mss not installed")

    monkeypatch.setattr(capture, "_open_mss", no_mss)
    with pytest.raises(capture.MonitorUnavailableError):
        capture.verify_frame(str(p), 2)


def main_state(argv):
    from awscreen.cli import main
    return main(argv)
