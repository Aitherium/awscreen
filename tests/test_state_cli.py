"""`awscreen state`: exit codes, the proposer report, reuse, never inventing elements. Offline."""
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
import socket  # noqa: E402
import urllib.request  # noqa: E402

import pytest  # noqa: E402
from _synthetic import draw_blank, draw_screen  # noqa: E402
from awscreen import regions  # noqa: E402
from awscreen.cli import main  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise OSError("network disabled in this test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(regions.OmniParserProposer, "_health",
                        lambda self: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setenv("AWVISION_URL", "https://127.0.0.1:1")


@pytest.fixture
def syn(tmp_path):
    img, truth = draw_screen()
    p = tmp_path / "syn.png"
    img.save(p)
    return p, truth


def _close(a, b, tol=2):
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def test_blobs_no_inspect_matches_truth(syn, capsys):
    p, truth = syn
    assert main(["state", str(p), "--no-inspect", "--proposer", "blobs", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "awscreen.screen_state/1"
    assert doc["proposer"]["ran"] == "blobs" and doc["timing"]["inspect_ms"] == 0
    boxes = [e["bbox"] for e in doc["elements"]]
    for name in ("save", "cancel", "input", "hello"):
        assert any(_close(b, truth[name]) for b in boxes), name
    for k in ("capture_ms", "propose_ms", "crop_ms", "neurons_ms", "merge_ms", "total_ms"):
        assert k in doc["timing"]


def test_auto_chain_reports_each_attempt(syn, capsys):
    p, _ = syn
    assert main(["state", str(p), "--no-inspect", "--json"]) == 0
    tried = json.loads(capsys.readouterr().out)["proposer"]["tried"]
    assert tried[0]["name"] == "omniparser" and not tried[0]["ok"]
    assert "health probe" in tried[0]["reason"]
    assert [t["name"] for t in tried][-1] == "blobs"


def test_blank_image_exits_1(tmp_path, capsys):
    p = tmp_path / "blank.png"
    draw_blank().save(p)
    assert main(["state", str(p), "--no-inspect"]) == 1


def test_missing_file_exits_2(tmp_path):
    assert main(["state", str(tmp_path / "nope.png"), "--no-inspect"]) == 2


def test_requested_omniparser_down_exits_2(syn, capsys):
    p, _ = syn
    assert main(["state", str(p), "--proposer", "omniparser"]) == 2
    assert "omniparser" in capsys.readouterr().err


def test_unreachable_vision_exits_2_never_invents(syn, capsys):
    p, _ = syn
    rc = main(["state", str(p), "--proposer", "blobs", "--json", "--timeout", "2"])
    captured = capsys.readouterr()
    assert rc == 2
    doc = json.loads(captured.out)
    assert not [e for e in doc["elements"] if e["source"] == "vlm" and e["status"] == "ok"]


def test_self_test_exits_0():
    assert main(["state", "--self-test"]) == 0


def test_prev_identical_frame_reuses(syn, tmp_path, capsys):
    p, _ = syn
    out = tmp_path / "prev.json"
    assert main(["state", str(p), "--no-inspect", "--proposer", "blobs", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["state", str(p), "--proposer", "blobs", "--prev", str(out), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["reused"] is True and doc["timing"]["inspect_ms"] == 0
    assert len(doc["elements"]) == len(json.loads(out.read_text())["elements"])


def test_prev_reuse_with_a_new_origin_rebases_absolute_bboxes(syn, tmp_path, capsys):
    """The frame cache keys on the pixels only, but an element bbox is ABSOLUTE screen px.
    Reusing across a different --origin must shift those bboxes by the origin delta, or a
    caller on a second monitor gets frame.origin saying the new monitor and bboxes still
    in the old monitor's coordinates (the 1920-px error the verifier measured)."""
    p, _ = syn
    out = tmp_path / "prev.json"
    assert main(["state", str(p), "--no-inspect", "--proposer", "blobs",
                 "--out", str(out)]) == 0
    capsys.readouterr()
    prev = json.loads(out.read_text())
    assert main(["state", str(p), "--no-inspect", "--proposer", "blobs",
                 "--prev", str(out), "--origin", "1920,0", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["reused"] is True and doc["frame"]["origin"] == [1920, 0]
    assert len(doc["elements"]) == len(prev["elements"])
    for e, pe in zip(doc["elements"], prev["elements"]):
        assert e["bbox_image"] == pe["bbox_image"]                       # image px: same
        assert e["bbox"] == [pe["bbox"][0] + 1920, pe["bbox"][1],
                             pe["bbox"][2] + 1920, pe["bbox"][3]]        # screen px: shifted


def test_screen_capture_reports_the_monitors_real_origin(syn, tmp_path, monkeypatch):
    """`state --screen` must localize in DESKTOP px: the captured monitor's real origin,
    not a hardcoded (0,0) (a second monitor's click would land 1920 px off)."""
    import awvision.sight as sight
    from awscreen.state import build_screen_state

    p, _ = syn
    monkeypatch.setattr(sight, "capture_screen_origin",
                        lambda out, monitor=1: (str(p), (1920, 0)))
    st = build_screen_state(None, screen=True, proposers=("blobs",), inspect=False,
                            capture_path=str(tmp_path / "cap.jpg"))
    assert st.frame.origin == (1920, 0)
    # An explicit --origin still wins over the capture's bookmark.
    st2 = build_screen_state(None, screen=True, proposers=("blobs",), inspect=False,
                             origin=(10, 20), capture_path=str(tmp_path / "cap2.jpg"))
    assert st2.frame.origin == (10, 20)
