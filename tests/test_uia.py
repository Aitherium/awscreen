"""UIA join: recorded fixture, role mapping, IoU/geometry rules, skip-with-reason."""
import json
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

import pytest  # noqa: E402
from _synthetic import draw_screen  # noqa: E402
from awscreen import uia  # noqa: E402
from awscreen.cli import main  # noqa: E402
from awscreen.state import Frame, ScreenState, build_screen_state  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "uia_notepad_tree.json"


def _put_screen(tmp_path):
    img, truth = draw_screen()
    p = tmp_path / "syn.png"
    img.save(p)
    return p, truth


def _state(tmp_path, **kw):
    p, truth = _put_screen(tmp_path)
    return build_screen_state(str(p), proposers=("blobs",), inspect=False, **kw), truth


def _by_automation(state, automation_id):
    return [e for e in state.elements
            if e.uia and e.uia.get("automation_id") == automation_id]


def test_recorded_fixture_loads_with_the_required_shape():
    snap = uia.UiaSnapshot.load(str(FIXTURE))
    assert snap.schema == "awscreen.uia/1"
    assert snap.source == "fixture" and snap.image_scale == 1.0
    assert snap.window["title"] == "Untitled - Notepad"
    assert len(snap.nodes) == 10
    for node in snap.nodes:
        assert {"name", "control_type", "rect", "is_enabled"} <= set(node)
        assert len(node["rect"]) == 4
        assert isinstance(node["is_enabled"], bool)


def test_role_mapping_keeps_menu_item_semantics():
    assert uia.role_for_control_type("ControlType.MenuItem") == "menuitem"
    assert uia.role_for_control_type("MenuItem") == "menuitem"
    assert uia.role_for_control_type("ControlType.Button") == "button"
    assert uia.role_for_control_type("Edit") == "input"
    assert uia.role_for_control_type("Hyperlink") == "link"
    assert uia.role_for_control_type("Totally-Unknown") == "other"


def test_join_matches_by_iou_and_keeps_the_control_type(tmp_path):
    state, truth = _state(tmp_path)
    state = uia.join_uia(state, uia.UiaSnapshot.load(str(FIXTURE)))

    save = _by_automation(state, "SaveItem")
    assert len(save) == 1
    e = save[0]
    assert e.role == "menuitem"                  # not "button"/label: a menu item
    assert "uia" in e.sources and "cv" in e.sources
    assert e.label == "Save"
    assert e.bbox_image == [100, 120, 260, 168]  # UIA geometry (no DOM matched)
    assert e.bbox == list(truth["save"])
    assert e.uia["iou"] > 0.9 and e.source == "uia"

    cancel = _by_automation(state, "CancelButton")[0]
    assert cancel.role == "button" and cancel.bbox_image == list(truth["cancel"])
    search = _by_automation(state, "SearchBox")[0]
    assert search.role == "input" and search.bbox_image == list(truth["input"])

    assert state.uia["ok"] is True and state.uia["matched"] == 3
    assert state.uia["added"] == 2                 # Format, Delete
    assert state.uia["skipped"] == 5               # Window, MenuBar, List, Group, off-screen
    assert state.uia["nodes"] == 10


def test_unmatched_nodes_are_added_and_disabled_is_not_interactable(tmp_path):
    state, _ = _state(tmp_path)
    state = uia.join_uia(state, uia.UiaSnapshot.load(str(FIXTURE)))

    fmt = [e for e in state.elements
           if e.source == "uia" and e.uia and e.uia.get("automation_id") == "FormatItem"]
    assert len(fmt) == 1 and fmt[0].role == "menuitem" and fmt[0].interactable is True
    assert fmt[0].bbox == [400, 100, 480, 124]

    delete = _by_automation(state, "DeleteItem")[0]
    assert delete.role == "menuitem" and delete.interactable is False

    # Structural chrome is skipped, never invented into a giant "element".
    assert not [e for e in state.elements
                if e.uia and e.uia.get("control_type") in ("Window", "MenuBar", "List")]


def test_origin_and_image_scale_are_mapped_exactly(tmp_path):
    state, _ = _state(tmp_path, origin=(1920, 0))
    state = uia.join_uia(state, uia.UiaSnapshot.from_dict({"nodes": [
        {"name": "Save", "control_type": "Button", "rect": [2020, 120, 160, 48],
         "is_enabled": True}]}))
    hit = _by_automation(state, "")
    assert len(hit) == 1
    assert hit[0].bbox_image == [100, 120, 260, 168]
    assert hit[0].bbox == [2020, 120, 2180, 168]

    small = ScreenState(frame=Frame(path="small.png", width=640, height=400))
    small = uia.join_uia(small, uia.UiaSnapshot.from_dict({
        "image_scale": 0.5, "nodes": [
            {"name": "X", "control_type": "Button", "rect": [200, 240, 320, 96],
             "is_enabled": True}]}))
    assert small.elements[0].bbox_image == [100, 120, 260, 168]


def test_dom_geometry_survives_a_uia_join(tmp_path):
    state, _ = _state(tmp_path, dom={"viewport": {"dpr": 1, "vw": 1280},
                                    "elements": [{"ref": "#save", "tag": "button",
                                                  "label": "Save",
                                                  "rect": [100, 120, 160, 48]}]})
    dom_box = [e for e in state.elements if e.dom and e.dom["ref"] == "#save"][0].bbox_image
    state = uia.join_uia(state, uia.UiaSnapshot.from_dict({"nodes": [
        {"name": "Save", "control_type": "MenuItem", "rect": [102, 121, 160, 48],
         "is_enabled": True}]}))
    hit = _by_automation(state, "")
    assert len(hit) == 1
    assert hit[0].bbox_image == dom_box       # DOM CSS geometry is exact; UIA does not win
    assert hit[0].dom["ref"] == "#save"
    assert hit[0].role == "menuitem"          # semantics still come from UIA
    assert {"dom", "uia"} <= set(hit[0].sources)


def test_off_windows_skips_with_a_reason_never_invents():
    for plat in ("linux", "darwin"):
        ok, why = uia.bridge_available(platform=plat)
        assert ok is False and "Windows" in why
        with pytest.raises(uia.UiaUnavailableError) as excinfo:
            uia.walk_focused_window(platform=plat)
        assert "Windows-only" in str(excinfo.value)


def test_bridge_stdout_parses_a_recorded_tree():
    snap = uia.parse_bridge_stdout(FIXTURE.read_text(encoding="utf-8-sig"))
    assert snap.source == "powershell" and len(snap.nodes) == 10
    for bad in ("", "not json", '{"nodes": []}'):
        with pytest.raises(ValueError):
            uia.parse_bridge_stdout(bad)


def test_bridge_stdout_tolerates_raw_control_characters_in_names():
    # Measured on a real Windows tree: PowerShell 5.1 ConvertTo-Json can leave a raw
    # tab/newline inside a string, which strict JSON rejects.
    raw = ('{"source": "powershell", "nodes": [{"name": "Save\tButton\n now", '
           '"control_type": "Button", "rect": [0, 0, 10, 10], "is_enabled": true}]}')
    snap = uia.parse_bridge_stdout(raw)
    assert snap.nodes[0]["name"] == "Save Button now"


def test_build_screen_state_accepts_a_recorded_snapshot(tmp_path):
    p, _ = _put_screen(tmp_path)
    snap = json.loads(FIXTURE.read_text(encoding="utf-8-sig"))
    state = build_screen_state(str(p), proposers=("blobs",), inspect=False, uia=snap)
    assert state.uia["ok"] is True and state.uia["matched"] == 3
    assert "uia_join_ms" in state.timing
    assert _by_automation(state, "SaveItem")[0].role == "menuitem"


def test_build_screen_state_skips_when_the_bridge_is_missing(tmp_path, monkeypatch):
    def unavailable(*a, **k):
        raise uia.UiaUnavailableError("UIA is Windows-only (platform='linux')")

    monkeypatch.setattr(uia, "walk_focused_window", unavailable)
    p, _ = _put_screen(tmp_path)
    state = build_screen_state(str(p), proposers=("blobs",), inspect=False, uia="auto")
    assert state.uia["attempted"] is True and state.uia["ok"] is False
    assert "Windows-only" in state.uia["reason"]
    assert not [e for e in state.elements if e.uia or e.source == "uia"]


def test_cli_joins_a_recorded_tree_and_prints_the_counts(tmp_path, capsys):
    p, _ = _put_screen(tmp_path)
    rc = main(["state", str(p), "--no-inspect", "--proposer", "blobs",
               "--uia", str(FIXTURE), "--json"])
    out, err = capsys.readouterr()
    assert rc == 0
    doc = json.loads(out)
    assert doc["uia"]["ok"] is True and doc["uia"]["matched"] == 3
    assert any(e.get("uia", {}).get("automation_id") == "SaveItem"
               and e["role"] == "menuitem" for e in doc["elements"])
    assert "uia: 10 node(s)" in err and "matched 3" in err


def test_cli_auto_uia_skips_with_a_printed_reason(tmp_path, monkeypatch, capsys):
    def unavailable(*a, **k):
        raise uia.UiaUnavailableError("UIA is Windows-only (platform='linux')")

    monkeypatch.setattr(uia, "walk_focused_window", unavailable)
    p, _ = _put_screen(tmp_path)
    rc = main(["state", str(p), "--no-inspect", "--proposer", "blobs", "--uia", "auto"])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "elements=" in out
    assert "uia: skipped (UIA is Windows-only" in err
