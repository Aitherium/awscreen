"""PixelActuator: plan resolution, exact coordinates, order, and the dry/live split.

No test clicks a real window. A recording backend stands in for the machine, so the
suite proves the PLAN -- the absolute points and the order they would be sent -- which
is the only part that can be checked without taking over someone's desktop.
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
from _synthetic import draw_screen  # noqa: E402
from awscreen.actuator import (  # noqa: E402
    ActuatorRefused,
    DryRunActuator,
    HttpBackend,
    LiveActuator,
    PixelActuator,
    PlanError,
    TargetNotFoundError,
    resolve_plan,
    run_plan,
)
from awscreen.cli import main  # noqa: E402
from awscreen.env import ScreenEnv  # noqa: E402
from awscreen.state import Element, Frame, ScreenState, build_screen_state  # noqa: E402

PLAN = {"actions": [
    {"kind": "click", "label": "save"},
    {"kind": "type", "label": "name", "text": "hello"},
    {"kind": "scroll", "direction": "down"},
]}


class RecordingBackend:
    """Callable-shaped backend: records exactly what LiveActuator sends."""

    def __init__(self):
        self.calls = []

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def type_text(self, text):
        self.calls.append(("type", text))

    def scroll(self, dy):
        self.calls.append(("scroll", dy))


@pytest.fixture
def state(tmp_path):
    img, truth = draw_screen()
    p = tmp_path / "syn.png"
    img.save(p)
    s = build_screen_state(str(p), proposers=("blobs",), inspect=False)
    for e in s.elements:
        if e.bbox_image == truth["save"]:
            e.label, e.role, e.interactable = "save", "button", True
        if e.bbox_image == truth["input"]:
            e.label, e.role, e.interactable = "name", "input", True
    return s


def test_implementations_satisfy_the_protocol():
    assert isinstance(DryRunActuator(), PixelActuator)
    assert isinstance(LiveActuator(RecordingBackend()), PixelActuator)


def test_live_actuator_runs_the_same_plan_exact_coordinates_and_order(state):
    resolved = resolve_plan(state, PLAN)
    # click -> (180,144); type target -> (500,320); scroll -> frame centre (640,400)
    assert [(a["kind"], a["x"], a["y"]) for a in resolved] == [
        ("click", 180, 144), ("type", 500, 320), ("scroll", 640, 400)]

    backend = RecordingBackend()
    live = run_plan(LiveActuator(backend), resolved)
    assert backend.calls == [("click", 180, 144), ("type", "hello"), ("scroll", 3)]
    # History carries the same absolute points, in the same order.
    assert [c["kind"] for c in live] == ["click", "type", "scroll"]
    assert (live[0]["x"], live[0]["y"]) == (180, 144)
    assert live[1]["text"] == "hello"
    assert live[2]["dy"] == 3 and (live[2]["x"], live[2]["y"]) == (640, 400)


def test_dry_run_records_the_same_plan_and_changes_nothing(state):
    resolved = resolve_plan(state, PLAN)
    dry = DryRunActuator()
    before = json.dumps(state.to_dict(), sort_keys=True)
    calls = run_plan(dry, resolved)
    assert calls == dry.calls
    assert [(c["kind"], c.get("x"), c.get("y")) for c in calls] == [
        (a["kind"], a["x"], a["y"]) for a in resolved]
    assert json.dumps(state.to_dict(), sort_keys=True) == before   # state untouched
    assert dry.dry_run is True


def test_resolve_uses_absolute_screen_pixels_including_origin():
    st = ScreenState(
        frame=Frame(path="m2.png", width=800, height=600, origin=(1920, 0)),
        elements=[Element(id="e0", region_id=None, bbox=[2020, 130, 2180, 178],
                          bbox_image=[100, 120, 260, 168], label="save", role="button")])
    click = resolve_plan(st, [{"kind": "click", "element_id": "e0"}])[0]
    assert (click["x"], click["y"]) == (2100, 154)
    scroll = resolve_plan(st, [{"kind": "scroll", "direction": "up"}])[0]
    assert (scroll["x"], scroll["y"]) == (2320, 300) and scroll["dy"] == -3


def test_resolve_element_id_and_xy_forms(state):
    el = [e for e in state.elements if e.label == "save"][0]
    by_id = resolve_plan(state, {"actions": [{"kind": "click", "element_id": el.id}]})[0]
    assert by_id["element_id"] == el.id and (by_id["x"], by_id["y"]) == (180, 144)
    explicit = resolve_plan(state, [{"kind": "click", "xy": [7, 9]}])[0]
    assert (explicit["x"], explicit["y"]) == (7, 9)


def test_missing_target_is_not_found_and_malformed_is_plan_error(state):
    with pytest.raises(TargetNotFoundError):
        resolve_plan(state, [{"kind": "click", "label": "definitely-absent"}])
    with pytest.raises(PlanError):                      # known target kind, no target
        resolve_plan(state, [{"kind": "click"}])
    with pytest.raises(PlanError):
        resolve_plan(state, [{"kind": "explode", "xy": [1, 2]}])
    with pytest.raises(PlanError):
        resolve_plan(state, {"plan": []})              # no 'actions'


def test_live_actuator_refuses_a_backend_missing_an_operation():
    with pytest.raises(ActuatorRefused):
        LiveActuator(object())


def test_http_backend_posts_one_json_request_per_action_without_network():
    posts = []

    def post(url, payload):
        posts.append((url, payload))
        return b"{}"

    backend = HttpBackend("http://127.0.0.1:9000/", post=post)
    backend.click(10, 20)
    backend.type_text("hi")
    backend.scroll(-2)
    assert posts == [
        ("http://127.0.0.1:9000/click", {"x": 10, "y": 20}),
        ("http://127.0.0.1:9000/type", {"text": "hi"}),
        ("http://127.0.0.1:9000/scroll", {"dy": -2}),
    ]


# ------------------------------------------------------------------------ ScreenEnv
def test_screen_env_default_is_dry_run_and_records(state):
    env = ScreenEnv(state=state)
    before = env.observe()
    action = [a for a in env.actions() if a.get("label") == "save"][0]
    _, reward, done, info = env.step(action)
    assert info["refused"] == "no executor" and info["dry_run"] is True
    assert info["recorded"] == {"kind": "click", "x": 180, "y": 144}
    assert env.actuator.calls == [info["recorded"]]
    assert env.observe() == before and reward == 0.0 and not done


def test_screen_env_with_live_actuator_acts_then_perceives(state):
    backend = RecordingBackend()
    env = ScreenEnv(state=state, actuator=LiveActuator(backend),
                    perceive_fn=lambda: state, reward_fn=lambda p, n, a: 0.25)
    action = [a for a in env.actions() if a.get("label") == "save"][0]
    _, reward, done, info = env.step(action)
    assert backend.calls == [("click", 180, 144)]
    assert info["acted"] and info["recorded"]["x"] == 180
    assert reward == 0.25 and done is False


# ----------------------------------------------------------------------------- CLI
def _write(state, tmp_path):
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(PLAN), encoding="utf-8")
    return sp, pp


def test_cli_act_dry_run_prints_resolved_absolute_points(state, tmp_path, capsys):
    sp, pp = _write(state, tmp_path)
    assert main(["act", "--plan", str(pp), "--state", str(sp), "--dry-run"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["dry_run"] is True
    assert [(a["kind"], a["x"], a["y"]) for a in doc["actions"]] == [
        ("click", 180, 144), ("type", 500, 320), ("scroll", 640, 400)]
    assert [c["kind"] for c in doc["calls"]] == ["click", "type", "scroll"]
    assert (doc["calls"][0]["x"], doc["calls"][0]["y"]) == (180, 144)


def test_cli_act_live_sends_to_the_backend_in_order(state, tmp_path, capsys, monkeypatch):
    import awscreen.actuator as actuator_mod

    backend = RecordingBackend()
    monkeypatch.setattr(actuator_mod, "HttpBackend", lambda *a, **k: backend)
    sp, pp = _write(state, tmp_path)
    assert main(["act", "--plan", str(pp), "--state", str(sp),
                 "--live", "--endpoint", "http://127.0.0.1:9000"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["dry_run"] is False
    assert backend.calls == [("click", 180, 144), ("type", "hello"), ("scroll", 3)]


def test_cli_act_missing_target_exits_1(state, tmp_path, capsys):
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps({"actions": [{"kind": "click", "label": "nope"}]}))
    assert main(["act", "--plan", str(pp), "--state", str(sp), "--dry-run"]) == 1
    assert "NOT FOUND" in capsys.readouterr().err


def test_cli_act_live_without_endpoint_exits_2(state, tmp_path, capsys):
    sp, pp = _write(state, tmp_path)
    assert main(["act", "--plan", str(pp), "--state", str(sp), "--live"]) == 2
    assert "--endpoint" in capsys.readouterr().err


def test_cli_act_text_format_names_points(state, tmp_path, capsys):
    sp, pp = _write(state, tmp_path)
    assert main(["act", "--plan", str(pp), "--state", str(sp), "--format", "text"]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out and "(180, 144)" in out and "dy=3" in out
