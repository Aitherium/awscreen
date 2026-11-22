"""ScreenEnv as a planner environment.

Conformance, actions, dry run, clone replay, ProblemSession."""
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

import copy  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402
from _synthetic import draw_screen  # noqa: E402
from awscreen.env import ReachedLabelScorer, ScreenEnv, action_key  # noqa: E402
from awscreen.state import build_screen_state  # noqa: E402


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


def _goal(state):
    d = copy.deepcopy(state.to_dict())
    d["elements"].append(dict(d["elements"][0], id="eX", label="Saved!"))
    return d


def test_conforms_to_environment_adapter():
    contracts = pytest.importorskip("awpredict.contracts")
    assert contracts.conforms(ScreenEnv, contracts.EnvironmentAdapter) == []
    for m in ("get_state_hash", "evaluate", "clone"):
        assert callable(getattr(ScreenEnv, m))


def test_actions_click_save_at_its_centre(state):
    env = ScreenEnv(state=state, type_slots=["hello"])
    acts = env.actions()
    click = [a for a in acts if a["kind"] == "click" and a["label"] == "save"]
    assert click and click[0]["xy"] == [180, 144]
    assert any(a["kind"] == "type" and a["text"] == "hello" for a in acts)
    assert {a.get("direction") for a in acts if a["kind"] == "scroll"} == {"up", "down"}
    assert len(ScreenEnv(state=state, max_branching=3).actions()) <= 3


def test_dry_run_step_refuses_and_calls_nothing(state):
    env = ScreenEnv(state=state, goal_label="saved")
    before = env.observe()
    nxt, r, done, info = env.step(env.actions()[0])
    assert info["refused"] == "no executor" and r == 0.0 and not done
    assert env.observe() == before


def test_real_step_uses_act_and_perceive(state, tmp_path):
    calls = []
    goal = _goal(state)
    tx = tmp_path / "tx.json"
    env = ScreenEnv(state=state, act_fn=calls.append, perceive_fn=lambda: goal,
                    goal_label="saved", transitions_path=str(tx))
    a = [x for x in env.actions() if x.get("label") == "save"][0]
    _, r, done, info = env.step(a)
    assert calls == [a] and r == 1.0 and done and info["reached"]
    data = json.loads(tx.read_text(encoding="utf-8"))
    assert data["transitions"][0]["action_key"] == action_key(a)


def test_clone_replays_seen_and_predicts_no_change_for_unseen(state):
    env = ScreenEnv(state=state, goal_label="saved")
    a = [x for x in env.actions() if x.get("label") == "save"][0]
    env.record(env.get_state_hash(), a, _goal(state), 1.0)
    sim = env.clone()
    _, r, done, info = sim.step(a)
    assert info["seen"] and r == 1.0 and done and sim.evaluate() == 1.0
    sim2 = env.clone()
    h = sim2.get_state_hash()
    _, r2, done2, info2 = sim2.step({"kind": "scroll", "xy": [640, 400], "direction": "up"})
    assert not info2["seen"] and r2 == 0.0 and not done2 and sim2.get_state_hash() == h
    assert env.get_state_hash() == h          # the clone never touched the original


def test_token_grid_marks_roles(state):
    g = ScreenEnv(state=state).token_grid()
    assert len(g) == 64 and all(len(row) == 64 for row in g)
    assert g[int(144 * 64 / 800)][int(180 * 64 / 1280)] != 0
    assert g[63][0] == 0


def test_problem_session_by_ref(state, tmp_path):
    pytest.importorskip("awpredict.contracts")
    problem = pytest.importorskip("awgym.gym.problem")
    from awpredict.contracts import Budget, ProblemSpec, Refusal, Score

    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(state.to_dict()), encoding="utf-8")
    tx = tmp_path / "tx.json"
    env = ScreenEnv(state=state, goal_label="saved", transitions_path=str(tx))
    a = [x for x in env.actions() if x.get("label") == "save"][0]
    env.record(env.get_state_hash(), a, _goal(state), 1.0)

    def spec(**kw):
        return ProblemSpec(problem_id="screen-demo", domain="screen",
                           adapter_ref="awscreen.env:ScreenEnv",
                           scorer_ref="awscreen.env:ReachedLabelScorer",
                           budget=Budget(steps=3),
                           adapter_kwargs=dict({"state_path": str(sp),
                                                "transitions_path": str(tx),
                                                "goal_label": "saved"}, **kw))

    dry = problem.ProblemSession(spec(), problem.const_policy(a)).run()
    assert isinstance(dry.outcome.refusal, Refusal) and dry.outcome.score is None

    sim = problem.ProblemSession(spec(simulate=True), problem.const_policy(a)).run()
    assert isinstance(sim.outcome.score, Score) and sim.outcome.score.value == 1.0


def test_scorer_refuses_empty():
    pytest.importorskip("awpredict.contracts")
    from awpredict.contracts import Refusal
    assert isinstance(ReachedLabelScorer().score([]), Refusal)
