"""ScreenEnv: a ScreenState as an environment a planner can search over.

It satisfies the awpredict ``EnvironmentAdapter`` protocol STRUCTURALLY (``domain``,
``observe``, ``actions``, ``step`` returning a 4-tuple) without importing awpredict, and
also carries the MCTS environment members (``get_state_hash``, ``evaluate``, ``clone``).

* ``step`` never acts on its own. With no injected executor it uses a
  :class:`~awscreen.actuator.DryRunActuator`: the action is RECORDED and the state is
  unchanged (reward 0, ``info.refused = 'no executor'``). Pass ``actuator=`` for a
  :class:`~awscreen.actuator.LiveActuator`, or ``act_fn=`` for a raw callback.
* ``clone`` is a SIMULATION that replays only recorded ``(state_hash, action_key) ->
  next_state`` transitions; an unseen pair predicts no change and reward 0, so a search
  cannot invent progress.

Played by a ProblemSession through a string reference, no registry::

    ProblemSpec(problem_id="screen-demo", domain="screen",
                adapter_ref="awscreen.env:ScreenEnv",
                scorer_ref="awscreen.env:ReachedLabelScorer",
                adapter_kwargs={"state_path": "state.json", "transitions_path": "tx.json",
                                "goal_label": "save"})
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .actuator import DryRunActuator
from .state import INTERACTIVE_ROLES, ROLES, ScreenState

_BUCKET = 16
ROLE_IDS = {r: i + 1 for i, r in enumerate(ROLES)}   # 0 = empty cell


def _as_state(s: Any) -> ScreenState:
    if isinstance(s, ScreenState):
        return s
    if isinstance(s, dict):
        return ScreenState.from_dict(s)
    raise TypeError(f"expected ScreenState or dict, got {type(s).__name__}")


def describe(state: ScreenState) -> str:
    """Stable descriptor: one line per element, sorted by (y1, x1), coordinates bucketed."""
    rows = []
    for e in sorted(state.elements, key=lambda e: (e.bbox[1], e.bbox[0])):
        if e.status != "ok":
            continue
        b = "/".join(str(int(v) // _BUCKET) for v in e.bbox)
        th = hashlib.sha1((e.text or "").encode("utf-8")).hexdigest()[:8]
        rows.append(f"{e.role}|{(e.label or '').strip().lower()}|{b}|{th}")
    return "\n".join(rows)


def action_key(action: dict) -> str:
    xy = action.get("xy") or (0, 0)
    return "%s@%d,%d|%s" % (action.get("kind"), int(xy[0]) // _BUCKET, int(xy[1]) // _BUCKET,
                            action.get("text") or action.get("direction") or "")


class ScreenEnv:
    domain = "screen"

    def __init__(self, state: Any = None, state_path: Optional[str] = None,
                 act_fn: Optional[Callable[[dict], Any]] = None,
                 actuator: Optional[Any] = None,
                 perceive_fn: Optional[Callable[[], Any]] = None,
                 reward_fn: Optional[Callable[[ScreenState, ScreenState, dict], float]] = None,
                 goal_label: Optional[str] = None, type_slots: Any = None,
                 transitions_path: Optional[str] = None, max_branching: int = 24,
                 simulate: bool = False) -> None:
        if state is None and state_path:
            state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        if state is None:
            raise ValueError("ScreenEnv needs a state or a state_path")
        self.state = _as_state(state)
        self._initial = copy.deepcopy(self.state)
        self.act_fn = act_fn
        # DryRunActuator is the DEFAULT: an env that was never told to act records the
        # action and changes nothing. An injected actuator wins over act_fn.
        self._actuator_injected = actuator is not None
        if actuator is not None:
            self.actuator = actuator
        elif act_fn is None:
            self.actuator = DryRunActuator()
        else:
            self.actuator = None
        self.perceive_fn = perceive_fn
        self.reward_fn = reward_fn
        self.goal_label = (goal_label or "").strip().lower() or None
        if isinstance(type_slots, dict):
            type_slots = list(type_slots.values())
        self.type_slots = [str(t) for t in (type_slots or [])]
        self.transitions_path = transitions_path
        self.max_branching = int(max_branching)
        self.simulate = bool(simulate)
        self.transitions: Dict[str, dict] = {}
        if transitions_path and Path(transitions_path).is_file():
            data = json.loads(Path(transitions_path).read_text(encoding="utf-8"))
            for t in data.get("transitions") or []:
                self.transitions[f"{t['hash']}::{t['action_key']}"] = t

    # ---------------------------------------------------------------- protocol
    def reset(self) -> ScreenState:
        self.state = copy.deepcopy(self._initial)
        return self.state

    def observe(self, env_state: Any = None) -> str:
        return describe(_as_state(env_state) if env_state is not None else self.state)

    def actions(self) -> List[dict]:
        out: List[dict] = []
        for e in sorted(self.state.elements, key=lambda e: (e.bbox[1], e.bbox[0])):
            if e.status != "ok":
                continue
            if e.interactable or e.role in INTERACTIVE_ROLES:
                out.append({"kind": "click", "element_id": e.id, "xy": list(e.center),
                            "label": e.label})
            if e.role == "input":
                for text in self.type_slots:
                    out.append({"kind": "type", "element_id": e.id, "xy": list(e.center),
                                "text": text, "label": e.label})
        fr = self.state.frame
        mid = [fr.origin[0] + fr.width // 2, fr.origin[1] + fr.height // 2]
        scrolls = [{"kind": "scroll", "element_id": None, "xy": mid, "direction": d}
                   for d in ("down", "up")]
        return out[: max(0, self.max_branching - len(scrolls))] + scrolls

    def step(self, action: dict) -> Tuple[ScreenState, float, bool, Dict[str, Any]]:
        prev = self.state
        key = action_key(action)
        h = self.get_state_hash()
        if self.simulate:
            t = self.transitions.get(f"{h}::{key}")
            if t is None:
                return prev, 0.0, False, {"simulated": True, "seen": False, "reached": False}
            nxt = ScreenState.from_dict(t["next_state"])
            self.state = nxt
            reached = self._reached(nxt)
            return nxt, float(t.get("reward", 0.0)), reached, {
                "simulated": True, "seen": True, "reached": reached}
        if self.actuator is not None:
            recorded = self.actuator.apply(action)
            if self.actuator.dry_run:
                info: Dict[str, Any] = {"dry_run": True, "recorded": recorded,
                                        "reached": False}
                if not self._actuator_injected:
                    # Never told to act: recorded, refused to touch the machine.
                    info["refused"] = "no executor"
                return prev, 0.0, False, info
            return self._advance(prev, action, key, h, {"recorded": recorded})
        if self.act_fn is None:            # defensive; the constructor guarantees one
            return prev, 0.0, False, {"refused": "no executor", "dry_run": True,
                                      "reached": False}
        self.act_fn(action)
        return self._advance(prev, action, key, h, {})

    def _advance(self, prev: ScreenState, action: dict, key: str, h: int,
                 info: Dict[str, Any]) -> Tuple[ScreenState, float, bool, Dict[str, Any]]:
        nxt = _as_state(self.perceive_fn()) if self.perceive_fn else prev
        reached = self._reached(nxt)
        if self.reward_fn is not None:
            reward = float(self.reward_fn(prev, nxt, action))
        else:
            reward = 1.0 if reached and not self._reached(prev) else 0.0
        self._record(h, key, nxt, reward)
        self.state = nxt
        return nxt, reward, reached, dict({"acted": True, "reached": reached}, **info)

    # ---------------------------------------------------------------- MCTS side
    def get_state_hash(self) -> int:
        return int(hashlib.sha1(self.observe().encode("utf-8")).hexdigest()[:15], 16)

    def evaluate(self) -> float:
        return 1.0 if self._reached(self.state) else 0.0

    def clone(self) -> "ScreenEnv":
        c = ScreenEnv(state=copy.deepcopy(self.state), reward_fn=self.reward_fn,
                      goal_label=self.goal_label, type_slots=list(self.type_slots),
                      max_branching=self.max_branching, simulate=True)
        c.transitions = dict(self.transitions)
        c._initial = copy.deepcopy(self._initial)
        return c

    def token_grid(self, state: Any = None, size: int = 64) -> List[List[int]]:
        """``size`` x ``size`` grid of role ids (0 = empty) over the frame."""
        st = _as_state(state) if state is not None else self.state
        img_w, img_h = max(1, st.frame.width), max(1, st.frame.height)
        grid = [[0] * size for _ in range(size)]
        for e in sorted(st.elements, key=lambda e: -((e.bbox_image[2] - e.bbox_image[0])
                                                      * (e.bbox_image[3] - e.bbox_image[1]))):
            if e.status != "ok":
                continue
            x1 = min(size - 1, int(e.bbox_image[0] * size / img_w))
            y1 = min(size - 1, int(e.bbox_image[1] * size / img_h))
            x2 = max(x1, min(size - 1, int((e.bbox_image[2] - 1) * size / img_w)))
            y2 = max(y1, min(size - 1, int((e.bbox_image[3] - 1) * size / img_h)))
            rid = ROLE_IDS.get(e.role, ROLE_IDS["other"])
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    grid[y][x] = rid
        return grid

    # ---------------------------------------------------------------- helpers
    def _reached(self, st: ScreenState) -> bool:
        return bool(self.goal_label) and bool(st.find(self.goal_label))

    def record(self, state_hash: int, action: dict, next_state: Any, reward: float) -> None:
        """Teach the simulator one observed transition (e.g. from a real run)."""
        self._record(state_hash, action_key(action), _as_state(next_state), reward)

    def _record(self, h: int, key: str, nxt: ScreenState, reward: float) -> None:
        self.transitions[f"{h}::{key}"] = {"hash": str(h), "action_key": key,
                                           "next_state": nxt.to_dict(),
                                           "reward": float(reward)}
        if self.transitions_path:
            p = Path(self.transitions_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps({"transitions": list(self.transitions.values())}),
                           encoding="utf-8")
            tmp.replace(p)


class ReachedLabelScorer:
    """Scores an episode 1.0 when the goal label appeared, 0.0 when it did not; REFUSES
    when there are no transitions or every step was a dry run (nothing was tried)."""

    metric = "reached_label"
    minimize = False

    def score(self, episode: Any):
        from awpredict.contracts import Refusal, Score  # lazy: only under a runner

        steps = list(episode or [])
        if not steps:
            return Refusal("no transitions")
        metas = [getattr(t, "meta", None) or {} for t in steps]
        if all(m.get("dry_run") for m in metas):
            return Refusal("dry run only: no executor acted, nothing to judge")
        reached = any(m.get("reached") for m in metas)
        return Score(self.metric, 1.0 if reached else 0.0,
                     evidence=f"{len(steps)} steps, reached={reached}")
