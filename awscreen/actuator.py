"""PixelActuator: turn a resolved action into a click, a type or a scroll.

``ScreenEnv`` can plan over a ``ScreenState`` but, on its own, never touches the machine.
This module is the missing half: a small protocol with two implementations and a plan
resolver that turns element references into ABSOLUTE desktop points.

* :class:`DryRunActuator` records every action and changes nothing. It is the DEFAULT,
  so a caller who never asked to act cannot click by accident.
* :class:`LiveActuator` dispatches to an injected backend -- three callables shaped like
  ``click(x, y)``, ``type_text(text)`` / ``type(text)`` and ``scroll(dy)``. The backend is
  usually an :class:`HttpBackend` (one JSON POST per action), but any object or mapping
  with those callables works, which is how a test proves the plan without clicking a
  real window.

Plans are plain JSON. An action names its target by ``element_id``, by ``label`` (or
``target``), or by explicit ``xy``; :func:`resolve_plan` reads a ``ScreenState`` and
returns the absolute points a click would land on::

    {"actions": [
        {"kind": "click", "label": "save"},
        {"kind": "type", "element_id": "e2", "text": "hello"},
        {"kind": "scroll", "direction": "down"}
    ]}

A ``type`` action sends only the text (the field must already have focus); put an explicit
``click`` before it when the plan has to open the field first. A ``scroll`` with no point
uses the frame's centre; ``direction`` is ``up``/``down`` and ``dy`` overrides it.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

#: Wheel notches for a scroll action with no explicit ``dy``.
SCROLL_STEP = 3


class ActuatorRefused(RuntimeError):  # noqa: N818 - public name, mirrors ExecutorRefused
    """The actuator cannot act as asked; the message says what is missing."""


class PlanError(ValueError):
    """The plan or its resolved output is malformed (the CLI maps this to exit 2)."""


class TargetNotFoundError(PlanError):
    """A plan references an element or label the ScreenState does not hold (exit 1)."""


@runtime_checkable
class PixelActuator(Protocol):
    """What a click needs: three operations, a history and a dry-run flag."""

    dry_run: bool
    calls: List[dict]

    def click(self, x: int, y: int) -> dict: ...

    def type_text(self, text: str) -> dict: ...

    def scroll(self, dy: int) -> dict: ...

    def apply(self, action: Mapping[str, Any]) -> dict: ...


# --------------------------------------------------------------------------- helpers
def _as_int(value: Any, name: str) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        raise PlanError(f"{name} must be a number, got {value!r}") from None


def _xy_of(action: Mapping[str, Any]) -> Optional[tuple]:
    """An explicit point: ``x``/``y`` first, then ``xy`` as a pair."""
    if action.get("x") is not None and action.get("y") is not None:
        return _as_int(action["x"], "x"), _as_int(action["y"], "y")
    xy = action.get("xy")
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        return _as_int(xy[0], "xy[0]"), _as_int(xy[1], "xy[1]")
    return None


def _scroll_dy(action: Mapping[str, Any]) -> int:
    if action.get("dy") is not None:
        return _as_int(action["dy"], "dy")
    direction = str(action.get("direction") or "down").strip().lower()
    if direction not in ("up", "down"):
        raise PlanError(f"scroll direction must be up/down, got {direction!r}")
    return -SCROLL_STEP if direction == "up" else SCROLL_STEP


def _find_element(state: Any, action: Mapping[str, Any]):
    elements = [e for e in getattr(state, "elements", []) if e.status == "ok"]
    element_id = action.get("element_id")
    if element_id is not None:
        for el in elements:
            if el.id == element_id:
                return el
        raise TargetNotFoundError(f"no ok element with id {element_id!r}")
    needle = str(action.get("label") or action.get("target") or "").strip().lower()
    if not needle:
        return None
    for el in elements:
        if needle in (el.label or "").lower() or needle in (el.text or "").lower():
            return el
    raise TargetNotFoundError(f"no ok element matching label {needle!r}")


def _frame_centre(state: Any) -> tuple:
    frame = getattr(state, "frame", None)
    if frame is None:
        raise PlanError("a scroll with no point needs a frame, and the state has none")
    ox, oy = frame.origin
    return int(ox) + int(frame.width) // 2, int(oy) + int(frame.height) // 2


def resolve_action(state: Any, action: Mapping[str, Any]) -> dict:
    """Resolve one action to an absolute-point record (no side effects)."""
    if not isinstance(action, Mapping):
        raise PlanError(f"each action must be an object, got {type(action).__name__}")
    kind = str(action.get("kind") or "").strip().lower()
    if kind not in ("click", "type", "scroll"):
        raise PlanError(f"unknown action kind {kind!r} (click|type|scroll)")

    point = _xy_of(action)
    element = None
    if point is None and kind != "scroll":
        element = _find_element(state, action)
        if element is None:
            raise PlanError(f"{kind} needs element_id, label or xy")
        point = (int(element.center[0]), int(element.center[1]))
    elif point is None:                       # a scroll with no explicit point
        element = _find_element(state, action)
        point = ((int(element.center[0]), int(element.center[1])) if element is not None
                 else _frame_centre(state))

    out: Dict[str, Any] = {"kind": kind, "x": point[0], "y": point[1]}
    if element is not None:
        out["element_id"] = element.id
        if element.label:
            out["label"] = element.label
    if kind == "type":
        out["text"] = str(action.get("text") or "")
    if kind == "scroll":
        out["dy"] = _scroll_dy(action)
    return out


def resolve_plan(state: Any, plan: Any) -> List[dict]:
    """Resolve a plan (``{"actions": [...]}`` or a bare list) against ``state``."""
    if isinstance(plan, Mapping):
        actions = plan.get("actions")
        if actions is None:
            raise PlanError("plan object needs an 'actions' list")
    elif isinstance(plan, (list, tuple)):
        actions = plan
    else:
        raise PlanError(f"plan must be an object or list, got {type(plan).__name__}")
    if not isinstance(actions, (list, tuple)):
        raise PlanError("'actions' must be a list")
    return [resolve_action(state, a) for a in actions]


def run_plan(actuator: "PixelActuator", resolved: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Apply resolved actions in order; return the actuator's history."""
    for action in resolved:
        actuator.apply(action)
    return list(actuator.calls)


# ------------------------------------------------------------------------ actuators
class _Recorder:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def _record(self, entry: dict) -> dict:
        self.calls.append(entry)
        return entry

    def apply(self, action: Mapping[str, Any]) -> dict:
        kind = str(action.get("kind") or "").strip().lower()
        point = _xy_of(action) or (None, None)
        if kind == "click":
            return self.click(point[0], point[1])
        if kind == "type":
            entry = self.type_text(str(action.get("text") or ""))
            entry["x"], entry["y"] = point
            return entry
        if kind == "scroll":
            entry = self.scroll(_scroll_dy(action))
            entry["x"], entry["y"] = point
            return entry
        raise PlanError(f"unknown action kind {kind!r} (click|type|scroll)")


class DryRunActuator(_Recorder):
    """Record every action; touch nothing. This is the default."""

    dry_run = True

    def click(self, x: Any, y: Any) -> dict:
        return self._record({"kind": "click", "x": _as_int(x, "x"), "y": _as_int(y, "y")})

    def type_text(self, text: str) -> dict:
        return self._record({"kind": "type", "text": str(text)})

    def scroll(self, dy: Any) -> dict:
        return self._record({"kind": "scroll", "dy": _as_int(dy, "dy")})


def _pick(backend: Any, *names: str) -> Optional[Callable[..., Any]]:
    for name in names:
        fn = None
        if isinstance(backend, Mapping):
            fn = backend.get(name)
        else:
            fn = getattr(backend, name, None)
        if callable(fn):
            return fn
    return None


class LiveActuator(_Recorder):
    """Dispatch click/type/scroll to an injected backend, then record what was sent.

    ``backend`` is an object (or mapping) exposing ``click(x, y)``, ``type_text`` or
    ``type`` (``text``), and ``scroll(dy)``. The three callables may also be passed
    directly as keyword arguments. A missing operation is refused up front, never
    silently skipped.
    """

    dry_run = False

    def __init__(self, backend: Any = None, *,
                 click: Optional[Callable[..., Any]] = None,
                 type_text: Optional[Callable[..., Any]] = None,
                 scroll: Optional[Callable[..., Any]] = None) -> None:
        super().__init__()
        if backend is not None:
            click = click or _pick(backend, "click")
            type_text = type_text or _pick(backend, "type_text", "type")
            scroll = scroll or _pick(backend, "scroll")
        missing = [n for n, fn in (("click", click), ("type", type_text),
                                   ("scroll", scroll)) if not callable(fn)]
        if missing:
            raise ActuatorRefused(
                "LiveActuator needs callable backend operations; missing: "
                + ", ".join(missing))
        self._click = click
        self._type = type_text
        self._scroll = scroll

    def click(self, x: Any, y: Any) -> dict:
        x, y = _as_int(x, "x"), _as_int(y, "y")
        self._click(x, y)
        return self._record({"kind": "click", "x": x, "y": y})

    def type_text(self, text: str) -> dict:
        text = str(text)
        self._type(text)
        return self._record({"kind": "type", "text": text})

    def scroll(self, dy: Any) -> dict:
        dy = _as_int(dy, "dy")
        self._scroll(dy)
        return self._record({"kind": "scroll", "dy": dy})


class HttpBackend:
    """One JSON POST per action: ``/click``, ``/type``, ``/scroll``.

    ``post`` is injectable so a test (or a caller with its own client) never opens a
    socket: it is called as ``post(url, payload)``. The default uses ``urllib``.
    """

    def __init__(self, base_url: str, post: Optional[Callable[[str, dict], Any]] = None,
                 timeout_s: float = 10.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self._post = post or self._urllib_post

    def _urllib_post(self, url: str, payload: dict) -> Any:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
            return resp.read()

    def click(self, x: Any, y: Any) -> Any:
        return self._post(self.base_url + "/click", {"x": int(x), "y": int(y)})

    def type_text(self, text: str) -> Any:
        return self._post(self.base_url + "/type", {"text": str(text)})

    #: Accept the verb spelling callers write in a plan/backend contract.
    type = type_text

    def scroll(self, dy: Any) -> Any:
        return self._post(self.base_url + "/scroll", {"dy": int(dy)})
