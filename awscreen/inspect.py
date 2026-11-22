"""Ask a small vision model about ONE crop, through the awvision resolver.

The endpoint and model are resolved ONCE before a fan-out (``awvision.resolve_pinned``:
locked, one ``/v1/models`` probe, refuses the plain-http mesh fallback), so every crop of a
screen goes to the same scheduler-routed model with a short prompt and a short answer.
The model defaults to ``bonsai2-27b-5090`` (:data:`DEFAULT_VLM_MODEL`); ``--model`` on
``awscreen state`` beats ``AWSCREEN_VLM_MODEL``, which beats awvision's own settings.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

from .state import ROLES, CropTransform

# The crop-inspection model. 2026-11-20 owner decision, measured on the same three UI
# crops: bonsai2-27b-5090 labelled roles correctly (text/button) where bonsai-4b-5090
# got them wrong, for 1-2 s more per crop (1.3-2.6 s vs 0.5-1.5 s). So the 27B is the
# default here; the 4B lane is freed for OCR duty (see CHANGELOG; the OCR proposer
# already exists and skips honestly when no engine is installed). Override, in order:
# the ``--model`` CLI flag, then ``AWSCREEN_VLM_MODEL``, then awvision's own
# ``AWVISION_MODEL`` / host-config model.
DEFAULT_VLM_MODEL = "bonsai2-27b-5090"
VLM_MODEL_ENV = "AWSCREEN_VLM_MODEL"

CROP_PROMPT = (
    "This is a small crop of a computer screen. Reply with ONLY a JSON object: "
    '{"role": one of button|input|text|icon|link|image|other|none, '
    '"label": short name of the control or content, '
    '"text": the exact visible text, copied character for character (keep case, '
    'punctuation and spacing), or "" if there is none}.'
)


@dataclass
class CropTask:
    region_id: str
    b64: str
    media_type: str
    transform: CropTransform
    features: dict = field(default_factory=dict)
    question: str = CROP_PROMPT


@dataclass
class InspectResult:
    region_id: str
    status: str                  # ok | timeout | error | skipped
    role: str = "none"
    label: str = ""
    text: str = ""
    confidence: float = 0.0
    raw: str = ""
    ms: float = 0.0
    error: str = ""


Inspector = Callable[[CropTask], InspectResult]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_reply(raw: str) -> dict:
    """``{role, label, text, structured}`` from a model reply: JSON, fenced JSON, or text."""
    s = (raw or "").strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    obj = None
    for cand in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s and "}" in s else ""):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            break
        except ValueError:
            continue
    if isinstance(obj, dict):
        role = str(obj.get("role") or "other").strip().lower()
        return {"role": role if role in ROLES else "other",
                "label": str(obj.get("label") or "").strip()[:120],
                "text": str(obj.get("text") or "").strip()[:500], "structured": True}
    text = (raw or "").strip()
    return {"role": "other", "label": text[:60], "text": text[:500], "structured": False}


def resolve_vlm_model(explicit: Optional[str] = None,
                      environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The model named by the caller: flag, then ``AWSCREEN_VLM_MODEL``, then
    ``AWVISION_MODEL``; ``None`` when nothing is named, so the awvision host config and
    then :data:`DEFAULT_VLM_MODEL` decide."""
    env = os.environ if environ is None else environ
    for name in (explicit, env.get(VLM_MODEL_ENV), env.get("AWVISION_MODEL")):
        if name and name.strip():
            return name.strip()
    return None


def vision_inspector(endpoint: Optional[str] = None, model: Optional[str] = None,
                     max_tokens: int = 96, timeout_s: float = 45.0) -> Inspector:
    """An Inspector over awvision. Resolves (endpoint, model) NOW, once; raises
    ``awvision.vision.VisionUnavailable`` when nothing usable answers."""
    from awvision.vision import get_vision_response_b64, resolve_model, resolve_pinned

    named = resolve_vlm_model(model)
    if endpoint is None:
        endpoint, pinned_model = resolve_pinned(require_scheduler=True)
        model = named or pinned_model or DEFAULT_VLM_MODEL
    else:
        model = named or resolve_model(endpoint)

    def inspect(task: CropTask) -> InspectResult:
        t0 = time.perf_counter()
        try:
            raw = get_vision_response_b64(task.b64, task.media_type, task.question,
                                          endpoint, model, max_tokens=max_tokens,
                                          timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 - a failed crop is a result, not a crash
            status = "timeout" if "did not answer within" in str(exc) else "error"
            return InspectResult(region_id=task.region_id, status=status,
                                 ms=round((time.perf_counter() - t0) * 1000, 1),
                                 error=f"{type(exc).__name__}: {str(exc)[:200]}")
        p = parse_reply(raw)
        return InspectResult(region_id=task.region_id, status="ok", role=p["role"],
                             label=p["label"], text=p["text"],
                             confidence=0.7 if p["structured"] else 0.4, raw=raw[:400],
                             ms=round((time.perf_counter() - t0) * 1000, 1))

    inspect.endpoint_host = urlparse(endpoint).hostname or ""  # type: ignore[attr-defined]
    inspect.model = model  # type: ignore[attr-defined]
    return inspect
