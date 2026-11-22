"""Bounded parallel inspection: the 4 cap, timeouts and errors as results, parsing. Offline."""
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

import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402

import pytest  # noqa: E402
from _synthetic import draw_screen  # noqa: E402
from awscreen.crops import make_crop  # noqa: E402
from awscreen.executors import HARD_CAP, LocalExecutor, clamp_concurrency  # noqa: E402
from awscreen.inspect import CropTask, InspectResult, parse_reply  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call in an offline test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def _tasks(n, extra=()):
    img, _ = draw_screen()
    tf = make_crop(img, (0, 0, 10, 10))[1]
    ids = [f"r{i}" for i in range(n)] + list(extra)
    return [CropTask(region_id=i, b64="", media_type="image/png", transform=tf) for i in ids]


def _inspector(peak, slow=(), boom=(), sleep_s=0.03):
    lock = threading.Lock()

    def insp(task):
        with lock:
            peak["now"] = peak.get("now", 0) + 1
            peak["max"] = max(peak.get("max", 0), peak["now"])
        try:
            if task.region_id in boom:
                raise RuntimeError("model exploded")
            time.sleep(1.5 if task.region_id in slow else sleep_s)
            return InspectResult(region_id=task.region_id, status="ok", role="button",
                                 label="x", confidence=0.7)
        finally:
            with lock:
                peak["now"] -= 1
    return insp


def test_cap_holds_when_sixteen_are_asked():
    peak = {}
    res = LocalExecutor().run(_tasks(20), _inspector(peak), max_concurrency=16, timeout_s=5)
    assert peak["max"] <= HARD_CAP and len(res) == 20
    assert all(r.status == "ok" for r in res)
    assert clamp_concurrency(16) == 4 and clamp_concurrency(0) == 1


def test_cap_slow_task_times_out_explicitly():
    peak = {}
    res = LocalExecutor().run(_tasks(3, extra=("slow",)), _inspector(peak, slow=("slow",)),
                              max_concurrency=16, timeout_s=0.3)
    by = {r.region_id: r for r in res}
    assert by["slow"].status == "timeout" and "0.3" in by["slow"].error
    assert all(by[f"r{i}"].status == "ok" for i in range(3))
    assert peak["max"] <= HARD_CAP


def test_raising_task_becomes_error_not_dropped():
    res = LocalExecutor().run(_tasks(2, extra=("bad",)), _inspector({}, boom=("bad",)),
                              max_concurrency=2, timeout_s=5)
    assert len(res) == 3
    bad = [r for r in res if r.region_id == "bad"][0]
    assert bad.status == "error" and "exploded" in bad.error


def test_order_is_preserved():
    res = LocalExecutor().run(_tasks(9), _inspector({}), max_concurrency=4, timeout_s=5)
    assert [r.region_id for r in res] == [f"r{i}" for i in range(9)]


def test_parse_reply_json_fenced_and_text():
    a = parse_reply('{"role":"button","label":"Save","text":"SAVE"}')
    assert a == {"role": "button", "label": "Save", "text": "SAVE", "structured": True}
    b = parse_reply('Sure!\n```json\n{"role": "LINK", "label": "Docs"}\n```')
    assert b["role"] == "link" and b["label"] == "Docs" and b["structured"]
    c = parse_reply("A blue button that says Save")
    assert c["role"] == "other" and not c["structured"] and "Save" in c["text"]
    d = parse_reply('{"role":"spaceship"}')
    assert d["role"] == "other"


def test_vision_inspector_turns_failure_into_timeout(monkeypatch):
    import awvision.vision as v
    from awscreen import inspect as insp_mod

    monkeypatch.delenv("AWSCREEN_VLM_MODEL", raising=False)
    monkeypatch.delenv("AWVISION_MODEL", raising=False)
    monkeypatch.setattr(v, "resolve_pinned", lambda require_scheduler=True: ("https://h:1", "m"))

    def fail(*a, **k):
        raise RuntimeError("Vision service at https://h:1 did not answer within 45s.")
    monkeypatch.setattr(v, "get_vision_response_b64", fail)
    inspector = insp_mod.vision_inspector()
    assert inspector.endpoint_host == "h" and inspector.model == "m"
    r = inspector(_tasks(1)[0])
    assert r.status == "timeout" and "did not answer" in r.error


def test_vision_inspector_passes_short_answer_and_timeout(monkeypatch):
    import awvision.vision as v
    from awscreen import inspect as insp_mod

    seen = {}

    def ok(b64, media, q, ep, model, *, max_tokens=None, timeout=None):
        seen.update(ep=ep, model=model, max_tokens=max_tokens, timeout=timeout)
        return '{"role":"button","label":"ignite","text":"ignite"}'
    monkeypatch.setattr(v, "get_vision_response_b64", ok)
    inspector = insp_mod.vision_inspector(endpoint="https://h:2", model="vlm", timeout_s=12)
    r = inspector(_tasks(1)[0])
    assert r.status == "ok" and r.label == "ignite" and r.role == "button"
    assert seen == {"ep": "https://h:2", "model": "vlm", "max_tokens": 96, "timeout": 12}


def _model_seeing(monkeypatch):
    import awvision.vision as v
    seen = {}

    def ok(b64, media, q, ep, model, *, max_tokens=None, timeout=None):
        seen["model"] = model
        return '{"role":"button","label":"ok","text":""}'
    monkeypatch.setattr(v, "resolve_pinned",
                        lambda require_scheduler=True: ("https://h:1", "pinned-model"))
    monkeypatch.setattr(v, "get_vision_response_b64", ok)
    return seen


def test_crop_model_default_is_the_5090_27b(monkeypatch):
    import awvision.vision as v
    from awscreen import inspect as insp_mod

    monkeypatch.delenv("AWSCREEN_VLM_MODEL", raising=False)
    monkeypatch.delenv("AWVISION_MODEL", raising=False)
    assert insp_mod.DEFAULT_VLM_MODEL == "bonsai2-27b-5090"
    assert insp_mod.VLM_MODEL_ENV == "AWSCREEN_VLM_MODEL"
    assert insp_mod.DEFAULT_VLM_MODEL == v._VISION_MODELS[0]


def test_crop_model_override_precedence(monkeypatch):
    from awscreen import inspect as insp_mod

    env = {"AWSCREEN_VLM_MODEL": "env-model", "AWVISION_MODEL": "awvision-model"}
    assert insp_mod.resolve_vlm_model("flag-model", env) == "flag-model"
    assert insp_mod.resolve_vlm_model(None, env) == "env-model"
    assert insp_mod.resolve_vlm_model(None, {"AWVISION_MODEL": "awvision-model"}) == \
        "awvision-model"
    assert insp_mod.resolve_vlm_model(None, {}) is None


def test_vision_inspector_flag_beats_the_pin(monkeypatch):
    from awscreen import inspect as insp_mod

    monkeypatch.delenv("AWSCREEN_VLM_MODEL", raising=False)
    seen = _model_seeing(monkeypatch)
    inspector = insp_mod.vision_inspector(model="flag-model")
    inspector(_tasks(1)[0])
    assert inspector.model == "flag-model" and seen["model"] == "flag-model"


def test_vision_inspector_env_beats_the_pin(monkeypatch):
    from awscreen import inspect as insp_mod

    monkeypatch.setenv("AWSCREEN_VLM_MODEL", "env-model")
    monkeypatch.delenv("AWVISION_MODEL", raising=False)
    seen = _model_seeing(monkeypatch)
    inspector = insp_mod.vision_inspector()
    inspector(_tasks(1)[0])
    assert inspector.model == "env-model" and seen["model"] == "env-model"


def test_vision_inspector_falls_back_to_the_pinned_model(monkeypatch):
    from awscreen import inspect as insp_mod

    monkeypatch.delenv("AWSCREEN_VLM_MODEL", raising=False)
    monkeypatch.delenv("AWVISION_MODEL", raising=False)
    seen = _model_seeing(monkeypatch)
    inspector = insp_mod.vision_inspector()
    inspector(_tasks(1)[0])
    assert inspector.model == "pinned-model" and seen["model"] == "pinned-model"
