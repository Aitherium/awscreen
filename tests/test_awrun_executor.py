"""awrun kind=screen: dispatch round-trip, malformed specs, swarm refused."""
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
import re  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("awrun.store")

from _synthetic import draw_screen  # noqa: E402
from awrun.dispatcher import dispatch_once  # noqa: E402
from awrun.store import RunStore  # noqa: E402
from awscreen.awrun_kind import SCREEN_KIND, run_screen  # noqa: E402
from awscreen.crops import crop_b64, make_crop  # noqa: E402
from awscreen.executors import (  # noqa: E402
    AwrunExecutor,
    ExecutorRefused,
    _NullRelay,
    write_manifest,
)
from awscreen.inspect import CropTask, InspectResult  # noqa: E402


def _fake(task):
    return InspectResult(region_id=task.region_id, status="ok", role="button",
                         label="save", text="SAVE", confidence=0.7)


def _tasks():
    img, t = draw_screen()
    out = []
    for i, k in enumerate(("save", "cancel")):
        crop, tf = make_crop(img, tuple(t[k]))
        out.append(CropTask(region_id=f"r{i}", b64=crop_b64(crop), media_type="image/png",
                            transform=tf))
    return out


def test_screen_item_round_trips_through_dispatch_once(tmp_path):
    store = RunStore(tmp_path)
    man = write_manifest(_tasks(), tmp_path / "screen" / "job1", max_concurrency=4,
                         timeout_s=5)
    item = store.submit(SCREEN_KIND, {"manifest": str(man), "max_concurrency": 4,
                                      "timeout_s": 5})
    assert item.gpu is None
    done = dispatch_once(store, worker_id="t",
                         run_fns={SCREEN_KIND: lambda it: run_screen(
                             it, run_root=tmp_path, inspector=_fake)},
                         kinds=[SCREEN_KIND], run_ids=[item.id], relay_client=_NullRelay())
    assert done is not None and done.status == "done"
    res = json.loads((man.parent / "results.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in res["results"]] == ["ok", "ok"]


def test_malformed_spec_is_code_2(tmp_path):
    class Item:
        spec = {}
    assert run_screen(Item())[0] == 2

    class Outside:
        spec = {"manifest": str(tmp_path.parent / "elsewhere.json")}
    code, msg = run_screen(Outside(), run_root=tmp_path)
    assert code == 2 and "outside" in msg


def test_executor_inline(tmp_path):
    ex = AwrunExecutor(root=tmp_path)
    res = ex.run(_tasks(), _fake, max_concurrency=16, timeout_s=5)
    assert [r.status for r in res] == ["ok", "ok"]
    item = RunStore(tmp_path).get(ex.last_item_id)
    assert item.kind == "screen" and item.status == "done"
    # a KEY naming base64 pixels; a random hex path segment can contain "b64"
    # paths only, never pixels
    assert not re.search(r"\"[^\"]*b64[^\"]*\"\s*:", json.dumps(item.spec))


@pytest.mark.parametrize("target", ["swarm", "remote", "aws"])
def test_executor_refuses_swarm_and_remote(target):
    with pytest.raises(ExecutorRefused):
        AwrunExecutor(target=target)
    with pytest.raises(ExecutorRefused):
        AwrunExecutor(root="https://example.invalid/awrun")


def test_manifest_round_trips_the_exact_crop_transform(tmp_path):
    """write_manifest ships the crop transform to an out-of-process worker; the actual
    crop_size must survive the JSON round-trip, or that worker maps a crop box back
    several px off on the short axis of a long flat crop."""
    from PIL import Image
    img = Image.new("RGB", (3840, 600), (0, 0, 0))
    crop, tf = make_crop(img, (0, 100, 3840, 130), margin=8, max_side=448)
    task = CropTask(region_id="r0", b64=crop_b64(crop), media_type="image/png",
                    transform=tf)
    man = write_manifest([task], tmp_path / "screen" / "job1", max_concurrency=1,
                         timeout_s=5)
    doc = json.loads(man.read_text(encoding="utf-8"))
    assert tuple(doc["tasks"][0]["transform"]["crop_size"]) == crop.size

    seen = {}

    def capture(t):
        seen["tf"] = t.transform
        return InspectResult(region_id=t.region_id, status="ok")

    class Item:
        spec = {"manifest": str(man), "max_concurrency": 1, "timeout_s": 5}
    code, _ = run_screen(Item(), run_root=tmp_path, inspector=capture)
    assert code == 0 and seen["tf"].crop_size == crop.size
    _, by = seen["tf"].to_image(crop.width, crop.height)
    assert abs(by - seen["tf"].crop_box[3]) < 1e-6


def _all_fail(task):
    return InspectResult(region_id=task.region_id, status="error",
                         error="gateway 502")


def test_zero_answers_is_code_2_the_same_as_awscreen_state(tmp_path):
    """Inspection was asked for and the vision endpoint answered NOTHING: this path must
    exit 2 ("could not judge"), exactly as `awscreen state` does -- not 0, which would
    read as "nothing was on screen"."""
    tasks = _tasks()
    man = write_manifest(tasks, tmp_path / "screen" / "job2", max_concurrency=4,
                         timeout_s=5)

    class Item:
        spec = {"manifest": str(man), "max_concurrency": 4, "timeout_s": 5}

    code, msg = run_screen(Item(), run_root=tmp_path, inspector=_all_fail)
    assert code == 2, msg
    assert "0 of 2" in msg
