"""awrun `screen` kind: one crop-inspection batch queued as a job.

    # inline, from AwrunExecutor (the default): submit + dispatch_once(run_ids=[id])
    # or a dedicated worker, filtered to kind screen:
    python -m awscreen.awrun_kind --root ~/.aither/awrun --once

item.spec = ``{"manifest": <path>, "max_concurrency": int, "timeout_s": float}``. The
manifest and its crop files live under the LOCAL awrun root; the spec carries paths, never
pixels. The RunFn runs the same capped LocalExecutor and writes ``results.json`` beside
the manifest. Exit code is JOB HEALTH: 0 the batch ran and at least one crop was answered
(read the message for how many), 2 = the spec or manifest is malformed, the vision
endpoint is unavailable, or EVERY crop failed (asked, answered nothing). Code 2 for
"could not judge" is the SAME contract ``awscreen state`` uses, so a verifier sees one
verdict from either path.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

SCREEN_KIND = "screen"


def run_screen(item, *, run_root=None, inspector=None) -> Tuple[int, str]:
    from .executors import LocalExecutor, clamp_concurrency
    from .inspect import CropTask
    from .state import CropTransform

    spec = getattr(item, "spec", None) or {}
    man_path = spec.get("manifest") if isinstance(spec, dict) else None
    if not man_path:
        return 2, "malformed screen item: spec.manifest is required"
    man = Path(man_path)
    if run_root is not None:
        try:
            man.resolve().relative_to(Path(run_root).resolve())
        except ValueError:
            return 2, f"refused: manifest {man} is outside the awrun root {run_root}"
    try:
        data = json.loads(man.read_text(encoding="utf-8"))
        tasks = []
        for t in data["tasks"]:
            tf = t["transform"]
            tasks.append(CropTask(
                region_id=str(t["region_id"]),
                b64=base64.b64encode((man.parent / t["file"]).read_bytes()).decode("ascii"),
                media_type=t.get("media_type") or "image/png",
                transform=CropTransform(crop_box=tuple(tf["crop_box"]),
                                        scale=float(tf["scale"]),
                                        origin=tuple(tf.get("origin") or (0, 0)),
                                        crop_size=tuple(tf.get("crop_size") or (0, 0))),
                features=t.get("features") or {}, question=t.get("question") or ""))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return 2, f"malformed screen manifest {man}: {type(exc).__name__}: {exc}"
    if inspector is None:
        from .inspect import vision_inspector
        try:
            inspector = vision_inspector(timeout_s=float(spec.get("timeout_s") or 45.0))
        except Exception as exc:  # noqa: BLE001 - VisionUnavailable: say so, run nothing
            return 2, f"vision unavailable: {type(exc).__name__}: {exc}"
    results = LocalExecutor().run(
        tasks, inspector, max_concurrency=clamp_concurrency(spec.get("max_concurrency", 4)),
        timeout_s=float(spec.get("timeout_s") or 45.0))
    out = man.parent / "results.json"
    out.write_text(json.dumps({"results": [asdict(r) for r in results]}, indent=1),
                   encoding="utf-8")
    ok = sum(1 for r in results if r.status == "ok")
    if results and ok == 0:
        # The SAME batch exits 2 from `awscreen state`: inspection was asked for and the
        # vision endpoint answered no crop at all. "Could not judge" must not read as
        # "nothing on screen" on this path either.
        return 2, (f"vision answered 0 of {len(results)} crops; "
                   f"results at {out}")
    return 0, f"{ok}/{len(results)} crops answered; results at {out}"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m awscreen.awrun_kind",
                                 description="awrun worker for kind=screen items")
    ap.add_argument("--root", default=None, help="awrun root (default: awrun runs_dir)")
    ap.add_argument("--once", action="store_true", help="dispatch at most one item")
    args = ap.parse_args(argv)
    try:
        from awrun.dispatcher import dispatch_once
        from awrun.store import RunStore, runs_dir
    except ImportError as exc:
        print(f"DEAD: awrun is not installed: {exc}", file=sys.stderr)
        return 2
    from .executors import _NullRelay

    root = Path(args.root).expanduser() if args.root else runs_dir()
    store = RunStore(root)

    def fn(it):
        return run_screen(it, run_root=root)

    n = 0
    while True:
        item = dispatch_once(store, worker_id="awscreen-worker", run_fns={SCREEN_KIND: fn},
                             kinds=[SCREEN_KIND], relay_client=_NullRelay())
        if item is None:
            break
        n += 1
        print(f"{item.id} {item.status} {(item.result or {}).get('message', '')}".strip())
        if args.once:
            break
    if n == 0:
        print("no queued screen items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
