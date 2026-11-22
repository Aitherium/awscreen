"""Run crop inspections in parallel, bounded.

``LocalExecutor`` is a thread pool HARD-capped at 4 workers whatever is asked, with a
per-task deadline: a task still running ``timeout_s`` after it started, or never started
before the batch deadline, comes back as ``status='timeout'``; one that raised comes back
as ``status='error'``. Nothing is dropped. The cap and the deadlines are the memory guard
for a starved host -- nothing else on a Windows host protects it.

``AwrunExecutor`` puts the same batch into the LOCAL awrun queue as ONE item of kind
``screen`` (file paths only, no pixels in the spec) and runs it inline or waits for a
worker. It refuses any swarm or remote target: crops of a desktop are private.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .inspect import CropTask, Inspector, InspectResult

HARD_CAP = 4


class ExecutorRefused(RuntimeError):  # noqa: N818 - public name, mirrors VisionUnavailable
    """The executor will not run this batch where it was asked to; the message says why."""


class Executor(Protocol):
    name: str

    def run(self, tasks: List[CropTask], inspector: Inspector, *,
            max_concurrency: int = 4, timeout_s: float = 45.0) -> List[InspectResult]: ...


def clamp_concurrency(n) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = HARD_CAP
    return max(1, min(HARD_CAP, n))


class LocalExecutor:
    name = "local"

    def run(self, tasks: List[CropTask], inspector: Inspector, *,
            max_concurrency: int = 4, timeout_s: float = 45.0) -> List[InspectResult]:
        if not tasks:
            return []
        n = clamp_concurrency(max_concurrency)
        timeout_s = max(0.01, float(timeout_s))
        started: Dict[int, float] = {}
        lock = threading.Lock()

        def _run(i: int, task: CropTask) -> InspectResult:
            with lock:
                started[i] = time.monotonic()
            return inspector(task)

        results: Dict[int, InspectResult] = {}
        pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="awscreen-crop")
        futs = {pool.submit(_run, i, t): i for i, t in enumerate(tasks)}
        # Every task gets its own timeout_s once started; the batch as a whole gets
        # one wave of timeout_s per n tasks plus one, after which the unstarted are late.
        batch_deadline = time.monotonic() + timeout_s * (math.ceil(len(tasks) / n) + 1)
        pending = set(futs)
        try:
            while pending:
                now = time.monotonic()
                with lock:
                    run_deadlines = [started[futs[f]] + timeout_s for f in pending
                                     if futs[f] in started]
                nxt = min(run_deadlines + [batch_deadline])
                done, pending = wait(pending, timeout=max(0.0, min(nxt - now, 0.25)),
                                     return_when=FIRST_COMPLETED)
                for f in done:
                    i = futs[f]
                    try:
                        res = f.result()
                        if not isinstance(res, InspectResult):
                            raise TypeError(f"inspector returned {type(res).__name__}")
                        results[i] = res
                    except Exception as exc:  # noqa: BLE001 - becomes an explicit element
                        results[i] = InspectResult(
                            region_id=tasks[i].region_id, status="error",
                            error=f"{type(exc).__name__}: {str(exc)[:200]}")
                now = time.monotonic()
                late = set()
                for f in pending:
                    i = futs[f]
                    with lock:
                        st = started.get(i)
                    if st is not None and now - st >= timeout_s:
                        late.add(f)
                        results[i] = InspectResult(
                            region_id=tasks[i].region_id, status="timeout",
                            ms=round((now - st) * 1000, 1),
                            error=f"no answer within {timeout_s:g}s")
                    elif st is None and now >= batch_deadline:
                        late.add(f)
                        f.cancel()
                        results[i] = InspectResult(
                            region_id=tasks[i].region_id, status="timeout",
                            error="not started before the batch deadline")
                pending -= late
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return [results[i] for i in range(len(tasks))]


def _awrun_root(root) -> Path:
    if root:
        return Path(root)
    from awrun.store import runs_dir
    return runs_dir()


class _NullRelay:
    """dispatch_once broadcasts to the relay by default; a screen batch stays on-box."""

    def send_text(self, **_kw) -> None:
        return None


def write_manifest(tasks: List[CropTask], job_dir: Path, *, max_concurrency: int,
                   timeout_s: float) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, t in enumerate(tasks):
        ext = "png" if t.media_type == "image/png" else "jpg"
        name = f"crop_{i:03d}.{ext}"
        (job_dir / name).write_bytes(base64.b64decode(t.b64))
        entries.append({"region_id": t.region_id, "file": name, "media_type": t.media_type,
                        "question": t.question, "features": t.features,
                        "transform": asdict(t.transform)})
    man = job_dir / "manifest.json"
    man.write_text(json.dumps({"tasks": entries, "max_concurrency": max_concurrency,
                               "timeout_s": timeout_s}, indent=1), encoding="utf-8")
    return man


class AwrunExecutor:
    """One durable kind=screen item in the LOCAL awrun store; same LocalExecutor inside."""

    name = "awrun"

    def __init__(self, root=None, run_inline: bool = True, poll_s: float = 1.0,
                 wait_s: float = 300.0, target: str = "local") -> None:
        if str(target).lower() != "local" or str(root or "").startswith(("http:", "https:")):
            raise ExecutorRefused(
                f"awscreen refuses awrun target {target!r}: screen crops never go to a "
                "swarm or remote runner (the swarm repo is public, and runners are spend)")
        self.root = root
        self.run_inline = run_inline
        self.poll_s = poll_s
        self.wait_s = wait_s
        self.last_item_id = ""

    def run(self, tasks: List[CropTask], inspector: Optional[Inspector], *,
            max_concurrency: int = 4, timeout_s: float = 45.0) -> List[InspectResult]:
        if not tasks:
            return []
        from awrun.dispatcher import dispatch_once
        from awrun.store import RunStore

        from .awrun_kind import SCREEN_KIND, run_screen

        root = _awrun_root(self.root)
        store = RunStore(root)
        job_dir = root / "screen" / uuid.uuid4().hex[:12]
        man = write_manifest(tasks, job_dir, max_concurrency=clamp_concurrency(
            max_concurrency), timeout_s=float(timeout_s))
        item = store.submit(SCREEN_KIND, {"manifest": str(man),
                                          "max_concurrency": clamp_concurrency(max_concurrency),
                                          "timeout_s": float(timeout_s)})
        self.last_item_id = item.id
        if self.run_inline:
            def fn(it):
                return run_screen(it, run_root=root, inspector=inspector)

            dispatch_once(store, worker_id="awscreen-inline", run_fns={SCREEN_KIND: fn},
                          kinds=[SCREEN_KIND], run_ids=[item.id], relay_client=_NullRelay())
        else:
            deadline = time.monotonic() + self.wait_s
            while time.monotonic() < deadline:
                cur = store.get(item.id)
                if cur is not None and cur.status in ("done", "failed", "cancelled"):
                    break
                time.sleep(self.poll_s)
        res_path = job_dir / "results.json"
        if not res_path.is_file():
            cur = store.get(item.id)
            raise RuntimeError(f"awrun item {item.id} produced no results.json "
                               f"(status={getattr(cur, 'status', '?')}, "
                               f"result={getattr(cur, 'result', None)})")
        data = json.loads(res_path.read_text(encoding="utf-8"))
        known = set(InspectResult.__dataclass_fields__)
        return [InspectResult(**{k: v for k, v in r.items() if k in known})
                for r in data.get("results") or []]
