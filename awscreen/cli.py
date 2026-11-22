"""awscreen CLI — find and locate clickable elements by description.

Exit codes:
  0 = elements found and displayed
  1 = no elements found matching description
  2 = cannot judge (API error, missing key, file not found)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

from .finder import Finder


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    _av = argv if argv is not None else sys.argv[1:]
    if _av[:1] == ["state"]:
        return state_main(list(_av[1:]))
    if _av[:1] == ["act"]:
        return act_main(list(_av[1:]))
    if _av[:1] == ["demo"]:
        from .demo import main as _demo_main
        return _demo_main(list(_av[1:]))
    ap = argparse.ArgumentParser(
        prog="awscreen",
        description="Find clickable elements by visual description",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test (offline, no API)",
    )
    ap.add_argument(
        "image",
        nargs="?",
        help="Path to screenshot image (PNG or JPEG)",
    )
    ap.add_argument(
        "description",
        nargs="?",
        help="Natural language description of element to find",
    )
    ap.add_argument(
        "--endpoint",
        help="local vision endpoint (or set AWSCREEN_URL / AWVISION_URL)",
    )
    ap.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )

    args = ap.parse_args(argv)

    if args.self_test:
        rc = _self_test()
        from ._selftest_state import main as _state_self_test
        return rc or _state_self_test()

    if not args.image or not args.description:
        ap.print_help()
        return 2

    # No credential: this talks to loopback. See finder.py for why.
    endpoint = getattr(args, "endpoint", None)

    try:
        finder = Finder(endpoint=endpoint)
    except ValueError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    try:
        screenshot = finder.load_image(args.image)
    except FileNotFoundError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    except (ValueError, ImportError) as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    try:
        elements = finder.find(screenshot, args.description)
    except ValueError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    if not elements:
        if args.format == "json":
            print(json.dumps({"found": 0, "elements": []}))
        else:
            print("No elements found matching description")
        return 1

    if args.format == "json":
        result = {
            "found": len(elements),
            "elements": [
                {
                    "x": e.x,
                    "y": e.y,
                    "width": e.width,
                    "height": e.height,
                    "confidence": e.confidence,
                    "description": e.description,
                }
                for e in elements
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        for i, elem in enumerate(elements, 1):
            print(
                f"Match {i}: ({elem.x}, {elem.y}) "
                f"[{elem.width}x{elem.height}] "
                f"confidence={elem.confidence:.2f} "
                f'"{elem.description}"'
            )

    return 0


def state_main(argv: list[str]) -> int:
    """``awscreen state``: one frame -> a ScreenState JSON.

    Exit: 0 at least one element; 1 no elements; 2 could not judge (unreadable image,
    an explicitly requested proposer unavailable, inspection asked for and the
    vision endpoint could not answer a single crop, or a frame that is not the monitor
    it was claimed to be).
    """
    ap = argparse.ArgumentParser(
        prog="awscreen state",
        description="What is on screen and where: regions -> crops -> parallel "
                    "inspection -> one ScreenState JSON")
    ap.add_argument("image", nargs="?", help="screenshot (PNG/JPEG)")
    ap.add_argument("--screen", action="store_true",
                    help="capture a monitor (the primary unless --monitor is given)")
    ap.add_argument("--monitor", type=int, default=None, metavar="N",
                    help="1-based monitor index (mss order); --screen captures it, and "
                         "an image file is VERIFIED to be that monitor's pixels. Without "
                         "it: the primary monitor, origin 0,0 for a file.")
    ap.add_argument("--proposer", default="auto",
                    choices=["auto", "omniparser", "ocr", "blobs", "grid"])
    ap.add_argument("--no-inspect", action="store_true",
                    help="neurons only: zero model calls")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip the local OCR text pass (text comes only from hints/VLM)")
    ap.add_argument("--text-only", action="store_true",
                    help="read only the exact strings for a region list (OCR, then VLM)")
    ap.add_argument("--regions", default=None,
                    help="region list for --text-only: a ScreenState JSON, "
                         "{'regions': [...]}, or a list of [x1,y1,x2,y2] boxes")
    ap.add_argument("--max-regions", type=int, default=24,
                    help="cap on model calls for this frame (default 24)")
    ap.add_argument("--concurrency", type=int, default=4, help="in flight, hard-capped at 4")
    ap.add_argument("--timeout", type=float, default=45.0, help="seconds per crop")
    ap.add_argument("--budget-ms", type=float, default=None,
                    help="wall-clock budget for the whole pipeline, in ms: neurons always "
                         "run, model calls continue only while a whole crop still fits, "
                         "and the rest are returned with status 'budget' (default: "
                         "AWSCREEN_BUDGET_MS, else unbounded)")
    ap.add_argument("--executor", default="local", choices=["local", "awrun"])
    ap.add_argument("--model", default=None,
                    help="crop-inspection vision model (default: AWSCREEN_VLM_MODEL, else "
                         "awscreen's bonsai2-27b-5090)")
    ap.add_argument("--dom", help="DOM snapshot JSON to join onto the pixels")
    ap.add_argument("--uia", nargs="?", const="auto", default=None,
                    help="UI Automation tree: omit the value (or pass 'auto') to walk the "
                         "focused window's UIA tree on Windows, or pass a recorded tree "
                         "JSON. Skipped with a printed reason when no bridge is available.")
    ap.add_argument("--prev", help="previous ScreenState JSON; identical regions are reused "
                                   "with zero model calls (whole-frame or per-region)")
    ap.add_argument("--origin", default=None,
                    help="desktop px of image (0,0), e.g. 1920,0 (default: the captured "
                         "or claimed monitor's real origin, else 0,0)")
    ap.add_argument("--out", help="also write the ScreenState JSON here")
    ap.add_argument("--json", action="store_true", help="print the full JSON")
    ap.add_argument("--self-test", action="store_true", help="offline checks, no network")
    args = ap.parse_args(argv)

    if args.self_test:
        from ._selftest_state import main as _state_self_test
        return _state_self_test()
    if not args.image and not args.screen:
        ap.print_help()
        return 2
    if args.text_only:
        return _text_only(args)

    from .capture import CaptureError
    from .state import ProposerUnavailable, ScreenState, build_screen_state

    try:
        dom = json.loads(Path(args.dom).read_text(encoding="utf-8")) if args.dom else None
        if args.uia and args.uia != "auto":
            uia_arg: object = json.loads(Path(args.uia).read_text(encoding="utf-8-sig"))
        else:
            uia_arg = args.uia
        prev = (ScreenState.from_dict(json.loads(Path(args.prev).read_text(encoding="utf-8")))
                if args.prev else None)
        origin = (tuple(int(v) for v in args.origin.split(","))
                  if args.origin else None)
    except (OSError, ValueError) as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    executor = None
    if args.executor == "awrun":
        from .executors import AwrunExecutor
        executor = AwrunExecutor()
    explicit = args.proposer != "auto"
    chain = ("omniparser", "ocr", "blobs", "grid") if not explicit else (args.proposer,)
    from .budget import resolve_budget_ms
    budget_ms = resolve_budget_ms(args.budget_ms)
    try:
        state = build_screen_state(
            args.image, screen=args.screen, proposers=chain, explicit_proposer=explicit,
            inspect=not args.no_inspect, max_regions=args.max_regions,
            executor=executor, max_concurrency=args.concurrency, timeout_s=args.timeout,
            dom=dom, uia=uia_arg, prev=prev, origin=origin, vlm_model=args.model,
            monitor=args.monitor, ocr=(False if args.no_ocr else None),
            budget_ms=budget_ms)
    except ProposerUnavailable as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    except CaptureError as exc:
        # A wrong frame/origin is "could not judge", never a guess with a 1920-px offset.
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"DEAD: cannot read image: {exc}", file=sys.stderr)
        return 2

    doc = state.to_dict()
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(json.dumps(doc, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps(doc, indent=1))
    else:
        t = state.timing
        print(f"proposer={state.proposer.get('ran') or 'none'} regions={len(state.regions)} "
              f"elements={len(state.elements)} model_calls={state.model.get('calls', 0)} "
              f"reused={state.reused} reused_regions={state.reused_regions} "
              f"total_ms={t.get('total_ms')}")
        for e in state.elements:
            print(f"  {e.id:>4} {e.role:<7} {e.status:<7} {e.bbox} "
                  f"[{','.join(e.sources)}] {e.label[:40]!r} {e.text[:40]!r}")

    u = state.uia or {}
    if u.get("attempted"):
        if u.get("ok"):
            print(f"uia: {u.get('nodes', 0)} node(s) from {u.get('source')}; "
                  f"matched {u.get('matched', 0)} added {u.get('added', 0)} "
                  f"skipped {u.get('skipped', 0)}", file=sys.stderr)
        else:
            print(f"uia: skipped ({u.get('reason')})", file=sys.stderr)

    m = state.model
    if not args.no_inspect and not state.reused:
        if m.get("error"):
            print(f"DEAD: vision inspection could not run: {m['error']}", file=sys.stderr)
            return 2
        if m.get("calls", 0) > 0 and m.get("ok", 0) == 0:
            print(f"DEAD: vision endpoint answered 0 of {m['calls']} crops "
                  f"(see elements with status timeout/error)", file=sys.stderr)
            return 2
    return 0 if state.elements else 1


def act_main(argv: list[str]) -> int:
    """``awscreen act``: resolve a plan against a ScreenState and (dry-)run it.

    Prints the ABSOLUTE points each action resolves to -- the exact desktop coordinates a
    click would land on -- then the recorded calls. Dry run is the default; ``--live``
    with ``--endpoint`` sends each action to an HTTP backend. Nothing here ever clicks a
    window unless the caller explicitly asks for it.

    Exit: 0 resolved (and run); 1 a plan target is not in the state; 2 could not judge
    (unreadable file, malformed plan, --live without --endpoint).
    """
    from . import actuator as _act
    from .state import ScreenState

    ap = argparse.ArgumentParser(
        prog="awscreen act",
        description="Resolve a JSON action plan against a ScreenState to absolute points")
    ap.add_argument("--plan", required=True, help="actions JSON: {'actions': [...]} or [...]")
    ap.add_argument("--state", required=True, help="ScreenState JSON from `awscreen state`")
    ap.add_argument("--dry-run", action="store_true",
                    help="record only, change nothing (this is also the default)")
    ap.add_argument("--live", action="store_true",
                    help="send actions to --endpoint (mutually exclusive with --dry-run)")
    ap.add_argument("--endpoint", help="HTTP backend base URL for --live")
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per HTTP action")
    ap.add_argument("--format", choices=["json", "text"], default="json",
                    help="where the resolved points go (default json)")
    ap.add_argument("--out", help="also write the resolved plan JSON here")
    args = ap.parse_args(argv)

    if args.live and args.dry_run:
        print("DEAD: choose --dry-run or --live, not both", file=sys.stderr)
        return 2
    live = bool(args.live)
    if live and not args.endpoint:
        print("DEAD: --live needs --endpoint", file=sys.stderr)
        return 2

    try:
        state_doc = json.loads(Path(args.state).expanduser().read_text(encoding="utf-8"))
        plan = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
        state = ScreenState.from_dict(state_doc)
    except (OSError, ValueError) as exc:
        print(f"DEAD: cannot read plan/state: {exc}", file=sys.stderr)
        return 2

    try:
        resolved = _act.resolve_plan(state, plan)
    except _act.TargetNotFoundError as exc:
        print(f"NOT FOUND: {exc}", file=sys.stderr)
        return 1
    except _act.PlanError as exc:
        print(f"DEAD: {exc}", file=sys.stderr)
        return 2

    try:
        if live:
            backend = _act.HttpBackend(args.endpoint, timeout_s=args.timeout)
            machine: _act.PixelActuator = _act.LiveActuator(backend)
        else:
            machine = _act.DryRunActuator()
        calls = _act.run_plan(machine, resolved)
    except (_act.ActuatorRefused, OSError) as exc:
        print(f"DEAD: cannot act: {exc}", file=sys.stderr)
        return 2

    result = {"dry_run": bool(machine.dry_run), "actions": resolved, "calls": calls}
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(result, indent=1))
    else:
        mode = "dry-run (nothing sent)" if machine.dry_run else "live"
        print(f"{mode}: {len(resolved)} action(s)")
        for a in resolved:
            extra = ""
            if a["kind"] == "type":
                extra = f" text={a['text']!r}"
            elif a["kind"] == "scroll":
                extra = f" dy={a['dy']}"
            print(f"  {a['kind']:<6} ({a['x']}, {a['y']}){extra}")
    return 0
def _load_regions(path: str):
    """Regions from a ScreenState JSON, a {'regions': [...]} record, or a bare box list."""
    from .state import Region, ScreenState

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and doc.get("schema") == "awscreen.screen_state/1":
        return list(ScreenState.from_dict(doc).regions)
    items = doc.get("regions") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        raise ValueError("expected a ScreenState, {'regions': [...]}, or a list of boxes")
    out = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("bbox_image")
            if not bbox or len(bbox) != 4:
                raise ValueError(f"region {i} has no [x1,y1,x2,y2] bbox")
            out.append(Region(id=str(item.get("id") or f"r{i}"),
                              bbox=tuple(int(v) for v in bbox),
                              proposer=str(item.get("proposer") or "list")))
        else:
            if len(item) != 4:
                raise ValueError(f"region {i} is not [x1,y1,x2,y2]")
            out.append(Region(id=f"r{i}", bbox=tuple(int(v) for v in item),
                              proposer="list"))
    return out


def _text_only(args) -> int:
    """``--text-only``: one exact string per region, OCR first and VLM only for the rest.

    Exit: 0 at least one string, 1 a lane ran and found none, 2 no lane could run.
    """
    from PIL import Image

    from .inspect import vision_inspector
    from .ocr import NullOcrProvider, make_ocr_provider, read_region_text
    from .regions import make_proposer
    from .regions import propose as _propose

    image = args.image
    if args.screen:
        from awvision.sight import capture_screen_origin
        capture = str(Path.home() / ".aither" / "awscreen" / "text_only_capture.jpg")
        Path(capture).parent.mkdir(parents=True, exist_ok=True)
        image, _origin = capture_screen_origin(capture)
    if not image or not Path(image).is_file():
        print(f"DEAD: image not found: {image}", file=sys.stderr)
        return 2
    try:
        with Image.open(image) as raw:
            img = raw.convert("RGB")
    except (OSError, ValueError) as exc:
        print(f"DEAD: cannot read image: {exc}", file=sys.stderr)
        return 2

    if args.regions:
        try:
            regions = _load_regions(args.regions)
        except (OSError, ValueError, KeyError) as exc:
            print(f"DEAD: cannot read region list: {exc}", file=sys.stderr)
            return 2
    else:
        explicit = args.proposer != "auto"
        chain = ("omniparser", "ocr", "blobs", "grid") if not explicit else (args.proposer,)
        built = [make_proposer(n) for n in chain]
        if explicit and len(built) == 1:
            ok, why = built[0].available()
            if not ok:
                print(f"DEAD: proposer {built[0].name!r} unavailable: {why}", file=sys.stderr)
                return 2
        regions, _report = _propose(img, built)
        if not regions:
            print("no regions proposed")
            return 1

    provider = (NullOcrProvider("disabled by --no-ocr") if args.no_ocr
                else make_ocr_provider("tesseract"))
    factory = None
    if not args.no_inspect:
        factory = lambda: vision_inspector(model=args.model, timeout_s=args.timeout)  # noqa: E731
    result = read_region_text(img, regions, ocr_provider=provider,
                              inspector_factory=factory, max_concurrency=args.concurrency,
                              timeout_s=args.timeout, use_vlm=not args.no_inspect)
    doc = {"schema": "awscreen.text_only/1", **result.to_dict()}
    if args.out:
        Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).expanduser().write_text(json.dumps(doc, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps(doc, indent=1))
    else:
        for t in result.texts:
            print(f"{t.region_id:>4} {t.bbox} [{t.source or '-'}] {t.text!r}")

    if any(t.text for t in result.texts):
        return 0
    if result.provider.get("ok") or result.model.get("calls", 0):
        return 1
    print(f"DEAD: no text lane could run: "
          f"{result.provider.get('reason') or result.model.get('error') or 'none'}",
          file=sys.stderr)
    return 2


def _self_test() -> int:
    """Self-test: verify the brick works offline.

    Tests:
    - Argument parsing
    - Image loading
    - JSON parsing
    - Finder initialization (without calling API)
    """
    try:
        # Test 1: no credential is REQUIRED, and the default is loopback.
        # This arm used to assert the opposite -- that a Finder without an API
        # key must fail. It was written against a hosted backend, which would
        # have shipped a brick that uploads a picture of your desktop to a
        # third party. The rule inverted with the design, and a self-test that
        # still pinned the old one would have kept it.
        finder = Finder()
        if not finder.endpoint.startswith("http"):
            print("FAIL: Finder has no endpoint")
            return 1
        if "localhost" not in finder.endpoint and "127.0.0.1" not in finder.endpoint:
            print("FAIL: default endpoint is not loopback: " + finder.endpoint)
            return 1
        # An api_key handed in by an older call site must be ignored, never
        # stored -- accepting a credential and then sending it somewhere is
        # worse than refusing it.
        if any("sk-probe" in str(v) for v in vars(Finder(api_key="sk-probe")).values()):
            print("FAIL: an api_key was retained")
            return 1

        # Test 2: Can initialize with key (but won't call API)
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test-key-for-testing")
            # This will fail if Anthropic is not installed, which is fine

        # Test 3: Test image loading with a dynamically created test image
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test-key")
            # Create a minimal valid PNG (1x1 pixel) dynamically
            png_bytes = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x055"
                b"\xcd\xfb\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(png_bytes)
                tmp.flush()
                tmp_path = tmp.name
            try:
                screenshot = finder.load_image(tmp_path)
                if screenshot.width <= 0 or screenshot.height <= 0:
                    print("FAIL: Image dimensions invalid")
                    return 1
            finally:
                Path(tmp_path).unlink()

        # Test 4: Test JSON parsing
        response = (
            '[{"x":10,"y":20,"width":100,"height":50,'
            '"confidence":0.95,"description":"button"}]'
        )
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test")
            elements = finder._parse_response(response)
            if len(elements) != 1:
                print(f"FAIL: Expected 1 element, got {len(elements)}")
                return 1
            elem = elements[0]
            if (
                elem.x != 10
                or elem.y != 20
                or elem.width != 100
                or elem.height != 50
                or elem.confidence != 0.95
            ):
                print(f"FAIL: Element values incorrect: {elem}")
                return 1

        # Test 5: Parse empty response
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test")
            elements = finder._parse_response("[]")
            if len(elements) != 0:
                print("FAIL: Empty response should return empty list")
                return 1

        # Test 6: Parse response with surrounding text
        response_with_text = (
            "I found these elements:\n"
            '[{"x":5,"y":10,"width":80,"height":30,"confidence":0.8,"description":"text"}]\n'
            "Done."
        )
        with contextlib.suppress(ImportError):
            finder = Finder(api_key="test")
            elements = finder._parse_response(response_with_text)
            if len(elements) != 1:
                print("FAIL: Should extract JSON from text")
                return 1

        # All tests passed
        return 0

    except Exception as exc:
        print(f"FAIL: Unexpected error in self-test: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
