"""Offline self-test for ``awscreen state``: one line per check, 0 only if all pass.

Never dials: OmniParser is not asked (blobs/grid run directly), the inspector is a fake,
and the awrun round-trip uses a temporary store with a null relay.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Tuple


def _close(a, b, tol=2) -> bool:
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def _fake_inspector(peak: dict, slow_ids=(), sleep_s=0.02):
    from .inspect import InspectResult
    lock = threading.Lock()

    def insp(task):
        with lock:
            peak["now"] = peak.get("now", 0) + 1
            peak["max"] = max(peak.get("max", 0), peak["now"])
        try:
            time.sleep(1.0 if task.region_id in slow_ids else sleep_s)
            return InspectResult(region_id=task.region_id, status="ok", role="button",
                                 label=f"fake {task.region_id}", text="", confidence=0.7,
                                 raw='{"role":"button"}')
        finally:
            with lock:
                peak["now"] -= 1
    insp.endpoint_host = "fake"
    insp.model = "fake"
    return insp


def run_checks() -> List[Tuple[str, bool, str]]:
    from PIL import Image

    from . import crops, dom, executors, merge, neurons, ocr, regions
    from . import uia as uia_mod
    from ._synthetic import draw_screen
    from .env import ScreenEnv
    from .inspect import CropTask, InspectResult
    from .state import Element, Region, build_screen_state

    out: List[Tuple[str, bool, str]] = []

    def check(name: str, fn: Callable[[], Tuple[bool, str]]):
        try:
            ok, why = fn()
        except Exception as exc:  # noqa: BLE001 - a crash is a failed check, named
            ok, why = False, f"{type(exc).__name__}: {exc}"
        out.append((name, bool(ok), why))

    img, truth = draw_screen()

    def c_blobs():
        regs = regions.BlobProposer().propose(img)
        boxes = [r.bbox for r in regs]
        miss = [k for k, t in truth.items() if not any(_close(b, t) for b in boxes)]
        return not miss, f"{len(regs)} regions; missing={miss}"
    check("proposers: blobs find every truth box within 2 px", c_blobs)

    def c_grid():
        g = regions.AdaptiveGridProposer()
        cells = g.cells(img.size)
        area = sum((c[2] - c[0]) * (c[3] - c[1]) for c in cells)
        regs = g.propose(img)
        return area == img.width * img.height and regs, \
            f"{len(cells)} cells tile {area}px; {len(regs)} inked regions"
    check("proposers: grid tiles exactly and keeps inked cells", c_grid)

    def c_crop():
        reg = Region(id="r0", bbox=tuple(truth["save"]), proposer="blobs")
        cr, tf = crops.make_crop(img, reg, margin=8, max_side=448, origin=(1920, 0))
        x, y = tf.to_image(*tf.from_screen(2020, 130))
        same = cr.getpixel((10, 10)) == img.getpixel(tf.to_image(10, 10))
        small, tf2 = crops.make_crop(img, (0, 0, 1280, 800), margin=0, max_side=448)
        bx = tf2.bbox_to_screen([0, 0, small.width, small.height])
        ok = (abs(x - 100) < 1e-6 and abs(y - 130) < 1e-6 and same
              and _close(bx, [0, 0, 1280, 800]))
        return ok, f"round-trip ({x},{y}); pixels equal={same}; downscale box={bx}"
    check("crop round-trip (margin, origin, downscale)", c_crop)

    def c_neurons():
        cr, _ = crops.make_crop(img, tuple(truth["save"]), margin=0)
        f = neurons.run_neurons(cr)
        blank = neurons.run_neurons(img.crop((600, 700, 700, 780)))
        ask_blank, why = neurons.worth_asking(blank)
        tm = neurons.template_match(img, cr)
        tm_ok = (not tm["available"]) or (tm["x"], tm["y"]) == (100, 120)
        return (tuple(f["dominant_rgb"]) == (37, 99, 235) and not ask_blank and tm_ok,
                f"dominant={f['dominant_rgb']} blank refused ({why}); template={tm}")
    check("neurons: dominant colour, blank gate, template match", c_neurons)

    def c_cap():
        peak: dict = {}
        tf = crops.make_crop(img, (0, 0, 10, 10))[1]
        tasks = [CropTask(region_id=f"r{i}", b64="", media_type="image/png", transform=tf)
                 for i in range(12)] + [CropTask(region_id="slow", b64="",
                                                 media_type="image/png", transform=tf)]
        res = executors.LocalExecutor().run(tasks, _fake_inspector(peak, slow_ids=("slow",)),
                                            max_concurrency=16, timeout_s=0.3)
        slow = [r for r in res if r.region_id == "slow"][0]
        return (peak.get("max", 99) <= 4 and slow.status == "timeout" and len(res) == 13,
                f"peak={peak.get('max')} (asked 16); slow -> {slow.status}")
    check("executor: cap 4 and timeout becomes an element", c_cap)

    def c_merge():
        a = Element(id="", region_id="r0", bbox=[0, 0, 100, 40], bbox_image=[0, 0, 100, 40],
                    label="", text="", role="other", confidence=0.4, source="cv",
                    sources=["cv"])
        b = Element(id="", region_id="r0", bbox=[2, 1, 101, 41], bbox_image=[2, 1, 101, 41],
                    label="Save", text="SAVE", text_source="vlm-ocr", role="button",
                    confidence=0.7, source="vlm", sources=["vlm", "vlm-ocr"])
        m = merge.merge_elements([a, b])
        return (len(m) == 1 and m[0].bbox == [2, 1, 101, 41] and m[0].role == "button"
                and m[0].text_source == "vlm-ocr" and "vlm-ocr" in m[0].sources
                and set(m[0].sources) == {"cv", "vlm", "vlm-ocr"}), \
            f"{len(m)} element(s), {m[0].sources}, text={m[0].text_source}"
    check("merge: IoU dedupe follows source priority and keeps the text reader", c_merge)

    def c_ocr():
        lines = [ocr.OcrLine(text="SAVE", bbox=tuple(truth["save"]), conf=0.99,
                             provider="tesseract-tsv"),
                 ocr.OcrLine(text="Hello world", bbox=tuple(truth["hello"]), conf=0.9,
                             provider="tesseract-tsv")]
        assigned = ocr.assign_lines([("save", truth["save"]), ("hello", truth["hello"])],
                                    lines)
        element = Element(id="", region_id="r0", bbox=list(truth["save"]),
                          bbox_image=list(truth["save"]), source="cv", sources=["cv"])
        wrote = ocr.attach_text([element], lines)
        skip_ok, skip_why = ocr.TesseractTsvProvider(binary=None).available()
        skipped = ocr.TesseractTsvProvider(binary=None).read(img)
        return (assigned.get("save", ("", 0))[0] == "SAVE"
                and assigned.get("hello", ("", 0))[0] == "Hello world"
                and wrote == 1 and element.text == "SAVE"
                and element.text_source == "tesseract-tsv" and "ocr" in element.sources
                and not skip_ok and skipped == []), \
            f"assigned={assigned}; element source={element.text_source}; " \
            f"absent engine skips ({skip_why})"
    check("ocr: line-to-region assignment and a graceful skip", c_ocr)

    def c_vlm_text():
        def insp(task):
            return InspectResult(region_id=task.region_id, status="ok", role="button",
                                 label="Save", text="SAVE", confidence=0.9,
                                 raw='{"role":"button","label":"Save","text":"SAVE"}')
        insp.endpoint_host = "fake"
        insp.model = "fake"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "syn.png"
            img.save(p)
            s = build_screen_state(str(p), proposers=("blobs",), inspect=True,
                                   inspector=insp, ocr=False)
        tagged = [e for e in s.elements
                  if e.text == "SAVE" and e.text_source == "vlm-ocr"
                  and "vlm-ocr" in e.sources]
        return bool(tagged) and s.model.get("calls", 0) > 0, \
            f"{len(tagged)} element(s) carry vlm-ocr text from {s.model.get('calls')} call(s)"
    check("vlm-ocr: the inspection call also returns the exact text", c_vlm_text)

    def c_dom():
        img2, _t2 = draw_screen(scale=2)
        st = build_screen_state.__wrapped__(img2) if hasattr(build_screen_state, "__wrapped__") \
            else None
        del st
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s2.png"
            img2.save(p)
            snap = {"viewport": {"dpr": 2, "vw": 1280, "vh": 800, "sx": 0, "sy": 0},
                    "elements": [{"ref": "#save", "tag": "button", "label": "Save",
                                  "rect": [100, 120, 160, 48]},
                                 {"ref": "#ghost", "tag": "a", "label": "ghost",
                                  "rect": [1000, 20, 80, 20]}]}
            s = build_screen_state(str(p), proposers=("blobs",), inspect=False, dom=snap)
        hit = [e for e in s.elements if e.dom and e.dom.get("ref") == "#save"]
        ghost = [e for e in s.elements if e.source == "dom" and e.dom.get("ref") == "#ghost"]
        ok2 = bool(hit) and "cv" in hit[0].sources and hit[0].dom["iou"] > 0.9 and ghost
        fit = dom.DomSnapshot.from_dict({"viewport": {"dpr": 1, "vw": 1920, "sx": 0,
                                                      "sy": 300},
                                         "image_scale": "fit", "elements": []})
        b = fit.css_to_image([150, 480, 240, 72], 1280, page=True)
        return ok2 and _close(b, [100, 120, 260, 168]), \
            f"dpr2 iou={hit[0].dom['iou'] if hit else None}; fit+scroll box={b}"
    check("dom-join: dpr2, fit and scroll", c_dom)

    def c_uia():
        x1, y1, x2, y2 = truth["save"]
        snap = uia_mod.UiaSnapshot.from_dict({"source": "fixture", "nodes": [
            {"name": "Save", "control_type": "MenuItem", "rect": [x1, y1, x2 - x1, y2 - y1],
             "is_enabled": True},
            {"name": "Format", "control_type": "MenuItem", "rect": [400, 100, 80, 24],
             "is_enabled": True},
            {"name": "Offscreen", "control_type": "Button", "rect": [5000, 10, 40, 20],
             "is_enabled": True}]})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "syn.png"
            img.save(p)
            s = build_screen_state(str(p), proposers=("blobs",), inspect=False)
        s = uia_mod.join_uia(s, snap)
        hit = [e for e in s.elements if e.uia and e.uia["name"] == "Save"]
        fmt = [e for e in s.elements if e.source == "uia" and e.uia["name"] == "Format"]
        linux_ok, linux_why = uia_mod.bridge_available(platform="linux")
        ok = (bool(hit) and hit[0].role == "menuitem" and "uia" in hit[0].sources
              and bool(fmt) and fmt[0].role == "menuitem" and fmt[0].interactable
              and not linux_ok and "Windows" in linux_why and s.uia.get("skipped") == 1)
        return ok, (f"matched={s.uia.get('matched')} added={s.uia.get('added')} "
                    f"skipped={s.uia.get('skipped')}; Save role="
                    f"{hit[0].role if hit else None}; off-Windows skip={not linux_ok}")
    check("uia-join: menu item keeps its control type; off Windows skips", c_uia)

    def c_env():
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "syn.png"
            img.save(p)
            s = build_screen_state(str(p), proposers=("blobs",), inspect=True,
                                   inspector=_fake_inspector({}))
        for e in s.elements:
            if e.bbox_image and _close(e.bbox_image, truth["save"]):
                e.label = "save"
        env = ScreenEnv(state=s, goal_label="done")
        try:
            from awpredict.contracts import EnvironmentAdapter, conforms
            missing = conforms(ScreenEnv, EnvironmentAdapter)
            conf = "conforms" if not missing else f"missing {missing}"
        except ImportError:
            missing, conf = [], "awpredict: unavailable (structural members checked)"
            missing = [m for m in ("domain", "observe", "actions", "step")
                       if not hasattr(ScreenEnv, m)]
        acts = env.actions()
        click = [a for a in acts if a["kind"] == "click" and a.get("label") == "save"]
        dry = env.step(click[0])
        goal = json.loads(json.dumps(s.to_dict()))
        goal["elements"].append(dict(goal["elements"][0], label="done", id="eX"))
        env.record(env.get_state_hash(), click[0], goal, 1.0)
        sim = env.clone()
        _, r_seen, done_seen, _ = sim.step(click[0])
        sim2 = env.clone()
        _, r_unseen, _, info_unseen = sim2.step(acts[-1])
        ok = (not missing and click and click[0]["xy"] == [180, 144]
              and dry[3].get("refused") == "no executor" and r_seen == 1.0 and done_seen
              and r_unseen == 0.0 and info_unseen["seen"] is False)
        return ok, f"{conf}; {len(acts)} actions; clone replay r={r_seen}, unseen r={r_unseen}"
    check("env: conforms, dry-run refusal, clone replay", c_env)

    def c_act():
        from .actuator import DryRunActuator, LiveActuator, resolve_plan, run_plan
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "syn.png"
            img.save(p)
            st = build_screen_state(str(p), proposers=("blobs",), inspect=False)
        for e in st.elements:
            if e.bbox_image and _close(e.bbox_image, truth["save"]):
                e.label = "save"
            if e.bbox_image and _close(e.bbox_image, truth["input"]):
                e.label = "name"
        plan = {"actions": [{"kind": "click", "label": "save"},
                            {"kind": "type", "label": "name", "text": "hi"},
                            {"kind": "scroll", "direction": "down"}]}
        resolved = resolve_plan(st, plan)

        class Rec:
            def __init__(self):
                self.calls = []

            def click(self, x, y):
                self.calls.append(("click", x, y))

            def type_text(self, text):
                self.calls.append(("type", text))

            def scroll(self, dy):
                self.calls.append(("scroll", dy))

        rec = Rec()
        dry = run_plan(DryRunActuator(), resolved)
        live = run_plan(LiveActuator(rec), resolved)
        same = [(c["kind"], c.get("x"), c.get("y")) for c in dry] == \
               [(c["kind"], c.get("x"), c.get("y")) for c in live]
        ok = (resolved[0]["x"] == 180 and resolved[0]["y"] == 144
              and rec.calls == [("click", 180, 144), ("type", "hi"), ("scroll", 3)]
              and same)
        return ok, f"{len(resolved)} actions; live order={rec.calls}"
    check("act: plan resolves to absolute points, dry run == live order", c_act)

    def c_awrun():
        try:
            import awrun.store  # noqa: F401
        except ImportError:
            return True, "awrun: unavailable"
        with tempfile.TemporaryDirectory() as td:
            cr, tf = crops.make_crop(img, tuple(truth["save"]))
            tasks = [CropTask(region_id="r0", b64=crops.crop_b64(cr), media_type="image/png",
                              transform=tf)]
            ex = executors.AwrunExecutor(root=td)
            res = ex.run(tasks, _fake_inspector({}), max_concurrency=4, timeout_s=5)
            try:
                executors.AwrunExecutor(target="swarm")
                refused = False
            except executors.ExecutorRefused:
                refused = True
        return (len(res) == 1 and res[0].status == "ok" and refused,
                f"item {ex.last_item_id}: {res[0].status if res else 'none'}; swarm refused")
    check("awrun: kind=screen round-trip, swarm refused", c_awrun)

    def c_vlm_model():
        from .inspect import DEFAULT_VLM_MODEL, resolve_vlm_model
        flag = resolve_vlm_model("flag-model", {"AWSCREEN_VLM_MODEL": "env-model"})
        env = resolve_vlm_model(None, {"AWSCREEN_VLM_MODEL": "env-model"})
        unset = resolve_vlm_model(None, {})
        return (DEFAULT_VLM_MODEL == "bonsai2-27b-5090" and flag == "flag-model"
                and env == "env-model" and unset is None), \
            f"default={DEFAULT_VLM_MODEL}; flag > env > awvision"
    check("vlm: crop model default and override precedence", c_vlm_model)

    def c_budget():
        from .budget import Budget, resolve_budget_ms
        now = [0.0]
        b = Budget(5000, lambda: now[0])
        cap = b.cap_for(2000, 1)          # two whole crops in five seconds
        now[0] = 3.0
        mid_fits = b.fits(2000) and not b.fits(2001)
        now[0] = 5.0
        flag = resolve_budget_ms(4000, {"AWSCREEN_BUDGET_MS": "7000"})
        env = resolve_budget_ms(None, {"AWSCREEN_BUDGET_MS": "7000"})
        return (cap == 2 and mid_fits and b.expired() and flag == 4000 and env == 7000), \
            f"cap={cap}; 2s fits@3s={mid_fits}; expired@5s; flag={flag}, env={env}"
    check("budget: cap shrinks to whole crops, clock stops calls", c_budget)
    def c_monitors():
        from . import capture
        from ._synthetic import draw_two_monitors

        frames, factory = draw_two_monitors()
        saved = capture._open_mss
        capture._open_mss = factory
        try:
            with tempfile.TemporaryDirectory() as td:
                out = str(Path(td) / "right.png")
                _p, mon = capture.capture(out, monitor=2)
                st = build_screen_state(out, monitor=2, proposers=("blobs",),
                                        inspect=False)
                primary = Path(td) / "primary.png"
                frames[0][1].save(primary)
                mismatch = False
                try:
                    capture.verify_frame(str(primary), 2)
                except capture.CaptureMonitorMismatchError:
                    mismatch = True
        finally:
            capture._open_mss = saved
        place = frames[1][2]["place"]
        hit = [e for e in st.elements if _close(e.bbox_image, place, 3)]
        ok = (mon.origin == (1280, 0) and st.frame.origin == (1280, 0) and hit
              and _close(hit[0].bbox, [place[0] + 1280, place[1],
                                      place[2] + 1280, place[3]], 3) and mismatch)
        return ok, (f"monitor2 origin={mon.origin}; right-monitor element at "
                    f"{hit[0].bbox if hit else None}; wrong frame refused={mismatch}")
    check("capture: per-monitor origin, verified bytes", c_monitors)

    _ = Image  # keep the import honest: Pillow is required for every check above
    return out


def main() -> int:
    results = run_checks()
    for name, ok, why in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {why}")
    bad = [r for r in results if not r[1]]
    print(f"state self-test: {len(results) - len(bad)}/{len(results)} passed")
    return 0 if results and not bad else 1
