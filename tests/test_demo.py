"""`awscreen demo`: offline timeline, annotation, honest DEAD surfaces, CLI routing.

Drawn screens, a fake surface list, no model and no windows. The real pipeline still
runs (``inspect=False``) so the test exercises propose -> crop -> neurons for real.
"""
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

import pytest  # noqa: E402
from awscreen import demo  # noqa: E402
from awscreen.demo import (  # noqa: E402
    BOX_COLOUR,
    MODEL_BOX,
    Capture,
    DemoDeps,
    SurfaceSpec,
    annotate,
    run_demo,
    transcript,
)
from PIL import Image, ImageDraw  # noqa: E402


def _drawn(path: Path, label: str = "Sign in") -> Path:
    img = Image.new("RGB", (900, 500), (18, 22, 32))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 300, 100], fill=(40, 90, 200))
    d.text((60, 60), label, fill=(255, 255, 255))
    d.rectangle([340, 40, 700, 100], outline=(200, 200, 200), width=3)
    d.text((360, 60), "Search the platform", fill=(120, 220, 255))
    d.rectangle([40, 150, 260, 220], fill=(40, 120, 70))
    d.text((60, 175), "Documentation", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network escapes a demo test."""
    def boom(*_a, **_k):
        raise OSError("network disabled in this test")
    monkeypatch.setattr("urllib.request.urlopen", boom)


def _deps(tmp: Path, capture=None, video=None) -> DemoDeps:
    return DemoDeps(
        capture=capture or (lambda spec, out, env: Capture(image=str(_drawn(out)),
                                                            origin=(0, 0))),
        close=lambda _w: None,
        state=demo.default_state,
        context=lambda _out: {"tools": {"get_system_status": {"health": "healthy"}},
                              "processes": {"count": 1, "top_by_memory": []},
                              "ports": {"count": 2}, "har": {"requests": 1}},
        video=video or (lambda *a, **k: {"ok": True, "path": str(tmp / "v.mp4")}),
        recorder_start=lambda _p, _f: {"stub": True},
        recorder_stop=lambda _r: {"status": "skipped"},
    )


def test_self_test_passes():
    assert demo.self_test() == 0


def test_run_demo_writes_timeline_and_transcript(tmp_path):
    specs = [SurfaceSpec("desktop", "DESKTOP", kind="screen"),
             SurfaceSpec("terminal", "TERMINAL", opener="terminal"),
             SurfaceSpec("apis", "APIS", opener="browser_context")]
    params = {"out_dir": str(tmp_path), "proposer": "blobs", "inspect": False,
              "max_regions": 4, "concurrency": 2, "render_video": False}
    record = run_demo(specs, deps=_deps(tmp_path), params=params)
    assert record["schema"] == demo.SCHEMA
    assert record["totals"]["surfaces"] == 3
    assert record["totals"]["ok"] == 3
    assert record["totals"]["elements"] > 0
    for surface in record["surfaces"]:
        assert surface["elements"] > 0, surface["name"]
        assert Path(surface["annotated"]).is_file(), surface["name"]
        assert Path(surface["image"]).is_file()
        assert "total_ms" in surface["timing"]
    # APIs surface carries the context section.
    apis = next(s for s in record["surfaces"] if s["name"] == "apis")
    assert apis["state"]["context"]["har"]["requests"] == 1
    assert (tmp_path / "context.json").is_file()
    assert (tmp_path / "demo.json").is_file()
    text = (tmp_path / "demo-transcript.md").read_text(encoding="utf-8")
    for name in ("desktop", "terminal", "apis"):
        assert f"**{name}**" in text
    assert "elements" in text and "ms" in text


def test_annotate_uses_two_colours_and_numbers(tmp_path):
    img = _drawn(tmp_path / "src.png")
    state = {
        "frame": {"origin": (0, 0), "width": 900, "height": 500},
        "proposer": {"ran": "blobs"},
        "model": {"model": "tiny", "calls": 1, "ok": 1},
        "elements": [
            {"id": "e0", "bbox": [40, 40, 300, 100], "bbox_image": [40, 40, 300, 100],
             "role": "button", "label": "Sign in", "confidence": 0.82,
             "source": "vlm", "sources": ["vlm"], "status": "ok"},
            {"id": "e1", "bbox": [340, 40, 700, 100], "bbox_image": [340, 40, 700, 100],
             "role": "input", "label": "Search", "confidence": 0.4,
             "source": "cv", "sources": ["cv"], "status": "ok"},
        ],
    }
    out = annotate(str(img), state, tmp_path / "a.png", "TEST SURFACE")
    img_out = Image.open(out).convert("RGB")
    colours = {colour for _count, colour in img_out.getcolors(maxcolors=1 << 24)}
    assert MODEL_BOX in colours
    assert BOX_COLOUR in colours
    assert Path(out).stat().st_size > 1000


def test_dead_surface_is_recorded_not_invented(tmp_path):
    def dead(_spec, _out, _env):
        return Capture(status="dead", detail="no window appeared")

    specs = [SurfaceSpec("browser", "BROWSER", opener="browser")]
    params = {"out_dir": str(tmp_path), "proposer": "blobs", "inspect": False,
              "render_video": False}
    record = run_demo(specs, deps=_deps(tmp_path, capture=dead), params=params)
    surface = record["surfaces"][0]
    assert surface["status"] == "dead"
    assert "no window appeared" in surface["error"]
    assert surface["elements"] == 0
    assert surface["annotated"] == ""
    assert "DEAD" in transcript(record)


def test_model_failure_is_zero_labelled_not_faked(tmp_path):
    def no_model(image, params):
        state = demo.default_state(image, {**params, "inspect": False})
        state["model"] = {"calls": 0, "ok": 0, "error": "VisionUnavailable: offline"}
        return state

    deps = _deps(tmp_path)
    deps.state = no_model
    specs = [SurfaceSpec("desktop", "DESKTOP", kind="screen")]
    params = {"out_dir": str(tmp_path), "proposer": "blobs", "inspect": False,
              "render_video": False}
    record = run_demo(specs, deps=deps, params=params)
    surface = record["surfaces"][0]
    assert surface["elements"] > 0
    assert surface["model_elements"] == 0
    assert surface["status"] == "ok"
    assert "0/0" in transcript(record)


def test_fit_frame_letterboxes_to_the_deliverable_canvas(tmp_path):
    src = _drawn(tmp_path / "src.png")
    out = demo._fit_frame(str(src), tmp_path / "fit.png", (1920, 1080))
    assert Image.open(out).size == (1920, 1080)


def test_cli_routes_the_demo_verb(monkeypatch):
    from awscreen import cli

    seen = {}

    def fake_demo(argv):
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(demo, "main", fake_demo)
    assert cli.main(["demo", "--self-test"]) == 7
    assert seen["argv"] == ["--self-test"]


def test_video_seam_gets_slides_and_capture(tmp_path):
    captured = {}

    def fake_video(slides, capture_video, out_path, params):
        captured["slides"] = list(slides)
        captured["capture_video"] = capture_video
        captured["out_path"] = str(out_path)
        return {"ok": True, "path": str(out_path), "media_id": 1}

    specs = [SurfaceSpec("desktop", "DESKTOP", kind="screen")]
    params = {"out_dir": str(tmp_path), "proposer": "blobs", "inspect": False,
              "render_video": True, "record": True}
    record = run_demo(specs, deps=_deps(tmp_path, video=fake_video), params=params)
    assert captured["slides"], "no annotated slide reached the render seam"
    assert captured["out_path"].endswith("vision-swarm-demo.mp4")
    assert record["video"]["status"] == "ok"
