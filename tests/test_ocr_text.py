"""Exact text: the OCR seam, vlm-ocr from inspection, and `--text-only`. Offline.

Both reading lanes are exercised without a real engine: a fake tesseract TSV built from
the fixture's known ink boxes, and a stubbed vision inspector that answers from the same
truth. The mixed screen proves exact strings for at least 8 of its 10 known strings.
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

import json  # noqa: E402
import socket  # noqa: E402
import urllib.request  # noqa: E402

import pytest  # noqa: E402
from _synthetic import draw_mixed_screen, draw_screen  # noqa: E402
from awscreen import ocr, regions  # noqa: E402
from awscreen._synthetic import KNOWN_TEXT  # noqa: E402
from awscreen.cli import main  # noqa: E402
from awscreen.inspect import CROP_PROMPT, InspectResult  # noqa: E402
from awscreen.merge import merge_elements  # noqa: E402
from awscreen.state import Element, build_screen_state  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise OSError("network disabled in this test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def _tsv_for(truth) -> str:
    """A tesseract TSV whose line boxes ARE the fixture's known text ink boxes."""
    rows = ["level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
            "width\theight\tconf\ttext\n"]
    for block, item in enumerate(truth.values(), start=1):
        x1, y1, x2, y2 = item["text_bbox"]
        rows.append(f"5\t1\t{block}\t1\t1\t1\t{x1}\t{y1}\t{max(1, x2 - x1)}\t"
                    f"{max(1, y2 - y1)}\t96\t{item['text']}\n")
    return "".join(rows)


@pytest.fixture
def fake_tesseract(monkeypatch):
    """Install a fake tesseract that returns one exact TSV line per fixture control."""
    def install(truth):
        def fake_run(argv, **kw):
            class Proc:
                returncode = 0
                stdout = _tsv_for(truth).encode("utf-8")
                stderr = b""

            return Proc()

        monkeypatch.setattr(regions.subprocess, "run", fake_run)
        monkeypatch.setattr(regions.TesseractTsvProposer, "_tsv_via_aithervision",
                            lambda self, img: None, raising=False)
        return ocr.TesseractTsvProvider(binary="tesseract")
    return install


def _text_in_crop(task, truth) -> str:
    """The fixture string whose ink box overlaps the crop most (a stubbed VLM read)."""
    cb = task.transform.crop_box
    best, best_area = "", 0.0
    for item in truth.values():
        tb = item["text_bbox"]
        ix = max(0, min(cb[2], tb[2]) - max(cb[0], tb[0]))
        iy = max(0, min(cb[3], tb[3]) - max(cb[1], tb[1]))
        if ix * iy > best_area:
            best, best_area = item["text"], ix * iy
    return best


def _stub_inspector(truth, calls):
    def insp(task):
        calls.append(task.question)
        text = _text_in_crop(task, truth)
        return InspectResult(region_id=task.region_id, status="ok", role="text",
                             label=text, text=text, confidence=0.9,
                             raw=json.dumps({"text": text}))
    insp.endpoint_host = "stub"
    insp.model = "stub"
    return insp


def _pairs(truth):
    return [(name, item["bbox"]) for name, item in truth.items()]


def test_ocr_provider_skips_gracefully_without_tesseract(monkeypatch):
    monkeypatch.setattr(regions.shutil, "which", lambda name: None)
    provider = ocr.TesseractTsvProvider()
    ok, why = provider.available()
    assert not ok and "tesseract" in why
    img, _ = draw_screen()
    assert provider.read(img) == []          # a skip, never a guess and never a raise


def test_tesseract_provider_reads_the_fake_tsv(fake_tesseract):
    img, truth = draw_mixed_screen()
    provider = fake_tesseract(truth)
    lines = provider.read(img)
    assert {line.text for line in lines} == set(KNOWN_TEXT)
    assert all(line.provider == "tesseract-tsv" for line in lines)


def test_assign_lines_gives_a_line_to_the_tightest_region():
    panel = ("panel", (0, 0, 500, 300))
    button = ("button", (10, 10, 110, 60))
    line = ocr.OcrLine(text="SAVE", bbox=(20, 20, 90, 50), conf=0.99,
                       provider="tesseract-tsv")
    assert ocr.assign_lines([panel, button], [line]) == {"button": ("SAVE", 0.99)}


def test_inspection_stores_exact_text_as_vlm_ocr(tmp_path):
    img, truth = draw_mixed_screen()
    path = tmp_path / "mixed.png"
    img.save(path)
    calls = []
    state = build_screen_state(str(path), proposers=("blobs",), inspect=True,
                               inspector=_stub_inspector(truth, calls), max_regions=10,
                               ocr=False)
    tagged = [e for e in state.elements if e.text_source == "vlm-ocr"]
    assert tagged, "no element carries vlm-ocr text"
    assert all("vlm-ocr" in e.sources and "vlm" in e.sources for e in tagged)
    # One inspection call per chosen crop, and every one used the crop prompt: the text
    # came from the call that was already happening, not from a second one.
    assert len(calls) == state.model["calls"]
    assert all(question == CROP_PROMPT for question in calls)
    found = [t for t in KNOWN_TEXT if any(t in e.text for e in state.elements)]
    assert len(found) >= 8, found


def test_read_region_text_with_fake_tesseract(fake_tesseract):
    img, truth = draw_mixed_screen()
    provider = fake_tesseract(truth)
    result = ocr.read_region_text(img, _pairs(truth), ocr_provider=provider, use_vlm=False)
    assert result.provider["ok"] and result.provider["lines"] == 10
    assert [t.text for t in result.texts] == [truth[n]["text"] for n, _ in _pairs(truth)]
    assert all(t.source == "tesseract-tsv" and t.status == "ok" for t in result.texts)


def test_read_region_text_uses_the_vlm_lane_when_ocr_is_absent():
    img, truth = draw_mixed_screen()
    calls = []
    result = ocr.read_region_text(img, _pairs(truth),
                                  ocr_provider=ocr.NullOcrProvider("no engine"),
                                  inspector=_stub_inspector(truth, calls), use_vlm=True)
    assert len(calls) == len(truth)
    assert all(t.source == "vlm-ocr" and t.status == "ok" for t in result.texts)
    assert [t.text for t in result.texts] == [truth[n]["text"] for n, _ in _pairs(truth)]


def test_mixed_screen_reads_at_least_eight_of_ten_known_strings(tmp_path, fake_tesseract):
    """The whole pipeline on a mixed screen: OCR fills every region, the model reads the
    five it is allowed to inspect, and their text comes from that same call."""
    img, truth = draw_mixed_screen()
    path = tmp_path / "mixed.png"
    img.save(path)
    provider = fake_tesseract(truth)
    calls = []
    state = build_screen_state(str(path), proposers=("blobs",), inspect=True,
                               inspector=_stub_inspector(truth, calls), max_regions=5,
                               ocr=provider)
    found = [t for t in KNOWN_TEXT if any(t in e.text for e in state.elements)]
    assert len(found) >= 8, (found, [(e.text, e.text_source) for e in state.elements])
    readers = {e.text_source for e in state.elements if e.text}
    assert "vlm-ocr" in readers and "tesseract-tsv" in readers, readers


def test_merge_keeps_the_vlm_ocr_reader_and_source():
    kept = Element(id="", region_id="r0", bbox=[0, 0, 10, 10], bbox_image=[0, 0, 10, 10],
                   label="Save", text="SAVE", text_source="vlm-ocr", role="button",
                   confidence=0.9, source="vlm", sources=["vlm", "vlm-ocr"])
    other = Element(id="", region_id="r0", bbox=[1, 1, 11, 11], bbox_image=[1, 1, 11, 11],
                    text="SAVE", text_source="tesseract-tsv", role="other",
                    confidence=0.4, source="cv", sources=["cv", "ocr"])
    merged = merge_elements([kept, other])
    assert len(merged) == 1
    assert merged[0].text_source == "vlm-ocr"
    assert {"vlm", "vlm-ocr", "cv", "ocr"} <= set(merged[0].sources)


def _write_regions(path, truth):
    doc = {"regions": [{"id": name, "bbox": item["bbox"]} for name, item in truth.items()]}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_cli_text_only_from_a_region_file(tmp_path, fake_tesseract, monkeypatch, capsys):
    img, truth = draw_mixed_screen()
    img_path = tmp_path / "mixed.png"
    img.save(img_path)
    provider = fake_tesseract(truth)
    monkeypatch.setattr(ocr, "make_ocr_provider",
                        lambda name="tesseract", **kw: provider)
    reg_path = _write_regions(tmp_path / "regions.json", truth)

    rc = main(["state", str(img_path), "--text-only", "--regions", str(reg_path),
               "--no-inspect", "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "awscreen.text_only/1"
    assert [t["text"] for t in doc["texts"]] == [truth[n]["text"] for n in truth]
    assert all(t["source"] == "tesseract-tsv" for t in doc["texts"])


def test_cli_text_only_accepts_a_screen_state_region_file(tmp_path, fake_tesseract,
                                                          monkeypatch, capsys):
    img, truth = draw_mixed_screen()
    img_path = tmp_path / "mixed.png"
    img.save(img_path)
    provider = fake_tesseract(truth)
    monkeypatch.setattr(ocr, "make_ocr_provider", lambda name="tesseract", **kw: provider)
    # A real state record from the same frame supplies the regions.
    assert main(["state", str(img_path), "--proposer", "blobs", "--no-inspect",
                 "--no-ocr", "--out", str(tmp_path / "state.json")]) == 0
    capsys.readouterr()
    rc = main(["state", str(img_path), "--text-only", "--no-inspect",
               "--regions", str(tmp_path / "state.json"), "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    found = [t for t in KNOWN_TEXT if any(t == r["text"] for r in doc["texts"])]
    assert len(found) >= 8, found


def test_cli_text_only_exit_2_when_no_lane_can_run(tmp_path, monkeypatch, capsys):
    img, truth = draw_mixed_screen()
    img_path = tmp_path / "mixed.png"
    img.save(img_path)
    monkeypatch.setattr(ocr, "make_ocr_provider",
                        lambda *a, **k: ocr.NullOcrProvider("no engine"))
    reg_path = _write_regions(tmp_path / "regions.json", truth)
    rc = main(["state", str(img_path), "--text-only", "--regions", str(reg_path),
               "--no-inspect"])
    assert rc == 2
    assert "no text lane" in capsys.readouterr().err


def test_cli_text_only_exit_1_when_a_lane_ran_and_found_nothing(tmp_path, monkeypatch,
                                                               capsys):
    img, truth = draw_mixed_screen()
    img_path = tmp_path / "mixed.png"
    img.save(img_path)
    reg_path = _write_regions(tmp_path / "regions.json", truth)

    class EmptyProvider:
        name = "empty"

        def available(self):
            return True, "ok"

        def read(self, img):
            return []

    monkeypatch.setattr(ocr, "make_ocr_provider", lambda *a, **k: EmptyProvider())
    rc = main(["state", str(img_path), "--text-only", "--regions", str(reg_path),
               "--no-inspect"])
    assert rc == 1
