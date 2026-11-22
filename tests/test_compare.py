"""Element-level screen diff: matching, far moves, and pixel deltas (offline)."""
import sys
from pathlib import Path

_PKGS = Path(__file__).resolve().parents[2]
for _n in ("awrun", "awgym", "awpredict", "awvision", "awscreen"):
    _p = _PKGS / _n
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from awscreen import compare  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def _state(*boxes):
    return {"elements": [
        {"id": f"e{i}", "bbox": list(b), "bbox_image": list(b),
         "label": label, "role": "button"} for i, (b, label) in enumerate(boxes)
    ]}


def test_iou_match_unchanged_and_moved():
    a = _state(([0, 0, 10, 10], "A"), ([50, 50, 60, 60], "B"))
    b = _state(([0, 0, 10, 10], "A"), ([70, 50, 80, 60], "B"))
    out = compare.compare_states(a, b)
    assert out["summary"]["unchanged"] == 1
    assert out["summary"]["moved"] == 1
    assert out["moved"][0]["dx"] == 20


def test_far_move_matched_by_label_not_removed_plus_added():
    a = _state(([0, 0, 10, 10], "SAVE"))
    b = _state(([500, 0, 510, 10], "SAVE"))
    out = compare.compare_states(a, b)
    assert out["summary"]["moved"] == 1
    assert out["summary"]["added"] == 0
    assert out["summary"]["removed"] == 0


def test_added_and_removed_buckets():
    a = _state(([0, 0, 10, 10], "ONE"))
    b = _state(([100, 100, 110, 110], "TWO"))
    out = compare.compare_states(a, b)
    assert out["summary"]["removed"] == 1
    assert out["summary"]["added"] == 1
    assert out["added"][0]["label"] == "TWO"


def test_pixel_delta_counts_changed_pixels():
    a = Image.new("RGB", (40, 40), (255, 255, 255))
    b = Image.new("RGB", (40, 40), (255, 255, 255))
    ImageDraw.Draw(b).rectangle([5, 5, 15, 15], fill=(0, 0, 0))
    delta = compare.pixel_delta(a, b, [0, 0, 20, 20])
    assert delta["pixels_changed"] > 0
    assert delta["pixels_total"] == 400


def test_compare_images_offline(tmp_path):
    def draw(path, left):
        img = Image.new("RGB", (300, 200), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([left, 20, left + 60, 60], fill=(37, 99, 235))
        img.save(path, "PNG")
        return str(path)

    a = draw(tmp_path / "a.png", 20)
    b = draw(tmp_path / "b.png", 40)
    out = compare.compare_images(a, b, proposer="blobs")
    assert out["summary"]["moved"] == 1
    assert out["moved"][0]["pixel_delta"]["pixels_changed"] > 0


def test_empty_states_diff_to_zero():
    out = compare.compare_states({"elements": []}, {"elements": []})
    assert out["summary"]["pairs"] == 0
    assert out["summary"]["added"] == 0


def test_state_without_elements_is_rejected():
    import pytest
    with pytest.raises(compare.CompareError):
        compare.compare_states({}, {"elements": []})
