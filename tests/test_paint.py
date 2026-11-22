"""Deterministic paint: a JSON manifest always renders the same pixels (offline)."""
import io
import sys
from pathlib import Path

_PKGS = Path(__file__).resolve().parents[2]
for _n in ("awrun", "awgym", "awpredict", "awvision", "awscreen"):
    _p = _PKGS / _n
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402
from awscreen import paint  # noqa: E402
from PIL import Image  # noqa: E402


def _manifest():
    return {
        "width": 40, "height": 20, "background": "#ffffff",
        "ops": [
            {"op": "rect", "bbox": [10, 5, 20, 10], "fill": "#ff0000"},
            {"op": "pixels", "pixels": [[0, 0, "#00ff00"], [1, 0, [0, 0, 255]]]},
        ],
    }


def test_same_manifest_same_bytes():
    a = paint.paint_b64(_manifest())
    b = paint.paint_b64(_manifest())
    assert a == b


def test_bbox_is_exclusive():
    img = paint.render_paint(_manifest())
    assert img.getpixel((10, 5)) == (255, 0, 0, 255)   # inside
    assert img.getpixel((19, 9)) == (255, 0, 0, 255)   # last inside pixel
    assert img.getpixel((20, 5)) == (255, 255, 255, 255)  # x2 excluded
    assert img.getpixel((10, 10)) == (255, 255, 255, 255)  # y2 excluded


def test_pixels_op_sets_exact_pixels():
    img = paint.render_paint(_manifest())
    assert img.getpixel((0, 0)) == (0, 255, 0, 255)
    assert img.getpixel((1, 0)) == (0, 0, 255, 255)


def test_png_roundtrips():
    data = paint.paint_png_bytes(_manifest())
    img = Image.open(io.BytesIO(data))
    assert img.size == (40, 20)


def test_describe_and_hash():
    info = paint.describe(_manifest())
    assert info["ops"] == 2
    assert info["manifest_sha256"] == paint.manifest_sha256(_manifest())
    assert (info["width"], info["height"]) == (40, 20)


def test_invalid_manifests_raise():
    with pytest.raises(paint.PaintError):
        paint.render_paint({"width": 0, "height": 10})
    with pytest.raises(paint.PaintError):
        paint.render_paint({"width": 10, "height": 10, "ops": [{"op": "nope"}]})
    with pytest.raises(paint.PaintError):
        paint.render_paint({"width": 10, "height": 10,
                            "ops": [{"op": "rect", "bbox": [5, 5, 1, 1]}]})


def test_hex_alpha_and_named_colours():
    manifest = {"width": 4, "height": 4, "ops": [
        {"op": "pixels", "pixels": [[0, 0, "#ff000080"], [1, 0, "red"]]}]}
    img = paint.render_paint(manifest)
    assert img.getpixel((0, 0)) == (255, 0, 0, 128)
    assert img.getpixel((1, 0)) == (255, 0, 0, 255)
