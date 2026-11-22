"""Crop + CropTransform: exact round-trips, clamping, downscale, pixel equality."""
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

from _synthetic import draw_screen  # noqa: E402
from awscreen.crops import crop_b64, make_crop  # noqa: E402
from awscreen.state import Region  # noqa: E402


def test_round_trip_with_margin_and_origin():
    img, t = draw_screen()
    crop, tf = make_crop(img, Region("r0", tuple(t["save"]), "blobs"), margin=8,
                         origin=(1920, 40))
    assert tf.crop_box == (92, 112, 268, 176) and tf.scale == 1.0
    assert crop.size == (176, 64)
    assert tf.to_image(8, 8) == (100, 120)
    assert tf.to_screen(8, 8) == (2020, 160)
    assert tf.from_screen(2020, 160) == (8, 8)
    assert tf.bbox_to_screen([8, 8, 168, 56]) == [2020, 160, 2180, 208]


def test_clamped_at_the_edges():
    img, _ = draw_screen()
    crop, tf = make_crop(img, (0, 0, 20, 20), margin=8)
    assert tf.crop_box == (0, 0, 28, 28)
    crop, tf = make_crop(img, (1270, 790, 1280, 800), margin=8)
    assert tf.crop_box == (1262, 782, 1280, 800)


def test_downscale_records_the_scale():
    img, _ = draw_screen()
    crop, tf = make_crop(img, (0, 0, 896, 100), margin=0, max_side=448)
    assert tf.scale == 0.5 and crop.size == (448, 50)
    assert tf.to_image(448, 50) == (896, 100)
    x, y = tf.from_screen(*tf.to_screen(123.0, 45.0))
    assert abs(x - 123.0) < 1e-9 and abs(y - 45.0) < 1e-9


def test_crop_pixels_equal_source_pixels():
    img, t = draw_screen()
    crop, tf = make_crop(img, tuple(t["cancel"]), margin=8)
    for cx, cy in [(0, 0), (8, 8), (100, 30), (crop.width - 1, crop.height - 1)]:
        ix, iy = tf.to_image(cx, cy)
        assert crop.getpixel((cx, cy)) == img.getpixel((int(ix), int(iy)))


def test_crop_b64_is_png_in_memory():
    import base64
    img, t = draw_screen()
    crop, _ = make_crop(img, tuple(t["save"]))
    assert base64.b64decode(crop_b64(crop))[:8] == b"\x89PNG\r\n\x1a\n"


def test_flat_crop_short_axis_maps_back_exactly():
    """The downscale resizes each axis independently (round(w*s) x round(h*s)); a single
    nominal ``scale`` divided into both axes drifts several px on the short side of a long
    flat crop (3840-wide, 22-px region -> 3.7 px; 7680-wide, 44-px -> 8.6 px). Recording
    the actual ``crop_size`` must make the mapping land on the true pixel."""
    from PIL import Image
    for w, h in ((3840, 22), (5120, 22), (7680, 44)):
        img = Image.new("RGB", (w, 600), (0, 0, 0))
        crop, tf = make_crop(img, (0, 100, w, 100 + h), margin=8, max_side=448)
        assert tf.crop_size == crop.size
        sx, sy = tf.scale_xy()
        assert sx != sy                                    # the axes really do differ
        bx, by = tf.to_image(crop.width, crop.height)
        assert abs(bx - tf.crop_box[2]) < 1e-6 and abs(by - tf.crop_box[3]) < 1e-6
        # and from_screen is the exact inverse on both axes
        ix, iy = tf.from_screen(*tf.to_screen(crop.width, crop.height))
        assert abs(ix - crop.width) < 1e-6 and abs(iy - crop.height) < 1e-6
