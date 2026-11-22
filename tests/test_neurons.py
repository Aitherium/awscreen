"""Pixel neurons: dominant colour, the blank gate, template match."""
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
from _synthetic import draw_screen  # noqa: E402
from awscreen import neurons  # noqa: E402
from awscreen.crops import make_crop  # noqa: E402


def test_dominant_colour_of_save_is_the_fill():
    img, t = draw_screen()
    crop, _ = make_crop(img, tuple(t["save"]), margin=0)
    assert tuple(neurons.run_neurons(crop)["dominant_rgb"]) == (37, 99, 235)


def test_blank_crop_is_refused():
    img, _ = draw_screen()
    f = neurons.run_neurons(img.crop((600, 700, 760, 780)))
    ok, why = neurons.worth_asking(f)
    assert not ok and "blank" in why
    assert neurons.guess_role(f)[0] == "none"


def test_inked_crop_is_asked():
    img, t = draw_screen()
    crop, _ = make_crop(img, tuple(t["save"]))
    ok, _why = neurons.worth_asking(neurons.run_neurons(crop))
    assert ok


def test_template_match_locates_save():
    img, t = draw_screen()
    crop, _ = make_crop(img, tuple(t["save"]), margin=0)
    res = neurons.template_match(img, crop)
    try:
        import numpy  # noqa: F401
    except ImportError:
        assert res == {"available": False, "reason": "unavailable: numpy not importable"}
        return
    assert res["available"] and (res["x"], res["y"]) == (100, 120)


def test_template_match_reports_unavailable_without_numpy(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("no numpy")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    img, t = draw_screen()
    res = neurons.template_match(img, img.crop((0, 0, 10, 10)))
    assert res["available"] is False and "unavailable" in res["reason"]


@pytest.mark.parametrize("key", ["edge_density", "ink_ratio", "lum_mean", "lum_std", "aspect"])
def test_feature_keys_present(key):
    img, t = draw_screen()
    assert key in neurons.run_neurons(img.crop((0, 0, 50, 50)))
