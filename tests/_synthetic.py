"""Synthetic screen fixture; the drawer ships in the package so --self-test can use it."""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from awscreen._synthetic import (  # noqa: E402
    BG,
    KNOWN_TEXT,
    RIGHT_MONITOR_ORIGIN,
    FakeMss,
    _FakeShot,
    draw_blank,
    draw_mixed_screen,
    draw_screen,
    draw_screen_right,
    draw_two_monitors,
)

__all__ = ["BG", "KNOWN_TEXT", "RIGHT_MONITOR_ORIGIN", "FakeMss", "_FakeShot",
           "draw_blank", "draw_mixed_screen", "draw_screen", "draw_screen_right",
           "draw_two_monitors"]
