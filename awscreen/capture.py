"""Capture ONE monitor, and keep the desktop point its image (0,0) really sits on.

``mss`` is the only local capture that knows there is more than one monitor. It is used
when importable:

* ``mss.monitors[0]`` is the union of every screen; ``1..N`` are the physical monitors,
  each with its own ``left``/``top`` on the desktop (a right-hand monitor commonly starts
  at ``left=1920``, a stacked one at ``top=-1080``).
* A frame is VERIFIED against the monitor it claims to be: its pixel size must equal that
  monitor's ``width`` x ``height``.

The alternatives are worse than refusing. Pillow's ``ImageGrab.grab()`` can only grab the
primary display; labelling that frame "monitor 2" puts every element 1920 px off and
nothing says so. So requesting monitor != 1 without ``mss`` raises
:class:`MonitorUnavailableError`, and a frame whose size does not match its claimed monitor
raises :class:`CaptureMonitorMismatchError`. ``awscreen state`` turns either into exit 2
("could not judge") -- never a silent wrong offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image

__all__ = [
    "CaptureError",
    "CaptureMonitorMismatchError",
    "Monitor",
    "MonitorOutOfRangeError",
    "MonitorUnavailableError",
    "capture",
    "list_monitors",
    "verify_frame",
]


class CaptureError(RuntimeError):
    """A monitor could not be captured or could not be verified. The message names why."""


class MonitorUnavailableError(CaptureError):
    """The monitor-aware capture lane (``mss``) cannot run on this host."""


class MonitorOutOfRangeError(CaptureError):
    """The requested monitor index is not present."""


class CaptureMonitorMismatchError(CaptureError):
    """The frame's pixels do not belong to the monitor they are claimed to be."""


@dataclass(frozen=True)
class Monitor:
    """One physical monitor: its 1-based mss index and its desktop rectangle."""

    index: int          # mss monitor index: 0 = every screen, 1..N = physical monitors
    left: int
    top: int
    width: int
    height: int

    @property
    def origin(self) -> Tuple[int, int]:
        """Desktop pixel where this monitor's image (0,0) sits."""
        return (self.left, self.top)

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def as_dict(self) -> dict:
        return {"index": self.index, "left": self.left, "top": self.top,
                "width": self.width, "height": self.height}

    def __str__(self) -> str:
        return (f"monitor {self.index}: {self.width}x{self.height} "
                f"at desktop ({self.left},{self.top})")


def _open_mss():
    """The single seam tests replace: a fresh ``mss`` context manager.

    Kept as a module function (not imported at module scope) so a host without ``mss``
    can still import awscreen, and so the two-monitor fixture can inject a fake.
    """
    import mss  # type: ignore[import-not-found]  # optional dependency

    return mss.mss()


def _monitors_from(sct: Any) -> List[Monitor]:
    """Read physical monitors (index 0 is the union and is skipped)."""
    try:
        raw = list(sct.monitors)[1:]
    except Exception as exc:  # noqa: BLE001 - a capture lib that cannot enumerate
        raise MonitorUnavailableError(f"mss could not enumerate monitors: {exc}") from exc
    out: List[Monitor] = []
    for i, m in enumerate(raw, 1):
        try:
            w, h = int(m["width"]), int(m["height"])
            left, top = int(m.get("left", 0)), int(m.get("top", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise MonitorUnavailableError(
                f"mss monitor {i} has no usable rectangle: {exc}") from exc
        if w <= 0 or h <= 0:
            raise MonitorUnavailableError(f"mss monitor {i} reports an empty size {w}x{h}")
        out.append(Monitor(i, left, top, w, h))
    if not out:
        raise MonitorUnavailableError("mss reported no physical monitors")
    return out


def list_monitors(open_mss: Optional[Any] = None) -> List[Monitor]:
    """Every physical monitor, in mss order (1-based). Raises MonitorUnavailableError."""
    maker = open_mss or _open_mss
    try:
        sct = maker()
    except ImportError as exc:
        raise MonitorUnavailableError("monitor capture needs mss: pip install mss") from exc
    except Exception as exc:  # noqa: BLE001 - a broken capture backend is unavailable
        raise MonitorUnavailableError(f"mss could not start: {exc}") from exc
    try:
        with sct:
            return _monitors_from(sct)
    except MonitorUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MonitorUnavailableError(f"mss could not enumerate monitors: {exc}") from exc


def get_monitor(index: int, open_mss: Optional[Any] = None) -> Monitor:
    """The physical monitor at ``index`` (1-based). Raises for a missing index."""
    if int(index) < 1:
        raise MonitorOutOfRangeError(
            f"monitor {index} does not exist: mss indexes physical monitors from 1")
    found = list_monitors(open_mss=open_mss)
    if int(index) > len(found):
        have = ", ".join(str(m) for m in found)
        raise MonitorOutOfRangeError(
            f"monitor {index} does not exist; this host reports {len(found)}: {have}")
    return found[int(index) - 1]


def _save(img: Image.Image, out: str) -> None:
    """PNG for a ``.png`` target (lossless, deterministic), JPEG otherwise."""
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".png":
        img.save(p, "PNG")
    else:
        img.convert("RGB").save(p, "JPEG", quality=80)


def _pillow_primary(out: str) -> Tuple[str, Monitor]:
    """Pillow's primary-display grab, origin (0,0) by definition."""
    try:
        from PIL import ImageGrab  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MonitorUnavailableError("screen capture needs Pillow: pip install pillow") from exc
    try:
        img = ImageGrab.grab().convert("RGB")
    except (OSError, ValueError) as exc:
        raise CaptureError(f"Pillow could not grab the primary screen: {exc}") from exc
    if img.width <= 0 or img.height <= 0:
        raise CaptureError("Pillow produced an empty frame")
    _save(img, out)
    return out, Monitor(1, 0, 0, img.width, img.height)


def capture(out: str, monitor: int = 1,
            open_mss: Optional[Any] = None) -> Tuple[str, Monitor]:
    """Capture ``monitor`` (1-based) into ``out``; return ``(path, Monitor)``.

    With ``mss`` the monitor's own pixels are grabbed and its real desktop origin is
    returned. Without ``mss`` only monitor 1 can be captured (Pillow, origin (0,0));
    asking for any other monitor raises :class:`MonitorUnavailableError` instead of silently
    handing back the primary display labelled as that monitor.
    """
    index = int(monitor)
    maker = open_mss or _open_mss
    try:
        sct = maker()
    except ImportError as exc:
        if index != 1:
            raise MonitorUnavailableError(
                f"monitor {index} needs mss (pip install mss); Pillow can only grab the "
                "primary monitor, and its frame would be mislabelled") from exc
        return _pillow_primary(out)
    except Exception as exc:  # noqa: BLE001 - one bad backend is the same as absent
        if index != 1:
            raise MonitorUnavailableError(
                f"monitor {index} needs mss and it could not start ({exc}); Pillow can "
                "only grab the primary monitor") from exc
        return _pillow_primary(out)

    try:
        with sct:
            mon_list = _monitors_from(sct)
            if index < 1 or index > len(mon_list):
                have = ", ".join(str(m) for m in mon_list)
                raise MonitorOutOfRangeError(
                    f"monitor {index} does not exist; this host reports "
                    f"{len(mon_list)}: {have}")
            mon = mon_list[index - 1]
            rect = sct.monitors[index]
            shot = sct.grab(rect)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    except (MonitorUnavailableError, MonitorOutOfRangeError):
        raise
    except Exception as exc:  # noqa: BLE001 - a partial grab is a failed grab
        raise CaptureError(f"mss could not capture monitor {index}: {exc}") from exc

    if img.size != mon.size:
        raise CaptureMonitorMismatchError(
            f"grabbed {img.size[0]}x{img.size[1]} for {mon}; mss returned a frame that "
            "does not match the monitor it was asked for")
    _save(img, out)
    return out, mon


def verify_frame(path: str, monitor: int,
                 open_mss: Optional[Any] = None) -> Monitor:
    """Verify that ``path``'s pixels are exactly ``monitor``'s; return that Monitor.

    Raises MonitorUnavailableError when mss cannot answer (so the claim cannot be checked)
    and CaptureMonitorMismatchError when the dimensions differ. A caller that asked for a
    monitor gets either the monitor or an error -- never an unchecked origin.
    """
    mon = get_monitor(monitor, open_mss=open_mss)
    try:
        with Image.open(path) as im:
            size = im.size
    except (OSError, ValueError) as exc:
        raise CaptureError(f"cannot read {path}: {exc}") from exc
    if size != mon.size:
        raise CaptureMonitorMismatchError(
            f"{path} is {size[0]}x{size[1]} but {mon}: the frame does not belong to the "
            "claimed monitor, so any click computed from it would be offset")
    return mon
