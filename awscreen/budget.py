"""A latency budget: spend a deadline in order, and name what did not fit.

``awscreen state`` spends a frame's time in a fixed order -- capture, propose, crop,
neurons (all cheap and local), then model calls. A budget does not change that order; it
decides WHEN TO STOP: neurons always run, then a vision call is made only while the
remaining time still covers a whole crop. A crop that gets no call is never dropped and
never invented -- its neuron-only element is returned with ``status = 'budget'`` so a
reader can see exactly what was not looked at.

The per-crop cost is an ESTIMATE (:data:`DEFAULT_CALL_ESTIMATE_MS`), because latency is
not known before the first call; the wave runner re-checks the clock after every wave, so
even a wrong estimate cannot push the state past the deadline.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Mapping, Optional

#: The CLI/env name that turns a budget on without a flag.
BUDGET_ENV = "AWSCREEN_BUDGET_MS"

#: What one crop costs the 27B lane: measured 1.3-2.6 s on the three UI crops (see
#: ``inspect.py``), rounded up so the planner is conservative rather than late.
DEFAULT_CALL_ESTIMATE_MS = 2600.0

#: Element status for an element the clock left no whole crop for.
BUDGET_STATUS = "budget"


def resolve_budget_ms(explicit: Optional[float] = None,
                      environ: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """The budget named by the caller: the flag, then ``AWSCREEN_BUDGET_MS``.

    ``None`` means unbounded (today's behaviour). ``0`` is honoured and means "no model
    calls fit". A malformed env value is ignored, not guessed.
    """
    def _as_ms(raw):
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None

    if explicit is not None:
        return _as_ms(explicit)
    env = os.environ if environ is None else environ
    return _as_ms(env.get(BUDGET_ENV))


class Budget:
    """A wall-clock deadline measured from construction.

    ``clock`` is a zero-argument seconds callable; tests inject a fake one so the
    deadline is deterministic (the real one is ``time.monotonic``).
    """

    def __init__(self, total_ms: float, clock: Optional[Callable[[], float]] = None) -> None:
        self.total_ms = max(0.0, float(total_ms))
        self._clock = clock or time.monotonic
        self._started = self._clock()

    def elapsed_ms(self) -> float:
        return max(0.0, (self._clock() - self._started) * 1000.0)

    def remaining_ms(self) -> float:
        return max(0.0, self.total_ms - self.elapsed_ms())

    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def fits(self, cost_ms: float) -> bool:
        """True when a whole operation costing ``cost_ms`` still fits."""
        return self.remaining_ms() >= max(0.0, float(cost_ms))

    def cap_for(self, call_ms: float, concurrency: int) -> int:
        """How many crops the remaining time covers: whole waves of ``concurrency``.

        A wave costs about ``call_ms`` however many crops run in it, so the cap is
        ``floor(remaining / call_ms) * concurrency``.
        """
        conc = max(1, int(concurrency))
        per = max(1.0, float(call_ms))
        return int(self.remaining_ms() // per) * conc

    def as_dict(self) -> dict:
        return {"budget_ms": round(self.total_ms, 1),
                "elapsed_ms": round(self.elapsed_ms(), 1),
                "remaining_ms": round(self.remaining_ms(), 1)}
