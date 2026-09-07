"""One shared lock for the heavy GPU passes, so they queue instead of thrashing.

WHY
---
Two GPU-heavy passes can run at the same time on a 6 GB card:

  * the best-face scan, started speculatively in the background every time a
    person result is opened (faces_gallery.prepare_best_face), so that pressing
    "Save Face" later is instant
  * the Track Person target pass (track_target.retrack), which the officer is
    actively waiting for

Measured on this machine: the Track Person pass takes 2.9 s on its own, and 34.6 s
while a face-prepare scan is running - a 12x slowdown. Neither pass is doing
anything wrong; they are simply fighting for the same GPU, and both end up slower
than if they had taken turns.

WHAT THIS CHANGES
-----------------
Scheduling only. Every pass still examines exactly the same frames with the same
models, thresholds and decision logic, and produces the same result - it just may
wait briefly before starting. Nothing here can alter an outcome.

PRIORITY
--------
The face-prepare scan is speculative: it only pre-warms a cache, and Save Face
works correctly (just slower) without it. So a user-facing Track Person request
takes priority, and the scan steps aside rather than making the officer wait
~19 s behind work they never asked for.

HOW (and why not a plain mutex)
-------------------------------
A plain exclusive lock was tried first and measured: it does NOT help the officer.
It converts "both passes thrash" into "the officer queues behind a ~19 s scan they
never asked for" - the Track Person wall time stayed at ~34 s either way.

So this is COOPERATIVE instead. The user-facing pass simply marks itself active;
the speculative scan checks that between frames and pauses until it clears. The
scan is never cancelled and never skips a frame, so the face it eventually picks is
bit-identical - it just finishes a little later.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

_counter_lock = threading.Lock()
_priority_waiting = 0


@contextmanager
def hold(priority: bool = False):
    """Mark a user-facing GPU pass as active for its duration.

    Deliberately NOT an exclusive lock: blocking the officer behind speculative
    background work is the problem this exists to avoid.
    """
    global _priority_waiting
    if not priority:
        yield
        return
    with _counter_lock:
        _priority_waiting += 1
    try:
        yield
    finally:
        with _counter_lock:
            _priority_waiting -= 1


def priority_active() -> bool:
    """True while a user-facing GPU pass is running."""
    with _counter_lock:
        return _priority_waiting > 0


def yield_to_priority(poll: float = 0.05, max_wait: float = 180.0) -> None:
    """Pause speculative background GPU work while a user request is running.

    Called between frames of the best-face scan. It only ever WAITS - it does not
    skip work or shorten the scan - so the scan's result cannot change.
    `max_wait` is a safety valve so a stuck priority flag can never wedge the
    background worker forever.
    """
    waited = 0.0
    while priority_active() and waited < max_wait:
        time.sleep(poll)
        waited += poll
