"""Background prefetch for iterators.

The build pipeline is a serial producer (loader I/O in the main process)
feeding parallel consumers (worker pool). Without overlap, wall-clock is
``load + process`` per chunk; :func:`prefetch_iter` runs the producer on a
daemon thread with a bounded queue so the next chunk loads while the current
one is being processed, bringing wall-clock toward ``max(load, process)``.

Exceptions raised by the producer are re-raised at the consumer's ``next()``
call site, preserving the unwrapped iteration's error semantics. Closing the
generator early (``break`` / ``.close()``) signals the producer to stop; the
thread is a daemon, so a producer blocked inside the underlying iterable can
never prevent interpreter exit.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import cast

_STOP_POLL_SECONDS = 0.1


def prefetch_iter[T](iterable: Iterable[T], *, depth: int = 2) -> Iterator[T]:
    """Yield from *iterable*, producing up to *depth* items ahead on a thread.

    Item order is preserved. The producer thread starts on the first
    ``next()`` call (generator semantics), holds at most *depth* finished
    items, and blocks (backpressure) once the queue is full.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def _put(message: tuple[str, object]) -> None:
        """Enqueue, but abandon the message if the consumer has gone away."""
        while not stop.is_set():
            try:
                q.put(message, timeout=_STOP_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def _produce() -> None:
        try:
            for item in iterable:
                _put(("item", item))
                if stop.is_set():
                    return
            _put(("done", None))
        except BaseException as exc:
            _put(("error", exc))

    thread = threading.Thread(target=_produce, name="prefetch-producer", daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = q.get()
            if kind == "item":
                yield cast("T", payload)
            elif kind == "done":
                return
            else:
                assert isinstance(payload, BaseException)
                raise payload
    finally:
        stop.set()
