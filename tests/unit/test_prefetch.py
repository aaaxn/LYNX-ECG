"""Tests for the background prefetch iterator."""

from __future__ import annotations

import threading
import time

import pytest

from dataset.prefetch import prefetch_iter

pytestmark = pytest.mark.unit


def test_order_and_completeness():
    assert list(prefetch_iter(range(100))) == list(range(100))


def test_empty_iterable():
    assert list(prefetch_iter(iter(()))) == []


def test_invalid_depth_rejected():
    with pytest.raises(ValueError, match="depth"):
        list(prefetch_iter(range(3), depth=0))


def test_producer_exception_reraised_at_consumer():
    def gen():
        yield 1
        yield 2
        raise RuntimeError("loader blew up")

    it = prefetch_iter(gen())
    assert next(it) == 1
    assert next(it) == 2
    with pytest.raises(RuntimeError, match="loader blew up"):
        next(it)


def test_immediate_producer_exception():
    def gen():
        raise FileNotFoundError("missing source")
        yield  # pragma: no cover

    with pytest.raises(FileNotFoundError, match="missing source"):
        next(prefetch_iter(gen()))


def test_backpressure_bounded_by_depth():
    """The producer never runs more than depth items ahead of the consumer."""
    produced = []

    def gen():
        for i in range(50):
            produced.append(i)
            yield i

    it = prefetch_iter(gen(), depth=2)
    assert next(it) == 0
    time.sleep(0.3)
    assert len(produced) <= 4
    assert list(it) == list(range(1, 50))
    assert produced == list(range(50))


def test_early_close_stops_producer():
    stopped = threading.Event()

    def gen():
        try:
            yield from range(10_000)
        finally:
            stopped.set()

    it = prefetch_iter(gen(), depth=2)
    assert next(it) == 0
    it.close()
    assert stopped.wait(timeout=5.0), "producer did not stop after consumer close"
