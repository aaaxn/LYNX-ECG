"""Unit tests for example-weighted micro-batch gradient accumulation.

These exercise ``training.train.run_accumulated_training_step`` and
``run_train_epoch`` (added for ``OptimConfig.gradient_accumulation_steps``)
without a GPU, a real model, or a distributed job: ``forward_batch`` is
monkeypatched to run a plain ``nn.Linear`` on a trivial batch so the tests
isolate the accumulation *arithmetic*: the ``*n_i`` weighting, the single
``÷N`` normalization, the once-per-window clip/step, and the leftover partial
window, rather than the CODE100 model.

The load-bearing property: a window's accumulated gradient must equal the
gradient of the mean loss over the *concatenated* examples, i.e. weighting each
micro-batch by its example count (not ``1/num_micro_batches``). The unequal-size
leftover test is what distinguishes the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from config.train_models import OptimConfig
from training import train

pytestmark = pytest.mark.unit


@dataclass
class _FakeOutput:
    """Minimal stand-in for ``models.hier_model.ModelOutput``, only the fields
    the training step reads."""

    logits: torch.Tensor


class _FakeBatch:
    """A collated micro-batch: ``x`` features and ``y`` targets, with the
    ``.to(device)`` a PyG ``Batch`` supports."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def to(self, device: torch.device, non_blocking: bool = False) -> _FakeBatch:
        return self


def _fake_forward_batch(model: nn.Module, batch: _FakeBatch, device: torch.device) -> _FakeOutput:
    logits = model(batch.x.to(device))
    return _FakeOutput(
        logits=logits,
    )


_DEVICE = torch.device("cpu")


def _grads(model: nn.Module) -> list[torch.Tensor]:
    return [p.grad.detach().clone() for p in model.parameters()]


def _reference_mean_loss_grads(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> list[torch.Tensor]:
    """Gradient of the mean loss over the whole (concatenated) batch, the
    ground truth a full accumulation window must reproduce."""
    model.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    grads = _grads(model)
    model.zero_grad(set_to_none=True)
    return grads


def _split(x: torch.Tensor, y: torch.Tensor, sizes: list[int]) -> list[_FakeBatch]:
    windowed: list[_FakeBatch] = []
    start = 0
    for n in sizes:
        windowed.append(_FakeBatch(x[start : start + n], y[start : start + n]))
        start += n
    return windowed


class TestFullWindowEquivalence:
    def test_matches_true_larger_batch(self, monkeypatch):
        """Four equal micro-batches accumulate into the same gradient as one
        forward/backward over their concatenation, the memory-limited run
        reproducing the reference effective batch size."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        torch.manual_seed(0)
        d, c, micro, accum = 4, 3, 5, 4
        n = micro * accum
        x = torch.randn(n, d)
        y = (torch.rand(n, c) > 0.5).float()

        model = nn.Linear(d, c)
        expected = _reference_mean_loss_grads(model, x, y)

        window = _split(x, y, [micro] * accum)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        stats = train.run_accumulated_training_step(
            model, optimizer, window, _DEVICE, F.binary_cross_entropy_with_logits, None
        )
        got = _grads(model)

        assert len(stats) == accum
        assert sum(s.batch_size for s in stats) == n
        for exp, act in zip(expected, got, strict=True):
            assert torch.allclose(exp, act, atol=1e-6, rtol=1e-5)


class TestLeftoverPartialWindow:
    def test_unequal_micro_batches_are_example_weighted(self, monkeypatch):
        """A leftover window of *unequal* micro-batches (2 then 3) must weight by
        example count: its gradient equals the mean over the 5 concatenated
        examples. A naive ``1/num_micro_batches`` average would be wrong here."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        torch.manual_seed(1)
        d, c = 4, 2
        x = torch.randn(5, d)
        y = (torch.rand(5, c) > 0.5).float()

        model = nn.Linear(d, c)
        expected = _reference_mean_loss_grads(model, x, y)

        window = _split(x, y, [2, 3])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        stats = train.run_accumulated_training_step(
            model, optimizer, window, _DEVICE, F.binary_cross_entropy_with_logits, None
        )
        got = _grads(model)

        assert [s.batch_size for s in stats] == [2, 3]
        for exp, act in zip(expected, got, strict=True):
            assert torch.allclose(exp, act, atol=1e-6, rtol=1e-5)

    def test_naive_uniform_average_would_disagree(self, monkeypatch):
        """Guardrail: the reference mean gradient must NOT match a uniform
        ``1/num_micro_batches`` average of per-micro-batch mean gradients when
        sizes differ, proving the example-weighting is doing real work."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        torch.manual_seed(2)
        d, c = 4, 2
        x = torch.randn(5, d)
        y = (torch.rand(5, c) > 0.5).float()
        model = nn.Linear(d, c)

        expected = _reference_mean_loss_grads(model, x, y)

        naive: list[torch.Tensor] = []
        for sl in (slice(0, 2), slice(2, 5)):
            model.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(x[sl]), y[sl]).backward()
            naive.append(_grads(model))
        model.zero_grad(set_to_none=True)
        uniform = [(a + b) / 2 for a, b in zip(naive[0], naive[1], strict=True)]

        assert not all(
            torch.allclose(exp, unf, atol=1e-6, rtol=1e-5)
            for exp, unf in zip(expected, uniform, strict=True)
        )


class TestOptimizerStepAccounting:
    @pytest.mark.parametrize("window_sizes", [[3], [3, 3], [3, 3, 2]])
    def test_one_zero_grad_clip_and_step_per_window(self, monkeypatch, window_sizes):
        """Regardless of window size, a window zeros gradients once, clips once,
        and steps the optimizer once."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        clip_calls = {"n": 0}
        real_clip = nn.utils.clip_grad_norm_

        def counting_clip(*args, **kwargs):
            clip_calls["n"] += 1
            return real_clip(*args, **kwargs)

        monkeypatch.setattr(nn.utils, "clip_grad_norm_", counting_clip)

        torch.manual_seed(3)
        d, c = 4, 2
        model = nn.Linear(d, c)
        window = [
            _FakeBatch(torch.randn(n, d), (torch.rand(n, c) > 0.5).float()) for n in window_sizes
        ]
        optimizer = MagicMock(wraps=torch.optim.SGD(model.parameters(), lr=0.0))

        train.run_accumulated_training_step(
            model, optimizer, window, _DEVICE, F.binary_cross_entropy_with_logits, 1.0
        )

        assert optimizer.zero_grad.call_count == 1
        assert optimizer.step.call_count == 1
        assert clip_calls["n"] == 1

    def test_single_micro_batch_window_delegates_to_run_training_step(self, monkeypatch):
        """A window of one delegates to ``run_training_step`` (the benchmark's
        one-step source) so accumulation=1 is bit-for-bit unchanged; a larger
        window takes the accumulation path instead."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        calls = {"n": 0}
        real_step = train.run_training_step

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real_step(*args, **kwargs)

        monkeypatch.setattr(train, "run_training_step", spy)

        torch.manual_seed(4)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        one = [_FakeBatch(torch.randn(3, 4), (torch.rand(3, 2) > 0.5).float())]
        train.run_accumulated_training_step(
            model, optimizer, one, _DEVICE, F.binary_cross_entropy_with_logits, None
        )
        assert calls["n"] == 1

        two = [
            _FakeBatch(torch.randn(3, 4), (torch.rand(3, 2) > 0.5).float()),
            _FakeBatch(torch.randn(3, 4), (torch.rand(3, 2) > 0.5).float()),
        ]
        train.run_accumulated_training_step(
            model, optimizer, two, _DEVICE, F.binary_cross_entropy_with_logits, None
        )
        assert calls["n"] == 1


class TestIterWindows:
    @pytest.mark.parametrize(
        ("count", "size", "expected"),
        [
            (10, 4, [4, 4, 2]),
            (8, 4, [4, 4]),
            (5, 1, [1, 1, 1, 1, 1]),
            (3, 5, [3]),
            (0, 4, []),
        ],
    )
    def test_chunk_sizes(self, count, size, expected):
        windows = list(train._iter_windows(range(count), size))
        assert [len(w) for w in windows] == expected


class TestRunTrainEpoch:
    def _loader(self, num_micro: int, per: int, d: int, c: int) -> list[_FakeBatch]:
        torch.manual_seed(7)
        return [
            _FakeBatch(torch.randn(per, d), (torch.rand(per, c) > 0.5).float())
            for _ in range(num_micro)
        ]

    def test_steps_once_per_window_with_partial(self, monkeypatch):
        """10 micro-batches at accumulation=4 -> 3 optimizer steps (windows
        [4, 4, 2]); totals count all 10 forward passes and all examples."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        d, c, per = 4, 2, 3
        loader = self._loader(10, per, d, c)
        model = nn.Linear(d, c)
        optimizer = MagicMock(wraps=torch.optim.SGD(model.parameters(), lr=0.0))
        cfg = OptimConfig(gradient_accumulation_steps=4, grad_clip_norm=None)

        totals = train.run_train_epoch(
            model, optimizer, loader, _DEVICE, F.binary_cross_entropy_with_logits, cfg, epoch=1
        )

        assert optimizer.step.call_count == 3
        assert totals.example_count == 10 * per

    def test_accumulation_one_steps_per_micro_batch(self, monkeypatch):
        """The default (no accumulation): one optimizer step per micro-batch,
        preserving pre-accumulation behavior."""
        monkeypatch.setattr(train, "forward_batch", _fake_forward_batch)
        d, c, per = 4, 2, 3
        loader = self._loader(5, per, d, c)
        model = nn.Linear(d, c)
        optimizer = MagicMock(wraps=torch.optim.SGD(model.parameters(), lr=0.0))
        cfg = OptimConfig(gradient_accumulation_steps=1, grad_clip_norm=None)

        totals = train.run_train_epoch(
            model, optimizer, loader, _DEVICE, F.binary_cross_entropy_with_logits, cfg, epoch=1
        )

        assert optimizer.step.call_count == 5
        assert totals.example_count == 5 * per
