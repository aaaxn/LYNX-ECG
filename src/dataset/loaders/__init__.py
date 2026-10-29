"""Streaming loaders for raw ECG datasets.

The builder consumes ``ExamSample`` instances. Each loader is an iterable
that yields ``ExamSample`` for one exam at a time, opening file handles
inside the iterator (so multi-process workers can each open their own).

The factory :func:`get_loader` resolves ``(dataset_source, source_variant)``
through the registry in ``dataset.sources`` and instantiates the
appropriate concrete loader. Loaders yield raw native-rate samples;
:func:`prepare_exam_sample` applies the config-driven band-pass / resample
transforms per sample.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np

from dataset.sources import get_source_variant, resolve_source_path

if TYPE_CHECKING:
    from config.models import AmplitudeNormalizationConfig, SignalFilterConfig


CODE_6CLASS_LABEL_NAMES: tuple[str, ...] = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")


@dataclass(frozen=True)
class ExamSample:
    """One exam streamed from a source.

    ``signal`` has shape ``[n_leads, n_samples]`` (leads first, samples last).
    ``labels`` is a 1-D multi-hot ``uint8`` array.
    """

    exam_id: str
    patient_id: str
    strat_fold: int
    labels: np.ndarray
    label_names: tuple[str, ...]
    label_taxonomy: str
    signal: np.ndarray
    sampling_rate: int


class ExamLoader(Protocol):
    """A loader is any iterable of ``ExamSample`` with optional length."""

    def __iter__(self) -> Iterator[ExamSample]: ...

    def __len__(self) -> int: ...


def prepare_exam_sample(
    sample: ExamSample,
    *,
    signal_filter: SignalFilterConfig | None = None,
    resample_to_hz: int | None = None,
    amplitude_normalization: AmplitudeNormalizationConfig | None = None,
    graph_leads: Sequence[int] | None = None,
) -> ExamSample:
    """Apply the config-driven source transforms to one exam.

    Order is fixed: the optional zero-phase band-pass runs first, at the
    sample's **native** rate. Then the optional anti-aliased
    resample re-stamps ``sampling_rate``, and finally the optional per-exam
    amplitude normalization rescales the (already filtered, resampled) signal
    into a gain-invariant canonical scale. Labels, ids, and taxonomy pass
    through unchanged. Resampling to the sample's own rate is a no-op.

    This is a pure per-sample function (SOS design is memoized per parameter
    set) so the build can run it inside the parallel workers instead of the
    single loader process. At high job counts the serial loader is the
    build's wall-clock cap.

    Raises:
        DegenerateAmplitudeError: ``amplitude_normalization`` is set and the
            exam's amplitude scale is zero or non-finite.
    """
    if signal_filter is not None:
        from dataset.filtering import apply_bandpass, cached_bandpass_sos

        sos = cached_bandpass_sos(
            sampling_rate=int(sample.sampling_rate),
            highpass_hz=float(signal_filter.highpass_hz),
            lowpass_hz=float(signal_filter.lowpass_hz),
            order=int(signal_filter.order),
        )
        sample = replace(sample, signal=apply_bandpass(sample.signal, sos=sos))

    if resample_to_hz is not None and int(resample_to_hz) != int(sample.sampling_rate):
        from dataset.resampling import resample_signal

        signal = resample_signal(sample.signal, sample.sampling_rate, int(resample_to_hz))
        sample = replace(sample, signal=signal, sampling_rate=int(resample_to_hz))

    if amplitude_normalization is not None:
        if graph_leads is None:
            raise ValueError("amplitude_normalization requires graph_leads to derive the scale")
        from dataset.normalization import normalize_amplitude

        sample = replace(
            sample,
            signal=normalize_amplitude(
                sample.signal, graph_leads, method=amplitude_normalization.method
            ),
        )

    return sample


def get_loader(
    dataset_source: str,
    source_variant: str,
    *,
    limit_exams: int | None = None,
) -> Iterable[ExamSample]:
    """Resolve and instantiate the loader for the given source/variant.

    Loaders yield **native-rate, untransformed** samples; the config-driven
    band-pass / resample transforms are applied per sample by
    :func:`prepare_exam_sample` (the build runs it inside the parallel
    workers).
    """
    sv = get_source_variant(dataset_source, source_variant)
    path = resolve_source_path(dataset_source, source_variant)

    if sv.dataset_source == "ptbxl":
        from dataset.loaders.ptbxl import PTBXLLoader

        loader: Iterable[ExamSample] = PTBXLLoader(
            release_dir=path.parent,
            sampling_rate=sv.sampling_rate,
            limit_exams=limit_exams,
        )
    elif sv.dataset_source == "codetest":
        from dataset.loaders.code15 import Code15Loader

        loader = Code15Loader(
            hdf5_path=path,
            sampling_rate=sv.sampling_rate,
            limit_exams=limit_exams,
        )
    elif sv.dataset_source == "code100":
        from dataset.loaders.code100 import Code100Loader

        loader = Code100Loader(
            root_dir=path,
            sampling_rate=sv.sampling_rate,
            limit_exams=limit_exams,
        )
    else:
        raise ValueError(f"No loader registered for dataset_source={sv.dataset_source!r}")

    return loader


__all__ = ["ExamLoader", "ExamSample", "get_loader", "prepare_exam_sample"]
