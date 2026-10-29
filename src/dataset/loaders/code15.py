"""CODE-test loader.

Reads the CODE-15 packed HDF5 schema that ``scripts/dataset/pack_codetest.py``
writes for CODE-test:

- ``tracings``   ``[N, T, L]`` float32 (sample-major; transposed to ``[L, T]``).
- ``exam_id``    ``[N]`` int, converted to string by the loader.
- ``patient_id`` ``[N]`` int, optional (``codetest`` lacks this key).
- ``labels``     ``[N, 6]`` uint8 multi-hot.
- ``label_names`` attr. The **physical** column order of ``labels``. The raw
  CODE files pack the classes as ``[1dAVb, RBBB, LBBB, SB, ST, AF]`` (ST before
  AF), so the loader remaps columns to the canonical
  :data:`~dataset.loaders.CODE_6CLASS_LABEL_NAMES` order *by name*. Column
  position is never trusted. Doing so silently swaps
  AF and ST. The attr is required; a file without it is rejected.

Label taxonomy: ``codetest_6class`` (the six CODE classes, cardiologist-annotated).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from dataset.labels import (
    LabelContractError,
    validate_label_matrix,
    validate_label_names,
)
from dataset.loaders import CODE_6CLASS_LABEL_NAMES, ExamSample

CODETEST_LABEL_TAXONOMY = "codetest_6class"


class Code15Loader:
    """Iterable loader over CODE-test exams in a CODE-15 packed HDF5 file."""

    def __init__(
        self,
        hdf5_path: Path,
        sampling_rate: int,
        limit_exams: int | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.sampling_rate = int(sampling_rate)
        self.limit_exams = limit_exams

    def __len__(self) -> int:
        import h5py

        with h5py.File(self.hdf5_path, "r") as f:
            n = int(f["exam_id"].shape[0])
        return min(n, self.limit_exams) if self.limit_exams else n

    @staticmethod
    def _decode_id(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _canonical_permutation(
        self, h5_file, label_names: tuple[str, ...]
    ) -> tuple[tuple[str, ...], list[int]]:
        """Return validated physical names and their permutation to canonical order.

        ``label_names[i]`` (the canonical order this loader emits) is found at
        physical column ``perm[i]`` in the raw ``labels`` dataset. Contract
        failures are raised before any label value is converted.
        """
        context = f"HDF5 source {self.hdf5_path} 'label_names'"
        if "label_names" not in h5_file.attrs:
            raise LabelContractError(
                f"{context}: attr is required to map physical label columns to the "
                f"canonical order {list(label_names)}"
            )
        raw_names = validate_label_names(
            [self._decode_id(v) for v in h5_file.attrs["label_names"]],
            context=context,
        )
        missing = [name for name in label_names if name not in raw_names]
        extra = [name for name in raw_names if name not in label_names]
        if missing or extra or len(raw_names) != len(label_names):
            raise LabelContractError(
                f"{context} must contain exactly the canonical classes once each; "
                f"missing={missing}, extra={extra}"
            )
        return raw_names, [raw_names.index(name) for name in label_names]

    def __iter__(self) -> Iterator[ExamSample]:
        for batch in self.iter_batches(batch_size=1):
            yield from batch

    def iter_batches(self, batch_size: int) -> Iterator[list[ExamSample]]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        import h5py

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 source not found: {self.hdf5_path}")

        label_names = CODE_6CLASS_LABEL_NAMES
        taxonomy = CODETEST_LABEL_TAXONOMY

        with h5py.File(self.hdf5_path, "r") as f:
            signal_key = "tracings" if "tracings" in f else "signal"
            source_exam_count = int(f["exam_id"].shape[0])
            n = source_exam_count
            if self.limit_exams is not None:
                n = min(n, int(self.limit_exams))

            exam_ids = f["exam_id"][:n]
            patient_ids = f["patient_id"][:n] if "patient_id" in f else None

            if "labels" not in f:
                raise ValueError(
                    f"HDF5 source {self.hdf5_path} has no 'labels' dataset; "
                    "CODE-test sources require [N, 6] multi-hot labels."
                )

            raw_names, perm = self._canonical_permutation(f, label_names)
            labels_ds = f["labels"]
            expected_shape = (source_exam_count, len(raw_names))
            if labels_ds.ndim != 2 or labels_ds.shape != expected_shape:
                raise ValueError(
                    f"HDF5 source {self.hdf5_path} 'labels' has shape "
                    f"{labels_ds.shape}; expected {expected_shape} "
                    f"({source_exam_count} rows for {source_exam_count} exams, "
                    f"{len(raw_names)} columns to match 'label_names' attr "
                    f"{list(raw_names)})."
                )
            raw_labels = labels_ds[:n]
            labels_all = validate_label_matrix(
                raw_labels[:, perm],
                num_exams=n,
                num_classes=len(label_names),
                context=f"HDF5 source {self.hdf5_path} 'labels'",
            )

            signal_ds = f[signal_key]
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                signals_raw = np.asarray(signal_ds[start:stop], dtype=np.float32)
                if signals_raw.ndim != 3:
                    raise ValueError(
                        f"Unexpected signal batch shape {signals_raw.shape}; expected 3-D"
                    )
                signals = np.transpose(signals_raw, (0, 2, 1))

                batch: list[ExamSample] = []
                for offset in range(stop - start):
                    k = start + offset
                    sig = signals[offset]
                    if sig.ndim != 2:
                        raise ValueError(
                            f"Unexpected signal shape at exam {k}: {sig.shape}; expected 2-D"
                        )

                    exam_id = self._decode_id(exam_ids[k])
                    patient_id = (
                        self._decode_id(patient_ids[k]) if patient_ids is not None else "unknown"
                    )
                    labels = labels_all[k].copy()

                    batch.append(
                        ExamSample(
                            exam_id=exam_id,
                            patient_id=patient_id,
                            strat_fold=-1,
                            labels=labels,
                            label_names=label_names,
                            label_taxonomy=taxonomy,
                            signal=sig,
                            sampling_rate=self.sampling_rate,
                        )
                    )
                yield batch
