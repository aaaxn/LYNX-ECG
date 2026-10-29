"""CODE-100 (full CODE) loader.

Streams the *first* WFDB record (``_N1``) of each annotated exam from the raw
CODE dataset staged as a WFDB tree under ``$ECG_DATASETS_ROOT/code100/``:

- ``annotations.csv``. Per-exam attributes + the six binary labels
  (``1dAVb, RBBB, LBBB, SB, AF, ST``), keyed by ``id_exam`` / ``id_patient``.
- ``S<XXX>0000/TNMG<id_exam>_N<k>.hea`` + ``.dat``. WFDB records. Exam ``E``
  lives in shard ``S<floor(E/10000)*10000>`` (zero-padded to 7 digits). An exam
  may have several successive records (``_Nk``); this loader takes ``_N1`` only,
  matching the CODE-15 / Nature-paper convention.

The raw CODE waveforms store the **8 physically-recorded leads**
(``DI, DII, V1..V6``); this loader reconstructs the standard **12-lead** signal
(``DI, DII, DIII, AVR, AVL, AVF, V1..V6``, identical to code15/codetest) by
deriving the four augmented limb leads as exact linear combinations of DI and
DII (see :func:`derive_twelve_lead`).

CODE-100 is natively mixed-rate (predominantly 300 Hz with a ~2.7% 600 Hz
minority). The dataset HDF5 stores a single file-level ``sampling_rate``, so
this loader keeps only records whose header rate equals the configured
``sampling_rate`` and skips records at any other rate, yielding a genuinely
homogeneous dataset. Annotated exams whose ``_N1`` record is missing on disk
are likewise skipped. Both skip counts are logged at end of iteration.

Label taxonomy: ``code100_6class`` (same six classes and order as CODE-15).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from dataset.labels import validate_label_matrix
from dataset.loaders import CODE_6CLASS_LABEL_NAMES, ExamSample

logger = logging.getLogger(__name__)

CODE100_LABEL_TAXONOMY = "code100_6class"

CODE100_STORED_LEADS: tuple[str, ...] = ("DI", "DII", "V1", "V2", "V3", "V4", "V5", "V6")
TWELVE_LEAD_ORDER: tuple[str, ...] = (
    "DI",
    "DII",
    "DIII",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

_SHARD_SPAN = 10_000


def derive_twelve_lead(signal: np.ndarray, lead_names: Sequence[str]) -> np.ndarray:
    """Reconstruct the standard 12-lead signal from the 8 stored CODE leads.

    A diagnostic 12-lead ECG has only 8 independent signals: ``DI``, ``DII`` and
    ``V1..V6`` (what CODE stores). The four augmented limb leads are exact linear
    combinations of ``DI`` and ``DII``, the same construction code15/codetest use::

        DIII = DII - DI
        AVR  = -(DI + DII) / 2
        AVL  = DI - DII / 2
        AVF  = DII - DI / 2

    ``signal`` is ``[n_stored_leads, T]``; leads are matched to ``lead_names`` **by
    name** (case-insensitive), so the result is correct regardless of input row
    order. Returns ``[12, T]`` float32 in :data:`TWELVE_LEAD_ORDER`.

    Raises ``ValueError`` if any of the 8 required stored leads is absent from
    ``lead_names`` (surplus leads are ignored).
    """
    sig = np.asarray(signal, dtype=np.float32)
    pos = {str(name).upper(): i for i, name in enumerate(lead_names)}
    missing = [lead for lead in CODE100_STORED_LEADS if lead not in pos]
    if missing:
        raise ValueError(
            f"cannot derive 12-lead signal: stored leads {missing} absent from "
            f"{[str(n).upper() for n in lead_names]}"
        )

    di = sig[pos["DI"]]
    dii = sig[pos["DII"]]
    derived = {
        "DIII": dii - di,
        "AVR": -(di + dii) / 2.0,
        "AVL": di - dii / 2.0,
        "AVF": dii - di / 2.0,
    }

    out = np.empty((len(TWELVE_LEAD_ORDER), sig.shape[1]), dtype=np.float32)
    for i, lead in enumerate(TWELVE_LEAD_ORDER):
        out[i] = derived[lead] if lead in derived else sig[pos[lead]]
    return out


def shard_dir_name(exam_id: int) -> str:
    """Return the shard directory holding ``exam_id`` (e.g. 120000 -> 'S0120000')."""
    return f"S{(exam_id // _SHARD_SPAN) * _SHARD_SPAN:07d}"


class Code100Loader:
    """Iterable loader over annotated CODE-100 exams (first WFDB record each)."""

    def __init__(
        self,
        root_dir: Path,
        sampling_rate: int,
        limit_exams: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sampling_rate = int(sampling_rate)
        self.limit_exams = limit_exams
        self._annotations_path = self.root_dir / "annotations.csv"
        self._df = None

    def _load_annotations(self):
        import pandas as pd

        if self._df is None:
            if not self._annotations_path.exists():
                raise FileNotFoundError(f"CODE-100 annotations not found: {self._annotations_path}")
            usecols = ["id_exam", "id_patient", *CODE_6CLASS_LABEL_NAMES]
            df = pd.read_csv(self._annotations_path, usecols=usecols)
            if self.limit_exams is not None:
                df = df.head(int(self.limit_exams))
            self._df = df.reset_index(drop=True)
        return self._df

    def __len__(self) -> int:
        return len(self._load_annotations())

    def _record_path(self, exam_id: int) -> Path:
        return self.root_dir / shard_dir_name(exam_id) / f"TNMG{exam_id}_N1"

    def __iter__(self) -> Iterator[ExamSample]:
        import wfdb

        df = self._load_annotations()
        exam_ids = df["id_exam"].to_numpy()
        patient_ids = df["id_patient"].to_numpy()
        context = f"CODE-100 annotations {self._annotations_path}"
        validated_columns = [
            validate_label_matrix(
                df[[name]].to_numpy(),
                num_exams=len(df),
                num_classes=1,
                context=f"{context} column {name!r}",
            )
            for name in CODE_6CLASS_LABEL_NAMES
        ]
        labels_all = np.column_stack(validated_columns)

        n_missing = 0
        n_off_rate = 0
        n_bad_leads = 0
        for i in range(len(df)):
            exam_id = int(exam_ids[i])
            record_path = self._record_path(exam_id)
            if not record_path.with_suffix(".hea").exists():
                n_missing += 1
                continue

            record = wfdb.rdrecord(str(record_path))
            fs = round(float(record.fs))
            if fs != self.sampling_rate:
                n_off_rate += 1
                continue

            stored = np.asarray(record.p_signal, dtype=np.float32).T
            try:
                signal = derive_twelve_lead(stored, record.sig_name)
            except ValueError as exc:
                n_bad_leads += 1
                logger.warning("CODE-100: skipping exam %d with unexpected leads: %s", exam_id, exc)
                continue

            yield ExamSample(
                exam_id=str(exam_id),
                patient_id=str(int(patient_ids[i])),
                strat_fold=-1,
                labels=labels_all[i].copy(),
                label_names=CODE_6CLASS_LABEL_NAMES,
                label_taxonomy=CODE100_LABEL_TAXONOMY,
                signal=signal,
                sampling_rate=fs,
            )

        if n_missing or n_off_rate or n_bad_leads:
            logger.warning(
                "CODE-100: skipped %d/%d annotated exams "
                "(%d missing _N1 record, %d off-rate (!= %d Hz), %d unexpected lead set)",
                n_missing + n_off_rate + n_bad_leads,
                len(df),
                n_missing,
                n_off_rate,
                self.sampling_rate,
                n_bad_leads,
            )
