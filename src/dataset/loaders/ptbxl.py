"""PTB-XL loader.

Streams one exam at a time. Reads ``ptbxl_database.csv`` for metadata and
maps PTB-XL SCP codes onto the CODE 6-class taxonomy (``1dAVb, RBBB, LBBB, SB,
AF, ST``) by strict code presence, so a CODE-trained checkpoint can be scored
on PTB-XL as an external evaluation set.

Signal data comes from the 500 Hz WFDB records under ``records500/``.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from dataset.loaders import CODE_6CLASS_LABEL_NAMES, ExamSample

PTBXL_CODE6_STRICT_TAXONOMY = "code6_ptbxl_strict_scp_v1.0.3"

_CODE6_STRICT_CODES: dict[str, tuple[str, ...]] = {
    "1dAVb": ("1AVB",),
    "RBBB": ("CRBBB",),
    "LBBB": ("CLBBB",),
    "SB": ("SBRAD",),
    "AF": ("AFIB",),
    "ST": ("STACH",),
}


def code6_scp_mapping() -> dict[str, tuple[str, ...]]:
    """Return the strict SCP-code -> CODE-6 mapping."""
    return {label: tuple(codes) for label, codes in _CODE6_STRICT_CODES.items()}


def code6_labels_from_scp(scp_codes: dict, mapping: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Multi-hot CODE-6 label vector for one exam, by SCP-code presence.

    ``scp_codes`` is PTB-XL's per-exam ``{code: likelihood}`` dict; only the keys
    (presence) are read, never the likelihood values. The output is ordered by
    :data:`CODE_6CLASS_LABEL_NAMES`.
    """
    present = set(scp_codes)
    labels = np.zeros(len(CODE_6CLASS_LABEL_NAMES), dtype=np.uint8)
    for i, label in enumerate(CODE_6CLASS_LABEL_NAMES):
        if any(code in present for code in mapping[label]):
            labels[i] = 1
    return labels


class PTBXLLoader:
    """Iterable loader over the 500 Hz PTB-XL exams."""

    def __init__(
        self,
        release_dir: Path,
        sampling_rate: int,
        limit_exams: int | None = None,
    ) -> None:
        if int(sampling_rate) != 500:
            raise ValueError(f"PTB-XL loader reads the 500 Hz records; got {sampling_rate}")
        self._release_dir = Path(release_dir)
        self.sampling_rate = int(sampling_rate)
        self.limit_exams = limit_exams

    def __len__(self) -> int:
        df = self._metadata_df
        return int(min(len(df), self.limit_exams) if self.limit_exams is not None else len(df))

    @functools.cached_property
    def _metadata_df(self):
        """Fully prepared metadata + per-exam labels, built once per instance.

        A pure function of the immutable constructor inputs (``_release_dir``,
        ``limit_exams``): parsed on first access and cached
        in the instance ``__dict__``, so ``__len__`` and ``__iter__`` share one
        parse instead of re-reading the CSVs on every call. Treat the returned
        frame and its nested arrays as READ-ONLY. ``__len__`` only reads it;
        ``__iter__`` reads rows and copies each cached label array at the
        ``ExamSample`` yield boundary so caller mutation cannot corrupt the
        cache. Different limits live on different instances, so the cache never
        crosses them.
        """
        import pandas as pd

        db_path = self._release_dir / "ptbxl_database.csv"
        if not db_path.exists():
            raise FileNotFoundError(
                f"PTB-XL metadata not found under {self._release_dir}; expected ptbxl_database.csv"
            )

        df = pd.read_csv(db_path, index_col="ecg_id")
        required_columns = {"patient_id", "strat_fold"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"PTB-XL metadata is missing required columns: {missing_columns}")
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)
        if self.limit_exams is not None:
            df = df.head(int(self.limit_exams))

        for column in ("patient_id", "strat_fold"):
            raw = df[column]
            boolean = raw.map(lambda value: isinstance(value, (bool, np.bool_))).to_numpy()
            numeric = pd.to_numeric(raw, errors="coerce")
            invalid = boolean | numeric.isna() | ~np.isfinite(numeric.to_numpy())
            invalid |= numeric.to_numpy() != np.floor(numeric.to_numpy())
            if column == "patient_id":
                invalid |= numeric.to_numpy() <= 0
            else:
                invalid |= (numeric.to_numpy() < 1) | (numeric.to_numpy() > 10)
            if bool(np.any(invalid)):
                raise ValueError(f"PTB-XL metadata column {column!r} contains invalid values")

        mapping = code6_scp_mapping()
        df["_labels"] = df.scp_codes.apply(lambda scp: code6_labels_from_scp(scp, mapping))
        return df

    def __iter__(self) -> Iterator[ExamSample]:
        import wfdb

        df = self._metadata_df
        records_root = self._release_dir

        for ecg_id, row in df.iterrows():
            relative = row["filename_hr"]
            wfdb_path = (records_root / relative).with_suffix("")
            record = wfdb.rdrecord(str(wfdb_path))
            sig = np.asarray(record.p_signal, dtype=np.float32).T

            yield ExamSample(
                exam_id=str(int(ecg_id)),
                patient_id=str(int(row["patient_id"])),
                strat_fold=int(row["strat_fold"]),
                labels=np.asarray(row["_labels"], dtype=np.uint8).copy(),
                label_names=CODE_6CLASS_LABEL_NAMES,
                label_taxonomy=PTBXL_CODE6_STRICT_TAXONOMY,
                signal=sig,
                sampling_rate=self.sampling_rate,
            )
