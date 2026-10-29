"""Pack the public CODE-Test release into the HDF5 the dataset loaders read.

CODE-Test (Zenodo 10.5281/zenodo.3625006) ships ``ecg_tracings.hdf5``,
``attributes.csv``, and ``annotations/gold_standard.csv``. The loaders read one
file, ``$ECG_DATASETS_ROOT/code15/codetest_raw.hdf5``, with the tracings, the
cardiologist gold-standard labels, and their names. CODE-Test has no exam ids,
so the row index is the exam id.

Usage:
    uv run python scripts/dataset/pack_codetest.py \\
        --release-dir /path/to/codetest/data \\
        --output "$ECG_DATASETS_ROOT/code15/codetest_raw.hdf5"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LEAD_NAMES = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABEL_NAMES = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
SAMPLING_RATE_HZ = 400


def pack(release_dir: Path, output: Path) -> None:
    with h5py.File(release_dir / "ecg_tracings.hdf5", "r") as source:
        tracings = source["tracings"][()].astype(np.float32)
    gold = pd.read_csv(release_dir / "annotations" / "gold_standard.csv")
    attributes = pd.read_csv(release_dir / "attributes.csv")
    if not len(tracings) == len(gold) == len(attributes):
        raise ValueError(
            f"row counts differ: tracings {len(tracings)}, labels {len(gold)}, "
            f"attributes {len(attributes)}"
        )
    labels = gold[LABEL_NAMES].to_numpy(dtype=np.uint8)
    with h5py.File(output, "x") as h5:
        h5.create_dataset("tracings", data=tracings)
        h5.create_dataset("exam_id", data=np.arange(len(tracings), dtype=np.int64))
        h5.create_dataset("labels", data=labels)
        h5.create_dataset("age", data=attributes["age"].to_numpy(dtype=np.int32))
        h5.create_dataset("is_male", data=(attributes["sex"] == "M").to_numpy(dtype=np.int8))
        h5.attrs["label_names"] = np.array(LABEL_NAMES, dtype=object)
        h5.attrs["lead_names"] = np.array(LEAD_NAMES, dtype=object)
        h5.attrs["num_exams"] = np.int64(len(tracings))
        h5.attrs["sampling_rate"] = np.int64(SAMPLING_RATE_HZ)
        h5.attrs["source_dataset"] = "codetest"
        h5.attrs["exam_id_semantics"] = "local_synthetic_row_index"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pack(args.release_dir, args.output)


if __name__ == "__main__":
    main()
