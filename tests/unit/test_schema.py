"""Tests for the current HDF5 dataset contract."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import pytest

from dataset.records import GraphRecord
from dataset.schema import (
    GRAPH_SCALAR_SPECS,
    require_dataset_contract,
)
from tests.fixtures.synthetic_graph_hdf5 import build_tiny_hdf5

pytestmark = [pytest.mark.unit]


def test_every_graph_scalar_spec_maps_to_a_graph_record_attribute():
    record_fields = {f.name for f in dataclasses.fields(GraphRecord)}
    for spec in GRAPH_SCALAR_SPECS:
        assert spec.kind == "graph", spec.name
        is_field = spec.graph_field in record_fields
        is_property = isinstance(getattr(GraphRecord, spec.graph_field, None), property)
        assert is_field or is_property, (
            f"{spec.name}: graph_field {spec.graph_field!r} is not a GraphRecord attribute"
        )


def test_require_dataset_contract_rejects_nonbinary_stored_labels(tmp_path: Path):
    path = Path(build_tiny_hdf5(tmp_path))
    with h5py.File(path, "r+") as h5:
        h5["exam_labels"][0, 0] = 2

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="binary"):
        require_dataset_contract(h5, path)


def test_require_dataset_contract_rejects_duplicate_root_label_names(tmp_path: Path):
    path = Path(build_tiny_hdf5(tmp_path))
    with h5py.File(path, "r+") as h5:
        h5.attrs["exam_label_names"] = ["1dAVb", "1dAVb", "LBBB", "SB", "AF"]

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="unique"):
        require_dataset_contract(h5, path)
