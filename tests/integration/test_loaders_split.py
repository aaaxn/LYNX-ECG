"""Regression tests for the build_datasets / build_loaders_from_datasets split."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.loaders import build_datasets

pytestmark = [pytest.mark.integration]


def test_build_datasets_routes_test_to_alternate_path(monkeypatch):
    """When test_dataset_path is given, train/val use dataset_path and test uses the alternate."""
    from config.train_models import SubgraphRepresentationConfig

    representation = SubgraphRepresentationConfig(
        adjacency="temporal_chain",
        edge_attributes="zero",
    )
    constructed: list[dict] = []

    class RecordingDataset:
        def __init__(
            self,
            path,
            indices,
            *,
            num_leads,
            arrays=None,
            symmetrize_edges=False,
            subgraph_representation=None,
        ):
            constructed.append(
                {
                    "path": path,
                    "arrays": arrays,
                    "symmetrize_edges": symmetrize_edges,
                    "subgraph_representation": subgraph_representation,
                }
            )

        def __len__(self):
            return 1

    arrays_by_path: dict[str, object] = {}

    def _fake_from_hdf5(path: object) -> object:
        return arrays_by_path.setdefault(str(path), object())

    monkeypatch.setattr(
        "training.loaders.GraphIndexArrays",
        type("FakeGIA", (), {"from_hdf5": staticmethod(_fake_from_hdf5)}),
    )
    monkeypatch.setattr("training.loaders.ExamGraphDataset", RecordingDataset)

    build_datasets(
        dataset_path=Path("/tmp/main.h5"),
        num_leads=12,
        train_idx=[0],
        val_idx=[1],
        test_idx=[2],
        test_dataset_path=Path("/tmp/test.h5"),
        subgraph_representation=representation,
    )

    assert len(constructed) == 3
    assert constructed[0]["path"] == Path("/tmp/main.h5")
    assert constructed[1]["path"] == Path("/tmp/main.h5")
    assert constructed[2]["path"] == Path("/tmp/test.h5")
    assert all(c["subgraph_representation"] is representation for c in constructed)

    assert constructed[0]["arrays"] is constructed[1]["arrays"]
    assert constructed[2]["arrays"] is not constructed[0]["arrays"]
