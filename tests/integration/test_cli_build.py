"""End-to-end integration tests for the ``dataset`` CLI.

Generates a tiny synthetic CODE-test HDF5 source under a tmp
``ECG_DATASETS_ROOT`` and exercises the build CLI against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import cli.build as build_cli
import ecg_artifact_io as artifact_io
from cli.build import main as build_main
from cli.qa import main as qa_main
from dataset.loaders import ExamSample
from dataset.records import ExamMetadata, ExamResult, GraphRecord, LeadSummary, pack_graphs

pytestmark = pytest.mark.integration

_CODE_PHYSICAL_LABEL_NAMES = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]


def _make_success_result(sample):
    graph = GraphRecord(
        exam_id=sample.exam_id,
        lead=0,
        beat_index=0,
        boundary_type=0,
        start_sample=0,
        end_sample=1,
        r_peak_sample=1,
        rpeak_polarity=1,
        original_num_nodes=2,
        node_x=np.zeros((2, 1), dtype=np.float32),
        node_t=np.zeros(2, dtype=np.float32),
        node_original_sample=np.arange(2, dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    metadata = ExamMetadata(
        exam_id=sample.exam_id,
        patient_id=sample.patient_id,
        strat_fold=sample.strat_fold,
        labels=sample.labels,
        label_names=sample.label_names,
        label_taxonomy=sample.label_taxonomy,
    )
    return ExamResult(
        exam_id=sample.exam_id,
        success=True,
        metadata=metadata,
        leads=[LeadSummary(lead=0, valid=True, n_graphs=1)],
        packed=pack_graphs([graph]),
    )


class _FakeWriterBase:
    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self._n_graphs = 0
        self._n_nodes = 0
        self._n_edges = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    @property
    def n_graphs(self) -> int:
        return self._n_graphs

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def n_edges(self) -> int:
        return self._n_edges

    def _tally(self, packed):
        self._n_graphs += packed.n_graphs
        self._n_nodes += int(packed.node_counts.sum())
        self._n_edges += int(packed.edge_counts.sum())

    def add_packed_exam_batch(self, exam_batches):
        for _metadata, packed in exam_batches:
            self._tally(packed)


class _PassingQA:
    success = True

    def errors(self):
        return []


def _qrs_pattern(n_samples: int, sampling_rate: int = 400, heart_rate: int = 70) -> np.ndarray:
    """Hand-crafted ECG-like signal with periodic QRS spikes.

    Deterministic, fast, and detectable by NeuroKit2 ecg_peaks. Avoids the
    nk.ecg_simulate path that has shown intermittent hangs under pytest on
    Python 3.14 + NumPy 2.x.
    """
    period_samples = int(sampling_rate * 60 / heart_rate)
    sig = 0.05 * np.sin(2 * np.pi * np.arange(n_samples) / sampling_rate * 1.0)
    spike_offset = period_samples // 2
    for r in range(spike_offset, n_samples, period_samples):
        for k in range(-2, 3):
            j = r + k
            if 0 <= j < n_samples:
                sig[j] += 1.0 * (1.0 - abs(k) / 3.0)
        t_offset = int(0.08 * sampling_rate)
        for k in range(-3, 4):
            j = r + t_offset + k
            if 0 <= j < n_samples:
                sig[j] += 0.15 * (1.0 - abs(k) / 4.0)
    return sig.astype(np.float32)


def _make_synthetic_codetest(path: Path, n_exams: int = 5, n_samples: int = 4096) -> None:
    """Write a tiny CODE-test HDF5 (the packed CODE-15 layout) with synthetic ECG signals."""
    rng = np.random.default_rng(42)
    sigs = np.zeros((n_exams, n_samples, 12), dtype=np.float32)
    for k in range(n_exams):
        hr = 60 + (k * 5) % 40
        sig = _qrs_pattern(n_samples, sampling_rate=400, heart_rate=hr)
        for lead in range(12):
            sigs[k, :, lead] = (
                sig * (0.7 + 0.05 * lead)
                + rng.standard_normal(n_samples).astype(np.float32) * 0.001
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=sigs)
        f.create_dataset("exam_id", data=np.arange(n_exams, dtype=np.int64))
        labels = np.zeros((n_exams, 6), dtype=np.uint8)
        labels[:, 0] = 1
        f.create_dataset("labels", data=labels)
        f.attrs["label_names"] = np.array(_CODE_PHYSICAL_LABEL_NAMES, dtype=object)
        f.create_dataset("age", data=np.full(n_exams, 50, dtype=np.int32))
        f.create_dataset("is_male", data=np.ones(n_exams, dtype=np.int8))


def _make_smoke_config(repo_root: Path, hdf5_path: Path) -> Path:
    config_path = repo_root / "configs" / "dataset" / "smoke_test.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_source": "codetest",
        "source_variant": "raw_400hz",
        "sampling_rate": 400,
        "graph_leads": [0, 1, 11],
        "node_reduction": {
            "method": "turning_points",
            "turning_points": {"cutoff_hz": 40.0, "max_gap_ms": 80.0},
        },
    }
    config_path.write_text(json.dumps(payload, indent=2))
    return config_path


def _h5_observables(path: Path) -> tuple[dict, dict]:
    """Return consumer-visible attributes and dataset values."""

    def observable(value):
        array = np.asarray(value)
        return str(array.dtype), array.shape, repr(array.tolist())

    with h5py.File(path, "r") as h5:
        attrs = {name: observable(value) for name, value in h5.attrs.items()}
        datasets = {}

        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                datasets[name] = observable(item[()])

        h5.visititems(collect)
    return attrs, datasets


def _staging_paths(output_path: Path) -> list[Path]:
    return list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_dataset_cli_smoke_build_passes(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))

    _make_synthetic_codetest(datasets_root / "code15" / "codetest_raw.hdf5", n_exams=5)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, datasets_root / "code15" / "codetest_raw.hdf5")

    code = build_main(
        [
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--jobs",
            "1",
        ]
    )
    assert code == 0

    output_dir = repo_root / "graph_datasets" / "codetest"
    h5_files = list(output_dir.glob("*.h5"))
    assert len(h5_files) == 1, f"expected 1 HDF5, got {h5_files}"
    output_h5 = h5_files[0]

    reports = list((repo_root / "reports").iterdir())
    assert len(reports) == 1
    assert (reports[0] / "report.json").exists()

    qa_code = qa_main(["--h5", str(output_h5), "--profile", "full"])
    assert qa_code == 0


def test_dataset_cli_resampled_build_records_effective_rate(tmp_path, monkeypatch):
    """A build with ``resample_to_hz`` set produces graphs at the resampled rate:
    the HDF5 records the effective rate (and the source rate for provenance), and
    the output filename carries the resample token."""
    datasets_root = tmp_path / "ecg_datasets"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(datasets_root / "code15" / "codetest_raw.hdf5", n_exams=5)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = repo_root / "configs" / "dataset" / "resample_test.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "dataset_source": "codetest",
                "source_variant": "raw_400hz",
                "sampling_rate": 400,
                "resample_to_hz": 300,
                "graph_leads": [0, 1, 11],
                "node_reduction": {
                    "method": "turning_points",
                    "turning_points": {"cutoff_hz": 40.0},
                },
            }
        )
    )

    code = build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])
    assert code == 0

    output_h5 = next((repo_root / "graph_datasets" / "codetest").glob("*.h5"))
    assert "resample-300hz" in output_h5.name
    with h5py.File(output_h5, "r") as f:
        assert int(f.attrs["sampling_rate"]) == 300
        assert int(f.attrs["source_sampling_rate"]) == 400


def test_dataset_cli_overwrite_protection(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(datasets_root / "code15" / "codetest_raw.hdf5", n_exams=3)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, datasets_root / "code15" / "codetest_raw.hdf5")

    assert (
        build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])
        == 0
    )
    output_h5 = next((repo_root / "graph_datasets" / "codetest").glob("*.h5"))
    with h5py.File(output_h5, "r") as h5:
        assert h5["exam_id"].shape == (3,)

    assert (
        build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])
        == 2
    )

    _make_synthetic_codetest(datasets_root / "code15" / "codetest_raw.hdf5", n_exams=5)
    assert (
        build_main(
            [
                "--config",
                str(config_path),
                "--repo-root",
                str(repo_root),
                "--jobs",
                "1",
                "--overwrite",
            ]
        )
        == 0
    )
    with h5py.File(output_h5, "r") as h5:
        assert h5["exam_id"].shape == (5,)
    assert qa_main(["--h5", str(output_h5), "--profile", "full"]) == 0
    assert _staging_paths(output_h5) == []


def test_dataset_cli_failed_overwrite_preserves_public_artifact(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    source_path = datasets_root / "code15" / "codetest_raw.hdf5"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(source_path, n_exams=3)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, source_path)
    args = ["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"]
    assert build_main(args) == 0

    output_h5 = next((repo_root / "graph_datasets" / "codetest").glob("*.h5"))
    original = _h5_observables(output_h5)
    _make_unfittable_codetest(source_path, n_exams=3)

    assert build_main([*args, "--overwrite"]) == 1
    assert _h5_observables(output_h5) == original
    assert qa_main(["--h5", str(output_h5), "--profile", "full"]) == 0
    assert _staging_paths(output_h5) == []


def test_dataset_cli_failed_quick_qa_preserves_public_artifact(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    source_path = datasets_root / "code15" / "codetest_raw.hdf5"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(source_path, n_exams=3)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, source_path)
    args = ["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"]
    assert build_main(args) == 0

    output_h5 = next((repo_root / "graph_datasets" / "codetest").glob("*.h5"))
    original = _h5_observables(output_h5)
    qa_paths = []

    class FailedQA:
        success = False

        def errors(self):
            return [type("Issue", (), {"code": "injected", "message": "rejected"})()]

    def reject_staged(path, *, profile):
        qa_paths.append(Path(path))
        assert profile == "quick"
        assert Path(path) != output_h5
        return FailedQA()

    monkeypatch.setattr(build_cli, "validate_dataset", reject_staged)

    assert build_main([*args, "--overwrite"]) == 1
    assert len(qa_paths) == 1
    assert _h5_observables(output_h5) == original
    assert _staging_paths(output_h5) == []


def test_dataset_cli_publication_race_preserves_winner(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    source_path = datasets_root / "code15" / "codetest_raw.hdf5"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(source_path, n_exams=3)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, source_path)
    output_paths = []
    real_link = artifact_io.os.link

    def publish_winner_then_link(staged_path, output_path):
        staged_path = Path(staged_path)
        output_path = Path(output_path)
        output_path.write_bytes(staged_path.read_bytes())
        with h5py.File(output_path, "r+") as h5:
            h5.attrs["race_winner"] = True
        output_paths.append(output_path)
        return real_link(staged_path, output_path)

    monkeypatch.setattr(artifact_io.os, "link", publish_winner_then_link)

    code = build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])

    assert code == 2
    assert len(output_paths) == 1
    output_h5 = output_paths[0]
    with h5py.File(output_h5, "r") as h5:
        assert bool(h5.attrs["race_winner"])
        assert h5["exam_id"].shape == (3,)
    assert qa_main(["--h5", str(output_h5), "--profile", "full"]) == 0
    assert _staging_paths(output_h5) == []


def test_dataset_cli_missing_source_exits_2(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, datasets_root / "code15" / "codetest_raw.hdf5")

    code = build_main(
        [
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--jobs",
            "1",
        ]
    )
    assert code == 2


def test_dataset_cli_invalid_first_source_label_exits_2_without_hdf5(tmp_path, monkeypatch):
    datasets_root = tmp_path / "ecg_datasets"
    source_path = datasets_root / "code15" / "codetest_raw.hdf5"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(source_path, n_exams=3)
    with h5py.File(source_path, "r+") as source:
        labels = np.asarray(source["labels"])
        labels[0, _CODE_PHYSICAL_LABEL_NAMES.index("AF")] = 2
        source["labels"][...] = labels

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, source_path)

    assert (
        build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])
        == 2
    )

    output_dir = repo_root / "graph_datasets" / "codetest"
    assert not list(output_dir.glob("*.h5"))
    assert not list(output_dir.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("second_names", "second_taxonomy"),
    [
        (("RBBB", "1dAVb", "LBBB", "SB", "AF", "ST"), "codetest_6class"),
        (("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"), "other_taxonomy"),
    ],
)
def test_dataset_cli_later_label_contract_failure_is_not_a_skip(
    tmp_path,
    monkeypatch,
    second_names,
    second_taxonomy,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = repo_root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_source": "ptbxl",
                "source_variant": "500hz",
                "sampling_rate": 500,
                "graph_leads": [0],
                "node_reduction": {"method": "none"},
                "build": {"jobs": 1, "chunk_size": 1},
                "visualization": {"smoke_during_build": False},
            }
        )
    )
    names = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

    def sample(exam_id: str, label_names: tuple[str, ...], taxonomy: str) -> ExamSample:
        return ExamSample(
            exam_id=exam_id,
            patient_id=f"p{exam_id}",
            strat_fold=-1,
            labels=np.zeros(len(label_names), dtype=np.uint8),
            label_names=label_names,
            label_taxonomy=taxonomy,
            signal=np.zeros((1, 16), dtype=np.float32),
            sampling_rate=500,
        )

    first = sample("1", names, "codetest_6class")
    second = sample("2", second_names, second_taxonomy)

    class LaterContractLoader:
        def __len__(self):
            return 2

        def __iter__(self):
            return iter((first, second))

        def iter_batches(self, batch_size):
            assert batch_size == 1
            yield [first]
            yield [second]

    processed_exam_ids: list[str] = []

    def fake_process_exam(sample, config):
        processed_exam_ids.append(sample.exam_id)
        return _make_success_result(sample)

    monkeypatch.setattr(build_cli, "get_loader", lambda *args, **kwargs: LaterContractLoader())
    monkeypatch.setattr(build_cli, "process_exam", fake_process_exam)

    assert build_main(["--config", str(config_path), "--repo-root", str(repo_root)]) == 2
    assert processed_exam_ids == ["1"]

    output_dir = repo_root / "graph_datasets" / "ptbxl"
    assert output_dir.exists()
    assert not list(output_dir.glob("*.h5"))
    assert not list(output_dir.glob(".*.tmp"))
    skipped_paths = list(repo_root.rglob("skipped.jsonl"))
    assert len(skipped_paths) == 1
    assert "exam_unexpected_error" not in skipped_paths[0].read_text()


def _make_unfittable_codetest(path: Path, n_exams: int = 3, n_samples: int = 4096) -> None:
    """Synthetic source that produces zero valid graphs (flat signal, no detectable R-peaks)."""
    sigs = np.full((n_exams, n_samples, 12), 0.0, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=sigs)
        f.create_dataset("exam_id", data=np.arange(n_exams, dtype=np.int64))
        labels = np.zeros((n_exams, 6), dtype=np.uint8)
        labels[:, 0] = 1
        f.create_dataset("labels", data=labels)
        f.attrs["label_names"] = np.array(_CODE_PHYSICAL_LABEL_NAMES, dtype=object)
        f.create_dataset("age", data=np.full(n_exams, 50, dtype=np.int32))
        f.create_dataset("is_male", data=np.ones(n_exams, dtype=np.int8))


def test_dataset_cli_no_valid_graphs_is_hard_error(tmp_path, monkeypatch):
    """Empty output is never a successful build; partial file must not be promoted."""
    datasets_root = tmp_path / "ecg_datasets"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_unfittable_codetest(datasets_root / "code15" / "codetest_raw.hdf5", n_exams=3)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, datasets_root / "code15" / "codetest_raw.hdf5")

    code = build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])
    assert code == 1

    output_dir = repo_root / "graph_datasets" / "codetest"
    if output_dir.exists():
        assert list(output_dir.glob("*.h5")) == []
    assert list(output_dir.glob("*.partial.h5")) == [] if output_dir.exists() else True


def test_dataset_cli_with_jobs_2(tmp_path, monkeypatch):
    """Parallel build path with --jobs 2 should also succeed."""
    datasets_root = tmp_path / "ecg_datasets"
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(datasets_root))
    _make_synthetic_codetest(datasets_root / "code15" / "codetest_raw.hdf5", n_exams=4)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = _make_smoke_config(repo_root, datasets_root / "code15" / "codetest_raw.hdf5")

    code = build_main(
        [
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--jobs",
            "2",
        ]
    )
    assert code == 0
    output_files = list((repo_root / "graph_datasets" / "codetest").glob("*.h5"))
    assert len(output_files) == 1
    qa_code = qa_main(["--h5", str(output_files[0]), "--profile", "full"])
    assert qa_code == 0


def test_build_writes_each_chunk_before_processing_all_exams(tmp_path, monkeypatch):
    """Default build path must not accumulate all processed graph results before writing."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_path = repo_root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_source": "codetest",
                "source_variant": "raw_400hz",
                "sampling_rate": 400,
                "graph_leads": [0],
                "node_reduction": {"method": "none"},
                "build": {"jobs": 1, "chunk_size": 2000},
                "visualization": {"smoke_during_build": False},
            }
        )
    )

    total_samples = 2001
    label_names = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")
    samples = [
        ExamSample(
            exam_id=f"{idx:04d}",
            patient_id=f"p{idx}",
            strat_fold=-1,
            labels=np.zeros(6, dtype=np.uint8),
            label_names=label_names,
            label_taxonomy="codetest_6class",
            signal=np.zeros((1, 16), dtype=np.float32),
            sampling_rate=400,
        )
        for idx in range(total_samples)
    ]

    class FakeLoader:
        def __iter__(self):
            return iter(samples)

        def __len__(self):
            return len(samples)

    processed_count = 0
    write_processed_counts: list[int] = []

    def fake_process_exam(sample, config):
        nonlocal processed_count
        processed_count += 1
        return _make_success_result(sample)

    class FakeWriter(_FakeWriterBase):
        def add_packed_exam_batch(self, exam_batches):
            for _metadata, packed in exam_batches:
                write_processed_counts.append(processed_count)
                self._tally(packed)

    monkeypatch.setattr(build_cli, "get_loader", lambda *args, **kwargs: FakeLoader())
    monkeypatch.setattr(build_cli, "process_exam", fake_process_exam)
    monkeypatch.setattr(build_cli, "DatasetWriter", FakeWriter)
    monkeypatch.setattr(build_cli, "validate_dataset", lambda *args, **kwargs: _PassingQA())

    code = build_main(["--config", str(config_path), "--repo-root", str(repo_root), "--jobs", "1"])

    assert code == 0
    assert write_processed_counts
    assert min(write_processed_counts) < total_samples
