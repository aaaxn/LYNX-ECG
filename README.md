# LYNX-ECG

Code for *Accurate and Auditable ECG Classification via Visibility Graph Neural Networks* (Xavier et al., under review).

![LYNX-ECG pipeline. (a) Each beat of a 12-lead ECG keeps its turning points, which form a natural visibility graph. (b) A GINE encoder embeds each beat graph, a GATv2 encoder and attention pooling combine the beats of each lead, and attention over the 12 leads feeds an MLP head with six outputs.](assets/lynx_ecg_pipeline.png)

LYNX-ECG (Lead-level visibilitY Networks for eXplainable ECG) classifies a 12-lead ECG into six labels: 1dAVb, RBBB, LBBB, SB, AF, and ST. It turns each heartbeat into a sparse visibility graph of timestamped turning points and reads the graphs with a hierarchical graph neural network. On CODE-Test the model reaches a macro F1 score of 0.889 ± 0.004 over three seeds, higher than five retrained baselines including ECGFounder. On the external PTB-XL cohort it reaches 0.730 ± 0.001. Its CODE100 graphs need 164.6 GB of HDF5 storage, which is 14% of the 1,175.6 GB that graphs of every sample need. Every node keeps its time stamp, so the paper deletes waveform regions, changes beat timing, and removes leads, then scores the exam again. For the three conduction labels, the clinically expected region has the largest Shapley contribution in 220 of 238 correct deletion verdicts (92.4%).

This repository contains the dataset pipeline, the model, the training and evaluation commands, and the explanation analyses that the paper reports. It contains no ECG data and no trained weights. The paper retrains each baseline from its own public code, so the baselines are not part of this repository.

## Method

**Graph construction.** A fourth-order zero-phase Butterworth filter keeps 0.5 to 40 Hz. The pipeline resamples the signal to 300 Hz when needed and divides it by the median across leads of the 95th-percentile amplitude. A polarity-aware NeuroKit2 detector finds R peaks in each lead, and beats end at the midpoints between consecutive R peaks. In each beat, the nodes are the local extrema of a 40 Hz low-passed copy, the R peak, and the two beat endpoints. The gap rule inserts evenly spaced support nodes until no two consecutive nodes are more than 80 ms apart. Natural visibility joins two nodes when every node between them lies below the line that joins them. Edges point forward in time. A node stores its amplitude and its time within the beat. An edge stores the time difference and the amplitude difference.

**Model.** A three-layer GINE encoder with hidden width 128 maps each beat graph to one embedding. A two-layer, four-head GATv2 encoder exchanges information between adjacent beats of the same lead, and gated attention pools the beats into one embedding per lead. Softmax attention over the available leads gives the exam embedding. The graph has no edges between leads, so lead embeddings combine only in this last pooling step. A two-layer MLP head returns six logits. The model has 327,371 trainable parameters.

**Training.** Each epoch uses every exam that is positive for at least one label and 220,514 negative exams, drawn without replacement and redrawn every epoch. The loss is binary cross-entropy. A 2% patient-hash split of CODE100 selects the checkpoint with the best validation macro F1. Every reported F1, precision, and recall uses the fixed threshold 0.5.

**Explanation audits.** Integrated Gradients attributes a class logit to node features and reports the lift of each region. Structural deletion removes the nodes of the delineated PR, QRS, and T regions and links the last remaining node before each deleted region to the first remaining node after it, so each beat graph stays connected. Exact Shapley values split the score drop among the regions. Timing interventions shuffle, equalize, or rescale the beats of an exam. An exact decomposition over the 12 leads gives the Shapley value of each lead.

## Repository layout

| Path | Contents |
|---|---|
| `src/dataset/` | Preprocessing, R peaks, beats, node reduction, visibility graphs, HDF5 writer, and quality checks |
| `src/models/` | GINE beat encoder, GATv2 beat-sequence encoder, pooling, and classification head |
| `src/training/` | Graph loading, splits, dynamic sampling, training loop, evaluation, and metrics |
| `src/explain/` | Integrated Gradients, delineation, region deletion, and exact Shapley games |
| `src/config/` | Pydantic schemas for dataset and training configs |
| `src/cli/` | Console commands `dataset`, `dataset-qa`, `train`, `evaluate`, and `delineation-qc` |
| `src/ecg_artifact_io/`, `src/ecg_checkpoint_io/`, `src/ecg_paired_predictions/` | Readers and writers for artifacts, checkpoints, and paired prediction files |
| `configs/dataset/`, `configs/train/`, `configs/explain/` | Dataset, training, and explanation configs of the paper, and the smoke configs |
| `scripts/` | Evaluation, explanation, and benchmark workflows. See [scripts/README.md](scripts/README.md). |
| `tests/` | Unit and integration tests, and opt-in smoke tests |
| `assets/` | The pipeline figure of this README |

## Installation

Install [uv](https://docs.astral.sh/uv/) and run this command from the repository root.

```bash
uv sync --python 3.12
```

The project needs Python 3.12. On Linux x86_64 the lock file pins PyTorch 2.11 with CUDA 13.0 wheels, and PyTorch Geometric with matching `pyg-lib`, `torch-scatter`, and `torch-sparse` wheels from data.pyg.org. On macOS it pins the PyTorch 2.11 wheel. The project declares no PyTorch dependency for other platforms. Full-size training needs a CUDA GPU. The quick check below runs on a CPU.

## Data

The repository holds no ECG data. Set `ECG_DATASETS_ROOT` to the directory that holds the raw datasets.

```bash
export ECG_DATASETS_ROOT=/path/to/ecg_datasets
```

| Dataset | Role | Access | Source | Path under `$ECG_DATASETS_ROOT` |
|---|---|---|---|---|
| CODE (CODE100) | Training and validation, 2,237,125 and 46,378 exams after quality control | Restricted | On request, see below | `code100/` with `annotations.csv` and the `S<XXX>0000/` WFDB folders |
| CODE-Test | Test, 827 exams | Open | [Zenodo 10.5281/zenodo.3625006](https://doi.org/10.5281/zenodo.3625006) | `code15/codetest_raw.hdf5`, made with `scripts/dataset/pack_codetest.py` |
| PTB-XL 1.0.3 | External test, 21,799 exams | Open | [PhysioNet 10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45) | `ptbxl/physionet.org/files/ptb-xl/1.0.3/` |

The Telehealth Network of Minas Gerais recorded the CODE exams. Researchers at educational or research institutions request access from the Network. The request gives the name of the principal investigator, the host organization, contact details, and the scientific purpose. The Network reviews each request, restricts use to non-commercial research, and grants access after the requester signs a data use agreement. This repository does not redistribute CODE waveforms.

PTB-XL labels follow the exact SCP-ECG codes 1AVB, CRBBB, CLBBB, SBRAD, AFIB, and STACH. The CODE100 loader keeps the records stored at 300 Hz and skips the records at other rates.

Pack CODE-Test after you download and extract its release. The release directory holds `ecg_tracings.hdf5`, `attributes.csv`, and `annotations/gold_standard.csv`.

```bash
uv run python scripts/dataset/pack_codetest.py \
    --release-dir /path/to/codetest/data \
    --output "$ECG_DATASETS_ROOT/code15/codetest_raw.hdf5"
```

## Quick check

This check builds 200 PTB-XL exams with the paper preprocessing and trains the paper architecture for two epochs on a CPU. It needs PTB-XL under `ECG_DATASETS_ROOT`. On a 64-core workstation it takes about 20 seconds.

```bash
H5=graph_datasets/ptbxl/ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__bp0p5-40hz-o4__resample-300hz__norm-p95med__natural__leads-0-1-2-3-4-5-6-7-8-9-10-11__limit-200.h5
uv run dataset --config configs/dataset/smoke/ptbxl_tp_smoke_200.json
uv run dataset-qa --h5 "$H5"
uv run train --config configs/train/smoke/ptbxl_tp_smoke.json --device cpu
uv run evaluate --ckpt results/ptbxl_tp_smoke_s55/best_f1_val.pt --dataset "$H5" --device cpu
```

The build writes the HDF5 file to `graph_datasets/`. Training writes `best_f1_val.pt`, `history.json`, and `test_metrics.json` to `results/ptbxl_tp_smoke_s55/`, together with other run files. The smoke run trains on 200 exams, so its metrics only show that the pipeline runs.

Run the test suites with these commands.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -m smoke
```

The smoke tests need PTB-XL under `ECG_DATASETS_ROOT`. One test also needs the HDF5 file from the quick check.

## Training and evaluation

The `--help` text of each workflow lists its inputs. Build the datasets first, because the training presets read them from `graph_datasets/`. Each result covers the seeds 55, 1111, and 55555. Run each training config once per seed with `--seed`. Each run writes to `results/<run_name>_s<seed>/`. For example, `uv run train --config configs/train/code100_tp.json --seed 1111` writes to `results/code100_tp_s1111/`.

| Paper result | Configs | Workflow |
|---|---|---|
| Table 1, Extended Data Table 1 | `configs/train/code100_tp.json` | `train`, `eval_ptbxl_code6.sh` |
| Table 2 (node selection) | `configs/train/code100_{uniform_k30,all_points}.json` | `train` |
| Extended Data Table 2 (topology, edge attributes) | `configs/train/code100_tp_topology2x2_*.json` | `train` |
| Supplementary Note 6 (message direction) | `configs/train/code100_tp_symmetrized.json` | `export_graph_prediction_cohort.py`, `paired_bootstrap_models.py` |
| Fig. 3a, Extended Data Table 3 | Main-model checkpoints | `clinical_alignment_gate.py` |
| Fig. 3b | `configs/explain/codetest_region_delineation_qc.json` | `delineation-qc`, `prominence_availability_shapley.py`, `prominence_availability_analysis.py` |
| Extended Data Table 4 | `configs/explain/codetest_region_delineation_qc.json` | `prominence_support_factorial.py` |
| Fig. 4, Supplementary Tables 10 and 12 to 14 | `configs/explain/codetest_region_delineation_qc.json` | `rhythm_lead_audit.py`, `rhythm_lead_summary.py` |
| Supplementary Note 7 (efficiency) | `configs/train/code100_tp.json` | `efficiency_eval.py`, `benchmark_training.py` |

A retrained model gives close but not identical numbers, because GPU training is not bit-reproducible.

## Citation

```bibtex
@unpublished{xavier2026lynxecg,
  title  = {Accurate and Auditable ECG Classification via Visibility Graph Neural Networks},
  author = {Xavier, Artur and Evangelista, Guilherme Henrique Gomes and Khan, Aitazaz and Ribeiro, Antonio Luiz Pinho and Zaki, Mohammed J. and Parthasarathy, Srinivasan and Meira Jr., Wagner},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

The code uses the [MIT License](LICENSE). The ECG datasets keep their own licenses and access terms, which [Data](#data) describes.
