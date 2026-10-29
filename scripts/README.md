# Workflows

`scripts/` holds the command-line workflows the paper uses. Shared logic lives in `src/`. Run a Python workflow from the repository root:

```bash
uv run python scripts/<area>/<name>.py --help
```

The `--help` text of each workflow lists its inputs. The README maps each paper result to its configs and workflows.

| Workflow | Role |
|---|---|
| `dataset/pack_codetest.py` | Pack the public CODE-Test release into the HDF5 the loaders read. |
| `evaluation/eval_ptbxl_code6.sh` | Score checkpoints on PTB-XL Strict-SCP at the 0.5 threshold. |
| `evaluation/efficiency_eval.py` | Count parameters and measure graph reduction, latency, memory, and FLOPs. |
| `evaluation/export_graph_prediction_cohort.py` | Export aligned LYNX-ECG predictions with checkpoint provenance. |
| `evaluation/paired_bootstrap_models.py` | Compare models on the same exams with an exam bootstrap. |
| `explain/clinical_alignment_gate.py` | Integrated Gradients lift in the expected and comparison regions. |
| `explain/prominence_availability_shapley.py` | Eight-state PR-QRS-T deletion game over every exam, under both availability rules. |
| `explain/prominence_availability_analysis.py` | Aggregate the three seeds into the regional verdicts. |
| `explain/prominence_support_factorial.py` | Compare delineated and fixed-window regions on the same beats. |
| `explain/rhythm_lead_audit.py` | Beat-sequence, rate, and lead interventions and exact 12-lead Shapley values. |
| `explain/rhythm_lead_summary.py` | Aggregate the three seeds of the rhythm and lead audit. |
| `training/benchmark_training.py` | Measure loader, step, end-to-end, and evaluation throughput. |

Generated reports stay out of git. Report readers check the provenance their producer records, so pass the paths each command names.
