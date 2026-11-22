# CODE v2 patient-level splits (90/5/5, seed 2026)

`splits.csv.gz` assigns every exam of the v2 CODE index to train, val or test. It is the split behind every v2 model.


## Columns

| Column | Type | Meaning |
|---|---|---|
| `exam_id` | int64 | CODE `id_exam` |
| `split` | str | `train`, `val` or `test` |
| `in_code15` | bool | the exam belongs to a patient who has at least one exam in CODE-15% (Zenodo 4916206, `exams.csv`, 345,779 exams / 233,770 patients) |

The file has no patient IDs, labels, signals or file paths.

## How the split was made

- **Universe:** every exam in the CODE training H5 files that also appears in `annotations_nosamitrop.csv` (SaMi-Trop excluded).
- **Removed:**
  - 7,601 exams not in the annotations;
  - 4,280 all-zero recordings.
- **Result:** 2,310,514 usable exams.
- **Shuffle:** unique patients were shuffled with `numpy.random.default_rng(2026)`. The first 90% of patients went to train, the next 5% to val, the last 5% to test.
- **Patient level:** all exams of a patient are in one split.
- **No subsampling:** val and test keep natural prevalence.

| Split | ECGs | Patients | ECGs with `in_code15` | ECGs with `in_code15 == False` |
|---|---:|---:|---:|---:|
| train | 2,079,099 | 1,399,614 | 309,638 | 1,769,461 |
| val | 115,852 | 77,756 | 17,467 | 98,385 |
| test | 115,563 | 77,757 | 17,134 | **98,429** |
| total | 2,310,514 | 1,555,127 | 344,239 | 1,966,275 |

**Checks run on export:**
- no duplicate `exam_id`;
- every usable exam appears exactly once;
- 0 patients shared between any two splits;
- the file was re-read and its counts matched.

`test` with `in_code15 == False` (98,429 ECGs / 66,120 patients) is our **test_minus_code15** set. It is identical, exam for exam, to the set used for our CODE-15%-pretrained models.

## File

- `splits.csv.gz`: 6,475,647 bytes
- sha256 `64de90e3fbccbb5b23a4f358bdb4204aacdeb6ab4fd3cb523a3e6e1f9058d8bd`

```python
import pandas as pd
s = pd.read_csv("splits.csv.gz")          # exam_id, split, in_code15
train_ids = s.exam_id[s.split == "train"]
test_minus_code15 = s.exam_id[(s.split == "test") & ~s.in_code15]
```

## Protocol to match our numbers

1. Train on `train`. Select the checkpoint on `val`.
2. On `val`, choose a threshold for each class that maximises F1.
3. Freeze the thresholds and score `test` once.
4. Report per-class F1 for 1dAVb, RBBB, LBBB, SB, AF and ST, plus the macro (unweighted mean of the six).
5. Models pretrained on CODE-15% (for example ST-MEM and ECG-JEPA) are scored only on test rows with `in_code15 == False` (test_minus_code15). Other models are scored on the full test split.

CODE-Test (827 ECGs, cardiologist gold standard) is a separate cohort and not part of this split.
