# Cultural Lens Conditioning for Cross-Cultural Misogynistic Meme Detection

Code and model predictions accompanying the MSc thesis of the same name
(Vinayak Madhukar Jaybhaye, School of Computer Science, University of Galway, 2026).

## What is and is not here

This repository contains **code and model predictions only**. The CC-MMD 2026 corpora
(MDMD, CMMD, and the MAMI partition) are **not redistributed**: MAMI is governed by the
SemEval-2022 Task 5 data-use agreement, and MDMD and CMMD were supplied to registered
participants by the CC-MMD 2026 organisers. Obtain them from those sources.

Prediction files are included for the runs listed under **Coverage** below, so those
numbers can be recomputed from this repository plus the official gold-label files, without
access to the meme images. Six cells in the thesis are **not** covered — see Coverage and
`PREDICTIONS_CHECKLIST.md`.

## Layout

```
notebooks/        inference notebooks, one per pipeline version (see mapping below)
predictions/      model outputs, one CSV per configuration: image_id,label
gold/             official *_test_labelled.csv files
score_and_analyse.py   all scoring and analysis
example_results.json   sample output, for comparison
requirements.txt
```

## Reproducing the reported numbers

```bash
pip install -r requirements.txt
python score_and_analyse.py --gold-dir gold --pred-dir predictions/submission --out results.json
```

Run once per prediction folder (`submission`, `zeroshot`, `demos`, `malayalam_exemplars`,
`content_markers`, `perception_markers`). The script prints macro-F1, accuracy, and
gold/predicted positive rates per configuration, and writes per-class precision/recall/F1,
confusion matrices, inter-annotator agreement, and the cross-lens error attribution to the
`--out` file. `example_results.json` shows the expected shape.

Configurations with no prediction file in the chosen folder are reported as `[skip]` and
omitted; this is expected, since most folders hold a single condition.

Prediction files are named `{corpus}_{lens}.csv`, where corpus is one of
`tamil`, `malayalam`, `cmmd`, `mami`, and lens is one of `original`, `irish`, `chinese`,
`indian`.

## Which notebook produced which result

Results in the thesis come from successive versions of the pipeline. This mapping is
required to reproduce any individual row.

| Thesis location | Condition | Notebook |
|---|---|---|
| Table 7.1 | zero-shot, model comparison | `notebooks/Zero_Shot_Model_comparison.ipynb` |
| Table 7.2, row 2 | + lens-matched demonstrations, scene-graph CoT | `notebooks/demo_scene_graph_Cot.ipynb` |
| Table 7.2, row 3 | + content markers | `notebooks/content_markers.ipynb` |
| Table 7.2, row 4 | + perception markers (full method) | `notebooks/perception_marker.ipynb` |
| Table 7.3 | Malayalam, three exemplar conditions | `notebooks/malayalam_three_exampler.ipynb` |
| Table 7.5, Figure 7.2, §7.5.1, §7.7 | official shared-task submission | `notebooks/official_submission.ipynb` |

The submitted system is an **earlier pipeline** than any row of Table 7.2: it predates both
the cultural markers and the prompt refinements of Chapter 5. Its re-scored figures differ
from the "+ lens-matched demonstrations" row it most closely resembles (Section 7.6 of the
thesis).

## Coverage

Released, and verified to reproduce the thesis values exactly:

| Folder | Files | Reproduces |
|---|---|---|
| `submission/` | 11 configurations | Table 7.5, Figure 7.2, §7.5.1, §7.7 |
| `zeroshot/` | `malayalam_original.csv` | Table 7.2 row 1 (Malayalam), Table 7.3, Table 7.4 |
| `demos/` | `malayalam_original.csv` | Table 7.2 row 2 (Malayalam), Table 7.3, Table 7.4 |
| `malayalam_exemplars/` | `malayalam_original.csv` | Table 7.3, Table 7.4 |
| `content_markers/` | `mami_indian.csv`, `mami_chinese.csv` | Table 7.2 row 3, Table 7.4 |
| `perception_markers/` | `mami_indian.csv`, `mami_chinese.csv` | Table 7.2 row 4, Table 7.4 |

**Not retained.** Prediction files were not kept for the zero-shot and lens-matched
demonstration runs on Tamil, on the Chinese corpus, and on the Indian lens of the Western
corpus, nor for the LLaVA-1.5-7B and Gemma3-4b rows of Table 7.1. Those cells are recorded
from run logs and cannot be recomputed from this release. This is disclosed in Section 6.4
of the thesis. Affected: Table 7.2 zero-shot Tamil (0.674) and Chinese (0.735); Table 7.2
demonstrations Tamil (0.695), Chinese (0.800) and MAMI-Indian (0.646); all of Table 7.1
except the Qwen2.5-VL-7B figures that coincide with the above.

## Item counts

Three different Tamil test-set sizes appear across the runs, for reasons disclosed in
Section 6.4 of the thesis:

- **356** — the full official Tamil test set.
- **355** — the archived submission files in `predictions/submission/`, whose exemplar pool
  excluded one test item by identifier collision. The per-class counts in §7.5.1 and §7.7
  are over this set.
- **353** — the runs behind Table 7.1 and Table 7.2, which predate the fix described below
  and excluded three items.

Malayalam is 200 throughout in this release; the pre-fix runs used 198. CMMD (340) and MAMI
(1,000) are unaffected.

## Known issues in earlier runs

Two faults were found during the work and are disclosed in the thesis (Sections 6.4 and
7.6). Both are fixed in `notebooks/malayalam_three_exampler.ipynb`, the most recent
notebook.

1. **Exemplar exclusion matched bare filename IDs rather than full paths.** Because item
   IDs are not unique across corpora or splits, this dropped legitimate test items that
   shared an integer with a development-split exemplar: 3 from Tamil (356 → 353) and 2 from
   Malayalam (200 → 198). CMMD and MAMI were unaffected. Results produced before the fix are
   reported over the reduced item counts and are labelled as such in the thesis.

2. **`batch_classify` reverse-derived the partition name from `(label_col, country)`.**
   Three partitions share that pair, so the lookup returned the first match and every
   Malayalam run silently used the Tamil exemplar pool regardless of configuration. Fixed by
   passing the partition name explicitly.

## Environment

Qwen2.5-VL-7B-Instruct via `transformers`, greedy decoding, 512-token generation budget.
Run on NVIDIA A100 (40GB) and T4; 4-bit NF4 quantisation where GPU memory required it,
bfloat16 otherwise. See the notebooks for exact settings. Scoring requires only
`scikit-learn` (`requirements.txt`); the script reads CSVs with the standard library.

## Citation

[[Add thesis citation once submitted.]]
