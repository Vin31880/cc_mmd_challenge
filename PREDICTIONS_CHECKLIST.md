# Prediction files — status

All 18 released files were re-scored against the official gold-label files with
`score_and_analyse.py`; every value reproduces the thesis exactly.

## Released (18 files)

| Folder | Files | Reproduces |
|---|---|---|
| `submission/` | 11 configurations | Table 7.5, Figure 7.2, §7.5.1, §7.7 |
| `zeroshot/` | `malayalam_original.csv` | Table 7.2 row 1 (Malayalam), Table 7.3, Table 7.4 |
| `demos/` | `malayalam_original.csv` | Table 7.2 row 2 (Malayalam), Table 7.3, Table 7.4 |
| `malayalam_exemplars/` | `malayalam_original.csv` | Table 7.3, Table 7.4 |
| `content_markers/` | `mami_indian.csv`, `mami_chinese.csv` | Table 7.2 row 3, Table 7.4 |
| `perception_markers/` | `mami_indian.csv`, `mami_chinese.csv` | Table 7.2 row 4, Table 7.4 |

### Verified macro-F1

`submission/` — tamil original 0.596, irish 0.599, chinese 0.571; malayalam original 0.734,
irish 0.746, chinese 0.759; cmmd original 0.750, indian 0.803, irish 0.747; mami indian
0.606, chinese 0.727.

Malayalam exemplar conditions (Table 7.3) — zero-shot 0.817, Malayalam exemplars 0.779,
Tamil exemplars 0.607.

Markers on the Western corpus (Table 7.2) — content 0.684 (Indian) / 0.482 (Chinese);
perception 0.675 (Indian) / 0.674 (Chinese).

### Verified predicted-positive rates (Table 7.4)

mami indian: content 37.9%, perception 22.9% (gold 38.3%).
mami chinese: content 12.2%, perception 79.8% (gold 59.2%).
malayalam: zero-shot 48.0%, Malayalam exemplars 29.0%, Tamil exemplars 15.5% (gold 40.0%).

## Not retained

Prediction files were not kept for the zero-shot and lens-matched demonstration runs on
Tamil, on the Chinese corpus, and on the Indian lens of the Western corpus, nor for the
LLaVA-1.5-7B and Gemma3-4b rows of Table 7.1. Those cells are recorded from run logs and
cannot be recomputed from this release. This is stated in §6.4 of the thesis.

Affected cells: Table 7.2 zero-shot Tamil (0.674) and Chinese (0.735); Table 7.2
demonstrations Tamil (0.695), Chinese (0.800), MAMI-Indian (0.646); all of Table 7.1 except
the Qwen2.5-VL-7B figures that coincide with the above.

## Reproducing

```bash
python score_and_analyse.py --gold-dir gold --pred-dir predictions/<folder> --out results.json
```

Place the official `*_test_labelled.csv` files in `gold/` first; the corpora are not
redistributed here (see §6.5 of the thesis and `gold/README.md`).
