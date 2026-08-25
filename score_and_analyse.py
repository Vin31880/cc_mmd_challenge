#!/usr/bin/env python3
"""
CC-MMD 2026 — scoring and analysis for all figures reported in the thesis.

Reproduces, from prediction files + official gold labels:
  * macro-F1 / accuracy for every configuration          (Tables 7.1, 7.2, 7.6)
  * per-class precision / recall / F1                    (Table 7.7)
  * predicted-vs-gold positive rates                     (Figure 7.2)
  * confusion matrices
  * cross-lens error attribution                         (Section 7.5.1)
  * inter-annotator agreement between lenses             (Sections 7.3, 7.7)

Usage:
    python score_and_analyse.py --gold-dir GOLD --pred-dir PRED

Gold files expected in GOLD (as released by the CC-MMD organisers):
    MAMI_test_labelled.csv           image_id, transcriptions, indian_labels, chinese_labels
    MDMD_Tamil_test_labelled.csv     image_id, transcriptions, original_labels, irish_labels, chinese_labels
    MDMD_Malaylam_test_labelled.csv  (same columns as Tamil)
    CMMD_test_labelled.csv           image_id, transcriptions, original_labels, indian_labels, irish_labels

Prediction files: CSV with columns image_id,label  (label in {0,1}), or the
combined Task B format with one column per culture.
"""
import argparse, csv, itertools, json, os, sys
from collections import Counter

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

POSITIVE = {"1", "misogyny", "misogynistic"}


def norm_id(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def to_int(v):
    s = str(v).strip().lower()
    if s in POSITIVE:
        return 1
    if s in {"0", "not-misogyny", "not misogyny", "not-misogynistic"}:
        return 0
    raise ValueError(f"unrecognised label {v!r}")


def load(path, id_col="image_id"):
    with open(path, encoding="utf-8-sig") as f:
        return {norm_id(r[id_col]): r for r in csv.DictReader(f)}


def align(gold, gold_col, pred, pred_col="label"):
    """Return (y_true, y_pred) over ids present in both, sorted numerically."""
    common = sorted(set(gold) & set(pred), key=lambda x: int(x))
    y_true, y_pred = [], []
    for k in common:
        if gold[k].get(gold_col) in (None, ""):
            continue
        y_true.append(to_int(gold[k][gold_col]))
        y_pred.append(int(str(pred[k][pred_col]).strip()))
    return y_true, y_pred


def report(name, y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                 zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    gold_pos, pred_pos = sum(y_true) / len(y_true), sum(y_pred) / len(y_pred)
    return {
        "configuration": name,
        "n": len(y_true),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "neg": {"precision": round(p[0], 3), "recall": round(r[0], 3), "f1": round(f[0], 3)},
        "pos": {"precision": round(p[1], 3), "recall": round(r[1], 3), "f1": round(f[1], 3)},
        "gold_positive_rate": round(gold_pos, 3),
        "pred_positive_rate": round(pred_pos, 3),
        "calibration_ratio": round(pred_pos / gold_pos, 2) if gold_pos else None,
        "confusion": {"tn": int(cm[0][0]), "fp": int(cm[0][1]),
                      "fn": int(cm[1][0]), "tp": int(cm[1][1])},
    }


def agreement(gold, cols):
    """Inter-annotator agreement between every pair of lenses (no model involved)."""
    out = {}
    for a, b in itertools.combinations(cols, 2):
        ids = [k for k in gold if gold[k].get(a) and gold[k].get(b)]
        out[f"{a} vs {b}"] = round(
            sum(to_int(gold[k][a]) == to_int(gold[k][b]) for k in ids) / len(ids), 3)
    return out


def cross_lens_attribution(gold, target, other, pred):
    """How many 'errors' under `target` agree with the `other` population?
    Reproduces the Section 7.5.1 figures."""
    yt, yp = align(gold, target, pred)
    ids = sorted(set(gold) & set(pred), key=lambda x: int(x))
    fp = [k for k in ids if to_int(gold[k][target]) == 0 and int(pred[k]["label"]) == 1]
    fn = [k for k in ids if to_int(gold[k][target]) == 1 and int(pred[k]["label"]) == 0]
    fp_other = sum(1 for k in fp if to_int(gold[k][other]) == 1)
    fn_other = sum(1 for k in fn if to_int(gold[k][other]) == 0)
    total, recovered = len(fp) + len(fn), fp_other + fn_other
    yt_other = [to_int(gold[k][other]) for k in ids]
    yp_all = [int(pred[k]["label"]) for k in ids]
    return {
        "false_positives": len(fp),
        "fp_agreeing_with_other_population": fp_other,
        "false_negatives": len(fn),
        "fn_agreeing_with_other_population": fn_other,
        "errors_recovered_under_other_lens": f"{recovered}/{total}"
                                             f" ({recovered / total:.1%})",
        "macro_f1_vs_target_labels": round(f1_score(yt, yp, average="macro"), 3),
        "macro_f1_vs_other_labels": round(f1_score(yt_other, yp_all, average="macro"), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    G = {
        "tamil":     ("MDMD_Tamil_test_labelled.csv",
                      ["original_labels", "irish_labels", "chinese_labels"]),
        "malayalam": ("MDMD_Malaylam_test_labelled.csv",
                      ["original_labels", "irish_labels", "chinese_labels"]),
        "cmmd":      ("CMMD_test_labelled.csv",
                      ["original_labels", "indian_labels", "irish_labels"]),
        "mami":      ("MAMI_test_labelled.csv",
                      ["indian_labels", "chinese_labels"]),
    }

    results, agreements = [], {}
    for corpus, (fname, cols) in G.items():
        gpath = os.path.join(a.gold_dir, fname)
        if not os.path.exists(gpath):
            print(f"[skip] missing gold file {gpath}", file=sys.stderr)
            continue
        gold = load(gpath)
        agreements[corpus] = agreement(gold, cols)
        for col in cols:
            lens = col.replace("_labels", "")
            ppath = os.path.join(a.pred_dir, f"{corpus}_{lens}.csv")
            if not os.path.exists(ppath):
                print(f"[skip] no predictions for {corpus}/{lens}", file=sys.stderr)
                continue
            yt, yp = align(gold, col, load(ppath))
            results.append(report(f"{corpus}_{lens}", yt, yp))

    extra = {}
    mami_gold = os.path.join(a.gold_dir, G["mami"][0])
    mami_pred = os.path.join(a.pred_dir, "mami_indian.csv")
    if os.path.exists(mami_gold) and os.path.exists(mami_pred):
        extra["cross_lens_attribution_mami_indian"] = cross_lens_attribution(
            load(mami_gold), "indian_labels", "chinese_labels", load(mami_pred))

    payload = {"configurations": results,
               "inter_annotator_agreement": agreements,
               **extra}
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)

    hdr = f"{'configuration':<24}{'n':>6}{'macroF1':>9}{'acc':>8}{'gold+':>8}{'pred+':>8}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['configuration']:<24}{r['n']:>6}{r['macro_f1']:>9.3f}"
              f"{r['accuracy']:>8.3f}{r['gold_positive_rate']:>8.3f}"
              f"{r['pred_positive_rate']:>8.3f}")
    print(f"\nfull results -> {a.out}")


if __name__ == "__main__":
    main()
