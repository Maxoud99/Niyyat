"""
Evaluate intent attribution against ground-truth mask.

Ground truth:
  - mask=0  → no error       → true label = 0
  - mask=1  → error present, ALL errors are INTENTIONAL → true label = 1

LLM predictions (intent_labels.csv):
  -  1 → intentional error detected  (CORRECT on error cells)
  - -1 → unintentional error detected (WRONG on error cells)
  -  0 → no error detected            (MISSED on error cells)

Metrics computed on ERROR CELLS ONLY (723 cells, all true label = 1):
  Intentional class (1):
    Precision = TP / (TP + FP)  where FP = cells predicted 1 but truly 0
                                 (here we evaluate on error-cells only so FP=0 → P=1.0)
    Recall    = TP / (TP + FN)  = correct / 723
    F1        = 2*P*R / (P+R)

  Unintentional class (-1):
    All predicted -1 are false positives (true label is 1, not -1).
    Precision = 0 / predicted_as_-1  = 0.0
    Recall    = 0 / 0  (no true -1 exists) → defined as 0.0
    F1        = 0.0

  Accuracy  = correct_intentional / total_error_cells
"""
import pandas as pd
import numpy as np
import os

MASK_PATH   = "/home/mohamed/error_injector/llms_baseline/tabfact/outputs/error_detection/run_20251222_163837/error_mask_updated.csv"
BASE_OUT    = "/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based/LLMs-fact/outputs"
RESULTS_DIR = "/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based/LLMs-fact/evaluation_results"
VARIANTS    = [
    "bareminimum-gemini", "info-gemini", "few-shots-gemini",
    "baremin-mixtral", "info-mixtral", "few-shots-mixtral",
    "baremin-llama", "info-llama", "few-shots-llama",
    "baremin-qwen", "info-qwen", "few-shots-qwen",
    "baremin-r1-qwen", "info-r1-qwen", "few-shots-r1-qwen",
]
ATTR_COLS   = ["subject_entity", "subject_type", "subject_location", "value", "claim_domain", "metric"]

os.makedirs(RESULTS_DIR, exist_ok=True)

mask_df   = pd.read_csv(MASK_PATH)
mask_flat = mask_df[ATTR_COLS].values.flatten().astype(int)
error_idx = np.where(mask_flat == 1)[0]
N         = len(error_idx)   # 723

# ── Compute metrics ───────────────────────────────────────────────────────────
rows = []
per_col_rows = []

for variant in VARIANTS:
    pred_flat = pd.read_csv(f"{BASE_OUT}/{variant}/intent_labels.csv")[ATTR_COLS].values.flatten().astype(int)
    preds = pred_flat[error_idx]   # predictions on true-error cells only

    tp_int  = int((preds ==  1).sum())   # correctly called intentional
    fp_int  = 0                          # no true-0 cells here (error cells only)
    fn_int  = int((preds == -1).sum()) + int((preds == 0).sum())  # wrong or missed
    fp_unint= int((preds == -1).sum())   # wrongly called unintentional
    missed  = int((preds ==  0).sum())

    # Intentional metrics
    p_int = tp_int / (tp_int + fp_int) if (tp_int + fp_int) > 0 else 0.0
    r_int = tp_int / (tp_int + fn_int) if (tp_int + fn_int) > 0 else 0.0
    f_int = (2 * p_int * r_int / (p_int + r_int)) if (p_int + r_int) > 0 else 0.0

    # Unintentional metrics (no true unintentional exists → P=R=F1=0)
    p_unint = 0.0
    r_unint = 0.0
    f_unint = 0.0

    acc = tp_int / N

    rows.append({
        "variant":            variant,
        "total_error_cells":  N,
        "correct_intentional": tp_int,
        "wrong_unintentional": fp_unint,
        "missed":             missed,
        "accuracy":           round(acc,    4),
        "intentional_precision": round(p_int,  4),
        "intentional_recall":    round(r_int,  4),
        "intentional_f1":        round(f_int,  4),
        "unintentional_precision": round(p_unint, 4),
        "unintentional_recall":    round(r_unint, 4),
        "unintentional_f1":        round(f_unint, 4),
    })

    # Per-column
    pred_df = pd.read_csv(f"{BASE_OUT}/{variant}/intent_labels.csv")
    for col in ATTR_COLS:
        m = mask_df[col].values.astype(int)
        p = pred_df[col].values.astype(int)
        err = m == 1
        n_err = err.sum()
        if n_err == 0:
            per_col_rows.append({"variant": variant, "column": col,
                                  "n_errors": 0, "accuracy": None,
                                  "intentional_precision": None, "intentional_recall": None, "intentional_f1": None})
            continue
        tp  = int(((p ==  1) & err).sum())
        fp_u= int(((p == -1) & err).sum())
        ms  = int(((p ==  0) & err).sum())
        fn  = fp_u + ms
        pr  = 1.0
        rc  = tp / n_err
        f1c = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        per_col_rows.append({
            "variant": variant, "column": col, "n_errors": int(n_err),
            "correct": tp, "wrong_unintentional": fp_u, "missed": ms,
            "accuracy":              round(tp / n_err, 4),
            "intentional_precision": round(pr,  4),
            "intentional_recall":    round(rc,  4),
            "intentional_f1":        round(f1c, 4),
        })

# ── Save to CSV ───────────────────────────────────────────────────────────────
overall_df  = pd.DataFrame(rows)
per_col_df  = pd.DataFrame(per_col_rows)

overall_path  = os.path.join(RESULTS_DIR, "overall_metrics.csv")
per_col_path  = os.path.join(RESULTS_DIR, "per_column_metrics.csv")
overall_df.to_csv(overall_path,  index=False)
per_col_df.to_csv(per_col_path,  index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"Total error cells (all intentional): {N}\n")
print(f"{'Variant':<22}  {'Acc':>6}  {'INT-P':>6}  {'INT-R':>6}  {'INT-F1':>7}  {'UNINT-P':>8}  {'UNINT-R':>8}  {'UNINT-F1':>9}  Correct  Wrong  Missed")
print("─" * 105)
for r in rows:
    print(f"{r['variant']:<22}  {r['accuracy']:>6.4f}  "
          f"{r['intentional_precision']:>6.4f}  {r['intentional_recall']:>6.4f}  {r['intentional_f1']:>7.4f}  "
          f"{r['unintentional_precision']:>8.4f}  {r['unintentional_recall']:>8.4f}  {r['unintentional_f1']:>9.4f}  "
          f"{r['correct_intentional']:>7}  {r['wrong_unintentional']:>5}  {r['missed']:>6}")

print(f"\nResults saved to:\n  {overall_path}\n  {per_col_path}")
