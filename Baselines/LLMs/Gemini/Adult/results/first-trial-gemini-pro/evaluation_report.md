# Evaluation Report (Intent vs Unintentional Only)

**Compared columns (15):** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

- Total cells (all): 293085
- Used cells (GT in {1,-1}): 28256 | Dropped clean (GT=0): 264829
- Predicted clean within used subset (counted as FN): 0

## Per-class Metrics

|   class_label | class_name    |   support_true |   precision |   recall |       f1 |    TP |   FP |   FN |
|--------------:|:--------------|---------------:|------------:|---------:|---------:|------:|-----:|-----:|
|             1 | intentional   |          13291 |    0.765864 | 0.883530 | 0.820500 | 11743 | 3590 | 1548 |
|            -1 | unintentional |          14965 |    0.880214 | 0.760107 | 0.815763 | 11375 | 1548 | 3590 |

## Overall Metrics (intent-only)

| metric                           |   precision |   recall |       f1 |   matches |   mismatches |   n_cells_used |   n_cells_total |   n_clean_dropped |   predicted_clean_inside_used |
|:---------------------------------|------------:|---------:|---------:|----------:|-------------:|---------------:|----------------:|------------------:|------------------------------:|
| overall_micro (intent-only)      |    0.818163 | 0.818163 | 0.818163 |     23118 |         5138 |          28256 |          293085 |            264829 |                             0 |
| overall_macro (mean over {1,-1}) |    0.823039 | 0.821819 | 0.818132 |     23118 |         5138 |          28256 |          293085 |            264829 |                             0 |

## Confusion Matrix 2×2 (true rows × pred columns)

|                    |   Pred=1 |   Pred=-1 |
|:-------------------|---------:|----------:|
| 1 (intentional)    |    11743 |      1548 |
| -1 (unintentional) |     3590 |     11375 |