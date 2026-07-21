# Evaluation Report (Intent vs Unintentional Only)

**Compared columns (15):** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

- Total cells (all): 293085
- Used cells (GT in {1,-1}): 28256 | Dropped clean (GT=0): 264829
- Predicted clean within used subset (counted as FN): 0

## Per-class Metrics

|   class_label | class_name    |   support_true |   precision |   recall |       f1 |   TP |   FP |   FN |
|--------------:|:--------------|---------------:|------------:|---------:|---------:|-----:|-----:|-----:|
|             1 | intentional   |          13291 |    0.584613 | 0.661500 | 0.620685 | 8792 | 6247 | 4499 |
|            -1 | unintentional |          14965 |    0.659605 | 0.582559 | 0.618693 | 8718 | 4499 | 6247 |

## Overall Metrics (intent-only)

| metric                           |   precision |   recall |       f1 |   matches |   mismatches |   n_cells_used |   n_cells_total |   n_clean_dropped |   predicted_clean_inside_used |
|:---------------------------------|------------:|---------:|---------:|----------:|-------------:|---------------:|----------------:|------------------:|------------------------------:|
| overall_micro (intent-only)      |    0.619691 | 0.619691 | 0.619691 |     17510 |        10746 |          28256 |          293085 |            264829 |                             0 |
| overall_macro (mean over {1,-1}) |    0.622109 | 0.622030 | 0.619689 |     17510 |        10746 |          28256 |          293085 |            264829 |                             0 |

## Confusion Matrix 2×2 (true rows × pred columns)

|                    |   Pred=1 |   Pred=-1 |
|:-------------------|---------:|----------:|
| 1 (intentional)    |     8792 |      4499 |
| -1 (unintentional) |     6247 |      8718 |