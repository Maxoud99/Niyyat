# Evaluation Report (Intent vs Unintentional Only)

**Compared columns (15):** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

- Total cells (all): 293085
- Used cells (GT in {1,-1}): 28256 | Dropped clean (GT=0): 264829
- Predicted clean within used subset (counted as FN): 0

## Per-class Metrics

|   class_label | class_name    |   support_true |   precision |   recall |       f1 |    TP |   FP |   FN |
|--------------:|:--------------|---------------:|------------:|---------:|---------:|------:|-----:|-----:|
|             1 | intentional   |          13291 |    0.662785 | 0.959145 | 0.783889 | 12748 | 6486 |  543 |
|            -1 | unintentional |          14965 |    0.939814 | 0.566589 | 0.706966 |  8479 |  543 | 6486 |

## Overall Metrics (intent-only)

| metric                           |   precision |   recall |       f1 |   matches |   mismatches |   n_cells_used |   n_cells_total |   n_clean_dropped |   predicted_clean_inside_used |
|:---------------------------------|------------:|---------:|---------:|----------:|-------------:|---------------:|----------------:|------------------:|------------------------------:|
| overall_micro (intent-only)      |    0.751239 | 0.751239 | 0.751239 |     21227 |         7029 |          28256 |          293085 |            264829 |                             0 |
| overall_macro (mean over {1,-1}) |    0.801299 | 0.762867 | 0.745428 |     21227 |         7029 |          28256 |          293085 |            264829 |                             0 |

## Confusion Matrix 2×2 (true rows × pred columns)

|                    |   Pred=1 |   Pred=-1 |
|:-------------------|---------:|----------:|
| 1 (intentional)    |    12748 |       543 |
| -1 (unintentional) |     6486 |      8479 |