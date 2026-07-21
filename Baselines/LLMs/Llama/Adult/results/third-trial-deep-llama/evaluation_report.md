# Evaluation Report (Intent vs Unintentional Only)

**Compared columns (15):** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

- Total cells (all): 293085
- Used cells (GT in {1,-1}): 28256 | Dropped clean (GT=0): 264829
- Predicted clean within used subset (counted as FN): 0

## Per-class Metrics

|   class_label | class_name    |   support_true |   precision |   recall |       f1 |    TP |   FP |   FN |
|--------------:|:--------------|---------------:|------------:|---------:|---------:|------:|-----:|-----:|
|             1 | intentional   |          13291 |    0.699939 | 0.432624 | 0.534734 |  5750 | 2465 | 7541 |
|            -1 | unintentional |          14965 |    0.623721 | 0.835282 | 0.714163 | 12500 | 7541 | 2465 |

## Overall Metrics (intent-only)

| metric                           |   precision |   recall |       f1 |   matches |   mismatches |   n_cells_used |   n_cells_total |   n_clean_dropped |   predicted_clean_inside_used |
|:---------------------------------|------------:|---------:|---------:|----------:|-------------:|---------------:|----------------:|------------------:|------------------------------:|
| overall_micro (intent-only)      |    0.645881 | 0.645881 | 0.645881 |     18250 |        10006 |          28256 |          293085 |            264829 |                             0 |
| overall_macro (mean over {1,-1}) |    0.661830 | 0.633953 | 0.624449 |     18250 |        10006 |          28256 |          293085 |            264829 |                             0 |

## Confusion Matrix 2×2 (true rows × pred columns)

|                    |   Pred=1 |   Pred=-1 |
|:-------------------|---------:|----------:|
| 1 (intentional)    |     5750 |      7541 |
| -1 (unintentional) |     2465 |     12500 |