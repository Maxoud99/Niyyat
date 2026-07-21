# Evaluation Report (Intent vs Unintentional Only)

**Compared columns (15):** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

- Total cells (all): 293085
- Used cells (GT in {1,-1}): 28256 | Dropped clean (GT=0): 264829
- Predicted clean within used subset (counted as FN): 0

## Per-class Metrics

|   class_label | class_name    |   support_true |   precision |   recall |       f1 |    TP |    FP |    FN |
|--------------:|:--------------|---------------:|------------:|---------:|---------:|------:|------:|------:|
|             1 | intentional   |          13291 |    0.563600 | 0.106012 | 0.178456 |  1409 |  1091 | 11882 |
|            -1 | unintentional |          14965 |    0.538671 | 0.927097 | 0.681417 | 13874 | 11882 |  1091 |

## Overall Metrics (intent-only)

| metric                           |   precision |   recall |       f1 |   matches |   mismatches |   n_cells_used |   n_cells_total |   n_clean_dropped |   predicted_clean_inside_used |
|:---------------------------------|------------:|---------:|---------:|----------:|-------------:|---------------:|----------------:|------------------:|------------------------------:|
| overall_micro (intent-only)      |    0.540876 | 0.540876 | 0.540876 |     15283 |        12973 |          28256 |          293085 |            264829 |                             0 |
| overall_macro (mean over {1,-1}) |    0.551135 | 0.516554 | 0.429937 |     15283 |        12973 |          28256 |          293085 |            264829 |                             0 |

## Confusion Matrix 2×2 (true rows × pred columns)

|                    |   Pred=1 |   Pred=-1 |
|:-------------------|---------:|----------:|
| 1 (intentional)    |     1409 |     11882 |
| -1 (unintentional) |     1091 |     13874 |