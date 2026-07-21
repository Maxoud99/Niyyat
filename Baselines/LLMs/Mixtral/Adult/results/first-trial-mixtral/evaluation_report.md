# Evaluation Report (Intent vs Unintentional Only)

**Compared columns (15):** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class

- Total cells (all): 293085
- Used cells (GT in {1,-1}): 28256 | Dropped clean (GT=0): 264829
- Predicted clean within used subset (counted as FN): 0

## Per-class Metrics

|   class_label | class_name    |   support_true |   precision |   recall |       f1 |    TP |    FP |    FN |
|--------------:|:--------------|---------------:|------------:|---------:|---------:|------:|------:|------:|
|             1 | intentional   |          13291 |    0.586526 | 0.033406 | 0.063212 |   444 |   313 | 12847 |
|            -1 | unintentional |          14965 |    0.532819 | 0.979085 | 0.690090 | 14652 | 12847 |   313 |

## Overall Metrics (intent-only)

| metric                           |   precision |   recall |       f1 |   matches |   mismatches |   n_cells_used |   n_cells_total |   n_clean_dropped |   predicted_clean_inside_used |
|:---------------------------------|------------:|---------:|---------:|----------:|-------------:|---------------:|----------------:|------------------:|------------------------------:|
| overall_micro (intent-only)      |    0.534258 | 0.534258 | 0.534258 |     15096 |        13160 |          28256 |          293085 |            264829 |                             0 |
| overall_macro (mean over {1,-1}) |    0.559673 | 0.506245 | 0.376651 |     15096 |        13160 |          28256 |          293085 |            264829 |                             0 |

## Confusion Matrix 2×2 (true rows × pred columns)

|                    |   Pred=1 |   Pred=-1 |
|:-------------------|---------:|----------:|
| 1 (intentional)    |      444 |     12847 |
| -1 (unintentional) |      313 |     14652 |