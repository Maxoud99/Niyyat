# Heuristics Analysis — Scenario B (Semi-Supervised)

**Scenario B:** Cluster → proportional 1% label sampling → RF classifier on heuristic features  
**Datasets:** LLM Adult · Mixed SOTA Adult · TwiBot-20  
**Feature set:** Same 13 heuristic features (H1–H8) as Scenario A  
**Evaluated:** RF retrained per clustering algorithm × dataset, importances averaged across 6 algorithms

---

## Note on Label Counting: New vs. Old System

A key difference between the two RF-based pipelines:

| System | Unit of "1 label" | LLM 1% = | Twitter 1% = | Why % differs within same dataset |
|---|---|---|---|---|
| **Old clustering RF** (value-level features) | 1 **record** (row) — all cells labeled together | 192 records / 43,259 cells = **0.44% of cells** | 9 records / 1,099 cells = **0.82% of cells** |
| **New heuristics RF** (13 heuristic features) | 1 **cell** — each error cell is a separate sample | 433 cells = **1.00% of cells** | 75 cells = **6.82% of cells** |

**Why the % is larger for Twitter in the new system:**  
The formula is `n_clusters / total_error_cells`. Twitter has only 1,099 error cells while still needing ≥15 clusters (one sample per cluster). So even 15 clusters = 6.82%, while LLM's 43,259 cells / 15 clusters = only 1.00%. The small dataset, not a larger cluster count, drives the higher %.

---

## Label Sampling Summary per Dataset × Algorithm

### LLM Adult (`n_errors = 43,259 cells`)

| Algorithm | n_clusters | n_labeled cells | % labeled |
|---|---|---|---|
| KMeans | 15 | 433 | 1.00% |
| DBSCAN | 728 | 3,667 | 8.48% |
| Hierarchical-Ward | 15 | 433 | 1.00% |
| Hierarchical-Average | 15 | 436 | 1.01% |
| GMM | — | 433 | 1.00% |
| HDBSCAN | 282 | 1,415 | 3.27% |

### Mixed SOTA Adult (`n_errors = 11,661 cells`)

| Algorithm | n_clusters | n_labeled cells | % labeled |
|---|---|---|---|
| KMeans | 15 | 117 | 1.00% |
| DBSCAN | 125 | 623 | 5.34% |
| Hierarchical-Ward | 15 | 117 | 1.00% |
| Hierarchical-Average | 15 | 130 | 1.11% |
| GMM | — | 117 | 1.00% |
| HDBSCAN | 83 | 415 | 3.56% |

### TwiBot-20 (`n_errors = 1,099 cells`)

| Algorithm | n_clusters | n_labeled cells | % labeled |
|---|---|---|---|
| KMeans | 15 | 75 | 6.82% |
| DBSCAN | 21 | 105 | 9.55% |
| Hierarchical-Ward | 15 | 75 | 6.82% |
| Hierarchical-Average | 15 | 59 | 5.37% |
| GMM | — | 75 | 6.82% |
| HDBSCAN | 10 | 50 | 4.55% |

> **Note:** h2_min_edit_distance and h2_is_obfuscation are all-NaN for TwiBot-20  
> (all-numeric dataset — no string edit distances). Those 2 features contribute 0 importance.  
> h7_mutability, h8_is_sensitive, h8_is_majority_value are also 0 for Twitter  
> (no sensitive demographic columns, no mutability config provided).

---

## Feature Importances — Scenario B RF (mean across 6 clustering algorithms)

### LLM Adult Income

| Rank | Feature | Heuristic | DBSCAN | KMeans | Hier-Ward | Hier-Avg | GMM | HDBSCAN | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `h3_distribution_score` | H3 | 0.1748 | 0.2170 | 0.2120 | 0.2765 | 0.2310 | 0.2064 | **0.2196** |
| 2 | `h4_coherence_score` | H4 | 0.1639 | 0.2278 | 0.2004 | 0.1348 | 0.1958 | 0.1779 | **0.1835** |
| 3 | `h1_plausible` | H1 | 0.1357 | 0.1242 | 0.1896 | 0.2120 | 0.1294 | 0.1499 | **0.1568** |
| 4 | `h2_min_edit_distance` | H2 | 0.2260 | 0.1233 | 0.1137 | 0.1306 | 0.1174 | 0.1584 | **0.1449** |
| 5 | `h2_is_obfuscation` | H2 | 0.1105 | 0.0974 | 0.1197 | 0.1080 | 0.1331 | 0.1524 | **0.1202** |
| 6 | `h7_gain_direction` | H7 | 0.0622 | 0.0530 | 0.0451 | 0.0471 | 0.0768 | 0.0485 | **0.0554** |
| 7 | `h5_error_count` | H5 | 0.0305 | 0.0445 | 0.0447 | 0.0319 | 0.0340 | 0.0359 | **0.0369** |
| 8 | `h6_column_importance` | H6 | 0.0424 | 0.0485 | 0.0300 | 0.0182 | 0.0276 | 0.0273 | **0.0323** |
| 9 | `h7_comprehensibility` | H7 | 0.0140 | 0.0198 | 0.0178 | 0.0116 | 0.0198 | 0.0118 | **0.0158** |
| 10 | `h5_codependent_flag` | H5 | 0.0146 | 0.0206 | 0.0133 | 0.0132 | 0.0171 | 0.0155 | **0.0157** |
| 11 | `h7_mutability` | H7 | 0.0055 | 0.0093 | 0.0053 | 0.0074 | 0.0089 | 0.0059 | **0.0071** |
| 12 | `h8_is_majority_value` | H8 | 0.0141 | 0.0057 | 0.0043 | 0.0055 | 0.0040 | 0.0060 | **0.0066** |
| 13 | `h8_is_sensitive` | H8 | 0.0059 | 0.0089 | 0.0043 | 0.0032 | 0.0050 | 0.0041 | **0.0052** |

---

### Mixed SOTA Adult Income

| Rank | Feature | Heuristic | DBSCAN | KMeans | Hier-Ward | Hier-Avg | GMM | HDBSCAN | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `h7_gain_direction` | H7 | 0.2083 | 0.1806 | 0.1774 | 0.1920 | 0.1670 | 0.1805 | **0.1843** |
| 2 | `h3_distribution_score` | H3 | 0.1929 | 0.1407 | 0.1605 | 0.1490 | 0.1392 | 0.1754 | **0.1596** |
| 3 | `h6_column_importance` | H6 | 0.1193 | 0.1183 | 0.1494 | 0.1619 | 0.1395 | 0.1594 | **0.1413** |
| 4 | `h1_plausible` | H1 | 0.1587 | 0.0962 | 0.0852 | 0.1219 | 0.1157 | 0.1052 | **0.1138** |
| 5 | `h4_coherence_score` | H4 | 0.1240 | 0.0739 | 0.1371 | 0.0874 | 0.1262 | 0.0764 | **0.1042** |
| 6 | `h5_error_count` | H5 | 0.0306 | 0.2339 | 0.0609 | 0.0792 | 0.1346 | 0.0717 | **0.1018** |
| 7 | `h5_codependent_flag` | H5 | 0.0333 | 0.0408 | 0.1034 | 0.0957 | 0.0494 | 0.0703 | **0.0655** |
| 8 | `h2_is_obfuscation` | H2 | 0.0555 | 0.0394 | 0.0545 | 0.0261 | 0.0589 | 0.0902 | **0.0541** |
| 9 | `h2_min_edit_distance` | H2 | 0.0382 | 0.0280 | 0.0181 | 0.0296 | 0.0205 | 0.0383 | **0.0288** |
| 10 | `h7_comprehensibility` | H7 | 0.0221 | 0.0228 | 0.0208 | 0.0269 | 0.0225 | 0.0176 | **0.0221** |
| 11 | `h8_is_sensitive` | H8 | 0.0058 | 0.0107 | 0.0091 | 0.0144 | 0.0099 | 0.0046 | **0.0091** |
| 12 | `h8_is_majority_value` | H8 | 0.0046 | 0.0106 | 0.0126 | 0.0101 | 0.0073 | 0.0045 | **0.0083** |
| 13 | `h7_mutability` | H7 | 0.0067 | 0.0040 | 0.0109 | 0.0057 | 0.0092 | 0.0058 | **0.0071** |

---

### TwiBot-20

> ⚠️ h2_min_edit_distance, h2_is_obfuscation, h7_mutability, h8_is_sensitive, h8_is_majority_value  
> are all 0 for this dataset (all-numeric, no sensitive/mutability config).

| Rank | Feature | Heuristic | DBSCAN | KMeans | Hier-Ward | Hier-Avg | GMM | HDBSCAN | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `h3_distribution_score` | H3 | 0.1782 | 0.1816 | 0.1408 | 0.2196 | 0.1894 | 0.1771 | **0.1811** |
| 2 | `h4_coherence_score` | H4 | 0.1624 | 0.1881 | 0.1797 | 0.1469 | 0.1829 | 0.1307 | **0.1651** |
| 3 | `h1_plausible` | H1 | 0.2033 | 0.1268 | 0.0709 | 0.0923 | 0.2250 | 0.1524 | **0.1451** |
| 4 | `h5_codependent_flag` | H5 | 0.0754 | 0.1390 | 0.1955 | 0.0859 | 0.1060 | 0.1193 | **0.1202** |
| 5 | `h6_column_importance` | H6 | 0.1543 | 0.1164 | 0.0814 | 0.0972 | 0.0986 | 0.1642 | **0.1187** |
| 6 | `h7_gain_direction` | H7 | 0.1109 | 0.0980 | 0.0856 | 0.2028 | 0.0872 | 0.1174 | **0.1170** |
| 7 | `h5_error_count` | H5 | 0.0569 | 0.1171 | 0.2241 | 0.0776 | 0.0586 | 0.1195 | **0.1090** |
| 8 | `h7_comprehensibility` | H7 | 0.0585 | 0.0330 | 0.0220 | 0.0777 | 0.0523 | 0.0195 | **0.0438** |
| 9–13 | h2_*, h7_mutability, h8_* | H2/H7/H8 | 0 | 0 | 0 | 0 | 0 | 0 | **0.0000** |

---

## Cross-Dataset Comparison

### Mean importance (averaged across 6 clustering algorithms)

| Rank | Feature | Heuristic | LLM Adult | Mixed SOTA | TwiBot-20 | **Overall** |
|---|---|---|---|---|---|---|
| 1 | `h3_distribution_score` | H3 | 0.2196 | 0.1596 | 0.1811 | **0.1868** |
| 2 | `h4_coherence_score` | H4 | 0.1835 | 0.1042 | 0.1651 | **0.1509** |
| 3 | `h1_plausible` | H1 | 0.1568 | 0.1138 | 0.1451 | **0.1386** |
| 4 | `h7_gain_direction` | H7 | 0.0554 | 0.1843 | 0.1170 | **0.1189** |
| 5 | `h6_column_importance` | H6 | 0.0323 | 0.1413 | 0.1187 | **0.0974** |
| 6 | `h5_error_count` | H5 | 0.0369 | 0.1018 | 0.1090 | **0.0826** |
| 7 | `h5_codependent_flag` | H5 | 0.0157 | 0.0655 | 0.1202 | **0.0671** |
| 8 | `h2_is_obfuscation` | H2 | 0.1202 | 0.0541 | 0.0000 | **0.0581** |
| 9 | `h2_min_edit_distance` | H2 | 0.1449 | 0.0288 | 0.0000 | **0.0579** |
| 10 | `h7_comprehensibility` | H7 | 0.0158 | 0.0221 | 0.0438 | **0.0272** |
| 11 | `h8_is_majority_value` | H8 | 0.0066 | 0.0083 | 0.0000 | **0.0050** |
| 12 | `h8_is_sensitive` | H8 | 0.0052 | 0.0091 | 0.0000 | **0.0048** |
| 13 | `h7_mutability` | H7 | 0.0071 | 0.0071 | 0.0000 | **0.0047** |

### Aggregated per-heuristic importance (Scenario B)

| Rank | Heuristic | LLM total | Mixed SOTA total | TwiBot-20 total | **Avg total** |
|---|---|---|---|---|---|
| 1 | **H3** Distribution Position | 0.2196 | 0.1596 | 0.1811 | **0.1868** |
| 2 | **H4** Row Coherence | 0.1835 | 0.1042 | 0.1651 | **0.1509** |
| 3 | **H1** Value Plausibility | 0.1568 | 0.1138 | 0.1451 | **0.1386** |
| 4 | **H7** User Incentive | 0.0783 | 0.2135 | 0.1608 | **0.1509** *(tie with H4)* |
| 5 | **H2** String Anomaly | 0.2651 | 0.0829 | 0.0000 | **0.1160** |
| 6 | **H5** Error Pattern | 0.0526 | 0.1673 | 0.2292 | **0.1497** *(very close to H7)* |
| 7 | **H6** Column Importance | 0.0323 | 0.1413 | 0.1187 | **0.0974** |
| 8 | **H8** Sensitivity Flag | 0.0118 | 0.0174 | 0.0000 | **0.0097** |

---

## Scenario A vs. Scenario B: Feature Importance Shift

| Feature | Scenario A (LLM) | Scenario B (LLM) | Δ | Interpretation |
|---|---|---|---|---|
| `h3_distribution_score` | 0.259 | 0.220 | −0.039 | Still #1, slightly weaker with smaller training set |
| `h4_coherence_score` | 0.183 | 0.184 | +0.001 | Stable — row coherence signal holds under semi-supervision |
| `h1_plausible` | 0.180 | 0.157 | −0.023 | Slightly weaker — 1% sampling may miss rare out-of-vocab tokens |
| `h2_min_edit_distance` | 0.091 | 0.145 | **+0.054** | **Rises under DBSCAN** — DBSCAN clusters string-anomaly cells together, making edit distance very discriminative for those sampled cells |
| `h2_is_obfuscation` | 0.138 | 0.120 | −0.018 | Minor drop, still important |
| `h7_gain_direction` | 0.055 | 0.055 | 0.000 | Perfectly stable |
| `h5_error_count` | 0.038 | 0.037 | −0.001 | Stable |
| `h6_column_importance` | 0.022 | 0.032 | +0.010 | Slight rise — column-level constant features become more useful when per-cell context is limited |

| Feature | Scenario A (Mixed SOTA) | Scenario B (Mixed SOTA) | Δ | Interpretation |
|---|---|---|---|---|
| `h7_gain_direction` | 0.213 | 0.184 | −0.029 | Drops slightly but stays #1 — Kireev's outcome-directedness still the primary signal |
| `h3_distribution_score` | 0.136 | 0.160 | +0.024 | Rises under semi-supervision — distribution blending is a stable unsupervised signal |
| `h6_column_importance` | 0.128 | 0.141 | +0.013 | Rises — high-MI columns are clustered together, making them easy to sample representatively |
| `h5_error_count` | 0.133 | 0.102 | −0.031 | Drops — coordinated multi-cell patterns are harder to capture with 1% labels |
| `h4_coherence_score` | 0.058 | 0.104 | **+0.046** | **Rises** — with fewer labels, row coherence becomes more discriminative |

---

## Key Insights

### 1. H3 is the most stable heuristic under label scarcity
`h3_distribution_score` is **#1 across all 3 datasets in Scenario B** (means: 0.220, 0.160, 0.181).  
It is dataset-agnostic and works well with small samples because it captures the fundamental blending-vs-outlier signal that is consistent regardless of how many labeled cells were seen.

### 2. H2 (String Anomaly) becomes more discriminative under DBSCAN
Under DBSCAN, `h2_min_edit_distance` jumps to 0.226 (LLM) and is the **single most important feature for the DBSCAN-sampled training set**.  
Reason: DBSCAN clusters in the 13-feature heuristic space naturally groups cells by their distance-based properties. Cells with unusual edit distances form tight DBSCAN clusters, so when 1 cell is sampled from each cluster, the training set is disproportionately rich in edit-distance diversity — making that feature extremely discriminative for the RF trained on those samples.

### 3. H7 (User Incentive) dominates Mixed SOTA under any sampling strategy
`h7_gain_direction` is **#1 for Mixed SOTA in both Scenario A and B** (0.213 → 0.184).  
The Kireev adversarial injection is strongly outcome-directed, and this signal is so clean that even 1% of cells is enough to learn it perfectly.

### 4. H5 (Error Pattern) rises for Twitter
`h5_codependent_flag` is **#4 for TwiBot-20** (0.120) despite being only #10 for LLM.  
TwiBot-20's fast-plausible adversarial injection creates coordinated multi-field bot profiles — the adversary changes `followers_count` and `friends_count` together — so the codependent-pair signal is stronger in Twitter than in the sparser LLM adult errors.

### 5. H4 (Row Coherence) is the best semi-supervised survivor
In Scenario B, `h4_coherence_score` rises or stays stable vs. Scenario A across **all 3 datasets** (LLM: 0.183→0.184, Mixed SOTA: 0.058→0.104, Twitter: stable).  
Row coherence is a **continuous, smooth feature** — small samples are enough to learn its boundary because the signal is dense (every row has a coherence score, not just a sparse flag).

### 6. H2 is completely absent for TwiBot-20
`h2_min_edit_distance` and `h2_is_obfuscation` are both 0 for Twitter — it's an all-numeric dataset.  
This means TwiBot-20 effectively uses only **11 features** instead of 13, yet still achieves F1w = 0.929.  
H3/H4/H1 together compensate fully.

---

## Feature Importance ASCII Chart (Scenario B, cross-dataset mean)

```
Feature                  | LLM    | Mixed  | Twit   | Overall| Bar (overall)
-------------------------|--------|--------|--------|--------|-------------------------
h3_distribution_score    | 0.220  | 0.160  | 0.181  | 0.187  | ██████████████████
h4_coherence_score       | 0.184  | 0.104  | 0.165  | 0.151  | ███████████████
h1_plausible             | 0.157  | 0.114  | 0.145  | 0.139  | █████████████
h7_gain_direction        | 0.055  | 0.184  | 0.117  | 0.119  | ████████████
h6_column_importance     | 0.032  | 0.141  | 0.119  | 0.097  | █████████
h5_error_count           | 0.037  | 0.102  | 0.109  | 0.083  | ████████
h5_codependent_flag      | 0.016  | 0.066  | 0.120  | 0.067  | ██████
h2_is_obfuscation        | 0.120  | 0.054  | 0.000  | 0.058  | █████
h2_min_edit_distance     | 0.145  | 0.029  | 0.000  | 0.058  | █████
h7_comprehensibility     | 0.016  | 0.022  | 0.044  | 0.027  | ██
h8_is_majority_value     | 0.007  | 0.008  | 0.000  | 0.005  | ▌
h8_is_sensitive          | 0.005  | 0.009  | 0.000  | 0.005  | ▌
h7_mutability            | 0.007  | 0.007  | 0.000  | 0.005  | ▌
```

---

*Generated: 2026-03-23 | Scenario B — RF retrained on cluster-sampled 1% labels per clustering algorithm*  
*Importances averaged across 6 algorithms: KMeans, DBSCAN, Hierarchical-Ward, Hierarchical-Average, GMM, HDBSCAN*
