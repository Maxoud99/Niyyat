# Niyyat

Datasets, baselines, and system code for **Niyyat**, an intent-attribution
system for tabular data: given a dirty table and a binary error mask, decide
whether each erroneous cell was changed intentionally (e.g., a rational actor
gaming a classifier or masking a sensitive attribute) or by accident (typos,
sensor noise), with no clean reference values required.

This is the artifact repository for:

> Mohamed Abdelmaksoud, Konrad Rieck, and Ziawasch Abedjan. **NIYYAT:
> Intent Attribution for Erroneous Cells in Tabular Data.**
> PVLDB, Vol. 20 (target), 2027. *(paper under preparation; DOI and public
> preprint link to follow.)*

```
Niyyat/
├── Datasets/      Adult, TwitterBot, TabFact, eBay, Credit Card Fraud — the dirty-data benchmarks
├── Baselines/     LLMs (per model), FraudDetector, Random
├── System/        Heuristics Attribution, Reference-Augmented, User-guided Attribution
└── Scripts/       Wrappers that run the pipeline (see Scripts/README.md)
```

`Results/` and `fixes/` are excluded from this repo (`.gitignore`) — they hold
generated run artifacts and working notes, not source material.

## Datasets/

| Folder | Notes |
|---|---|
| `Adult/LLM-based` | Gemini-2.5-generated intentional/unintentional edits over a UCI Adult Income subsample |
| `Adult/Mixed-sota-based` | tab_err (unintentional) + Kireev greedy-search adversarial attack (intentional) |
| `Adult/TFM-based` | TabPFN-guided masking + imputation |
| `TwitterBot/LLM-based` | Gemini-2.5-generated bot-evasion edits over TwiBot-20 profiles |
| `TwitterBot/Mixed-sota-based` | TwiBot-20-based, same tab_err + Kireev pipeline as Adult-Mixed |
| `TwitterBot/TFM-based` | TabPFN-guided masking over TwiBot-20 |
| `TabFact` | `final/` = cleaned dirty-data variant used for attribution; `original_splits/` = raw TabFact train/val/test |
| `eBay` | ~10,345-row eBay listings (4 marketplaces); intent inferred from seller behavior, validated against human annotation |
| `CreditCardFraud` | Confirmed card-issuer fraud transactions; intent label is the real fraud determination, not a proxy. The two large CSVs here are tracked with **Git LFS** |

## System/

Three attribution-method families:

- **Heuristics Attribution** — H1-H8 dirty-data-only feature engineering + classifier (Regime I: dirty table and error mask alone).
- **Reference-Augmented** — methods that lean on an external reference beyond the dirty data itself: the LLM engine (clean-row-conditioned prompting), `fingerprint/` (heuristics augmented with statistics computed from clean reference values), and `declarative/` (a user's natural-language domain description → LLM-derived integrity constraints).
- **User-guided Attribution** — `clustering-organized/` (6 clustering algorithms × 5 label-propagation methods, ~1% proportional label budget) and `no_clustering/` (the uniform-random-sampling ablation of the same pipeline).
- **Shared-Core** — Stage-1 ensemble error detection, preprocessing, evaluation, and imputation code all three attribution methods depend on.

## Baselines/

- **LLMs/{Gemini,Llama,Qwen,Deepseek-R1,Mixtral,GPT,Claude}** — each model's folder has `Adult/`, `TwitterBot/`, `TabFact/` subfolders with `code/` + `results/`. GPT only has an Adult implementation. Claude has no implementation — `Baselines/LLMs/Claude/NOTE.md` explains why and points at the closest analogues to adapt.
- **FraudDetector** — ECOD + Leave-One-Out cell attribution (zero-label, oracle-tuned threshold).
- **Random** — random/constant/probability guessing strategies, the minimum-performance floor.

## A note on Scripts/

`Scripts/*.sh` were written as wrappers around this project's original,
larger working tree (`llms_baseline/`) and reference absolute paths outside
this repo. They are kept for documentation of the pipeline shape, but will
not run as-is against a bare clone of this repo — see `Scripts/README.md`
for what each one does and what it expects on disk.

## Requirements

Git LFS is required to pull the Credit Card Fraud CSVs:
```
git lfs install
git clone <this repo>
```
