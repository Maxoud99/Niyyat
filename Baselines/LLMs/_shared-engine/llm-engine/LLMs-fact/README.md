# TabFact — Intent Attribution via Gemini

Feature-level intent attribution on the cleaned TabFact dataset.
Given (dirty record, Gemini-derived correct record, unified error mask),
each error cell is classified as:

| Label | Meaning |
|:-----:|---------|
| **1** | INTENTIONAL — deliberate factual manipulation (entity/type/location swap, domain flip, meaning-altering value change). |
| **-1** | UNINTENTIONAL — typo, case / spacing drift, abbreviation, ±1–2 numeric drift. |
| **0** | No change (mask = 0). |

## Inputs

| File | Source |
|------|--------|
| `datasets/final/combined_dataset_TRF_no_claim.csv` | The cleaned dataset (984 rows, includes `metric` column) |
| `outputs/error_detection/run_20251222_163837/error_mask_updated.csv` | Aligned error mask (984 rows, binary 0/1 per feature) |
| `outputs/error_detection/run_20251222_163837/explanations.json` | Gemini detection explanations — provides the **correct value** for each flagged cell |
| `datasets/final/combined_dataset_TRF.csv` | Original dirty CSV (1002 rows) — used only to align new → original row indices when looking up corrections |

The corrected record is reconstructed in memory: for every mask=1 cell the
corresponding `correct_value` from `explanations.json` replaces the dirty cell;
everything else is copied verbatim from the dirty dataset.

## Attributed columns

`subject_entity`, `subject_type`, `subject_location`, `value`, `claim_domain`

`metric` and `is_factual` are excluded (metric has no errors; is_factual is the label).

## Files

| File | Description |
|------|-------------|
| `_gemini_base.py` | Shared base — data loading, alignment, Gemini call, JSON parsing, pipeline orchestrator |
| `intent-attribution-baremin-gemini.py` | **Bare-minimum** prompt: role + strict JSON rules, no rubric |
| `intent-attribution-info-gemini.py` | **Info** prompt: role + rubric + per-cell DIFFS |
| `intent-attribution-few-shots-gemini.py` | **Few-shots** prompt: info + 3 in-context examples |
| `run_all_gemini.py` | One-click runner: executes all three variants sequentially, prints a comparison table |

## Outputs (per variant)

```
outputs/<variant>-gemini/
 ├── intent_labels.csv        # per-cell decision mask  (1 / -1 / 0)
 ├── intent_explanations.csv  # per-cell reason, dirty value, correct value
 ├── run_stats.json           # timing + histograms + decision counts
 └── per_chunk_times.csv      # per-chunk / per-LLM-call timing
```

## Quick start

```bash
cd tabfact/intent-attribution

# Run all three variants on the full dataset
python run_all_gemini.py

# Smoke-test on 30 rows only
python run_all_gemini.py --max-records 30

# Run a single variant
python intent-attribution-info-gemini.py --max-records 50

# Skip one variant in the run-all
python run_all_gemini.py --skip bareminimum
```

## Notes for the student

* Each variant uses `gemini-2.5-pro`, temperature 0, deterministic.
* Requests are chunked (`CHUNK_SIZE = 10` records per call) to keep prompts bounded.
* On API or JSON-parse failure, the cell defaults to `-1` (unintentional) as a
  safety fallback and the fallback is logged in `run_stats.json → status_hist`.
* The `_align_new_to_orig` helper recovers the mapping from each row of the
  cleaned dataset back to its original row index so that per-row corrections
  can be fetched from `explanations.json`.
