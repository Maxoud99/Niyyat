# Declarative Heuristics (Family C) — Results & Ablation

**Revision 2: intent-signal rules, replacing denial-constraint validity rules.**

## What changed and why

The original version of Family C asked the LLM to extract generic
**denial-constraint / validity rules** from the analyst's description
("age must be in [17,90]", "education must match education-num") and then
assumed, uniformly, that *any* violation of *any* rule was evidence of
**intentional** manipulation. That assumption is wrong in both directions:

- Many intentional manipulations (TwiBot-20 bot-evasion edits, TabPFN's
  within-distribution Adult-TFM injections) are specifically designed to
  **stay inside valid ranges** — they violate nothing, so a validity-only
  rule set has no signal to find. This was already diagnosed in Revision 1
  of this document ("TwiBot-20's intentional errors stay within valid
  ranges by design, so few cells violate the structural constraints").
- Some validity violations have **no plausible deliberate motive at all**
  (a negative age, an out-of-range hours-per-week) — these are textbook
  *accidental* corruption, yet Revision 1 counted every one of them as
  evidence of intent, working against the classifier on exactly the cells
  where the signal should point the other way.

Revision 2 replaces validity rules with **intent-signal rules**: each rule
is a domain claim of the form *"if a flagged cell matches pattern P, that
is evidence the change was {intentional | unintentional}, because P matches
{a gain-targeted / fairness-masking / obfuscation motive | no plausible
motive at all}."* Every rule carries an explicit signed `intent_signal`
(+1 or −1) and a `motive` tag, extracted directly from the LLM rather than
inferred by a blanket assumption. Critically, intent-signal rules can fire
on **valid, in-range values** — "capital-gain > 0" or "race = White" are
not violations of anything, but they are still domain evidence of a
gain-targeted or fairness-masking motive when they occur on a cell already
flagged as changed. This is exactly the gap that made Revision 1 fail on
TwiBot-20 and Adult-TFM.

**Schema change** (`extractor.py`, `evaluator.py`):

| | Revision 1 (validity) | Revision 2 (intent-signal) |
|---|---|---|
| Rule meaning | "constraint satisfied/violated" | "pattern present, signed evidence" |
| Per-rule feature | `violated_Ck` ∈ {0,1} | `signal_Ck` ∈ {−intent_signal, 0, +intent_signal} |
| Row aggregate | `row_total_violations` (count) | `row_signal_sum` (signed sum) |
| Column aggregate | `n_col_violated`, `col_violation_ratio` | `cell_signal_sum`, `cell_signal_ratio` |
| Polarity | assumed uniformly intentional | explicit per-rule `intent_signal` + `motive` |

The dataset description files (`configs/*.txt`) were rewritten to give the
LLM the domain knowledge needed to ground these motives: which columns and
directions are gain-targeted for each dataset's actual prediction target,
which attributes are sensitive (fairness-masking), what counts as an
obfuscation placeholder, and what counts as a no-motive noise pattern.
Constraints were re-extracted from scratch (`--force-reextract`) for all
three description files; rule counts grew from 18/19/8 (Revision 1) to
31/25/9 (Revision 2) because the new prompt asks for rules in four motive
classes instead of one validity pass.

Everything else is unchanged from Revision 1: DBSCAN clustering fixed
across every scenario, cluster-proportional 1% sampling, Random Forest with
`class_weight="balanced"`. All deltas below are attributable to the rule
semantics alone.

---

## Rule inventory and coverage (Revision 2)

| Dataset | Rules extracted | Active (fired ≥1×) | Coverage | Cells w/ +1 signal | Cells w/ −1 signal |
|---|---:|---:|---:|---:|---:|
| Adult-LLM | 31 | 24 | 60.8% | 15,038 | 22,166 |
| Adult-Mixed | 31 | 26 | 68.1% | 5,644 | 3,532 |
| Adult-TFM | 31 | 10 | 41.2% | 3,781 | 7 |
| TwitterBot-LLM V2 | 25 | 6 | 15.4% | 152 | 21 |
| TwitterBot-Mixed | 25 | 3 | 10.6% | 135 | 56 |
| TwitterBot-TFM | 25 | 4 | 16.2% | 92 | 108 |
| TabFact | 9 | 3 | 60.0% | 0 | 1,347 |

Coverage rose substantially on every dataset relative to Revision 1
(e.g. Adult-TFM: 0.1% → 41.2%; TwitterBot-TFM: 9.3% → 16.2%), because
gain-direction and fairness-masking rules fire on *valid* values that the
old validity-only rule set could never see.

**TabFact anomaly (harmless).** TabFact is single-class (100% intentional
by construction), so any classifier trivially scores F1-w = 1.000
regardless of features. The new rule set's "no-motive / noise" range
checks (implausible claim values, blank entities) do fire on 1,347 of
2,245 cells here — but on this corpus, *even* an implausible-looking value
is intentional (a refutation only needs to be factually wrong, not subtle),
so these rules are mis-signed for TabFact specifically. It costs nothing
because the metric is degenerate, but it is a useful reminder that the
"no plausible motive ⇒ unintentional" heuristic assumes a domain where
*unintentional* errors exist at all; TabFact's corpus of deliberately
refuted claims does not have that comparison class.

---

## All-Dataset Summary — Revision 1 (validity) vs. Revision 2 (intent-signal), F1-weighted

| Dataset | C (R1) | C (R2) | ΔC | B+C (R1) | B+C (R2) | ΔBC | (B+)+C (R1) | (B+)+C (R2) | ΔBplusC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adult-LLM | 0.807 | 0.791 | −1.6 pp | 0.895 | 0.860 | −3.5 pp | 0.902 | 0.897 | −0.6 pp |
| Adult-Mixed | 0.794 | 0.813 | **+1.9 pp** | 0.952 | 0.958 | +0.6 pp | 0.969 | 0.974 | +0.5 pp |
| Adult-TFM | 0.613 | **0.728** | **+11.5 pp** | 0.910 | 0.921 | +1.1 pp | 0.914 | 0.901 | −1.3 pp |
| TwitterBot-LLM V2 | 0.808 | 0.730 | −7.8 pp | 0.802 | 0.733 | −6.9 pp | 0.811 | **0.824** | +1.3 pp |
| TwitterBot-Mixed | 0.632 | 0.632 | 0.0 pp | 0.843 | 0.835 | −0.8 pp | 0.776 | 0.776 | 0.0 pp |
| TwitterBot-TFM | 0.472 | **0.616** | **+14.4 pp** | 0.778 | **0.816** | +3.8 pp | 0.730 | 0.728 | −0.2 pp |
| TabFact ‡ | 1.000 | 1.000 | 0.0 pp | 1.000 | 1.000 | 0.0 pp | 1.000 | 1.000 | 0.0 pp |

‡ Single-class, trivial regardless of revision.

---

## Where it helped, where it hurt, and why

**TwitterBot-TFM (+14.4 pp standalone C) and Adult-TFM (+11.5 pp standalone
C) — the two datasets Revision 1 explicitly flagged as failures.**
Revision 1's own write-up said of Adult-TFM: *"only 3 of 18 constraints are
active... TabPFN generates within-distribution errors by design."* That
diagnosis is exactly what intent-signal rules are designed to fix: TFM's
injected values don't break any *validity* rule, but they can still trip a
*gain-direction* rule (a capital-gain or hours-per-week pushed into the
high-income range is still gain-targeted even if it's "valid"). Feature
importance confirms the mechanism — on Adult-TFM, `signal_C1`
(capital-gain > 0, gain-targeted) and `signal_C3` (hours-per-week in
[40,99]) together carry 14.3% importance, dimensions that did not exist
under the old validity-only scheme. The same pattern holds on
TwitterBot-TFM, where `row_signal_sum` (the signed row-level aggregate)
alone carries 48.6% importance — most of it now coming from rules that
fire on *valid* profile-completeness values, not invalid ones.

**Adult-LLM regresses modestly (−0.6 to −3.5 pp across all three C-involving
scenarios).** The dominant Revision 1 signal here — the
education/education-num cross-field mismatch — is still present in
Revision 2 (now `signal_C9`, explicitly tagged `gain_targeted`) and still
fires on the same cells. The regression is most plausibly a **dimensionality
cost**: Revision 2 has 31 rules vs. Revision 1's 18, so the C-scenario
feature space grows from 22 to 35 dimensions at the *same* 433-seed budget,
giving DBSCAN a sparser space to cluster from cells that mostly carry
zeros for any single rule. This is the same dimensionality/seed-budget
tradeoff already documented for Heuristic vs. Reference-Augmented
(§ Heuristic vs. Reference-Augmented in the main paper) — more features
are not free at a fixed label budget.

**TwitterBot-LLM V2's standalone C and B+C drop (−7.8, −6.9 pp), but
(B+)+C improves (+1.3 pp).** The Gemini-driven adversarial strategy on this
dataset is idiosyncratic enough that Revision 1's narrow, mostly-noise rule
set (2 active rules, both incidental) happened to correlate with this
specific injector's footprint. Revision 2's broader rule set (6 active
rules, mostly genuine bot-evasion gain-direction signals such as
`default_profile`/`geo_enabled`/`has_location`) is more principled but less
narrowly tuned to this one injector's quirks, and the standalone/B+C
scenarios pay a small price for it at only 10 seeds. The full combination
recovers and slightly exceeds Revision 1, consistent with the
already-documented pattern that this dataset's gains come from
*combining* evidence sources, not from any one of them alone.

**TwitterBot-Mixed and TabFact are essentially unchanged.** TwitterBot-Mixed's
active rule count stayed small (3 vs. 2 active rules before) and the fired
cells largely overlap with the old scheme's; TabFact is degenerate
regardless of rule semantics (see above).

---

## Per-Dataset Full Metrics (Revision 2)

### Adult-LLM (43,259 cells)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.746 | 0.823 | 0.791 |
| B + C | 0.833 | 0.879 | 0.860 |
| (B+) + C | 0.881 | 0.908 | 0.897 |

### Adult-Mixed (11,661 cells)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.720 | 0.858 | 0.813 |
| B + C | 0.937 | 0.969 | 0.958 |
| (B+) + C | 0.960 | 0.981 | 0.974 |

### Adult-TFM (9,199 cells)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.629 | 0.776 | 0.728 |
| B + C | 0.874 | 0.944 | 0.921 |
| (B+) + C | 0.846 | 0.928 | 0.901 |

### TwitterBot-LLM V2 (1,043 cells)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.668 | 0.771 | 0.730 |
| B + C | 0.696 | 0.757 | 0.733 |
| (B+) + C | 0.808 | 0.834 | **0.824** |

### TwitterBot-Mixed (1,099 cells)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.281 | 0.791 | 0.632 |
| B + C | 0.764 | 0.867 | 0.835 |
| (B+) + C | 0.706 | 0.808 | 0.776 |

### TwitterBot-TFM (1,184 cells)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.537 | 0.641 | 0.616 |
| B + C | 0.626 | 0.876 | **0.816** |
| (B+) + C | 0.538 | 0.787 | 0.728 |

### TabFact (2,245 cells) ‡

All scenarios score F1-w = 1.000 (single-class, see anomaly note above).

---

## Research Questions (Status, Revision 2)

| RQ | Hypothesis | Status |
|---|---|---|
| RQ1 — C vs. LLM | C > Best LLM on ≥1 dataset | **Still not confirmed.** Adult-LLM: 0.791 vs. 0.818 (−2.7 pp). Adult-Mixed: 0.813 vs. 0.917 (−10.4 pp). Closest approach is TwitterBot-LLM V2's (B+)+C at 0.824, but that combines C with B+, not C alone. |
| RQ2 — B+C vs. B | B+C > standalone Heuristic (B) on ≥1 dataset | **Confirmed only on Adult-TFM** (B+C 0.921 vs. B 0.917, +0.4 pp) — the one dataset where the new gain-direction rules align with TFM's within-distribution injection mechanism. B+C trails standalone B everywhere else: Adult-LLM (0.860 vs. 0.885, −2.5 pp), Adult-Mixed (0.958 vs. 0.966, −0.8 pp), TwitterBot-LLM V2 (0.733 vs. 0.804, −7.1 pp), TwitterBot-Mixed (0.835 vs. 0.848, −1.3 pp), TwitterBot-TFM (0.816 vs. 0.946, −13.0 pp). |
| RQ3 — (B+)+C vs. B+ | Residual signal beyond standalone Reference-Augmented (B+) | **Confirmed on TwitterBot-LLM V2 (+16.2 pp, 0.824 vs. 0.662) and TwitterBot-Mixed (+2.7 pp, 0.776 vs. 0.749)** — both datasets where B+'s larger feature space already clusters poorly from few seeds, so the constraint features stabilise rather than dilute. Degrades on Adult-LLM (−4.6 pp), Adult-Mixed (−1.5 pp), Adult-TFM (−3.0 pp), TwitterBot-TFM (−12.4 pp), where standalone B+ is already strong and the extra rule dimensions cost more than they add. |
| RQ4 — Coverage predicts gain | Higher coverage ⇒ higher standalone C utility | **Weaker than before.** Coverage rose on every dataset (table above) but standalone C only improved on 3 of 6 non-trivial datasets. Motive *alignment* with the actual injection mechanism (gain-direction rules matching TFM's within-distribution edits) predicts standalone C utility better than raw coverage alone. |

---

## Output Files

| Location | Contents |
|---|---|
| `results/constraints/adult_income_constraints.json` | 31 intent-signal rules — Adult-LLM, Adult-Mixed, Adult-TFM |
| `results/constraints/twibot20_constraints.json` | 25 intent-signal rules — all three TwitterBot datasets |
| `results/constraints/tabfact_constraints.json` | 9 intent-signal rules — TabFact |
| `results/{dataset}/{C,BC,BplusC}/features.csv` | Per-cell feature matrix (`signal_C1..Ck`, `n_applicable`, `cell_signal_sum`, `cell_signal_ratio`, `row_signal_sum`) |
| `results/{dataset}/{C,BC,BplusC}/metrics.json` | Accuracy, F1-w, F1-INT, F1-UNINT |
| `results/{dataset}/{C,BC,BplusC}/feature_importance.csv` | RF feature importances |
| `results/summary_declarative.csv` | All datasets × scenarios summary |

## Reproducing

```bash
cd error_detection_system/src/attribution/declarative
# Re-extract intent-signal rules from the rewritten configs/*.txt (one-time, cached)
python3 -c "from extractor import extract_constraints; extract_constraints('configs/adult_income.txt', output_json='results/constraints/adult_income_constraints.json', force_reextract=True)"
# Run all datasets / scenarios
python run_all_declarative.py
```
