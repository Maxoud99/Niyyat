#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Declarative Intent-Signal Evaluator
=====================================

Takes a list of intent-indicative rules (from extractor.py) and applies them
to a dirty dataset + binary error mask, producing a per-erroneous-cell
feature matrix suitable for the clustering → RF attribution pipeline.

Each rule fires on a row when its pattern is PRESENT (expression evaluates
True) and carries a signed `intent_signal` (+1 = evidence of intentional
change, -1 = evidence of unintentional/noise change). This is the key
departure from a denial-constraint evaluator: a fired rule does not mean
"this cell is invalid", it means "this cell carries domain evidence in a
specific direction", and that direction is read directly off the rule
rather than assumed uniformly for every violation.

Feature columns per erroneous cell:
  - signal_C1, signal_C2, ...  : intent_signal of rule Ck if it fires AND
                                  this column is in Ck's scope, else 0
  - n_applicable                : how many rules involve this column
  - cell_signal_sum             : signed sum of fired signals for this column
  - cell_signal_ratio           : cell_signal_sum / n_applicable
  - row_signal_sum              : signed sum of all fired signals anywhere in
                                   the row (row-level, shared across the row's
                                   flagged cells)

Attribution principle: a cell (row_i, col_c) inherits the signal of rule Ck
if and only if:
  1. Ck's pattern fires in row_i, AND
  2. col_c appears in Ck's columns list

This ensures that each cell is only credited/charged for rules it
participates in, not for rules fired by other columns in the same row.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Safe builtins available inside constraint expressions
_SAFE_BUILTINS = {
    "__builtins__": {},
    "int": int,
    "float": float,
    "str": str,
    "len": len,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "bool": bool,
    "list": list,
    "set": set,
    "dict": dict,
    "isinstance": isinstance,
    "any": any,
    "all": all,
}


# ─────────────────────────────────────────────────────────────────────────────
# Expression evaluation — safe, per-row
# ─────────────────────────────────────────────────────────────────────────────

def _eval_pattern(expression: str, row_dict: Dict[str, str]) -> Optional[bool]:
    """
    Evaluate one rule's pattern expression for one row.

    Returns:
      True  — pattern is PRESENT (rule fires)
      False — pattern is absent (no signal from this rule on this row)
      None  — expression failed (missing column, type error) → not applicable
    """
    try:
        result = eval(expression, _SAFE_BUILTINS, {"row": row_dict})
        return bool(result)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintEvaluator:
    """
    Evaluates intent-indicative rules on a dirty dataset and produces a
    per-erroneous-cell feature matrix.

    Parameters
    ----------
    constraints : List[Dict]
        List of rule dicts, each with: id, description, columns, expression,
        intent_signal (+1/-1), motive.
        Produced by extractor.extract_constraints().
    """

    def __init__(self, constraints: List[Dict]):
        if not constraints:
            raise ValueError("Constraint list is empty.")
        self.constraints = constraints
        self.constraint_ids = [c["id"] for c in constraints]
        self.signals = {c["id"]: int(c.get("intent_signal", 1)) for c in constraints}
        # Pre-build column → constraint indices mapping
        self._col_to_constraints: Dict[str, List[int]] = {}
        for i, c in enumerate(constraints):
            for col in c.get("columns", []):
                self._col_to_constraints.setdefault(col, []).append(i)

    def _row_patterns(self, row_dict: Dict[str, str]) -> Dict[str, Optional[bool]]:
        """
        Evaluate all rule patterns for one row.
        Returns {constraint_id: True/False/None}.
        """
        results = {}
        for c in self.constraints:
            outcome = _eval_pattern(c["expression"], row_dict)
            # outcome=True → pattern present (rule fires), False → absent, None → n/a
            results[c["id"]] = outcome
        return results

    def extract_features(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce a feature matrix with one row per erroneous cell.

        Parameters
        ----------
        dirty_df : DataFrame
            The dirty dataset (N rows × P columns).
        mask_df : DataFrame
            Binary error mask (N rows × P columns). 1 = erroneous cell.

        Returns
        -------
        DataFrame with columns:
            row_idx, column_name,
            signal_C1 ... signal_Ck,
            n_applicable, cell_signal_sum, cell_signal_ratio,
            row_signal_sum
        """
        if len(dirty_df) != len(mask_df):
            raise ValueError(
                f"dirty_df rows ({len(dirty_df)}) != mask_df rows ({len(mask_df)})"
            )

        feature_cols = [c for c in mask_df.columns if c in dirty_df.columns]
        n_constraints = len(self.constraints)
        signal_col_names = [f"signal_{cid}" for cid in self.constraint_ids]

        records = []
        n_errors_found = 0

        for row_idx in range(len(dirty_df)):
            # Build row dict (all values as strings)
            row_dict = {col: str(dirty_df.iloc[row_idx][col]) for col in dirty_df.columns}

            # Evaluate all rule patterns for this row
            row_patterns = self._row_patterns(row_dict)

            # Row-level summary: signed sum of every rule that fires anywhere in the row
            row_signal_sum = sum(
                self.signals[cid] for cid, outcome in row_patterns.items() if outcome is True
            )

            # For each erroneous cell in this row
            for col in feature_cols:
                mask_val = str(mask_df.iloc[row_idx][col]).strip()
                if mask_val not in ("1", "1.0"):
                    continue

                n_errors_found += 1

                # Which rules involve this column?
                applicable_idxs = self._col_to_constraints.get(col, [])
                n_applicable = len(applicable_idxs)

                # Per-rule signed signal for this cell
                signal_flags = {}
                for i, c in enumerate(self.constraints):
                    cid = c["id"]
                    col_name = f"signal_{cid}"
                    outcome = row_patterns[cid]
                    # Credit/charge the signal only if this column is in the
                    # rule's scope AND the rule's pattern fired.
                    if i in applicable_idxs and outcome is True:
                        signal_flags[col_name] = self.signals[cid]
                    else:
                        signal_flags[col_name] = 0

                cell_signal_sum = sum(signal_flags.values())
                cell_signal_ratio = (
                    cell_signal_sum / n_applicable if n_applicable > 0 else 0.0
                )

                rec = {
                    "row_idx": row_idx,
                    "column_name": col,
                }
                rec.update(signal_flags)
                rec["n_applicable"]      = n_applicable
                rec["cell_signal_sum"]   = cell_signal_sum
                rec["cell_signal_ratio"] = cell_signal_ratio
                rec["row_signal_sum"]    = row_signal_sum

                records.append(rec)

        if not records:
            print("[evaluator] Warning: no erroneous cells found in mask.")
            return pd.DataFrame(columns=["row_idx", "column_name"] + signal_col_names +
                                        ["n_applicable", "cell_signal_sum",
                                         "cell_signal_ratio", "row_signal_sum"])

        features_df = pd.DataFrame(records)

        # Ensure all signal_* columns exist (some may be missing if no cell had them)
        for col_name in signal_col_names:
            if col_name not in features_df.columns:
                features_df[col_name] = 0

        print(
            f"[evaluator] {n_errors_found} erroneous cells across {len(dirty_df)} rows | "
            f"{n_constraints} intent-signal rules"
        )
        signal_counts = (features_df[signal_col_names] != 0).sum()
        for cid, cnt in zip(self.constraint_ids, signal_counts):
            if cnt > 0:
                polarity = "+1 intentional" if self.signals[cid] == 1 else "-1 unintentional"
                print(f"  {cid} ({polarity}): {int(cnt)} cells fired")

        return features_df

    def constraint_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all loaded intent-signal rules."""
        rows = []
        for c in self.constraints:
            rows.append({
                "id": c["id"],
                "description": c.get("description", ""),
                "columns": ", ".join(c.get("columns", [])),
                "expression": c.get("expression", ""),
                "intent_signal": c.get("intent_signal", 1),
                "motive": c.get("motive", "unspecified"),
            })
        return pd.DataFrame(rows)
