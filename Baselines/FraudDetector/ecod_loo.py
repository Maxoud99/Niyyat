"""
ecod_loo.py
ECOD fraud detector + Leave-One-Out cell-level intent attribution.

Algorithm:
  1. Fit ECOD on the clean rows (numeric-encoded).
  2. For each dirty row with flagged cells:
     a. Score the full dirty row -> score_dirty
     b. For each flagged cell c:
        - Build row_reverted (dirty row with cell c reset to clean value)
        - Score row_reverted -> score_reverted
        - delta[c] = score_dirty - score_reverted
     c. Predict +1 (intentional) if delta[c] > threshold, else -1 (unintentional)

ECOD reference:
  Li et al. "ECOD: Unsupervised Outlier Detection Using Empirical
  Cumulative Distribution Functions." IEEE TKDE 2022.
  https://doi.org/10.1109/TKDE.2022.3159715
"""

import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEncoder:
    """
    Ordinal-encodes categorical columns and passes numeric columns through.
    Fitted on clean data; applied to clean + dirty rows consistently.
    Unknown categories at transform time are mapped to -1 (treated as rare).
    """

    def __init__(self):
        self._cat_cols   = []
        self._num_cols   = []
        self._encoder    = None
        self._all_cols   = []

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        self._all_cols = list(df.columns)
        self._cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._num_cols = [c for c in self._all_cols if c not in self._cat_cols]

        if self._cat_cols:
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            self._encoder.fit(df[self._cat_cols].astype(str))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self._all_cols)), dtype=float)
        col_idx = {c: i for i, c in enumerate(self._all_cols)}

        if self._num_cols:
            for col in self._num_cols:
                out[:, col_idx[col]] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if self._cat_cols and self._encoder is not None:
            cat_data = self._encoder.transform(df[self._cat_cols].astype(str))
            for j, col in enumerate(self._cat_cols):
                out[:, col_idx[col]] = cat_data[:, j]

        # Replace NaN with column median (computed from this batch)
        for j in range(out.shape[1]):
            col_vals = out[:, j]
            nan_mask = np.isnan(col_vals)
            if nan_mask.any():
                median = np.nanmedian(col_vals)
                out[nan_mask, j] = median if not np.isnan(median) else 0.0

        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# ECOD wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ECODDetector:
    """
    Wraps PyOD ECOD with ordinal encoding.
    Provides score_rows() which returns raw anomaly scores (higher = more anomalous).
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self._enc   = FeatureEncoder()
        self._model = ECOD(contamination=contamination)
        self._fitted = False

    def fit(self, clean: pd.DataFrame) -> "ECODDetector":
        X = self._enc.fit_transform(clean)
        self._model.fit(X)
        self._fitted = True
        return self

    def score_rows(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores for each row (higher = more anomalous)."""
        assert self._fitted, "Call fit() first."
        X = self._enc.transform(df)
        return self._model.decision_function(X)   # raw scores

    def score_batch(self, rows: list) -> np.ndarray:
        """Score a list of pd.Series in one batched call (avoids per-row overhead)."""
        df = pd.DataFrame(rows)
        return self.score_rows(df)


# ─────────────────────────────────────────────────────────────────────────────
# Leave-One-Out cell attribution  (batched per dirty row)
# ─────────────────────────────────────────────────────────────────────────────

def loo_cell_attribution(
    detector    : ECODDetector,
    clean_row   : pd.Series,
    dirty_row   : pd.Series,
    flagged_cols: list,
) -> dict:
    """
    Build a batch of (1 + len(flagged_cols)) rows, score them in one call:
      row 0          : full dirty row           -> score_dirty
      row 1 .. k     : dirty row with col_j     -> score_reverted[j]
    delta[col] = score_dirty - score_reverted[col]
    """
    batch_rows = [dirty_row]
    for col in flagged_cols:
        rev = dirty_row.copy()
        rev[col] = clean_row[col]
        batch_rows.append(rev)

    scores = detector.score_batch(batch_rows)
    score_dirty = scores[0]
    return {col: float(score_dirty - scores[i + 1])
            for i, col in enumerate(flagged_cols)}


# ─────────────────────────────────────────────────────────────────────────────
# Predict intent from delta scores
# ─────────────────────────────────────────────────────────────────────────────

def predict_intent(deltas: dict, threshold: float = 0.0) -> dict:
    """
    +1 (intentional)   if delta >  threshold  (cell drives the anomaly)
    -1 (unintentional) otherwise
    """
    return {col: (1 if d > threshold else -1) for col, d in deltas.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Full dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_loo_on_dataset(
    detector    : ECODDetector,
    clean       : pd.DataFrame,
    dirty       : pd.DataFrame,
    mask        : pd.DataFrame,   # +1=INT, -1=UNINT, 0=clean
    feature_cols: list,
    threshold   : float = 0.0,
    verbose     : bool  = True,
) -> pd.DataFrame:
    """
    Run batched LOO attribution on all dirty rows.
    Returns a DataFrame with columns:
      row_idx, col, y_true (+1/-1), y_pred (+1/-1), delta_score
    """
    records = []

    dirty_row_idx = mask.index[(mask != 0).any(axis=1)].tolist()
    iterator = tqdm(dirty_row_idx, desc="LOO attribution") if verbose else dirty_row_idx

    for i in iterator:
        flagged_cols = [c for c in feature_cols if mask.at[i, c] != 0]
        if not flagged_cols:
            continue

        # Use same-index clean row if available, else last clean row
        clean_row = clean.iloc[i][feature_cols] if i < len(clean) else clean.iloc[-1][feature_cols]
        dirty_row = dirty.iloc[i][feature_cols]

        deltas = loo_cell_attribution(detector, clean_row, dirty_row, flagged_cols)
        preds  = predict_intent(deltas, threshold)

        for col in flagged_cols:
            records.append(dict(
                row_idx     = i,
                col         = col,
                y_true      = int(mask.at[i, col]),
                y_pred      = preds[col],
                delta_score = deltas[col],
            ))

    return pd.DataFrame(records)
