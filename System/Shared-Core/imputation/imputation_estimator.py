#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MICE-Based Imputation Estimator with Dual Confidence
=====================================================

Dataset-agnostic imputation module that:
1. Auto-detects numerical vs. categorical columns
2. Trains per-column Random Forest models on clean rows
3. Imputes corrupted cells using MICE (iterative) for multi-corruption rows
4. Extracts dual confidence signals: tree variance + OOB error

Input contract:
  - data_df:  pd.DataFrame with N rows × P feature columns
  - mask_df:  pd.DataFrame with N rows × P columns, values in {-1, 0, 1}
              (0 = clean, ±1 = corrupted)

Output:
  - Per corrupted cell: (imputed_value, sigma_tree, sigma_oob, confidence)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MICEImputer:
    """
    Modified MICE imputer for corrupted (not missing) data.

    Trains one RF model per column on clean rows, then imputes flagged
    corrupted cells. For rows with multiple corrupted cells, uses iterative
    MICE rounds so that each column's imputation benefits from the others.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees per RF model.
    max_depth : int or None, default=None
        Max tree depth. None = unlimited.
    max_rounds : int, default=5
        Maximum MICE iteration rounds for multi-corruption rows.
    convergence_tol : float, default=1e-3
        Stop iterating when relative change in estimates < tol.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_rounds: int = 5,
        convergence_tol: float = 1e-3,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_rounds = max_rounds
        self.convergence_tol = convergence_tol
        self.random_state = random_state
        self.verbose = verbose

        # Fitted state
        self.models_: Dict[str, object] = {}           # col_name -> RF model
        self.label_encoders_: Dict[str, LabelEncoder] = {}  # col_name -> encoder
        self.col_types_: Dict[str, str] = {}            # col_name -> "numerical" | "categorical"
        self.col_is_integer_: Dict[str, bool] = {}      # col_name -> True if source dtype is integer
        self.col_defaults_: Dict[str, object] = {}      # col_name -> median or mode
        self.oob_errors_: Dict[str, float] = {}         # col_name -> OOB error
        self.feature_columns_: List[str] = []           # ordered list of feature columns
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data_df: pd.DataFrame, mask_df: pd.DataFrame) -> "MICEImputer":
        """
        Train per-column RF models on clean rows.

        Parameters
        ----------
        data_df : pd.DataFrame, shape (N, P)
            Full dataset (clean + corrupted rows). Only feature columns
            (no 'is_erroneous' or index columns).
        mask_df : pd.DataFrame, shape (N, P)
            Mask with same columns as data_df. Values: 0=clean, ±1=corrupted.

        Returns
        -------
        self
        """
        self._validate_inputs(data_df, mask_df)
        self.feature_columns_ = list(data_df.columns)

        # Step 1: Detect column types
        self._detect_column_types(data_df)

        # Step 2: Identify clean rows (all mask values = 0)
        clean_mask = (mask_df == 0).all(axis=1)
        n_clean = clean_mask.sum()
        n_dirty = (~clean_mask).sum()

        if self.verbose:
            print(f"\n{'='*70}")
            print("MICE IMPUTER — FIT")
            print(f"{'='*70}")
            print(f"Dataset:       {data_df.shape[0]} rows × {data_df.shape[1]} columns")
            print(f"Clean rows:    {n_clean} ({100*n_clean/len(data_df):.1f}%) — training data")
            print(f"Dirty rows:    {n_dirty} ({100*n_dirty/len(data_df):.1f}%) — need imputation")
            print(f"Numerical:     {sum(1 for v in self.col_types_.values() if v == 'numerical')} columns")
            print(f"Categorical:   {sum(1 for v in self.col_types_.values() if v == 'categorical')} columns")

        if n_clean < 10:
            raise ValueError(
                f"Only {n_clean} clean rows found. Need at least 10 to train imputer."
            )

        data_clean = data_df[clean_mask].copy()

        # Step 3: Fit label encoders on clean data (categorical columns only)
        self._fit_label_encoders(data_clean)

        # Step 4: Encode clean data for training
        data_clean_encoded = self._encode_dataframe(data_clean)

        # Step 5: Train one RF model per column
        for col in self.feature_columns_:
            X_train = data_clean_encoded.drop(columns=[col])
            y_train = data_clean_encoded[col]

            if self.col_types_[col] == "numerical":
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)
            self.models_[col] = model

            # Extract OOB error
            oob_score = model.oob_score_  # R² for regressor, accuracy for classifier
            if self.col_types_[col] == "numerical":
                # OOB R² can be negative; convert to error = 1 - R²
                self.oob_errors_[col] = max(0.0, 1.0 - oob_score)
            else:
                # OOB accuracy; convert to error = 1 - accuracy
                self.oob_errors_[col] = max(0.0, 1.0 - oob_score)

            if self.verbose:
                print(f"  ✓ {col:30s}  type={self.col_types_[col]:12s}  "
                      f"OOB_score={oob_score:.4f}  OOB_error={self.oob_errors_[col]:.4f}")

        # Store column defaults (median for numerical, mode for categorical)
        for col in self.feature_columns_:
            if self.col_types_[col] == "numerical":
                self.col_defaults_[col] = data_clean[col].median()
            else:
                self.col_defaults_[col] = data_clean[col].mode().iloc[0]

        self.is_fitted_ = True

        if self.verbose:
            print(f"\n✓ Fitted {len(self.models_)} per-column RF models.")

        return self

    def impute(
        self,
        data_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        correct_data_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Impute corrupted cells and return results with confidence.

        Parameters
        ----------
        data_df : pd.DataFrame, shape (N, P)
            Full dataset with corrupted values still present.
        mask_df : pd.DataFrame, shape (N, P)
            Mask: 0=clean, ±1=corrupted.
        correct_data_df : pd.DataFrame, shape (N, P), optional
            Ground truth correct values aligned row-for-row with data_df.
            When provided, a ``correct_value`` column is added to the output
            so imputation quality can be measured per-cell.
            When None (default), ``correct_value`` is set to NaN for all rows.

        Returns
        -------
        results : pd.DataFrame
            One row per corrupted cell with columns:
            - row_idx:         index of the row in data_df
            - column:          name of the corrupted column
            - observed_value:  the corrupted value (x_tilde)
            - correct_value:   the ground-truth value (x_star); NaN if not provided
            - imputed_value:   the estimated correct value (x_hat_star)
            - intent_label:    ground truth from mask (1=intentional, -1=unintentional)
            - col_type:        "numerical" or "categorical"
            - sigma_tree:      std of per-tree predictions (row-specific uncertainty)
            - sigma_oob:       OOB error for this column (column-level uncertainty)
            - confidence:      1 / (1 + sigma_tree)
            - mice_rounds:     number of MICE rounds used for this row
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before impute().")

        self._validate_inputs(data_df, mask_df)

        # Validate optional correct_data_df
        if correct_data_df is not None:
            if correct_data_df.shape[0] != data_df.shape[0]:
                raise ValueError(
                    f"correct_data_df row count ({correct_data_df.shape[0]}) must match "
                    f"data_df row count ({data_df.shape[0]})."
                )
            # Align index
            correct_data_df = correct_data_df.copy()
            correct_data_df.index = data_df.index

        # Identify dirty rows
        dirty_mask = (mask_df != 0).any(axis=1)
        dirty_indices = data_df.index[dirty_mask].tolist()

        if self.verbose:
            n_dirty = len(dirty_indices)
            # Count corrupted cells per row
            corrupted_per_row = (mask_df.loc[dirty_mask] != 0).sum(axis=1)
            n_single = (corrupted_per_row == 1).sum()
            n_multi = (corrupted_per_row > 1).sum()
            total_cells = (mask_df != 0).sum().sum()
            print(f"\n{'='*70}")
            print("MICE IMPUTER — IMPUTE")
            print(f"{'='*70}")
            print(f"Dirty rows:    {n_dirty}")
            print(f"  Single-cell: {n_single} (direct prediction)")
            print(f"  Multi-cell:  {n_multi} (MICE iterative)")
            print(f"Total cells:   {total_cells}")
            if correct_data_df is not None:
                print(f"  Ground truth: provided ✓")
            else:
                print(f"  Ground truth: not provided (correct_value will be NaN)")

        results = []

        for row_idx in dirty_indices:
            row = data_df.loc[row_idx]
            mask_row = mask_df.loc[row_idx]
            corrupted_cols = [c for c in self.feature_columns_ if mask_row[c] != 0]

            if len(corrupted_cols) == 1:
                # Single-cell: direct prediction
                col = corrupted_cols[0]
                imputed_val, sigma_tree = self._predict_single_cell(row, col)
                correct_val = (
                    correct_data_df.loc[row_idx, col]
                    if correct_data_df is not None and col in correct_data_df.columns
                    else float("nan")
                )
                results.append({
                    "row_idx": row_idx,
                    "column": col,
                    "observed_value": row[col],
                    "correct_value": correct_val,
                    "imputed_value": imputed_val,
                    "intent_label": int(mask_row[col]),
                    "col_type": self.col_types_[col],
                    "sigma_tree": sigma_tree,
                    "sigma_oob": self.oob_errors_[col],
                    "confidence": 1.0 / (1.0 + sigma_tree),
                    "mice_rounds": 0,
                })
            else:
                # Multi-cell: MICE iterative
                cell_results = self._impute_multi_cell(
                    row, mask_row, corrupted_cols
                )
                # Inject correct_value into each cell result
                for cell in cell_results:
                    col = cell["column"]
                    cell["correct_value"] = (
                        correct_data_df.loc[row_idx, col]
                        if correct_data_df is not None and col in correct_data_df.columns
                        else float("nan")
                    )
                results.extend(cell_results)

        results_df = pd.DataFrame(results)

        # Reorder columns so correct_value is adjacent to observed_value
        ordered_cols = [
            "row_idx", "column", "observed_value", "correct_value", "imputed_value",
            "intent_label", "col_type", "sigma_tree", "sigma_oob", "confidence", "mice_rounds",
        ]
        results_df = results_df[[c for c in ordered_cols if c in results_df.columns]]

        if self.verbose and len(results_df) > 0:
            print(f"\n✓ Imputed {len(results_df)} corrupted cells across {len(dirty_indices)} rows.")
            print(f"  Mean confidence:  {results_df['confidence'].mean():.4f}")
            print(f"  Mean sigma_tree:  {results_df['sigma_tree'].mean():.4f}")
            print(f"  Mean sigma_oob:   {results_df['sigma_oob'].mean():.4f}")

        return results_df

    def get_oob_errors(self) -> Dict[str, float]:
        """Return per-column OOB errors."""
        return dict(self.oob_errors_)

    def get_column_types(self) -> Dict[str, str]:
        """Return detected column types."""
        return dict(self.col_types_)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _validate_inputs(self, data_df: pd.DataFrame, mask_df: pd.DataFrame):
        """Check that data and mask are compatible."""
        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("data_df must be a pandas DataFrame.")
        if not isinstance(mask_df, pd.DataFrame):
            raise TypeError("mask_df must be a pandas DataFrame.")
        if data_df.shape != mask_df.shape:
            raise ValueError(
                f"Shape mismatch: data_df={data_df.shape}, mask_df={mask_df.shape}"
            )
        if list(data_df.columns) != list(mask_df.columns):
            raise ValueError("Column mismatch between data_df and mask_df.")

    def _detect_column_types(self, data_df: pd.DataFrame):
        """Auto-detect numerical vs. categorical columns, and track integer dtypes."""
        for col in data_df.columns:
            if data_df[col].dtype in ("object", "category", "string"):
                self.col_types_[col] = "categorical"
                self.col_is_integer_[col] = False
            else:
                self.col_types_[col] = "numerical"
                # Track whether source column is integer so imputed values can be rounded
                self.col_is_integer_[col] = pd.api.types.is_integer_dtype(data_df[col])

    def _fit_label_encoders(self, data_clean: pd.DataFrame):
        """Fit a LabelEncoder per categorical column on clean data."""
        for col in self.feature_columns_:
            if self.col_types_[col] == "categorical":
                le = LabelEncoder()
                le.fit(data_clean[col].astype(str))
                self.label_encoders_[col] = le

    def _encode_value(self, col: str, value) -> object:
        """Encode a single value for a column (handles unseen labels)."""
        if self.col_types_[col] == "numerical":
            try:
                return float(value)
            except (ValueError, TypeError):
                return self.col_defaults_[col]
        else:
            le = self.label_encoders_[col]
            val_str = str(value)
            if val_str in le.classes_:
                return le.transform([val_str])[0]
            else:
                # Unseen category → encode as -1 (out-of-vocabulary)
                return -1

    def _decode_value(self, col: str, encoded_value) -> object:
        """Decode an encoded value back to original space."""
        if self.col_types_[col] == "numerical":
            return encoded_value
        else:
            le = self.label_encoders_[col]
            idx = int(round(encoded_value))
            if 0 <= idx < len(le.classes_):
                return le.classes_[idx]
            else:
                return le.classes_[0]  # fallback

    def _encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode an entire DataFrame (categorical → integer, numerical → float)."""
        encoded = pd.DataFrame(index=df.index)
        for col in self.feature_columns_:
            if self.col_types_[col] == "numerical":
                encoded[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                    self.col_defaults_.get(col, 0)
                )
            else:
                le = self.label_encoders_[col]
                vals = df[col].astype(str)
                encoded[col] = vals.apply(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
        return encoded

    def _encode_row(self, row: pd.Series) -> pd.Series:
        """Encode a single row."""
        encoded = pd.Series(index=self.feature_columns_, dtype=float)
        for col in self.feature_columns_:
            encoded[col] = self._encode_value(col, row[col])
        return encoded

    def _build_feature_vector(
        self, encoded_row: pd.Series, target_col: str
    ) -> np.ndarray:
        """Build input feature vector for predicting target_col (all cols except target)."""
        cols = [c for c in self.feature_columns_ if c != target_col]
        return encoded_row[cols].values.astype(float).reshape(1, -1)

    def _predict_with_trees(
        self, col: str, X: np.ndarray
    ) -> Tuple[object, float]:
        """
        Predict using per-tree predictions to get both estimate and sigma_tree.

        Returns
        -------
        imputed_value : in original space (decoded if categorical)
        sigma_tree : float, std of per-tree predictions
        """
        model = self.models_[col]

        # Get individual tree predictions
        tree_preds = np.array([
            tree.predict(X)[0] for tree in model.estimators_
        ])

        if self.col_types_[col] == "numerical":
            imputed_encoded = np.mean(tree_preds)
            sigma_tree = np.std(tree_preds, ddof=1) if len(tree_preds) > 1 else 0.0
            # Preserve integer dtype: round to nearest integer for integer source columns
            if self.col_is_integer_.get(col, False):
                imputed_value = int(round(imputed_encoded))
            else:
                imputed_value = float(imputed_encoded)
        else:
            # For classifier: mode of tree predictions, sigma = 1 - agreement ratio
            from collections import Counter
            counts = Counter(tree_preds.astype(int))
            most_common = counts.most_common(1)[0]
            imputed_encoded = most_common[0]
            agreement = most_common[1] / len(tree_preds)
            sigma_tree = 1.0 - agreement  # 0 = perfect agreement, 1 = no agreement
            imputed_value = self._decode_value(col, imputed_encoded)

        return imputed_value, sigma_tree

    def _predict_single_cell(
        self, row: pd.Series, col: str
    ) -> Tuple[object, float]:
        """Predict a single corrupted cell (row has only one corruption)."""
        encoded_row = self._encode_row(row)
        X = self._build_feature_vector(encoded_row, col)
        return self._predict_with_trees(col, X)

    def _impute_multi_cell(
        self, row: pd.Series, mask_row: pd.Series, corrupted_cols: List[str]
    ) -> List[dict]:
        """
        MICE iterative imputation for a row with multiple corrupted cells.

        Initialize corrupted cells with column default (median/mode),
        then iteratively re-impute each corrupted column using the latest
        estimates for other corrupted columns.
        """
        # Initialize working copy of the row (encoded)
        encoded_row = self._encode_row(row)

        # Set corrupted cells to column defaults as initial estimates
        for col in corrupted_cols:
            if self.col_types_[col] == "numerical":
                encoded_row[col] = float(self.col_defaults_[col])
            else:
                encoded_row[col] = self._encode_value(col, self.col_defaults_[col])

        # Store previous round estimates for convergence check
        prev_estimates = {col: encoded_row[col] for col in corrupted_cols}

        n_rounds = 0
        for round_num in range(1, self.max_rounds + 1):
            n_rounds = round_num

            for col in corrupted_cols:
                X = self._build_feature_vector(encoded_row, col)
                model = self.models_[col]

                if self.col_types_[col] == "numerical":
                    tree_preds = np.array([
                        t.predict(X)[0] for t in model.estimators_
                    ])
                    encoded_row[col] = np.mean(tree_preds)
                else:
                    tree_preds = np.array([
                        t.predict(X)[0] for t in model.estimators_
                    ])
                    from collections import Counter
                    counts = Counter(tree_preds.astype(int))
                    encoded_row[col] = counts.most_common(1)[0][0]

            # Check convergence
            converged = True
            for col in corrupted_cols:
                old = prev_estimates[col]
                new = encoded_row[col]
                if self.col_types_[col] == "numerical":
                    denom = max(abs(old), abs(new), 1e-10)
                    if abs(new - old) / denom > self.convergence_tol:
                        converged = False
                else:
                    if new != old:
                        converged = False
                prev_estimates[col] = new

            if converged:
                break

        # Final pass: get per-tree predictions for sigma_tree
        cell_results = []
        for col in corrupted_cols:
            X = self._build_feature_vector(encoded_row, col)
            imputed_value, sigma_tree = self._predict_with_trees(col, X)

            cell_results.append({
                "row_idx": row.name,
                "column": col,
                "observed_value": row[col],
                "imputed_value": imputed_value,
                "intent_label": int(mask_row[col]),
                "col_type": self.col_types_[col],
                "sigma_tree": sigma_tree,
                "sigma_oob": self.oob_errors_[col],
                "confidence": 1.0 / (1.0 + sigma_tree),
                "mice_rounds": n_rounds,
            })

        return cell_results
