# TICKET 000 — Project Setup (implement this FIRST, before any H ticket)

## Goal
Create the shared infrastructure that all heuristic tickets depend on.
Tickets H1–H8 cannot be implemented without this.

## Location
Create a new folder:
```
/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/heuristics/
```

## Files to create

### 1. `heuristics/__init__.py`
Empty for now. Will be populated as heuristics are added.

### 2. `heuristics/base.py`
The abstract base class ALL heuristics inherit from.

```python
from abc import ABC, abstractmethod
import pandas as pd


class BaseHeuristic(ABC):
    """
    Abstract base class for all intent attribution heuristics.

    Design contract
    ---------------
    - fit()     : learns column statistics from the dirty dataset using ONLY
                  clean cells (mask_df == 0). Must set self.is_fitted = True.
    - compute() : returns one row per erroneous cell (mask_df == 1) as a
                  DataFrame with columns [row_idx, col_name, <heuristic features>].
    - All heuristics are DATASET AGNOSTIC: no hardcoded column names or paths.
      Anything dataset-specific is passed as an argument.
    """

    def __init__(self):
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame, **kwargs) -> "BaseHeuristic":
        """
        Learn column statistics from dirty_df using only clean cells (mask == 0).

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset (contains errors, but no labels).
        mask_df : pd.DataFrame
            Binary mask aligned with dirty_df. 0 = clean cell, 1 = erroneous cell.
            Same shape and column names as dirty_df.
        **kwargs
            Heuristic-specific configuration (e.g. col_types override).

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def compute(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute heuristic features for every erroneous cell (mask_df == 1).

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as dirty_df, values in {0, 1})

        Returns
        -------
        pd.DataFrame with columns:
            row_idx  (int)  : row index of the erroneous cell
            col_name (str)  : column name of the erroneous cell
            <heuristic-specific feature columns>
        One row per erroneous cell. Rows where mask == 0 are excluded.
        """
        ...

    def fit_compute(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Convenience: fit then compute in one call."""
        return self.fit(dirty_df, mask_df, **kwargs).compute(dirty_df, mask_df)

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}.fit() must be called before compute()."
            )

    @staticmethod
    def _get_error_positions(mask_df: pd.DataFrame):
        """Return list of (row_idx, col_name) for all erroneous cells."""
        positions = []
        for col in mask_df.columns:
            for row_idx in mask_df.index[mask_df[col] == 1]:
                positions.append((row_idx, col))
        return positions
```

## Invariants that ALL tickets must respect

1. `mask_df` has the same shape and column names as `dirty_df`. Values are 0 or 1.
2. `fit()` uses ONLY rows/cells where `mask_df == 0` to learn statistics.
3. `compute()` returns ONLY rows where `mask_df == 1`.
4. Output DataFrame always has `row_idx` (int) and `col_name` (str) as first two columns.
5. No hardcoded column names anywhere in any heuristic file.
6. Dependencies: only `numpy`, `pandas`, `scikit-learn` (all already installed).
