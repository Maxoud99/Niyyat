"""
Imputation-Based Correct Value Estimation for Intent Attribution
================================================================

This package estimates correct values for corrupted cells using MICE
(Multivariate Imputation by Chained Equations) with Random Forest models,
and produces confidence-weighted diagnostic features for downstream
intent classification (intentional vs. unintentional errors).

Dataset-agnostic: works with any CSV + mask pair following the contract:
  - Data file:  N rows × P columns (numerical + categorical)
  - Mask file:  N rows × P columns (0 = clean, 1 = intentional, -1 = unintentional)

Modules:
  - imputation_estimator: MICE-based imputation with dual confidence
  - diagnostic_features:  Confidence-weighted feature engineering (WRC, direction, etc.)
  - run_pipeline:         End-to-end pipeline orchestrator
"""

from .imputation_estimator import MICEImputer
from .diagnostic_features import DiagnosticFeatureExtractor
from .run_pipeline import ImputationPipeline
from .evaluate import ImputationEvaluator

__all__ = [
    "MICEImputer",
    "DiagnosticFeatureExtractor",
    "ImputationPipeline",
    "ImputationEvaluator",
]
