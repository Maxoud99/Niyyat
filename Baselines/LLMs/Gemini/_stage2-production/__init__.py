"""
Stage 2 Cell-Level Attribution Detectors
-----------------------------------------
Given a record known to be erroneous, identify which specific cells contain errors.

Available detectors:
- gemini_attributor: LLM-based attribution using Google Gemini
- ml_attributor: Supervised ML-based attribution (requires training labels)
- statistical_attributor: Unsupervised statistical methods (local comparison, deviation)
"""

from .gemini_attributor import GeminiCellAttributor

__all__ = ['GeminiCellAttributor']
