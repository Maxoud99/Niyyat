#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction helpers for the no-clustering ablation.

Identical logic to declarative/run_all_declarative.py's B / B+ feature
extraction — copied verbatim so this module has no import-time dependency
on a script-shaped file (run_all_declarative.py guards its imports with
sys.path tricks meant for __main__ use).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from attribution.heuristics.pipeline import AttributionPipeline


def extract_b_features(dirty: pd.DataFrame,
                        blind: pd.DataFrame,
                        cfg: dict) -> Optional[pd.DataFrame]:
    """12 structural heuristic features (Scenario B) per erroneous cell."""
    try:
        blind_aligned = blind.reindex(columns=dirty.columns, fill_value=0)
        pipe = AttributionPipeline(
            target_col=cfg.get("target_col", None),
            codependent_pairs=cfg.get("codependent_pairs", []),
            sensitive_cols=cfg.get("sensitive_cols", []),
        )
        pipe.fit(dirty, blind_aligned)
        feat_df = pipe.compute_features(dirty, blind_aligned)
        feat_df = feat_df.reset_index()
        feat_df = feat_df.rename(columns={"col_name": "column_name"})
        print(f"  [B features] shape: {feat_df.shape}")
        return feat_df
    except Exception as e:
        print(f"  [warning] B feature extraction failed: {e}")
        return None


def extract_statistical_features(dirty: pd.DataFrame,
                                  clean: pd.DataFrame,
                                  mask: pd.DataFrame) -> Optional[pd.DataFrame]:
    """10 statistical features (the B+ addition) per erroneous cell."""
    try:
        clean = clean.reset_index(drop=True)
        dirty = dirty.reset_index(drop=True)
        mask  = mask.reset_index(drop=True)

        if len(dirty) == 3 * len(clean):
            clean = clean.loc[clean.index.repeat(3)].reset_index(drop=True)

        shared_cols = [c for c in dirty.columns if c in clean.columns]
        clean = clean[shared_cols].reset_index(drop=True)
        dirty = dirty[shared_cols].reset_index(drop=True)

        all_cols = [c for c in mask.columns if c in shared_cols]
        col_enc  = {c: i for i, c in enumerate(all_cols)}

        mask_r = mask[all_cols].reset_index(drop=True)
        mask_long = mask_r.copy()
        mask_long.index.name = "row_idx"
        mask_long = mask_long.reset_index().melt(
            id_vars="row_idx", var_name="col_name", value_name="intent"
        )
        mask_long = mask_long[mask_long["intent"] != 0].reset_index(drop=True)

        if len(mask_long) == 0:
            return None

        row_idxs  = mask_long["row_idx"].values
        col_names = mask_long["col_name"].values

        def _get_vals(df, rows, cols):
            arr     = df.to_numpy()
            col_map = {c: j for j, c in enumerate(df.columns)}
            cidxs   = [col_map.get(c, 0) for c in cols]
            return arr[rows, cidxs]

        orig_vals = _get_vals(clean, row_idxs, col_names)
        new_vals  = _get_vals(dirty, row_idxs, col_names)

        def _to_num(arr):
            out = np.zeros(len(arr), dtype=float)
            for i, v in enumerate(arr):
                try:
                    out[i] = float(v) if pd.notna(v) else 0.0
                except (ValueError, TypeError):
                    out[i] = 0.0
            return out

        orig_num  = _to_num(orig_vals)
        new_num   = _to_num(new_vals)
        diff      = new_num - orig_num
        magnitude = np.abs(diff)
        for i, (ov, nv) in enumerate(zip(orig_vals, new_vals)):
            if str(ov) != str(nv) and magnitude[i] == 0.0:
                magnitude[i] = 1.0

        feat_enc = np.array([col_enc.get(c, 0) for c in col_names], dtype=float)

        per_col_le: Dict = {}
        for col in all_cols:
            if col not in dirty.columns:
                continue
            orig_s = clean[col].astype(str).tolist() if col in clean.columns else []
            new_s  = dirty[col].astype(str).tolist()
            per_col_le[col] = LabelEncoder().fit(sorted(set(orig_s + new_s)))

        orig_enc = np.full(len(mask_long), -1, dtype=float)
        new_enc  = np.full(len(mask_long), -1, dtype=float)
        for i, (col, ov, nv) in enumerate(zip(col_names, orig_vals, new_vals)):
            le = per_col_le.get(col)
            if le is None:
                continue
            try:
                orig_enc[i] = float(le.transform([str(ov)])[0])
            except ValueError:
                pass
            try:
                new_enc[i] = float(le.transform([str(nv)])[0])
            except ValueError:
                pass

        stat_df = pd.DataFrame({
            "row_idx":                     row_idxs,
            "column_name":                 col_names,
            "stat_change_magnitude":       magnitude,
            "stat_relative_change":        magnitude / (np.abs(orig_num) + 1.0),
            "stat_change_direction":       np.sign(diff),
            "stat_original_magnitude":     np.abs(orig_num),
            "stat_new_magnitude":          np.abs(new_num),
            "stat_original_log":           np.log1p(np.abs(orig_num)),
            "stat_new_log":                np.log1p(np.abs(new_num)),
            "stat_feature_name_encoded":   feat_enc,
            "stat_original_value_encoded": orig_enc,
            "stat_new_value_encoded":      new_enc,
        })
        print(f"  [Statistical features] shape: {stat_df.shape}")
        return stat_df

    except Exception as e:
        print(f"  [warning] Statistical feature extraction failed: {e}")
        return None


def combine_b_and_stat(b_feats: pd.DataFrame,
                        stat_feats: pd.DataFrame) -> pd.DataFrame:
    """Merge B (13) and statistical (10) features → 23 features."""
    merged = b_feats.merge(stat_feats, on=["row_idx", "column_name"], how="inner")
    print(f"  [B+ combined] shape: {merged.shape}")
    return merged


def attach_labels(features_df: pd.DataFrame, gt_mask: pd.DataFrame) -> pd.DataFrame:
    """Vectorized label attach: melt gt_mask (+1/-1/0) and inner-join on (row_idx, column_name)."""
    melted = gt_mask.copy()
    melted.index.name = "row_idx"
    melted = melted.reset_index().melt(
        id_vars="row_idx", var_name="column_name", value_name="intent_label"
    )
    melted = melted[melted["intent_label"] != 0].copy()
    melted["intent_label"] = melted["intent_label"].astype(int)
    merged = features_df.merge(melted, on=["row_idx", "column_name"], how="inner")
    return merged
