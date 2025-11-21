# file: drift_detector.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def psi(expected, actual, buckets=10):
    """Population Stability Index for 1D arrays."""
    eps = 1e-8
    breakpoints = np.linspace(0, 100, buckets+1)
    expected_percents = np.percentile(expected, breakpoints)
    act_percents = np.percentile(actual, breakpoints)
    # Instead use hist
    expected_counts, _ = np.histogram(expected, bins=expected_percents)
    actual_counts, _ = np.histogram(actual, bins=expected_percents)
    expected_ratios = expected_counts / (expected_counts.sum() + eps)
    actual_ratios = actual_counts / (actual_counts.sum() + eps)
    psi_val = np.sum((expected_ratios - actual_ratios) * np.log((expected_ratios + eps) / (actual_ratios + eps)))
    return float(psi_val)

def detect_drift_featurewise(baseline_df: pd.DataFrame, current_df: pd.DataFrame,
                             psi_threshold=0.2, ks_pval_threshold=0.01):
    """
    Returns dict {feature: {"psi": val, "ks_pvalue": p, "drift_flag": True/False}}
    """
    results = {}
    for col in baseline_df.columns:
        base = baseline_df[col].values
        cur = current_df[col].values
        try:
            psi_val = psi(base, cur)
            ks_stat, ks_p = ks_2samp(base, cur)
        except Exception:
            psi_val = float("nan")
            ks_p = 1.0

        flag = False
        if not np.isnan(psi_val) and psi_val > psi_threshold:
            flag = True
        if ks_p < ks_pval_threshold:
            flag = True

        results[col] = {"psi": float(psi_val), "ks_pvalue": float(ks_p), "drift_flag": flag}
    return results
