# drift_monitor.py

import os
import pickle
import pandas as pd

from drift_detector import detect_drift_featurewise


BASELINE_DIR = "drift_baselines"
os.makedirs(BASELINE_DIR, exist_ok=True)


# -------------------------------------------------------
# Save baseline RAW feature distributions (initial round)
# -------------------------------------------------------
def save_baseline(client_id: int, X_raw: pd.DataFrame):
    """
    Save RAW training features for baseline drift detection.
    X_raw must be the **raw pre-normalized dataframe**.
    """
    path = f"{BASELINE_DIR}/baseline_client_{client_id}.pkl"

    with open(path, "wb") as f:
        pickle.dump(X_raw, f)

    print(f"[Drift] Baseline saved for client {client_id} at {path}")


# -------------------------------------------------------
# Load baseline RAW features
# -------------------------------------------------------
def load_baseline(client_id: int):
    path = f"{BASELINE_DIR}/baseline_client_{client_id}.pkl"
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------
# Perform drift detection for a client
# -------------------------------------------------------
def check_drift(client_id: int, current_raw_df: pd.DataFrame,
                psi_threshold: float = 0.2,
                ks_threshold: float = 0.01):
    """
    Compare current raw features vs stored baseline.
    Returns per-feature drift results.

    current_raw_df = X_train_raw or X_test_raw (raw dataframe)
    """
    baseline = load_baseline(client_id)

    if baseline is None:
        print(f"[Drift] No baseline found for client {client_id}. Creating baseline now.")
        save_baseline(client_id, current_raw_df)
        return None  # No drift computed in first round

    # Align columns for safe comparison
    common_cols = list(baseline.columns.intersection(current_raw_df.columns))
    base_aligned = baseline[common_cols]
    curr_aligned = current_raw_df[common_cols]

    # Run actual drift detection
    results = detect_drift_featurewise(
        baseline_df=base_aligned,
        current_df=curr_aligned,
        psi_threshold=psi_threshold,
        ks_pval_threshold=ks_threshold,
    )

    # Count drift flags
    drifted = [c for c, r in results.items() if r["drift_flag"]]

    if drifted:
        print(f"[⚠ Drift Detected] Client {client_id}: {len(drifted)} features drifted.")
    else:
        print(f"[OK] No drift detected for client {client_id}.")

    return results


# -------------------------------------------------------
# Utility: Pretty drift summary
# -------------------------------------------------------
def summarize_drift(results: dict):
    """
    Helper function to print human-readable drift report.
    """
    if results is None:
        print("No drift results (likely first communication round).")
        return

    print("\n=== DRIFT SUMMARY ===")
    for col, r in results.items():
        flag = "⚠" if r["drift_flag"] else "OK"
        print(f"{col:30} | PSI={r['psi']:.4f} | KS p={r['ks_pvalue']:.4f} | {flag}")
    print("======================\n")
