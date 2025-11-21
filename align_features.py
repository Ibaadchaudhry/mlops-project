# align_features.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def align_client_feature_spaces(client_datasets):
    """
    client_datasets: dict[client_id] = {
        "X_train_raw": pd.DataFrame,
        "X_test_raw": pd.DataFrame,
        "X_train_norm": np.ndarray,   # optional (will be recomputed)
        "X_test_norm": np.ndarray,
        "y_train": np.ndarray,
        "y_test": np.ndarray,
    }
    Returns: same structure but with X_*_raw reindexed to global union columns,
             and X_*_norm recomputed client-side using StandardScaler fit on local train.
    """
    all_columns = set()
    for cid, ds in client_datasets.items():
        all_columns.update(ds["X_train_raw"].columns)

    all_columns = sorted(all_columns)

    aligned = {}
    for cid, ds in client_datasets.items():
        Xtr_raw = ds["X_train_raw"].reindex(columns=all_columns, fill_value=0)
        Xte_raw = ds["X_test_raw"].reindex(columns=all_columns, fill_value=0)

        scaler = StandardScaler()
        Xtr_norm = scaler.fit_transform(Xtr_raw)
        Xte_norm = scaler.transform(Xte_raw)

        aligned[cid] = {
            "X_train_raw": Xtr_raw,
            "X_test_raw": Xte_raw,
            "X_train_norm": Xtr_norm,
            "X_test_norm": Xte_norm,
            "y_train": ds["y_train"],
            "y_test": ds["y_test"],
            "feature_columns": all_columns,
        }

    return aligned
