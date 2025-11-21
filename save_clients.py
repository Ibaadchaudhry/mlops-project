# save_clients.py
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_ingestion import load_federated_data


def main():

    # Load raw per-client data
    raw = load_federated_data()

    # --- 1. Build global one-hot encoder on ALL clients ---
    all_frames = []

    for cid, ds in raw.items():
        temp = ds["X_train_raw"].copy()
        temp["__cid__"] = cid
        all_frames.append(temp)

    global_df = pd.concat(all_frames, axis=0)
    global_ohe = pd.get_dummies(global_df.drop(columns=["__cid__"]), drop_first=False)
    global_cols = list(global_ohe.columns)

    # --- 2. Process each client with aligned columns + normalization ---
    aligned = {}

    for cid, ds in raw.items():
        # ds already contains X_train_raw and X_test_raw as DataFrames from data_ingestion
        Xtr = pd.get_dummies(ds["X_train_raw"], drop_first=False)
        Xte = pd.get_dummies(ds["X_test_raw"], drop_first=False)

        Xtr = Xtr.reindex(columns=global_cols, fill_value=0)
        Xte = Xte.reindex(columns=global_cols, fill_value=0)

        scaler = StandardScaler()
        Xtr_norm = scaler.fit_transform(Xtr)
        Xte_norm = scaler.transform(Xte)

        aligned[cid] = {
            # include RAW aligned DataFrames so server can run drift checks
            "X_train_raw": Xtr,
            "X_test_raw": Xte,
            "X_train_norm": Xtr_norm.astype("float32"),
            "X_test_norm": Xte_norm.astype("float32"),
            "y_train": ds["y_train"],
            "y_test": ds["y_test"],
            "feature_columns": global_cols,
        }

    with open("client_datasets.pkl", "wb") as f:
        pickle.dump(aligned, f)

    print("Saved client_datasets.pkl with clients:", list(aligned.keys()))


if __name__ == "__main__":
    main()
