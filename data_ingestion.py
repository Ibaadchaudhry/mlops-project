"""
Federated Data Ingestion System using flwr-datasets
---------------------------------------------------

- Loads Adult Census Income dataset
- Splits into 80-20 train/test per client
- Applies Dirichlet partitioning to create 3 nodes
- Handles missing values
- Generates RAW version (for drift detection)
- Generates NORMALIZED version (for training)
"""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

NUM_CLIENTS = 3
DIRICHLET_ALPHA = 0.5


def load_federated_data(num_clients: int = NUM_CLIENTS):

    fds = FederatedDataset(
        dataset="scikit-learn/adult-census-income",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=num_clients,
                alpha=DIRICHLET_ALPHA,
                seed=42,
                partition_by="income"
            )
        },
    )

    label_encoder = LabelEncoder()

    client_datasets = {}

    for client_id in range(num_clients):
        partition = fds.load_partition(client_id, "train")
        df = partition.to_pandas()

        # Encode labels BEFORE splitting
        df["income"] = label_encoder.fit_transform(df["income"])

        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        X_train = train_df.drop(columns=["income"])
        y_train = train_df["income"].to_numpy()

        X_test = test_df.drop(columns=["income"])
        y_test = test_df["income"].to_numpy()

        # Identify column types
        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = X_train.select_dtypes(include=["object"]).columns

        # Missing-value imputation
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])

        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode RAW version
        X_train_raw = pd.get_dummies(X_train, drop_first=True)
        X_test_raw = pd.get_dummies(X_test, drop_first=True)

        # Align test columns
        X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)

        # NORMALIZED version
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_raw)
        X_test_norm = scaler.transform(X_test_raw)

        client_datasets[client_id] = {
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
            "X_train_norm": X_train_norm.astype(np.float32),
            "X_test_norm": X_test_norm.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "y_test": y_test.astype(np.float32),
        }

    return client_datasets


if __name__ == "__main__":
    data = load_federated_data()
    print("\nData ingestion complete!\n")
    for cid, ds in data.items():
        print(f"Client {cid}: Train={len(ds['y_train'])}, Test={len(ds['y_test'])}")
