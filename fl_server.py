# fl_server.py
import flwr as fl
import multiprocessing
import pickle
from model import TabularMLP
from fl_client import FLClient
from drift_detector import detect_drift_featurewise
import pandas as pd
import os
import json
import numpy as np
import numbers

def aggregate_metrics(results):
    """Aggregate metrics across clients (weighted by number of examples)."""
    total_examples = sum(num_examples for num_examples, _ in results)
    avg_metrics = {}
    for num_examples, metrics in results:
        for k, v in metrics.items():
            avg_metrics[k] = avg_metrics.get(k, 0) + v * num_examples
    for k in avg_metrics:
        avg_metrics[k] /= total_examples
    return avg_metrics



def on_fit_config_fn(rnd):
    return {
        "local_epochs": 5,
        "sample_fraction": 0.3 + (0.05 * rnd)  # increases each round
    }


# ---- Drift-aware strategy ----
class DriftAwareStrategy(fl.server.strategy.FedAvg):
    """
    Subclass FedAvg to run drift detection after evaluate aggregation.
    """

    def __init__(self, *args, baseline_client_id=0, output_dir="drift_reports", **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_client_id = baseline_client_id
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # --- Create models folder ---
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Try to preload client raw data if available
        try:
            with open("client_datasets.pkl", "rb") as f:
                self._client_datasets = pickle.load(f)
        except Exception:
            self._client_datasets = None

    def aggregate_fit(self, rnd, results, failures):
        # Get aggregated parameters
        aggregated, metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            print(f"[Server] Saving global model after round {rnd} ...")

            from flwr.common import parameters_to_ndarrays
            import torch
            from model import TabularMLP

            # Convert FL parameters → list of numpy arrays
            weights = parameters_to_ndarrays(aggregated)

            # Create model with correct input dim
            first_w = weights[0]
            input_dim = first_w.shape[1]
            model = TabularMLP(input_dim=input_dim)

            # Convert numpy arrays → tensors
            state_dict = {
                k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)
            }
            model.load_state_dict(state_dict)

            # Save inside models/
            save_path = os.path.join(self.model_dir, f"global_model_round_{rnd}.pt")
            torch.save(model.state_dict(), save_path)

            print(f"[Server] Saved: {save_path}")

        return aggregated, metrics
        

    def aggregate_evaluate(self, rnd, results, failures):
        # First call parent to get aggregated metrics
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        # --- Aggregate evaluation metrics into a single dict (robust to Flower result shape) ---
        pairs = []
        for r in results:
            # Common shapes:
            # 1) (num_examples, metrics)
            # 2) (client, (loss, num_examples, metrics))  [current Flower evaluate returns loss first]
            # 3) (client, (num_examples, metrics))  [older variants]
            try:
                # Shape (num_examples, metrics)
                if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], (int, np.integer)):
                    num_examples = int(r[0])
                    metrics = r[1] or {}
                    pairs.append((num_examples, metrics))
                    continue
            except Exception:
                pass

            try:
                client, res = r
                # If res is a tuple
                if isinstance(res, tuple):
                    # Common expected shapes:
                    # (loss, num_examples, metrics)
                    # (num_examples, metrics)
                    # Be robust to numpy/scalar types
                    if len(res) >= 3 and isinstance(res[0], numbers.Number) and isinstance(res[1], (int, np.integer)) and isinstance(res[-1], dict):
                        loss_val = float(res[0])
                        num_examples = int(res[1])
                        metrics = dict(res[-1] or {})
                        metrics["loss"] = loss_val
                        pairs.append((num_examples, metrics))
                        continue
                    if len(res) == 3 and isinstance(res[0], numbers.Number) and isinstance(res[1], numbers.Number) and isinstance(res[2], dict):
                        # fallback pattern
                        loss_val = float(res[0])
                        num_examples = int(res[1])
                        metrics = dict(res[2] or {})
                        metrics["loss"] = loss_val
                        pairs.append((num_examples, metrics))
                        continue
                    if len(res) == 2 and isinstance(res[0], (int, np.integer)) and isinstance(res[1], dict):
                        num_examples = int(res[0])
                        metrics = dict(res[1] or {})
                        pairs.append((num_examples, metrics))
                        continue
                # Fallback: try attributes (e.g., EvaluateRes objects)
                num_examples = getattr(res, "num_examples", None)
                metrics = getattr(res, "metrics", None) or {}
                # Try to extract loss attribute if present
                loss_attr = getattr(res, "loss", None)
                try:
                    metrics = dict(metrics)
                except Exception:
                    metrics = {}
                if loss_attr is not None and isinstance(loss_attr, numbers.Number):
                    metrics["loss"] = float(loss_attr)
                if num_examples is not None:
                    pairs.append((int(num_examples), metrics))
            except Exception:
                # skip malformed result
                continue

        if pairs:
            try:
                avg_metrics = aggregate_metrics(pairs)
            except Exception:
                avg_metrics = {}
        else:
            avg_metrics = {}

        # (debug prints removed) - parsing is now robust to EvaluateRes shapes

        # Persist aggregated metrics per round into models/metrics_history.json
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            mh_path = os.path.join(self.model_dir, "metrics_history.json")
            history = []
            if os.path.exists(mh_path):
                try:
                    with open(mh_path, "r", encoding="utf-8") as fh:
                        history = json.load(fh) or []
                except Exception:
                    history = []
            history.append({"round": int(rnd), "metrics": avg_metrics})
            with open(mh_path, "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)
            print(f"[Server] Appended metrics for round {rnd} to {mh_path}")
        except Exception as e:
            print(f"[Server] Failed to persist metrics history: {e}")

        # Run drift detection if we have client raw data
        if self._client_datasets is None:
            try:
                with open("client_datasets.pkl", "rb") as f:
                    self._client_datasets = pickle.load(f)
            except Exception:
                print("[DriftAwareStrategy] No client_datasets.pkl found; skipping drift detection.")
                return aggregated

        clients = self._client_datasets
        baseline_id = self.baseline_client_id

        if baseline_id not in clients:
            print(f"[DriftAwareStrategy] Baseline client {baseline_id} not found in client_datasets; skipping drift.")
            return aggregated

        baseline_df = clients[baseline_id].get("X_train_raw")
        if baseline_df is None:
            print("[DriftAwareStrategy] Baseline raw DataFrame missing; skipping drift.")
            return aggregated

        print(f"\n[DriftAwareStrategy] Round {rnd}: running drift detection against baseline client {baseline_id}")

        for cid, ds in clients.items():
            # skip baseline
            if cid == baseline_id:
                continue
            current_df = ds.get("X_train_raw")
            if current_df is None:
                print(f"[DriftAwareStrategy] Client {cid} has no X_train_raw; skipping.")
                continue

            # Ensure matching columns: reindex to baseline columns (baseline defines the feature space)
            try:
                cur_reindexed = current_df.reindex(columns=baseline_df.columns, fill_value=0)
            except Exception:
                cur_reindexed = current_df

            drift_results = detect_drift_featurewise(baseline_df=baseline_df, current_df=cur_reindexed)

            # Summary
            num_flagged = sum(1 for v in drift_results.values() if v["drift_flag"])
            print(f"[DriftAwareStrategy] Client {cid}: {num_flagged} features flagged for drift")

            # Save round-specific CSV
            df_out = pd.DataFrame(drift_results).T
            out_path = os.path.join(self.output_dir, f"drift_round_{rnd}_client_{cid}.csv")
            df_out.to_csv(out_path)
            print(f"[DriftAwareStrategy] Saved drift report: {out_path}")

        return aggregated
    


# ---- server starter (top-level) ----
def start_flower_server(num_rounds, num_clients):
    strategy = DriftAwareStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=on_fit_config_fn,
        baseline_client_id=0,
        output_dir="drift_reports",
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

# ---- client entry (top-level) ----
def start_client_process(cid, client_data, input_dim):
    client = FLClient(cid=cid, client_data=client_data, input_dim=input_dim)
    # convert to Flower client and start
    try:
        # Newer Flower API
        fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
    except Exception:
        # Fallback for older Flower versions
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

def main():
    # Load prepared client_datasets (created by save_clients.py)
    with open("client_datasets.pkl", "rb") as f:
        client_datasets = pickle.load(f)

    num_clients = len(client_datasets)
    rounds = 10

    # pick input dim
    any_ds = next(iter(client_datasets.values()))
    input_dim = any_ds["X_train_norm"].shape[1]

    # Start server process
    server_proc = multiprocessing.Process(target=start_flower_server, args=(rounds, num_clients))
    server_proc.start()

    # NOTE: small sleep may help clients find server (optional)
    import time
    time.sleep(1.0)

    # Start client processes
    client_procs = []
    for cid, ds in client_datasets.items():
        p = multiprocessing.Process(target=start_client_process, args=(cid, ds, input_dim))
        p.start()
        client_procs.append(p)

    # Wait for clients to finish
    for p in client_procs:
        p.join()

    # Wait for server
    server_proc.join()

if __name__ == "__main__":
    # Windows: ensure spawn method and top-level entry point
    multiprocessing.set_start_method("spawn")
    main()
