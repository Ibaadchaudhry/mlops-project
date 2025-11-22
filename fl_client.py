# fl_client.py
import flwr as fl
import numpy as np
import torch
from model import TabularMLP
from train_utils import train_local, evaluate_model

class FLClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementation compatible with newer Flower that passes config to methods.
    """

    def __init__(self, cid: int, client_data: dict, input_dim: int, device=None):
        self.cid = cid
        self.data = client_data
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TabularMLP(input_dim)

    # New API: accept config parameter (may be None)
    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, _), param in zip(state_dict.items(), parameters):
            state_dict[k] = torch.from_numpy(param)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config=None):
        # Set global weights
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 1)) if config else 1

        X_full = self.data["X_train_norm"]
        y_full = self.data["y_train"]

        total = len(y_full)

        # --- Sampling per round ---
        sample_fraction = float(config.get("sample_fraction", 0.5))  # 50% by default
        sample_size = max(1, int(total * sample_fraction))

        indices = np.random.choice(total, size=sample_size, replace=False)
        X = X_full[indices]
        y = y_full[indices].astype(np.float32)

        print(f"[Client {self.cid}] Round sampling: {sample_size}/{total}")

        # Train with sampled data
        train_local(self.model, X, y, epochs=local_epochs)

        return self.get_parameters(), sample_size, {}


    # fl_client.py (only changed evaluate method shown)
    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        X_val = self.data["X_test_norm"]
        y_val = self.data["y_test"].astype(np.float32)
        metrics = evaluate_model(self.model, X_val, y_val)
        # ensure serializable metrics (no None)
        if metrics.get("auc") is None:
            metrics["auc"] = -1.0
        if metrics.get("accuracy") is None:
            metrics["accuracy"] = -1.0
        # return (loss, num_examples, metrics). Use 1 - AUC as proxy loss if AUC valid
        loss = 1.0 - metrics["auc"] if metrics["auc"] >= 0.0 else 1.0
        # Also include loss inside the metrics dict for redundancy so server
        # can pick it up whether Flower sends attributes or tuple positions.
        try:
            metrics["loss"] = float(loss)
        except Exception:
            metrics["loss"] = None
        return float(loss), len(y_val), metrics

