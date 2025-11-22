# train_utils.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

def train_local(model, X, y, epochs=5, batch_size=64, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, y, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        preds = model(xb).cpu().numpy()
    auc = roc_auc_score(y, preds) if len(np.unique(y)) > 1 else float("nan")
    acc = accuracy_score(y, (preds >= 0.5).astype(int))
    return {"auc": float(auc) if not np.isnan(auc) else None, "accuracy": float(acc)}
