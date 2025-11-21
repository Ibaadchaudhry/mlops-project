# model.py
import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))
