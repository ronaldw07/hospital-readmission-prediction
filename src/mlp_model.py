"""
PyTorch MLP classifier with sklearn-compatible interface.

Wraps a two-hidden-layer feedforward neural network so it can be used
alongside sklearn/XGBoost/LightGBM models in the evaluation pipeline.
Includes StandardScaler preprocessing and BCEWithLogitsLoss with
pos_weight to handle the class imbalance in the readmission dataset.

Author: Ronald Wen
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _MLP(nn.Module):
    """Two-hidden-layer feedforward network for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MLPClassifier:
    """Sklearn-compatible wrapper around a PyTorch MLP for binary classification.

    Fits a StandardScaler on training features before passing them to the
    network.  predict_proba returns a (n, 2) array matching sklearn convention
    so the model integrates seamlessly with the existing evaluation pipeline.

    Args:
        hidden_dims: Sizes of hidden layers.
        dropout: Dropout probability applied after each hidden layer.
        epochs: Number of full passes over the training set.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        pos_weight: Weight for positive class in BCEWithLogitsLoss (handles
            class imbalance — set to n_neg / n_pos).
        random_state: Seed for reproducibility.

    Author: Ronald Wen
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 512,
        pos_weight: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.net_: _MLP | None = None
        self.device_ = torch.device('cpu')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPClassifier:
        """Fit scaler and train the MLP on (X, y)."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(np.array(y), dtype=torch.float32)

        input_dim = X_t.shape[1]
        self.net_ = _MLP(input_dim, self.hidden_dims, self.dropout)

        pw = torch.tensor([self.pos_weight], dtype=torch.float32) if self.pos_weight else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net_.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net_(xb), yb)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}  loss={loss.item():.4f}")

        self.net_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) probability array matching sklearn convention."""
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.net_(X_t)
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions at 0.5 threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
