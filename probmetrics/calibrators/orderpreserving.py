import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import Calibrator


class _FCModel(nn.Module):
    """
    From https://github.com/AmirooR/IntraOrderPreservingCalibration

    Standard Fully Connected base model as specified by 'config.json'.
    Maps (n_classes -> hidden -> n_classes).
    """

    def __init__(self, in_features: int, hidden_sizes: list, out_features: int):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _OrderPreservingModel(nn.Module):
    """
    From https://github.com/AmirooR/IntraOrderPreservingCalibration

    Internal PyTorch module adapting the provided OrderPreservingModel.
    """

    def __init__(
        self,
        base_model: nn.Module,
        invariant: bool = True,
        residual: bool = False,
        num_classes: int = 10,
        m_activation=F.softplus,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.invariant = invariant
        self.m_activation = m_activation
        self.residual = residual

    def compute_u(self, sorted_logits: torch.Tensor) -> torch.Tensor:
        diffs = sorted_logits[:, :-1] - sorted_logits[:, 1:]
        diffs = torch.cat(
            (
                diffs,
                torch.ones((diffs.shape[0], 1), dtype=diffs.dtype, device=diffs.device),
            ),
            dim=1,
        )
        assert torch.all(diffs >= 0), f"diffs should be positive: {diffs}"
        return diffs.flip([1])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        _, unsorted_indices = torch.sort(sorted_indices, descending=False)

        # [B, C]
        u = self.compute_u(sorted_logits)
        inp = sorted_logits if self.invariant else logits
        m = self.base_model(inp)

        m[:, 1:] = self.m_activation(m[:, 1:].clone())
        m[:, 0] = 0
        um = torch.cumsum(u * m, 1).flip([1])
        out = torch.gather(um, 1, unsorted_indices)

        if self.residual:
            out = out + logits
        return out


class IntraOrderPreservingCalibrator(Calibrator):
    """
    TODO work in progress, slow for now.

    From https://github.com/AmirooR/IntraOrderPreservingCalibration

    Order-Preserving Calibrator using a neural network mapping.

    Inputs (X) must be probabilities.
    Supports both multi-class (2D) and binary (1D or 2D) probabilities.

    Reference
    ---------
    Amir Rahimi, Amirreza Shaban, Ching-An Cheng, Richard Hartley, Byron Boots. Intra
    Order-preserving Functions for Calibration of Multi-Class Neural Networks. Advances
    in Neural Information Processing Systems, 2020.
    """
    def __init__(
        self,
        invariant: bool = True,
        residual: bool = False,
        hidden_sizes: tuple = (150, 150),
        num_epochs: int = 40,
        lr: float = 0.005,
        weight_decay: float = 0.0005,
        batch_size: int = 256,
    ):
        super().__init__()

        self.invariant = invariant
        self.residual = residual
        self.hidden_sizes = hidden_sizes
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the calibrator using Adam and CrossEntropyLoss."""

        # Convert 1D binary probabilities to 2D
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            X = np.column_stack((1.0 - X.flatten(), X.flatten()))

        self.n_classes_ = X.shape[1]

        # CrossEntropyLoss expects 1D integer class indices
        if y.ndim == 2 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)

        # Reverse-engineer logits safely from probabilities
        logits = np.log(np.clip(X, 1e-15, 1.0))

        X_tensor = torch.tensor(logits, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        bs = self.batch_size if len(X) > self.batch_size else len(X)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        # Build base FC model and wrap it in the Order Preserving Model
        base_model = _FCModel(
            in_features=self.n_classes_,
            hidden_sizes=self.hidden_sizes,
            out_features=self.n_classes_,
        )

        self.model_ = _OrderPreservingModel(
            base_model=base_model,
            invariant=self.invariant,
            residual=self.residual,
            num_classes=self.n_classes_,
        ).to(self.device)

        # Optimizer & Loss as defined in config.json and calibrate.py
        optimizer = optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for epoch in range(self.num_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output_batch = self.model_(batch_x)
                loss = criterion(output_batch, batch_y)
                loss.backward()
                optimizer.step()

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities."""
        if self.model_ is None:
            raise RuntimeError(
                "Calibrator must be fitted before calling predict_proba."
            )

        # Convert 1D binary probabilities to 2D
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            X = np.column_stack((1.0 - X.flatten(), X.flatten()))

        # Reverse-engineer logits safely from probabilities
        logits = np.log(np.clip(X, 1e-15, 1.0))
        X_tensor = torch.tensor(logits, dtype=torch.float32).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            calibrated_logits = self.model_(X_tensor)
            probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()

        return probs
