import torch
import torchmin
import warnings
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .base import Calibrator
from probmetrics.utils import expand_binary_probs, multiclass_probs_to_logits
from probmetrics.distributions import (
    CategoricalDistribution,
    CategoricalLogits,
)

from typing import Callable


def bisection_search(f: Callable[[float], float], a: float, b: float, n_steps: int):
    for _ in range(n_steps):
        c = a + 0.5 * (b - a)
        f_c = f(c)
        if f_c > 0:
            b = c
        else:
            a = c

    return 0.5 * (a + b)


class TemperatureScalingCalibrator(Calibrator):
    """
    Fast, optimized implementation. By default, uses bisection search on the gradient of
    the cross entropy for a scaling (inverse temperature) parameter.

    Reference
    ---------
    Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern
    neural networks. International Conference on Machine Learning, 2017.
    """
    def __init__(
        self,
        opt: str = "bisection",
        max_bisection_steps: int = 30,
        lr: float = 0.1,
        max_iter: int = 200,
        use_inv_temp: bool = True,
        inv_temp_init: float = 1 / 1.5,
    ):
        super().__init__()
        self.lr = lr
        self.max_bisection_steps = max_bisection_steps
        self.max_iter = max_iter
        self.use_inv_temp = use_inv_temp
        self.inv_temp_init = inv_temp_init
        self.opt = opt

    def _get_loss_grad(self, invtemp: float, logits: torch.Tensor, y: torch.Tensor):
        part_1 = torch.mean(
            torch.sum(logits * torch.softmax(invtemp * logits, dim=-1), dim=-1)
        )
        part_2 = torch.mean(logits[torch.arange(logits.shape[0]), y])
        return (part_1 - part_2).item()

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ):
        logits = y_pred.get_logits()
        labels = y_true_labels

        if self.opt in ["lbfgs", "lbfgs_line_search"]:
            self._fit_lbfgs(logits, labels)
        elif self.opt == "bisection":
            self._fit_bisection(logits, labels)
        else:
            raise ValueError(f'Unknown optimizer "{self.opt}"')

    def _fit_lbfgs(self, logits: torch.Tensor, labels: torch.Tensor):
        # following https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
        param = nn.Parameter(
            torch.ones(1, device=logits.device)
            * (self.inv_temp_init if self.use_inv_temp else 1 / self.inv_temp_init)
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [param],
            lr=self.lr,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe" if self.opt == "lbfgs_line_search" else None,
        )

        def eval():
            optimizer.zero_grad()
            y_pred = (
                logits * param[:, None]
                if self.use_inv_temp
                else logits / param[:, None]
            )
            loss = criterion(y_pred, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        self.invtemp_ = param.item() if self.use_inv_temp else 1 / param.item()

    def _fit_bisection(self, logits: torch.Tensor, labels: torch.Tensor):
        objective_grad = lambda u, l=logits, tar=labels: self._get_loss_grad(
            np.exp(u), l, tar
        )

        # should reach about float32 accuracy
        # need log_2(32) = 5 steps to get to length 1 and then 24 more steps to get to float32 epsilon (2^{-24})
        self.invtemp_ = np.exp(
            bisection_search(
                objective_grad, a=-16, b=16, n_steps=self.max_bisection_steps
            )
        )

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        return CategoricalLogits(self.invtemp_ * y_pred.get_logits())


### Other temperature scaling implementations ###


class GuoTemperatureScalingCalibrator(Calibrator):
    """
    Adapted from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Reference
    ---------
    Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern
    neural networks. International Conference on Machine Learning, 2017.
    """
    def __init__(self):
        super().__init__()

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ):
        self.temp_ = nn.Parameter(torch.ones(1) * 1.5)

        logits = y_pred.get_logits()
        labels = y_true_labels

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nll_criterion = nn.CrossEntropyLoss().to(device)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temp_], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() / self.temp_)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temp = self.temp_.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temp


class AutoGluonTemperatureScalingCalibrator(Calibrator):
    """
    Adapted from
    https://github.com/autogluon/autogluon/blob/c1181326cf6b7e3b27a7420273f1a82808d939e2/core/src/autogluon/core/calibrate/temperature_scaling.py#L9
    https://github.com/autogluon/autogluon/blob/28a242ebe8d55ba770c991b9db153ab4623c9abd/tabular/src/autogluon/tabular/trainer/abstract_trainer.py#L4433-L4457

    Reference
    ---------
    Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern
    neural networks. International Conference on Machine Learning, 2017.
    """
    def __init__(self, init_val: float = 1, max_iter: int = 200, lr: float = 0.1):
        super().__init__()
        self.init_val = init_val
        self.max_iter = max_iter
        self.lr = lr

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ):
        y_val_tensor = y_true_labels
        temperature_param = torch.nn.Parameter(torch.ones(1).fill_(self.init_val))
        logits = y_pred.get_logits()

        is_invalid = torch.isinf(logits).any().tolist()
        if is_invalid:
            return

        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [temperature_param], lr=self.lr, max_iter=self.max_iter
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        optimizer_trajectory = []

        if self.init_val != 1.0:
            # need to check 1.0 as well since AutoGluon does it outside
            optimizer_trajectory.append(
                (nll_criterion(logits, y_val_tensor).item(), 1.0)
            )

        def temperature_scale_step():
            optimizer.zero_grad()
            temp = temperature_param.unsqueeze(1).expand(logits.size(0), logits.size(1))
            new_logits = logits / temp
            loss = nll_criterion(new_logits, y_val_tensor)
            loss.backward()
            scheduler.step()
            optimizer_trajectory.append((loss.item(), temperature_param.item()))
            return loss

        optimizer.step(temperature_scale_step)

        try:
            best_loss_index = np.nanargmin(np.array(optimizer_trajectory)[:, 0])
        except ValueError:
            self.temperature = 1.0
            return
        temperature_scale = float(np.array(optimizer_trajectory)[best_loss_index, 1])

        if np.isnan(temperature_scale) or temperature_scale <= 0.0:
            self.temperature = 1.0
            return

        self.temp_ = temperature_scale

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() / self.temp_)


class AutoGluonInverseTemperatureScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/autogluon/autogluon/blob/c1181326cf6b7e3b27a7420273f1a82808d939e2/core/src/autogluon/core/calibrate/temperature_scaling.py#L9
    # but optimizing the inverse temperature instead
    def __init__(self, init_val: float = 1, max_iter: int = 200, lr: float = 0.1):
        super().__init__()
        self.init_val = init_val
        self.max_iter = max_iter
        self.lr = lr

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ):
        y_val_tensor = y_true_labels
        inv_temperature_param = torch.nn.Parameter(torch.ones(1).fill_(self.init_val))
        logits = y_pred.get_logits()

        is_invalid = torch.isinf(logits).any().tolist()
        if is_invalid:
            return

        nll_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [inv_temperature_param], lr=self.lr, max_iter=self.max_iter
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        optimizer_trajectory = []

        def temperature_scale_step():
            optimizer.zero_grad()
            inv_temp = inv_temperature_param.unsqueeze(1).expand(
                logits.size(0), logits.size(1)
            )
            new_logits = logits * inv_temp
            loss = nll_criterion(new_logits, y_val_tensor)
            loss.backward()
            scheduler.step()
            optimizer_trajectory.append((loss.item(), inv_temperature_param.item()))
            return loss

        optimizer.step(temperature_scale_step)

        try:
            best_loss_index = np.nanargmin(np.array(optimizer_trajectory)[:, 0])
        except ValueError:
            return
        inv_temperature_scale = float(
            np.array(optimizer_trajectory)[best_loss_index, 1]
        )

        if np.isnan(inv_temperature_scale):
            return

        self.inv_temp_ = inv_temperature_scale

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        with torch.no_grad():
            return CategoricalLogits(y_pred.get_logits() * self.inv_temp_)


class TorchUncertaintyTemperatureScalingCalibrator(TemperatureScalingCalibrator):
    # adapted from
    # https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/3a021d2e34e183b8aad3a0345e6d750c08c72af3/torch_uncertainty/post_processing/calibration/scaler.py#L1
    # https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/3a021d2e34e183b8aad3a0345e6d750c08c72af3/torch_uncertainty/post_processing/calibration/temperature_scaler.py#L9
    def __init__(self, init_val: float = 1, lr: float = 0.1, max_iter: int = 100):
        super().__init__(
            opt="lbfgs",
            lr=lr,
            max_iter=max_iter,
            use_inv_temp=False,
            inv_temp_init=1.0 / init_val,
        )
        # need to save these values here to comply with sklearn cloneability conventions
        self.init_val = init_val
        self.lr = lr
        self.max_iter = max_iter


class NetcalTemperatureScalingCalibrator(Calibrator):
    # this one does nothing due to https://github.com/EFS-OpenSource/calibration-framework/issues/61
    def __init__(self):
        super().__init__()
        try:
            from netcal.scaling import TemperatureScaling
        except ImportError as e:
            raise ImportError("The 'netcal' package is required.") from e

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from netcal.scaling import TemperatureScaling

        self.cal_ = TemperatureScaling()

        if X.shape[1] == 2:
            # binary, convert
            X = X[:, 1]
        self.cal_.fit(X, y, random_state=0)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == 2:
            # binary, convert
            X = X[:, 1]
        probs = self.cal_.transform(X)
        probs = expand_binary_probs(probs)
        return probs


class TorchcalTemperatureScalingCalibrator(Calibrator):
    # adapted from
    # https://github.com/rishabh-ranjan/torchcal/blob/3fb65f6423d33d680cd68c7f40a0259d41e8fb0b/torchcal.py#L8
    def __init__(self):
        super().__init__()

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ):
        y_pred_logits = y_pred.get_logits()

        temp = torch.ones(1, device=y_pred_logits.device)

        def loss(t):
            return torch.nn.functional.cross_entropy(y_pred_logits / t, y_true_labels)

        res = torchmin.minimize(loss, temp, method="newton-exact")
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
        else:
            temp = res.x

        self.temp_ = temp.item()

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        return CategoricalLogits(y_pred.get_logits() / self.temp_)


class ETSCalibrator(Calibrator):
    """
    From https://github.com/zhang64-llnl/Mix-n-Match-Calibration
    Optimizes a temperature parameter and then finds the optimal ensemble weights
    between the scaled predictions, original predictions, and a uniform distribution.

    Inputs (X) must be probabilities.
    Supports both multi-class (2D) and binary (1D or 2D) probabilities.

    Reference
    ---------
    Jize Zhang, Bhavya Kailkhura, and T. Yong-Jin Han. Mix-n-match: Ensemble and
    compositional methods for uncertainty calibration in deep learning. International
    Conference on Machine Learning, 2020.
    """

    def __init__(self, loss: str = "mse"):
        super().__init__()
        try:
            from scipy import optimize
        except ImportError as e:
            raise ImportError("The 'scipy' package is required.") from e
        
        if loss not in ["mse", "ce"]:
            raise ValueError(
                "Loss must be either 'mse' (Mean Squared Error) or 'ce' (Cross-Entropy)."
            )
        self.loss = loss

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the calibrator.
        X: array of probabilities (n_samples, n_classes) or (n_samples,) for binary.
        y: labels (n_samples,) or one-hot encoded (n_samples, n_classes).
        """
        from scipy import optimize
        # Convert 1D binary probabilities to 2D (Class 0: 1 - p, Class 1: p)
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            X = np.column_stack((1.0 - X.flatten(), X.flatten()))
        self.n_classes_ = X.shape[1]
        logits = multiclass_probs_to_logits(X)

        # Convert labels to one-hot if they are purely 1D indices
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_one_hot = np.eye(self.n_classes_)[y.flatten()]
        else:
            y_one_hot = y

        # 1. Optimize Temperature (t) using the derived logits
        bnds_t = ((0.05, 5.0),)
        t_target_func = self._ll_t if self.loss == "ce" else self._mse_t
        res_t = optimize.minimize(
            t_target_func,
            x0=[1.0],
            args=(logits, y_one_hot),
            method="L-BFGS-B",
            bounds=bnds_t,
            tol=1e-12,
        )
        self.t_ = res_t.x[0]

        # 2. Optimize Ensemble Weights (w0, w1, w2)
        p1 = X  # Original probabilities
        p0 = self._stable_softmax(logits / self.t_)  # Temperature-scaled probabilities
        p2 = np.ones_like(p0) / self.n_classes_  # Uniform distribution

        bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        w_target_func = self._ll_w if self.loss == "ce" else self._mse_w
        res_w = optimize.minimize(
            w_target_func,
            x0=[1.0, 0.0, 0.0],
            args=(p0, p1, p2, y_one_hot),
            method="SLSQP",
            constraints=constraints,
            bounds=bnds_w,
            tol=1e-12,
        )
        self.w_ = res_w.x

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts calibrated probabilities.
        X: array of probabilities (n_samples, n_classes) or (n_samples,) for binary.
        """
        # Convert 1D binary probabilities to 2D (Class 0: 1 - p, Class 1: p)
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            X = np.column_stack((1.0 - X.flatten(), X.flatten()))
        self.n_classes_ = X.shape[1]
        logits = multiclass_probs_to_logits(X)

        p1 = X
        p0 = self._stable_softmax(logits / self.t_)
        p2 = np.ones_like(p0) / self.n_classes_

        p_calibrated = self.w_[0] * p0 + self.w_[1] * p1 + self.w_[2] * p2

        # Ensure mathematical strictness of probabilities
        return p_calibrated / np.sum(p_calibrated, axis=1, keepdims=True)

    @staticmethod
    def _stable_softmax(x: np.ndarray) -> np.ndarray:
        """Helper to prevent np.exp() overflow errors."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    @staticmethod
    def _mse_t(t: float, logit: np.ndarray, label: np.ndarray) -> float:
        p = ETSCalibrator._stable_softmax(logit / t)
        return np.mean((p - label) ** 2)

    @staticmethod
    def _ll_t(t: float, logit: np.ndarray, label: np.ndarray) -> float:
        p = ETSCalibrator._stable_softmax(logit / t)
        p = np.clip(p, 1e-20, 1 - 1e-20)
        return -np.sum(label * np.log(p)) / p.shape[0]

    @staticmethod
    def _mse_w(
        w: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, label: np.ndarray
    ) -> float:
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        p = p / np.sum(p, axis=1, keepdims=True)
        return np.mean((p - label) ** 2)

    @staticmethod
    def _ll_w(
        w: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, label: np.ndarray
    ) -> float:
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        p = np.clip(p, 1e-20, 1 - 1e-20)
        return -np.sum(label * np.log(p)) / p.shape[0]


def _softplus_fn(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    """Core mathematical mapping for the Softplus recalibration."""
    return -(c / a) * torch.log(u + torch.exp(-a * (x - b)))


class _PyTorchSoftplus(nn.Module):
    """
    Internal PyTorch module adapting the original code for standard 2D
    (n_samples, n_classes) input.
    """

    def __init__(
        self, method: str, T: float = 1.0, use_shift: bool = False, n_classes: int = 2
    ):
        super().__init__()

        # Initialize defaults
        init_a = 0.135 if method == "SPU" else 1.0
        init_c = T if method == "SPTS" else 1.0

        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.b = nn.Parameter(torch.ones(1) * 1.0)
        self.c = nn.Parameter(torch.ones(1) * init_c)
        self.u = nn.Parameter(torch.ones(1) * 0.1)

        # Apply specific freezing rules based on the method
        if method in ["SP1", "SPTS"]:
            self.c.requires_grad = False

        self.use_shift = use_shift
        if self.use_shift:
            # Learned weight vector for cross-class shifting
            self.w = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_shift:
            # Dot product of x and w gives a shift per sample: (n_samples,)
            # We unsqueeze to (n_samples, 1) so it broadcasts across all classes
            shift = (x @ self.w).unsqueeze(1)
            x = x + shift

        # The original code squares a, c, and u to enforce strict positivity
        return _softplus_fn(x, self.a**2, self.b, self.c**2, self.u**2)


class SoftPlusCalibrator(Calibrator):
    """
    TODO work in progress, very slow for now.

    Softplus Recalibrator using PyTorch for optimization.
    Supports SPU, SP1, and SPTS, with or without the cross-class 'shift' mechanism.

    Inputs (X) must be probabilities.
    Supports both multi-class (2D) and binary (1D or 2D) probabilities.

    Reference
    ---------
    Christopher Qian, Feng Liang, and Jason Adams. Extending Temperature Scaling with
    Homogenizing Maps. Journal of Machine Learning Research 26(161):1-46, 2025
    """

    def __init__(
        self,
        method: str = "SPU",
        T: float = 1.0,
        use_shift: bool = False,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
    ):
        super().__init__()
        if method not in ["SPU", "SP1", "SPTS"]:
            raise ValueError("Method must be one of 'SPU', 'SP1', or 'SPTS'.")

        self.method = method
        self.T = T
        self.use_shift = use_shift
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the calibrator using Adam and CrossEntropyLoss."""
        # Convert 1D binary probabilities to 2D (Class 0: 1 - p, Class 1: p)
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            X = np.column_stack((1.0 - X.flatten(), X.flatten()))
        self.n_classes_ = X.shape[1]
        logits = multiclass_probs_to_logits(X)

        # CrossEntropyLoss expects 1D integer class indices, not one-hot
        if y.ndim == 2 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)

        X_tensor = torch.tensor(logits, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        # Fall back to len(X) batch size if the dataset is tiny, matching original script logic
        bs = self.batch_size if len(X) > self.batch_size else len(X)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)

        self.model_ = _PyTorchSoftplus(
            method=self.method,
            T=self.T,
            use_shift=self.use_shift,
            n_classes=self.n_classes_,
        ).to(self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.n_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                logits_calibrated = self.model_(batch_x)
                loss = criterion(logits_calibrated, batch_y)
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
        logits = multiclass_probs_to_logits(X)
        X_tensor = torch.tensor(logits, dtype=torch.float32).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            calibrated_logits = self.model_(X_tensor)
            # Apply softmax to get final probabilities
            probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()

        return probs
