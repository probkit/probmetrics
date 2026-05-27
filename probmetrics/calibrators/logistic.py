import torch
import logging
import torchmin
import warnings
import numpy as np
from typing import Optional, List
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from .base import Calibrator, WrapCalibratorAsClassifier
from probmetrics.distributions import (
    CategoricalDistribution,
    CategoricalLogits,
)

from probmetrics.saga import (
    fit_affine_scaling,
    warm_up_affine_scaling,
    fit_quadratic_scaling,
    warm_up_quadratic_scaling,
    fit_vector_scaling,
    warm_up_vector_scaling,
    fit_matrix_scaling,
    warm_up_matrix_scaling,
)

from probmetrics.utils import (
    validate_probabilities,
    flatten_binary_probs,
    expand_binary_probs,
    binary_probs_to_logits,
    multiclass_probs_to_logits,
)

logger = logging.getLogger(__name__)


class VectorScalingCalibrator(Calibrator):
    """
    Adapted from https://github.com/rishabh-ranjan/torchcal

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
        y_pred_logits = y_pred.get_logits()
        num_classes = y_pred_logits.shape[1]

        def loss(temp_bias):
            temp, bias = temp_bias.split([num_classes, num_classes])
            return torch.nn.functional.cross_entropy(
                y_pred_logits / temp + bias, y_true_labels
            )

        temp = torch.ones(num_classes, device=y_pred_logits.device)
        bias = torch.zeros(num_classes, device=y_pred_logits.device)
        temp_bias = torch.cat([temp, bias])
        res = torchmin.minimize(loss, temp_bias, method="bfgs")
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
            self.temp_, self.bias_ = temp, bias
        else:
            self.temp_, self.bias_ = res.x.split([num_classes, num_classes])

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        return CategoricalLogits(y_pred.get_logits() / self.temp_ + self.bias_)


class SVSCalibrator(Calibrator):
    """
    Multiclass post-hoc calibration with a structured scaling scheme
    $softmax((aI + diag(v))x + b)$ on the logits x.

    Numba functions are warmed-up the first time the class is initialized with the
    'saga' optimizer.

    A penalty (one of [None, "mcp", "lasso", "ridge"]) is applied to the intercept (b)
    and vector scaling (v) parameters, with respective regularization strength
    lambda_intercept*(k**rho)/(n**tau) and lambda_diagonal*(k**rho)/(n**tau).

    Instead of fitting the global scaling parameter jointly with the other parameters,
    the logits are pre-processed with temperature scaling (with no regularization) and a
    is then fixed to a=1.

    [!] solver = 'bfgs' only supports 'ridge' penalty.
    [!] For solver = 'bfgs', max_iter and tol are ignored and default torchmin
    parameters are used.

    Reference
    ---------
    Eugène Berta, David Holzmüller, Michael I. Jordan, and Francis Bach. Structured
    matrix scaling for multi-class calibration. International Conference on Artificial
    Intelligence and Statistics, 2026.
    """

    _warmed_up_saga = False
    _ALLOWED_PENALTIES = [None, "mcp", "lasso", "ridge"]
    _ALLOWED_OPTIMIZERS = ["saga", "bfgs"]

    def __init__(
        self,
        penalty: str = "ridge",
        rho: float = 1.0,
        tau: float = 1.0,
        lambda_intercept: float = 1.0,
        lambda_diagonal: float = 1.0,
        opt: str = "bfgs",
        max_iter: int = 500,
        tol: float = 1e-5,
        print_init_info: bool = True,
    ):
        super().__init__()

        if penalty not in self._ALLOWED_PENALTIES:
            raise ValueError(
                f"Penalty '{penalty}' not recognized. Choose from {self._ALLOWED_PENALTIES}."
            )

        if opt not in self._ALLOWED_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{opt}' not recognized. Choose from {self._ALLOWED_OPTIMIZERS}."
            )

        if opt == "bfgs" and penalty != "ridge":
            raise ValueError(
                "The 'bfgs' optimizer only supports the 'ridge' penalty, use 'saga' instead."
            )

        if opt == "saga":
            try:
                from numba import njit
            except ImportError as e:
                raise ImportError(
                    "The 'numba' package is required for optimizer 'saga'."
                ) from e

            if not SVSCalibrator._warmed_up_saga:
                if print_init_info:
                    logger.info(
                        "First SVSCalibrator instantiation with 'saga' - warming up Numba functions..."
                    )
                warm_up_vector_scaling()
                SVSCalibrator._warmed_up_saga = True
                if print_init_info:
                    logger.info("Warmed up!")

        self.penalty = penalty
        self.rho = rho
        self.tau = tau
        self.lambda_intercept = lambda_intercept
        self.lambda_diagonal = lambda_diagonal
        self.max_iter = max_iter
        self.tol = tol
        self.opt = opt
        self.print_init_info = print_init_info

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        n, k = X.shape

        from .factory import get_calibrator

        self.ts_ = get_calibrator("ts-mix")
        self.ts_.fit(X, y)
        scaled_probs = self.ts_.predict_proba(X)

        reg_intercept = self.lambda_intercept * (k**self.rho) / (n**self.tau)
        reg_diagonal = self.lambda_diagonal * (k**self.rho) / (n**self.tau)

        logits = multiclass_probs_to_logits(scaled_probs)

        if self.opt == "saga":
            self.v_, self.b_ = fit_vector_scaling(
                logits,
                y,
                penalty=self.penalty,
                reg_intercept=reg_intercept,
                reg_diagonal=reg_diagonal,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        elif self.opt == "bfgs":
            self.v_, self.b_ = self._fit_bfgs(
                torch.tensor(logits),
                torch.as_tensor(y, dtype=torch.long),
                num_classes=k,
                reg_intercept=reg_intercept,
                reg_diagonal=reg_diagonal,
            )

    def _fit_bfgs(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        num_classes,
        reg_intercept,
        reg_diagonal,
    ):
        K = num_classes

        initial_params = torch.zeros(2 * K, device=logits.device, dtype=logits.dtype)

        def loss(params):
            v_delta = params[:K]
            b = params[K:]

            scaled_logits = logits * (1.0 + v_delta) + b
            ce = torch.nn.functional.cross_entropy(scaled_logits, y)

            r_b = reg_intercept * b.pow(2).sum()
            r_v = reg_diagonal * v_delta.pow(2).sum()

            return ce + r_b + r_v

        method = "l-bfgs" if initial_params.numel() > 1000 else "bfgs"
        res = torchmin.minimize(loss, initial_params, method=method)

        v_delta_final = res.x[:K]
        b_final = res.x[K:]

        return (
            1.0 + v_delta_final
        ).detach().cpu().numpy(), b_final.detach().cpu().numpy()

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        scaled_probs = self.ts_.predict_proba(X)
        logits = multiclass_probs_to_logits(scaled_probs)
        calibrated_logits = self.v_.reshape(1, -1) * logits + self.b_.reshape(1, -1)
        calibrated_logits -= np.max(calibrated_logits, axis=1, keepdims=True)
        exp_logits = np.exp(calibrated_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class MatrixScalingCalibrator(Calibrator):
    """
    Adapted from https://github.com/rishabh-ranjan/torchcal

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
        y_pred_logits = y_pred.get_logits()

        num_classes = y_pred_logits.shape[1]

        num_params = num_classes * (num_classes + 1)
        if num_params <= 1000:
            method = "bfgs"
        else:
            method = "l-bfgs"

        def loss(itemp_bias):
            itemp, bias = itemp_bias.split([num_classes**2, num_classes])
            itemp = itemp.view(num_classes, num_classes)
            return torch.nn.functional.cross_entropy(
                y_pred_logits @ itemp + bias, y_true_labels
            )

        itemp = torch.eye(
            num_classes,
            num_classes,
            device=y_pred_logits.device,
            dtype=y_pred_logits.dtype,
        )
        bias = torch.zeros(
            num_classes, device=y_pred_logits.device, dtype=y_pred_logits.dtype
        )
        itemp_bias = torch.cat([itemp.view(-1), bias])
        res = torchmin.minimize(loss, itemp_bias, method=method)
        if not res.success:
            warnings.warn(
                f"{self.__class__}: {res.message} Not updating calibrator params."
            )
            self.itemp_, self.bias_ = itemp, bias
        else:
            self.itemp_, self.bias_ = res.x.split([num_classes**2, num_classes])
            self.itemp_ = self.itemp_.view(num_classes, num_classes)

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        return CategoricalLogits(y_pred.get_logits() @ self.itemp_ + self.bias_)


class SMSCalibrator(Calibrator):
    """
    Multiclass post-hoc calibration with a structured scaling scheme
    $softmax( (aI + diag(v) + off_diag(M) )x + b)$ on the logits x.

    Numba functions are warmed-up the first time the class is initialized with the
    'saga' optimizer.

    A penalty (one of ["mcp", "lasso", "group_lasso", "ridge"]) is applied to the
    intercept (b), vector scaling (v) and off diagonal (M) parameters, with respective
    regularization strength lambda_intercept*(k**rho)/(n**tau),
    lambda_diagonal*(k**rho)/(n**tau) and lambda_off_diagonal*((k*(k-1))**rho)/(n**tau).

    Instead of fitting the global scaling parameter jointly with the other parameters,
    the logits are pre-processed with temperature scaling (with no regularization) and a
    is then fixed to a=1.

    [!] The 'bfgs' solver only supports 'ridge' penalty.
    [!] If 'bfgs' solver is used, max_iter and tol are ignored and default torchmin
    parameters are used.

    Reference
    ---------
    Eugène Berta, David Holzmüller, Michael I. Jordan, and Francis Bach. Structured
    matrix scaling for multi-class calibration. International Conference on Artificial
    Intelligence and Statistics, 2026.
    """

    _warmed_up_saga = False
    _ALLOWED_PENALTIES = [None, "mcp", "lasso", "group_lasso", "ridge"]
    _ALLOWED_OPTIMIZERS = ["saga", "bfgs"]

    def __init__(
        self,
        penalty: str = "ridge",
        rho: float = 1.0,
        tau: float = 1.0,
        lambda_intercept: float = 1.0,
        lambda_diagonal: float = 1.0,
        lambda_off_diagonal: float = 1.0,
        opt: str = "bfgs",
        max_iter: int = 500,
        tol: float = 1e-5,
        print_init_info: bool = True,
    ):
        super().__init__()

        if penalty not in self._ALLOWED_PENALTIES:
            raise ValueError(
                f"Penalty '{penalty}' not recognized. Choose from {self._ALLOWED_PENALTIES}."
            )

        if opt not in self._ALLOWED_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{opt}' not recognized. Choose from {self._ALLOWED_OPTIMIZERS}."
            )

        if opt == "bfgs" and penalty != "ridge":
            raise ValueError(
                "The 'bfgs' optimizer only supports the 'ridge' penalty, use 'saga' instead."
            )

        if opt == "saga":
            try:
                from numba import njit
            except ImportError as e:
                raise ImportError(
                    "The 'numba' package is required for optimizer 'saga'."
                ) from e
        
            if not SMSCalibrator._warmed_up_saga:
                if print_init_info:
                    logger.info(
                        "First SMSCalibrator instantiation with 'saga' - warming up Numba functions..."
                    )
                warm_up_matrix_scaling()
                SMSCalibrator._warmed_up_saga = True
                if print_init_info:
                    logger.info("Warmed up!")

        self.penalty = penalty
        self.rho = rho
        self.tau = tau
        self.lambda_intercept = lambda_intercept
        self.lambda_diagonal = lambda_diagonal
        self.lambda_off_diagonal = lambda_off_diagonal
        self.max_iter = max_iter
        self.tol = tol
        self.opt = opt
        self.print_init_info = print_init_info

    def _get_logits(
        self, X: np.ndarray, append_bias: bool = True
    ) -> tuple[np.ndarray, int, int]:
        logits = multiclass_probs_to_logits(X)
        if append_bias:
            # Adding a constant term to logit vectors to fit the intercept.
            logits = np.hstack([logits, np.ones((len(logits), 1))])
        return logits

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the calibration matrix W."""
        validate_probabilities(X)

        n, k = X.shape

        from .factory import get_calibrator

        self.ts_ = get_calibrator("ts-mix")
        self.ts_.fit(X, y)
        scaled_probs = self.ts_.predict_proba(X)

        reg_intercept = self.lambda_intercept * (k**self.rho) / (n**self.tau)
        reg_diagonal = self.lambda_diagonal * (k**self.rho) / (n**self.tau)
        reg_off_diagonal = (
            self.lambda_off_diagonal * ((k * (k - 1)) ** self.rho) / (n**self.tau)
        )

        if self.opt == "saga":
            logits = self._get_logits(scaled_probs)
            self.W_ = fit_matrix_scaling(
                logits,
                y,
                penalty=self.penalty,
                max_iter=self.max_iter,
                tol=self.tol,
                init_scaling=1.0,
                reg_intercept=reg_intercept,
                reg_diagonal=reg_diagonal,
                reg_off_diagonal=reg_off_diagonal,
            )

        elif self.opt == "bfgs":
            logits = self._get_logits(scaled_probs, append_bias=False)
            self.W_ = self._fit_bfgs(
                torch.tensor(logits),
                torch.as_tensor(y, dtype=torch.long),
                num_classes=k,
                reg_intercept=reg_intercept,
                reg_diagonal=reg_diagonal,
                reg_off_diagonal=reg_off_diagonal,
            )

    def _fit_bfgs(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        num_classes,
        reg_intercept,
        reg_diagonal,
        reg_off_diagonal,
    ):
        K = num_classes

        initial_params = torch.zeros(
            K * (K + 1), device=logits.device, dtype=logits.dtype
        )

        def loss(params):
            W_delta = params[: K * K].view(K, K)
            b = params[K * K :]

            scaled_logits = logits + torch.nn.functional.linear(logits, W_delta, b)
            ce = torch.nn.functional.cross_entropy(scaled_logits, y)

            W_diag = W_delta.diagonal()
            r_i = reg_intercept * b.pow(2).sum()
            r_d = reg_diagonal * W_diag.pow(2).sum()
            r_o = reg_off_diagonal * (W_delta.pow(2).sum() - W_diag.pow(2).sum())

            return ce + r_i + r_d + r_o

        method = "l-bfgs" if initial_params.numel() > 1000 else "bfgs"
        res = torchmin.minimize(loss, initial_params, method=method)

        flat_result = res.x
        W_delta_final = flat_result[: K * K].view(K, K)
        b_final = flat_result[K * K :]

        with torch.no_grad():
            W_final = (
                torch.eye(K, device=logits.device, dtype=logits.dtype) + W_delta_final
            )
            return torch.hstack([W_final, b_final.unsqueeze(1)]).detach().cpu().numpy()

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        scaled_probs = self.ts_.predict_proba(X)
        logits = self._get_logits(scaled_probs)
        calibrated_logits = logits @ self.W_.T
        calibrated_logits -= np.max(calibrated_logits, axis=1, keepdims=True)
        exp_logits = np.exp(calibrated_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def logloss_np(y_true: np.ndarray, y_proba: np.ndarray):
    return -np.mean(np.take_along_axis(np.log(y_proba), y_true[:, None], axis=1))


class DirichletCalibrator(Calibrator):
    """
    From https://github.com/dirichletcal/dirichlet_python

    Reference
    ---------
    Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and
    Peter Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class
    probabilities with Dirichlet calibration. Advances in Neural Information Processing
    Systems, 2019.
    """
    def __init__(
        self,
        n_cv: int = 0,
        use_odir: bool = False,
        reg_lambda: float = 0.0,
        reg_mu: Optional[float] = None,
        reg_lambda_grid: Optional[List[float]] = None,
        reg_mu_grid: Optional[List[float]] = None,
    ):
        super().__init__()
        try:
            from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
        except ImportError as e:
            raise ImportError("The 'dirichletcal' package is required.") from e

        self.n_cv = n_cv
        self.use_odir = use_odir
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.reg_lambda_grid = reg_lambda_grid
        self.reg_mu_grid = reg_mu_grid

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator

        if self.n_cv == 0:
            self.cal_ = FullDirichletCalibrator(
                reg_lambda=self.reg_lambda, reg_mu=self.reg_mu
            )
        elif self.n_cv >= 2:
            reg_lambda_grid = self.reg_lambda_grid or [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            reg_mu_grid = self.reg_mu_grid or [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            calibrator = FullDirichletCalibrator(
                reg_lambda=reg_lambda_grid,
                reg_mu=reg_mu_grid if self.use_odir else None,
            )
            calibrator = WrapCalibratorAsClassifier(calibrator)
            skf = StratifiedKFold(n_splits=self.n_cv, shuffle=True, random_state=0)
            self.cal_ = GridSearchCV(
                calibrator,
                param_grid={
                    "calib__reg_lambda": reg_lambda_grid,
                    "calib__reg_mu": reg_mu_grid if self.use_odir else [None],
                },
                cv=skf,
                # use this because scikit-learn's logloss scorer fails in case of
                # missing classes
                scoring=lambda est, X_test, y_test: -logloss_np(
                    y_test, est.predict_proba(X_test)
                ),
            )
        else:
            raise ValueError(f"n_cv must be either 0 or >=2, but is {self.n_cv}")

        self.cal_.fit(X, y)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        return self.cal_.predict_proba(X)


class LogisticCalibrator(Calibrator):
    """
    By default, uses 'quadratic-scaling' for binary classification and 'sms' for
    multiclass classification.
    """

    VALID_BINARY_TYPES = ["linear", "affine", "quadratic"]
    VALID_MULTICLASS_TYPES = ["svs", "sms"]

    def __init__(self, binary_type: str = "quadratic", multiclass_type: str = "sms"):
        super().__init__()

        if binary_type not in self.VALID_BINARY_TYPES:
            raise ValueError(
                f"Invalid binary type '{binary_type}'. Must be one of {self.VALID_BINARY_TYPES}"
            )
        self.binary_type = binary_type

        if multiclass_type not in self.VALID_MULTICLASS_TYPES:
            raise ValueError(
                f"Invalid multiclass type '{multiclass_type}'. Must be one of {self.VALID_MULTICLASS_TYPES}"
            )
        self.multiclass_type = multiclass_type

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        n_classes = len(np.unique(y))

        if n_classes == 2:
            if self.binary_type == "affine":
                self.cal_ = BinaryLogisticCalibrator(type="affine")
            elif self.binary_type == "linear":
                self.cal_ = BinaryLogisticCalibrator(type="linear")
            elif self.binary_type == "quadratic":
                self.cal_ = BinaryLogisticCalibrator(type="quadratic")

        else:
            if self.multiclass_type == "svs":
                self.cal_ = SVSCalibrator()
            elif self.multiclass_type == "sms":
                self.cal_ = SMSCalibrator()

        self.cal_.fit(X, y)
        return self

    def predict(self, X):
        if self.cal_ is None:
            raise RuntimeError("LogisticCalibrator must be fitted before prediction.")
        return self.cal_.predict(X)

    def _predict_proba_impl(self, X):
        if self.cal_ is None:
            raise RuntimeError("LogisticCalibrator must be fitted before prediction.")
        return self.cal_.predict_proba(X)


class BinaryLogisticCalibrator(Calibrator):
    """
    Binary post-hoc calibration with linear, affine or quadratic scaling
    on the binary logits using sklearn's LogisticRegression.

    This class fits a model of the form:
    P(y=1) = sigma(calibrated_logit)

    Where `calibrated_logit` depends on the `type`:
    - 'linear':    calibrated_logit = w * logit (Temperature scaling with inverse
    temperature)
    - 'affine':    calibrated_logit = b + w * logit (Platt scaling)
    - 'quadratic': calibrated_logit = b + w1 * logit + w2 * logit^2
    """

    VALID_TYPES = ["linear", "affine", "quadratic", "beta"]

    def __init__(self, type: str = "affine"):
        """
        Args:
            type (str, optional): The type of scaling.
                One of ['linear', 'affine', 'quadratic'].
                Defaults to 'affine'.
        """
        super().__init__()

        if type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid type '{type}'. Must be one of {self.VALID_TYPES}"
            )
        self.type = type

    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        log_p, log_1_minus_p = binary_probs_to_logits(X)
        logits = log_p - log_1_minus_p

        if self.type == "linear":
            return logits

        ones = np.ones_like(logits)

        if self.type == "affine":
            return np.hstack([ones, logits])

        elif self.type == "quadratic":
            return np.hstack([ones, logits, np.square(logits)])

        if self.type == "beta":  # TODO Check that this is indeed beta
            return np.hstack([ones, log_p, log_1_minus_p])

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the logistic regression calibrator."""
        from sklearn.linear_model import LogisticRegression

        self.cal_ = LogisticRegression(
            penalty=None, fit_intercept=False, solver="lbfgs"
        )

        features = self._get_logits(X)
        self.cal_.fit(features, y)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        features = self._get_logits(X)
        return self.cal_.predict_proba(features)


class RegularizedAffineScalingCalibrator(Calibrator):
    """
    Binary post-hoc calibration with Platt scaling (~binary logistic regression)
    $sigmoid(ax+b)$ on the logits x.
    Uses the SAGA algorithm to fit the scaling and intercept parameters a & b.
    Numba functions are warmed-up the first time the class is initialized (~1s).
    A penalty (one of ["mcp", "lasso", "ridge"]) can be applied to the intercept
    parameter b, with regularization strength lambda_intercept.
    To apply no regularization, leave penalty to None.
    """

    _warmed_up = False

    def __init__(
        self,
        penalty: str = None,
        lambda_intercept: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        print_init_info: bool = True,
    ):
        super().__init__()
        self.penalty = penalty
        self.lambda_intercept = lambda_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.print_init_info = print_init_info

        try:
            from numba import njit
        except ImportError as e:
            raise ImportError("The 'numba' package is required.") from e

        if not RegularizedAffineScalingCalibrator._warmed_up:
            if self.print_init_info:
                print(
                    "First RegularizedAffineScalingCalibrator instantiation - warming up Numba functions..."
                )
            warm_up_affine_scaling()
            RegularizedAffineScalingCalibrator._warmed_up = True
            if self.print_init_info:
                print("Warmed up!")

    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        log_p, log_1_minus_p = binary_probs_to_logits(X)
        logits = log_p - log_1_minus_p
        ones = np.ones_like(logits)
        return np.hstack([ones, logits])

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        logits = self._get_logits(X)
        self.w_ = fit_affine_scaling(
            logits,
            y,
            penalty=self.penalty,
            reg_intercept=self.lambda_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        logits = self._get_logits(X)
        logits = logits.dot(self.w_).reshape(-1, 1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.hstack([1 - probs, probs])


class RegularizedQuadraticScalingCalibrator(Calibrator):
    """
    Binary post-hoc calibration with quadratic scaling $sigmoid(a + bx + cx^2)$ on the
    logits x.
    Uses the SAGA algorithm to fit the intercept, scaling and quadratic parameters a, b
    & c.
    Numba functions are warmed-up the first time the class is initialized (~1s).
    A penalty (one of ["mcp", "lasso", "ridge"]) can be applied to the intercept and
    quadratic parameters a & c,
    with respective regularization strength lambda_intercept and lambda_quadratic.
    To apply no regularization, leave penalty to None.
    """

    _warmed_up = False

    def __init__(
        self,
        penalty: str = None,
        lambda_intercept: float = 0.0,
        lambda_quadratic: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        print_init_info: bool = True,
    ):
        super().__init__()
        self.penalty = penalty
        self.lambda_intercept = lambda_intercept
        self.lambda_quadratic = lambda_quadratic
        self.max_iter = max_iter
        self.tol = tol
        self.print_init_info = print_init_info

        try:
            from numba import njit
        except ImportError as e:
            raise ImportError("The 'numba' package is required.") from e

        if not RegularizedQuadraticScalingCalibrator._warmed_up:
            if self.print_init_info:
                print(
                    "First RegularizedQuadraticScalingCalibrator instantiation - warming up Numba functions..."
                )
            warm_up_quadratic_scaling()
            RegularizedQuadraticScalingCalibrator._warmed_up = True
            if self.print_init_info:
                print("Warmed up!")

    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        log_p, log_1_minus_p = binary_probs_to_logits(X)
        logits = log_p - log_1_minus_p
        ones = np.ones_like(logits)
        return np.hstack([ones, logits, np.square(logits)])

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        logits = self._get_logits(X)
        self.w_ = fit_quadratic_scaling(
            logits,
            y,
            penalty=self.penalty,
            reg_intercept=self.lambda_intercept,
            reg_quadratic=self.lambda_quadratic,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        logits = self._get_logits(X)
        logits = logits.dot(self.w_).reshape(-1, 1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.hstack([1 - probs, probs])


class BetacalCalibrator(Calibrator):
    """
    From https://github.com/betacal/python/

    Reference
    ---------
    Meelis Kull, Telmo Silva Filho, and Peter Flach. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary classifiers.
    International Conference on Artificial Intelligence and Statistics, 2017.
    """
    def __init__(self, parameters="abm"):
        super().__init__()
        try:
            from betacal import BetaCalibration
        except ImportError as e:
            raise ImportError("The 'betacal' package is required.") from e
        self.parameters = parameters

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        probs = flatten_binary_probs(X)
        from betacal import BetaCalibration

        self.cal_ = BetaCalibration(parameters=self.parameters)
        self.cal_.fit(probs, y)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        probs = flatten_binary_probs(X)
        # If logistic has coef[0] = 0 after .fit, an exeption occurs in .predict,
        # very rare
        try:
            probs = self.cal_.predict(probs)
        except Exception:
            # Manually, following the intial implem, removing the problematic if
            df = probs.reshape(-1, 1)
            eps = np.finfo(df.dtype).eps
            df = np.clip(df, eps, 1 - eps)

            x = np.hstack((df, 1.0 - df))
            x = np.log(x)
            x[:, 1] *= -1
            probs = self.cal_.calibrator_.lr_.predict_proba(x)[:, 1]

        probs = expand_binary_probs(probs)
        return probs
