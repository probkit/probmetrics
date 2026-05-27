import torch
import warnings
import numpy as np

from .base import Calibrator
from .sklearn import SklearnCalibrator

from probmetrics.utils import flatten_binary_probs, expand_binary_probs
from probmetrics.distributions import CategoricalDistribution, CategoricalProbs


class CenteredIsotonicRegressionCalibrator(Calibrator):
    """
    From https://github.com/mathijs02/cir-model

    Reference
    ---------
    Assaf P. Oron and Nancy Flournoy. Centered isotonic regression: point and interval
    estimation for dose-response studies. Statistics in Biopharmaceutical Research,
    9(3):258-267, 2017.
    """

    def __init__(self):
        super().__init__()
        try:
            from cir_model import CenteredIsotonicRegression
        except ImportError as e:
            raise ImportError("The 'cir_model' package is required.") from e

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO
        # We have to use float64 since with float32 it can happen that somewhere
        # internally a rounding error occurs (?) and interp1d thinks that it is asked to
        # extrapolate at the boundary value, which causes it to output nan values, which
        # are seen as a separate possible value but (y==value) is empty because nan==nan
        # is false, so an error occurs because the method attempts to take an average of
        # an empty set.
        from cir_model import CenteredIsotonicRegression

        self.cal_ = CenteredIsotonicRegression()

        probs = flatten_binary_probs(X)
        probs = probs.astype(np.float64)
        y = y.astype(np.float64)
        self.cal_.fit(probs, y)
        self.min_ = np.min(probs)
        self.max_ = np.max(probs)

    def _predict_proba_impl(self, X):
        probs = flatten_binary_probs(X)
        probs = probs.astype(np.float64)
        # have to clip since CenteredIsotonicRegression refuses to extrapolate (?)
        probs = np.clip(probs, self.min_, self.max_)
        probs = self.cal_.transform(probs)
        probs = np.clip(probs, 0.0, 1.0)
        probs = expand_binary_probs(probs)
        return probs


class BinaryVennAbersCalibrator(Calibrator):
    """
    From https://github.com/ip200/venn-abers

    Reference
    ---------
    Vladimir Vovk, Ivan Petej, and Valentina Fedorova. Large-scale probabilistic
    predictors with and without guarantees of validity. Advances in Neural Information
    Processing Systems, 2015.
    """

    def __init__(self):
        super().__init__()
        try:
            from venn_abers import VennAbers
        except ImportError as e:
            raise ImportError("The 'venn_abers' package is required.") from e

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from venn_abers import VennAbers

        self.cal_ = VennAbers()
        probs = flatten_binary_probs(X)
        probs = expand_binary_probs(probs)
        self.cal_.fit(probs, y)

    def _predict_proba_impl(self, X):
        probs = flatten_binary_probs(X)
        probs = expand_binary_probs(probs)
        return self.cal_.predict_proba(probs)[0]


class VennAbersCalibrator(Calibrator):
    """
    From https://github.com/ip200/venn-abers

    Reference
    ---------
    Vladimir Vovk, Ivan Petej, and Valentina Fedorova. Large-scale probabilistic
    predictors with and without guarantees of validity. Advances in Neural Information
    Processing Systems, 2015.
    """

    def __init__(self, use_ovo: bool = False):
        super().__init__()
        try:
            from venn_abers import VennAbersCalibrator
        except ImportError as e:
            raise ImportError("The 'venn_abers' package is required.") from e
        self.use_ovo = use_ovo

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from venn_abers import VennAbersCalibrator

        self.cal_ = VennAbersCalibrator()
        self.X_ = np.copy(X)
        self.y_ = np.copy(y)

    def _predict_proba_impl(self, X):
        return self.cal_.predict_proba(
            p_cal=self.X_,
            y_cal=self.y_,
            p_test=X,
            p0_p1_output=False,
            va_type="one_vs_one" if self.use_ovo else "one_vs_all",
        )


class NetcalENIRCalibrator(Calibrator):
    """
    From https://github.com/EFS-OpenSource/calibration-framework

    Reference
    ---------
    Naeini, Mahdi Pakdaman and Cooper, Gregory F.. Binary classifier calibration using
    an ensemble of near isotonic regression models. International Conference on Data
    Mining, 2016.
    """
    def __init__(self):
        super().__init__()
        try:
            from netcal.binning import ENIR
        except ImportError as e:
            raise ImportError("The 'netcal' package is required.") from e
        self.defaulted_to_IR = False

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from netcal.binning import ENIR

        self.cal_ = ENIR()
        probs = flatten_binary_probs(X)
        # netcal.binning.ENIR returns an error when .fit() is called with separable data
        # (AUC=1).
        try:
            self.cal_.fit(probs, y)
        except Exception as e:
            warnings.warn(
                f"ENIR failed during .fit(), defaulting to Isotonic regression."
            )
            self.cal_ = SklearnCalibrator(method="isotonic", cv="prefit")
            self.cal_.fit(probs, y)
            self.defaulted_to_IR = True

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        probs = flatten_binary_probs(X)
        if self.defaulted_to_IR:
            probs = self.cal_.predict_proba(probs)
        else:
            probs = self.cal_.transform(probs)
        probs = expand_binary_probs(probs)
        return probs


class KernelCalibrator(Calibrator):
    """
    Beta / Dirichlet kernel from
    https://github.com/tpopordanoska/ece-kde/blob/main/ece_kde.py, initially used to
    estimate calibration errors, converted to a post-hoc calibration method here.

    Reference
    ---------
    Teodora Popordanoska, Raphael Sayer, and Matthew Blaschko. A consistent and
    differentiable Lp canonical calibration error estimator. Advances in Neural
    Information Processing Systems, 2022.
    """

    def __init__(self, bandwidth=None, device="cpu"):
        """
        :param bandwidth: The bandwidth of the kernel. If None, LOO MLE is used to find
        it during fit.
        :param device: The device type: "cpu" or "cuda"
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.device = device

    def _get_heuristic_bandwidth(self, N: int, num_classes: int) -> float:
        """Select a default bandwidth using Scott's Rule adapted for the variance of
        Beta/Dirichlet kernels on a simplex."""
        # Intrinsic dimensionality: d = 1 for binary, d = C - 1 for multiclass
        d = num_classes - 1 if num_classes > 1 else 1
        return N ** (-2.0 / (d + 4.0))

    def _get_bandwidth(self, f):
        """
        Select a bandwidth based on maximizing the leave-one-out likelihood (LOO MLE).
        """
        bandwidths = torch.cat(
            (
                torch.logspace(start=-5, end=-1, steps=15),
                torch.linspace(0.2, 1, steps=5),
            )
        ).to(self.device)

        max_b = -1
        max_l = float("-inf")
        n = len(f)

        for b in bandwidths:
            log_kern = self._get_kernel_matrix(f, f, b, loo=True)
            log_fhat = torch.logsumexp(log_kern, 1) - torch.log(
                torch.tensor(n - 1.0, device=self.device)
            )
            l = torch.sum(log_fhat).item()

            if l > max_l:
                max_l = l
                max_b = b.item()

        return max_b

    def _get_kernel_matrix(self, z, zi, bandwidth, loo=False):
        """
        Computes the pairwise log-kernel matrix between evaluation points `z` and fitted
        points `zi`.
        """
        # Clamp inputs safely to prevent log(0)
        z_c = torch.clamp(z, 1e-10, 1 - 1e-10)
        zi_c = torch.clamp(zi, 1e-10, 1 - 1e-10)

        if z.shape[1] == 1:
            # Beta Kernel for Binary Classification
            z_uns = z_c.unsqueeze(1)  # Shape: (N, 1, 1)
            zi_uns = zi_c.unsqueeze(0)  # Shape: (1, M, 1)
            p = zi_uns / bandwidth + 1
            q = (1 - zi_uns) / bandwidth + 1

            log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
            log_num = (p - 1) * torch.log(z_uns) + (q - 1) * torch.log(1 - z_uns)
            log_kern = (log_num - log_beta).squeeze(-1)  # Shape: (N, M)
        else:
            # Dirichlet Kernel for Multiclass Classification
            alphas = zi_c / bandwidth + 1  # Shape: (M, C)
            log_beta = torch.sum(torch.lgamma(alphas), dim=1) - torch.lgamma(
                torch.sum(alphas, dim=1)
            )  # (M,)
            log_num = torch.matmul(torch.log(z_c), (alphas - 1).T)  # Shape: (N, M)
            log_kern = log_num - log_beta.unsqueeze(0)  # Shape: (N, M)

        if loo:
            # Trick: set diagonal to -inf for leave-one-out calculation
            diag_idx = torch.arange(len(z))
            log_kern[diag_idx, diag_idx] = torch.finfo(torch.float).min

        return log_kern

    def _fit_torch_impl(
        self,
        y_pred: CategoricalDistribution,
        y_true_labels: torch.Tensor,
        max_fit_samples: int = 10000,
    ):
        """
        Fits the kernel function by storing the calibration set and calculating the
        bandwidth.
        """
        with torch.no_grad():
            probs = y_pred.get_probs()

            if probs.ndim == 1:
                probs = probs.unsqueeze(1)

            self.num_classes_ = probs.shape[1]

            # TODO: SPEED FIX: Subsample the fit set
            N_total = probs.shape[0]
            if N_total > max_fit_samples:
                indices = torch.randperm(N_total, device=self.device)[:max_fit_samples]
                fit_probs = probs[indices]
                fit_labels = y_true_labels[indices]
            else:
                fit_probs = probs
                fit_labels = y_true_labels

            # TODO: we use heuristic bandwith instead of tuned bandwidth
            if self.bandwidth is None:
                self.bandwidth = self._get_heuristic_bandwidth(
                    len(fit_probs), self.num_classes_
                )
            # if self.bandwidth is None:
            #     self.bandwidth = self._get_bandwidth(fit_probs)

            self.X_fit_ = fit_probs
            self.y_fit_ = fit_labels

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        """
        Predicts calibrated probabilities by evaluating the Nadaraya-Watson estimator.
        """
        with torch.no_grad():
            X_t = y_pred.get_probs()
            is_1d = X_t.ndim == 1
            if is_1d:
                X_t = X_t.unsqueeze(1)

            log_kern = self._get_kernel_matrix(
                X_t, self.X_fit_, self.bandwidth, loo=False
            )
            log_den = torch.logsumexp(log_kern, dim=1)

            if self.num_classes_ == 1:
                # Binary Formulation
                log_kern_y = log_kern.clone()
                # Mask out entries where y != 1 (effectively multiplying by y in log
                # space)
                log_kern_y[:, self.y_fit_ == 0] = torch.finfo(torch.float).min

                log_num = torch.logsumexp(log_kern_y, dim=1)
                prob_1 = torch.exp(log_num - log_den)
                prob_0 = 1.0 - prob_1
                return CategoricalProbs(torch.stack([prob_0, prob_1], dim=1))

            else:
                # Canonical Multiclass Formulation
                calibrated_probs = []
                for k in range(self.num_classes_):
                    log_kern_k = log_kern.clone()
                    # Mask out entries where y != k
                    log_kern_k[:, self.y_fit_ != k] = torch.finfo(torch.float).min
                    log_num_k = torch.logsumexp(log_kern_k, dim=1)
                    calibrated_probs.append(torch.exp(log_num_k - log_den))

                res = torch.stack(calibrated_probs, dim=1)
                # return res.cpu().numpy()
                return CategoricalProbs(res)
