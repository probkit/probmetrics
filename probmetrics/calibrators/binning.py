import numpy as np

from .base import Calibrator
from probmetrics.utils import flatten_binary_probs, expand_binary_probs


class BinaryHistogramBinningCalibrator(Calibrator):
    """
    Binary Histogram Binning Calibrator.

    This calibrator partitions the uncalibrated predicted probabilities into
    discrete bins and calculates the empirical probability of the positive
    class within each bin. During prediction, it maps uncalibrated probabilities
    to the empirical probability of their corresponding bin.

    Parameters
    ----------
    n_bins : int, default=10
        The number of bins to partition the probability space into.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        - 'uniform': Bins have identical widths (e.g., [0, 0.1), [0.1, 0.2)).
        - 'quantile': Bins have varying widths but contain approximately the
          same number of samples.

    Attributes
    ----------
    bin_edges_ : ndarray of shape (n_bins + 1,)
        The edges defining the bins. Learned during `fit`.
    bin_values_ : ndarray of shape (n_bins,)
        The calibrated probability assigned to each bin. Learned during `fit`.

    Reference
    ---------
    Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from
    decision trees and naive bayesian classifiers. International Conference on Machine
    Learning, 2001.
    """

    def __init__(self, n_bins: int = 10, strategy: str = "uniform"):
        super().__init__()

        if strategy not in ["uniform", "quantile"]:
            raise ValueError(
                f"Strategy must be 'uniform' or 'quantile', got {strategy!r}."
            )

        self.strategy = strategy
        self.n_bins = n_bins

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        probs = flatten_binary_probs(X)

        if self.strategy == "uniform":
            self.bin_edges_ = np.linspace(0.0, 1.0, self.n_bins + 1)
        else:
            self.bin_edges_ = np.unique(
                np.quantile(probs, np.linspace(0.0, 1.0, self.n_bins + 1))
            )
            if len(self.bin_edges_) == 1:
                # Fallback if variance is 0
                self.bin_edges_ = np.array([0.0, 1.0])

        bin_indices = np.digitize(probs, self.bin_edges_, right=False) - 1
        max_idx = len(self.bin_edges_) - 2

        # Ensures exact 1.0s fall in the final bin rather than creating an out-of-bounds
        # index
        bin_indices = np.clip(bin_indices, 0, max_idx)

        bin_sums = np.bincount(
            bin_indices, weights=y, minlength=len(self.bin_edges_) - 1
        )
        bin_counts = np.bincount(bin_indices, minlength=len(self.bin_edges_) - 1)

        # Fallback for empty bins: Global prior probability of the positive class
        # (Safer than midpoints if the underlying model is severely miscalibrated)
        global_prior = np.mean(y) if len(y) > 0 else 0.5
        fallback_values = np.full(max_idx + 1, global_prior)

        self.bin_values_ = np.divide(
            bin_sums, bin_counts, out=fallback_values, where=(bin_counts > 0)
        )

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        probs = flatten_binary_probs(X)
        bin_indices = np.digitize(probs, self.bin_edges_, right=False) - 1
        bin_indices = np.clip(
            bin_indices, 0, len(self.bin_edges_) - 2
        )  # Ensures exact 1.0s fall in the final bin
        probs = self.bin_values_[bin_indices]
        probs = expand_binary_probs(probs)
        return probs


class BinaryPlattBinnerCalibrator(Calibrator):
    """
    From https://github.com/p-lambda/verified_calibration

    Reference
    ---------
    Ananya Kumar, Percy S. Liang, and Tengyu Ma. Verified uncertainty calibration.
    Advances in neural information processing systems, 2019.
    """

    def __init__(self):
        super().__init__()
        try:
            from calibration import PlattBinnerCalibrator
        except ImportError as e:
            raise ImportError("The 'calibration' package is required.") from e

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        probs = flatten_binary_probs(X)
        from calibration import PlattBinnerCalibrator

        self.cal_ = PlattBinnerCalibrator(num_calibration=None, num_bins=10)
        self.cal_.train_calibration(probs, y)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        probs = flatten_binary_probs(X)
        probs = self.cal_.calibrate(probs)
        probs = expand_binary_probs(probs)
        return probs


class NetcalBBQCalibrator(Calibrator):
    """
    From https://github.com/EFS-OpenSource/calibration-framework

    Reference
    ---------
    Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well
    calibrated probabilities using bayesian binning. AAAI Conference on Artificial
    Intelligence, 2015.
    """

    def __init__(self):
        super().__init__()
        try:
            from netcal.binning import BBQ
        except ImportError as e:
            raise ImportError("The 'netcal' package is required.") from e

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from netcal.binning import BBQ

        self.cal_ = BBQ()
        self.cal_.fit(X, y)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        probs = self.cal_.transform(X)
        probs = expand_binary_probs(probs)
        return probs
