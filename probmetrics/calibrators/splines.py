import numpy as np

from .base import Calibrator
from probmetrics.utils import flatten_binary_probs, expand_binary_probs


class MLISplineCalibrator(Calibrator):
    """
    Uses smoothing splines to determine the calibration function.
    From https://github.com/numeristical/splinecalib

    Reference
    ---------
    Brian Lucena. Spline-based probability calibration. arXiv preprint arXiv:1809.07751,
    https://arxiv.org/abs/1809.07751, 2018.
    """
    def __init__(self):
        super().__init__()
        try:
            from splinecalib import SplineCalib
        except ImportError as e:
            raise ImportError("The 'splinecalib' package is required.") from e

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from splinecalib import SplineCalib

        self.cal_ = SplineCalib()
        if len(np.unique(X)) < 3:  # This causes a bug.
            self.cal_ = None
        else:
            self.cal_.fit(X, y)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        if self.cal_ is not None:
            probs = self.cal_.calibrate(X)
        else:
            probs = X
        probs = expand_binary_probs(probs)
        return probs


class _GuptaSpline:
    """
    Internal helper class for calculating smoothing splines.
    From https://github.com/kartikgupta-at-anu/spline-calibration
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, kx: np.ndarray, runout: str = "parabolic"
    ):
        self.kx = kx
        self.delta = kx[1] - kx[0]
        self.nknots = len(kx)
        self.runout = runout

        m_from_ky = self.ky_to_M()
        my_from_ky = np.concatenate([m_from_ky, np.eye(len(kx))], axis=0)
        y_from_my = self.my_to_y(x)
        y_from_ky = y_from_my @ my_from_ky

        # Find the least squares solution
        ky = np.linalg.lstsq(y_from_ky, y, rcond=-1)[0]

        self.ky = ky
        self.my = my_from_ky @ ky

    def my_to_y(self, vecx: np.ndarray) -> np.ndarray:
        ndata = len(vecx)
        mM = np.zeros((ndata, self.nknots))
        my = np.zeros((ndata, self.nknots))

        for i, xx in enumerate(vecx):
            j = int(np.floor((xx - self.kx[0]) / self.delta))
            j = max(0, min(j, self.nknots - 2))
            x = xx - j * self.delta

            mM[i, j] = (
                -(x**3) / (6.0 * self.delta) + x**2 / 2.0 - 2.0 * self.delta * x / 6.0
            )
            mM[i, j + 1] = x**3 / (6.0 * self.delta) - self.delta * x / 6.0
            my[i, j] = -x / self.delta + 1.0
            my[i, j + 1] = x / self.delta

        return np.concatenate([mM, my], axis=1)

    def my_to_dy(self, vecx: np.ndarray) -> np.ndarray:
        ndata = len(vecx)
        mM = np.zeros((ndata, self.nknots))
        my = np.zeros((ndata, self.nknots))

        for i, xx in enumerate(vecx):
            j = int(np.floor((xx - self.kx[0]) / self.delta))
            j = max(0, min(j, self.nknots - 2))
            x = xx - j * self.delta

            mM[i, j] = -(x**2) / (2.0 * self.delta) + x - 2.0 * self.delta / 6.0
            mM[i, j + 1] = x**2 / (2.0 * self.delta) - self.delta / 6.0
            my[i, j] = -1.0 / self.delta
            my[i, j + 1] = 1.0 / self.delta

        return np.concatenate([mM, my], axis=1)

    def ky_to_M(self) -> np.ndarray:
        A = 4.0 * np.eye(self.nknots - 2)
        for i in range(1, self.nknots - 2):
            A[i - 1, i] = 1.0
            A[i, i - 1] = 1.0

        if self.runout == "parabolic":
            A[0, 0] = 5.0
            A[-1, -1] = 5.0
        elif self.runout == "cubic":
            A[0, 0] = 6.0
            A[0, 1] = 0.0
            A[-1, -1] = 6.0
            A[-1, -2] = 0.0

        B = np.zeros((self.nknots - 2, self.nknots))
        for i in range(0, self.nknots - 2):
            B[i, i] = 1.0
            B[i, i + 1] = -2.0
            B[i, i + 2] = 1.0

        B = B * (6 / self.delta**2)

        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B

        if self.runout == "natural":
            z0 = np.zeros((1, self.nknots))
            z1 = np.zeros((1, self.nknots))
        elif self.runout == "parabolic":
            z0 = AinvB[0]
            z1 = AinvB[-1]
        elif self.runout == "cubic":
            z0 = 2.0 * AinvB[0] - AinvB[1]
            z1 = 2.0 * AinvB[-1] - AinvB[-2]

        z0 = z0.reshape((1, -1))
        z1 = z1.reshape((1, -1))

        return np.concatenate([z0, AinvB, z1], axis=0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.my_to_y(x) @ self.my

    def evaluate_deriv(self, x: np.ndarray) -> np.ndarray:
        return self.my_to_dy(x) @ self.my


class CDFSplineCalibrator(Calibrator):
    """
    Approximates the empirical cumulative distribution using a differentiable function
    via splines.

    Based on https://github.com/kartikgupta-at-anu/spline-calibration

    Reference
    ---------
    Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian
    Sminchisescu, and Richard Hartley. Calibration of neural networks using splines.
    International Conference on Learning Representations, 2021.
    """

    def __init__(self, spline_method: str = "natural", splines: int = 6):
        super().__init__()
        self.spline_method = spline_method
        self.splines = splines

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        probs = flatten_binary_probs(X)

        # Sort the data according to score
        order = probs.argsort()
        sorted_probs = probs[order]
        sorted_labels = y[order]

        nsamples = len(sorted_probs)
        integrated_accuracy = np.cumsum(sorted_labels) / nsamples
        integrated_scores = np.cumsum(sorted_probs) / nsamples
        percentiles = np.linspace(0.0, 1.0, nsamples)

        kx = np.linspace(0.0, 1.0, self.splines)
        spline = _GuptaSpline(
            percentiles,
            integrated_accuracy - integrated_scores,
            kx,
            runout=self.spline_method,
        )

        # Evaluate the spline and store arrays for interpolation
        acc = spline.evaluate_deriv(percentiles)
        acc += sorted_probs

        self.calib_scores_ = sorted_probs
        self.calib_acc_ = acc

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        probs = flatten_binary_probs(X)
        # Use numpy's optimized linear interpolation instead of a custom Python
        # implementation
        probs = np.interp(probs, self.calib_scores_, self.calib_acc_)
        probs = expand_binary_probs(probs)
        return probs
