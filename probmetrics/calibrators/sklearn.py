import sklearn
import numpy as np

from .base import Calibrator

from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


class IdentityEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def fit(self, X, y):
        # somehow it can fail if we don't do the np.asarray()
        self.classes_ = np.asarray(list(range(self.n_classes)))
        # self.classes_ = np.unique(y)  # if we do this we have a problem for missing
        # classes
        return self

    def predict_proba(self, X):
        return X

    def predict(self, X):
        # having this as a dummy here for the FrozenEstimator solution
        # which hovewer doesn't work right now
        raise NotImplementedError()


class SklearnCalibrator(Calibrator):
    def __init__(self, method: str, cv: str = "prefit"):
        assert cv == "prefit", (
            "Other methods than prefit not intended and not implemented for sklearn > 1.6"
        )
        super().__init__()
        self.method = method
        self.cv = cv

    @staticmethod
    def _normalize_rows(A: np.ndarray) -> np.ndarray:
        """Row-normalize so rows sum to 1; safe if a row sums to 0."""
        s = A.sum(axis=1, keepdims=True)
        s = np.where(s == 0.0, 1.0, s)
        return A / s

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim == 1:
            X = np.stack((1.0 - X, X), axis=1)

        y = np.asarray(y)

        self.n_classes_ = X.shape[-1]
        self.classes_ = np.arange(self.n_classes_)

        # Fit label encoder up-front (used for the missing-class path)
        self.label_encoder_ = LabelEncoder()
        y_mapped = self.label_encoder_.fit_transform(y)
        self.present_classes_ = self.label_encoder_.classes_
        self.missing_classes_ = np.setdiff1d(self.classes_, self.present_classes_)

        # ---------- Case 1: no missing classes -> original formulation ----------
        if self.missing_classes_.size == 0:
            est = IdentityEstimator(self.n_classes_)
            est.fit(X, y)

            try:
                # sklearn >= 1.6
                from sklearn.frozen import FrozenEstimator

                self.calib_ = CalibratedClassifierCV(
                    FrozenEstimator(est), method=self.method
                )
            except ImportError:
                # sklearn < 1.6
                self.calib_ = CalibratedClassifierCV(
                    est, method=self.method, cv=self.cv
                )

            self.calib_.fit(X, y)
            return

        # --- Case 2: missing classes -> slice + normalize X_present, use y_mapped ---
        X_present = self._normalize_rows(X[:, self.present_classes_])

        est = IdentityEstimator(len(self.present_classes_))
        est.fit(X_present, y_mapped)

        try:
            from sklearn.frozen import FrozenEstimator

            self.calib_ = CalibratedClassifierCV(
                FrozenEstimator(est), method=self.method
            )
        except ImportError:
            self.calib_ = CalibratedClassifierCV(est, method=self.method, cv=self.cv)

        self.calib_.fit(X_present, y_mapped)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = np.stack((1.0 - X, X), axis=1)

        # If fit saw all classes, keep original behavior
        if self.missing_classes_.size == 0:
            return self.calib_.predict_proba(X)

        # Missing classes: normalize the same sliced X_present and zero-fill missing
        # classes (Option A)
        X_present = self._normalize_rows(X[:, self.present_classes_])
        p_present = self.calib_.predict_proba(X_present)

        p_full = np.zeros((X.shape[0], self.n_classes_), dtype=p_present.dtype)
        p_full[:, self.present_classes_] = p_present
        return p_full
