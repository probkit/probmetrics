import torch
import sklearn
import numpy as np

from probmetrics.distributions import (
    CategoricalDistribution,
    CategoricalProbs,
)
from probmetrics.utils import validate_probabilities

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class Calibrator(BaseEstimator, ClassifierMixin):
    """
    Calibrator base class. To implement,
    - override at least one of (_fit_impl, _fit_torch_impl)
    - override at least one of (_predict_proba_impl, _predict_proba_torch_impl)
    """

    def __init__(self):
        assert self.__class__.fit == Calibrator.fit
        assert self.__class__.fit_torch == Calibrator.fit_torch
        assert self.__class__.predict_proba == Calibrator.predict_proba
        assert self.__class__.predict_proba_torch == Calibrator.predict_proba_torch

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Calibrator":
        """Maps to _fit_impl if implemented, falls back on _fit_torch_impl otherwise.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array, got {type(y).__name__}")

        validate_probabilities(X)

        self.classes_ = list(range(X.shape[-1]))

        # If subclass implemented the numpy version, use it
        if self.__class__._fit_impl != Calibrator._fit_impl:
            self._fit_impl(X, y)
            return self

        # Fallback to torch routing if numpy is not implemented
        if self.__class__._fit_torch_impl != Calibrator._fit_torch_impl:
            self._fit_torch_impl(
                y_pred=CategoricalProbs(torch.as_tensor(X)),
                y_true_labels=torch.as_tensor(y, dtype=torch.long),
            )
            return self

        raise NotImplementedError(
            "Subclasses must implement _fit_impl or _fit_torch_impl."
        )

    def fit_torch(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ) -> "Calibrator":
        """Maps to _fit_torch_impl if implemented, falls back on _fit_impl otherwise.
        """
        if not isinstance(y_true_labels, torch.Tensor):
            raise TypeError(
                f"y_true_labels must be a torch.Tensor, got {type(y_true_labels).__name__}"
            )
        if not isinstance(y_pred, CategoricalDistribution):
            raise TypeError(
                f"y_pred must be a CategoricalDistribution, got {type(y_pred).__name__}"
            )
        # default implementation, using sklearn
        self.classes_ = list(range(y_pred.get_n_classes()))

        # If subclass implemented the torch version, use it
        if self.__class__._fit_torch_impl != Calibrator._fit_torch_impl:
            self._fit_torch_impl(y_pred, y_true_labels)
            return self

        # Fallback to numpy routing if torch is not implemented
        if self.__class__._fit_impl != Calibrator._fit_impl:
            self._fit_impl(
                y_pred.get_probs().detach().cpu().numpy(),
                y_true_labels.detach().cpu().numpy(),
            )
            return self

        raise NotImplementedError(
            "Subclasses must implement _fit_impl or _fit_torch_impl."
        )

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ) -> None:
        raise NotImplementedError()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Maps to _predict_proba_impl if implemented, falls back on
        _predict_proba_torch_impl otherwise.
        """
        check_is_fitted(self)
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}")
        validate_probabilities(X)

        # If subclass implemented the numpy version, use it
        if self.__class__._predict_proba_impl != Calibrator._predict_proba_impl:
            return self._predict_proba_impl(X)

        # Fallback to torch routing if numpy is not implemented
        if (
            self.__class__._predict_proba_torch_impl
            != Calibrator._predict_proba_torch_impl
        ):
            return (
                self._predict_proba_torch_impl(CategoricalProbs(torch.as_tensor(X)))
                .get_probs()
                .numpy()
            )

        raise NotImplementedError(
            "Subclasses must implement _predict_proba_impl or _predict_proba_torch_impl."
        )

    def predict_proba_torch(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        """Maps to _predict_proba_torch_impl if implemented, falls back on
        _predict_proba_impl otherwise.
        """
        check_is_fitted(self)
        if not isinstance(y_pred, CategoricalDistribution):
            raise TypeError(
                f"y_pred must be a CategoricalDistribution, got {type(y_pred).__name__}"
            )

        # If subclass implemented the torch version, use it
        if (
            self.__class__._predict_proba_torch_impl
            != Calibrator._predict_proba_torch_impl
        ):
            return self._predict_proba_torch_impl(y_pred)

        # Fallback to numpy routing if torch is not implemented
        if self.__class__._predict_proba_impl != Calibrator._predict_proba_impl:
            y_pred_probs = y_pred.get_probs()
            probs = self._predict_proba_impl(y_pred_probs.detach().cpu().numpy())
            return CategoricalProbs(
                torch.as_tensor(
                    probs, device=y_pred_probs.device, dtype=y_pred_probs.dtype
                )
            )

        raise NotImplementedError(
            "Subclasses must implement _predict_proba_impl or _predict_proba_torch_impl."
        )
    
    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        raise NotImplementedError()

    def predict(self, X):
        y_probs = self.predict_proba(X)
        class_idxs = np.argmax(y_probs, axis=-1)
        return np.asarray(self.classes_)[class_idxs]


class ApplyToLogitsCalibrator(Calibrator):
    def __init__(self, calib: BaseEstimator):
        super().__init__()
        self.calib = calib

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.calib_ = sklearn.base.clone(self.calib)
        self.calib_.fit(np.log(X + 1e-10), y)

    def _predict_proba_impl(self, X):
        return self.calib_.predict_proba(np.log(X + 1e-10))


class MulticlassOneVsOneCalibrator(Calibrator):
    def __init__(self, binary_calibrator: BaseEstimator):
        super().__init__()
        self.binary_calibrator = binary_calibrator

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes_ = X.shape[-1]
        self.bin_calibs_ = []

        if self.n_classes_ == 2:
            # binary classification
            bin_calib = sklearn.base.clone(self.binary_calibrator)
            bin_calib.fit(X, y)
            self.bin_calibs_.append(bin_calib)
        else:
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    idxs = np.logical_or((y == i), (y == j))
                    # idxs = np.arange(X.shape[0])
                    bin_probs = np.stack([X[idxs, j], X[idxs, i]], axis=-1)
                    bin_probs += 1e-30
                    bin_probs /= np.sum(bin_probs, axis=-1, keepdims=True)
                    # print(f'{np.any(np.isnan(bin_probs))=}')
                    bin_labels = (y[idxs] == i).astype(np.int32)
                    bin_calib = sklearn.base.clone(self.binary_calibrator)
                    bin_calib.fit(bin_probs, bin_labels)
                    self.bin_calibs_.append(bin_calib)

    def _predict_proba_impl(self, X):
        if self.n_classes_ == 2:
            # binary classification
            return self.bin_calibs_[0].predict_proba(X)
        else:
            # use PKPD formula in http://proceedings.mlr.press/v60/manokhin17a/manokhin17a.pdf
            pair_probs = [[None] * self.n_classes_ for i in range(self.n_classes_)]
            multi_probs = []
            calib_idx = 0
            for i in range(self.n_classes_):
                for j in range(i + 1, self.n_classes_):
                    bin_probs = np.stack([X[:, j], X[:, i]], axis=-1)
                    bin_probs += 1e-30
                    bin_probs /= np.sum(bin_probs, axis=-1, keepdims=True)
                    pred_probs = self.bin_calibs_[calib_idx].predict_proba(bin_probs)
                    pair_probs[i][j] = pred_probs[:, 1]
                    pair_probs[j][i] = pred_probs[:, 0]
                    # if i == 0 and j == 1:
                    #     print(f'{bin_probs=}')
                    #     print(f'{pred_probs=}')
                    calib_idx += 1

            for i in range(self.n_classes_):
                sum_inv_probs = sum(
                    [
                        1.0 / (1e-30 + pair_probs[i][j])
                        for j in range(self.n_classes_)
                        if j != i
                    ]
                )
                multi_probs.append(
                    1.0 / np.clip(sum_inv_probs - (self.n_classes_ - 2), 1e-30, np.inf)
                )

            multi_probs = np.stack(multi_probs, axis=-1)
            multi_probs = np.clip(multi_probs, a_min=1e-30, a_max=np.inf)
            multi_probs = multi_probs / np.sum(multi_probs, axis=-1, keepdims=True)
            return multi_probs


class MulticlassOneVsRestCalibrator(Calibrator):
    def __init__(self, binary_calibrator: BaseEstimator):
        super().__init__()
        self.binary_calibrator = binary_calibrator

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes_ = X.shape[-1]
        self.bin_calibs_ = []

        X = X.copy()
        X[np.isnan(X)] = 0.0

        if self.n_classes_ == 2:
            # binary classification
            bin_calib = sklearn.base.clone(self.binary_calibrator)
            bin_calib.fit(X, y)
            self.bin_calibs_.append(bin_calib)
        else:
            for i in range(self.n_classes_):
                pos_probs = X[:, i]
                neg_probs = 1.0 - pos_probs
                bin_probs = np.stack([neg_probs, pos_probs], axis=-1)
                bin_labels = (y == i).astype(np.int32)
                bin_calib = sklearn.base.clone(self.binary_calibrator)
                bin_calib.fit(bin_probs, bin_labels)
                self.bin_calibs_.append(bin_calib)

    def _predict_proba_impl(self, X):
        X = X.copy()
        X[np.isnan(X)] = 0.0
        if self.n_classes_ == 2:
            # binary classification
            return self.bin_calibs_[0].predict_proba(X)
        else:
            multi_probs = []
            for i in range(self.n_classes_):
                pos_probs = X[:, i]
                neg_probs = 1.0 - pos_probs
                bin_probs = np.stack([neg_probs, pos_probs], axis=-1)
                pred_probs = self.bin_calibs_[i].predict_proba(bin_probs)
                multi_probs.append(pred_probs[:, 1])
            multi_probs = np.stack(multi_probs, axis=-1)
            multi_probs = np.clip(multi_probs, a_min=1e-30, a_max=np.inf)
            multi_probs = multi_probs / np.sum(multi_probs, axis=-1, keepdims=True)
            return multi_probs


class MixtureCalibrator(Calibrator):
    def __init__(
        self,
        calibrator: Calibrator,
        output_constant: float = 1.0,
        input_constant: float = 0.0,
    ):
        super().__init__()
        self.calibrator = calibrator
        self.output_constant = output_constant
        self.input_constant = input_constant

    def _get_mixture(self, dist_1: CategoricalDistribution, unif_coef: float):
        probs = dist_1.get_probs()
        unif_probs = (1.0 / probs.shape[-1]) * torch.ones_like(probs)
        return CategoricalProbs((1.0 - unif_coef) * probs + unif_coef * unif_probs)

    def _fit_torch_impl(
        self, y_pred: CategoricalDistribution, y_true_labels: torch.Tensor
    ):
        self.n_samples_ = y_pred.get_n_samples()
        if self.input_constant != 0.0:
            y_pred = self._get_mixture(
                y_pred, self.input_constant / (self.n_samples_ + 1)
            )
        self.cal_ = sklearn.base.clone(self.calibrator)
        self.cal_.fit_torch(y_pred, y_true_labels)

    def _predict_proba_torch_impl(
        self, y_pred: CategoricalDistribution
    ) -> CategoricalDistribution:
        if self.input_constant != 0.0:
            y_pred = self._get_mixture(
                y_pred, self.input_constant / (self.n_samples_ + 1)
            )
        y_pred = self.cal_.predict_proba_torch(y_pred)
        if self.output_constant != 0.0:
            y_pred = self._get_mixture(
                y_pred, self.output_constant / (self.n_samples_ + 1)
            )

        return y_pred


class WrapCalibratorAsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, calib: BaseEstimator):
        self.calib = calib

    def fit(self, X, y):
        # print(f'Fitting calibrator: {self.calib}')
        self.calib_ = sklearn.base.clone(self.calib)
        self.calib_.fit(X, y)
        self.classes_ = list(range(np.max(y) + 1))
        return self

    def predict_proba(self, X):
        return self.calib_.predict_proba(X)
