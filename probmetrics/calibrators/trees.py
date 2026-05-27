import warnings
import numpy as np
from .base import Calibrator
from probmetrics.utils import binary_probs_to_logits, multiclass_probs_to_logits


class LightGBMCalibrator(Calibrator):
    def __init__(self, use_cv: bool = True):
        super().__init__()
        try:
            import lightgbm as lgb
            from sklearn.model_selection import StratifiedKFold
        except ImportError as e:
            raise ImportError(
                "The 'lightgbm' and 'scikit-learn' packages are required."
            ) from e
        self.use_cv = use_cv

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        import lightgbm as lgb
        from sklearn.model_selection import StratifiedKFold

        self.n_classes_ = X.shape[1] if len(X.shape) > 1 and X.shape[1] > 1 else 2
        self.cal_models_ = []

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
            objective = "binary"
            metric = "binary_logloss"
        else:
            logits = multiclass_probs_to_logits(X)
            objective = "multiclass"
            metric = "multi_logloss"

        params = {
            "objective": objective,
            "metric": metric,
            "max_depth": 3,
            "num_leaves": 8,
            "verbose": -1,
        }
        if self.n_classes_ > 2:
            params["num_class"] = self.n_classes_

        if self.use_cv:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(logits, y):
                train_data = lgb.Dataset(logits[train_idx], label=y[train_idx])
                val_data = lgb.Dataset(
                    logits[val_idx], label=y[val_idx], reference=train_data
                )

                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
                self.cal_models_.append(model)
        else:
            train_data = lgb.Dataset(logits, label=y)

            model = lgb.train(params, train_data)
            self.cal_models_.append(model)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            logits = multiclass_probs_to_logits(X)

        all_preds = []
        for model in self.cal_models_:
            pred = model.predict(logits)
            if self.n_classes_ == 2:
                pred = np.vstack([1 - pred, pred]).T
            all_preds.append(pred)

        return np.mean(all_preds, axis=0)


class XGBoostCalibrator(Calibrator):
    def __init__(self, use_cv: bool = True):
        super().__init__()
        try:
            import xgboost as xgb
            from sklearn.model_selection import StratifiedKFold
        except ImportError as e:
            raise ImportError(
                "The 'xgboost' and 'scikit-learn' packages are required."
            ) from e
        self.use_cv = use_cv

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold

        self.n_classes_ = X.shape[1] if len(X.shape) > 1 and X.shape[1] > 1 else 2
        self.n_observed_classes_ = len(np.unique(y))

        if self.n_observed_classes_ != self.n_classes_:
            warnings.warn(
                "# classes X.shape[1] is different from # unique labels len(np.unique(y))"
            )
            self.observed_classes_ = np.unique(y)
            class_to_index = {
                cls: idx for idx, cls in enumerate(self.observed_classes_)
            }
            y_remapped = np.array([class_to_index[c] for c in y])
        else:
            y_remapped = y

        self.cal_models_ = []

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            logits = multiclass_probs_to_logits(X)
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "max_depth": 3,
        }
        if self.n_classes_ > 2:
            params["num_class"] = self.n_observed_classes_

        if self.use_cv:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(logits, y_remapped):
                train_kwargs = {"data": logits[train_idx], "label": y_remapped[train_idx]}
                val_kwargs = {"data": logits[val_idx], "label": y_remapped[val_idx]}

                dtrain = xgb.DMatrix(**train_kwargs)
                dval = xgb.DMatrix(**val_kwargs)

                model = xgb.train(
                    params,
                    dtrain,
                    evals=[(dval, "eval")],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
                self.cal_models_.append(model)
        else:
            dtrain_kwargs = {"data": logits, "label": y_remapped}
            dtrain = xgb.DMatrix(**dtrain_kwargs)
            model = xgb.train(params, dtrain)
            self.cal_models_.append(model)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            logits = multiclass_probs_to_logits(X)

        dtest_kwargs = {"data": logits}
        dtest = xgb.DMatrix(**dtest_kwargs)

        all_preds = []
        for model in self.cal_models_:
            pred = model.predict(dtest)
            if self.n_classes_ == 2:
                pred = np.vstack([1 - pred, pred]).T
            all_preds.append(pred)
        preds = np.mean(all_preds, axis=0)

        if self.n_observed_classes_ != self.n_classes_:
            full_preds = np.zeros((preds.shape[0], self.n_classes_))
            full_preds[:, self.observed_classes_] = preds
            preds = full_preds

        return preds


class CatBoostCalibrator(Calibrator):
    def __init__(self, use_cv: bool = True):
        super().__init__()
        try:
            from catboost import CatBoostClassifier, Pool
            from sklearn.model_selection import StratifiedKFold
        except ImportError as e:
            raise ImportError(
                "The 'catboost' and 'scikit-learn' packages are required."
            ) from e
        self.use_cv = use_cv

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from catboost import CatBoostClassifier, Pool
        from sklearn.model_selection import StratifiedKFold

        self.n_classes_ = X.shape[1] if len(X.shape) > 1 and X.shape[1] > 1 else 2
        self.n_observed_classes_ = len(np.unique(y))

        if self.n_observed_classes_ != self.n_classes_:
            warnings.warn(
                "# classes X.shape[1] is different from # unique labels len(np.unique(y))"
            )
            self.observed_classes_ = np.unique(y)
            class_to_index = {
                cls: idx for idx, cls in enumerate(self.observed_classes_)
            }
            y_remapped = np.array([class_to_index[c] for c in y])
        else:
            y_remapped = y

        self.cal_models_ = []

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            logits = multiclass_probs_to_logits(X)

        if self.use_cv:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(logits, y_remapped):
                train_kwargs = {"data": logits[train_idx], "label": y_remapped[train_idx]}
                val_kwargs = {"data": logits[val_idx], "label": y_remapped[val_idx]}

                train_pool = Pool(**train_kwargs)
                val_pool = Pool(**val_kwargs)

                model = CatBoostClassifier(
                    depth=3,
                    loss_function="Logloss" if self.n_classes_ == 2 else "MultiClass",
                    iterations=1000,
                    early_stopping_rounds=50,
                    verbose=False,
                )
                model.fit(train_pool, eval_set=val_pool, verbose=False)
                self.cal_models_.append(model)
        else:
            pool_kwargs = {"data": logits, "label": y_remapped}
            pool = Pool(**pool_kwargs)

            model = CatBoostClassifier(depth=3, verbose=False)
            model.fit(pool)
            self.cal_models_.append(model)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        from catboost import Pool

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            logits = multiclass_probs_to_logits(X)

        pool_kwargs = {"data": logits}
        pool = Pool(**pool_kwargs)

        preds = [model.predict_proba(pool) for model in self.cal_models_]
        preds = np.mean(preds, axis=0)
    
        if self.n_observed_classes_ != self.n_classes_:
            full_preds = np.zeros((preds.shape[0], self.n_classes_))
            full_preds[:, self.observed_classes_] = preds
            preds = full_preds

        return preds


class BinaryCatBoostCalibrator(Calibrator):
    def __init__(self, tiny=False, monotone=False):
        super().__init__()
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError as e:
            raise ImportError("The 'catboost' packages is required.") from e
        self.tiny = tiny
        self.monotone = monotone
        self.model_params = {
            "verbose": 0,
            "allow_writing_files": False,
            "iterations": 100,
        }

        if self.tiny:
            self.model_params["depth"] = 3

        if self.monotone:
            self.model_params["monotone_constraints"] = "(1)"

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from catboost import CatBoostClassifier, Pool

        self.n_classes_ = X.shape[1] if len(X.shape) > 1 and X.shape[1] > 1 else 2

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            raise ValueError("This calibrator is made for binary predictions only.")

        pool_kwargs = {"data": logits, "label": y}
        pool = Pool(**pool_kwargs)
        model = CatBoostClassifier(**self.model_params)
        model.fit(pool)
        self.cal_ = model

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        from catboost import Pool

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            raise ValueError("This calibrator is made for binary predictions only.")

        pool_kwargs = {"data": logits}
        pool = Pool(**pool_kwargs)

        return self.cal_.predict_proba(pool)


class InitLogitCatBoostCalibrator(Calibrator):
    def __init__(self, init_logits=True):
        super().__init__()
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError as e:
            raise ImportError("The 'catboost' package is required.") from e
        self.init_logits = init_logits
        self.model_params = {
            "iterations": 50,
            "learning_rate": 0.1,
            "depth": 4,
            "verbose": 0,
            "allow_writing_files": False,
        }

    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        from catboost import CatBoostClassifier, Pool

        self.n_classes_ = X.shape[1] if len(X.shape) > 1 and X.shape[1] > 1 else 2

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            logits = multiclass_probs_to_logits(X)

        pool_kwargs = {"data": logits, "label": y}
        if self.init_logits:
            pool_kwargs["baseline"] = logits
        pool = Pool(**pool_kwargs)

        self.cal_ = CatBoostClassifier(**self.model_params)
        self.cal_.fit(pool)

    def _predict_proba_impl(self, X: np.ndarray) -> np.ndarray:
        from catboost import Pool

        if self.n_classes_ == 2:
            log_p, log_1_minus_p = binary_probs_to_logits(X)
            logits = log_p - log_1_minus_p
        else:
            logits = multiclass_probs_to_logits(X)

        pool_kwargs = {"data": logits}
        if self.init_logits:
            pool_kwargs["baseline"] = logits

        pool = Pool(**pool_kwargs)

        return self.cal_.predict_proba(pool)
