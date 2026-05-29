[![test](https://github.com/dholzmueller/probmetrics/actions/workflows/testing.yml/badge.svg)](https://github.com/dholzmueller/probmetrics/actions/workflows/testing.yml)
[![Downloads](https://img.shields.io/pypi/dm/probmetrics)](https://pypistats.org/packages/probmetrics)


# Probmetrics: Classification metrics and post-hoc calibration

This package (PyTorch-based) currently contains
- post-hoc calibration methods, in particular:
  - a fast and accurate temperature scaling implementation described in [1]
  - an implementation of structured matrix scaling (SMS), a regularized version of matrix scaling introduced in [2]
  - implementations for all the post-hoc calibration methods in the CalArena benchmark [4]
- classification metrics, especially metrics for assessing the quality of probabilistic
  predictions, in particular:
  - our classifier based $L_p$ calibration error estimators from [3]

It accompanies our papers  
[1] [Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195)
(see also: [vision experiments](https://github.com/eugeneberta/RefineThenCalibrate-Vision), [tabular experiments](https://github.com/dholzmueller/pytabkit), [theory](https://github.com/eugeneberta/RefineThenCalibrate-Theory))  
[2] [Structured Matrix Scaling for Multi-Class Calibration](https://arxiv.org/abs/2511.03685)
(see also: [experiments](https://github.com/eugeneberta/LogisticCalibrationBenchmark))  
[3] [A Variational Estimator for Lp Calibration Errors](https://arxiv.org/abs/2602.24230)
(see also [all experiments](https://github.com/ElSacho/Evaluating_Lp_Calibration_Errors))  
[4] [CalArena: A Large-Scale Post-Hoc Calibration Benchmark](https://arxiv.org/abs/2605.30188)
(see also [leaderboards and experiments](https://github.com/probkit/CalArena))  

Please cite us if you use this repository for research purposes.


## Installation

Probmetrics is available via
```bash
pip install probmetrics
```
To obtain all functionality, run `pip install 'probmetrics[extra,dev,dirichletcal]'`.
- extra installs:
  - numba for SAGA based optimizers in logistic calibrators SVS, SMS and others.
  - relplot for smooth ECE (only works with scikit-learn versions <= 1.6).
  - catboost, lightgbm for our classifier-based $L_p$ calibration error metrics.
  - other packages supporting more post-hoc calibration methods: netcal (BBQ, ENIR, Netcal TS),
    uncertainty-calibration (Scaling-binning), betacal (Beta calibration), splinecalib
    (Spline calibration), venn-abers (Venn-Abers calibration), cir-model (CIR), scipy
    (ETS).
- dev installs more packages for development (especially testing)
- dirichletcal installs Dirichlet calibration, which however only works for Python 3.12
upwards.

## Using post-hoc calibration methods

You can create a calibrator by importing it from `probmetrics.calibrators`
```python
from probmetrics.calibrators import TemperatureScalingCalibrator

calib = TemperatureScalingCalibrator()
```

or using the `get_calibrator` factory function:
```python
from probmetrics.calibrators import get_calibrator

calib = get_calibrator("logistic")
```

<!-- Below are the main supported methods:
- `'logistic'` defaults to structured matrix scaling (SMS) for multiclass 
  and quadratic scaling for binary calibration. 
  We recommend using `'logistic'` for best results, especially on multiclass problems. 
  It can be slow for larger numbers of classes. Only runs on CPU. 
  For the SAGA version (not the default),
  the first call is slower due to numba compilation.
- `'svs'`: Structured vector scaling (SVS) for multiclass problems, 
  faster than SMS for multiclass while being almost as good in many cases.
- `'affine-scaling'`: Affine scaling for binary problems, 
  underperforms `'logistic'` (quadratic scaling) in our benchmarks but preserves AUC.
- `'temp-scaling'`: Our 
  [highly efficient implementation of temperature scaling](https://arxiv.org/abs/2501.19195)
  that, unlike some other implementations, does not suffer from optimization issues. 
  Temperature scaling is not as expressive as matrix or vector scaling variants,
  but it is faster and has the least overfitting risk.
- `'ts-mix'`: Same as `'temp-scaling'` but with Laplace smoothing 
  (slightly preferable for logloss). Can also be achieved using 
  `get_calibrator('temp-scaling', calibrate_with_mixture=True)`
- `'isotonic'` Isotonic regression from scikit-learn. 
  Isotonic variants can be good for binary classification with enough data (around 10K samples or more)
- `'ivap'` Inductive Venn-Abers predictor (a version of isotonic regression, slow but a bit better)
- `'cir'` Centered isotonic regression (slightly better and slower than isotonic)
- `'dircal'` Dirichlet calibration (slow, logistic performs better in our experiments)
- `'dircal-cv'` Dirichlet calibration optimized with cross-validation (very slow) -->

You can find all post-hoc calibration functions implemented in `probmetrics/calibrators`.
A subset of those methods is supported by the `get_calibrator` function, find details in `probmetrics/calibrators/factory.py`.


### Usage with numpy

```python
import numpy as np

probas = np.asarray([[0.1, 0.9]])  # shape = (n_samples, n_classes)
labels = np.asarray([1])  # shape = (n_samples,)

calib.fit(probas, labels)
calibrated_probas = calib.predict_proba(probas) # shape = (n_samples, n_classes)
```

### Usage with PyTorch

The PyTorch version can be used directly with GPU tensors, which is leveraged by our
temperature scaling implementation but not by most other methods.
For temperature scaling, this could accelerate things, but the CPU version can be faster 
for smaller validation sets (around 1K-10K samples).

```python
from probmetrics.distributions import CategoricalProbs
import torch

probas = torch.as_tensor([[0.1, 0.9]])
labels = torch.as_tensor([1])

# if you have logits, you can use CategoricalLogits instead
calib.fit_torch(CategoricalProbs(probas), labels)
result = calib.predict_proba_torch(CategoricalProbs(probas))
calibrated_probas = result.get_probs()
```

## Using metrics

We provide classification and calibration metrics, which can be used as follows:

```python
import torch
from probmetrics.metrics import Metrics

# computing multiple metrics at once is more efficient than computing them individually
metrics = Metrics.from_names([
    "logloss",
    "brier", # for binary, this is 2x the brier from sklearn
    "accuracy",
    "class-error",
    "auroc-ovr", # one-vs-rest
    "auroc-ovo-sklearn", # one-vs-one (can be slow!)
    "ece-15", # Expected calibration error with 15 bins
    "rmsce-15", # Root mean squared calibration error with 15 bins
    "mce-15", # Maximum calibration error with 15 bins
    "smece", # Smooth ECE, requires the relplot package.
])

y_true = torch.tensor(...)
y_pred = torch.tensor(...)

results = metrics.compute_all_from_labels_probs(y_true, y_pred)
print(results["brier"].item())
```

While there are some classes for regression metrics, they are not implemented yet.

The following function returns a list of all metric names:
```python
from probmetrics.metrics import Metrics, MetricType
Metrics.get_available_names(metric_type=MetricType.CLASS)
```

### Using our refinement and calibration metrics

We provide estimators for refinement error (loss after post-hoc calibration) and
calibration error (loss improvement through post-hoc calibration).
They can be used as follows:

```python
from probmetrics.metrics import Metrics

metrics = Metrics.from_names([
    # Mean test logloss after post-hoc calibration with ts-mix:
    "refinement_logloss_ts-mix_all",

    # Difference in mean test logloss before/after calibration with ts-mix:
    "calib-err_logloss_ts-mix_all",

    # Same thing for Brier score:
    "refinement_brier_ts-mix_all",
    "calib-err_brier_ts-mix_all",

    # Using the L1, L2 and Linf calibration error estimators (CatBoost based) described
    # in [3]:
    "calib-err_proper-L1-binary-as-1d_WS_CatboostClassifier_all",
    "calib-err_proper-L2-binary-as-1d_WS_CatboostClassifier_all",
    "calib-err_proper-Linf-binary-as-1d_WS_CatboostClassifier_all",
])

y_true = torch.tensor(...)
y_logits = torch.tensor(...)

results = metrics.compute_all_from_labels_logits(y_true, y_logits)
print(results['refinement_logloss_ts-mix_all'].item())
```

### Advanced calibration, confidence, and top-class metrics

Beyond standard metrics, you can evaluate proper Lp calibration errors for 
any p, as well as isolate specific types of errors like over-confidence, 
under-confidence, and top-class errors. 

```python
from probmetrics.metrics import (
  ProperLpLoss,
  BrierLoss,
  OverConfidenceLoss,
  UnderConfidenceLoss,
  TopClassLoss
)

# Evaluate proper Lp calibration errors for any p
lp_loss_l1 = ProperLpLoss(p=1)  # Evaluate E[ \| Y - E[Y|f(X)] \|_1 ] 
lp_loss_l2 = ProperLpLoss(p=2)  # Evaluate E[ \| Y - E[Y|f(X)] \|_2 ] 

# Evaluate over-confidence and under-confidence 
# (Initialize via string name or by passing a metric object)
over_brier = OverConfidenceLoss.from_name("brier")
under_L1 = UnderConfidenceLoss.from_name("proper-L1")

# Evaluate top-class error with any accompanying loss
topclass_brier = TopClassLoss(BrierLoss(binary_as_multiclass=False))
topclass_L1 = TopClassLoss.from_name("proper-L1")

# Compose wrappers (e.g., top-class with underconfidence for proper-L1)
under_topclass_l1 = TopClassLoss(UnderConfidenceLoss.from_name("proper-L1"))
over_topclass_brier = TopClassLoss(OverConfidenceLoss(BrierLoss()))

# Some metrics are listed by default, here are some of them
metrics = metrics = Metrics.from_names([
    # Estimate  E[ \| Y - E[Y|f(X)] \|_1 ] and treat binary predictions as scalars with
    # shapes (n,1)
    "proper-L1-binary-as-1d"

    # Estimate  E[ \| Y - E[Y|f(X)] \|_2 ] and treat binary predictions as vectors with
    # shapes (n,2)
    "proper-L2",

    "topclass-proper-L1-binary-as-1d", # Estimate L1 calibration error of top class 
    "topclass-under-proper-L1-binary-as-1d", # Estimate L1-overconfidence of top class 
    "topclass-over-proper-L1-binary-as-1d", # Estimate L1-underconfidence of top class
])
```

> **Note:** Over- and under-confidence metrics are designed for binary classification.
For multi-class in a top-class fashion, please use `TopClassLoss(OverConfidenceLoss(your_metric))`.

Once those losses are defined, you can evaluate the calibration error by doing:

```python
from probmetrics.metrics import MetricsWithCalibration, CombinedMetrics
from probmetrics.classifiers import WS_CatboostClassifier, WS_LGBMClassifier
from probmetrics.splitters import CVSplitter

loss = ProperLpLoss(p=2) 

metrics = MetricsWithCalibration(
    loss,
    calibrator=WS_CatboostClassifier(), # classifier used to recalibrate the predictions
    val_splitter=CVSplitter(n_cv=5) # cross-validation splitter
)

# Or use combined metrics to evaluate multiple metrics while fitting the post-hoc
# calibrator only once
combined_losses = CombinedMetrics([
    ProperLpLoss(p=1), 
    OverConfidenceLoss.from_name("brier"), 
    OverConfidenceLoss.from_name("proper-L1") , 
    UnderConfidenceLoss.from_name("proper-L1" ), 
    UnderConfidenceLoss(BrierLoss()),
    BrierLoss()
])

metrics = MetricsWithCalibration(
    combined_losses,
    calibrator=WS_LGBMClassifier(), 
    val_splitter=CVSplitter(n_cv=5)
)

y_true = torch.tensor(...)
y_prob = torch.tensor(...)

results = metrics.compute_all_from_labels_probs(y_true, y_prob)
```

The `calibrator` argument is used to recalibrate the original predictions. 
Any class inheriting from sklearn.base.ClassifierMixin (i.e., follows the scikit-learn 
classifier API) and implementing `predict_proba()` can be used.
We recommend using `WS_CatboostClassifier` with default parameters. 
"WS" stands for "Warm Started", as predictions are initialized at the original predicted
$f(x)$ values (see [3]). 

#### Binary vs. multiclass formatting

The library internally stores predictions in a multiclass format with shape
`(n_samples, n_classes)`.
For binary classification, for some metrics you can control whether to treat the output
as a two-column distribution or a single-column probability using the
`binary_as_multiclass` parameter.
For example, for `BrierLoss()`, using `binary_as_multiclass=False` will yield the
scikit-learn formula, while `binary_as_multiclass=True` will yield twice the value.

Setting `binary_as_multiclass=False` tells the loss function to treat 
`(n_samples, 2)` predictions as a single-column `(n_samples, 1)` probability.
The loss then internally transforms the data to 
binary labels $Y \in {0, 1}$ and the probability
column $f(X) \in [0, 1]$ for the calculation.

Those features are also valid with the `TopClassLoss`.
The `TopClassLoss` wrapper focuses the loss calculation on the class with 
the highest predicted probability. The behavior changes based on your binary setting,
for instance:

| Configuration                                                | Estimate                                                                         | Description                                                                                                                                                                                                                                 |
|:-------------------------------------------------------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TopClassLoss(ProperLpLoss(p=1))`                            | $\mathbb{E}[ \lvert Z - \mathbb{E}[Z \mid \max f(X)] \rvert ]$                   | Scalar probability: $\max f(X)$ is the scalar probability of the top class of $f(X)$; $Z \in \{0, 1\}$ equals $1$ if the label is what the top-class predicted and $0$ otherwise. Evaluates the absolute error of the top-class prediction. |
| `TopClassLoss(ProperLpLoss(p=1, binary_as_multiclass=True))` | $\mathbb{E}[ \Vert \mathbf{Z} - \mathbb{E}[\mathbf{Z} \mid \max f(X)] \Vert_1 ]$ | Vectorized: $\mathbf{Z}$ is a one-hot vector. Calculates the $L_1$ norm of the error vector.                                                                                                                                                |

When used inside `MetricsWithCalibration`, `TopClassLoss` will choose the top-class 
based on $f(X)$ instead of $g(f(X))$ so the loss difference uses the same choice of top class for both terms.

## Contributors
- David Holzmüller
- Eugène Berta
- Sacha Braun

## Releases

- v1.3.0 by [@eugeneberta](https://github.com/eugeneberta):
  - Splitted calibrators.py into several subfiles and implemented many new calibrators,
  most of which are benchmarked and ranked in the CalArena leaderboard.
  Among others, added Binary histogram binning, Scaling-binning (from
  uncertainty-calibration), BBQ (from netcal), Beta calibration (from betacal), ENIR
  (from netcal), binary and multiclass Kernel calibration (using beta and dirichlet
  kernels from ece_kde), Spline calibration (from splinecalib), CDF-Spline calibration,
  Ensemble Temperature Scaling, tree based calibration with CatBoost, LightGBM and
  XGBoost.  
  - Re-structured the base `Calibrator` class to differentiate
  `_predict_proba_torch_impl` from `_predict_proba_impl`.
  - Added Kuiper and Kolmogorov-Smirnov binary calibration metrics.
  - Deprecated python 3.9, added python 3.13 and 3.14 support.
- v1.2.0 by [@elsacho](https://github.com/elsacho): Added new proper loss functions:
  - ProperLpLoss(p=p): Metrics to evaluate $E[ \Vert f(X) - E[Y|f(X)] \Vert_p ]$ where
    $f(X)$ are the predictions of the classifier, $p >= 1$, including `p=float("inf")`
  - TopClassLoss: A wrapper to variationally evaluate top-class errors.
  - OverConfidenceLoss & UnderConfidenceLoss: Wrappers to variationally evaluate 
    over/under-confidence in binary predictors.
  - MetricsWithCalibration can now handle arbitrary classifiers and Lp-type losses.
  - New classifiers:  Added `WS_CatboostClassifier` and `WS_LGBMClassifier` for 
    evaluating calibration errors.
  - removed sklearn < 1.7 constraint.
- v1.1.0 by [@eugeneberta](https://github.com/eugeneberta): Improvements to the SVS and
  SMS calibrators:
  - logit pre-processing with `'ts-mix'` is now automatic, 
    and the global scaling parameter $\alpha$ is fixed to 1. This yields:
    - improved performance on our tabular and computer vision benchmarks 
      (see the arxiv v2 of the SMS paper, coming soon).
    - faster convergence.
    - ability to compute the duality gap in closed form for stopping SAGA solvers, 
      which we implement in this version.
  - improved L-BFGS solvers, much faster than in the previous version. 
    Now used in SVS and SMS by default.
  - the default binary calibrator in `LogisticCalibrator` is now quadratic scaling 
    instead of affine scaling, this can be changed back by using 
    `LogisticCalibrator(binary_type='affine')`.
- v1.0.0 by [@eugeneberta](https://github.com/eugeneberta): New post-hoc calibrators
  like `'logistic'` including structured matrix scaling (SMS), structured vector scaling
  (SVS), affine scaling, and quadratic scaling.
- v0.0.2 by [@dholzmueller](https://github.com/dholzmueller):
  - Removed numpy<2.0 constraint
  - allow 1D vectors in CategoricalLogits / CategoricalProbs
  - add TorchCal temperature scaling
  - minor fixes in AutoGluon temperature scaling 
    that shouldn't affect the performance in practice
- v0.0.1 by [@dholzmueller](https://github.com/dholzmueller):
  Initial release with classification metrics, 
  calibration/refinement metrics, and some post-hoc calibration methods.
