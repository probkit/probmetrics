# probmetrics/calibrators/__init__.py

from .base import (
    Calibrator,
    ApplyToLogitsCalibrator,
    WrapCalibratorAsClassifier,
    MulticlassOneVsOneCalibrator,
    MulticlassOneVsRestCalibrator,
    MixtureCalibrator,
)

from .logistic import (
    VectorScalingCalibrator,
    SVSCalibrator,
    MatrixScalingCalibrator,
    SMSCalibrator,
    DirichletCalibrator,
    LogisticCalibrator,
    BinaryLogisticCalibrator,
    RegularizedAffineScalingCalibrator,
    RegularizedQuadraticScalingCalibrator,
    BetacalCalibrator,
)

from .temperature_scaling import (
    TemperatureScalingCalibrator,
    GuoTemperatureScalingCalibrator,
    AutoGluonTemperatureScalingCalibrator,
    AutoGluonInverseTemperatureScalingCalibrator,
    TorchUncertaintyTemperatureScalingCalibrator,
    TorchcalTemperatureScalingCalibrator,
    NetcalTemperatureScalingCalibrator,
    ETSCalibrator,
    SoftPlusCalibrator,
)

from .nonparametric import (
    NetcalENIRCalibrator,
    CenteredIsotonicRegressionCalibrator,
    BinaryVennAbersCalibrator,
    VennAbersCalibrator,
    KernelCalibrator,
)

from .binning import (
    BinaryHistogramBinningCalibrator,
    BinaryPlattBinnerCalibrator,
    NetcalBBQCalibrator,
)

from .splines import (
    MLISplineCalibrator,
    CDFSplineCalibrator,
)

from .sklearn import (
    SklearnCalibrator,
)

from .trees import (
    CatBoostCalibrator,
    LightGBMCalibrator,
    XGBoostCalibrator,
    BinaryCatBoostCalibrator,
)

from .factory import get_calibrator


__all__ = [
    "Calibrator",
    "MulticlassOneVsOneCalibrator",
    "MulticlassOneVsRestCalibrator",
    "MixtureCalibrator",
    "get_calibrator",
    "VectorScalingCalibrator",
    "SVSCalibrator",
    "MatrixScalingCalibrator",
    "SMSCalibrator",
    "DirichletCalibrator",
    "LogisticCalibrator",
    "BinaryLogisticCalibrator",
    "RegularizedAffineScalingCalibrator",
    "RegularizedQuadraticScalingCalibrator",
    "BetacalCalibrator",
    "TemperatureScalingCalibrator",
    "GuoTemperatureScalingCalibrator",
    "AutoGluonTemperatureScalingCalibrator",
    "AutoGluonInverseTemperatureScalingCalibrator",
    "TorchUncertaintyTemperatureScalingCalibrator",
    "TorchcalTemperatureScalingCalibrator",
    "NetcalTemperatureScalingCalibrator",
    "ETSCalibrator",
    "SoftPlusCalibrator",
    "NetcalENIRCalibrator",
    "CenteredIsotonicRegressionCalibrator",
    "BinaryVennAbersCalibrator",
    "VennAbersCalibrator",
    "KernelCalibrator",
    "BinaryHistogramBinningCalibrator",
    "BinaryPlattBinnerCalibrator",
    "NetcalBBQCalibrator",
    "MLISplineCalibrator",
    "CDFSplineCalibrator",
    "SklearnCalibrator",
    "CatBoostCalibrator",
    "LightGBMCalibrator",
    "XGBoostCalibrator",
    "BinaryCatBoostCalibrator",
]
