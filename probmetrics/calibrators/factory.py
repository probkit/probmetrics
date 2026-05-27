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
)


def get_calibrator(
    calibration_method: str,
    calibrate_with_mixture: bool = False,
    cal_mixture_output_constant: float = 1.0,
    cal_mixture_input_constant: float = 0.0,
    **config,
) -> Calibrator:
    if calibration_method == "platt":
        cal = SklearnCalibrator(method="sigmoid", cv="prefit")
    elif calibration_method == "platt-logits":
        cal = ApplyToLogitsCalibrator(SklearnCalibrator(method="sigmoid", cv="prefit"))
    elif calibration_method == "isotonic":
        cal = SklearnCalibrator(method="isotonic", cv="prefit")
    elif calibration_method == "ivap":
        cal = VennAbersCalibrator(use_ovo=config.get("va_use_ovo", False))
    elif calibration_method == "ivap-ovr":
        cal = MulticlassOneVsRestCalibrator(BinaryVennAbersCalibrator())
    elif calibration_method == "ivap-ovo":
        cal = MulticlassOneVsOneCalibrator(BinaryVennAbersCalibrator())
    elif calibration_method == "isotonic-naive-ovr":
        # should be the same as 'isotonic', this is just to check
        cal = MulticlassOneVsRestCalibrator(
            SklearnCalibrator(method="isotonic", cv="prefit")
        )
    elif calibration_method == "cir":
        cal = MulticlassOneVsRestCalibrator(CenteredIsotonicRegressionCalibrator())
    elif calibration_method in ["temp-scaling", "ts-mix"]:
        cal = TemperatureScalingCalibrator(
            opt=config.get("ts_opt", "bisection"),
            max_bisection_steps=config.get("ts_max_bisection_steps", 30),
            lr=config.get("ts_lr", 0.1),
            max_iter=config.get("ts_max_iter", 200),
            use_inv_temp=config.get("ts_use_inv_temp", True),
            inv_temp_init=config.get("ts_inv_temp_init", 1 / 1.5),
        )
    elif calibration_method == "linear-scaling":
        cal = BinaryLogisticCalibrator(type="linear")
    elif calibration_method == "affine-scaling":
        cal = BinaryLogisticCalibrator(type="affine")
    elif calibration_method == "quadratic-scaling":
        cal = BinaryLogisticCalibrator(type="quadratic")
    elif calibration_method == "beta":
        cal = BetacalCalibrator(parameters=config.get("beta_parameters", "abm"))
    elif calibration_method == "svs":
        cal = SVSCalibrator(
            penalty=config.get("svs_penalty", "ridge"),
            rho=config.get("svs_rho", 1.0),
            tau=config.get("svs_tau", 1.0),
            lambda_intercept=config.get("svs_lambda_intercept", 1.0),
            lambda_diagonal=config.get("svs_lambda_diagonal", 1.0),
            opt=config.get("svs_opt", "bfgs"),
            max_iter=config.get("svs_max_iter", 500),
            tol=config.get("svs_tol", 1e-5),
            print_init_info=config.get("svs_print_init_info", True),
        )
    elif calibration_method == "sms":
        cal = SMSCalibrator(
            penalty=config.get("sms_penalty", "ridge"),
            rho=config.get("sms_rho", 1.0),
            tau=config.get("sms_tau", 1.0),
            lambda_intercept=config.get("sms_lambda_intercept", 1.0),
            lambda_diagonal=config.get("sms_lambda_diagonal", 1.0),
            lambda_off_diagonal=config.get("sms_lambda_off_diagonal", 1.0),
            opt=config.get("sms_opt", "bfgs"),
            max_iter=config.get("sms_max_iter", 500),
            tol=config.get("sms_tol", 1e-5),
            print_init_info=config.get("sms_print_init_info", True),
        )
    elif calibration_method in ["logistic", "logistic-mix"]:
        cal = LogisticCalibrator(
            binary_type=config.get("logistic_binary_type", "affine"),
            multiclass_type=config.get("logistic_multiclass_type", "sms"),
        )
    elif calibration_method == "torchunc-ts":
        cal = TorchUncertaintyTemperatureScalingCalibrator(
            init_val=config.get("ts_init_val", 1),
            lr=config.get("ts_lr", 0.1),
            max_iter=config.get("ts_max_iter", 100),
        )
    elif calibration_method == "guo-ts":
        cal = GuoTemperatureScalingCalibrator()
    elif calibration_method == "torchcal-ts":
        cal = TorchcalTemperatureScalingCalibrator()
    elif calibration_method == "autogluon-ts":
        cal = AutoGluonTemperatureScalingCalibrator(
            init_val=config.get("ts_init_val", 1),
            max_iter=config.get("ts_max_iter", 200),
            lr=config.get("ts_lr", 0.1),
        )
    elif calibration_method == "autogluon-inv-ts":
        cal = AutoGluonInverseTemperatureScalingCalibrator(
            init_val=config.get("ts_init_val", 1),
            max_iter=config.get("ts_max_iter", 200),
            lr=config.get("ts_lr", 0.1),
        )
    elif calibration_method == "dircal":
        cal = DirichletCalibrator(
            n_cv=0,
            reg_lambda=config.get("dircal_reg_lambda", 1e-3),
            reg_mu=config.get("dircal_reg_mu", 1e-3),
        )
    elif calibration_method == "dircal-cv":
        cal = DirichletCalibrator(
            n_cv=config.get("dircal_n_cv", 5),
            use_odir=config.get("dircal_use_odir", False),
            reg_lambda_grid=config.get("dircal_reg_lambda_grid", None),
            reg_mu_grid=config.get("dircal_reg_mu_grid", None),
        )
    else:
        raise ValueError(f'Unknown calibration method "{calibration_method}"')

    if calibrate_with_mixture or calibration_method.endswith("-mix"):
        cal = MixtureCalibrator(
            cal,
            output_constant=cal_mixture_output_constant,
            input_constant=cal_mixture_input_constant,
        )

    return cal
