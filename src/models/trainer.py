"""Módulo de entrenamiento de modelos predictivos y benchmarking."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import BUSINESS_MATRIX, RANDOM_STATE
from src.models.evaluation import compute_financial_utility
from src.models.preprocessing import create_preprocessor


@dataclass
class ModelEvaluationResult:
    """Contenedor estructurado para resultados de evaluación de modelos."""

    model_name: str
    pipeline: Any
    pr_auc: float
    roc_auc: float
    f1_default: float
    precision_default: float
    recall_default: float
    utility_default: float
    y_proba: np.ndarray


def create_candidate_classifiers(
    scale_pos_weight: float = 5.0,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Inicializa los estimadores candidatos con control de desbalance."""
    return {
        "LightGBM Baseline": LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=4,
        ),
    }


def create_candidate_pipelines(
    scale_pos_weight: float = 1.0,
    random_state: int = RANDOM_STATE,
    feature_version: str = "v1",
) -> dict[str, Pipeline]:
    """Crea los pipelines scikit-learn completos combinando preprocesamiento y clasificador."""
    classifiers = create_candidate_classifiers(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    return {
        name: Pipeline(
            [
                (
                    "preprocessor",
                    create_preprocessor(
                        random_state=random_state, feature_version=feature_version
                    ),
                ),
                ("classifier", clf),
            ]
        )
        for name, clf in classifiers.items()
    }


def train_and_evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    x_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    calibrate: bool = False,
    calibration_method: str = "isotonic",
) -> ModelEvaluationResult:
    """Entrena un pipeline y calcula métricas diagnósticas y de utilidad sobre validación."""
    y_val_arr = np.asarray(y_val)

    pipeline.fit(x_train, y_train)

    if calibrate and hasattr(pipeline, "predict_proba"):
        try:
            from sklearn.frozen import FrozenEstimator

            calibrator = CalibratedClassifierCV(
                estimator=FrozenEstimator(pipeline),
                method=calibration_method,
            )
        except (ImportError, AttributeError):
            calibrator = CalibratedClassifierCV(
                estimator=pipeline,
                method=calibration_method,
                cv="prefit",
            )
        calibrator.fit(x_val, y_val)
        y_proba = calibrator.predict_proba(x_val)[:, 1]
        active_pipeline: Any = calibrator
    else:
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(x_val)[:, 1]
        else:
            y_proba = pipeline.decision_function(x_val)
        active_pipeline = pipeline

    y_pred_default = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_val_arr, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pr_auc = float(average_precision_score(y_val_arr, y_proba))
    roc_auc = float(roc_auc_score(y_val_arr, y_proba))
    f1 = float(f1_score(y_val_arr, y_pred_default, zero_division=0))
    prec = float(precision_score(y_val_arr, y_pred_default, zero_division=0))
    rec = float(recall_score(y_val_arr, y_pred_default, zero_division=0))
    utility = compute_financial_utility(tp, fp, fn, tn, matrix=BUSINESS_MATRIX)

    return ModelEvaluationResult(
        model_name=name,
        pipeline=active_pipeline,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        f1_default=f1,
        precision_default=prec,
        recall_default=rec,
        utility_default=utility,
        y_proba=y_proba,
    )
