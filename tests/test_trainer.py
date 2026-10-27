"""Pruebas unitarias para el módulo de entrenamiento de modelos."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.trainer import (
    ModelEvaluationResult,
    create_candidate_classifiers,
    create_candidate_pipelines,
    train_and_evaluate_model,
)


def test_create_candidate_classifiers():
    """Verifica la inicialización de los clasificadores candidatos."""
    classifiers = create_candidate_classifiers(scale_pos_weight=5.0)
    assert "LightGBM Baseline" in classifiers
    assert "LightGBM" in classifiers
    assert "XGBoost" in classifiers


def test_create_candidate_pipelines():
    """Verifica la creación de pipelines completos con preprocesador."""
    pipelines = create_candidate_pipelines(scale_pos_weight=1.0)
    assert "LightGBM Baseline" in pipelines
    assert "LightGBM" in pipelines
    assert "XGBoost" in pipelines
    for pipe in pipelines.values():
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_train_and_evaluate_model():
    """Verifica el flujo completo de entrenamiento y evaluación sobre un dataset sintético."""
    np.random.seed(42)
    x_train = pd.DataFrame(
        np.random.randn(200, 5), columns=[f"f_{i}" for i in range(5)]
    )
    y_train = (x_train["f_0"] + np.random.randn(200) > 0.8).astype(int)

    x_val = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y_val = (x_val["f_0"] > 0.3).astype(int)

    classifiers = create_candidate_classifiers()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifiers["LightGBM Baseline"]),
        ]
    )

    result = train_and_evaluate_model(
        "LightGBM Baseline",
        pipe,
        x_train,
        y_train,
        x_val,
        y_val,
    )

    assert isinstance(result, ModelEvaluationResult)
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.precision_default <= 1.0
    assert 0.0 <= result.recall_default <= 1.0


def test_train_and_evaluate_model_calibrated():
    """Verifica el entrenamiento con calibración de probabilidades activa."""
    np.random.seed(42)
    x_train = pd.DataFrame(
        np.random.randn(200, 5), columns=[f"f_{i}" for i in range(5)]
    )
    y_train = (x_train["f_0"] + np.random.randn(200) > 0.8).astype(int)

    x_val = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    y_val = (x_val["f_0"] > 0.3).astype(int)

    classifiers = create_candidate_classifiers()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifiers["LightGBM Baseline"]),
        ]
    )

    result = train_and_evaluate_model(
        "LightGBM Baseline Calibrated",
        pipe,
        x_train,
        y_train,
        x_val,
        y_val,
        calibrate=True,
    )

    assert isinstance(result, ModelEvaluationResult)
    assert np.all(result.y_proba >= 0.0)
    assert np.all(result.y_proba <= 1.0)
