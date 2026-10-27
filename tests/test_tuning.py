"""Pruebas unitarias para optimización de hiperparámetros con Optuna."""

import numpy as np

from src.models.tuning import (
    TuningResult,
    tune_attention_classifier,
    tune_lightgbm,
    tune_xgboost,
)


def test_tune_lightgbm_smoke() -> None:
    rng = np.random.RandomState(42)
    n_train, n_val, n_feat = 80, 40, 5

    x_train = rng.randn(n_train, n_feat)
    y_train = (rng.rand(n_train) > 0.8).astype(int)

    x_val = rng.randn(n_val, n_feat)
    y_val = (rng.rand(n_val) > 0.8).astype(int)

    result = tune_lightgbm(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        n_trials=2,
        random_state=42,
    )

    assert isinstance(result, TuningResult)
    assert result.model_name == "LightGBM Optuna"
    assert "n_estimators" in result.best_params
    assert hasattr(result.best_estimator, "predict_proba")
    assert 0.0 <= result.best_pr_auc <= 1.0


def test_tune_xgboost_smoke() -> None:
    rng = np.random.RandomState(42)
    n_train, n_val, n_feat = 80, 40, 5

    x_train = rng.randn(n_train, n_feat)
    y_train = (rng.rand(n_train) > 0.8).astype(int)

    x_val = rng.randn(n_val, n_feat)
    y_val = (rng.rand(n_val) > 0.8).astype(int)

    result = tune_xgboost(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        n_trials=2,
        random_state=42,
    )

    assert isinstance(result, TuningResult)
    assert result.model_name == "XGBoost Optuna"
    assert "learning_rate" in result.best_params
    assert hasattr(result.best_estimator, "predict_proba")
    assert 0.0 <= result.best_pr_auc <= 1.0


def test_tune_attention_smoke() -> None:
    rng = np.random.RandomState(42)
    n_train, n_val, n_feat = 60, 30, 4

    x_train = rng.randn(n_train, n_feat).astype(np.float32)
    y_train = (rng.rand(n_train) > 0.8).astype(int)

    x_val = rng.randn(n_val, n_feat).astype(np.float32)
    y_val = (rng.rand(n_val) > 0.8).astype(int)

    result = tune_attention_classifier(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        n_trials=2,
        random_state=42,
    )

    assert isinstance(result, TuningResult)
    assert result.model_name == "Tabular Attention Optuna"
    assert "embed_dim" in result.best_params
    assert hasattr(result.best_estimator, "predict_proba")
    assert 0.0 <= result.best_pr_auc <= 1.0
