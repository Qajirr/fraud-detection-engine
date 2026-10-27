"""Módulo de optimización bayesiana de hiperparámetros con Optuna."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.models.attention import TabularAttentionClassifier

# Silenciar logs verbosos de Optuna por defecto
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    """Contenedor para resultados de optimización de hiperparámetros."""

    model_name: str
    best_params: dict[str, Any]
    best_pr_auc: float
    study: optuna.Study
    best_estimator: Any


def tune_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> TuningResult:
    """Optimiza hiperparámetros de LightGBM maximizando PR-AUC en validación."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": random_state,
            "n_jobs": 4,
            "verbose": -1,
        }
        clf = LGBMClassifier(**params)
        clf.fit(x_train, y_train)
        preds_proba = clf.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, preds_proba))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        **study.best_params,
        "random_state": random_state,
        "n_jobs": 4,
        "verbose": -1,
    }
    best_model = LGBMClassifier(**best_params)
    best_model.fit(x_train, y_train)

    return TuningResult(
        model_name="LightGBM Optuna",
        best_params=study.best_params,
        best_pr_auc=study.best_value,
        study=study,
        best_estimator=best_model,
    )


def tune_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    random_state: int = 42,
) -> TuningResult:
    """Optimiza hiperparámetros de XGBoost maximizando PR-AUC en validación."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": random_state,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 4,
        }
        clf = XGBClassifier(**params)
        clf.fit(x_train, y_train)
        preds_proba = clf.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, preds_proba))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = {
        **study.best_params,
        "random_state": random_state,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 4,
    }
    best_model = XGBClassifier(**best_params)
    best_model.fit(x_train, y_train)

    return TuningResult(
        model_name="XGBoost Optuna",
        best_params=study.best_params,
        best_pr_auc=study.best_value,
        study=study,
        best_estimator=best_model,
    )


def tune_attention_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 15,
    random_state: int = 42,
) -> TuningResult:
    """Optimiza hiperparámetros de la Red Tabular con Atención maximizando PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        embed_dim = trial.suggest_categorical("embed_dim", [16, 32])
        num_heads = trial.suggest_categorical("num_heads", [2, 4])
        ffn_dim = trial.suggest_categorical("ffn_dim", [32, 64])
        dropout = trial.suggest_float("dropout", 0.05, 0.25)
        lr = trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True)
        alpha_focal = trial.suggest_float("alpha_focal", 0.25, 0.75)
        gamma_focal = trial.suggest_float("gamma_focal", 1.5, 3.0)

        clf = TabularAttentionClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            learning_rate=lr,
            epochs=10,
            alpha_focal=alpha_focal,
            gamma_focal=gamma_focal,
            random_state=random_state,
        )
        clf.fit(x_train, y_train, x_val=x_val, y_val=y_val)
        val_proba = clf.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, val_proba))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_clf = TabularAttentionClassifier(
        **study.best_params,
        epochs=15,
        random_state=random_state,
    )
    best_clf.fit(x_train, y_train, x_val=x_val, y_val=y_val)

    return TuningResult(
        model_name="Tabular Attention Optuna",
        best_params=study.best_params,
        best_pr_auc=study.best_value,
        study=study,
        best_estimator=best_clf,
    )
