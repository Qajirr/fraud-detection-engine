"""Pipeline de experimentación para Feature Engineering 2.0 y benchmarking OOT."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from src.config import (
    PROCESSED_DATA_V2_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_OOT_START_DT,
    VAL_START_DT,
)
from src.features.builder import build_feature_dataset_v2
from src.models.evaluation import (
    compute_detailed_metrics,
    find_optimal_business_threshold,
)
from src.models.preprocessing import (
    ALL_MODEL_FEATURES,
    ALL_MODEL_FEATURES_V2,
    create_preprocessor,
)


def load_or_create_features_v2(
    raw_path: Path = RAW_DATA_PATH,
    save_path: Path = PROCESSED_DATA_V2_PATH,
) -> pl.DataFrame:
    """Carga el dataset v2 persistido o ejecuta la construcción causal si no existe."""
    if save_path.exists():
        print(f"Cargando dataset enriquecido v2 desde {save_path}...")
        return pl.read_parquet(save_path)

    print(f"Generando dataset v2 sin fuga temporal desde {raw_path}...")
    return build_feature_dataset_v2(raw_path=raw_path, save_path=save_path)


def get_temporal_partitions(df: pl.DataFrame):
    """Divide el dataset en Train, Val y Test OOT respetando estrictamente el orden temporal."""
    train_pl = df.filter(pl.col("timestamp") < VAL_START_DT)
    val_pl = df.filter(
        (pl.col("timestamp") >= VAL_START_DT)
        & (pl.col("timestamp") < TEST_OOT_START_DT)
    )
    test_pl = df.filter(pl.col("timestamp") >= TEST_OOT_START_DT)

    return train_pl, val_pl, test_pl


def run_experiment_features_v2(
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Ejecuta el experimento comparativo de XGBoost con Features v1 vs Features v2."""
    df_v2 = load_or_create_features_v2()

    train_pl, val_pl, test_pl = get_temporal_partitions(df_v2)

    # Convertir a pandas para compatibilidad scikit-learn
    train_df = train_pl.to_pandas()
    val_df = val_pl.to_pandas()
    test_df = test_pl.to_pandas()

    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_val = val_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)

    # Hiperparámetros optimizados con Optuna (Campeón XGBoost)
    xgb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 5.0,
        "random_state": random_state,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 4,
    }

    results = {}

    # 1. Benchmark: XGBoost con Features v1 (Línea base de comparación)
    print("\n--- Entrenando XGBoost con Features v1 ---")
    prep_v1 = create_preprocessor(random_state=random_state, feature_version="v1")
    x_tr_v1 = prep_v1.fit_transform(train_df[ALL_MODEL_FEATURES], y_train)
    x_val_v1 = prep_v1.transform(val_df[ALL_MODEL_FEATURES])
    x_te_v1 = prep_v1.transform(test_df[ALL_MODEL_FEATURES])

    clf_v1 = XGBClassifier(**xgb_params)
    clf_v1.fit(x_tr_v1, y_train)

    val_proba_v1 = clf_v1.predict_proba(x_val_v1)[:, 1]
    th_res_v1 = find_optimal_business_threshold(y_val, val_proba_v1)
    test_proba_v1 = clf_v1.predict_proba(x_te_v1)[:, 1]
    metrics_v1 = compute_detailed_metrics(
        y_test, test_proba_v1, threshold=th_res_v1.optimal_threshold
    )
    metrics_v1["model_name"] = "XGBoost (Features v1)"
    metrics_v1["val_optimal_threshold"] = th_res_v1.optimal_threshold
    metrics_v1["savings_vs_inaction"] = metrics_v1["Financial_Utility_USD"] - (
        -180.0 * int(np.sum(y_test))
    )
    results["v1"] = metrics_v1

    # 2. Experimento Fase A: XGBoost con Features 2.0
    print("\n--- Entrenando XGBoost con Features 2.0 ---")
    prep_v2 = create_preprocessor(random_state=random_state, feature_version="v2")
    x_tr_v2 = prep_v2.fit_transform(train_df[ALL_MODEL_FEATURES_V2], y_train)
    x_val_v2 = prep_v2.transform(val_df[ALL_MODEL_FEATURES_V2])
    x_te_v2 = prep_v2.transform(test_df[ALL_MODEL_FEATURES_V2])

    clf_v2 = XGBClassifier(**xgb_params)
    clf_v2.fit(x_tr_v2, y_train)

    val_proba_v2 = clf_v2.predict_proba(x_val_v2)[:, 1]
    th_res_v2 = find_optimal_business_threshold(y_val, val_proba_v2)
    test_proba_v2 = clf_v2.predict_proba(x_te_v2)[:, 1]
    metrics_v2 = compute_detailed_metrics(
        y_test, test_proba_v2, threshold=th_res_v2.optimal_threshold
    )
    metrics_v2["model_name"] = "XGBoost (Features 2.0)"
    metrics_v2["val_optimal_threshold"] = th_res_v2.optimal_threshold
    metrics_v2["savings_vs_inaction"] = metrics_v2["Financial_Utility_USD"] - (
        -180.0 * int(np.sum(y_test))
    )
    results["v2"] = metrics_v2

    # Comparación de Utilidad y Rendimiento
    diff_utility = (
        metrics_v2["Financial_Utility_USD"] - metrics_v1["Financial_Utility_USD"]
    )
    diff_pr_auc = metrics_v2["PR-AUC"] - metrics_v1["PR-AUC"]
    diff_fp = metrics_v2["FP"] - metrics_v1["FP"]
    diff_tp = metrics_v2["TP"] - metrics_v1["TP"]

    print("\n================== RESULTADOS FASE A ==================")
    print(
        f"v1 Utilidad: US$ {metrics_v1['Financial_Utility_USD']:,.2f} | PR-AUC: {metrics_v1['PR-AUC']:.4f} | TP: {metrics_v1['TP']} | FP: {metrics_v1['FP']}"
    )
    print(
        f"v2 Utilidad: US$ {metrics_v2['Financial_Utility_USD']:,.2f} | PR-AUC: {metrics_v2['PR-AUC']:.4f} | TP: {metrics_v2['TP']} | FP: {metrics_v2['FP']}"
    )
    print(f"Ganancia Neta Incremental: US$ {diff_utility:+,.2f}")
    print(f"Delta PR-AUC: {diff_pr_auc:+.4f}")
    print(f"Delta Fraudes Detectados (VP): {diff_tp:+d}")
    print(f"Delta Falsas Alarmas (FP): {diff_fp:+d}")
    print("=======================================================\n")

    return {
        "metrics_v1": metrics_v1,
        "metrics_v2": metrics_v2,
        "diff_utility": diff_utility,
        "diff_pr_auc": diff_pr_auc,
        "diff_tp": diff_tp,
        "diff_fp": diff_fp,
    }


if __name__ == "__main__":
    run_experiment_features_v2()
