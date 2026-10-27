"""Diagnóstico fino y rescate de Falsos Negativos (FN) mediante Destilación 3.0 en Test OOT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shap
from xgboost import XGBClassifier

from src.config import (
    MCC_DESCRIPTIONS,
    PRESENTATION_DIR,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_OOT_END_DT,
    TEST_OOT_START_DT,
    TRAIN_START_DT,
    VAL_START_DT,
)
from src.features.builder import build_feature_dataset_v3
from src.models.fn_analysis_plots import save_diagnostic_plots
from src.models.preprocessing import (
    ALL_MODEL_FEATURES,
    ALL_MODEL_FEATURES_V2,
    ALL_MODEL_FEATURES_V3,
    create_preprocessor,
)
from src.models.threshold import (
    compute_financial_utility,
    find_optimal_business_threshold,
)


@dataclass(frozen=True)
class FalseNegativeDiagnosticResult:
    """Contenedor estructurado del diagnóstico y rescate de Falsos Negativos en Test OOT."""

    optimal_threshold: float
    total_frauds: int
    detected_vp: int
    missed_fn: int
    false_positives: int
    card_level_recall_burst: float
    card_level_recall_single: float
    fn_after_prior_vp: int
    v3_optimal_threshold: float
    v3_detected_vp: int
    v3_missed_fn: int
    v3_false_positives: int
    v3_utility: float
    v3_stateful_vp: int
    v3_stateful_utility: float
    sequential_breakdown: pd.DataFrame
    archetype_breakdown: pd.DataFrame
    shap_gap_table: pd.DataFrame


def classify_fn_archetype(row: pd.Series) -> str:
    """Clasifica cada Falso Negativo en un arquetipo operativo mutuamente excluyente."""
    if row["y_proba"] >= 0.12:
        return "1. Casi-Alerta (0.12-0.27)"
    if row["total_frauds_on_card"] >= 2 and row["fraud_seq_num"] == 1:
        return "2. 'Paciente Cero' de Ráfaga (1ra tx)"
    if row["channel"] == "POS" and row["entry_mode"] in ("chip", "contactless"):
        return "3. POS Chip/Contactless Aislado"
    if (
        row["channel"] in ("ECOMMERCE", "APP")
        and row["is_country_mismatch"] == 0
        and row["cards_seen_by_device_24h"] <= 1
    ):
        return "4. E-Commerce/App Doméstico Aislado"
    if row["channel"] == "ATM":
        return "5. Retiro ATM Diurno Aislado"
    return "6. Rubro Cotidiano de Bajo Ticket"


def _train_and_score(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
    version: str,
) -> tuple[np.ndarray, float, XGBClassifier, list[str], np.ndarray]:
    """Entrena XGBoost en Train, calibra umbral en Val y puntúa Test OOT."""
    y_train = train_df[TARGET_COL].to_pandas()
    y_val = val_df[TARGET_COL].to_numpy()

    prep = create_preprocessor(feature_version=version)
    Xt_tr = prep.fit_transform(train_df.select(features).to_pandas(), y_train)
    Xt_va = prep.transform(val_df.select(features).to_pandas())
    Xt_te = prep.transform(test_df.select(features).to_pandas())

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=5.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    clf.fit(Xt_tr, y_train)

    p_val = clf.predict_proba(Xt_va)[:, 1]
    res_val = find_optimal_business_threshold(y_val, p_val, resolution=1000)
    p_test = clf.predict_proba(Xt_te)[:, 1]
    feat_names = prep.get_feature_names_out().tolist()

    return p_test, res_val.optimal_threshold, clf, feat_names, Xt_te


def run_false_negative_experiment(
    raw_csv_path: Path | None = None,
    features_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> FalseNegativeDiagnosticResult:
    """Ejecuta el experimento de diagnóstico y rescate de Falsos Negativos."""
    csv_file = raw_csv_path or RAW_DATA_PATH
    raw_df = pl.read_csv(csv_file).with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    df_v3 = build_feature_dataset_v3(
        df=raw_df,
        save_path=None
        if features_dir is None
        else features_dir / "transacciones_features_v3.parquet",
    )

    tr_v3 = df_v3.filter(
        (pl.col("timestamp") >= TRAIN_START_DT) & (pl.col("timestamp") < VAL_START_DT)
    )
    va_v3 = df_v3.filter(
        (pl.col("timestamp") >= VAL_START_DT)
        & (pl.col("timestamp") < TEST_OOT_START_DT)
    )
    te_v3 = df_v3.filter(
        (pl.col("timestamp") >= TEST_OOT_START_DT)
        & (pl.col("timestamp") <= TEST_OOT_END_DT)
    )

    feat_v1 = [f for f in ALL_MODEL_FEATURES if f in df_v3.columns]
    feat_v2 = [f for f in ALL_MODEL_FEATURES_V2 if f in df_v3.columns]
    feat_v3 = [f for f in ALL_MODEL_FEATURES_V3 if f in df_v3.columns]

    p_v1, t_v1, clf_v1, feat_v1_names, Xt_te_v1 = _train_and_score(
        tr_v3, va_v3, te_v3, feat_v1, version="v1"
    )
    p_v2, t_v2, _, _, _ = _train_and_score(tr_v3, va_v3, te_v3, feat_v2, version="v2")
    p_v3, t_v3, _, _, _ = _train_and_score(tr_v3, va_v3, te_v3, feat_v3, version="v3")

    y_test_np = te_v3[TARGET_COL].to_numpy()
    frauds_pd = (
        te_v3.filter(pl.col(TARGET_COL) == 1)
        .select(
            [
                "transaction_id",
                "card_id",
                "customer_id",
                "timestamp",
                "amount",
                "channel",
                "entry_mode",
                "mcc",
                "is_country_mismatch",
                "cards_seen_by_device_24h",
                "is_fraud",
            ]
        )
        .to_pandas()
    )
    frauds_idx = (
        te_v3.with_row_index().filter(pl.col(TARGET_COL) == 1)["index"].to_numpy()
    )
    frauds_pd["y_proba"] = p_v1[frauds_idx]
    frauds_pd["p_v1"] = p_v1[frauds_idx]
    frauds_pd["p_v2"] = p_v2[frauds_idx]
    frauds_pd["p_v3"] = p_v3[frauds_idx]
    frauds_pd["mcc_desc"] = (
        frauds_pd["mcc"].map(MCC_DESCRIPTIONS).fillna(frauds_pd["mcc"].astype(str))
    )

    frauds_pd = frauds_pd.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    frauds_pd["fraud_seq_num"] = frauds_pd.groupby("card_id").cumcount() + 1
    frauds_pd["total_frauds_on_card"] = frauds_pd.groupby("card_id")[
        "transaction_id"
    ].transform("count")

    frauds_pd["detected_v1"] = frauds_pd["p_v1"] >= t_v1
    frauds_pd["detected_v2"] = frauds_pd["p_v2"] >= t_v2
    frauds_pd["detected_v3"] = frauds_pd["p_v3"] >= t_v3

    seq_df = (
        frauds_pd.groupby("fraud_seq_num")
        .agg(
            total_fraudes=("is_fraud", "count"),
            vp_v1=("detected_v1", "sum"),
            vp_v2=("detected_v2", "sum"),
            vp_v3=("detected_v3", "sum"),
        )
        .reset_index()
    )
    seq_df["recall_pct"] = (seq_df["vp_v1"] / seq_df["total_fraudes"] * 100).round(1)
    seq_df["fn_rate_pct"] = (100.0 - seq_df["recall_pct"]).round(1)
    seq_df["recall_v2_pct"] = (seq_df["vp_v2"] / seq_df["total_fraudes"] * 100).round(1)
    seq_df["recall_v3_pct"] = (seq_df["vp_v3"] / seq_df["total_fraudes"] * 100).round(1)

    card_det = (
        frauds_pd.groupby("card_id")
        .agg(
            detected=("detected_v1", "any"),
            total_frauds=("total_frauds_on_card", "first"),
        )
        .reset_index()
    )
    burst_recall = float(card_det[card_det["total_frauds"] >= 2]["detected"].mean())
    single_recall = float(card_det[card_det["total_frauds"] == 1]["detected"].mean())

    frauds_pd["cum_vp_before_v1"] = frauds_pd.groupby("card_id")["p_v1"].transform(
        lambda s: (s >= t_v1).shift(1, fill_value=False).cumsum()
    )
    fn_after_vp = len(
        frauds_pd[(frauds_pd["p_v1"] < t_v1) & (frauds_pd["cum_vp_before_v1"] > 0)]
    )

    frauds_pd["cum_vp_before_v3"] = frauds_pd.groupby("card_id")["p_v3"].transform(
        lambda s: (s >= t_v3).shift(1, fill_value=False).cumsum()
    )
    v3_vp = int(np.sum((y_test_np == 1) & (p_v3 >= t_v3)))
    v3_fp = int(np.sum((y_test_np == 0) & (p_v3 >= t_v3)))
    v3_fn = int(np.sum((y_test_np == 1) & (p_v3 < t_v3)))
    v3_util = compute_financial_utility(v3_vp, v3_fp, v3_fn)

    v3_extra_stateful = len(
        frauds_pd[(frauds_pd["p_v3"] < t_v3) & (frauds_pd["cum_vp_before_v3"] > 0)]
    )
    v3_stateful_vp = v3_vp + v3_extra_stateful
    v3_stateful_util = compute_financial_utility(
        v3_stateful_vp, v3_fp, len(frauds_pd) - v3_stateful_vp
    )

    fn_only = frauds_pd[frauds_pd["p_v1"] < t_v1].copy()
    fn_only["refined_archetype"] = fn_only.apply(classify_fn_archetype, axis=1)
    fn_only["rescued_v3"] = fn_only["p_v3"] >= t_v3
    arch_df = (
        fn_only.groupby("refined_archetype")
        .agg(
            casos=("is_fraud", "count"),
            rescatados_v3=("rescued_v3", lambda s: int(s.sum())),
            restantes_v3=("rescued_v3", lambda s: int((~s).sum())),
            score_medio=("p_v1", "mean"),
            score_medio_v3=("p_v3", "mean"),
            monto_mediano=("amount", "median"),
        )
        .reset_index()
        .sort_values("casos", ascending=False)
    )
    arch_df["pct"] = (arch_df["casos"] / len(fn_only) * 100).round(1)
    arch_df["tasa_rescate_pct"] = (
        arch_df["rescatados_v3"] / arch_df["casos"] * 100
    ).round(1)

    explainer = shap.TreeExplainer(clf_v1)
    vp_idx = np.where((y_test_np == 1) & (p_v1 >= t_v1))[0]
    fn_idx = np.where((y_test_np == 1) & (p_v1 < t_v1))[0]
    shap_vp = explainer.shap_values(Xt_te_v1[vp_idx], check_additivity=False)
    shap_fn = explainer.shap_values(Xt_te_v1[fn_idx], check_additivity=False)

    shap_gap = pd.DataFrame(
        {
            "feature": feat_v1_names,
            "mean_shap_VP": np.mean(shap_vp, axis=0),
            "mean_shap_FN": np.mean(shap_fn, axis=0),
            "gap_VP_to_FN": np.mean(shap_vp, axis=0) - np.mean(shap_fn, axis=0),
        }
    ).sort_values("gap_VP_to_FN", ascending=False)

    out_dir = figures_dir or (PRESENTATION_DIR / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_diagnostic_plots(seq_df, arch_df, shap_gap, out_dir)

    return FalseNegativeDiagnosticResult(
        optimal_threshold=float(t_v1),
        total_frauds=len(frauds_pd),
        detected_vp=len(vp_idx),
        missed_fn=len(fn_idx),
        false_positives=int(np.sum((y_test_np == 0) & (p_v1 >= t_v1))),
        card_level_recall_burst=burst_recall,
        card_level_recall_single=single_recall,
        fn_after_prior_vp=fn_after_vp,
        v3_optimal_threshold=float(t_v3),
        v3_detected_vp=v3_vp,
        v3_missed_fn=v3_fn,
        v3_false_positives=v3_fp,
        v3_utility=float(v3_util),
        v3_stateful_vp=v3_stateful_vp,
        v3_stateful_utility=float(v3_stateful_util),
        sequential_breakdown=seq_df,
        archetype_breakdown=arch_df,
        shap_gap_table=shap_gap,
    )


if __name__ == "__main__":
    res = run_false_negative_experiment()
    print(
        f"V1: VP={res.detected_vp}, FN={res.missed_fn} | "
        f"V3: VP={res.v3_detected_vp}, FN={res.v3_missed_fn}, Util=US${res.v3_utility:,.0f} | "
        f"V3+Bloqueo Plástico: VP={res.v3_stateful_vp}, Util=US${res.v3_stateful_utility:,.0f}"
    )
