"""Pruebas unitarias para la Destilación 3.0 de variables causales sin fuga temporal."""

from datetime import datetime

import pandas as pd
import polars as pl

from src.features.builder import build_feature_dataset_v3
from src.models.fn_analysis import classify_fn_archetype
from src.models.preprocessing import ALL_MODEL_FEATURES_V3, create_preprocessor


def test_compute_distilled_v3_features_no_leakage() -> None:
    """Verifica que las 14 variables de Destilación 3.0 se calculen sin fuga temporal."""
    raw_df = pl.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4],
            "card_id": [10, 10, 10, 20],
            "customer_id": [100, 100, 100, 200],
            "merchant_id": [500, 500, 501, 500],
            "timestamp": [
                datetime(2026, 3, 1, 10, 0, 0),
                datetime(2026, 3, 2, 12, 0, 0),
                datetime(2026, 3, 6, 15, 0, 0),
                datetime(2026, 4, 20, 18, 0, 0),
            ],
            "amount": [50.0, 60.0, 400.0, 50.0],
            "mcc": [5411, 5411, 5411, 5411],
            "merchant_country": ["CL", "CL", "US", "CL"],
            "channel": ["POS", "POS", "ECOMMERCE", "POS"],
            "entry_mode": ["chip", "chip", "keyed", "contactless"],
            "ip_country": [None, None, "US", None],
            "device_id": [None, None, "dev_abc", None],
            "ip_address": [None, None, "190.1.1.1", None],
            "is_fraud": [0, 0, 1, 0],
        }
    )

    v3_df = build_feature_dataset_v3(df=raw_df, save_path=None)

    assert len(v3_df) == 4
    expected_v3_cols = {
        "card_prior_tx_count",
        "ratio_amount_card_expanding_mean",
        "ratio_amount_card_prev_max",
        "zscore_amount_segment",
        "is_above_segment_p90",
        "is_new_mcc_for_card",
        "is_new_country_for_card",
        "is_new_entry_mode_for_card",
        "is_new_device_for_customer",
        "is_new_ip_for_customer",
        "merchant_tx_count_24h",
        "merchant_distinct_cards_24h",
        "is_dormant_wakeup_shock",
        "dormancy_shock_index",
    }
    assert expected_v3_cols.issubset(set(v3_df.columns))

    row3 = v3_df.filter(pl.col("transaction_id") == 3).row(0, named=True)
    assert row3["card_prior_tx_count"] == 2
    assert row3["is_new_country_for_card"] == 1
    assert row3["is_new_entry_mode_for_card"] == 1
    assert row3["is_new_device_for_customer"] == 1
    assert row3["is_new_ip_for_customer"] == 1
    assert row3["is_dormant_wakeup_shock"] == 1


def test_preprocessor_v3_pipeline() -> None:
    """Verifica que el preprocesador scikit-learn soporte feature_version='v3'."""
    prep = create_preprocessor(cv_target_encoder=2, feature_version="v3")
    assert prep is not None
    assert len(ALL_MODEL_FEATURES_V3) == 47


def test_classify_fn_archetype_categories() -> None:
    """Verifica la clasificación mutuamente excluyente de los arquetipos de FN."""
    row_near = pd.Series(
        {
            "y_proba": 0.18,
            "total_frauds_on_card": 1,
            "fraud_seq_num": 1,
            "channel": "POS",
            "entry_mode": "chip",
            "is_country_mismatch": 0,
            "cards_seen_by_device_24h": 0,
        }
    )
    assert "Casi-Alerta" in classify_fn_archetype(row_near)

    row_p0 = pd.Series(
        {
            "y_proba": 0.02,
            "total_frauds_on_card": 3,
            "fraud_seq_num": 1,
            "channel": "POS",
            "entry_mode": "chip",
            "is_country_mismatch": 0,
            "cards_seen_by_device_24h": 0,
        }
    )
    assert "Paciente Cero" in classify_fn_archetype(row_p0)
