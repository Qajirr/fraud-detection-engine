"""Pruebas unitarias para el módulo de ingeniería de características 2.0."""

from datetime import datetime

import polars as pl
import pytest

from src.features.builder import build_feature_dataset_v2
from src.features.customer import compute_customer_rolling_features
from src.features.flags import add_impossible_geo_jump_flag
from src.features.mcc_profile import add_mcc_amount_ratio


@pytest.fixture
def sample_transactions_v2() -> pl.DataFrame:
    """Fixture con transacciones representativas multi-tarjeta y multi-país."""
    return pl.DataFrame(
        {
            "transaction_id": ["tx1", "tx2", "tx3", "tx4", "tx5"],
            "timestamp": [
                datetime(2026, 3, 1, 10, 0),  # tx1: card_A en CHL (cust_1)
                datetime(
                    2026, 3, 1, 10, 30
                ),  # tx2: card_A en USA 30min después (salto imposible)
                datetime(
                    2026, 3, 1, 10, 45
                ),  # tx3: card_B en CHL (mismo cust_1, 2da tarjeta a los 45m)
                datetime(2026, 3, 1, 12, 0),  # tx4: card_C en CHL (cust_2)
                datetime(2026, 3, 1, 16, 0),  # tx5: card_A en USA 5.5h después
            ],
            "card_id": ["card_A", "card_A", "card_B", "card_C", "card_A"],
            "customer_id": ["cust_1", "cust_1", "cust_1", "cust_2", "cust_1"],
            "merchant_id": ["m1", "m2", "m3", "m4", "m5"],
            "mcc": [5411, 5411, 5411, 7995, 5411],  # 5411 Supermercado
            "merchant_country": ["CHL", "USA", "CHL", "CHL", "USA"],
            "channel": ["POS", "ECOMMERCE", "POS", "POS", "ECOMMERCE"],
            "entry_mode": ["chip", "ecommerce", "chip", "chip", "ecommerce"],
            "amount": [50.0, 500.0, 75.0, 100.0, 200.0],
            "device_id": [None, "dev_1", None, None, "dev_1"],
            "ip_address": [None, "1.1.1.1", None, None, "1.1.1.1"],
            "ip_country": [None, "USA", None, None, "USA"],
            "is_fraud": [0, 1, 0, 0, 1],
        }
    )


def test_add_impossible_geo_jump_flag(sample_transactions_v2: pl.DataFrame):
    """Verifica que un cambio de país en menos de 2 horas sea marcado como salto imposible."""
    res = add_impossible_geo_jump_flag(sample_transactions_v2)

    assert "is_impossible_geo_jump" in res.columns
    tx2 = res.filter(pl.col("transaction_id") == "tx2").to_dicts()[0]
    # tx2 es card_A en USA solo 30 min (1800s) después de estar en CHL
    assert tx2["is_impossible_geo_jump"] == 1

    # tx1 es la primera de card_A -> no hay salto
    tx1 = res.filter(pl.col("transaction_id") == "tx1").to_dicts()[0]
    assert tx1["is_impossible_geo_jump"] == 0

    # tx5 es card_A en USA 5 horas después de tx3 (que era card_B) y 5.5h después de tx2
    tx5 = res.filter(pl.col("transaction_id") == "tx5").to_dicts()[0]
    assert tx5["is_impossible_geo_jump"] == 0


def test_compute_customer_rolling_features(sample_transactions_v2: pl.DataFrame):
    """Verifica la agregación de transacciones y plásticos a nivel cliente en ventanas móviles."""
    res = compute_customer_rolling_features(sample_transactions_v2)

    assert "customer_tx_count_1h" in res.columns
    assert "customer_tx_count_24h" in res.columns
    assert "customer_amount_sum_24h" in res.columns
    assert "customer_distinct_cards_24h" in res.columns

    # tx1 (cust_1): primera transacción -> 0 previas
    tx1 = res.filter(pl.col("transaction_id") == "tx1").to_dicts()[0]
    assert tx1["customer_tx_count_1h"] == 0
    assert tx1["customer_amount_sum_24h"] == 0.0

    # tx3 (cust_1 con card_B a las 11:00): ve tx1 (10:00) y tx2 (10:30) dentro de 1h
    tx3 = res.filter(pl.col("transaction_id") == "tx3").to_dicts()[0]
    assert tx3["customer_tx_count_1h"] == 2
    assert tx3["customer_tx_count_24h"] == 2
    assert tx3["customer_amount_sum_24h"] == 550.0  # 50 + 500
    assert tx3["customer_distinct_cards_24h"] == 2  # ha usado card_A y card_B


def test_add_mcc_amount_ratio(sample_transactions_v2: pl.DataFrame):
    """Verifica el cálculo de ratio de monto respecto a la mediana del rubro MCC."""
    res = add_mcc_amount_ratio(sample_transactions_v2)
    assert "ratio_amount_mcc_median" in res.columns

    # tx2 ($500) en supermercado donde el resto son montos menores -> ratio > 1.0
    tx2 = res.filter(pl.col("transaction_id") == "tx2").to_dicts()[0]
    assert tx2["ratio_amount_mcc_median"] > 1.0


def test_build_feature_dataset_v2_pipeline(sample_transactions_v2: pl.DataFrame):
    """Verifica la ejecución integral del pipeline de features 2.0."""
    res = build_feature_dataset_v2(df=sample_transactions_v2, save_path=None)

    assert res.height == sample_transactions_v2.height
    v2_expected_cols = [
        "is_impossible_geo_jump",
        "burst_ratio_1h_24h",
        "customer_tx_count_1h",
        "customer_tx_count_24h",
        "customer_amount_sum_24h",
        "customer_distinct_cards_24h",
        "ratio_amount_mcc_median",
        "card_distinct_merchants_24h",
        "card_distinct_countries_24h",
    ]
    for col in v2_expected_cols:
        assert col in res.columns
