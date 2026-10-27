"""Pruebas unitarias para el módulo de ingeniería de características."""

from datetime import datetime

import polars as pl
import pytest

from src.features.builder import build_feature_dataset
from src.features.flags import add_risk_flags
from src.features.velocity import (
    compute_card_rolling_features,
    compute_card_time_deltas,
    compute_digital_multiplicity,
)


@pytest.fixture
def sample_transactions() -> pl.DataFrame:
    """Fixture con transacciones representativas en orden cronológico."""
    return pl.DataFrame(
        {
            "transaction_id": ["tx1", "tx2", "tx3", "tx4", "tx5"],
            "timestamp": [
                datetime(2026, 3, 1, 2, 0),  # Madrugada (hora 2)
                datetime(2026, 3, 1, 2, 30),  # Madrugada +30m misma tarjeta
                datetime(2026, 3, 1, 4, 0),  # Madrugada +2h misma tarjeta
                datetime(2026, 3, 1, 14, 0),  # Tarde otra tarjeta
                datetime(2026, 3, 1, 14, 10),  # Tarde digital mismatch
            ],
            "card_id": ["card_A", "card_A", "card_A", "card_B", "card_C"],
            "customer_id": ["cust_1", "cust_1", "cust_1", "cust_2", "cust_3"],
            "merchant_id": ["m1", "m2", "m3", "m4", "m5"],
            "mcc": [7995, 5411, 6051, 5812, 5311],  # 7995 y 6051 son de alto riesgo
            "merchant_country": ["CHL", "CHL", "USA", "CHL", "USA"],
            "channel": ["POS", "POS", "ECOMMERCE", "ATM", "ECOMMERCE"],
            "entry_mode": ["swipe", "chip", "ecommerce", "pin", "ecommerce"],
            "amount": [100.0, 50.0, 300.0, 20.0, 500.0],
            "device_id": [None, None, "dev_1", None, "dev_1"],  # dev_1 compartido
            "ip_address": [None, None, "1.1.1.1", None, "2.2.2.2"],
            "ip_country": [None, None, "USA", None, "CHL"],  # tx5 tiene mismatch
            "is_fraud": [0, 0, 1, 0, 1],
        }
    )


def test_add_risk_flags(sample_transactions: pl.DataFrame):
    """Verifica que los flags de riesgo se calculen correctamente sin falsos positivos."""
    res = add_risk_flags(sample_transactions)

    assert "is_country_mismatch" in res.columns
    assert "is_high_risk_mcc" in res.columns
    assert "is_swipe" in res.columns
    assert "is_digital_channel" in res.columns
    assert "is_card_present" in res.columns
    assert "is_night_transaction" in res.columns

    # tx1 es swipe, noche, mcc 7995 (riesgo), POS presencial
    tx1 = res.filter(pl.col("transaction_id") == "tx1").to_dicts()[0]
    assert tx1["is_swipe"] == 1
    assert tx1["is_night_transaction"] == 1
    assert tx1["is_high_risk_mcc"] == 1
    assert tx1["is_country_mismatch"] == 0  # ip_country es null
    assert tx1["is_card_present"] == 1
    assert tx1["is_digital_channel"] == 0

    # tx5 es mismatch geográfico (USA vs CHL) y digital
    tx5 = res.filter(pl.col("transaction_id") == "tx5").to_dicts()[0]
    assert tx5["is_country_mismatch"] == 1
    assert tx5["is_digital_channel"] == 1
    assert tx5["is_card_present"] == 0


def test_compute_card_time_deltas(sample_transactions: pl.DataFrame):
    """Verifica el cálculo de tiempo entre transacciones de la misma tarjeta."""
    res = compute_card_time_deltas(sample_transactions)

    tx1 = res.filter(pl.col("transaction_id") == "tx1").to_dicts()[0]
    tx2 = res.filter(pl.col("transaction_id") == "tx2").to_dicts()[0]

    # tx1 es primera transacción de card_A
    assert tx1["delta_time_last_tx"] == -1.0
    assert tx1["is_first_card_tx"] == 1

    # tx2 ocurrió 30 min (1800 seg) después
    assert tx2["delta_time_last_tx"] == 1800.0
    assert tx2["is_first_card_tx"] == 0


def test_compute_card_rolling_features_causality(sample_transactions: pl.DataFrame):
    """Verifica que las ventanas móviles sean estrictamente causales (sin fuga hacia el futuro)."""
    res = compute_card_rolling_features(sample_transactions)

    # tx1: Primera transacción de card_A -> conteo y suma acumulada previa deben ser 0
    tx1 = res.filter(pl.col("transaction_id") == "tx1").to_dicts()[0]
    assert tx1["card_tx_count_1h"] == 0
    assert tx1["card_tx_count_6h"] == 0
    assert tx1["card_amount_sum_1h"] == 0.0
    assert tx1["card_amount_sum_6h"] == 0.0

    # tx2: Ocurrió a las 02:30 -> Dentro de 1h y 6h de tx1 ($100)
    tx2 = res.filter(pl.col("transaction_id") == "tx2").to_dicts()[0]
    assert tx2["card_tx_count_1h"] == 1
    assert tx2["card_tx_count_6h"] == 1
    assert tx2["card_amount_sum_1h"] == 100.0
    assert tx2["card_amount_sum_6h"] == 100.0

    # tx3: Ocurrió a las 04:00 -> Más de 1h desde tx1 y tx2, pero dentro de 6h y 24h
    tx3 = res.filter(pl.col("transaction_id") == "tx3").to_dicts()[0]
    assert tx3["card_tx_count_1h"] == 0
    assert tx3["card_tx_count_6h"] == 2
    assert tx3["card_tx_count_24h"] == 2
    assert tx3["card_amount_sum_6h"] == 150.0  # tx1(100) + tx2(50)
    assert tx3["card_amount_sum_24h"] == 150.0


def test_compute_digital_multiplicity(sample_transactions: pl.DataFrame):
    """Verifica la detección de dispositivos compartidos por múltiples tarjetas."""
    res = compute_digital_multiplicity(sample_transactions)

    # dev_1 es usado por card_A (tx3) y por card_C (tx5) dentro de 24h
    tx5 = res.filter(pl.col("transaction_id") == "tx5").to_dicts()[0]
    assert tx5["cards_seen_by_device_24h"] == 2

    # tx4 es ATM -> device_id es null, cards_seen debe ser 0
    tx4 = res.filter(pl.col("transaction_id") == "tx4").to_dicts()[0]
    assert tx4["cards_seen_by_device_24h"] == 0


def test_build_feature_dataset_pipeline(sample_transactions: pl.DataFrame):
    """Verifica que el pipeline de construcción maestro opere de extremo a extremo sin error."""
    res = build_feature_dataset(df=sample_transactions, save_path=None)

    assert isinstance(res, pl.DataFrame)
    assert res.height == sample_transactions.height
    expected_cols = [
        "is_country_mismatch",
        "is_high_risk_mcc",
        "is_card_present",
        "delta_time_last_tx",
        "card_tx_count_1h",
        "card_tx_count_6h",
        "card_amount_sum_1h",
        "card_amount_sum_6h",
        "card_tx_count_24h",
        "card_amount_sum_24h",
        "ratio_amount_mean_7d",
        "cards_seen_by_device_24h",
    ]
    for col in expected_cols:
        assert col in res.columns
