"""Pruebas unitarias para el módulo de análisis de redes y grafos."""

from datetime import datetime

import polars as pl
import pytest

from src.features.network import (
    build_bipartite_network,
    compute_graph_degree_metrics,
    compute_network_rolling_features,
    detect_fraud_rings,
)


@pytest.fixture
def network_transactions() -> pl.DataFrame:
    """Dataset con transacciones estructuradas para análisis de redes."""
    return pl.DataFrame(
        {
            "transaction_id": ["tx1", "tx2", "tx3", "tx4", "tx5"],
            "timestamp": [
                datetime(2026, 3, 1, 10, 0),
                datetime(2026, 3, 1, 11, 0),
                datetime(2026, 3, 1, 12, 0),
                datetime(2026, 3, 1, 13, 0),
                datetime(2026, 3, 2, 18, 0),  # > 24h después
            ],
            "card_id": ["card_1", "card_2", "card_1", "card_3", "card_1"],
            "merchant_id": ["m_A", "m_B", "m_C", "m_A", "m_A"],
            "merchant_country": ["CHL", "CHL", "USA", "CHL", "CHL"],
            "device_id": ["dev_X", "dev_X", "dev_Y", "dev_Z", "dev_X"],
            "ip_address": ["ip_1", "ip_1", "ip_2", "ip_3", "ip_1"],
        }
    )


def test_build_bipartite_network(network_transactions: pl.DataFrame):
    """Verifica la construcción del grafo heterogéneo."""
    g = build_bipartite_network(network_transactions)

    assert "card_card_1" in g.nodes
    assert "card_card_2" in g.nodes
    assert "dev_dev_X" in g.nodes
    assert "ip_ip_1" in g.nodes

    # card_1 y dev_X deben estar conectados
    assert g.has_edge("card_card_1", "dev_dev_X")
    assert g.has_edge("card_card_2", "dev_dev_X")


def test_detect_fraud_rings(network_transactions: pl.DataFrame):
    """Verifica la detección de anillos de fraude compartiendo dispositivos/IPs."""
    g = build_bipartite_network(network_transactions)
    rings = detect_fraud_rings(g, min_cards=2)

    assert len(rings) >= 1
    top_ring = rings[0]
    # card_1 y card_2 comparten dev_X e ip_1
    assert top_ring["card_count"] >= 2
    assert "card_card_1" in top_ring["nodes"]
    assert "card_card_2" in top_ring["nodes"]


def test_compute_network_rolling_features(network_transactions: pl.DataFrame):
    """Verifica que las métricas de red en ventanas causales no tengan fuga temporal."""
    res = compute_network_rolling_features(network_transactions)

    assert "card_distinct_merchants_24h" in res.columns
    assert "card_distinct_countries_24h" in res.columns

    # tx1 es primera de card_1 (comercio m_A, país CHL)
    tx1 = res.filter(pl.col("transaction_id") == "tx1").to_dicts()[0]
    assert tx1["card_distinct_merchants_24h"] == 1
    assert tx1["card_distinct_countries_24h"] == 1

    # tx3 es segunda de card_1 a las 12:00 (comercio m_C, país USA)
    tx3 = res.filter(pl.col("transaction_id") == "tx3").to_dicts()[0]
    assert tx3["card_distinct_merchants_24h"] == 2
    assert tx3["card_distinct_countries_24h"] == 2


def test_compute_graph_degree_metrics(network_transactions: pl.DataFrame):
    """Verifica el cálculo de grados de conectividad."""
    g = build_bipartite_network(network_transactions)
    degrees = compute_graph_degree_metrics(g)

    # dev_X conecta con card_1 y card_2 -> grado >= 2
    assert degrees["dev_dev_X"] >= 2
