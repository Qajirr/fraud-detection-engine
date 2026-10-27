"""Pruebas unitarias para las utilidades de visualización en src.utils.plots."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.utils.plots import (
    plot_amount_distribution,
    plot_channel_entry_matrix,
    plot_conditional_missingness,
    plot_geographic_mismatch_risk,
    plot_hourly_seasonality,
    plot_mcc_risk,
    save_interactive_plot,
)


@pytest.fixture
def sample_missing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["POS", "POS", "ECOMMERCE", "ECOMMERCE"],
            "feature": ["device_id", "ip_address", "device_id", "ip_address"],
            "null_pct": [100.0, 100.0, 0.0, 0.0],
        }
    )


@pytest.fixture
def sample_amount_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [15.5, 45.0, 1200.0, 25.0, 500.0],
            "is_fraud": [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_hourly_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": list(range(24)),
            "total_tx": [1000] * 24,
            "fraud_count": [3] * 24,
            "fraud_rate_pct": [0.3] * 24,
        }
    )


@pytest.fixture
def sample_mcc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mcc": [7995, 6051, 5411],
            "description": ["Apuestas", "Cripto", "Supermercado"],
            "total_tx": [5000, 3000, 50000],
            "fraud_count": [250, 120, 15],
            "fraud_rate_pct": [5.0, 4.0, 0.03],
        }
    )


@pytest.fixture
def sample_matrix_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["POS", "POS", "ECOMMERCE"],
            "entry_mode": ["chip", "swipe", "ecommerce"],
            "fraud_rate_pct": [0.1, 1.2, 0.8],
        }
    )


@pytest.fixture
def sample_geo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mismatch_label": ["Coincide (Mismo País)", "Descalce (País Distinto)"],
            "total_tx": [90000, 10000],
            "fraud_count": [180, 250],
            "fraud_rate_pct": [0.2, 2.5],
        }
    )


def test_plot_conditional_missingness(sample_missing_df: pd.DataFrame):
    fig = plot_conditional_missingness(sample_missing_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_amount_distribution(sample_amount_df: pd.DataFrame):
    fig = plot_amount_distribution(sample_amount_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_hourly_seasonality(sample_hourly_df: pd.DataFrame):
    fig = plot_hourly_seasonality(sample_hourly_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_mcc_risk(sample_mcc_df: pd.DataFrame):
    fig = plot_mcc_risk(sample_mcc_df, top_n=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_channel_entry_matrix(sample_matrix_df: pd.DataFrame):
    fig = plot_channel_entry_matrix(sample_matrix_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_plot_geographic_mismatch_risk(sample_geo_df: pd.DataFrame):
    fig = plot_geographic_mismatch_risk(sample_geo_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_save_interactive_plot(tmp_path: Path):
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    saved_path = save_interactive_plot(fig, "test_chart", output_dir=tmp_path)
    assert saved_path.exists()
    assert saved_path.suffix == ".html"
    assert saved_path.stat().st_size > 0
