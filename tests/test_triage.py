"""Pruebas unitarias para el módulo de política tri-zona (Triage)."""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.models.triage import (
    compute_triage_metrics,
    find_optimal_dual_thresholds,
    find_optimal_dual_thresholds_optuna,
    plot_triage_surface_plotly,
    plot_triage_zones_plotly,
)


@pytest.fixture
def synthetic_triage_data():
    """Genera datos sintéticos con etiquetas y probabilidades para pruebas de triage."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    y_true = np.zeros(n_samples, dtype=int)
    # 20 fraudes (2%)
    fraud_indices = rng.choice(n_samples, size=20, replace=False)
    y_true[fraud_indices] = 1

    # Probabilidades sintéticas
    y_proba = rng.beta(0.5, 20.0, size=n_samples)
    y_proba[fraud_indices] = rng.beta(5.0, 1.0, size=20)  # scores altos para fraudes

    return y_true, y_proba


def test_compute_triage_metrics_validity(synthetic_triage_data):
    """Verifica que el cálculo de métricas en 3 zonas sea exacto y consistente."""
    y_true, y_proba = synthetic_triage_data
    t_low = 0.15
    t_high = 0.70

    res = compute_triage_metrics(y_true, y_proba, t_low=t_low, t_high=t_high)

    assert res.t_low == t_low
    assert res.t_high == t_high
    assert res.green_total + res.yellow_total + res.red_total == len(y_true)
    assert res.total_vp == res.yellow_vp + res.red_vp
    assert res.total_fp == res.yellow_fp + res.red_fp
    assert res.total_fn == res.green_fn

    # Comprobación de fórmula de utilidad exacta del banco
    expected_utility = (
        (120.0 * res.total_vp) - (30.0 * res.total_fp) - (180.0 * res.total_fn)
    )
    assert pytest.approx(res.total_utility) == expected_utility


def test_compute_triage_metrics_asymmetric_costs(synthetic_triage_data):
    """Verifica que los costos diferenciados para zona amarilla y roja se calculen bien."""
    y_true, y_proba = synthetic_triage_data
    t_low = 0.20
    t_high = 0.70

    res = compute_triage_metrics(
        y_true,
        y_proba,
        t_low=t_low,
        t_high=t_high,
        cost_fp_yellow=10.0,
        cost_fp_red=60.0,
    )

    expected_utility = (
        (120.0 * res.total_vp)
        - (10.0 * res.yellow_fp)
        - (60.0 * res.red_fp)
        - (180.0 * res.total_fn)
    )
    assert pytest.approx(res.total_utility) == expected_utility


def test_compute_triage_metrics_invalid_thresholds(synthetic_triage_data):
    """Verifica que se lance ValueError si t_low >= t_high."""
    y_true, y_proba = synthetic_triage_data
    with pytest.raises(ValueError, match="estrictamente menor"):
        compute_triage_metrics(y_true, y_proba, t_low=0.5, t_high=0.3)


def test_find_optimal_dual_thresholds(synthetic_triage_data):
    """Verifica que la optimización en malla encuentre umbrales válidos."""
    y_true, y_proba = synthetic_triage_data
    best_res, grid_data = find_optimal_dual_thresholds(
        y_true, y_proba, min_precision_high=0.75, n_thresholds=20
    )

    assert best_res.t_low < best_res.t_high
    assert best_res.total_utility > -180.0 * int(np.sum(y_true))
    assert "thresholds" in grid_data
    assert "grid_utility" in grid_data


def test_find_optimal_dual_thresholds_optuna(synthetic_triage_data):
    """Verifica que la optimización continua con Optuna converja a umbrales válidos."""
    y_true, y_proba = synthetic_triage_data
    best_res = find_optimal_dual_thresholds_optuna(
        y_true, y_proba, min_precision_high=0.75, n_trials=15, random_state=42
    )

    assert best_res.t_low < best_res.t_high
    assert best_res.total_utility > -180.0 * int(np.sum(y_true))
    assert best_res.red_precision >= 0.75 or best_res.red_total == 0


def test_plot_triage_figures(synthetic_triage_data):
    """Verifica que se generen correctamente los gráficos interactivos en Plotly."""
    y_true, y_proba = synthetic_triage_data
    best_res, grid_data = find_optimal_dual_thresholds(
        y_true, y_proba, min_precision_high=0.75, n_thresholds=15
    )

    fig_zones = plot_triage_zones_plotly(y_true, y_proba, best_res)
    assert isinstance(fig_zones, go.Figure)

    fig_surface = plot_triage_surface_plotly(grid_data, best_res.t_low, best_res.t_high)
    assert isinstance(fig_surface, go.Figure)
