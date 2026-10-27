"""Pruebas unitarias para el módulo de optimización de umbrales financieros."""

import numpy as np
import plotly.graph_objects as go
import pytest

from src.models.threshold import (
    BusinessThresholdResult,
    compute_financial_utility,
    compute_vectorized_threshold_metrics,
    find_optimal_business_threshold,
    plot_business_utility_tradeoff,
)


@pytest.fixture
def synthetic_threshold_data():
    """Genera datos sintéticos de prueba con prevalencia baja."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    y_true = np.zeros(n_samples, dtype=int)
    fraud_idx = rng.choice(n_samples, size=30, replace=False)
    y_true[fraud_idx] = 1

    # Probabilidades sintéticas
    y_proba = rng.beta(0.5, 20.0, size=n_samples)
    y_proba[fraud_idx] = rng.beta(5.0, 1.5, size=30)

    return y_true, y_proba


def test_compute_financial_utility():
    """Valida el cálculo aritmético exacto de la función de costo financiero."""
    # VP = 10 (+1200), FP = 5 (-150), FN = 2 (-360) -> 1200 - 150 - 360 = 690
    utility = compute_financial_utility(tp=10, fp=5, fn=2, vn=100)
    assert utility == 690.0

    # Inacción: 0 VP, 0 FP, 10 FN -> -1800
    assert compute_financial_utility(tp=0, fp=0, fn=10) == -1800.0


def test_compute_vectorized_threshold_metrics(synthetic_threshold_data):
    """Verifica que la vectorización calcule exactamente métricas y matrices."""
    y_true, y_proba = synthetic_threshold_data
    thresholds = np.array([0.1, 0.5, 0.9])

    utils, precs, _recs, tps, fps, fns = compute_vectorized_threshold_metrics(
        y_true, y_proba, thresholds
    )

    assert len(utils) == len(thresholds)
    assert len(precs) == len(thresholds)
    assert len(tps) == len(thresholds)

    # Validar que a mayor umbral, menor número de alertas (TP + FP)
    alerts = tps + fps
    assert alerts[0] >= alerts[1] >= alerts[2]

    # Validar coherencia de TP + FN = total positivos
    total_pos = int(np.sum(y_true))
    np.testing.assert_array_equal(tps + fns, total_pos)


def test_find_optimal_business_threshold_resolution(synthetic_threshold_data):
    """Verifica la búsqueda de umbral en grilla de alta resolución (1000 pasos)."""
    y_true, y_proba = synthetic_threshold_data
    res = find_optimal_business_threshold(y_true, y_proba, steps=1000)

    assert isinstance(res, BusinessThresholdResult)
    assert 0.0 < res.optimal_threshold < 1.0
    assert res.max_utility > -180.0 * int(np.sum(y_true))

    expected_util = compute_financial_utility(
        res.tp_at_optimal, res.fp_at_optimal, res.fn_at_optimal, res.tn_at_optimal
    )
    assert pytest.approx(res.max_utility) == expected_util
    assert len(res.thresholds) == 1000
    assert len(res.utilities) == 1000


def test_find_optimal_business_threshold_exact(synthetic_threshold_data):
    """Verifica el modo de thresholds exactos derivados de los valores de predicción."""
    y_true, y_proba = synthetic_threshold_data
    res = find_optimal_business_threshold(y_true, y_proba, exact_thresholds=True)

    assert isinstance(res, BusinessThresholdResult)
    assert 0.0 < res.optimal_threshold < 1.0
    assert res.max_utility > -180.0 * int(np.sum(y_true))


def test_find_optimal_business_threshold_constraints(synthetic_threshold_data):
    """Verifica el comportamiento cuando se imponen restricciones operativas."""
    y_true, y_proba = synthetic_threshold_data

    # Restricción de precisión alta
    res_constrained = find_optimal_business_threshold(
        y_true, y_proba, steps=500, min_precision=0.70
    )
    assert (
        res_constrained.precision_at_optimal >= 0.70
        or res_constrained.tp_at_optimal == 0
    )

    # Restricción de alertas máximas
    res_cap = find_optimal_business_threshold(y_true, y_proba, steps=500, max_alerts=15)
    assert (res_cap.tp_at_optimal + res_cap.fp_at_optimal) <= 15


def test_plot_business_utility_tradeoff(synthetic_threshold_data):
    """Verifica que el gráfico Plotly se renderice con plantilla plotly_white."""
    y_true, y_proba = synthetic_threshold_data
    res = find_optimal_business_threshold(y_true, y_proba, steps=200)

    fig = plot_business_utility_tradeoff(res, model_name="Test Model")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.layout.title.text.startswith("Curva de Utilidad")
