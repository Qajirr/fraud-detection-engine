"""Pruebas unitarias para la evaluación diagnóstica y optimización financiera."""

import numpy as np
from plotly.graph_objects import Figure

from src.models.evaluation import (
    BusinessThresholdResult,
    compute_detailed_metrics,
    compute_financial_utility,
    find_optimal_business_threshold,
    plot_business_utility_tradeoff,
    plot_confusion_matrix,
)


def test_compute_financial_utility():
    """Valida la fórmula de utilidad financiera: 120*TP - 30*FP - 180*FN."""
    # Caso 1: 10 TP, 5 FP, 2 FN
    # Utilidad = 10*120 - 5*30 - 2*180 = 1200 - 150 - 360 = 690
    utility = compute_financial_utility(tp=10, fp=5, fn=2, vn=100)
    assert utility == 690.0

    # Caso 2: Todo falso negativo
    utility_fn = compute_financial_utility(tp=0, fp=0, fn=10, vn=100)
    assert utility_fn == -1800.0


def test_find_optimal_business_threshold():
    """Verifica que la optimización encuentre el umbral que maximiza la ganancia neta."""
    np.random.seed(42)
    y_true = np.array([0] * 90 + [1] * 10)
    # Probabilidades con señal predictiva clara
    y_proba = np.where(
        y_true == 1,
        np.random.uniform(0.7, 0.95, 100),
        np.random.uniform(0.01, 0.3, 100),
    )

    result = find_optimal_business_threshold(y_true, y_proba, steps=50)

    assert isinstance(result, BusinessThresholdResult)
    assert 0.01 <= result.optimal_threshold <= 0.99
    assert result.max_utility > 0
    assert result.tp_at_optimal > 0
    assert len(result.thresholds) == 50
    assert len(result.utilities) == 50


def test_compute_detailed_metrics():
    """Verifica el cómputo de métricas exhaustivas y presencia de campos requeridos."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.9])

    metrics = compute_detailed_metrics(y_true, y_proba, threshold=0.5)

    assert "Financial_Utility_USD" in metrics
    assert "PR-AUC" in metrics
    assert "ROC-AUC" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-Score" in metrics
    assert metrics["TP"] == 2
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0
    assert metrics["TN"] == 2


def test_plotly_figures_generation():
    """Verifica que las funciones de visualización retornen instancias válidas de Plotly Figure."""
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 10)
    y_proba = np.random.uniform(0, 1, 60)

    tuning_res = find_optimal_business_threshold(y_true, y_proba, steps=20)
    fig_util = plot_business_utility_tradeoff(tuning_res, model_name="TestModel")
    assert isinstance(fig_util, Figure)

    cm = np.array([[45, 5], [2, 8]])
    fig_cm = plot_confusion_matrix(cm, threshold=0.5, model_name="TestModel")
    assert isinstance(fig_cm, Figure)
