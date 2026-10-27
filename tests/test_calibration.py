"""Pruebas unitarias para el módulo de calibración de probabilidades."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.models.calibration import (
    calibrate_fraud_model,
    evaluate_probability_calibration,
    plot_reliability_curve_plotly,
)


@pytest.fixture
def synthetic_val_data():
    """Genera datos sintéticos desbalanceados para calibración."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    x = rng.randn(n_samples, 4)
    # Target desbalanceado ~1%
    logits = x[:, 0] * 1.5 + x[:, 1] * 0.5 - 3.0
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_samples) < probs).astype(int)
    return x, y


def test_calibrate_fraud_model(synthetic_val_data):
    """Verifica el ajuste y predicción con CalibratedClassifierCV."""
    x, y = synthetic_val_data
    base_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    base_model.fit(x, y)

    calibrated = calibrate_fraud_model(base_model, x, y, method="isotonic")
    cal_probs = calibrated.predict_proba(x)[:, 1]

    assert cal_probs.shape == (len(y),)
    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)


def test_evaluate_probability_calibration(synthetic_val_data):
    """Verifica el cálculo de Brier Score y curvas de calibración."""
    _x, y = synthetic_val_data
    rng = np.random.RandomState(42)
    uncal_probs = rng.uniform(0.1, 0.9, size=len(y))
    cal_probs = rng.uniform(0.01, 0.1, size=len(y))

    metrics = evaluate_probability_calibration(y, uncal_probs, cal_probs)

    assert metrics.brier_score_uncalibrated >= 0.0
    assert metrics.brier_score_calibrated >= 0.0
    assert metrics.expected_calibration_error >= 0.0


def test_plot_reliability_curve_plotly(synthetic_val_data):
    """Verifica que la curva de confiabilidad se renderice como go.Figure."""
    _x, y = synthetic_val_data
    rng = np.random.RandomState(42)
    uncal_probs = rng.uniform(0.1, 0.9, size=len(y))
    cal_probs = rng.uniform(0.01, 0.1, size=len(y))

    metrics = evaluate_probability_calibration(y, uncal_probs, cal_probs)
    fig = plot_reliability_curve_plotly(metrics, model_name="TestModel")

    assert fig is not None
    assert len(fig.data) == 3  # Diagonal, uncal, cal
