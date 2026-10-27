"""Módulo de calibración de probabilidades de fraude para estimadores desbalanceados."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


@dataclass
class CalibrationMetrics:
    """Métricas diagnósticas de confiabilidad probabilística."""

    brier_score_uncalibrated: float
    brier_score_calibrated: float
    expected_calibration_error: float
    prob_true_uncalibrated: np.ndarray
    prob_pred_uncalibrated: np.ndarray
    prob_true_calibrated: np.ndarray
    prob_pred_calibrated: np.ndarray


def calibrate_fraud_model(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    method: Literal["isotonic", "sigmoid"] = "isotonic",
) -> CalibratedClassifierCV:
    """Ajusta un calibrador scikit-learn sobre el conjunto de validación.

    En scikit-learn moderno (1.6+) se utiliza FrozenEstimator; en versiones previas cv='prefit'.
    Garantiza que el clasificador optimizado con scale_pos_weight devuelva probabilidades
    posteriores fidedignas alineadas con la prevalencia real del fenómeno (0.3%).
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
        )
    except (ImportError, AttributeError):
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )

    calibrator.fit(x_val, y_val)
    return calibrator


def compute_expected_calibration_error(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    bin_counts: np.ndarray | None = None,
) -> float:
    """Calcula el Expected Calibration Error (ECE) ponderado por bin."""
    abs_diff = np.abs(prob_true - prob_pred)
    if bin_counts is not None and np.sum(bin_counts) > 0:
        weights = bin_counts / np.sum(bin_counts)
        return float(np.sum(weights * abs_diff))
    return float(np.mean(abs_diff))


def evaluate_probability_calibration(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Calcula y compara métricas de confiabilidad antes y después de calibrar."""
    y_arr = np.asarray(y_true)
    brier_uncal = float(brier_score_loss(y_arr, y_proba_uncal))
    brier_cal = float(brier_score_loss(y_arr, y_proba_cal))

    p_true_uncal, p_pred_uncal = calibration_curve(
        y_arr, y_proba_uncal, n_bins=n_bins, strategy="uniform"
    )
    p_true_cal, p_pred_cal = calibration_curve(
        y_arr, y_proba_cal, n_bins=n_bins, strategy="uniform"
    )

    ece = compute_expected_calibration_error(p_true_cal, p_pred_cal)

    return CalibrationMetrics(
        brier_score_uncalibrated=brier_uncal,
        brier_score_calibrated=brier_cal,
        expected_calibration_error=ece,
        prob_true_uncalibrated=p_true_uncal,
        prob_pred_uncalibrated=p_pred_uncal,
        prob_true_calibrated=p_true_cal,
        prob_pred_calibrated=p_pred_cal,
    )


def plot_reliability_curve_plotly(
    metrics: CalibrationMetrics,
    model_name: str = "XGBoost",
) -> go.Figure:
    """Genera curva de confiabilidad interactiva Plotly comparando calibración."""
    fig = go.Figure()

    # Línea ideal (perfectamente calibrada)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#7f8c8d", "width": 1.5},
            name="Calibración Perfecta (Diagonal)",
        )
    )

    # Curva sin calibrar
    fig.add_trace(
        go.Scatter(
            x=metrics.prob_pred_uncalibrated,
            y=metrics.prob_true_uncalibrated,
            mode="lines+markers",
            marker={"size": 8, "color": "#e74c3c"},
            line={"color": "#e74c3c", "width": 2},
            name=f"{model_name} Sin Calibrar (Brier: {metrics.brier_score_uncalibrated:.5f})",
        )
    )

    # Curva calibrada
    fig.add_trace(
        go.Scatter(
            x=metrics.prob_pred_calibrated,
            y=metrics.prob_true_calibrated,
            mode="lines+markers",
            marker={"size": 8, "color": "#2ecc71"},
            line={"color": "#2ecc71", "width": 2.5},
            name=f"{model_name} Calibrado Isotonic (Brier: {metrics.brier_score_calibrated:.5f})",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title={
            "text": f"Curva de Confiabilidad Probabilística (Reliability Curve) - {model_name}",
            "font": {"size": 18, "family": "Arial"},
        },
        xaxis={
            "title": "Probabilidad Media Predicha",
            "range": [0, 1],
            "showgrid": True,
        },
        yaxis={
            "title": "Fracción Real de Positivos (Fraude)",
            "range": [0, 1],
            "showgrid": True,
        },
        legend={
            "x": 0.05,
            "y": 0.95,
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "bordercolor": "#bdc3c7",
            "borderwidth": 1,
        },
        height=520,
    )
    return fig
