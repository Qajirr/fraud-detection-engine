"""Módulo de evaluación diagnóstica, métricas de rendimiento y gráficos Plotly."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import BUSINESS_MATRIX, BusinessCostMatrix
from src.models.threshold import (
    BusinessThresholdResult,
    compute_financial_utility,
    find_optimal_business_threshold,
    plot_business_utility_tradeoff,
)

__all__ = [
    "BusinessThresholdResult",
    "compute_detailed_metrics",
    "compute_financial_utility",
    "find_optimal_business_threshold",
    "plot_business_utility_tradeoff",
    "plot_confusion_matrix",
    "plot_precision_recall_curves",
    "plot_roc_curves",
]


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
) -> dict[str, Any]:
    """Calcula el conjunto exhaustivo de métricas de clasificación y negocio."""
    y_arr = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pr_auc = float(average_precision_score(y_arr, y_proba))
    roc_auc = float(roc_auc_score(y_arr, y_proba))
    prec = float(precision_score(y_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_arr, y_pred, zero_division=0))
    f05 = float(fbeta_score(y_arr, y_pred, beta=0.5, zero_division=0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    utility = compute_financial_utility(tp, fp, fn, tn, matrix=matrix)

    return {
        "threshold": round(threshold, 4),
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "F0.5-Score": round(f05, 4),
        "Specificity": round(specificity, 4),
        "Financial_Utility_USD": round(utility, 2),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def plot_precision_recall_curves(
    results: dict[str, Any],
    y_val: np.ndarray,
) -> go.Figure:
    """Genera curvas Precision-Recall comparativas para los modelos candidatos."""
    fig = go.Figure()
    y_arr = np.asarray(y_val)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, res) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_arr, res.y_proba)
        fig.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=f"{name} (AUPRC = {res.pr_auc:.4f})",
                line={"color": colors[i % len(colors)], "width": 2},
            )
        )

    baseline = float(np.mean(y_arr))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"Baseline (Prevalencia = {baseline:.4%})",
            line={"color": "#7f7f7f", "dash": "dash"},
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Curvas Precision-Recall (AUPRC) - Comparación de Modelos",
        xaxis_title="Recall (Sensibilidad)",
        yaxis_title="Precision (Precisión)",
        xaxis={"range": [0.0, 1.05]},
        yaxis={"range": [0.0, 1.05]},
        legend={"x": 0.02, "y": 0.15},
    )
    return fig


def plot_roc_curves(
    results: dict[str, Any],
    y_val: np.ndarray,
) -> go.Figure:
    """Genera curvas ROC comparativas."""
    fig = go.Figure()
    y_arr = np.asarray(y_val)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_arr, res.y_proba)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (ROC-AUC = {res.roc_auc:.4f})",
                line={"color": colors[i % len(colors)], "width": 2},
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Curvas ROC - Comparación de Modelos (AUC Azar = 0.50)",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        xaxis={"range": [0.0, 1.05]},
        yaxis={"range": [0.0, 1.05]},
        legend={"x": 0.6, "y": 0.15},
    )
    return fig


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    threshold: float,
    model_name: str,
) -> go.Figure:
    """Genera un mapa de calor interactivo de la matriz de confusión con Plotly."""
    labels = ["Legítima (0)", "Fraude (1)"]
    cm_arr = np.asarray(cm)
    z_text = [[f"{val:,}" for val in row] for row in cm_arr]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverinfo="text",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Matriz de Confusión: {model_name} (Umbral = {threshold:.2f})",
        xaxis_title="Predicción del Modelo",
        yaxis_title="Etiqueta Real",
        yaxis={"autorange": "reversed"},
    )
    return fig
