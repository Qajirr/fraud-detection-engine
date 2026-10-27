"""Módulo de optimización de umbrales de decisión financieros y análisis de sensibilidad."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from src.config import BUSINESS_MATRIX, BusinessCostMatrix


@dataclass
class BusinessThresholdResult:
    """Resultado estructurado de la calibración de umbral optimizando utilidad financiera."""

    optimal_threshold: float
    max_utility: float
    tp_at_optimal: int
    fp_at_optimal: int
    fn_at_optimal: int
    tn_at_optimal: int
    precision_at_optimal: float
    recall_at_optimal: float
    f1_at_optimal: float
    thresholds: np.ndarray
    utilities: np.ndarray
    precisions: np.ndarray
    recalls: np.ndarray


def compute_financial_utility(
    tp: int,
    fp: int,
    fn: int,
    vn: int = 0,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
) -> float:
    """Calcula la utilidad económica neta según la matriz de costos unitarios del emisor."""
    return (
        matrix.profit_tp * tp
        - matrix.cost_fp * fp
        - matrix.cost_fn * fn
        + matrix.benefit_vn * vn
    )


def compute_vectorized_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evalúa vectorizadamente la matriz de confusión y métricas financieras para un array de umbrales."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    total_pos = int(np.sum(y_arr == 1))

    # Ordenar por score descendente para cálculo de suma acumulativa O(N log N)
    sort_idx = np.argsort(p_arr)[::-1]
    sorted_p = p_arr[sort_idx]
    sorted_y = y_arr[sort_idx]

    # Para cada umbral candidato, encontrar cuántos elementos caen en predicción positiva (p >= th)
    # searchsorted en array decreciente invertido
    pos_counts = len(sorted_p) - np.searchsorted(
        sorted_p[::-1], thresholds, side="left"
    )

    # Sumas acumuladas de verdaderos positivos en los primeros k elementos
    cum_positives = np.concatenate(([0], np.cumsum(sorted_y)))

    tp_arr = cum_positives[pos_counts]
    fp_arr = pos_counts - tp_arr
    fn_arr = total_pos - tp_arr

    utilities = (
        matrix.profit_tp * tp_arr - matrix.cost_fp * fp_arr - matrix.cost_fn * fn_arr
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.where(pos_counts > 0, tp_arr / pos_counts, 0.0)
        recalls = np.where(total_pos > 0, tp_arr / total_pos, 0.0)

    return (
        utilities.astype(float),
        precisions.astype(float),
        recalls.astype(float),
        tp_arr.astype(int),
        fp_arr.astype(int),
        fn_arr.astype(int),
    )


def find_optimal_business_threshold(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray,
    steps: int = 1000,
    exact_thresholds: bool = False,
    min_precision: float | None = None,
    max_alerts: int | None = None,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
) -> BusinessThresholdResult:
    """Encuentra el umbral óptimo maximizando la utilidad financiera mediante evaluación vectorizada.

    Parameters
    ----------
    y_true : np.ndarray | list[int]
        Etiquetas binarias reales (0 o 1).
    y_proba : np.ndarray
        Probabilidades predichas de fraude P(Y=1).
    steps : int, default=1000
        Número de particiones en la grilla de alta resolución.
    exact_thresholds : bool, default=False
        Si es True, evalúa sobre los valores únicos reales de probabilidad.
    min_precision : float | None, default=None
        Restricción de precisión mínima obligatoria en la decisión positiva.
    max_alerts : int | None, default=None
        Restricción de capacidad máxima de alertas (TP + FP).
    matrix : BusinessCostMatrix, default=BUSINESS_MATRIX
        Ponderaciones económicas oficiales del banco.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    if exact_thresholds:
        candidate_thresholds = np.unique(np.clip(p_arr, 0.001, 0.999))
        candidate_thresholds.sort()
    else:
        candidate_thresholds = np.linspace(0.005, 0.995, steps)

    (
        utilities,
        precisions,
        recalls,
        tp_arr,
        fp_arr,
        fn_arr,
    ) = compute_vectorized_threshold_metrics(
        y_arr, p_arr, candidate_thresholds, matrix=matrix
    )

    valid_mask = np.ones(len(candidate_thresholds), dtype=bool)

    if min_precision is not None:
        valid_mask &= precisions >= min_precision

    if max_alerts is not None:
        valid_mask &= (tp_arr + fp_arr) <= max_alerts

    if not np.any(valid_mask):
        # Fallback al óptimo libre si ninguna combinación cumple las restricciones
        best_idx = int(np.argmax(utilities))
    else:
        masked_utilities = np.where(valid_mask, utilities, -np.inf)
        best_idx = int(np.argmax(masked_utilities))

    best_th = float(candidate_thresholds[best_idx])
    best_tp = int(tp_arr[best_idx])
    best_fp = int(fp_arr[best_idx])
    best_fn = int(fn_arr[best_idx])
    best_tn = int(len(y_arr) - np.sum(y_arr == 1) - best_fp)

    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])
    best_f1 = (
        float(2 * (best_prec * best_rec) / (best_prec + best_rec))
        if (best_prec + best_rec) > 0
        else 0.0
    )

    # Para graficación suave, si se usaron exact_thresholds, muestrear grilla uniforme
    if exact_thresholds:
        plot_ths = np.linspace(0.01, 0.99, min(steps, 500))
        (
            plot_utils,
            plot_precs,
            plot_recs,
            _,
            _,
            _,
        ) = compute_vectorized_threshold_metrics(y_arr, p_arr, plot_ths, matrix=matrix)
    else:
        plot_ths = candidate_thresholds
        plot_utils = utilities
        plot_precs = precisions
        plot_recs = recalls

    return BusinessThresholdResult(
        optimal_threshold=best_th,
        max_utility=float(utilities[best_idx]),
        tp_at_optimal=best_tp,
        fp_at_optimal=best_fp,
        fn_at_optimal=best_fn,
        tn_at_optimal=best_tn,
        precision_at_optimal=best_prec,
        recall_at_optimal=best_rec,
        f1_at_optimal=best_f1,
        thresholds=plot_ths,
        utilities=plot_utils,
        precisions=plot_precs,
        recalls=plot_recs,
    )


def plot_business_utility_tradeoff(
    tuning: BusinessThresholdResult,
    model_name: str = "Modelo",
) -> go.Figure:
    """Visualiza la ganancia económica neta en función del umbral de decisión."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=tuning.thresholds,
            y=tuning.utilities,
            mode="lines",
            name="Utilidad Financiera Neta (US$)",
            line={"color": "#2ca02c", "width": 3},
        )
    )

    fig.add_vline(
        x=tuning.optimal_threshold,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Óptimo = {tuning.optimal_threshold:.3f} (Max: US${tuning.max_utility:,.0f})",
        annotation_position="top left",
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Curva de Utilidad Financiera Neta vs. Umbral de Decisión ({model_name})",
        xaxis_title="Umbral de Decisión (Threshold)",
        yaxis_title="Ganancia Financiera Neta (US$)",
        legend={"x": 0.05, "y": 0.9},
    )
    return fig
