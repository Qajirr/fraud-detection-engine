"""Funciones de visualización interactiva Plotly para la política tri-zona (Triage)."""

from typing import Any

import numpy as np
import plotly.graph_objects as go


def plot_triage_zones_plotly(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    result: Any,
    title: str = "Política de Decisión Tri-Zona: Semáforo de Riesgo Transaccional",
) -> go.Figure:
    """Genera gráfico interactivo de distribución con franjas semáforo Verde, Amarilla y Roja."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    p_legit = p_arr[y_arr == 0]
    p_fraud = p_arr[y_arr == 1]

    fig = go.Figure()

    # Histograma de legítimas
    fig.add_trace(
        go.Histogram(
            x=p_legit,
            name="Transacciones Legítimas",
            nbinsx=60,
            histnorm="probability density",
            marker_color="rgba(44, 123, 182, 0.6)",
            opacity=0.6,
        )
    )

    # Histograma de fraudes
    fig.add_trace(
        go.Histogram(
            x=p_fraud,
            name="Transacciones Fraude",
            nbinsx=60,
            histnorm="probability density",
            marker_color="rgba(215, 25, 28, 0.7)",
            opacity=0.7,
        )
    )

    # Bandas sombreadas para el semáforo de 3 zonas
    fig.add_vrect(
        x0=0.0,
        x1=result.t_low,
        fillcolor="rgba(46, 204, 113, 0.15)",
        layer="below",
        line_width=1,
        line_color="green",
        annotation_text="ZONA VERDE<br>Aprobación Directa<br>(VN: $0 | FN: -$180)",
        annotation_position="top left",
    )
    fig.add_vrect(
        x0=result.t_low,
        x1=result.t_high,
        fillcolor="rgba(241, 196, 15, 0.20)",
        layer="below",
        line_width=1,
        line_color="goldenrod",
        annotation_text="ZONA AMARILLA<br>Revisión 2da Instancia<br>(Fricción: -$30 | VP: +$120)",
        annotation_position="top left",
    )
    fig.add_vrect(
        x0=result.t_high,
        x1=1.0,
        fillcolor="rgba(231, 76, 60, 0.20)",
        layer="below",
        line_width=1,
        line_color="red",
        annotation_text=f"ZONA ROJA<br>Bloqueo Inmediato<br>(Precisión: {result.red_precision:.1%})",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_white",
        title=f"<b>{title}</b><br><sup>T_low = {result.t_low:.3f} | T_high = {result.t_high:.3f} | Utilidad: US$ {result.total_utility:,.0f}</sup>",
        xaxis_title="Score de Probabilidad P(Fraude)",
        yaxis_title="Densidad de Probabilidad",
        barmode="overlay",
        legend={"x": 0.4, "y": 0.95, "bgcolor": "rgba(255,255,255,0.8)"},
        width=1000,
        height=550,
    )

    return fig


def plot_triage_surface_plotly(
    grid_data: dict[str, Any],
    optimal_t_low: float,
    optimal_t_high: float,
    title: str = "Superficie de Utilidad Financiera Neta: Optimización de Doble Umbral",
) -> go.Figure:
    """Genera heatmap interactivo 2D mostrando la ganancia neta en función de (T_low, T_high)."""
    thresholds = grid_data["thresholds"]
    z_matrix = grid_data["grid_utility"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=thresholds,
            y=thresholds,
            colorscale="Viridis",
            colorbar={"title": "Utilidad (US$)"},
            hoverongaps=False,
        )
    )

    # Punto óptimo
    fig.add_trace(
        go.Scatter(
            x=[optimal_t_high],
            y=[optimal_t_low],
            mode="markers+text",
            marker={
                "color": "red",
                "size": 14,
                "symbol": "cross",
                "line": {"width": 2},
            },
            text=[f"Óptimo ({optimal_t_low:.3f}, {optimal_t_high:.3f})"],
            textposition="top center",
            name="Par Óptimo (T_low*, T_high*)",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"<b>{title}</b><br><sup>Evaluación en Grid Search sobre Validación OOT</sup>",
        xaxis_title="Umbral Superior T_high (Bloqueo Directo)",
        yaxis_title="Umbral Inferior T_low (Envío a Revisión)",
        width=900,
        height=600,
    )

    return fig
