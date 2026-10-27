"""Generador de visualización comparativa interactiva Plotly para el Experimento Fase A."""

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PRESENTATION_DIR


def generate_experiment_comparison_figure(
    metrics_v1: dict,
    metrics_v2: dict,
    save_path: Path | None = None,
) -> go.Figure:
    """Crea una figura interactiva de 2 paneles comparando métricas financieras y operativas."""
    if save_path is None:
        save_path = PRESENTATION_DIR / "figures" / "17_comparacion_features_v2.html"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Impacto Financiero Neto en Test OOT (Mayo 2026)",
            "Diagnóstico de Clasificación y Falsas Alarmas",
        ),
    )

    models = ["XGBoost (Features v1)", "XGBoost (Features 2.0)"]
    net_utilities = [
        metrics_v1["Financial_Utility_USD"],
        metrics_v2["Financial_Utility_USD"],
    ]
    savings = [metrics_v1["savings_vs_inaction"], metrics_v2["savings_vs_inaction"]]

    # Panel 1: Utilidad Neta y Ahorro vs Inacción
    fig.add_trace(
        go.Bar(
            name="Utilidad Neta (US$)",
            x=models,
            y=net_utilities,
            marker_color=["#1f77b4", "#2ca02c"],
            text=[f"US$ {u:,.0f}" for u in net_utilities],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Ahorro vs Inacción (US$)",
            x=models,
            y=savings,
            marker_color=["#aec7e8", "#98df8a"],
            text=[f"+US$ {s:,.0f}" for s in savings],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Falsas Alarmas (FP) y Fraudes Detectados (VP)
    fps = [metrics_v1["FP"], metrics_v2["FP"]]
    vps = [metrics_v1["TP"], metrics_v2["TP"]]

    fig.add_trace(
        go.Bar(
            name="Falsas Alarmas (FP)",
            x=models,
            y=fps,
            marker_color=["#d62728", "#ff9896"],
            text=[f"{fp:,} FPs" for fp in fps],
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Fraudes Detectados (VP)",
            x=models,
            y=vps,
            marker_color=["#2ca02c", "#1f77b4"],
            text=[f"{vp:,} VPs" for vp in vps],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        title=(
            "Experimento Fase A: Impacto de Feature Engineering 2.0<br>"
            "<sup>Ganancia incremental de +US$ 3.780 y reducción de 236 falsas alarmas (-26.5%)</sup>"
        ),
        barmode="group",
        height=500,
        width=1000,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    return fig


if __name__ == "__main__":
    v1_sample = {
        "Financial_Utility_USD": -25260.0,
        "savings_vs_inaction": 144840.0,
        "FP": 892,
        "TP": 572,
    }
    v2_sample = {
        "Financial_Utility_USD": -21480.0,
        "savings_vs_inaction": 148620.0,
        "FP": 656,
        "TP": 561,
    }
    generate_experiment_comparison_figure(v1_sample, v2_sample)
    print("Figura 17_comparacion_features_v2.html generada exitosamente.")
