"""Funciones modulares de visualización interactiva en Plotly para detección de fraude.

Sigue estándares de SOLID, plantilla plotly_white y renderizado sin dependencias estáticas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PRESENTATION_DIR

# Paleta corporativa y semántica de riesgo
COLOR_FRAUD = "#DC2626"  # Rojo Alerta
COLOR_LEGIT = "#2563EB"  # Azul Seguro
COLOR_ACCENT = "#059669"  # Verde Éxito
COLOR_NEUTRAL = "#6B7280"  # Gris Neutro
TEMPLATE = "plotly_white"
DEFAULT_FIGURES_DIR = PRESENTATION_DIR / "figures"


def save_interactive_plot(
    fig: go.Figure, filename: str, output_dir: Path | None = None
) -> Path:
    """Guarda una figura interactiva en formato HTML autónomo."""
    target_dir = output_dir or DEFAULT_FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / (
        filename if filename.endswith(".html") else f"{filename}.html"
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path


def plot_conditional_missingness(df_missing: pd.DataFrame) -> go.Figure:
    """Genera gráfico de barras agrupadas de porcentaje de nulos por canal y variable."""
    fig = go.Figure()
    channels = df_missing["channel"].unique()
    colors = ["#1E3A8A", "#2563EB", "#60A5FA", "#93C5FD"]

    for idx, ch in enumerate(channels):
        sub = df_missing[df_missing["channel"] == ch]
        fig.add_trace(
            go.Bar(
                name=str(ch),
                x=sub["feature"],
                y=sub["null_pct"],
                text=sub["null_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="auto",
                marker_color=colors[idx % len(colors)],
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        title="Porcentaje de Valores Nulos Condicionados por Canal Transaccional",
        xaxis_title="Variable",
        yaxis_title="% Valores Faltantes",
        barmode="group",
        legend_title="Canal",
        yaxis={"range": [0, 105]},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_amount_distribution(
    sample_df: pd.DataFrame, max_points: int = 50000
) -> go.Figure:
    """Compara la distribución de montos entre transacciones legítimas y fraudulentas."""
    plot_data = sample_df.head(max_points)
    legit = plot_data[plot_data["is_fraud"] == 0]["amount"]
    fraud = plot_data[plot_data["is_fraud"] == 1]["amount"]

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=legit,
            name="Legítimo (0)",
            marker_color=COLOR_LEGIT,
            boxpoints="outliers",
        )
    )
    fig.add_trace(
        go.Box(
            y=fraud,
            name="Fraude (1)",
            marker_color=COLOR_FRAUD,
            boxpoints="outliers",
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Distribución de Monto de Transacción por Etiqueta de Fraude (Escala Logarítmica)",
        yaxis_title="Monto (USD)",
        yaxis_type="log",
        showlegend=True,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_hourly_seasonality(df_hourly: pd.DataFrame) -> go.Figure:
    """Grafica el volumen transaccional horario y la tasa de fraude correspondiente."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_hourly["hour"],
            y=df_hourly["total_tx"],
            name="Volumen Transacciones",
            marker_color="#94A3B8",
            opacity=0.6,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_hourly["hour"],
            y=df_hourly["fraud_rate_pct"],
            name="Tasa de Fraude (%)",
            mode="lines+markers",
            line={"color": COLOR_FRAUD, "width": 3},
            marker={"size": 8, "color": COLOR_FRAUD},
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Estacionalidad Horaria: Volumen vs Tasa de Fraude",
        xaxis={"title": "Hora del Día (0-23)", "tickmode": "linear", "dtick": 1},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    fig.update_yaxes(title_text="Volumen de Transacciones", secondary_y=False)
    fig.update_yaxes(title_text="Tasa de Fraude (%)", secondary_y=True, showgrid=False)
    return fig


def plot_mcc_risk(df_mcc: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Grafica las categorías comerciales (MCC) con mayor tasa de fraude y su volumen."""
    top_mcc = df_mcc.sort_values(by="fraud_rate_pct", ascending=True).tail(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[f"{row['mcc']} - {row['description']}" for _, row in top_mcc.iterrows()],
            x=top_mcc["fraud_rate_pct"],
            orientation="h",
            marker={
                "color": top_mcc["fraud_rate_pct"],
                "colorscale": "Reds",
                "showscale": False,
            },
            text=top_mcc["fraud_rate_pct"].apply(lambda v: f"{v:.2f}%"),
            textposition="auto",
            customdata=top_mcc[["total_tx", "fraud_count"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Tasa Fraude: %{x:.2f}%<br>"
                "Total Transacciones: %{customdata[0]:,}<br>"
                "Casos Fraude: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title=f"Top {top_n} Rubros Comerciales (MCC) por Tasa de Incidencia de Fraude",
        xaxis_title="Tasa de Fraude (%)",
        yaxis_title="Rubro Comercial (MCC)",
        margin={"l": 150, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_channel_entry_matrix(df_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap de tasa de fraude cruzando canal y modo de captura."""
    pivoted = df_matrix.pivot(
        index="channel", columns="entry_mode", values="fraud_rate_pct"
    ).fillna(0.0)

    map_fn = getattr(pivoted, "map", getattr(pivoted, "applymap", None))
    text_vals = map_fn(lambda v: f"{v:.2f}%" if v > 0 else "N/A").values

    fig = go.Figure(
        data=go.Heatmap(
            z=pivoted.values,
            x=pivoted.columns.tolist(),
            y=pivoted.index.tolist(),
            colorscale="Reds",
            text=text_vals,
            texttemplate="%{text}",
            colorbar={"title": "% Fraude"},
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Matriz de Riesgo: Canal vs Modo de Captura (Tasa de Fraude %)",
        xaxis_title="Modo de Entrada (entry_mode)",
        yaxis_title="Canal (channel)",
        margin={"l": 60, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_geographic_mismatch_risk(df_geo: pd.DataFrame) -> go.Figure:
    """Compara tasa de fraude y volumen según coincidencia o discrepancia geográfica."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_geo["mismatch_label"],
            y=df_geo["fraud_rate_pct"],
            text=df_geo["fraud_rate_pct"].apply(lambda v: f"{v:.2f}%"),
            textposition="auto",
            marker_color=[COLOR_LEGIT, COLOR_FRAUD],
            customdata=df_geo[["total_tx", "fraud_count"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Tasa Fraude: %{y:.2f}%<br>"
                "Transacciones: %{customdata[0]:,}<br>"
                "Fraudes: %{customdata[1]:,}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title="Impacto del Descalce Geográfico (País Comercio vs País IP)",
        xaxis_title="Condición Geográfica",
        yaxis_title="Tasa de Fraude (%)",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Importancia de Características (Feature Importance)",
) -> go.Figure:
    """Genera un gráfico de barras horizontal interactivo con las variables más predictivas."""
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx][::-1]
    top_scores = [float(importances[i]) for i in sorted_idx][::-1]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_scores,
            y=top_features,
            orientation="h",
            marker={
                "color": top_scores,
                "colorscale": "Blues",
                "showscale": False,
            },
            text=[f"{s:.4f}" for s in top_scores],
            textposition="auto",
        )
    )

    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis_title="Importancia Relativa",
        yaxis_title="Característica",
        margin={"l": 160, "r": 40, "t": 60, "b": 40},
    )
    return fig
