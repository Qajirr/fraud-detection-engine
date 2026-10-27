"""Módulo de explicabilidad de modelos con TreeSHAP, Waterfall, redes de grafos y gráficos interactivos Plotly."""

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compute_tree_shap_values(
    model: Any,
    x_data: np.ndarray | pd.DataFrame,
    check_additivity: bool = False,
) -> np.ndarray:
    """Calcula valores SHAP exactos usando TreeExplainer optimizado para árboles."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_data, check_additivity=check_additivity)

    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        return np.asarray(shap_vals[1])
    if hasattr(shap_vals, "values"):
        vals = shap_vals.values
        if vals.ndim == 3:
            return vals[:, :, 1]
        return vals
    return np.asarray(shap_vals)


def plot_shap_summary_plotly(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    title: str = "Importancia Global de Variables (Mean |SHAP Value|)",
) -> go.Figure:
    """Genera gráfico interactivo de barras con el impacto global promedio de cada variable."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    sorted_features = [feature_names[i] for i in top_indices][::-1]
    sorted_values = [mean_abs_shap[i] for i in top_indices][::-1]

    fig = go.Figure(
        go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation="h",
            marker={
                "color": sorted_values,
                "colorscale": "Viridis",
                "colorbar": {"title": "|SHAP| Medio"},
            },
            hovertemplate="<b>%{y}</b><br>Impacto Medio: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 18, "family": "Arial"}},
        xaxis={"title": "Mean |SHAP Value|", "showgrid": True},
        yaxis={"title": "Variable"},
        margin={"l": 180, "r": 40, "t": 60, "b": 50},
        height=520,
    )
    return fig


def plot_shap_dependence_plotly(
    shap_values: np.ndarray,
    x_data: np.ndarray,
    feature_names: list[str],
    feature_name: str,
    color_feature_name: str | None = None,
    sample_size: int = 5000,
    random_state: int = 42,
) -> go.Figure:
    """Genera gráfico de dispersión de dependencia parcial interactivo en Plotly."""
    feat_idx = feature_names.index(feature_name)
    n_rows = len(x_data)
    if n_rows > sample_size:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_rows, size=sample_size, replace=False)
    else:
        indices = np.arange(n_rows)

    x_vals = x_data[indices, feat_idx]
    y_vals = shap_values[indices, feat_idx]
    marker_opts: dict[str, Any] = {"size": 5, "opacity": 0.7}
    if color_feature_name and color_feature_name in feature_names:
        c_idx = feature_names.index(color_feature_name)
        marker_opts.update(
            color=x_data[indices, c_idx],
            colorscale="Thermal",
            colorbar={"title": color_feature_name},
            showscale=True,
        )
    else:
        marker_opts["color"] = "#2563EB"

    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=marker_opts,
            hovertemplate=f"<b>{feature_name}</b>: %{{x:.3f}}<br>SHAP: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title={
            "text": f"SHAP Dependence: {feature_name}",
            "font": {"size": 18, "family": "Arial"},
        },
        xaxis={"title": f"Valor de {feature_name}", "showgrid": True},
        yaxis={"title": "SHAP Value", "showgrid": True},
        height=480,
    )
    return fig


def plot_attention_heatmap_plotly(
    attention_matrix: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    title: str = "Matriz de Atención Inter-Variable (Self-Attention Weights)",
) -> go.Figure:
    """Genera un mapa de calor interactivo de pesos medios de auto-atención."""
    mean_attn = (
        np.mean(attention_matrix, axis=0)
        if attention_matrix.ndim == 3
        else attention_matrix
    )
    top_indices = np.argsort(np.sum(mean_attn, axis=1))[::-1][:top_k]
    sub_matrix = mean_attn[np.ix_(top_indices, top_indices)]
    sub_names = [feature_names[i] for i in top_indices]

    fig = go.Figure(
        data=go.Heatmap(
            z=sub_matrix,
            x=sub_names,
            y=sub_names,
            colorscale="Blues",
            colorbar={"title": "Peso Atención"},
            hovertemplate="Fuente: <b>%{y}</b><br>Destino: <b>%{x}</b><br>Peso: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 18, "family": "Arial"}},
        xaxis={"tickangle": -45},
        yaxis={"autorange": "reversed"},
        height=520,
    )
    return fig


def plot_shap_waterfall_plotly(
    base_value: float,
    shap_values_single: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray | None = None,
    top_n: int = 10,
    title: str = "SHAP Waterfall - Auditoría de Alerta de Fraude Individual",
) -> go.Figure:
    """Genera un gráfico Waterfall interactivo de decisión individual para comités o analistas."""
    top_indices = np.argsort(np.abs(shap_values_single))[::-1][:top_n]
    x_deltas = [base_value]
    measure = ["absolute"]
    y_labels = ["Valor Base E[f(x)]"]

    for idx in top_indices:
        val = float(shap_values_single[idx])
        name = feature_names[idx]
        y_labels.append(
            f"{name} = {feature_values[idx]:.2f}"
            if feature_values is not None
            else name
        )
        x_deltas.append(val)
        measure.append("relative")

    final_prediction = base_value + float(np.sum(shap_values_single))
    x_deltas.append(final_prediction)
    measure.append("total")
    y_labels.append("Score Final f(x)")

    fig = go.Figure(
        go.Waterfall(
            name="Contribución",
            orientation="v",
            measure=measure,
            x=y_labels,
            textposition="outside",
            text=[
                f"{v:+.3f}" if m == "relative" else f"{v:.3f}"
                for v, m in zip(x_deltas, measure)
            ],
            y=x_deltas,
            connector={"line": {"color": "#7f8c8d"}},
            decreasing={"marker": {"color": "#2ecc71"}},
            increasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#34495e"}},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "font": {"size": 18, "family": "Arial"}},
        xaxis={"tickangle": -35},
        yaxis={"title": "Contribución a Probabilidad de Fraude", "showgrid": True},
        height=520,
    )
    return fig


def plot_fraud_network_plotly(
    graph: nx.Graph,
    title: str = "Topología de Red: Anillos de Fraude y Dispositivos Compartidos",
) -> go.Figure:
    """Visualiza un subgrafo interactivo de entidades sospechosas en Plotly."""
    pos = nx.spring_layout(graph, seed=42, k=0.5)
    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1.2, "color": "#bdc3c7"},
        hoverinfo="none",
        mode="lines",
    )

    node_types = nx.get_node_attributes(graph, "node_type")
    type_styles = {
        "card": {"color": "#e74c3c", "symbol": "circle"},
        "device": {"color": "#3498db", "symbol": "square"},
        "ip": {"color": "#e67e22", "symbol": "diamond"},
        "merchant": {"color": "#9b59b6", "symbol": "triangle-up"},
    }

    node_x, node_y, node_text, node_colors, node_symbols = [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = node_types.get(node, "card")
        node_text.append(
            f"<b>{node}</b><br>Tipo: {ntype}<br>Grado: {graph.degree(node)}"
        )
        node_colors.append(type_styles.get(ntype, type_styles["card"])["color"])
        node_symbols.append(type_styles.get(ntype, type_styles["card"])["symbol"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker={
            "color": node_colors,
            "symbol": node_symbols,
            "size": 14,
            "line": {"width": 1.5, "color": "#2c3e50"},
        },
    )

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={"text": title, "font": {"size": 18, "family": "Arial"}},
            template="plotly_white",
            showlegend=False,
            margin={"b": 40, "l": 40, "r": 40, "t": 60},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=540,
        ),
    )


def save_figure_html(fig: go.Figure, output_path: str | Path) -> None:
    """Guarda la figura Plotly en formato HTML interactivo autocontenido."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
