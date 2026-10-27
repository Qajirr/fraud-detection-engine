"""Pruebas unitarias para TreeSHAP, mapas de atención y gráficos interactivos Plotly."""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from lightgbm import LGBMClassifier

from src.models.explainability import (
    compute_tree_shap_values,
    plot_attention_heatmap_plotly,
    plot_fraud_network_plotly,
    plot_shap_dependence_plotly,
    plot_shap_summary_plotly,
    plot_shap_waterfall_plotly,
)


def test_tree_shap_values_computation() -> None:
    rng = np.random.RandomState(42)
    n_samples, n_feat = 80, 5
    x = rng.randn(n_samples, n_feat)
    y = (rng.rand(n_samples) > 0.7).astype(int)

    clf = LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    clf.fit(x, y)

    shap_vals = compute_tree_shap_values(clf, x)
    assert shap_vals.shape == (n_samples, n_feat)
    assert not np.isnan(shap_vals).any()


def test_plotly_explainability_figures() -> None:
    rng = np.random.RandomState(42)
    n_samples, n_feat = 50, 4
    shap_vals = rng.randn(n_samples, n_feat)
    x_data = rng.randn(n_samples, n_feat)
    feature_names = [f"feat_{i}" for i in range(n_feat)]

    fig_summary = plot_shap_summary_plotly(shap_vals, feature_names)
    assert isinstance(fig_summary, go.Figure)

    fig_dep = plot_shap_dependence_plotly(
        shap_values=shap_vals,
        x_data=x_data,
        feature_names=feature_names,
        feature_name="feat_0",
        color_feature_name="feat_1",
    )
    assert isinstance(fig_dep, go.Figure)

    dummy_attn = rng.rand(n_samples, n_feat, n_feat)
    fig_attn = plot_attention_heatmap_plotly(dummy_attn, feature_names)
    assert isinstance(fig_attn, go.Figure)

    # Test waterfall plot
    fig_waterfall = plot_shap_waterfall_plotly(
        base_value=0.003,
        shap_values_single=shap_vals[0],
        feature_names=feature_names,
    )
    assert isinstance(fig_waterfall, go.Figure)

    # Test fraud network plot
    g = nx.Graph()
    g.add_node("card_1", node_type="card")
    g.add_node("dev_1", node_type="device")
    g.add_edge("card_1", "dev_1")
    fig_net = plot_fraud_network_plotly(g)
    assert isinstance(fig_net, go.Figure)
