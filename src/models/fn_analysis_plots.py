"""Visualizaciones interactivas en Plotly para el diagnóstico de Falsos Negativos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_diagnostic_plots(
    seq_df: pd.DataFrame,
    arch_df: pd.DataFrame,
    shap_gap: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Genera las figuras interactivas Plotly (20, 21 y 22) sin colisión de texto."""
    seq_sub = seq_df[seq_df["fraud_seq_num"] <= 5].copy()
    x_lbl = [f"Fraude #{int(n)}" for n in seq_sub["fraud_seq_num"]]
    arch_sorted = arch_df.sort_values("casos", ascending=True)

    # Figura 20: Dinámica Secuencial y Anatomía de Falsos Negativos (sin solapamiento)
    fig_seq = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.28,
        subplot_titles=(
            "Recall vs. Omisión por Orden del Fraude (V1)",
            "Taxonomía de los 390 Falsos Negativos",
        ),
    )
    fig_seq.add_trace(
        go.Bar(
            name="Detectado (VP %)",
            x=x_lbl,
            y=seq_sub["recall_pct"],
            marker_color="#059669",
            text=[f"{v:.0f}%" for v in seq_sub["recall_pct"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    fig_seq.add_trace(
        go.Bar(
            name="Omitido (FN %)",
            x=x_lbl,
            y=seq_sub["fn_rate_pct"],
            marker_color="#DC2626",
            text=[f"{v:.0f}%" for v in seq_sub["fn_rate_pct"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    fig_seq.add_trace(
        go.Bar(
            name="Casos FN",
            y=arch_sorted["refined_archetype"],
            x=arch_sorted["casos"],
            orientation="h",
            marker_color="#2563EB",
            text=[
                f"{c} ({p:.1f}%)"
                for c, p in zip(arch_sorted["casos"], arch_sorted["pct"])
            ],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig_seq.update_yaxes(title_text="% de Fraudes", range=[0, 118], row=1, col=1)
    fig_seq.update_xaxes(title_text="Número de Casos FN", range=[0, 175], row=1, col=2)
    fig_seq.update_yaxes(automargin=True, row=1, col=2)
    fig_seq.update_layout(
        template="plotly_white",
        barmode="group",
        title={
            "text": "Diagnóstico de Falsos Negativos: Efecto 'Paciente Cero' y Taxonomía de Fuga",
            "x": 0.02,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.09,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"l": 60, "r": 70, "t": 110, "b": 60},
        height=560,
        width=1240,
        uniformtext_minsize=9,
        uniformtext_mode="show",
    )
    fig_seq.write_html(
        str(out_dir / "20_fn_anatomia_y_secuencia.html"), include_plotlyjs="cdn"
    )

    # Figura 21: Brecha TreeSHAP entre VP y FN (con rango expandido y etiquetas limpias)
    top_gap = shap_gap.head(10).iloc[::-1]
    fig_shap = go.Figure()
    fig_shap.add_trace(
        go.Bar(
            name="SHAP Medio en VP (Detectados)",
            y=top_gap["feature"],
            x=top_gap["mean_shap_VP"],
            orientation="h",
            marker_color="#059669",
            text=[f"{v:+.2f}" for v in top_gap["mean_shap_VP"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig_shap.add_trace(
        go.Bar(
            name="SHAP Medio en FN (No Detectados)",
            y=top_gap["feature"],
            x=top_gap["mean_shap_FN"],
            orientation="h",
            marker_color="#DC2626",
            text=[f"{v:+.2f}" for v in top_gap["mean_shap_FN"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig_shap.update_layout(
        template="plotly_white",
        barmode="group",
        title={
            "text": "Explicabilidad TreeSHAP: Brecha de Señal entre Fraudes Detectados (VP) y Omitidos (FN)",
            "x": 0.02,
        },
        xaxis={
            "title": "Contribución Promedio al Log-Odds de Fraude (Valor SHAP)",
            "range": [-0.35, 2.35],
        },
        yaxis={"title": "Variable del Modelo", "automargin": True},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"l": 210, "r": 70, "t": 100, "b": 60},
        height=580,
        width=1120,
    )
    fig_shap.write_html(
        str(out_dir / "21_fn_vs_vp_shap_gap.html"), include_plotlyjs="cdn"
    )

    # Figura 22: Rescate de Falsos Negativos con Destilación 3.0
    fig_rescue = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.46, 0.54],
        horizontal_spacing=0.28,
        subplot_titles=(
            "Recall por Orden del Fraude (V1 vs V2 vs V3)",
            "Rescate por Arquetipo de Falso Negativo (Destilación V3)",
        ),
    )
    fig_rescue.add_trace(
        go.Bar(
            name="V1 Base (58.7%)",
            x=x_lbl,
            y=seq_sub["recall_pct"],
            marker_color="#94A3B8",
            text=[f"{v:.0f}%" for v in seq_sub["recall_pct"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    fig_rescue.add_trace(
        go.Bar(
            name="V2 Enriquecido (60.2%)",
            x=x_lbl,
            y=seq_sub["recall_v2_pct"],
            marker_color="#3B82F6",
            text=[f"{v:.0f}%" for v in seq_sub["recall_v2_pct"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    fig_rescue.add_trace(
        go.Bar(
            name="V3 Destilado (77.0%)",
            x=x_lbl,
            y=seq_sub["recall_v3_pct"],
            marker_color="#059669",
            text=[f"{v:.0f}%" for v in seq_sub["recall_v3_pct"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    fig_rescue.add_trace(
        go.Bar(
            name="Rescatados por V3",
            y=arch_sorted["refined_archetype"],
            x=arch_sorted["rescatados_v3"],
            orientation="h",
            marker_color="#059669",
            text=[
                f"{r} ({pct:.0f}%)"
                for r, pct in zip(
                    arch_sorted["rescatados_v3"], arch_sorted["tasa_rescate_pct"]
                )
            ],
            textposition="inside",
            insidetextanchor="middle",
        ),
        row=1,
        col=2,
    )
    fig_rescue.add_trace(
        go.Bar(
            name="FN Remanentes",
            y=arch_sorted["refined_archetype"],
            x=arch_sorted["restantes_v3"],
            orientation="h",
            marker_color="#DC2626",
            text=[f"{rem}" for rem in arch_sorted["restantes_v3"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=2,
    )
    fig_rescue.update_yaxes(title_text="Recall (%)", range=[0, 118], row=1, col=1)
    fig_rescue.update_xaxes(
        title_text="Casos FN Originales (Rescatados vs. Remanentes)",
        range=[0, 165],
        row=1,
        col=2,
    )
    fig_rescue.update_yaxes(automargin=True, row=1, col=2)
    fig_rescue.update_layout(
        template="plotly_white",
        barmode="group",
        title={
            "text": "Impacto de la Destilación 3.0: Rescate del 'Paciente Cero' y Arquetipos Complejos",
            "x": 0.02,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.09,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"l": 60, "r": 70, "t": 110, "b": 60},
        height=580,
        width=1260,
    )
    fig_rescue.write_html(
        str(out_dir / "22_rescate_fn_destilacion_v3.html"), include_plotlyjs="cdn"
    )
