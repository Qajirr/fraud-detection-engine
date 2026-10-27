"""Módulo de modelamiento, entrenamiento y evaluación financiera."""

from src.models.attention import (
    FocalLoss,
    TabularAttentionClassifier,
    TabularAttentionNet,
)
from src.models.calibration import (
    CalibrationMetrics,
    calibrate_fraud_model,
    evaluate_probability_calibration,
    plot_reliability_curve_plotly,
)
from src.models.evaluation import (
    compute_detailed_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.models.explainability import (
    compute_tree_shap_values,
    plot_attention_heatmap_plotly,
    plot_fraud_network_plotly,
    plot_shap_dependence_plotly,
    plot_shap_summary_plotly,
    plot_shap_waterfall_plotly,
    save_figure_html,
)
from src.models.threshold import (
    BusinessThresholdResult,
    compute_financial_utility,
    compute_vectorized_threshold_metrics,
    find_optimal_business_threshold,
    plot_business_utility_tradeoff,
)
from src.models.trainer import (
    ModelEvaluationResult,
    create_candidate_classifiers,
    train_and_evaluate_model,
)
from src.models.triage import (
    TriagePolicyResult,
    compute_triage_metrics,
    find_optimal_dual_thresholds,
    find_optimal_dual_thresholds_optuna,
    plot_triage_surface_plotly,
    plot_triage_zones_plotly,
)
from src.models.tuning import (
    TuningResult,
    tune_attention_classifier,
    tune_lightgbm,
    tune_xgboost,
)

__all__ = [
    "BusinessThresholdResult",
    "CalibrationMetrics",
    "FocalLoss",
    "ModelEvaluationResult",
    "TabularAttentionClassifier",
    "TabularAttentionNet",
    "TriagePolicyResult",
    "TuningResult",
    "calibrate_fraud_model",
    "compute_detailed_metrics",
    "compute_financial_utility",
    "compute_tree_shap_values",
    "compute_triage_metrics",
    "compute_vectorized_threshold_metrics",
    "create_candidate_classifiers",
    "evaluate_probability_calibration",
    "find_optimal_business_threshold",
    "find_optimal_dual_thresholds",
    "find_optimal_dual_thresholds_optuna",
    "plot_attention_heatmap_plotly",
    "plot_business_utility_tradeoff",
    "plot_confusion_matrix",
    "plot_fraud_network_plotly",
    "plot_precision_recall_curves",
    "plot_reliability_curve_plotly",
    "plot_roc_curves",
    "plot_shap_dependence_plotly",
    "plot_shap_summary_plotly",
    "plot_shap_waterfall_plotly",
    "plot_triage_surface_plotly",
    "plot_triage_zones_plotly",
    "save_figure_html",
    "train_and_evaluate_model",
    "tune_attention_classifier",
    "tune_lightgbm",
    "tune_xgboost",
]
