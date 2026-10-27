"""Módulo de política de decisión tri-zona (Triage) con doble umbral y matriz de costos financieros."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna

from src.config import BUSINESS_MATRIX, BusinessCostMatrix
from src.models.threshold import find_optimal_business_threshold
from src.models.triage_plots import (
    plot_triage_surface_plotly,
    plot_triage_zones_plotly,
)

# Silenciar logs verbosos de optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TriagePolicyResult:
    """Resultado cuantitativo de la política de decisión tri-zona (Semáforo de Riesgo)."""

    t_low: float
    t_high: float
    # Zona Verde (Aprobación Directa)
    green_total: int
    green_vn: int
    green_fn: int
    # Zona Amarilla (Revisión de 2da Instancia / Fricción Comercial)
    yellow_total: int
    yellow_vp: int
    yellow_fp: int
    yellow_precision: float
    # Zona Roja (Bloqueo Directo por Alta Certeza)
    red_total: int
    red_vp: int
    red_fp: int
    red_precision: float
    # Consolidado Global
    total_vp: int
    total_fp: int
    total_fn: int
    total_utility: float
    savings_vs_inaction: float
    auto_block_ratio: float
    review_rate: float


def compute_triage_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    t_low: float,
    t_high: float,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
    cost_fp_yellow: float | None = None,
    cost_fp_red: float | None = None,
) -> TriagePolicyResult:
    """Evalúa la política de tres zonas utilizando los costos unitarios del emisor."""
    if t_low >= t_high:
        raise ValueError(
            f"t_low ({t_low:.4f}) debe ser estrictamente menor que t_high ({t_high:.4f})"
        )

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    # Máscaras booleanas de partición
    mask_green = p_arr < t_low
    mask_yellow = (p_arr >= t_low) & (p_arr < t_high)
    mask_red = p_arr >= t_high

    # Zona Verde: Aprobación Directa (Sin fricción)
    green_vn = int(np.sum((y_arr == 0) & mask_green))
    green_fn = int(np.sum((y_arr == 1) & mask_green))
    green_total = green_vn + green_fn

    # Zona Amarilla: Revisión Humana / Call Center
    yellow_vp = int(np.sum((y_arr == 1) & mask_yellow))
    yellow_fp = int(np.sum((y_arr == 0) & mask_yellow))
    yellow_total = yellow_vp + yellow_fp
    yellow_prec = yellow_vp / yellow_total if yellow_total > 0 else 0.0

    # Zona Roja: Bloqueo Automatizado Directo
    red_vp = int(np.sum((y_arr == 1) & mask_red))
    red_fp = int(np.sum((y_arr == 0) & mask_red))
    red_total = red_vp + red_fp
    red_prec = red_vp / red_total if red_total > 0 else 0.0

    total_vp = yellow_vp + red_vp
    total_fp = yellow_fp + red_fp
    total_fn = green_fn

    c_fp_y = cost_fp_yellow if cost_fp_yellow is not None else matrix.cost_fp
    c_fp_r = cost_fp_red if cost_fp_red is not None else matrix.cost_fp

    total_utility = (
        (matrix.profit_tp * total_vp)
        - (c_fp_y * yellow_fp)
        - (c_fp_r * red_fp)
        - (matrix.cost_fn * total_fn)
    )

    inaction_loss = -matrix.cost_fn * int(np.sum(y_arr))
    savings = total_utility - inaction_loss

    total_alerts = yellow_total + red_total
    auto_block_ratio = (red_total / total_alerts) if total_alerts > 0 else 0.0
    review_rate = yellow_total / len(y_arr) if len(y_arr) > 0 else 0.0

    return TriagePolicyResult(
        t_low=float(t_low),
        t_high=float(t_high),
        green_total=green_total,
        green_vn=green_vn,
        green_fn=green_fn,
        yellow_total=yellow_total,
        yellow_vp=yellow_vp,
        yellow_fp=yellow_fp,
        yellow_precision=yellow_prec,
        red_total=red_total,
        red_vp=red_vp,
        red_fp=red_fp,
        red_precision=red_prec,
        total_vp=total_vp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_utility=float(total_utility),
        savings_vs_inaction=float(savings),
        auto_block_ratio=auto_block_ratio,
        review_rate=review_rate,
    )


def find_optimal_dual_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision_high: float = 0.80,
    anchor_t_low: float | None = None,
    n_thresholds: int = 80,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
) -> tuple[TriagePolicyResult, dict[str, Any]]:
    """Encuentra conjuntamente el par (T_low*, T_high*) optimizando P&L y pureza en zona roja."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    if anchor_t_low is None:
        single_opt = find_optimal_business_threshold(
            y_arr, p_arr, steps=1000, matrix=matrix
        )
        t_low_target = single_opt.optimal_threshold
    else:
        t_low_target = float(anchor_t_low)

    thresholds = np.linspace(0.02, 0.95, n_thresholds)
    grid_utility = np.zeros((n_thresholds, n_thresholds))

    best_result: TriagePolicyResult | None = None
    best_score = -np.inf

    # Evaluación en malla 2D
    for i, t_l in enumerate(thresholds):
        for j, t_h in enumerate(thresholds):
            if t_l >= t_h:
                grid_utility[i, j] = np.nan
                continue

            res = compute_triage_metrics(y_arr, p_arr, t_l, t_h, matrix=matrix)
            grid_utility[i, j] = res.total_utility

            # Ponderar P&L y cumplimiento de pureza en zona roja
            if res.red_total > 0 and res.red_precision >= min_precision_high:
                # Priorizar soluciones alineadas con el óptimo financiero de t_low
                score = res.total_utility - (1500.0 * abs(t_l - t_low_target))
                if score > best_score:
                    best_score = score
                    best_result = res

    if best_result is None:
        # Fallback si ninguna celda alcanza la precisión mínima
        max_idx = np.nanargmax(grid_utility)
        valid_indices = np.where(~np.isnan(grid_utility))
        i_opt = valid_indices[0][max_idx]
        j_opt = valid_indices[1][max_idx]
        best_result = compute_triage_metrics(
            y_arr, p_arr, thresholds[i_opt], thresholds[j_opt], matrix=matrix
        )

    grid_data = {
        "thresholds": thresholds,
        "grid_utility": grid_utility,
    }

    return best_result, grid_data


def find_optimal_dual_thresholds_optuna(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision_high: float = 0.85,
    anchor_t_low: float | None = None,
    n_trials: int = 60,
    random_state: int = 42,
    matrix: BusinessCostMatrix = BUSINESS_MATRIX,
) -> TriagePolicyResult:
    """Optimización bayesiana continua con Optuna para el par (T_low, T_high)."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_proba, dtype=float)

    if anchor_t_low is None:
        single_opt = find_optimal_business_threshold(
            y_arr, p_arr, steps=1000, matrix=matrix
        )
        t_low_base = single_opt.optimal_threshold
    else:
        t_low_base = float(anchor_t_low)

    def objective(trial: optuna.Trial) -> float:
        # Explorar en torno al óptimo financiero para T_low
        t_l = trial.suggest_float(
            "t_low", max(0.01, t_low_base - 0.08), min(0.60, t_low_base + 0.08)
        )
        t_h = trial.suggest_float("t_high", t_l + 0.05, 0.95)

        res = compute_triage_metrics(y_arr, p_arr, t_l, t_h, matrix=matrix)

        # Penalización severa si no cumple la precisión requerida en Zona Roja
        penalty = 0.0
        if res.red_total == 0 or res.red_precision < min_precision_high:
            penalty = 50000.0 * (min_precision_high - res.red_precision)

        # Recompensa por tasa de auto-bloqueo
        bonus_autoblock = res.auto_block_ratio * 1500.0

        return float(res.total_utility - penalty + bonus_autoblock)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_tl = float(study.best_params["t_low"])
    best_th = float(study.best_params["t_high"])

    return compute_triage_metrics(y_arr, p_arr, best_tl, best_th, matrix=matrix)


__all__ = [
    "TriagePolicyResult",
    "compute_triage_metrics",
    "find_optimal_dual_thresholds",
    "find_optimal_dual_thresholds_optuna",
    "plot_triage_surface_plotly",
    "plot_triage_zones_plotly",
]
