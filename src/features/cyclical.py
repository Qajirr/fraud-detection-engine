"""Transformador cíclico para componentes temporales de transacciones."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    """Codificador trigonométrico (sin/cos) para componentes periódicos temporales."""

    def __init__(self, include_raw: bool = True):
        self.include_raw = include_raw

    def fit(self, X: pd.DataFrame | pd.Series | np.ndarray, y: Any = None):
        """Ajuste stateless compatible con scikit-learn."""
        return self

    def transform(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Calcula senos y cosenos para hora del día y día de la semana."""
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            col = X
        else:
            col = pd.Series(np.asarray(X).ravel())

        # Asegurar tipo datetime
        dt_series = pd.to_datetime(col)

        hours = dt_series.dt.hour.to_numpy(dtype=float)
        days = dt_series.dt.dayofweek.to_numpy(dtype=float)

        hour_rad = 2.0 * np.pi * hours / 24.0
        hour_sin = np.sin(hour_rad)[:, np.newaxis]
        hour_cos = np.cos(hour_rad)[:, np.newaxis]

        day_rad = 2.0 * np.pi * days / 7.0
        day_sin = np.sin(day_rad)[:, np.newaxis]
        day_cos = np.cos(day_rad)[:, np.newaxis]

        features = [hour_sin, hour_cos, day_sin, day_cos]

        if self.include_raw:
            is_weekend = (days >= 5).astype(float)[:, np.newaxis]
            hours_raw = hours[:, np.newaxis]
            days_raw = days[:, np.newaxis]
            features.extend([hours_raw, days_raw, is_weekend])

        return np.hstack(features)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Nombres de características generadas."""
        prefix = input_features[0] if input_features else "time"
        names = [
            f"{prefix}_hour_sin",
            f"{prefix}_hour_cos",
            f"{prefix}_day_sin",
            f"{prefix}_day_cos",
        ]
        if self.include_raw:
            names.extend(
                [f"{prefix}_hour_raw", f"{prefix}_day_raw", f"{prefix}_is_weekend"]
            )
        return names
