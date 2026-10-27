"""Pruebas unitarias para el transformador cíclico temporal."""

import numpy as np
import pandas as pd

from src.features.cyclical import CyclicalTimeTransformer


def test_cyclical_time_transformer_shapes():
    """Verifica las dimensiones de salida de la transformación cíclica."""
    timestamps = pd.date_range("2026-03-01", periods=100, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    # Con variables crudas
    transformer = CyclicalTimeTransformer(include_raw=True)
    out = transformer.fit_transform(df)
    # Debería contener: hour_sin, hour_cos, day_sin, day_cos, hour_raw, day_raw, is_weekend = 7 columnas
    assert out.shape == (100, 7)

    names = transformer.get_feature_names_out(["timestamp"])
    assert len(names) == 7
    assert "timestamp_hour_sin" in names
    assert "timestamp_day_cos" in names
    assert "timestamp_is_weekend" in names


def test_cyclical_time_transformer_ranges():
    """Verifica que los valores de seno y coseno estén acotados en [-1, 1]."""
    timestamps = pd.date_range("2026-03-01", periods=48, freq="h")
    transformer = CyclicalTimeTransformer(include_raw=False)
    out = transformer.fit_transform(timestamps)

    # 4 columnas: hour_sin, hour_cos, day_sin, day_cos
    assert out.shape == (48, 4)
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)
