"""Constructor maestro del dataset enriquecido con variables causales de comportamiento."""

from pathlib import Path

import polars as pl

from src.config import (
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_V2_PATH,
    PROCESSED_DATA_V3_PATH,
    RAW_DATA_PATH,
)
from src.features.customer import compute_customer_rolling_features
from src.features.distillation import compute_distilled_v3_features
from src.features.flags import add_impossible_geo_jump_flag, add_risk_flags
from src.features.mcc_profile import add_mcc_amount_ratio
from src.features.network import compute_network_rolling_features
from src.features.velocity import (
    compute_card_rolling_features,
    compute_card_time_deltas,
    compute_digital_multiplicity,
)


def load_raw_transactions(csv_path: Path | str = RAW_DATA_PATH) -> pl.DataFrame:
    """Carga el archivo crudo de transacciones asegurando tipado correcto."""
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado en {path_obj}")

    return pl.read_csv(
        path_obj,
        try_parse_dates=True,
    ).with_columns(pl.col("timestamp").cast(pl.Datetime))


def build_feature_dataset(
    df: pl.DataFrame | None = None,
    raw_path: Path | str = RAW_DATA_PATH,
    save_path: Path | str | None = PROCESSED_DATA_PATH,
) -> pl.DataFrame:
    """Ejecuta el pipeline de Feature Engineering de extremo a extremo sin fuga temporal."""
    if df is None:
        df = load_raw_transactions(raw_path)

    base_df = df.with_columns(pl.col("timestamp").cast(pl.Datetime)).sort("timestamp")
    flagged_df = add_risk_flags(base_df)
    delta_df = compute_card_time_deltas(flagged_df)
    velocity_df = compute_card_rolling_features(delta_df)
    final_df = (
        compute_digital_multiplicity(velocity_df)
        .unique(subset=["transaction_id"], maintain_order=True)
        .sort("timestamp")
    )

    if save_path is not None:
        save_obj = Path(save_path)
        save_obj.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(save_obj)

    return final_df


def build_feature_dataset_v2(
    df: pl.DataFrame | None = None,
    raw_path: Path | str = RAW_DATA_PATH,
    save_path: Path | str | None = PROCESSED_DATA_V2_PATH,
) -> pl.DataFrame:
    """Ejecuta el pipeline ampliado de Feature Engineering 2.0 sin fuga temporal."""
    if df is None:
        df = load_raw_transactions(raw_path)

    base_df = df.with_columns(pl.col("timestamp").cast(pl.Datetime)).sort("timestamp")
    flagged_df = add_risk_flags(base_df)
    geo_df = add_impossible_geo_jump_flag(flagged_df)
    delta_df = compute_card_time_deltas(geo_df)
    velocity_df = compute_card_rolling_features(delta_df)
    digital_df = compute_digital_multiplicity(velocity_df)
    customer_df = compute_customer_rolling_features(digital_df)
    mcc_df = add_mcc_amount_ratio(customer_df)
    final_df = (
        compute_network_rolling_features(mcc_df)
        .unique(subset=["transaction_id"], maintain_order=True)
        .sort("timestamp")
    )

    if save_path is not None:
        save_obj = Path(save_path)
        save_obj.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(save_obj)

    return final_df


def build_feature_dataset_v3(
    df: pl.DataFrame | None = None,
    raw_path: Path | str = RAW_DATA_PATH,
    save_path: Path | str | None = PROCESSED_DATA_V3_PATH,
) -> pl.DataFrame:
    """Ejecuta el pipeline de Destilación 3.0 para rescate de Falsos Negativos complejos."""
    v2_df = build_feature_dataset_v2(df=df, raw_path=raw_path, save_path=None)
    final_df = (
        compute_distilled_v3_features(v2_df)
        .unique(subset=["transaction_id"], maintain_order=True)
        .sort("timestamp")
    )

    if save_path is not None:
        save_obj = Path(save_path)
        save_obj.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(save_obj)

    return final_df
