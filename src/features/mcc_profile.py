"""Perfilado de montos por rubro comercial (MCC) sin fuga temporal."""

import polars as pl


def compute_mcc_amount_medians(
    df: pl.DataFrame,
    val_start: str = "2026-04-16 00:00:00",
) -> pl.DataFrame:
    """Calcula la mediana de monto por MCC utilizando únicamente la partición de entrenamiento.

    Evita fuga temporal al restringir el cálculo al pasado histórico (Train).
    """
    train_df = df.filter(pl.col("timestamp") < pl.lit(val_start).str.to_datetime())
    if train_df.is_empty():
        train_df = df

    medians = train_df.group_by("mcc").agg(
        pl.col("amount").median().alias("mcc_median_amount")
    )
    return medians


def add_mcc_amount_ratio(
    df: pl.DataFrame,
    mcc_medians: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Calcula el ratio entre el monto de la transacción y la mediana histórica del rubro.

    Un ratio alto (ej. > 5.0) señala compras anómalamente grandes para ese tipo de comercio
    (ej. $1.000 en un restaurante o estación de servicio).
    """
    if mcc_medians is None:
        mcc_medians = compute_mcc_amount_medians(df)

    global_median = float(df["amount"].median() or 50.0)

    enriched = (
        df.join(mcc_medians, on="mcc", how="left")
        .with_columns(pl.col("mcc_median_amount").fill_null(global_median))
        .with_columns(
            (pl.col("amount") / (pl.col("mcc_median_amount") + 1e-3))
            .fill_null(1.0)
            .alias("ratio_amount_mcc_median")
        )
        .drop(["mcc_median_amount"])
    )
    return enriched
