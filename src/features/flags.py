"""Extracción de indicadores binarios de riesgo y descalces contextuales."""

import polars as pl

HIGH_RISK_MCCS: tuple[int, ...] = (7995, 6051, 5815, 5732)


def add_risk_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula flags binarios de riesgo basados en reglas de negocio y hallazgos del EDA.

    - is_country_mismatch: 1 si el país del comercio difiere del país de la IP (canales digitales).
    - is_high_risk_mcc: 1 si el MCC pertenece a categorías críticas (apuestas, cripto, etc.).
    - is_swipe: 1 si el modo de entrada en POS fue banda magnética.
    - is_digital_channel: 1 si la transacción fue originada en ECOMMERCE o APP.
    - is_card_present: 1 si la transacción fue presencial (POS o ATM).
    - is_night_transaction: 1 si la transacción ocurrió entre la 01:00 y las 05:59 horas.
    """
    ts_col = pl.col("timestamp")
    hour_expr = ts_col.dt.hour()

    # Mismatch geográfico: solo aplica si ambos países son no nulos
    geo_mismatch_expr = (
        pl.when(
            pl.col("merchant_country").is_not_null()
            & pl.col("ip_country").is_not_null()
        )
        .then(pl.col("merchant_country") != pl.col("ip_country"))
        .otherwise(False)
        .cast(pl.Int8)
        .alias("is_country_mismatch")
    )

    high_risk_mcc_expr = (
        pl.col("mcc").is_in(HIGH_RISK_MCCS).cast(pl.Int8).alias("is_high_risk_mcc")
    )

    is_swipe_expr = (
        (pl.col("entry_mode") == "swipe")
        .fill_null(False)
        .cast(pl.Int8)
        .alias("is_swipe")
    )

    is_digital_expr = (
        pl.col("channel")
        .is_in(["ECOMMERCE", "APP"])
        .fill_null(False)
        .cast(pl.Int8)
        .alias("is_digital_channel")
    )

    is_card_present_expr = (
        pl.col("channel")
        .is_in(["POS", "ATM"])
        .fill_null(False)
        .cast(pl.Int8)
        .alias("is_card_present")
    )

    is_night_expr = (
        ((hour_expr >= 1) & (hour_expr <= 5))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("is_night_transaction")
    )

    return df.with_columns(
        [
            geo_mismatch_expr,
            high_risk_mcc_expr,
            is_swipe_expr,
            is_digital_expr,
            is_card_present_expr,
            is_night_expr,
        ]
    )


def add_impossible_geo_jump_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Detecta transacciones consecutivas de la misma tarjeta en distintos países con delta < 2h.

    Un desplazamiento internacional en menos de 2 horas (7.200 segundos) representa
    una imposibilidad física o una brecha de seguridad evidente (clonación simultánea).
    """
    sorted_df = df.sort("timestamp")
    prev_country = pl.col("merchant_country").shift(1).over("card_id")
    time_diff = pl.col("timestamp").diff().dt.total_seconds().over("card_id")

    is_jump = (
        prev_country.is_not_null()
        & pl.col("merchant_country").is_not_null()
        & (pl.col("merchant_country") != prev_country)
        & (time_diff > 0)
        & (time_diff < 7200)
    )

    return sorted_df.with_columns(
        is_jump.fill_null(False).cast(pl.Int8).alias("is_impossible_geo_jump")
    )
