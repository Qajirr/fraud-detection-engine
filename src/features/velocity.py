"""Cálculo causal de velocidad transaccional y multiplicidad digital sin fuga temporal."""

import polars as pl


def compute_card_time_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula el tiempo en segundos desde la última transacción de la misma tarjeta."""
    return df.with_columns(
        [
            pl.col("timestamp")
            .diff()
            .dt.total_seconds()
            .over("card_id")
            .fill_null(-1.0)
            .alias("delta_time_last_tx"),
            pl.col("timestamp")
            .diff()
            .is_null()
            .over("card_id")
            .cast(pl.Int8)
            .alias("is_first_card_tx"),
        ]
    )


def compute_card_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula conteos y montos acumulados por tarjeta en ventanas móviles causales.

    Ventanas incluidas: 1h, 6h, 24h, 7d.
    Utiliza closed='right' y ajusta restando la transacción actual para garantizar
    que solo se utilice información estrictamente pasada (equivalente a closed='left'
    pero preservando el identificador exacto de la transacción).
    """
    sorted_df = df.sort("timestamp")

    # Ventana 1 hora
    roll_1h = (
        sorted_df.rolling(
            index_column="timestamp",
            period="1h",
            closed="right",
            group_by="card_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.count("amount").alias("_cnt_1h_inc"),
                pl.sum("amount").alias("_sum_1h_inc"),
            ]
        )
        .select(
            [
                "transaction_id",
                (pl.col("_cnt_1h_inc") - 1).alias("card_tx_count_1h"),
                pl.col("_sum_1h_inc"),
            ]
        )
    )

    # Ventana 6 horas
    roll_6h = (
        sorted_df.rolling(
            index_column="timestamp",
            period="6h",
            closed="right",
            group_by="card_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.count("amount").alias("_cnt_6h_inc"),
                pl.sum("amount").alias("_sum_6h_inc"),
            ]
        )
        .select(
            [
                "transaction_id",
                (pl.col("_cnt_6h_inc") - 1).alias("card_tx_count_6h"),
                pl.col("_sum_6h_inc"),
            ]
        )
    )

    # Ventana 24 horas
    roll_24h = (
        sorted_df.rolling(
            index_column="timestamp",
            period="24h",
            closed="right",
            group_by="card_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.count("amount").alias("_cnt_24h_inc"),
                pl.sum("amount").alias("_sum_24h_inc"),
            ]
        )
        .select(
            [
                "transaction_id",
                (pl.col("_cnt_24h_inc") - 1).alias("card_tx_count_24h"),
                pl.col("_sum_24h_inc"),
            ]
        )
    )

    # Ventana 7 días
    roll_7d = (
        sorted_df.rolling(
            index_column="timestamp",
            period="7d",
            closed="right",
            group_by="card_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.count("amount").alias("_cnt_7d_inc"),
                pl.sum("amount").alias("_sum_7d_inc"),
            ]
        )
        .select(
            [
                "transaction_id",
                (pl.col("_cnt_7d_inc") - 1).alias("card_tx_count_7d"),
                pl.col("_sum_7d_inc"),
            ]
        )
    )

    # Unir métricas con la tabla base y ajustar sumas restando el monto actual
    enriched = (
        df.join(roll_1h, on="transaction_id", how="left")
        .join(roll_6h, on="transaction_id", how="left")
        .join(roll_24h, on="transaction_id", how="left")
        .join(roll_7d, on="transaction_id", how="left")
        .with_columns(
            [
                pl.col("card_tx_count_1h").fill_null(0).cast(pl.Int32),
                pl.col("card_tx_count_6h").fill_null(0).cast(pl.Int32),
                pl.col("card_tx_count_24h").fill_null(0).cast(pl.Int32),
                pl.col("card_tx_count_7d").fill_null(0).cast(pl.Int32),
                (pl.col("_sum_1h_inc").fill_null(pl.col("amount")) - pl.col("amount"))
                .clip(lower_bound=0.0)
                .alias("card_amount_sum_1h"),
                (pl.col("_sum_6h_inc").fill_null(pl.col("amount")) - pl.col("amount"))
                .clip(lower_bound=0.0)
                .alias("card_amount_sum_6h"),
                (pl.col("_sum_24h_inc").fill_null(pl.col("amount")) - pl.col("amount"))
                .clip(lower_bound=0.0)
                .alias("card_amount_sum_24h"),
                (pl.col("_sum_7d_inc").fill_null(pl.col("amount")) - pl.col("amount"))
                .clip(lower_bound=0.0)
                .alias("card_amount_sum_7d"),
            ]
        )
        .drop(["_sum_1h_inc", "_sum_6h_inc", "_sum_24h_inc", "_sum_7d_inc"])
    )

    # Ratio de monto actual vs promedio histórico reciente de 7 días
    hist_mean_7d = (
        pl.when(pl.col("card_tx_count_7d") > 0)
        .then(pl.col("card_amount_sum_7d") / pl.col("card_tx_count_7d"))
        .otherwise(pl.col("amount"))
    )

    burst_expr = (
        pl.col("card_tx_count_1h") / ((pl.col("card_tx_count_24h") / 24.0) + 0.1)
    ).alias("burst_ratio_1h_24h")

    return enriched.with_columns(
        [
            (pl.col("amount") / (hist_mean_7d + 1e-3))
            .fill_null(1.0)
            .alias("ratio_amount_mean_7d"),
            burst_expr,
        ]
    )


def compute_digital_multiplicity(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula la multiplicidad de tarjetas asociadas a un device_id e ip_address en 24h."""
    sorted_df = df.sort("timestamp")

    # Dispositivos digitales no nulos
    dev_df = sorted_df.filter(pl.col("device_id").is_not_null())
    dev_multiplicity = (
        dev_df.rolling(
            index_column="timestamp",
            period="24h",
            closed="right",
            group_by="device_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.col("card_id").n_unique().alias("cards_seen_by_device_24h"),
            ]
        )
        .select(["transaction_id", "cards_seen_by_device_24h"])
    )

    # Direcciones IP no nulas
    ip_df = sorted_df.filter(pl.col("ip_address").is_not_null())
    ip_multiplicity = (
        ip_df.rolling(
            index_column="timestamp",
            period="24h",
            closed="right",
            group_by="ip_address",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.col("card_id").n_unique().alias("cards_seen_by_ip_24h"),
            ]
        )
        .select(["transaction_id", "cards_seen_by_ip_24h"])
    )

    return (
        df.join(dev_multiplicity, on="transaction_id", how="left")
        .join(ip_multiplicity, on="transaction_id", how="left")
        .with_columns(
            [
                pl.col("cards_seen_by_device_24h").fill_null(0).cast(pl.Int32),
                pl.col("cards_seen_by_ip_24h").fill_null(0).cast(pl.Int32),
            ]
        )
    )
