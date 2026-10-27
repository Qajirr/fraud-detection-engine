"""Cálculo causal de métricas de comportamiento a nivel cliente (customer_id)."""

import polars as pl


def compute_customer_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula agregaciones retrospectivas por cliente sin fuga temporal.

    - customer_tx_count_1h: conteo de transacciones del cliente en la última hora.
    - customer_tx_count_24h: conteo de transacciones del cliente en las últimas 24 horas.
    - customer_amount_sum_24h: monto acumulado por el cliente en las últimas 24 horas.
    - customer_distinct_cards_24h: plásticos distintos operados por el cliente en 24h.
    """
    sorted_df = df.sort("timestamp")

    # Ventana 1 hora por cliente
    roll_1h = (
        sorted_df.rolling(
            index_column="timestamp",
            period="1h",
            closed="right",
            group_by="customer_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.count("amount").alias("_cust_cnt_1h_inc"),
            ]
        )
        .select(
            [
                "transaction_id",
                (pl.col("_cust_cnt_1h_inc") - 1).alias("customer_tx_count_1h"),
            ]
        )
        .unique(subset=["transaction_id"])
    )

    # Ventana 24 horas por cliente
    roll_24h = (
        sorted_df.rolling(
            index_column="timestamp",
            period="24h",
            closed="right",
            group_by="customer_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.count("amount").alias("_cust_cnt_24h_inc"),
                pl.sum("amount").alias("_cust_sum_24h_inc"),
                pl.col("card_id").n_unique().alias("customer_distinct_cards_24h"),
            ]
        )
        .select(
            [
                "transaction_id",
                (pl.col("_cust_cnt_24h_inc") - 1).alias("customer_tx_count_24h"),
                pl.col("_cust_sum_24h_inc"),
                pl.col("customer_distinct_cards_24h"),
            ]
        )
        .unique(subset=["transaction_id"])
    )

    enriched = (
        df.join(roll_1h, on="transaction_id", how="left")
        .join(roll_24h, on="transaction_id", how="left")
        .with_columns(
            [
                pl.col("customer_tx_count_1h").fill_null(0).cast(pl.Int32),
                pl.col("customer_tx_count_24h").fill_null(0).cast(pl.Int32),
                (
                    pl.col("_cust_sum_24h_inc").fill_null(pl.col("amount"))
                    - pl.col("amount")
                )
                .clip(lower_bound=0.0)
                .alias("customer_amount_sum_24h"),
                pl.col("customer_distinct_cards_24h").fill_null(1).cast(pl.Int32),
            ]
        )
        .drop(["_cust_sum_24h_inc"])
    )

    return enriched
