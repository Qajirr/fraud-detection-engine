"""Destilación 3.0 de variables causales para rescate de Falsos Negativos complejos."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from src.config import VAL_START_DT


def compute_distilled_v3_features(
    df: pl.DataFrame,
    val_start_dt: datetime = VAL_START_DT,
) -> pl.DataFrame:
    """Agrega las 14 señales causales de Destilación 3.0 sin fuga temporal.

    Familias incluidas:
    1. Familia A (Perfil Expandido y Shock de Despertar):
       - card_prior_tx_count
       - ratio_amount_card_expanding_mean
       - ratio_amount_card_prev_max
       - zscore_amount_segment
       - is_above_segment_p90
       - is_dormant_wakeup_shock
       - dormancy_shock_index
    2. Familia B (Novedad de Entidad e Interacción por Tarjeta/Cliente):
       - is_new_mcc_for_card
       - is_new_country_for_card
       - is_new_entry_mode_for_card
       - is_new_device_for_customer
       - is_new_ip_for_customer
    3. Familia C (Densidad y Riesgo Reciente del Comercio en 24h):
       - merchant_tx_count_24h
       - merchant_distinct_cards_24h
    """
    sorted_df = df.sort("timestamp")

    # 1. Historial causal acumulado de la tarjeta (expanding window estrictamente pasado)
    sorted_df = sorted_df.with_columns(
        [
            pl.col("amount")
            .cum_sum()
            .over("card_id")
            .sub(pl.col("amount"))
            .alias("_card_cum_amt_prev"),
            pl.int_range(0, pl.len()).over("card_id").alias("card_prior_tx_count"),
            pl.col("amount")
            .shift(1)
            .cum_max()
            .over("card_id")
            .fill_null(pl.col("amount"))
            .alias("_card_cum_max_prev"),
        ]
    )

    card_expanding_mean = (
        pl.when(pl.col("card_prior_tx_count") > 0)
        .then(pl.col("_card_cum_amt_prev") / pl.col("card_prior_tx_count"))
        .otherwise(pl.col("amount"))
    )

    sorted_df = sorted_df.with_columns(
        [
            (pl.col("amount") / (card_expanding_mean + 1.0))
            .fill_null(1.0)
            .alias("ratio_amount_card_expanding_mean"),
            (pl.col("amount") / (pl.col("_card_cum_max_prev") + 1.0))
            .fill_null(1.0)
            .alias("ratio_amount_card_prev_max"),
        ]
    ).drop(["_card_cum_amt_prev", "_card_cum_max_prev"])

    # 2. Z-score robusto de monto dentro de (mcc, channel, entry_mode) aprendido en Train
    train_sub = sorted_df.filter(pl.col("timestamp") < val_start_dt)
    if train_sub.is_empty():
        train_sub = sorted_df

    segment_stats = train_sub.group_by(["mcc", "channel", "entry_mode"]).agg(
        [
            pl.col("amount").median().alias("_seg_median"),
            (pl.col("amount").quantile(0.75) - pl.col("amount").quantile(0.25))
            .clip(lower_bound=5.0)
            .alias("_seg_iqr"),
            pl.col("amount").quantile(0.90).alias("_seg_p90"),
        ]
    )

    sorted_df = (
        sorted_df.join(segment_stats, on=["mcc", "channel", "entry_mode"], how="left")
        .with_columns(
            [
                (
                    (pl.col("amount") - pl.col("_seg_median").fill_null(50.0))
                    / pl.col("_seg_iqr").fill_null(30.0)
                )
                .clip(lower_bound=-3.0, upper_bound=30.0)
                .fill_null(0.0)
                .alias("zscore_amount_segment"),
                (pl.col("amount") > pl.col("_seg_p90").fill_null(200.0))
                .cast(pl.Int8)
                .fill_null(0)
                .alias("is_above_segment_p90"),
            ]
        )
        .drop(["_seg_median", "_seg_iqr", "_seg_p90"])
    )

    # 3. Novedad conductual de entidad (primera aparición histórica causal)
    sorted_df = sorted_df.with_columns(
        [
            pl.int_range(0, pl.len()).over(["card_id", "mcc"]).alias("_card_mcc_seq"),
            pl.int_range(0, pl.len())
            .over(["card_id", "merchant_country"])
            .alias("_card_country_seq"),
            pl.int_range(0, pl.len())
            .over(["card_id", "channel", "entry_mode"])
            .alias("_card_mode_seq"),
            pl.int_range(0, pl.len())
            .over(["customer_id", "device_id"])
            .alias("_cust_dev_seq"),
            pl.int_range(0, pl.len())
            .over(["customer_id", "ip_address"])
            .alias("_cust_ip_seq"),
        ]
    )

    sorted_df = sorted_df.with_columns(
        [
            ((pl.col("_card_mcc_seq") == 0) & (pl.col("card_prior_tx_count") >= 2))
            .cast(pl.Int8)
            .alias("is_new_mcc_for_card"),
            ((pl.col("_card_country_seq") == 0) & (pl.col("card_prior_tx_count") >= 1))
            .cast(pl.Int8)
            .alias("is_new_country_for_card"),
            ((pl.col("_card_mode_seq") == 0) & (pl.col("card_prior_tx_count") >= 2))
            .cast(pl.Int8)
            .alias("is_new_entry_mode_for_card"),
            (
                pl.col("device_id").is_not_null()
                & (pl.col("_cust_dev_seq") == 0)
                & (pl.col("card_prior_tx_count") >= 1)
            )
            .cast(pl.Int8)
            .alias("is_new_device_for_customer"),
            (
                pl.col("ip_address").is_not_null()
                & (pl.col("_cust_ip_seq") == 0)
                & (pl.col("card_prior_tx_count") >= 1)
            )
            .cast(pl.Int8)
            .alias("is_new_ip_for_customer"),
        ]
    ).drop(
        [
            "_card_mcc_seq",
            "_card_country_seq",
            "_card_mode_seq",
            "_cust_dev_seq",
            "_cust_ip_seq",
        ]
    )

    # 4. Densidad y actividad reciente en el comercio (merchant_id) en 24h
    roll_merch_24h = (
        sorted_df.rolling(
            index_column="timestamp",
            period="24h",
            closed="right",
            group_by="merchant_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                (pl.count("amount") - 1).alias("merchant_tx_count_24h"),
                pl.col("card_id").n_unique().alias("merchant_distinct_cards_24h"),
            ]
        )
        .select(
            ["transaction_id", "merchant_tx_count_24h", "merchant_distinct_cards_24h"]
        )
        .unique(subset=["transaction_id"])
    )

    sorted_df = sorted_df.join(
        roll_merch_24h, on="transaction_id", how="left"
    ).with_columns(
        [
            pl.col("merchant_tx_count_24h").fill_null(0).cast(pl.Int32),
            pl.col("merchant_distinct_cards_24h").fill_null(1).cast(pl.Int32),
        ]
    )

    # 5. Índice de despertar súbito ("Dormant Card Wakeup Shock")
    sorted_df = sorted_df.with_columns(
        [
            (
                (pl.col("delta_time_last_tx") > 86400 * 3)
                & (pl.col("ratio_amount_mcc_median") > 2.5)
            )
            .cast(pl.Int8)
            .alias("is_dormant_wakeup_shock"),
            (
                pl.col("delta_time_last_tx").clip(lower_bound=0.0).log1p()
                * pl.col("ratio_amount_mcc_median").clip(lower_bound=0.0).log1p()
            ).alias("dormancy_shock_index"),
        ]
    )

    return sorted_df
