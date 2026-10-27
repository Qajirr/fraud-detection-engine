"""Constructor del pipeline de preprocesamiento scikit-learn sin fuga temporal."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
    TargetEncoder,
)

from src.features.cyclical import CyclicalTimeTransformer

NUMERICAL_FEATURES: list[str] = [
    "amount",
    "delta_time_last_tx",
    "card_tx_count_1h",
    "card_tx_count_24h",
    "card_tx_count_7d",
    "card_amount_sum_1h",
    "card_amount_sum_24h",
    "card_amount_sum_7d",
    "ratio_amount_mean_7d",
    "cards_seen_by_device_24h",
    "cards_seen_by_ip_24h",
]

NUMERICAL_FEATURES_V2: list[str] = NUMERICAL_FEATURES + [
    "card_tx_count_6h",
    "card_amount_sum_6h",
    "burst_ratio_1h_24h",
    "customer_tx_count_1h",
    "customer_tx_count_24h",
    "customer_amount_sum_24h",
    "customer_distinct_cards_24h",
    "ratio_amount_mcc_median",
    "card_distinct_merchants_24h",
    "card_distinct_countries_24h",
]

NUMERICAL_FEATURES_V3: list[str] = NUMERICAL_FEATURES_V2 + [
    "card_prior_tx_count",
    "ratio_amount_card_expanding_mean",
    "ratio_amount_card_prev_max",
    "zscore_amount_segment",
    "merchant_tx_count_24h",
    "merchant_distinct_cards_24h",
    "dormancy_shock_index",
]

BINARY_FEATURES: list[str] = [
    "is_country_mismatch",
    "is_high_risk_mcc",
    "is_swipe",
    "is_digital_channel",
    "is_card_present",
    "is_night_transaction",
    "is_first_card_tx",
]

BINARY_FEATURES_V2: list[str] = BINARY_FEATURES + [
    "is_impossible_geo_jump",
]

BINARY_FEATURES_V3: list[str] = BINARY_FEATURES_V2 + [
    "is_above_segment_p90",
    "is_new_mcc_for_card",
    "is_new_country_for_card",
    "is_new_entry_mode_for_card",
    "is_new_device_for_customer",
    "is_new_ip_for_customer",
    "is_dormant_wakeup_shock",
]

ONEHOT_FEATURES: list[str] = [
    "channel",
    "entry_mode",
]

TARGET_ENCODE_FEATURES: list[str] = [
    "mcc",
]

CYCLICAL_FEATURE: str = "timestamp"

ALL_MODEL_FEATURES: list[str] = (
    NUMERICAL_FEATURES
    + BINARY_FEATURES
    + ONEHOT_FEATURES
    + TARGET_ENCODE_FEATURES
    + [CYCLICAL_FEATURE]
)

ALL_MODEL_FEATURES_V2: list[str] = (
    NUMERICAL_FEATURES_V2
    + BINARY_FEATURES_V2
    + ONEHOT_FEATURES
    + TARGET_ENCODE_FEATURES
    + [CYCLICAL_FEATURE]
)

ALL_MODEL_FEATURES_V3: list[str] = (
    NUMERICAL_FEATURES_V3
    + BINARY_FEATURES_V3
    + ONEHOT_FEATURES
    + TARGET_ENCODE_FEATURES
    + [CYCLICAL_FEATURE]
)


def log1p_transform(x: np.ndarray) -> np.ndarray:
    """Transformación logarítmica con recorte de valores no negativos."""
    return np.log1p(np.maximum(x, 0.0))


def create_preprocessor(
    cv_target_encoder: int = 5,
    random_state: int = 42,
    feature_version: str = "v1",
) -> ColumnTransformer:
    """Construye el ColumnTransformer scikit-learn garantizando aislamiento temporal."""
    if feature_version == "v3":
        num_features = NUMERICAL_FEATURES_V3
        bin_features = BINARY_FEATURES_V3
    elif feature_version == "v2":
        num_features = NUMERICAL_FEATURES_V2
        bin_features = BINARY_FEATURES_V2
    else:
        num_features = NUMERICAL_FEATURES
        bin_features = BINARY_FEATURES

    num_pipeline = Pipeline(
        [
            (
                "log1p",
                FunctionTransformer(log1p_transform, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    onehot_pipeline = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            )
        ]
    )

    target_encoder = TargetEncoder(
        smooth="auto",
        cv=KFold(n_splits=cv_target_encoder, shuffle=True, random_state=random_state),
    )

    cyclical_transformer = CyclicalTimeTransformer(include_raw=False)

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("bin", "passthrough", bin_features),
            ("ohe", onehot_pipeline, ONEHOT_FEATURES),
            ("target_enc", target_encoder, TARGET_ENCODE_FEATURES),
            ("cyclical", cyclical_transformer, [CYCLICAL_FEATURE]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
