"""Configuración global, constantes de negocio y rutas para el motor de detección de fraude transaccional."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Rutas del Proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "transacciones.csv"
PROCESSED_DATA_PATH = DATA_DIR / "transacciones_features.parquet"
PROCESSED_DATA_V2_PATH = DATA_DIR / "transacciones_features_v2.parquet"
PROCESSED_DATA_V3_PATH = DATA_DIR / "transacciones_features_v3.parquet"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"


@dataclass(frozen=True)
class BusinessCostMatrix:
    """Matriz de Costo / Beneficio Financiero para la evaluación de decisiones."""

    profit_tp: float = 120.0  # US$ 120 (fraude prevenido, neto de gestión)
    cost_fp: float = 30.0  # -US$ 30 (costo fricción de cliente, call center)
    cost_fn: float = 180.0  # -US$ 180 (pérdida directa por fraude no detectado)
    benefit_vn: float = 0.0  # US$ 0


BUSINESS_MATRIX = BusinessCostMatrix()

# Definición de Columnas
TARGET_COL = "is_fraud"
TIMESTAMP_COL = "timestamp"
AMOUNT_COL = "amount"

ID_COLS = [
    "transaction_id",
    "card_id",
    "customer_id",
    "merchant_id",
]

CATEGORICAL_COLS = [
    "mcc",
    "merchant_country",
    "channel",
    "entry_mode",
    "ip_country",
]

MCC_DESCRIPTIONS: dict[int, str] = {
    4111: "Transporte de Pasajeros",
    4722: "Agencias de Viajes",
    4814: "Telecomunicaciones",
    5311: "Tienda Departamental",
    5411: "Supermercado",
    5499: "Tienda de Alimentos",
    5541: "Estación de Servicio",
    5651: "Vestuario Familiar",
    5732: "Tiendas de Electrónica",
    5812: "Restaurantes",
    5815: "Bienes Digitales",
    5816: "Juegos Digitales",
    5912: "Farmacias",
    5942: "Librerías",
    5944: "Joyerías",
    5945: "Jugueterías",
    5947: "Regalos y Novedades",
    5964: "Venta Directa",
    5999: "Retail Misceláneo",
    6011: "Cajero Automático (ATM)",
    6051: "Cuasi-Efectivo / Cripto",
    7399: "Servicios a Empresas",
    7995: "Apuestas y Casinos",
}

DIGITAL_COLS = [
    "device_id",
    "ip_address",
]

# Configuración de División Temporal Out-of-Time (OOT)
TRAIN_START = "2026-03-01 00:00:00"
VAL_START = "2026-04-16 00:00:00"
TEST_OOT_START = "2026-05-01 00:00:00"
TEST_OOT_END = "2026-05-29 23:59:59"

TRAIN_START_DT = datetime(2026, 3, 1, 0, 0, 0)
VAL_START_DT = datetime(2026, 4, 16, 0, 0, 0)
TEST_OOT_START_DT = datetime(2026, 5, 1, 0, 0, 0)
TEST_OOT_END_DT = datetime(2026, 5, 29, 23, 59, 59)

# Semilla de Reproducibilidad
RANDOM_STATE = 42
