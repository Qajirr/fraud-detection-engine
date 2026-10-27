# Presentación Ejecutiva (4 Diapositivas): Detección de Fraude Transaccional & Optimización Financiera

**Contexto:** Prevención de Fraude con Tarjeta & Modelos de Riesgo Financiero | **Caso de Estudio Institucional**  
**Entregables:** [notebooks/eda_y_modelado_fraude.ipynb](file:///C:/Users/vicen/Documents/test-bci/notebooks/eda_y_modelado_fraude.ipynb) | [presentacion_4_diapositivas.html](file:///C:/Users/vicen/Documents/test-bci/presentation/presentacion_4_diapositivas.html) | [CASE_STUDY.md](file:///C:/Users/vicen/Documents/test-bci/CASE_STUDY.md)

---

## 📊 Diapositiva 1: Principales Insights de los Datos (EDA Out-of-Time)

**KPIs Base (`Train + Val`: 674.696 tx | Test Mayo ciego):** Prevalencia **0,311%** (`3.113` fraudes en 1M tx) · Ticket Mediano **2,58x** (`US$ 93` vs `US$ 36`) · Nulos **59% Físicos** (`100%` en POS/ATM, `0%` en Digital).

- **Rubros Críticos (`MCC`):** Liquidación inmediata concentra el fraude: **Cripto/Cuasi-Efectivo `6051` (54,0%)**, **Apuestas `7995` (29,9%)** y **Joyerías `5944` (21,6%)**.
- **Vulnerabilidad Física y Horaria:** Banda magnética (`swipe`, **2,23%**) multiplica **31x** el riesgo vs. `chip` (**0,07%**); pico relativo en madrugada (**01:00–05:59 hrs**).
- **Geografía y Grafos (`NetworkX`):** `merchant_country != ip_country` eleva el fraude **8,1x** (`0,81%` vs `0,10%`); **251 anillos de colusión** multitarjeta detectados.

> 📈 **Gráfico en Diapositiva 1:** Tasa de Fraude por Rubro Crítico (`MCC`) y Modo de Captura (`Swipe` vs `Chip`).

---

## 🛠️ Diapositiva 2: Feature Engineering y Destilación Causal 3.0 (`Polars`, 0% Leakage)

**Diagnóstico de Falsos Negativos:** El **80,3% de los FN en V1 (`313/390`)** ocurría en el **1er fraude de la tarjeta ("Paciente Cero")** tras ~6 días de inactividad (`card_tx_count_1h = 0`).

- **V1–V2 (Velocidad y Grafos, 34 vars):** Ráfagas retrospectivas (`1h, 6h, 24h, 7d`), salto geográfico imposible (`< 2h`) y rotación de tarjetas por `device_id` / `ip_address` (24h).
- **Destilación V3 (`+13 vars` $\rightarrow$ 47 vars totales para capturar el "Paciente Cero"):**
  - **Novedad (*First-Time Seen*):** `is_new_device_for_customer` (**48,2% en FN vs. 4,2% en legítimas $\rightarrow$ 11,5x**), `is_new_country_for_card` (**30,3% en FN**) e `is_new_mcc_for_card`.
  - **Shock sobre Historia Total y Comercio (24h):** `ratio_amount_card_prev_max` (**2,52x**), `zscore_amount_segment`, `dormancy_shock_index` y `merchant_distinct_cards_24h`.

> 📈 **Gráfico en Diapositiva 2:** Salto en Tasa de Captura (Recall) según Posición en la Ráfaga (`Fraude #1 Paciente Cero`: **28,9% en V1 $\rightarrow$ 59,2% en V3**; `Fraude #2`: **84,4%**; `Fraude #3+`: **98,5%**).

---

## ⚙️ Diapositiva 3: Metodología Utilizada

$$\text{Ganancia Neta} = (+120 \times \text{VP}) - (30 \times \text{FP}) - (180 \times \text{FN}) \quad \Rightarrow \quad \text{Asimetría } 6:1 \text{ (FN vs FP)}$$

- **1. Split Cronológico OOT:** `Train` (Mar–15 Abr: `507k` tx) $\rightarrow$ `Val OOT` (16–30 Abr: `167k` tx) $\rightarrow$ `Test OOT` (Mayo: `325k` tx).
- **2. Pipeline y Calibración Isotónica:** `CyclicalTime` (`sin/cos`) + `TargetEncoder` OOF (`cv=5`) + `RobustScaler`. Calibración isotónica logra **Brier `0,00150`** (`ECE = 0,0000`).
- **3. Optimización de Umbral ($T^*$) y Triage:** Corte óptimo **$T^* = 0,128$** (maximiza dólares en Val OOT) + **Triage Tri-Zona** + **Bloqueo de Plástico** tras 1er VP + **TreeSHAP**.

> 📈 **Gráfico en Diapositiva 3:** Evolución de **PR-AUC** en Validación OOT y Test OOT (`LGBM Base 0,459` $\rightarrow$ `LGBM Tuned 0,559` $\rightarrow$ `XGBoost V1 0,614` $\rightarrow$ **`XGBoost V3 Champion 0,785`**).

---

## 🏆 Diapositiva 4: Resultados e Impacto Económico (Test Out-of-Time: Mayo 2026)

**KPIs Campeón V3 (`325.304` tx | `945` fraudes):** Utilidad Neta **+US$ 34.830** (**+US$ 39.030** con bloqueo de plástico) · Ahorro vs. Inacción **+US$ 204.930** · Recall **76,7%** (`725/945` VP) · **-45,1% Falsas Alarmas** (`419` vs `763` FP).

| Estrategia en Test OOT (Mayo 2026) | PR-AUC | Corte ($T^*$) | Fraudes Detectados (VP) | Falsas Alarmas (FP) | Utilidad Neta (US$) | Ahorro vs. Inacción |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Sin Modelo (Inacción)** | — | — | 0 (0,0%) | 0 | **-US$ 170.100** | US$ 0 |
| **1. `XGBoost V1` (Default `0.50`)** | 0,653 | `0,500` | 463 (49,0%) | 226 (67% Prec) | **-US$ 37.980** | +US$ 132.120 |
| **2. `XGBoost V1` (Umbral $T^*$)** | 0,653 | `0,268` | 555 (58,7%) | 763 (42% Prec) | **-US$ 26.490** | +US$ 143.610 |
| **3. `champion_model_v3` (Destilación 3.0)** | **0,785** | **`0,128`** | **725 (76,7%)** | **419 (63% Prec)** | **+US$ 34.830** | **+US$ 204.930** |
| **4. `champion_model_v3` + Bloqueo Plástico** | **0,785** | **`0,128`** | **739 (78,2%)** | **419 (64% Prec)** | **+US$ 39.030** | **+US$ 209.130** |

> 📈 **Gráfico en Diapositiva 4:** Cascada de Utilidad Financiera Neta en Test OOT (De **-US$ 170.100** por inacción y **-US$ 26.490** en V1 a **+US$ 34.830 / +US$ 39.030** en V3).
