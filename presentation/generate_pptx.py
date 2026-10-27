"""Generate a concise, visual 4-slide executive PowerPoint (.pptx) with native editable charts."""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(15, 23, 42)
BLUE_DARK = RGBColor(30, 58, 138)
BLUE_ACCENT = RGBColor(37, 99, 235)
BLUE_LIGHT = RGBColor(147, 197, 253)
SLATE = RGBColor(51, 65, 85)
MUTED = RGBColor(100, 116, 139)
LIGHT_BG = RGBColor(248, 250, 252)
BORDER_GRAY = RGBColor(203, 213, 225)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(5, 150, 105)
GREEN_SOFT = RGBColor(236, 253, 245)
RED = RGBColor(220, 38, 38)
AMBER = RGBColor(217, 119, 6)


def add_header_and_footer(
    slide, title_text: str, subtitle_text: str, slide_num: int
) -> None:
    """Render clean header and footer."""
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.95)
    )
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()

    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0.91), Inches(13.333), Inches(0.05)
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = BLUE_ACCENT
    accent.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(0.45), Inches(0.10), Inches(10.6), Inches(0.78)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = subtitle_text
    p2.font.size = Pt(11)
    p2.font.color.rgb = RGBColor(191, 219, 254)
    p2.font.name = "Calibri"

    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(11.45),
        Inches(0.22),
        Inches(1.45),
        Inches(0.45),
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = BLUE_DARK
    badge.line.color.rgb = BLUE_ACCENT
    bp = badge.text_frame.paragraphs[0]
    bp.text = f"Slide {slide_num} / 4"
    bp.alignment = PP_ALIGN.CENTER
    bp.font.size = Pt(11)
    bp.font.bold = True
    bp.font.color.rgb = WHITE
    bp.font.name = "Calibri"

    footer = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.15), Inches(13.333), Inches(0.35)
    )
    footer.fill.solid()
    footer.fill.fore_color.rgb = RGBColor(241, 245, 249)
    footer.line.color.rgb = BORDER_GRAY

    ftb = slide.shapes.add_textbox(
        Inches(0.45), Inches(7.17), Inches(12.4), Inches(0.28)
    )
    fp = ftb.text_frame.paragraphs[0]
    fp.text = "Fintech Card Fraud Detection & Financial Optimization Engine | Portfolio Case Study"
    fp.font.size = Pt(9.5)
    fp.font.color.rgb = MUTED
    fp.font.name = "Calibri"


def add_kpi_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    label: str,
    value: str,
    sub: str,
    accent_color: RGBColor,
) -> None:
    """Render a compact executive KPI card."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = BORDER_GRAY

    strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.09), Inches(height)
    )
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent_color
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(left + 0.16),
        Inches(top + 0.05),
        Inches(width - 0.22),
        Inches(height - 0.08),
    )
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = label.upper()
    p1.font.size = Pt(9)
    p1.font.bold = True
    p1.font.color.rgb = MUTED
    p1.font.name = "Calibri"

    p2 = tf.add_paragraph()
    p2.text = value
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.font.name = "Calibri"

    p3 = tf.add_paragraph()
    p3.text = sub
    p3.font.size = Pt(9.5)
    p3.font.color.rgb = SLATE
    p3.font.name = "Calibri"


def add_concise_panel(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    heading: str,
    bullets: list[tuple[str, str]],
) -> None:
    """Render a concise bullet card."""
    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = WHITE
    panel.line.color.rgb = BORDER_GRAY

    tb = slide.shapes.add_textbox(
        Inches(left + 0.18),
        Inches(top + 0.12),
        Inches(width - 0.32),
        Inches(height - 0.2),
    )
    tf = tb.text_frame
    tf.word_wrap = True

    hp = tf.paragraphs[0]
    hp.text = heading
    hp.font.size = Pt(13.5)
    hp.font.bold = True
    hp.font.color.rgb = BLUE_DARK
    hp.font.name = "Calibri"
    hp.space_after = Pt(6)

    for prefix, body in bullets:
        bp = tf.add_paragraph()
        bp.space_after = Pt(6)
        r1 = bp.add_run()
        r1.text = f"▪ {prefix}: "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = NAVY
        r1.font.name = "Calibri"

        r2 = bp.add_run()
        r2.text = body
        r2.font.size = Pt(11)
        r2.font.color.rgb = SLATE
        r2.font.name = "Calibri"


def add_chart_card(
    slide, left: float, top: float, width: float, height: float, title: str
):
    """Draw a container panel and title for a native chart."""
    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = WHITE
    panel.line.color.rgb = BORDER_GRAY

    tb = slide.shapes.add_textbox(
        Inches(left + 0.18), Inches(top + 0.10), Inches(width - 0.36), Inches(0.4)
    )
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = BLUE_DARK
    p.font.name = "Calibri"


def style_table(
    table, col_widths: list[float], highlight_rows: set[int] | None = None
) -> None:
    """Apply clean compact table formatting."""
    highlight_rows = highlight_rows or set()
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = Inches(width)

    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_idx in highlight_rows:
                cell.fill.fore_color.rgb = GREEN_SOFT
            elif r_idx % 2 == 0:
                cell.fill.fore_color.rgb = LIGHT_BG
            else:
                cell.fill.fore_color.rgb = WHITE

            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Calibri"
                if r_idx == 0:
                    paragraph.font.size = Pt(10)
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = WHITE
                elif r_idx in highlight_rows:
                    paragraph.font.size = Pt(9.5)
                    paragraph.font.bold = True
                    paragraph.font.color.rgb = RGBColor(6, 95, 70)
                else:
                    paragraph.font.size = Pt(9.5)
                    paragraph.font.bold = c_idx == 0
                    paragraph.font.color.rgb = NAVY if c_idx == 0 else SLATE


def build_deck(output_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: PRINCIPALES INSIGHTS DE LOS DATOS (EDA OOT) + GRÁFICO MCC
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    add_header_and_footer(
        s1,
        "1. Principales Insights de los Datos (EDA Out-of-Time)",
        "Marzo–Abril 2026 (674.696 tx, 0% Leakage) | Función Banco: +US$120 VP  −US$30 FP  −US$180 FN (Asimetría 6:1)",
        1,
    )

    add_kpi_card(
        s1,
        0.45,
        1.12,
        2.95,
        0.92,
        "Volumen / Tasa Fraude",
        "1M tx | 0,31%",
        "3.113 fraudes (~1 cada 321 tx)",
        RED,
    )
    add_kpi_card(
        s1,
        3.60,
        1.12,
        2.95,
        0.92,
        "Ticket Mediano",
        "2,58x Mayor",
        "Fraude US$ 93 vs Legítima US$ 36",
        AMBER,
    )
    add_kpi_card(
        s1,
        6.75,
        1.12,
        2.95,
        0.92,
        "Nulos Estructurales",
        "59% Físicos",
        "100% nulos POS/ATM; 0% Digital",
        BLUE_ACCENT,
    )
    add_kpi_card(
        s1,
        9.90,
        1.12,
        2.98,
        0.92,
        "Grafos (NetworkX)",
        "251 Anillos",
        "11.847 dispositivos multitarjeta",
        GREEN,
    )

    add_concise_panel(
        s1,
        0.45,
        2.20,
        5.85,
        4.75,
        "Hallazgos Clave del EDA (Sin Fuga Temporal)",
        [
            (
                "Nulos No Aleatorios",
                "100% de ausencia de IP/Device ocurre en canales físicos (POS/ATM); 100% completitud en ECOMMERCE/APP.",
            ),
            (
                "Concentración por Rubro (MCC)",
                "Cuasi-Efectivo/Cripto (6051: 54,0%), Apuestas (7995: 29,9%) y Joyerías (5944: 21,6%) concentran el drenaje de fondos.",
            ),
            (
                "Clonación Física (Swipe)",
                "En POS, banda magnética (2,23%) y digitada (1,62%) multiplican >30x el riesgo frente a Chip (0,07%).",
            ),
            (
                "Ventana Nocturna",
                "Entre 01:00 y 05:59 hrs cae el flujo legítimo pero se dispara la tasa relativa de ataques.",
            ),
            (
                "Mismatch Geográfico",
                "Cuando merchant_country != ip_country, el riesgo salta 8,1x (0,81% vs 0,10%).",
            ),
        ],
    )

    add_chart_card(
        s1,
        6.50,
        2.20,
        6.38,
        4.75,
        "Gráfico 1: Tasa de Fraude (%) en Segmentos Críticos vs. Tasa Base (0,32%)",
    )
    c1_data = CategoryChartData()
    c1_data.categories = [
        "Tasa Base (Global)",
        "POS Chip (Seguro)",
        "Mismatch País != IP",
        "POS Banda (Swipe)",
        "MCC 5944 Joyerías",
        "MCC 7995 Apuestas",
        "MCC 6051 Cripto",
    ]
    c1_data.add_series(
        "Tasa de Fraude (%)", (0.32, 0.07, 0.81, 2.23, 21.55, 29.87, 54.02)
    )
    c1_shape = s1.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        Inches(6.68),
        Inches(2.65),
        Inches(6.0),
        Inches(4.1),
        c1_data,
    )
    c1 = c1_shape.chart
    c1.has_legend = False
    plot1 = c1.plots[0]
    plot1.has_data_labels = True
    plot1.data_labels.font.size = Pt(9.5)
    plot1.data_labels.font.bold = True
    plot1.data_labels.number_format = '0.00"%"'
    series1 = plot1.series[0]
    series1.format.fill.solid()
    series1.format.fill.fore_color.rgb = RED

    # =========================================================================
    # SLIDE 2: FEATURE ENGINEERING + GRÁFICO RESCATE PACIENTE CERO
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    add_header_and_footer(
        s2,
        "2. Feature Engineering y Destilación Causal 3.0 (Polars, 0% Leakage)",
        "Diagnóstico de Falsos Negativos (Parte IV): El 80,3% de los FN en V1 ocurría en el 1er fraude de la tarjeta ('Paciente Cero')",
        2,
    )

    add_concise_panel(
        s2,
        0.45,
        1.15,
        6.15,
        5.80,
        "De 34 Variables (V1) a 47 Variables Causales (V3)",
        [
            (
                "V1–V2 Velocidad y Ráfagas",
                "Conteos y sumas móviles retrospectivas (1h, 6h, 24h, 7d) por tarjeta/cliente y burst_ratio_1h_24h.",
            ),
            (
                "V1–V2 Geo y Grafos (24h)",
                "Salto geográfico imposible (<2h) y tarjetas distintas por device_id / ip_address en 24h.",
            ),
            (
                "¿Por qué escapaban 390 FN en V1?",
                "313 FN (80,3%) eran el 1er golpe tras ~6 días de inactividad (card_tx_count_1h = 0), dejando ciegas las ventanas cortas.",
            ),
            (
                "V3 Familia A (Shock Histórico)",
                "ratio_amount_card_prev_max (2,52x en FN), media expandida, Z-Score por segmento (mcc, canal, captura) y dormancy_shock.",
            ),
            (
                "V3 Familia B (First-Time Seen)",
                "is_new_device_for_customer (48,2% en FN vs 4,2% en legítimas → 11,5x), país inédito (30,3%) y rubro nuevo.",
            ),
            (
                "V3 Familia C (Comercio 24h)",
                "merchant_distinct_cards_24h para detectar puntos de compromiso / comercios fachada.",
            ),
        ],
    )

    add_chart_card(
        s2,
        6.80,
        1.15,
        6.08,
        5.80,
        "Gráfico 2: Recall en Test OOT según Posición en la Ráfaga (V1 vs V3)",
    )
    c2_data = CategoryChartData()
    c2_data.categories = [
        "1er Fraude ('Paciente Cero')\n[441 casos]",
        "2do Fraude en Tarjeta\n[269 casos]",
        "3er+ Fraude (Ráfaga)\n[235 casos]",
        "Recall Global Test OOT\n[945 fraudes]",
    ]
    c2_data.add_series("Modelo V1 (34 vars)", (28.9, 82.9, 87.2, 58.7))
    c2_data.add_series("Campeón V3 (47 vars)", (59.2, 84.4, 98.5, 76.7))

    c2_shape = s2.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(6.95),
        Inches(1.65),
        Inches(5.78),
        Inches(4.55),
        c2_data,
    )
    c2 = c2_shape.chart
    c2.has_legend = True
    c2.legend.position = XL_LEGEND_POSITION.TOP
    c2.legend.include_in_layout = False
    c2.legend.font.size = Pt(10)

    plot2 = c2.plots[0]
    plot2.has_data_labels = True
    plot2.data_labels.font.size = Pt(9.5)
    plot2.data_labels.font.bold = True
    plot2.data_labels.number_format = '0.0"%"'
    plot2.series[0].format.fill.solid()
    plot2.series[0].format.fill.fore_color.rgb = BLUE_LIGHT
    plot2.series[1].format.fill.solid()
    plot2.series[1].format.fill.fore_color.rgb = GREEN

    note2 = s2.shapes.add_textbox(
        Inches(6.95), Inches(6.28), Inches(5.78), Inches(0.55)
    )
    np2 = note2.text_frame.paragraphs[0]
    np2.text = "💡 La Destilación 3.0 duplica la detección del 'Paciente Cero' (28,9% → 59,2%) y captura el 98,5% desde el 3er intento."
    np2.font.size = Pt(10)
    np2.font.bold = True
    np2.font.color.rgb = NAVY
    np2.font.name = "Calibri"

    # =========================================================================
    # SLIDE 3: METODOLOGÍA UTILIZADA + GRÁFICO BENCHMARKING PR-AUC
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    add_header_and_footer(
        s3,
        "3. Metodología Utilizada",
        "Split Cronológico Out-of-Time | Preprocesamiento sin Fuga | Calibración Isotónica | Optimización Financiera T*",
        3,
    )

    add_concise_panel(
        s3,
        0.45,
        1.15,
        6.15,
        5.80,
        "Arquitectura de Validación y Toma de Decisiones",
        [
            (
                "1. Split Out-of-Time Estricto",
                "Train (01 Mar–15 Abr: 507k tx) → Val OOT (16–30 Abr: 167k tx) → Test OOT ciego (Mayo: 325k tx, 945 fraudes).",
            ),
            (
                "2. Pipeline scikit-learn",
                "CyclicalTime (sin/cos), TargetEncoder Out-of-Fold (cv=5) sobre MCC, y log1p + RobustScaler ajustados solo en Train.",
            ),
            (
                "3. Calibración Isotónica",
                "Corrige scale_pos_weight en Val OOT logrando Brier Score = 0,00150 y ECE = 0,0000.",
            ),
            (
                "4. Optimización Directa de T*",
                "Como un FN (-US$ 180) cuesta 6x más que un FP (-US$ 30), el corte óptimo baja de 0.50 a T* = 0,128 en V3.",
            ),
            (
                "5. Triage + Bloqueo + SHAP",
                "Política Tri-Zona (Aprobación / 2da Instancia / Bloqueo Directo), revocación de plástico tras 1er VP y auditoría TreeSHAP.",
            ),
        ],
    )

    add_chart_card(
        s3,
        6.80,
        1.15,
        6.08,
        5.80,
        "Gráfico 3: Evolución de PR-AUC (Average Precision) por Arquitectura",
    )
    c3_data = CategoryChartData()
    c3_data.categories = [
        "LightGBM\nBaseline (V1)",
        "LightGBM\nTuned (V1)",
        "XGBoost V1\n(34 vars)",
        "XGBoost V2\n(39 vars)",
        "XGBoost V3\nChampion (47 vars)",
    ]
    c3_data.add_series("Validación OOT (Abril)", (0.459, 0.559, 0.614, 0.629, 0.751))
    c3_data.add_series("Test OOT (Mayo)", (0.495, 0.598, 0.653, 0.664, 0.785))

    c3_shape = s3.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(6.95),
        Inches(1.65),
        Inches(5.78),
        Inches(4.55),
        c3_data,
    )
    c3 = c3_shape.chart
    c3.has_legend = True
    c3.legend.position = XL_LEGEND_POSITION.TOP
    c3.legend.include_in_layout = False
    c3.legend.font.size = Pt(10)

    plot3 = c3.plots[0]
    plot3.has_data_labels = True
    plot3.data_labels.font.size = Pt(9)
    plot3.data_labels.font.bold = True
    plot3.data_labels.number_format = "0.000"
    plot3.series[0].format.fill.solid()
    plot3.series[0].format.fill.fore_color.rgb = BLUE_LIGHT
    plot3.series[1].format.fill.solid()
    plot3.series[1].format.fill.fore_color.rgb = BLUE_ACCENT

    note3 = s3.shapes.add_textbox(
        Inches(6.95), Inches(6.28), Inches(5.78), Inches(0.55)
    )
    np3 = note3.text_frame.paragraphs[0]
    np3.text = "💡 Sin sobreajuste temporal: El PR-AUC sube de 0,751 (Val) a 0,785 (Test OOT) y ROC-AUC alcanza 0,9807."
    np3.font.size = Pt(10)
    np3.font.bold = True
    np3.font.color.rgb = NAVY
    np3.font.name = "Calibri"

    # =========================================================================
    # SLIDE 4: RESULTADOS E IMPACTO ECONÓMICO + GRÁFICO UTILIDAD NETA
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    add_header_and_footer(
        s4,
        "4. Resultados e Impacto Económico (Test Out-of-Time: Mayo 2026)",
        "Evaluación ciega en 325.304 tx (945 fraudes) — De pérdida residual (-US$ 26,5k) a Utilidad Neta Positiva (+US$ 39,0k)",
        4,
    )

    add_kpi_card(
        s4,
        0.45,
        1.12,
        2.95,
        0.92,
        "Utilidad Neta Directa",
        "+US$ 34.830",
        "+US$ 39.030 con bloqueo plástico",
        GREEN,
    )
    add_kpi_card(
        s4,
        3.60,
        1.12,
        2.95,
        0.92,
        "Ahorro vs. Inacción",
        "+US$ 204.930",
        "Revertimos pérdida de -US$ 170,1k",
        GREEN,
    )
    add_kpi_card(
        s4,
        6.75,
        1.12,
        2.95,
        0.92,
        "Recall / Precisión V3",
        "76,7% / 63,4%",
        "725 VP capturados | PR-AUC 0,785",
        BLUE_ACCENT,
    )
    add_kpi_card(
        s4,
        9.90,
        1.12,
        2.98,
        0.92,
        "Menos Fricción Cliente",
        "-45,1% FP vs V1",
        "Baja de 763 a 419 FP (~14/día)",
        AMBER,
    )

    rows_s4 = [
        [
            "Estrategia en Test OOT (Mayo 2026)",
            "PR-AUC",
            "T*",
            "VP (Recall)",
            "FP (Prec.)",
            "FN",
            "Utilidad Neta",
            "Ahorro vs Base",
        ],
        [
            "0. Sin Modelo (Inacción)",
            "—",
            "—",
            "0 (0,0%)",
            "0",
            "945",
            "-US$ 170.100",
            "US$ 0",
        ],
        [
            "1. XGBoost V1 (Default 0.50)",
            "0,653",
            "0,500",
            "463 (49,0%)",
            "226 (67%)",
            "482",
            "-US$ 37.980",
            "+US$ 132.120",
        ],
        [
            "2. XGBoost V1 (Umbral T*)",
            "0,653",
            "0,268",
            "555 (58,7%)",
            "763 (42%)",
            "390",
            "-US$ 26.490",
            "+US$ 143.610",
        ],
        [
            "3. XGBoost V2 (Extendida)",
            "0,664",
            "0,280",
            "569 (60,2%)",
            "664 (46%)",
            "376",
            "-US$ 19.320",
            "+US$ 150.780",
        ],
        [
            "4. champion_model_v3 (Destilación 3.0)",
            "0,785",
            "0,128",
            "725 (76,7%)",
            "419 (63%)",
            "220",
            "+US$ 34.830",
            "+US$ 204.930",
        ],
        [
            "5. V3 + Bloqueo Instantáneo Plástico",
            "0,785",
            "0,128",
            "739 (78,2%)",
            "419 (64%)",
            "206",
            "+US$ 39.030",
            "+US$ 209.130",
        ],
    ]

    t4_shape = s4.shapes.add_table(
        len(rows_s4), 8, Inches(0.45), Inches(2.20), Inches(7.15), Inches(4.75)
    )
    t4 = t4_shape.table
    for r_idx, row in enumerate(rows_s4):
        for c_idx, val in enumerate(row):
            t4.cell(r_idx, c_idx).text = val
    style_table(
        t4, [2.15, 0.62, 0.58, 0.95, 0.88, 0.45, 0.76, 0.76], highlight_rows={5, 6}
    )

    add_chart_card(
        s4,
        7.78,
        2.20,
        5.10,
        4.75,
        "Gráfico 4: Utilidad Neta Mensual Banco (US$, Mayo 2026)",
    )
    c4_data = CategoryChartData()
    c4_data.categories = [
        "Inacción\n(Sin Modelo)",
        "V1 Default\n(T=0.50)",
        "V1 Óptimo\n(T*=0.268)",
        "V2 Óptimo\n(T*=0.280)",
        "Campeón V3\n(T*=0.128)",
        "V3 + Bloqueo\nPlástico",
    ]
    c4_data.add_series(
        "Utilidad Neta (US$)", (-170100, -37980, -26490, -19320, 34830, 39030)
    )

    c4_shape = s4.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(7.90),
        Inches(2.68),
        Inches(4.85),
        Inches(4.05),
        c4_data,
    )
    c4 = c4_shape.chart
    c4.has_legend = False
    plot4 = c4.plots[0]
    plot4.has_data_labels = True
    plot4.data_labels.font.size = Pt(8.5)
    plot4.data_labels.font.bold = True
    plot4.data_labels.number_format = '"$"#,##0;("$"#,##0)'

    series4 = plot4.series[0]
    for idx, color in enumerate([RED, AMBER, AMBER, AMBER, GREEN, GREEN]):
        pt = series4.points[idx]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = color

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"Visual 4-slide presentation saved to: {output_path}")


if __name__ == "__main__":
    out_file = (
        Path(__file__).resolve().parent
        / "Presentacion_Ejecutiva_Fraude_Financiero.pptx"
    )
    build_deck(out_file)
