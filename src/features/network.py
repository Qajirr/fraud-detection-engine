"""Módulo de Análisis de Redes y Teoría de Grafos para Detección de Fraude y PLAFT."""

from typing import Any

import networkx as nx
import polars as pl


def compute_network_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula métricas causales de diversidad de red por tarjeta en 24 horas sin fuga temporal.

    - card_distinct_merchants_24h: número de comercios distintos operados por la tarjeta en 24h.
    - card_distinct_countries_24h: número de países distintos operados por la tarjeta en 24h.
    """
    sorted_df = df.sort("timestamp")

    roll_merchants = (
        sorted_df.rolling(
            index_column="timestamp",
            period="24h",
            closed="right",
            group_by="card_id",
        )
        .agg(
            [
                pl.col("transaction_id").last().alias("transaction_id"),
                pl.col("merchant_id").n_unique().alias("card_distinct_merchants_24h"),
                pl.col("merchant_country")
                .n_unique()
                .alias("card_distinct_countries_24h"),
            ]
        )
        .select(
            [
                "transaction_id",
                "card_distinct_merchants_24h",
                "card_distinct_countries_24h",
            ]
        )
    )

    return df.join(roll_merchants, on="transaction_id", how="left").with_columns(
        [
            pl.col("card_distinct_merchants_24h").fill_null(1).cast(pl.Int32),
            pl.col("card_distinct_countries_24h").fill_null(1).cast(pl.Int32),
        ]
    )


def build_bipartite_network(
    df: pl.DataFrame,
    include_ip: bool = True,
    include_merchant: bool = False,
) -> nx.Graph:
    """Construye un grafo heterogéneo no dirigido conectando tarjetas con dispositivos e IPs.

    Nodos:
    - Tarjetas: tipo 'card'
    - Dispositivos: tipo 'device'
    - Direcciones IP: tipo 'ip'
    - Comercios: tipo 'merchant' (opcional)

    Aristas: presencia de transacción entre la tarjeta y la entidad física/digital.
    """
    graph = nx.Graph()

    # Filtrar registros relevantes
    card_col = "card_id"
    device_col = "device_id"
    ip_col = "ip_address"
    merchant_col = "merchant_id"

    # Seleccionar columnas no nulas para acelerar iteración
    pdf = df.select(
        [c for c in [card_col, device_col, ip_col, merchant_col] if c in df.columns]
    ).to_pandas()

    for _, row in pdf.iterrows():
        c_node = f"card_{row[card_col]}"
        graph.add_node(c_node, node_type="card")

        if (
            device_col in row
            and row[device_col] is not None
            and str(row[device_col]) != "nan"
        ):
            d_node = f"dev_{row[device_col]}"
            graph.add_node(d_node, node_type="device")
            graph.add_edge(c_node, d_node, relation="uses_device")

        if (
            include_ip
            and ip_col in row
            and row[ip_col] is not None
            and str(row[ip_col]) != "nan"
        ):
            ip_node = f"ip_{row[ip_col]}"
            graph.add_node(ip_node, node_type="ip")
            graph.add_edge(c_node, ip_node, relation="uses_ip")

        if include_merchant and merchant_col in row and row[merchant_col] is not None:
            m_node = f"mer_{row[merchant_col]}"
            graph.add_node(m_node, node_type="merchant")
            graph.add_edge(c_node, m_node, relation="operates_at")

    return graph


def detect_fraud_rings(
    graph: nx.Graph,
    min_cards: int = 2,
) -> list[dict[str, Any]]:
    """Identifica anillos o subgrafos sospechosos donde múltiples tarjetas comparten dispositivo o IP.

    Retorna lista de componentes conexas que representan anillos de fraude potenciales.
    """
    rings = []
    for component in nx.connected_components(graph):
        cards_in_comp = [n for n in component if n.startswith("card_")]
        devices_in_comp = [n for n in component if n.startswith("dev_")]
        ips_in_comp = [n for n in component if n.startswith("ip_")]

        if len(cards_in_comp) >= min_cards and (
            len(devices_in_comp) > 0 or len(ips_in_comp) > 0
        ):
            subg = graph.subgraph(component)
            rings.append(
                {
                    "card_count": len(cards_in_comp),
                    "device_count": len(devices_in_comp),
                    "ip_count": len(ips_in_comp),
                    "nodes": list(component),
                    "edges": list(subg.edges()),
                    "density": float(nx.density(subg)),
                }
            )

    return sorted(rings, key=lambda x: x["card_count"], reverse=True)


def compute_graph_degree_metrics(graph: nx.Graph) -> dict[str, int]:
    """Calcula el grado de conectividad de cada nodo en la red."""
    return dict(graph.degree())
