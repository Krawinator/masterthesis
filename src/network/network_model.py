# src/network/network_model.py

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import GRAPH_PATH, SLACK_NODE_ID  # <-- Slack aus config

logger = logging.getLogger(__name__)


@dataclass
class NetworkModel:
    """
    Container für alle Netzobjekte, die wir für PTDF/DC-Lastfluss brauchen.
    """
    nodes: pd.DataFrame          # index = node_id
    lines: pd.DataFrame          # index = line_id, columns: from_node, to_node, x, b, limit_a, length_km
    A: pd.DataFrame              # Inzidenzmatrix (lines x nodes)
    Bbus: pd.DataFrame           # Suszeptanzmatrix (nodes x nodes)
    slack_node: str              # gewählter Slack-Knoten
    Ar: pd.DataFrame             # reduzierte Inzidenzmatrix (ohne Slack-Spalte)
    B_rr: pd.DataFrame           # reduzierte B-Matrix (ohne Slack-Zeile/-Spalte)


# ---------------------------------------------------------------------
# Hilfsfunktionen zum Parsen der JSON-Struktur
# ---------------------------------------------------------------------


def _load_graph_json(path: Path) -> dict:
    """
    Lädt das JSON-Graph-File.

    Dein Format:
        [
          { "data": {...}, "position": {...} },
          { "data": {...}, "position": {...} },
          ...
        ]

    → Elemente mit data.source/target = Edges
      Elemente ohne source/target         = Nodes
    """
    logger.info("Lade Netzgraph aus %s ...", path)
    with path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    # Fall 1: Liste von Elementen (wie bei dir)
    if isinstance(graph, list):
        nodes_raw = []
        edges_raw = []
        for el in graph:
            data = el.get("data", {})
            if "source" in data and "target" in data:
                edges_raw.append(el)
            else:
                nodes_raw.append(el)
        logger.info("Graph geladen (Listenformat): %d Nodes, %d Edges.", len(nodes_raw), len(edges_raw))
        return {"nodes": nodes_raw, "edges": edges_raw}

    # Fallback: evtl. andere Struktur
    if "nodes" in graph:
        nodes_raw = graph["nodes"]
    elif "elements" in graph and "nodes" in graph["elements"]:
        nodes_raw = graph["elements"]["nodes"]
    else:
        raise ValueError("Konnte Nodes in whole_graph.json nicht finden (erwarte Liste oder 'nodes' / 'elements.nodes').")

    if "edges" in graph:
        edges_raw = graph["edges"]
    elif "elements" in graph and "edges" in graph["elements"]:
        edges_raw = graph["elements"]["edges"]
    else:
        raise ValueError("Konnte Edges in whole_graph.json nicht finden (erwarte Liste oder 'edges' / 'elements.edges').")

    logger.info("Graph geladen (Objektformat): %d Nodes, %d Edges.", len(nodes_raw), len(edges_raw))
    return {"nodes": nodes_raw, "edges": edges_raw}


def _extract_node_data(node_entry: dict) -> Tuple[str, Dict]:
    """
    Extrahiert (node_id, attrs) aus einem Node-Eintrag.

    Dein Format:
      {
        "data": {
          "id": "SHUW_E24",
          "label": "SHUW_E24",
          "features": {...},
          "type": "uw_field"
        },
        "position": {...}
      }

    → Wir flatten:
      - data-Felder (id, label, type, ...)
      - features-Felder (P_Datapoint_ID, Latitude_deg, ...)
      - position.x / position.y
    """
    data = node_entry.get("data", {})
    position = node_entry.get("position", {})

    node_id = data.get("id")
    if node_id is None:
        raise ValueError(f"Node ohne 'id' gefunden: {node_entry}")

    node_type = data.get("type") or data.get("node_type")
    label = data.get("label")

    features = data.get("features", {}) or {}

    attrs = {}
    attrs.update(data)      # id, label, type, features (wird gleich überschrieben)
    attrs.pop("features", None)
    attrs["id"] = node_id
    attrs["type"] = node_type
    attrs["label"] = label

    # Features flatten (z.B. P_Datapoint_ID, Latitude_deg, Longitude_deg, Strom_Limit_in_A, ...)
    for k, v in features.items():
        attrs[k] = v

    # Position (optional)
    if "x" in position:
        attrs["pos_x"] = position["x"]
    if "y" in position:
        attrs["pos_y"] = position["y"]

    return node_id, attrs


def _extract_edge_data(edge_entry: dict) -> Tuple[str, Dict]:
    """
    Extrahiert (line_id, attrs) aus einem Edge-Eintrag.

    Dein Format:
      {
        "data": {
          "id": "...",
          "source": "SHUW_E24",
          "target": "JUBO_A5",
          "label": "...",
          "features": {
            "Strom_Limit_in_A": 1435,
            "X_ohm_per_km": 0.6534,
            "length_km": 2.49,
            "X_total_ohm": 1.6269
          }
        }
      }

    Wir bestimmen:
      - from_node, to_node
      - x (Reaktanz in Ohm, hier X_total_ohm oder length_km * X_ohm_per_km)
      - b = 1/x
      - limit_a = Strom_Limit_in_A
      - length_km
    """
    data = edge_entry.get("data", {})

    line_id = data.get("id")
    if line_id is None:
        # Fallback: Source-Target als ID
        line_id = f"{data.get('source')}__{data.get('target')}"

    source = data.get("source")
    target = data.get("target")
    if source is None or target is None:
        raise ValueError(f"Edge ohne 'source'/'target' gefunden: {edge_entry}")

    features = data.get("features", {}) or {}

    # Länge
    length_km = features.get("length_km")

    # Reaktanz: X_total_ohm
    x_total = features.get("X_total_ohm")

    x = None
    if x_total is not None:
        x = x_total

    if x is None:
        logger.warning(
            "Leitung %s (%s -> %s) hat keine Reaktanz (X_total_ohm) – wird vorerst mit NaN-x geführt.",
            line_id, source, target
        )
        x = np.nan

    # Stromgrenze in A
    limit_a = features.get("Strom_Limit_in_A")

    attrs = {}
    attrs.update(data)
    attrs.pop("features", None)
    attrs["id"] = line_id
    attrs["from_node"] = source
    attrs["to_node"] = target
    attrs["x"] = float(x) if x is not None else np.nan
    attrs["length_km"] = float(length_km) if length_km is not None else np.nan
    attrs["limit_a"] = float(limit_a) if limit_a is not None else np.nan

    # Features ggf. zusätzlich flatten, falls du später noch mehr brauchst
    for k, v in features.items():
        attrs[k] = v

    return line_id, attrs


def _build_nodes_df(nodes_raw: List[dict]) -> pd.DataFrame:
    """
    Baut ein DataFrame mit allen Knoten.
    """
    records = []
    for entry in nodes_raw:
        node_id, attrs = _extract_node_data(entry)
        records.append(attrs)

    df_nodes = pd.DataFrame(records)
    df_nodes = df_nodes.set_index("id").sort_index()

    logger.info("Nodes-DataFrame aufgebaut: %d Knoten (Index=id).", len(df_nodes))
    if "type" in df_nodes.columns:
        logger.debug("Node-Typen-Verteilung:\n%s", df_nodes["type"].value_counts(dropna=False))

    return df_nodes


def _build_lines_df(edges_raw: List[dict]) -> pd.DataFrame:
    """
    Baut ein DataFrame mit allen Leitungen.
    Entfernt Leitungen ohne gültige Reaktanz (x NaN).
    """
    records = []
    for entry in edges_raw:
        line_id, attrs = _extract_edge_data(entry)
        records.append(attrs)

    df_lines = pd.DataFrame(records)
    df_lines = df_lines.set_index("id").sort_index()

    # Nur Leitungen mit gültiger Reaktanz behalten
    before = len(df_lines)
    df_lines = df_lines[~df_lines["x"].isna()].copy()
    after = len(df_lines)
    if after < before:
        logger.warning(
            "Es wurden %d Leitungen ohne Reaktanz entfernt (verbleibend: %d).",
            before - after,
            after,
        )

    # Suszeptanz b = 1/x
    df_lines["b"] = 1.0 / df_lines["x"]

    logger.info("Lines-DataFrame aufgebaut: %d Leitungen (Index=id).", len(df_lines))
    return df_lines


# ---------------------------------------------------------------------
# Matrices: Inzidenzmatrix A und Suszeptanzmatrix Bbus
# ---------------------------------------------------------------------


def _build_incidence_matrix(nodes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """
    Baut die Inzidenzmatrix A (Zeilen = Leitungen, Spalten = Knoten):
        A[l, from_node] = +1
        A[l, to_node]   = -1
    """
    node_ids = list(nodes.index)
    line_ids = list(lines.index)

    node_index = {nid: i for i, nid in enumerate(node_ids)}
    A = np.zeros((len(line_ids), len(node_ids)), dtype=float)

    for li, line_id in enumerate(line_ids):
        row = lines.loc[line_id]
        from_node = row["from_node"]
        to_node = row["to_node"]

        if from_node not in node_index or to_node not in node_index:
            logger.warning(
                "Leitung %s referenziert unbekannte Knoten (%s -> %s) – wird in A übersprungen.",
                line_id, from_node, to_node,
            )
            continue

        i_from = node_index[from_node]
        i_to = node_index[to_node]

        A[li, i_from] = 1.0
        A[li, i_to] = -1.0

    A_df = pd.DataFrame(A, index=line_ids, columns=node_ids)
    logger.info("Inzidenzmatrix A aufgebaut: %s", A_df.shape)
    return A_df


def _build_Bbus(nodes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """
    Baut die Suszeptanzmatrix Bbus (Knoten x Knoten) nach DC-Lastfluss:
        Für jede Leitung mit b:
            B[i,i] += b
            B[j,j] += b
            B[i,j] -= b
            B[j,i] -= b
    """
    node_ids = list(nodes.index)
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    N = len(node_ids)
    B = np.zeros((N, N), dtype=float)

    for line_id, row in lines.iterrows():
        from_node = row["from_node"]
        to_node = row["to_node"]
        b = row["b"]

        if from_node not in node_index or to_node not in node_index:
            logger.warning(
                "Leitung %s referenziert unbekannte Knoten (%s -> %s) – übersprungen in Bbus.",
                line_id, from_node, to_node,
            )
            continue

        i = node_index[from_node]
        j = node_index[to_node]

        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b

    B_df = pd.DataFrame(B, index=node_ids, columns=node_ids)
    logger.info("Bbus-Matrix aufgebaut: %s", B_df.shape)
    return B_df


def _choose_slack_node(nodes: pd.DataFrame, slack_node: Optional[str] = None) -> str:
    """
    Wählt den Slack-Knoten:

    Priorität:
      1. expliziter Funktionsparameter `slack_node`
      2. globale Konfiguration `SLACK_NODE_ID` aus config.py
      3. Fallback: erster 'uw_field'-Knoten
      4. Fallback: erster Knoten im Index
    """
    # 1) Explizit übergeben
    if slack_node is not None:
        if slack_node not in nodes.index:
            raise ValueError(f"Gewählter Slack-Knoten '{slack_node}' ist nicht im Node-Index.")
        logger.info("Slack-Knoten explizit gesetzt auf %s.", slack_node)
        return slack_node

    # 2) Aus globaler Config
    if SLACK_NODE_ID is not None:
        if SLACK_NODE_ID not in nodes.index:
            raise ValueError(
                f"Konfigurierter SLACK_NODE_ID='{SLACK_NODE_ID}' "
                f"ist nicht im Node-Index. Verfügbare Knoten-Beispiele: {list(nodes.index[:5])}"
            )
        logger.info("Slack-Knoten aus config.py gesetzt auf %s.", SLACK_NODE_ID)
        return SLACK_NODE_ID

    # 3) Heuristik: erster uw_field
    if "type" in nodes.columns:
        uw_nodes = nodes[nodes["type"] == "uw_field"].index.tolist()
        if uw_nodes:
            chosen = uw_nodes[0]
            logger.info("Slack-Knoten automatisch gewählt: %s (type=uw_field).", chosen)
            return chosen

    # 4) Fallback: erster Knoten
    chosen = nodes.index[0]
    logger.warning(
        "Kein expliziter Slack-Knoten und kein 'uw_field'-Knoten gefunden – "
        "Slack-Knoten wird auf ersten Knoten im Index gesetzt: %s.",
        chosen,
    )
    return chosen


def _reduce_matrices_for_slack(
    A: pd.DataFrame,
    Bbus: pd.DataFrame,
    slack_node: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entfernt die Slack-Spalte/Zeile aus A/Bbus:
        - Ar: A ohne Slack-Spalte
        - B_rr: Bbus ohne Slack-Zeile und Slack-Spalte
    """
    if slack_node not in Bbus.index:
        raise ValueError(f"Slack-Knoten {slack_node} nicht in Bbus vorhanden.")

    non_slack_nodes = [n for n in Bbus.index if n != slack_node]

    Ar = A.loc[:, non_slack_nodes].copy()
    B_rr = Bbus.loc[non_slack_nodes, non_slack_nodes].copy()

    logger.info(
        "Reduktion für Slack-Knoten %s durchgeführt: Ar=%s, B_rr=%s.",
        slack_node,
        Ar.shape,
        B_rr.shape,
    )

    return Ar, B_rr


# ---------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------


def load_network_model(slack_node: Optional[str] = None) -> NetworkModel:
    """
    Hauptfunktion zum Laden des Netzmodells aus whole_graph.json.

    Parameter
    ---------
    slack_node : Optional[str]
        ID des Slack-Knotens.
        - Wenn None: wird SLACK_NODE_ID aus config.py verwendet (bzw. Fallback-Heuristik).
        - Wenn gesetzt: überschreibt die globale Konfiguration.
    """
    graph = _load_graph_json(GRAPH_PATH)
    nodes_raw = graph["nodes"]
    edges_raw = graph["edges"]

    nodes_df = _build_nodes_df(nodes_raw)
    lines_df = _build_lines_df(edges_raw)

    # Inzidenz & Bbus
    A = _build_incidence_matrix(nodes_df, lines_df)
    Bbus = _build_Bbus(nodes_df, lines_df)

    # Slack wählen & Matrizen reduzieren
    slack = _choose_slack_node(nodes_df, slack_node=slack_node)
    Ar, B_rr = _reduce_matrices_for_slack(A, Bbus, slack_node=slack)

    logger.info("Netzmodell erfolgreich aufgebaut. Slack-Knoten: %s.", slack)

    return NetworkModel(
        nodes=nodes_df,
        lines=lines_df,
        A=A,
        Bbus=Bbus,
        slack_node=slack,
        Ar=Ar,
        B_rr=B_rr,
    )
