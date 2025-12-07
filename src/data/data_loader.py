# src/data/data_loader.py

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import logging
from datetime import datetime, timezone, timedelta

from src.config import (
    GRAPH_PATH,
    RAW_TS_DIR,
    RELEVANT_NODE_TYPES,
    HIST_START,
    BUCKET_FACTOR,
    BUCKET_UNIT,
    TIMEZONE,
    AGGREGATION,
)
from src.data.eiot_client import fetch_datapoint_raw, datapoint_json_to_series
from src.data.weather_loader import fetch_weather_open_meteo

logger = logging.getLogger(__name__)


def _load_existing_hist(node_id: str) -> pd.DataFrame:
    """
    Lädt ggf. vorhandene historische Zeitreihen (P_MW + Wetter) für einen Node
    aus RAW_TS_DIR/<node_id>_hist.csv.

    Gibt einen DataFrame mit DatetimeIndex zurück oder einen leeren DataFrame.
    """
    hist_path = RAW_TS_DIR / f"{node_id}_hist.csv"
    if not hist_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(hist_path, parse_dates=["timestamp"], index_col="timestamp")
    except Exception as e:
        logger.error(
            "Konnte bestehende Hist-Datei für %s nicht laden (%s) – ignoriere Datei.",
            node_id,
            e,
            exc_info=True,
        )
        return pd.DataFrame()

    if df.empty:
        return df

    # Sicherstellen, dass Index sortiert ist
    df = df.sort_index()
    logger.info(
        "Gefundene bestehende Hist-Daten für %s: von %s bis %s (Zeilen: %d)",
        node_id,
        df.index.min(),
        df.index.max(),
        len(df),
    )
    return df


def _bucket_timedelta() -> timedelta:
    """
    Liefert die Zeitauflösung eines Buckets als timedelta auf Basis von
    BUCKET_FACTOR und BUCKET_UNIT.
    """
    unit = (BUCKET_UNIT or "").upper()
    if unit == "MINUTE":
        return timedelta(minutes=BUCKET_FACTOR)
    if unit == "HOUR":
        return timedelta(hours=BUCKET_FACTOR)
    if unit == "DAY":
        return timedelta(days=BUCKET_FACTOR)

    logger.warning(
        "Unbekannte BUCKET_UNIT=%s – verwende timedelta(0). "
        "Inkrementelle Logik könnte zu Duplikaten führen.",
        BUCKET_UNIT,
    )
    return timedelta(0)


def _load_nodes_with_p_datapoint(graph_path: Path) -> Dict[str, str]:
    """
    Alte Helper-Funktion (nur Mapping Node -> P_Datapoint_ID).
    Kann später entfernt werden, falls nicht mehr genutzt.
    """
    logger.info(f"Lade Graph aus {graph_path} ...")

    with graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    node_to_p = {}

    for item in graph:
        data = item.get("data", {})
        # Kanten überspringen
        if "source" in data and "target" in data:
            continue

        node_id = data.get("id")
        node_type = (data.get("type") or "").strip()

        if node_type not in RELEVANT_NODE_TYPES:
            continue

        p_id = data.get("features", {}).get("P_Datapoint_ID")

        if node_id and isinstance(p_id, str) and p_id:
            node_to_p[node_id] = p_id
            logger.debug(f"Gefunden: Node {node_id} → Datapoint {p_id}")

    logger.info(f"{len(node_to_p)} relevante Knoten mit P_Datapoint_ID gefunden.")
    return node_to_p


def _load_node_metadata(graph_path: Path) -> pd.DataFrame:
    """
    Liest den Graphen und liefert Metadaten zu allen relevanten Knoten
    (uw_field, battery) als DataFrame.

    Columns:
        - node_id (Index)
        - node_type
        - P_Datapoint_ID
        - Latitude_deg
        - Longitude_deg
    """
    logger.info("Lade Knotendaten (inkl. P_Datapoint_ID und Lat/Lon) aus %s ...", graph_path)

    with graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    records = []

    for item in graph:
        data = item.get("data", {})
        # Kanten überspringen
        if "source" in data and "target" in data:
            continue

        node_id = data.get("id")
        node_type = (data.get("type") or "").strip()

        if node_type not in RELEVANT_NODE_TYPES:
            continue

        features = data.get("features", {}) or {}
        p_raw = features.get("P_Datapoint_ID")
        p_id = p_raw.strip() if isinstance(p_raw, str) else None
        lat = features.get("Latitude_deg")
        lon = features.get("Longitude_deg")

        if not node_id:
            continue

        records.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "P_Datapoint_ID": p_id,
                "Latitude_deg": lat,
                "Longitude_deg": lon,
            }
        )

    if not records:
        logger.warning("Keine relevanten Knoten im Graphen gefunden.")
        return pd.DataFrame(
            columns=["node_type", "P_Datapoint_ID", "Latitude_deg", "Longitude_deg"]
        ).set_index(pd.Index([], name="node_id"))

    df = pd.DataFrame.from_records(records).set_index("node_id")
    logger.info("Knotendaten geladen: %d Knoten.", len(df))
    return df


def load_all_data(
    start_time: datetime = HIST_START,
    end_time: datetime | None = None,
    weather_past_hours: int = 24,
    weather_forecast_hours: int = 24,
) -> Dict[str, Any]:
    """
    Zieht historische Wirkleistungs-Zeitreihen (P_MW) und Wetterdaten
    für alle relevanten Knoten.

    Nutzt ggf. bereits vorhandene CSVs in RAW_TS_DIR/<node_id>_hist.csv und
    lädt nur die fehlenden Intervalle nach.

    Returns:
        {
            "nodes": DataFrame mit Metadaten je Node,
            "measurements": Dict[node_id -> pd.Series (P_MW)],
            "weather_hist": Dict[node_id -> pd.DataFrame],
            "weather_forecast": Dict[node_id -> pd.DataFrame],
        }
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)

    # end_time sicher als UTC-datetime
    if end_time.tzinfo is None:
        end_time_utc = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time_utc = end_time.astimezone(timezone.utc)

    RAW_TS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starte Laden der historischen Wirkleistungs- und Wetterdaten ...")

    nodes_df = _load_node_metadata(GRAPH_PATH)

    measurements: Dict[str, pd.Series] = {}
    weather_hist: Dict[str, pd.DataFrame] = {}
    weather_forecast: Dict[str, pd.DataFrame] = {}

    # globaler Start immer als UTC-datetime
    if start_time.tzinfo is None:
        global_start_utc = start_time.replace(tzinfo=timezone.utc)
    else:
        global_start_utc = start_time.astimezone(timezone.utc)

    bucket_delta = _bucket_timedelta()

    for row in nodes_df.itertuples():
        node_id = row.Index
        dp_id = row.P_Datapoint_ID

        logger.info("Verarbeite Node %s (DP=%s) ...", node_id, dp_id)

        # --- Bereits vorhandene Hist-Daten laden (P_MW + Wetter) ---
        existing_hist_df = _load_existing_hist(node_id)
        existing_hist_end = existing_hist_df.index.max() if not existing_hist_df.empty else None

        # --- EIOT: Wirkleistungs-Zeitreihe holen (mit Chunking im Client) ---
        s = pd.Series(dtype="float64", name="P_MW")

        # effektive Startzeit abhängig von schon vorhandenen Daten
        effective_start = global_start_utc

        if existing_hist_end is not None:
            # existing_hist_end ist aktuell naiv – in UTC bringen
            if existing_hist_end.tzinfo is None:
                existing_end_utc = existing_hist_end.replace(tzinfo=timezone.utc)
            else:
                existing_end_utc = existing_hist_end.astimezone(timezone.utc)

            next_needed = existing_end_utc + bucket_delta
            # Wir wollen nichts vor global_start_utc und nichts doppelt
            effective_start = max(global_start_utc, next_needed)

            logger.info(
                "Node %s: vorhandene Daten bis %s, lade neu ab %s (globaler Start=%s).",
                node_id,
                existing_end_utc,
                effective_start,
                global_start_utc,
            )

        # Nur offene Intervalle laden
        if isinstance(dp_id, str) and dp_id:
            if effective_start < end_time_utc:
                try:
                    resp = fetch_datapoint_raw(
                        datapoint_id=dp_id,
                        start_time=effective_start,
                        end_time=end_time_utc,
                        aggregation=AGGREGATION,
                        bucket_factor=BUCKET_FACTOR,
                        bucket_unit=BUCKET_UNIT,
                        timezone_str=TIMEZONE,
                        chunk_days=7,  # z.B. 7 Tage pro Request
                    )
                    s_new = datapoint_json_to_series(resp, col_name="P_MW")
                    # bestehende + neue P_MW-Daten zusammenführen
                    if not existing_hist_df.empty and "P_MW" in existing_hist_df.columns:
                        s_old = existing_hist_df["P_MW"]
                        s = pd.concat([s_old, s_new])
                        s = s[~s.index.duplicated(keep="last")].sort_index()
                    else:
                        s = s_new
                except Exception as e:
                    logger.error(
                        "Fehler beim Abruf für Node %s: %s", node_id, e, exc_info=True
                    )
            else:
                logger.info(
                    "Node %s: vorhandene Daten reichen bereits bis >= end_time, "
                    "kein EIOT-Abruf nötig.",
                    node_id,
                )
                if not existing_hist_df.empty and "P_MW" in existing_hist_df.columns:
                    s = existing_hist_df["P_MW"]
        else:
            logger.warning(
                "Kein gültiger P_Datapoint_ID für Node %s – überspringe EIOT-Abruf.",
                node_id,
            )

        if s.empty:
            logger.warning("Keine Wirkleistungswerte für %s erhalten — übersprungen.", node_id)
        else:
            measurements[node_id] = s

        # --- Wetterdaten holen (hist + forecast) ---
        lat = getattr(row, "Latitude_deg")
        lon = getattr(row, "Longitude_deg")

        if pd.isna(lat) or pd.isna(lon):
            logger.warning(
                "Keine gültigen Koordinaten für Node %s (lat=%s, lon=%s) – überspringe Wetter.",
                node_id,
                lat,
                lon,
            )
            continue

        try:
            df_hist, df_fc = fetch_weather_open_meteo(
                latitude=float(lat),
                longitude=float(lon),
                past_hours=weather_past_hours,
                forecast_hours=weather_forecast_hours,
                timezone=TIMEZONE,
            )
        except Exception as e:
            logger.error("Fehler beim Wetterabruf für Node %s: %s", node_id, e, exc_info=True)
            continue

        weather_hist[node_id] = df_hist
        weather_forecast[node_id] = df_fc
        logger.info(
            "Wetterdaten für Node %s geladen (hist=%d, forecast=%d).",
            node_id,
            len(df_hist),
            len(df_fc),
        )

        # --- Kombinierte Zeitreihen speichern (P_MW + Wetter) ---

        # Basis: bestehende Hist-Daten, falls vorhanden
        hist_df = existing_hist_df.copy()

        # P_MW aus measurements (bereits alt+neu zusammengeführt)
        if node_id in measurements:
            p_series = measurements[node_id].rename("P_MW")
            # Index vereinigen und P_MW setzen (neue Werte überschreiben alte)
            if hist_df.empty:
                hist_df = p_series.to_frame()
            else:
                hist_df = hist_df.reindex(hist_df.index.union(p_series.index))
                hist_df["P_MW"] = p_series

        # aktuelle Wetter-Historie dazu
        if node_id in weather_hist:
            w_hist = weather_hist[node_id]
            if hist_df.empty:
                hist_df = w_hist.copy()
            else:
                # Index vereinigen, Spalten der Wetterdaten setzen/überschreiben
                hist_df = hist_df.reindex(hist_df.index.union(w_hist.index))
                for col in w_hist.columns:
                    hist_df[col] = w_hist[col]

        if not hist_df.empty:
            hist_df = hist_df.sort_index()
            hist_path = RAW_TS_DIR / f"{node_id}_hist.csv"
            hist_df.to_csv(hist_path, index_label="timestamp")
            logger.info(
                "Historische Zeitreihen für %s gespeichert (%d Zeilen, %d Spalten) nach %s",
                node_id,
                len(hist_df),
                hist_df.shape[1],
                hist_path,
            )

    logger.info("Datenladen abgeschlossen.")
    return {
        "nodes": nodes_df,
        "measurements": measurements,
        "weather_hist": weather_hist,
        "weather_forecast": weather_forecast,
    }
