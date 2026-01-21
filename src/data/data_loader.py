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
    WEATHER_FORECAST_DIR,
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


# -----------------------------------------------------------------------------
# TZ hygiene
# -----------------------------------------------------------------------------
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Macht den Index konsistent: tz-aware UTC, duplikatfrei, sortierbar.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    return out


def _ensure_utc_series(s: pd.Series) -> pd.Series:
    """
    Macht den Index konsistent: tz-aware UTC, duplikatfrei, sortierbar.
    """
    if s is None or s.empty:
        return s

    out = s.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    return out


def _to_utc_timestamp(ts) -> pd.Timestamp:
    """
    Robust: tz-naive als lokale TIMEZONE interpretieren, tz-aware nach UTC konvertieren.
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(TIMEZONE).tz_convert("UTC")
    return t.tz_convert("UTC")


def _load_existing_hist(node_id: str) -> pd.DataFrame:
    """
    Lädt ggf. vorhandene historische Zeitreihen (P_MW + Wetter) für einen Node
    aus RAW_TS_DIR/<node_id>_hist.csv.

    Gibt einen DataFrame mit DatetimeIndex zurück oder einen leeren DataFrame.

    Wichtig: Index wird hier direkt auf tz-aware UTC normalisiert, damit später
    union/sort keine tz-Mischung erzeugt.
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

    df = _ensure_utc_index(df).sort_index()
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


def _load_node_metadata(graph_path: Path) -> pd.DataFrame:
    """
    Liest den Graphen und liefert Metadaten zu allen relevanten Knoten
    (uw_field, battery, etc.) als DataFrame.

    Columns:
        - node_id (Index)
        - node_type
        - P_Datapoint_ID
        - Latitude_deg
        - Longitude_deg
        - DerivedSpec (dict oder NaN)
    """
    logger.info("Lade Knotendaten (inkl. P_Datapoint_ID, Lat/Lon, DerivedSpec) aus %s ...", graph_path)

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

        derived_spec = features.get("derived")

        if not node_id:
            continue

        records.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "P_Datapoint_ID": p_id,
                "Latitude_deg": lat,
                "Longitude_deg": lon,
                "DerivedSpec": derived_spec,
            }
        )

    if not records:
        logger.warning("Keine relevanten Knoten im Graphen gefunden.")
        return pd.DataFrame(
            columns=["node_type", "P_Datapoint_ID", "Latitude_deg", "Longitude_deg", "DerivedSpec"]
        ).set_index(pd.Index([], name="node_id"))

    df = pd.DataFrame.from_records(records).set_index("node_id")
    logger.info("Knotendaten geladen: %d Knoten.", len(df))
    return df


def _apply_derived_measurements(
    nodes_df: pd.DataFrame,
    measurements: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    """
    Ergänzt das measurements-Dict um abgeleitete Knoten basierend auf der
    'DerivedSpec'-Spalte im nodes_df.
    """
    for node_id, row in nodes_df.iterrows():
        spec = row.get("DerivedSpec")
        if not isinstance(spec, dict):
            continue  # kein derived node

        method = spec.get("method")
        terms = spec.get("terms", [])

        if method != "field_sum":
            logger.warning(
                "Derived-Method %s für Node %s wird aktuell nicht unterstützt.",
                method,
                node_id,
            )
            continue

        logger.info("Berechne derived Node %s via field_sum.", node_id)

        series_list = []
        for term in terms:
            src_node = term.get("node")
            coeff = term.get("coeff", 1.0)

            if src_node not in measurements:
                logger.warning(
                    "Derived-Node %s: Basis-Node %s hat keine Messreihe – überspringe diesen Term.",
                    node_id,
                    src_node,
                )
                continue

            s_src = _ensure_utc_series(measurements[src_node].astype("float64"))
            series_list.append(coeff * s_src)

        if not series_list:
            logger.warning(
                "Derived-Node %s: keine gültigen Basis-Messreihen gefunden – keine abgeleitete Reihe erzeugt.",
                node_id,
            )
            continue

        df = pd.concat(series_list, axis=1)
        s_derived = df.sum(axis=1)
        s_derived.name = "P_MW"

        s_derived = _ensure_utc_series(s_derived).sort_index()
        measurements[node_id] = s_derived

        logger.info(
            "Derived-Node %s: abgeleitete P_MW-Reihe erzeugt (%d Zeitschritte).",
            node_id,
            len(s_derived),
        )

    return measurements


def _apply_derived_weather(
    nodes_df: pd.DataFrame,
    weather_hist: Dict[str, pd.DataFrame],
    weather_forecast: Dict[str, pd.DataFrame],
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Übernimmt Wetterdaten für derived nodes von einem Basis-Node.
    Standard: vom ersten Term in spec["terms"].
    """
    for node_id, row in nodes_df.iterrows():
        spec = row.get("DerivedSpec")
        if not isinstance(spec, dict):
            continue

        terms = spec.get("terms", [])
        if not terms:
            continue

        ref_node = terms[0].get("node")
        if not ref_node:
            continue

        if ref_node in weather_hist:
            weather_hist[node_id] = _ensure_utc_index(weather_hist[ref_node].copy())
            logger.info(
                "Derived-Node %s: Wetter-Historie von %s übernommen (%d Zeilen).",
                node_id,
                ref_node,
                len(weather_hist[node_id]),
            )
        else:
            logger.warning(
                "Derived-Node %s: Referenz-Node %s hat keine Wetter-Historie.",
                node_id,
                ref_node,
            )

        if ref_node in weather_forecast:
            weather_forecast[node_id] = _ensure_utc_index(weather_forecast[ref_node].copy())
            logger.info(
                "Derived-Node %s: Wetter-Prognose von %s übernommen (%d Zeilen).",
                node_id,
                ref_node,
                len(weather_forecast[node_id]),
            )
        else:
            logger.warning(
                "Derived-Node %s: Referenz-Node %s hat keine Wetter-Prognose.",
                node_id,
                ref_node,
            )

    return weather_hist, weather_forecast



def load_all_data(
    start_time: datetime = HIST_START,
    end_time: datetime | None = None,
    weather_forecast_hours: int = 30,
) -> Dict[str, Any]:
    """
    Zieht historische Wirkleistungs-Zeitreihen (P_MW) und Wetterdaten
    für alle relevanten Knoten.

    Intern halten wir alle Indizes tz-aware in UTC, damit join/union/sort stabil bleibt.
    """
    # --- 0) Start-/Endzeit sauber als lokale Zeit und UTC ableiten ---

    if end_time is None:
        end_local = pd.Timestamp.now(tz=TIMEZONE)
    else:
        end_local = pd.Timestamp(end_time)
        if end_local.tzinfo is None:
            end_local = end_local.tz_localize(TIMEZONE)
        else:
            end_local = end_local.tz_convert(TIMEZONE)

    end_time_utc = end_local.tz_convert(timezone.utc).to_pydatetime()

    start_local = pd.Timestamp(start_time)
    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(TIMEZONE)
    else:
        start_local = start_local.tz_convert(TIMEZONE)

    global_start_utc = start_local.tz_convert(timezone.utc).to_pydatetime()

    RAW_TS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starte Laden der historischen Wirkleistungs- und Wetterdaten "
        "(global_start_local=%s, end_local=%s, global_start_utc=%s, end_time_utc=%s) ...",
        start_local,
        end_local,
        global_start_utc,
        end_time_utc,
    )

    nodes_df = _load_node_metadata(GRAPH_PATH)

    measurements: Dict[str, pd.Series] = {}
    weather_hist: Dict[str, pd.DataFrame] = {}
    weather_forecast: Dict[str, pd.DataFrame] = {}

    bucket_delta = _bucket_timedelta()

    # -------------------------------------------------------------------------
    # 1) Pro physikalischem Node: EIOT + Wetter laden und Hist-CSV schreiben
    # -------------------------------------------------------------------------
    for row in nodes_df.itertuples():
        node_id = row.Index
        dp_id = row.P_Datapoint_ID
        derived_spec = getattr(row, "DerivedSpec", None)

        if isinstance(derived_spec, dict):
            logger.info(
                "Node %s ist ein derived node – EIOT- und Wetter-Abruf werden später aus Basis-Knoten abgeleitet.",
                node_id,
            )
            continue

        logger.info("Verarbeite Node %s (DP=%s) ...", node_id, dp_id)

        existing_hist_df = _load_existing_hist(node_id)
        existing_hist_end = existing_hist_df.index.max() if not existing_hist_df.empty else None
        # --- sanitize existing hist: remove weather-only tail (P_MW NaNs) ---
        if (not existing_hist_df.empty) and ("P_MW" in existing_hist_df.columns):
            existing_hist_df["P_MW"] = pd.to_numeric(existing_hist_df["P_MW"], errors="coerce")
            existing_hist_df = existing_hist_df.dropna(subset=["P_MW"]).sort_index()
            existing_hist_end = existing_hist_df.index.max() if not existing_hist_df.empty else None


        s = pd.Series(dtype="float64", name="P_MW")
        effective_start_utc = global_start_utc

        if existing_hist_end is not None:
            # existing_hist_end ist nach _load_existing_hist() UTC-aware
            existing_end_utc = _to_utc_timestamp(existing_hist_end).to_pydatetime()
            next_needed_utc = existing_end_utc + bucket_delta
            effective_start_utc = max(global_start_utc, next_needed_utc)

            logger.info(
                "Node %s: vorhandene Daten bis %s (UTC), lade neu ab %s (globaler Start UTC=%s).",
                node_id,
                existing_end_utc,
                effective_start_utc,
                global_start_utc,
            )

        if isinstance(dp_id, str) and dp_id:
            if effective_start_utc < end_time_utc:
                try:
                    resp = fetch_datapoint_raw(
                        datapoint_id=dp_id,
                        start_time=effective_start_utc,
                        end_time=end_time_utc,
                        aggregation=AGGREGATION,
                        bucket_factor=BUCKET_FACTOR,
                        bucket_unit=BUCKET_UNIT,
                        timezone_str=TIMEZONE,
                        chunk_days=7,
                    )

                    s_new = datapoint_json_to_series(resp, col_name="P_MW")
                    s_new = _ensure_utc_series(s_new)

                    if not existing_hist_df.empty and "P_MW" in existing_hist_df.columns:
                        s_old = _ensure_utc_series(existing_hist_df["P_MW"])
                        s = pd.concat([x for x in [s_old, s_new] if x is not None and not x.empty])
                        s = _ensure_utc_series(s).sort_index()
                    else:
                        s = s_new.sort_index()

                except Exception as e:
                    logger.error("Fehler beim Abruf für Node %s: %s", node_id, e, exc_info=True)

            else:
                logger.info(
                    "Node %s: vorhandene Daten reichen bereits bis >= end_time, kein EIOT-Abruf nötig.",
                    node_id,
                )
                if not existing_hist_df.empty and "P_MW" in existing_hist_df.columns:
                    s = _ensure_utc_series(existing_hist_df["P_MW"])
        else:
            logger.warning("Kein gültiger P_Datapoint_ID für Node %s – überspringe EIOT-Abruf.", node_id)

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
                start_time=start_local,
                end_time=end_local,
                timezone=TIMEZONE,
                forecast_hours=weather_forecast_hours,
            )
        except Exception as e:
            logger.error("Fehler beim Wetterabruf für Node %s: %s", node_id, e, exc_info=True)
            continue

        df_hist = _ensure_utc_index(df_hist)
        df_fc = _ensure_utc_index(df_fc)

        weather_hist[node_id] = df_hist
        weather_forecast[node_id] = df_fc

        logger.info(
            "Wetterdaten für Node %s geladen (hist=%d, forecast=%d).",
            node_id,
            len(df_hist),
            len(df_fc),
        )

        # Forecast separat speichern (so wie forecast.py es erwartet)
        if df_fc is not None and not df_fc.empty:
            WEATHER_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
            fc_path = WEATHER_FORECAST_DIR / f"{node_id}_weather_forecast.csv"
            df_fc.to_csv(fc_path, index_label="timestamp")
            logger.info("Wetter-Forecast gespeichert: %s (Zeilen: %d)", fc_path, len(df_fc))
        else:
            logger.warning("Wetter-Forecast leer für node_id=%s -> keine CSV geschrieben", node_id)

        # --- Kombinierte Zeitreihen speichern (P_MW + Wetter) ---
        hist_df = _ensure_utc_index(existing_hist_df.copy())

        # 1) P_MW ist die führende Zeitachse (nur Messzeitpunkte!)
        p_series = pd.Series(dtype="float64", name="P_MW")
        if node_id in measurements:
            p_series = _ensure_utc_series(measurements[node_id]).rename("P_MW")

        # wenn es weder bestehende P_MW noch neue P_MW gibt -> nicht speichern
        if (hist_df.empty or "P_MW" not in hist_df.columns) and p_series.empty:
            logger.warning("Node %s: keine P_MW (alt+neu leer) -> keine RAW hist CSV geschrieben.", node_id)
            continue

        # 2) hist_df so bauen, dass Index == P_MW-Zeitpunkte
        if hist_df.empty or "P_MW" not in hist_df.columns:
            # keine brauchbare bestehende CSV -> starte rein mit P_MW
            hist_df = p_series.to_frame()
        else:
            # bestehende CSV ist Basis (aber bereits vorher sanitized, d.h. keine P_MW-NaNs mehr)
            if not p_series.empty:
                hist_df["P_MW"] = pd.to_numeric(hist_df["P_MW"], errors="coerce")

                idx_common = p_series.index.intersection(hist_df.index)
                p_common = pd.to_numeric(p_series.loc[idx_common], errors="coerce")

                mask = p_common.notna()
                hist_df.loc[idx_common[mask], "P_MW"] = p_common.loc[mask].astype(float)


                # falls EIOT neue Zeitpunkte liefert, die in hist_df noch nicht existieren:
                new_idx = p_series.index.difference(hist_df.index)
                if len(new_idx) > 0:
                    p_new = pd.to_numeric(p_series.loc[new_idx], errors="coerce").dropna()
                    if not p_new.empty:
                        hist_df = pd.concat([hist_df, p_new.to_frame("P_MW")], axis=0).sort_index()




        # 3) Wetter HIST nur an P_MW-Index dazujoinen
        if node_id in weather_hist and not hist_df.empty:
            w_hist = _ensure_utc_index(weather_hist[node_id])

            # --- FIX: vorhandene Wetterspalten entfernen, bevor wir neu joinen ---
            overlap = hist_df.columns.intersection(w_hist.columns)
            if len(overlap) > 0:
                hist_df = hist_df.drop(columns=overlap)

            hist_df = hist_df.join(w_hist, how="left")


        # 4) finale Hygiene: erzwinge, dass jede gespeicherte Zeile P_MW hat
        hist_df["P_MW"] = pd.to_numeric(hist_df["P_MW"], errors="coerce")
        hist_df = hist_df.dropna(subset=["P_MW"]).sort_index()

        if hist_df.empty:
            logger.warning("Node %s: hist_df leer nach dropna(P_MW) -> keine RAW hist CSV geschrieben.", node_id)
            continue

        hist_path = RAW_TS_DIR / f"{node_id}_hist.csv"
        hist_df.to_csv(hist_path, index_label="timestamp")
        logger.info(
            "Historische Zeitreihen für %s gespeichert (%d Zeilen, %d Spalten) nach %s",
            node_id,
            len(hist_df),
            hist_df.shape[1],
            hist_path,
        )


    # -------------------------------------------------------------------------
    # 2) Derived-Nodes: P_MW ableiten und Wetter übernehmen
    # -------------------------------------------------------------------------
    measurements = _apply_derived_measurements(nodes_df, measurements)
    weather_hist, weather_forecast = _apply_derived_weather(nodes_df, weather_hist, weather_forecast)
    # Derived forecast CSVs schreiben 
    for node_id, row in nodes_df.iterrows():
        spec = row.get("DerivedSpec")
        if not isinstance(spec, dict):
            continue

        df_fc = weather_forecast.get(node_id)
        if df_fc is None or df_fc.empty:
            logger.warning("Derived-Node %s: Wetter-Forecast leer -> keine CSV", node_id)
            continue

        WEATHER_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
        fc_path = WEATHER_FORECAST_DIR / f"{node_id}_weather_forecast.csv"
        _ensure_utc_index(df_fc).to_csv(fc_path, index_label="timestamp")
        logger.info("Derived-Node %s: Wetter-Forecast gespeichert: %s (Zeilen: %d)", node_id, fc_path, len(df_fc))

    # -------------------------------------------------------------------------
    # 3) CSVs für derived nodes schreiben
    # -------------------------------------------------------------------------
    for node_id, row in nodes_df.iterrows():
        spec = row.get("DerivedSpec")
        if not isinstance(spec, dict):
            continue

        if node_id not in measurements:
            logger.warning(
                "Derived-Node %s hat keine abgeleitete P_MW-Reihe – keine CSV geschrieben.",
                node_id,
            )
            continue

        hist_df = _ensure_utc_series(measurements[node_id]).rename("P_MW").to_frame()

        if node_id in weather_hist:
            w_hist = _ensure_utc_index(weather_hist[node_id])
            hist_df = hist_df.join(w_hist, how="left")

        # erzwingen, dass keine Wetter-only Zeilen drin sind
        hist_df["P_MW"] = pd.to_numeric(hist_df["P_MW"], errors="coerce")
        hist_df = hist_df.dropna(subset=["P_MW"]).sort_index()


        if hist_df.empty:
            logger.warning(
                "Derived-Node %s: kombinierter Hist-DataFrame ist leer – keine CSV geschrieben.",
                node_id,
            )
            continue

        hist_df = _ensure_utc_index(hist_df).sort_index()
        hist_path = RAW_TS_DIR / f"{node_id}_hist.csv"
        hist_df.to_csv(hist_path, index_label="timestamp")
        logger.info(
            "Derived-Historie für %s gespeichert (%d Zeilen, %d Spalten) nach %s",
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
