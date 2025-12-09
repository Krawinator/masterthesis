# src/data/data_cleaning.py

import logging
from typing import Dict, Any

import pandas as pd

from src.config import FREQ, MAX_GAP_STEPS, WEATHER_COLS  

logger = logging.getLogger(__name__)


def _reindex_and_interp_series(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=FREQ)
    s = s.reindex(full_index)
    s = s.interpolate(method="time", limit=MAX_GAP_STEPS, limit_direction="both")
    return s


def _reindex_and_interp_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_index)
    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(
                method="time", limit=MAX_GAP_STEPS, limit_direction="both"
            )
    return df


def clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nimmt die Struktur von load_all_data und gibt eine bereinigte Kopie zurück.
    """
    nodes = data["nodes"]

    # 1) P_MW bereinigen
    cleaned_measurements: Dict[str, pd.Series] = {}
    for node_id, s in data["measurements"].items():
        logger.info("Bereinige P_MW für Node %s ...", node_id)
        cleaned_measurements[node_id] = _reindex_and_interp_series(s)

    # 2) Wetterdaten bereinigen
    cleaned_weather_hist: Dict[str, pd.DataFrame] = {}
    for node_id, df in data["weather_hist"].items():
        logger.info("Bereinige Wetter-Historie für Node %s ...", node_id)
        cleaned_weather_hist[node_id] = _reindex_and_interp_weather(df)

    cleaned_weather_forecast: Dict[str, pd.DataFrame] = {}
    for node_id, df in data["weather_forecast"].items():
        cleaned_weather_forecast[node_id] = df.sort_index()

    return {
        "nodes": nodes,
        "measurements": cleaned_measurements,
        "weather_hist": cleaned_weather_hist,
        "weather_forecast": cleaned_weather_forecast,
    }
