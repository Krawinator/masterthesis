# src/data/data_cleaning.py

import logging
from typing import Dict, Any, List
from pathlib import Path

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


def _max_nan_block_steps(s: pd.Series) -> int:
    """
    Maximale Länge eines zusammenhängenden NaN-Blocks (in Zeitschritten).
    """
    if s.empty:
        return 0
    isna = s.isna()
    if not isna.any():
        return 0
    grp = (isna != isna.shift()).cumsum()
    # Für NaN-Blöcke ist isna==True, Summe innerhalb des Blocks = Blocklänge
    block_sizes = isna.groupby(grp).sum()
    block_sizes = block_sizes[block_sizes > 0]
    return int(block_sizes.max()) if not block_sizes.empty else 0


def _build_coverage_rows(
    cleaned_measurements: Dict[str, pd.Series],
    cleaned_weather_hist: Dict[str, pd.DataFrame],
) -> List[dict]:
    rows: List[dict] = []

    for node_id, s in cleaned_measurements.items():
        s = s.sort_index()
        n_total = int(len(s))
        n_nan = int(s.isna().sum())
        n_valid = int(n_total - n_nan)
        coverage = float(n_valid / n_total) if n_total > 0 else 0.0

        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()

        max_nan_steps = _max_nan_block_steps(s)

        # Wetter-Coverage nur grob: Anteil an Zeitschritten, wo wenigstens 1 Wetterspalte vorhanden ist
        w = cleaned_weather_hist.get(node_id)
        if w is None or w.empty:
            w_cov = 0.0
            w_any_nan = None
        else:
            w = w.sort_index()
            cols_present = [c for c in WEATHER_COLS if c in w.columns]
            if not cols_present:
                w_cov = 0.0
                w_any_nan = None
            else:
                any_present = w[cols_present].notna().any(axis=1)
                w_cov = float(any_present.mean())
                w_any_nan = int((~any_present).sum())

        rows.append(
            {
                "node_id": node_id,
                "freq": FREQ,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_nan": n_nan,
                "coverage_pct": round(100.0 * coverage, 3),
                "first_valid": first_valid,
                "last_valid": last_valid,
                "max_nan_block_steps": max_nan_steps,
                "max_nan_block_hours": round(max_nan_steps * pd.Timedelta(FREQ).total_seconds() / 3600.0, 3),
                "weather_any_present_coverage_pct": round(100.0 * w_cov, 3),
                "weather_any_missing_steps": w_any_nan,
            }
        )

    return rows


def _write_coverage_report(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["coverage_pct", "max_nan_block_steps"], ascending=[True, False])
    df.to_csv(out_path, index=False)
    logger.info("Data-Coverage-Report geschrieben: %s (rows=%d)", out_path, len(df))

    # Top 5 auffällige Nodes direkt ins Log
    if not df.empty:
        top = df.head(5)[["node_id", "coverage_pct", "first_valid", "last_valid", "max_nan_block_steps"]]
        logger.info("Auffällige Nodes (Top 5):\n%s", top.to_string(index=False))


def clean_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nimmt die Struktur von load_all_data und gibt eine bereinigte Kopie zurück.

    NEU:
      - Erstellt einen Coverage-/Betriebszeitraum-Report pro Node und speichert ihn als CSV:
        logs/data_coverage_report.csv
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

    # 3) Coverage-/Betriebszeitraum-Report schreiben
    try:
        rows = _build_coverage_rows(cleaned_measurements, cleaned_weather_hist)
        _write_coverage_report(rows, Path("logs") / "data_coverage_report.csv")
    except Exception as e:
        logger.warning("Konnte Coverage-Report nicht schreiben: %s", e)

    return {
        "nodes": nodes,
        "measurements": cleaned_measurements,
        "weather_hist": cleaned_weather_hist,
        "weather_forecast": cleaned_weather_forecast,
    }
