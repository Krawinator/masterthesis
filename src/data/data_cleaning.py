# src/data/data_cleaning.py

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import pandas as pd

from src.config import FREQ, MAX_GAP_STEPS, WEATHER_COLS, PREP_TS_DIR

logger = logging.getLogger(__name__)


def _reindex_and_interp_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s

    s = s.sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=FREQ)

    # Reindex auf Raster
    s = s.reindex(full_index)

    # Micro-gap fill: nur kurze Lücken
    s = s.interpolate(
        method="time",
        limit=int(MAX_GAP_STEPS),
        limit_direction="both",
    )
    return s


def _reindex_and_interp_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_index)

    # Nur die Wetterspalten, die es wirklich gibt
    cols_present = [c for c in WEATHER_COLS if c in df.columns]
    for col in cols_present:
        df[col] = df[col].interpolate(
            method="time",
            limit=int(MAX_GAP_STEPS),
            limit_direction="both",
        )
    return df


def _max_nan_block_steps(s: pd.Series) -> int:
    if s is None or s.empty:
        return 0
    isna = s.isna()
    if not isna.any():
        return 0
    grp = (isna != isna.shift()).cumsum()
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


def _write_prepared_csvs(
    *,
    cleaned_measurements: Dict[str, pd.Series],
    cleaned_weather_hist: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Schreibt pro Node eine CSV nach PREP_TS_DIR/<node_id>_hist.csv
    Inhalt: P_MW + Wetterspalten (sofern vorhanden), Index -> timestamp Spalte.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for node_id, s in cleaned_measurements.items():
        if s is None or s.empty:
            continue

        df_out = s.rename("P_MW").to_frame()

        w = cleaned_weather_hist.get(node_id)
        if w is not None and not w.empty:
            # join auf gleichem Raster; lässt NaNs stehen, wenn Wetter fehlt
            df_out = df_out.join(w, how="left")

        df_out = df_out.sort_index()
        df_out.index.name = "timestamp"

        path = out_dir / f"{node_id}_hist.csv"
        df_out.to_csv(path)
    logger.info("Prepared CSVs geschrieben nach: %s", out_dir.resolve())


def clean_data(data: Dict[str, Any], *, write_prepared: bool = True) -> Dict[str, Any]:
    """
    Reindex + Micro-gap interpolation (MAX_GAP_STEPS) für:
      - measurements (P_MW)
      - weather_hist

    Optional: schreibt die prepared CSVs nach PREP_TS_DIR, damit BESS-cleaning
    damit arbeiten kann.
    """
    nodes = data["nodes"]

    cleaned_measurements: Dict[str, pd.Series] = {}
    for node_id, s in (data.get("measurements") or {}).items():
        cleaned_measurements[node_id] = _reindex_and_interp_series(s)

    cleaned_weather_hist: Dict[str, pd.DataFrame] = {}
    for node_id, df in (data.get("weather_hist") or {}).items():
        cleaned_weather_hist[node_id] = _reindex_and_interp_weather(df)

    cleaned_weather_forecast: Dict[str, pd.DataFrame] = {}
    for node_id, df in (data.get("weather_forecast") or {}).items():
        cleaned_weather_forecast[node_id] = df.sort_index() if df is not None else df

    # Coverage report
    try:
        rows = _build_coverage_rows(cleaned_measurements, cleaned_weather_hist)
        _write_coverage_report(rows, Path("logs") / "data_coverage_report.csv")
    except Exception:
        logger.exception("Konnte Coverage-Report nicht schreiben.")

    # Prepared CSVs für die weiteren Schritte
    if write_prepared:
        try:
            _write_prepared_csvs(
                cleaned_measurements=cleaned_measurements,
                cleaned_weather_hist=cleaned_weather_hist,
                out_dir=Path(PREP_TS_DIR),
            )
        except Exception:
            logger.exception("Konnte prepared CSVs nicht schreiben.")

    return {
        "nodes": nodes,
        "measurements": cleaned_measurements,
        "weather_hist": cleaned_weather_hist,
        "weather_forecast": cleaned_weather_forecast,
    }
