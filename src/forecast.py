# src/forecast.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import load
from src.config import (
    CLEAN_TS_DIR,
    WEATHER_FORECAST_DIR,
    PRED_TS_DIR,
    WINNER_MODEL_PATH,
    WINNER_META_PATH,
    NODE_MODELS_DIR,
)

logger = logging.getLogger(__name__)

TIMESTAMP_COL_DEFAULT = "timestamp"
TARGET_COL_DEFAULT = "P_MW"

@dataclass(frozen=True)
class WinnerMeta:
    winner_approach: str
    winner_model_name: str
    H: int
    timestamp_col: str
    target_col: str
    raw_features: List[str]
    target_lags: List[int]
    fourier_k: int
    period_steps: int
    exog_cols: List[str]
    lag_cols: List[str]
    direct_h_features: List[str]

    @staticmethod
    def from_json(path: Path) -> "WinnerMeta":
        meta = json.loads(path.read_text(encoding="utf-8"))
        return WinnerMeta(
            winner_approach=str(meta.get("winner_approach", "DirectStacked")),
            winner_model_name=str(meta.get("winner_model_name", "UNKNOWN")),
            H=int(meta["H"]),
            timestamp_col=str(meta.get("timestamp_col", TIMESTAMP_COL_DEFAULT)),
            target_col=str(meta.get("target_col", TARGET_COL_DEFAULT)),
            raw_features=list(meta.get("raw_features", [])),
            target_lags=list(meta.get("target_lags", [])),
            fourier_k=int(meta.get("fourier_k", 3)),
            period_steps=int(meta.get("period_steps", 96)),
            exog_cols=list(meta.get("exog_cols", [])),
            lag_cols=list(meta.get("lag_cols", [])),
            direct_h_features=list(meta.get("direct_h_features", ["h", "h_over_H"])),
        )


def _ensure_dt(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Timestamp parse produced {bad} NaT values (check input format/timezone).")
    return ts


def _time_features(ts: pd.Series) -> pd.DataFrame:
    ts = _ensure_dt(ts)
    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)
    month = ts.dt.month.astype(int)

    out = pd.DataFrame(index=ts.index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def _fourier_day_features(ts: pd.Series, K: int, period_steps: int) -> pd.DataFrame:
    ts = _ensure_dt(ts)
    steps_per_hour = period_steps // 24
    minutes_per_step = 60 // steps_per_hour
    step_in_day = (ts.dt.hour * steps_per_hour + (ts.dt.minute // minutes_per_step)).astype(float)

    out = pd.DataFrame(index=ts.index)
    for k in range(1, K + 1):
        out[f"day_sin{k}"] = np.sin(2 * np.pi * k * step_in_day / period_steps)
        out[f"day_cos{k}"] = np.cos(2 * np.pi * k * step_in_day / period_steps)
    return out


def _build_lag_vector_from_history(hist_df: pd.DataFrame, target_col: str, lags: List[int]) -> np.ndarray:
    y = hist_df[target_col].astype(float).to_numpy()
    if len(y) < max(lags):
        raise ValueError(f"Not enough history for max lag={max(lags)} (have n={len(y)})")
    lag_vals = [float(y[-L]) for L in lags]
    return np.asarray(lag_vals, dtype=float)


def _make_direct_X_for_horizons(
    wf_df: pd.DataFrame,
    meta: WinnerMeta,
    lag_vec: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    ts = _ensure_dt(wf_df[meta.timestamp_col])
    base = pd.DataFrame(index=wf_df.index)

    for f in meta.raw_features:
        if f not in wf_df.columns:
            raise ValueError(f"Weather forecast missing required column: {f}")
        base[f] = wf_df[f].astype(float)

    base = pd.concat(
        [
            base,
            _time_features(ts),
            _fourier_day_features(ts, K=meta.fourier_k, period_steps=meta.period_steps),
        ],
        axis=1,
    )

    missing_exog = [c for c in meta.exog_cols if c not in base.columns]
    if missing_exog:
        raise ValueError(f"Cannot build exog_cols, missing: {missing_exog}")

    exog_mat = base[meta.exog_cols].astype(float).to_numpy()
    lag_mat = np.repeat(lag_vec.reshape(1, -1), repeats=len(horizons), axis=0)

    h = horizons.astype(float).reshape(-1, 1)
    h_over_H = (h / float(meta.H)).astype(float)

    X = np.concatenate([exog_mat, lag_mat, h, h_over_H], axis=1)
    return X


def _read_hist_clean(node_id: str, meta: WinnerMeta) -> pd.DataFrame:
    hist_path = Path(CLEAN_TS_DIR) / f"{node_id}_hist_clean.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history file: {hist_path}")

    hist = pd.read_csv(hist_path)
    if meta.timestamp_col not in hist.columns or meta.target_col not in hist.columns:
        raise ValueError(f"{hist_path.name} missing required cols: {meta.timestamp_col}, {meta.target_col}")

    hist[meta.timestamp_col] = pd.to_datetime(hist[meta.timestamp_col], utc=True, errors="coerce")

    hist = (
        hist.dropna(subset=[meta.timestamp_col, meta.target_col])
        .sort_values(meta.timestamp_col)
        .reset_index(drop=True)
    )

    if hist.empty:
        raise ValueError(f"{node_id}: history empty after dropping NA timestamp/target.")
    return hist


def _read_weather_forecast(node_id: str, meta: WinnerMeta) -> pd.DataFrame:
    wf_path = Path(WEATHER_FORECAST_DIR) / f"{node_id}_weather_forecast.csv"
    if not wf_path.exists():
        raise FileNotFoundError(f"Missing weather forecast file: {wf_path}")

    wf = pd.read_csv(wf_path)
    if meta.timestamp_col not in wf.columns:
        raise ValueError(f"{wf_path.name} missing timestamp col: {meta.timestamp_col}")

    wf[meta.timestamp_col] = pd.to_datetime(wf[meta.timestamp_col], utc=True, errors="coerce")
    wf = wf.sort_values(meta.timestamp_col).reset_index(drop=True)

    if wf.empty:
        raise ValueError(f"{node_id}: weather forecast file exists but is empty.")
    return wf

def load_meta_global() -> WinnerMeta:
    if not WINNER_META_PATH.exists():
        raise FileNotFoundError(f"Winner meta not found: {WINNER_META_PATH}")
    meta = WinnerMeta.from_json(WINNER_META_PATH)
    logger.info(
        "Loaded global meta: approach=%s, model=%s, H=%d",
        meta.winner_approach,
        meta.winner_model_name,
        meta.H,
    )
    return meta


def load_model_for_node(node_id: str) -> object:
    """Load the per-node model from NODE_MODELS_DIR/<node_id>/model.joblib."""

    base = Path(NODE_MODELS_DIR)
    if not base.exists():
        raise FileNotFoundError(f"Per-node model directory not found: {base}")

    model_path = base / node_id / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Per-node model not found: {model_path}")

    return load(model_path)



def load_model_for_node_or_fallback(node_id: str) -> object:
    """If a per-node model is missing, fall back to the global winner model."""
    try:
        model = load_model_for_node(node_id)
        return model
    except FileNotFoundError:
        if WINNER_MODEL_PATH.exists():
            logger.warning("No per-node model for %s. Falling back to global winner model.", node_id)
            return load(WINNER_MODEL_PATH)
        raise


def forecast_one_node(
    node_id: str,
    *,
    model: object,
    meta: WinnerMeta,
    overwrite: bool = True,
    max_hours_cap: Optional[float] = None,
) -> Path:
    """Write the day-ahead forecast to PRED_TS_DIR/<node_id>_pred.csv."""

    hist_df = _read_hist_clean(node_id, meta)
    last_hist_ts = hist_df[meta.timestamp_col].iloc[-1]

    wf = _read_weather_forecast(node_id, meta)

    if max_hours_cap is not None:
        cutoff = last_hist_ts + pd.Timedelta(hours=float(max_hours_cap))
        wf = wf[wf[meta.timestamp_col] <= cutoff].copy().reset_index(drop=True)

    if wf.empty:
        raise ValueError(f"{node_id}: weather forecast empty after cap/filtering.")

    lag_vec = _build_lag_vector_from_history(hist_df, meta.target_col, meta.target_lags)

    # Day-Ahead läuft maximal über H Schritte. Wenn Wetter kürzer ist, wird der Horizont entsprechend gekürzt.
    use_steps = min(len(wf), meta.H)
    wf_use = wf.iloc[:use_steps].copy().reset_index(drop=True)
    horizons = np.arange(1, use_steps + 1, dtype=int)

    X = _make_direct_X_for_horizons(wf_use, meta, lag_vec, horizons)
    y_pred = np.asarray(model.predict(X), dtype=float)


    # Output bleibt unskaliert; Forecast wird im gleichen Wertebereich wie P_MW gespeichert.
    out_raw = pd.DataFrame(
            {
                meta.timestamp_col: wf_use[meta.timestamp_col],
                "P_MW_pred": y_pred,
            }
    )

    PRED_TS_DIR.mkdir(parents=True, exist_ok=True)
    out_path_raw = Path(PRED_TS_DIR) / f"{node_id}_pred.csv"
    if out_path_raw.exists() and not overwrite:
        raise FileExistsError(f"{out_path_raw} exists and overwrite=False")
    out_raw.to_csv(out_path_raw, index=False)


    logger.info(
        "Forecast OK for %s | raw=[%.3f..%.3f] | out=%s",
        node_id,
        float(np.min(y_pred)),
        float(np.max(y_pred)),
        str(out_path_raw),
    )

    return out_path_raw


def forecast_all_nodes(
    *,
    overwrite: bool = True,
    max_hours_cap: Optional[float] = None,
    fallback_to_global_winner: bool = True,
) -> pd.DataFrame:
    """
    Batch-forecast all nodes for which a *_hist_clean.csv exists.
    Loads a per-node model for each node from NODE_MODELS_DIR.
    Optionally falls back to WINNER_MODEL_PATH if a per-node model is missing.
    """
    meta = load_meta_global()

    clean_dir = Path(CLEAN_TS_DIR)
    files = sorted(clean_dir.glob("*_hist_clean.csv"))

    logger.info(
        "Forecast batch start: nodes=%d, overwrite=%s, max_hours_cap=%s, clean_dir=%s, weather_dir=%s, out_dir=%s, NODE_MODELS_DIR=%s, fallback=%s",
        len(files),
        overwrite,
        max_hours_cap,
        str(clean_dir),
        str(WEATHER_FORECAST_DIR),
        str(PRED_TS_DIR),
        str(NODE_MODELS_DIR),
        fallback_to_global_winner,
    )

    rows = []
    for p in files:
        node_id = p.stem.replace("_hist_clean", "")
        try:
            if fallback_to_global_winner:
                model = load_model_for_node_or_fallback(node_id)
            else:
                model = load_model_for_node(node_id)

            pred_path = forecast_one_node(
                node_id,
                model=model,
                meta=meta,
                overwrite=overwrite,
                max_hours_cap=max_hours_cap,
            )
            n_pred = int(pd.read_csv(pred_path).shape[0])

            rows.append(
                {
                    "node_id": node_id,
                    "ok": True,
                    "pred_path": str(pred_path),
                    "n_pred": n_pred,
                    "error": "",
                }
            )
            logger.info("Forecast OK for %s → %s (n=%d)", node_id, pred_path, n_pred)

        except Exception:
            rows.append(
                {
                    "node_id": node_id,
                    "ok": False,
                    "pred_path": "",
                    "n_pred": 0,
                    "error": "see logs (stacktrace)",
                }
            )
            logger.exception("Forecast FAILED for %s", node_id)

    df = (
        pd.DataFrame(rows)
        .sort_values(["ok", "node_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    ok_n = int(df["ok"].sum())
    fail_n = int((~df["ok"]).sum())
    logger.info("Forecast batch done: ok=%d, fail=%d", ok_n, fail_n)

    return df
