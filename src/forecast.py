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
    PRED_NORMALIZED_DIR,
)

logger = logging.getLogger(__name__)

# -------------------------
# Constants
# -------------------------
TIMESTAMP_COL_DEFAULT = "timestamp"
TARGET_COL_DEFAULT = "P_MW"
PRED_COL_DEFAULT = "P_MW_pred" 
PRED_NORM_TS_DIR = PRED_NORMALIZED_DIR
CLIP_Q_LO = 0.01
CLIP_Q_HI = 0.99
K_CHECK = 8


# -------------------------
# Meta object
# -------------------------
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


# -------------------------
# Feature building helpers
# -------------------------
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


# -------------------------
# Normalization helpers (aus Notebook übernommen)
# -------------------------
def _hist_quantiles(hist: pd.Series, q_lo: float = CLIP_Q_LO, q_hi: float = CLIP_Q_HI) -> tuple[float, float]:
    arr = np.asarray(hist.to_numpy(), dtype=float)
    lo = float(np.quantile(arr, q_lo))
    hi = float(np.quantile(arr, q_hi))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _score_candidate(pred: np.ndarray, hist: pd.Series, k_check: int = K_CHECK) -> float:
    """
    Kleiner = besser.
    Heuristik: (1) Nähe zum letzten Hist-Level in den ersten K Steps,
               (2) Größenordnung vs. Hist-IQR
    """
    if len(pred) == 0:
        return float("inf")

    last = float(hist.iloc[-1])
    k = min(k_check, len(pred))
    head = pred[:k]

    level_err = float(np.mean(np.abs(head - last)))

    h = np.asarray(hist.to_numpy(), dtype=float)
    q25, q75 = np.quantile(h, [0.25, 0.75])
    iqr = max(float(q75 - q25), 1e-6)
    z = float(np.mean(np.abs((head - np.median(h)) / iqr)))

    return float(level_err + 0.3 * z)


def _normalize_pred_series(
    *,
    y_pred: np.ndarray,
    hist: pd.Series,
) -> dict:
    """
    Returns dict with:
      pred_raw, pred_norm, used_flip, shift, hist_lo, hist_hi
    """
    pred_raw = np.asarray(y_pred, dtype=float)

    # Candidate A: as is
    pred_a = pred_raw

    # Candidate B: flipped
    pred_b = -pred_raw

    sa = _score_candidate(pred_a, hist)
    sb = _score_candidate(pred_b, hist)
    used_flip = bool(sb < sa)
    pred_use = pred_b if used_flip else pred_a

    # Level alignment: pred[0] -> last_hist
    last_hist = float(hist.iloc[-1])
    first_pred = float(pred_use[0])
    shift = float(last_hist - first_pred)
    pred_aligned = pred_use + shift

    # Clip to hist quantiles
    hist_lo, hist_hi = _hist_quantiles(hist)
    pred_norm = np.clip(pred_aligned, hist_lo, hist_hi)

    return {
        "pred_raw": pred_raw,
        "pred_norm": pred_norm,
        "used_flip": used_flip,
        "shift": shift,
        "hist_lo": hist_lo,
        "hist_hi": hist_hi,
    }


def _read_hist_clean(node_id: str, meta: WinnerMeta) -> pd.DataFrame:
    hist_path = Path(CLEAN_TS_DIR) / f"{node_id}_hist_clean.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history file: {hist_path}")

    hist = pd.read_csv(hist_path)
    if meta.timestamp_col not in hist.columns or meta.target_col not in hist.columns:
        raise ValueError(
            f"{hist_path.name} missing required cols: {meta.timestamp_col}, {meta.target_col}"
        )

    hist[meta.timestamp_col] = pd.to_datetime(hist[meta.timestamp_col], utc=True, errors="coerce")

    hist = (
        hist.dropna(subset=[meta.timestamp_col, meta.target_col])   # <- wichtig
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


# -------------------------
# Core forecasting
# -------------------------
def load_winner() -> tuple[object, WinnerMeta]:
    if not WINNER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Winner model not found: {WINNER_MODEL_PATH}")
    if not WINNER_META_PATH.exists():
        raise FileNotFoundError(f"Winner meta not found: {WINNER_META_PATH}")

    model = load(WINNER_MODEL_PATH)
    meta = WinnerMeta.from_json(WINNER_META_PATH)

    logger.info(
        "Loaded winner model: approach=%s, model=%s, H=%d",
        meta.winner_approach,
        meta.winner_model_name,
        meta.H,
    )
    return model, meta


def forecast_one_node(
    node_id: str,
    *,
    model: object,
    meta: WinnerMeta,
    overwrite: bool = True,
    max_hours_cap: Optional[float] = None,
) -> Path:
    """
    Writes raw preds:
      PRED_TS_DIR/<node_id>_pred.csv

    Writes normalized preds:
      src/data/pred_normalized/<node_id>_pred.csv

    Both overwritten if overwrite=True.
    """
    hist_df = _read_hist_clean(node_id, meta)
    last_hist_ts = hist_df[meta.timestamp_col].iloc[-1]

    wf = _read_weather_forecast(node_id, meta)

    if max_hours_cap is not None:
        cutoff = last_hist_ts + pd.Timedelta(hours=float(max_hours_cap))
        wf = wf[wf[meta.timestamp_col] <= cutoff].copy().reset_index(drop=True)

    if wf.empty:
        raise ValueError(f"{node_id}: weather forecast empty after cap/filtering.")

    lag_vec = _build_lag_vector_from_history(hist_df, meta.target_col, meta.target_lags)

    use_steps = min(len(wf), meta.H)
    wf_use = wf.iloc[:use_steps].copy().reset_index(drop=True)
    horizons = np.arange(1, use_steps + 1, dtype=int)

    X = _make_direct_X_for_horizons(wf_use, meta, lag_vec, horizons)
    y_pred = np.asarray(model.predict(X), dtype=float)

    # --- NEW: normalize (flip-check + level-align + quantile-clip) ---
    hist_s = hist_df[meta.target_col].astype(float)
    norm = _normalize_pred_series(y_pred=y_pred, hist=hist_s)

    # --- RAW output ---
    out_raw = pd.DataFrame(
        {
            meta.timestamp_col: wf_use[meta.timestamp_col],
            "P_MW_pred": norm["pred_raw"],
        }
    )

    PRED_TS_DIR.mkdir(parents=True, exist_ok=True)
    out_path_raw = Path(PRED_TS_DIR) / f"{node_id}_pred.csv"
    if out_path_raw.exists() and not overwrite:
        raise FileExistsError(f"{out_path_raw} exists and overwrite=False")
    out_raw.to_csv(out_path_raw, index=False)

    # --- NORMALIZED output (separater Ordner) ---
    out_norm = pd.DataFrame(
        {
            meta.timestamp_col: wf_use[meta.timestamp_col],
            "P_MW_pred_raw": norm["pred_raw"],
            "used_flip": int(norm["used_flip"]),
            "shift_added_MW": float(norm["shift"]),
            "hist_q01": float(norm["hist_lo"]),
            "hist_q99": float(norm["hist_hi"]),
            "P_MW_pred_norm": norm["pred_norm"],
        }
    )

    PRED_NORM_TS_DIR.mkdir(parents=True, exist_ok=True)
    out_path_norm = PRED_NORM_TS_DIR / f"{node_id}_pred.csv"
    if out_path_norm.exists() and not overwrite:
        raise FileExistsError(f"{out_path_norm} exists and overwrite=False")
    out_norm.to_csv(out_path_norm, index=False)

    # optional: kurze Logzeile je Node
    logger.info(
        "Forecast+Norm OK for %s | flip=%s shift=%.3f | raw=[%.3f..%.3f] norm=[%.3f..%.3f] | out_norm=%s",
        node_id,
        norm["used_flip"],
        norm["shift"],
        float(np.min(norm["pred_raw"])),
        float(np.max(norm["pred_raw"])),
        float(np.min(norm["pred_norm"])),
        float(np.max(norm["pred_norm"])),
        str(out_path_norm),
    )

    # Rückgabe: RAW path (damit dein Batch summary gleich bleibt)
    return out_path_raw


def forecast_all_nodes(
    *,
    overwrite: bool = True,
    max_hours_cap: Optional[float] = None,
) -> pd.DataFrame:
    model, meta = load_winner()

    clean_dir = Path(CLEAN_TS_DIR)
    files = sorted(clean_dir.glob("*_hist_clean.csv"))

    logger.info(
        "Forecast batch start: nodes=%d, overwrite=%s, max_hours_cap=%s, clean_dir=%s, weather_dir=%s, out_dir=%s, out_norm_dir=%s",
        len(files),
        overwrite,
        max_hours_cap,
        str(clean_dir),
        str(WEATHER_FORECAST_DIR),
        str(PRED_TS_DIR),
        str(PRED_NORM_TS_DIR),
    )

    rows = []
    for p in files:
        node_id = p.stem.replace("_hist_clean", "")
        try:
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

    ok_n = int((df["ok"] == True).sum())
    fail_n = int((df["ok"] == False).sum())
    logger.info("Forecast batch done: ok=%d, fail=%d", ok_n, fail_n)

    return df
