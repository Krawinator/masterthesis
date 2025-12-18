# src/network/bess_cleaning.py

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from src.config import RAW_TS_DIR, CLEAN_TS_DIR

logger = logging.getLogger(__name__)


def fit_multi_ridge_regression(
    X: pd.DataFrame,
    y: pd.Series,
    ridge_alpha: float,
) -> Tuple[float, pd.Series, float, int]:
    """
    Multivariate Ridge Regression: y ≈ alpha + X @ beta

    Minimiert:
      ||y - (alpha + X beta)||^2 + ridge_alpha * ||beta||^2

    WICHTIG:
      - Intercept (alpha) wird NICHT regularisiert.
      - beta wird regularisiert (L2).

    Returns:
      alpha: float
      beta: pd.Series (Index = X.columns)
      r2: float
      n_fit: int
    """
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha muss >= 0 sein.")

    df = pd.concat([X, y.rename("y")], axis=1, join="inner").dropna()
    n_fit = len(df)
    if n_fit == 0:
        return 0.0, pd.Series(0.0, index=X.columns, dtype=float), np.nan, 0

    Xv = df[X.columns].to_numpy(dtype=float)
    yv = df["y"].to_numpy(dtype=float)

    # Intercept hinzufügen
    A = np.column_stack([np.ones(n_fit), Xv])  # (n_fit, 1+p)
    p = A.shape[1]

    # Regularisierung: Intercept nicht bestrafen
    reg = np.eye(p, dtype=float)
    reg[0, 0] = 0.0
    reg *= ridge_alpha

    coef = np.linalg.solve(A.T @ A + reg, A.T @ yv)

    alpha = float(coef[0])
    beta = pd.Series(coef[1:], index=X.columns, dtype=float)

    y_hat = alpha + Xv @ beta.to_numpy()
    ss_res = float(np.sum((yv - y_hat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return alpha, beta, r2, n_fit


def remove_bess_effects_from_csv(
    bess_file: str,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    min_overlap_points: int = 100,
    out_dir: Optional[str] = None,
    ridge_alpha: float = 1.0,
) -> Dict[str, pd.Series]:
    """
    Backwards-compatible Wrapper für genau 1 BESS-Datei.
    Nutzt Ridge (nicht OLS).
    """
    clean, _report = remove_bess_effects_from_csv_multi(
        bess_files=[bess_file],
        ts_col=ts_col,
        val_col=val_col,
        min_overlap_points=min_overlap_points,
        out_dir=out_dir,  # None => CLEAN_TS_DIR
        ridge_alpha=ridge_alpha,
    )
    return clean


def remove_bess_effects_from_csv_multi(
    bess_files: List[str],
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    min_overlap_points: int = 200,
    out_dir: Optional[str] = None,
    include_intercept_in_removal: bool = False,
    ridge_alpha: float = 1.0,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    Entfernt den BESS-Einfluss aus allen *_hist.csv Zeitreihen in RAW_TS_DIR
    mit multivariater Ridge Regression über mehrere BESS-Serien gleichzeitig.

    NEU (dein Wunsch):
      - Beim Schreiben werden NICHT nur timestamp + P_MW gespeichert,
        sondern die gesamte ursprüngliche CSV (inkl. Wetterspalten).
      - Es wird lediglich die Spalte `val_col` (P_MW) durch die bereinigte Version ersetzt.

    SPEICHERN:
      - Wenn out_dir=None -> speichert IMMER nach CLEAN_TS_DIR (config).
      - Wenn out_dir gesetzt -> speichert dorthin.

    Returns:
      (clean_series_dict, report_df)
    """
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha muss >= 0 sein.")

    meas_dir = Path(RAW_TS_DIR)
    if not meas_dir.exists():
        raise FileNotFoundError(f"RAW_TS_DIR existiert nicht: {meas_dir}")

    if not bess_files:
        raise ValueError("bess_files ist leer. Bitte Liste mit BESS *_hist.csv Dateinamen übergeben.")

    out_path = Path(CLEAN_TS_DIR) if out_dir is None else Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("BESS-clean Output-Verzeichnis: %s", out_path)

    # --- BESS Matrix X laden (nur die val_col Spalte nutzen, Wetterspalten ignorieren)
    bess_series: Dict[str, pd.Series] = {}
    for bf in bess_files:
        p = meas_dir / bf
        if not p.exists():
            raise FileNotFoundError(f"BESS-Datei nicht gefunden: {p}")

        dfb = pd.read_csv(p, parse_dates=[ts_col]).sort_values(ts_col).set_index(ts_col)

        if val_col not in dfb.columns:
            raise ValueError(f"{p} enthält keine Spalte {val_col!r}. Vorhanden: {list(dfb.columns)}")

        # Spaltenname der BESS-Serie = Dateiname (eindeutig)
        bess_series[bf] = dfb[val_col].astype(float).rename(bf)

    X_all = pd.concat(bess_series.values(), axis=1).sort_index()
    logger.info(
        "BESS-Matrix geladen: %d Serien, shape=%s, ridge_alpha=%.6f",
        len(bess_series),
        X_all.shape,
        ridge_alpha,
    )

    clean_series: Dict[str, pd.Series] = {}
    report_rows = []
    bess_file_set = set(bess_files)

    for csv_path in meas_dir.glob("*_hist.csv"):
        # BESS-Dateien selbst überspringen
        if csv_path.name in bess_file_set:
            continue

        node_id = csv_path.stem.replace("_hist", "")

        # WICHTIG: komplette CSV laden (inkl. Wetterspalten)
        df_full = pd.read_csv(csv_path, parse_dates=[ts_col]).sort_values(ts_col).set_index(ts_col)

        if val_col not in df_full.columns:
            logger.warning("Skip %s: keine Spalte %s", csv_path.name, val_col)
            continue

        y = df_full[val_col].astype(float)

        # --- Fit nur im Overlap (alle BESS + y)
        alpha, beta_vec, r2, n_fit = fit_multi_ridge_regression(
            X=X_all,
            y=y,
            ridge_alpha=ridge_alpha,
        )
        if n_fit < min_overlap_points:
            continue

        # --- Removal auf kompletter Node-Achse
        X_on_idx_raw = X_all.reindex(y.index)
        X_on_idx = X_on_idx_raw.fillna(0.0)

        removed = X_on_idx @ beta_vec
        if include_intercept_in_removal:
            removed = removed + alpha

        y_clean = y - removed
        clean_series[node_id] = y_clean

        # --- Diagnose
        df_diag = pd.concat([y.rename("y"), removed.rename("removed")], axis=1, join="inner").dropna()
        corr_y_removed = df_diag["y"].corr(df_diag["removed"]) if len(df_diag) >= 10 else np.nan

        report_rows.append(
            {
                "node_id": node_id,
                "n_fit": int(n_fit),
                "r2": float(r2) if pd.notna(r2) else np.nan,
                "corr_y_removed": float(corr_y_removed) if pd.notna(corr_y_removed) else np.nan,
                "beta_abs_sum": float(beta_vec.abs().sum()),
                "ridge_alpha": float(ridge_alpha),
                **{f"beta__{col}": float(beta_vec[col]) for col in beta_vec.index},
            }
        )

        logger.info(
            "BESS-clean(Ridge) %s: n_fit=%d r2=%.3f corr(y,removed)=%.3f |beta|_sum=%.3f",
            node_id,
            n_fit,
            r2 if pd.notna(r2) else -1.0,
            corr_y_removed if pd.notna(corr_y_removed) else 0.0,
            float(beta_vec.abs().sum()),
        )

        # --- GANZ WICHTIG: komplette DF speichern, nur P_MW ersetzen
        df_out = df_full.copy()
        df_out[val_col] = y_clean

        out_file = out_path / f"{node_id}_hist_clean.csv"
        df_out.to_csv(out_file, index_label=ts_col)

    report_df = pd.DataFrame(report_rows).sort_values(
        "corr_y_removed",
        key=lambda c: c.abs(),
        ascending=False,
    )

    logger.info(
        "BESS-Bereinigung (Ridge) fertig. Bereinigte Nodes: %d | Report-Zeilen: %d",
        len(clean_series),
        len(report_df),
    )

    return clean_series, report_df
