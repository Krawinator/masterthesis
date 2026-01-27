# src/network/bess_cleaning.py

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd

from src.config import RAW_TS_DIR, CLEAN_TS_DIR, GRAPH_PATH

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

    Returns:
      alpha: float
      beta: pd.Series (Index = X.columns)
      r2: float
      n_fit: int
    """
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha muss >= 0 sein.")

    df = pd.concat([X, y.rename("y")], axis=1, join="inner").dropna()
    n_fit = int(len(df))
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
    reg *= float(ridge_alpha)

    coef = np.linalg.solve(A.T @ A + reg, A.T @ yv)

    alpha = float(coef[0])
    beta = pd.Series(coef[1:], index=X.columns, dtype=float)

    y_hat = alpha + Xv @ beta.to_numpy()
    ss_res = float(np.sum((yv - y_hat) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return alpha, beta, r2, n_fit

def _load_battery_node_ids_from_graph(graph_path: Path) -> List[str]:
    """
    Liest whole_graph.json (Cytoscape-Style) und gibt node_ids zurück, deren type == "battery" ist.
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"GRAPH_PATH nicht gefunden: {graph_path.resolve()}")

    data = json.loads(graph_path.read_text(encoding="utf-8"))
    battery_ids: List[str] = []

    for it in data:
        d = it.get("data", {})
        if "source" in d and "target" in d:
            continue
        nid = d.get("id")
        ntype = (d.get("type") or "").strip()
        if nid and ntype == "battery":
            battery_ids.append(str(nid))

    battery_ids = sorted(set(battery_ids))
    return battery_ids


def _battery_hist_filenames(raw_dir: Path, battery_node_ids: List[str]) -> List[str]:
    """
    Mappt battery node_ids -> <node_id>_hist.csv und filtert auf existierende Dateien.
    """
    files = []
    missing = []
    for nid in battery_node_ids:
        name = f"{nid}_hist.csv"
        p = raw_dir / name
        if p.exists():
            files.append(name)
        else:
            missing.append(name)

    if missing:
        logger.warning("BESS-Dateien fehlen in meas_dir (werden ignoriert): %s", ", ".join(missing))

    return files

def remove_bess_effects_from_csv(
    bess_file: str,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    min_overlap_points: int = 100,
    out_dir: Optional[str] = None,
    ridge_alpha: float = 1.0,
) -> Dict[str, pd.Series]:
    """
    Wrapper für genau 1 BESS-Datei.
    """
    clean = remove_bess_effects_from_csv_multi(
        bess_files=[bess_file],
        ts_col=ts_col,
        val_col=val_col,
        min_overlap_points=min_overlap_points,
        out_dir=out_dir,  
        ridge_alpha=ridge_alpha,
    )
    return clean


def remove_bess_effects_from_csv_multi(
    bess_files: List[str],
    meas_dir: Optional[Path] = None,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    min_overlap_points: int = 200,
    out_dir: Optional[str] = None,
    include_intercept_in_removal: bool = False,
    ridge_alpha: float = 1.0,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    Entfernt den BESS-Einfluss aus allen *_hist.csv Zeitreihen in meas_dir
    mit multivariater Ridge Regression über mehrere BESS-Serien gleichzeitig.

    Beim Schreiben wird die komplette ursprüngliche CSV (inkl. Wetterspalten) übernommen.
    Es wird nur val_col (P_MW) ersetzt.

    Returns:
      (clean_series_dict, report_df)
    """
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha muss >= 0 sein.")

    meas_dir = Path(RAW_TS_DIR) if meas_dir is None else Path(meas_dir)
    if not meas_dir.exists():
        raise FileNotFoundError(f"meas_dir existiert nicht: {meas_dir.resolve()}")
    logger.info("BESS-clean Input-Verzeichnis (meas_dir): %s", meas_dir.resolve())

    if not bess_files:
        raise ValueError("bess_files ist leer. Bitte Liste mit BESS *_hist.csv Dateinamen übergeben.")

    out_path = Path(CLEAN_TS_DIR) if out_dir is None else Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("BESS-clean Output-Verzeichnis: %s", out_path.resolve())

    #  BESS Matrix X laden (nur val_col)
    bess_series: Dict[str, pd.Series] = {}
    for bf in bess_files:
        p = meas_dir / bf
        if not p.exists():
            raise FileNotFoundError(f"BESS-Datei nicht gefunden: {p.resolve()}")

        dfb = pd.read_csv(p, parse_dates=[ts_col]).sort_values(ts_col).set_index(ts_col)

        if val_col not in dfb.columns:
            raise ValueError(f"{p.name} enthält keine Spalte {val_col!r}. Vorhanden: {list(dfb.columns)}")

        bess_series[bf] = dfb[val_col].astype(float).rename(bf)

    X_all = pd.concat(bess_series.values(), axis=1).sort_index()
    logger.info(
        "BESS-Matrix geladen: n=%d, shape=%s, ridge_alpha=%.6f",
        len(bess_series),
        tuple(X_all.shape),
        float(ridge_alpha),
    )

    clean_series: Dict[str, pd.Series] = {}
    report_rows: List[Dict[str, Any]] = []
    bess_file_set = set(bess_files)

    written = 0
    skipped_no_fit = 0

    for csv_path in sorted(meas_dir.glob("*_hist.csv")):
        # BESS-Dateien selbst überspringen
        if csv_path.name in bess_file_set:
            continue

        node_id = csv_path.stem.replace("_hist", "")

        df_full = pd.read_csv(csv_path, parse_dates=[ts_col]).sort_values(ts_col).set_index(ts_col)

        if val_col not in df_full.columns:
            logger.warning("Skip %s: keine Spalte %s", csv_path.name, val_col)
            continue

        y = df_full[val_col].astype(float)

        # Fit im Overlap
        alpha, beta_vec, r2, n_fit = fit_multi_ridge_regression(
            X=X_all,
            y=y,
            ridge_alpha=ridge_alpha,
        )
        if n_fit < int(min_overlap_points):
            skipped_no_fit += 1
            continue

        # Removal auf kompletter Node-Achse
        X_on_idx = X_all.reindex(y.index).fillna(0.0)

        removed = X_on_idx @ beta_vec
        if include_intercept_in_removal:
            removed = removed + float(alpha)

        y_clean = y - removed
        clean_series[node_id] = y_clean

        # Diagnose
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
            "BESS-clean %s: n_fit=%d r2=%.3f corr(y,removed)=%.3f |beta|_sum=%.3f",
            node_id,
            int(n_fit),
            float(r2) if pd.notna(r2) else -1.0,
            float(corr_y_removed) if pd.notna(corr_y_removed) else 0.0,
            float(beta_vec.abs().sum()),
        )

        # komplette DF speichern, nur P_MW ersetzen
        df_out = df_full.copy()
        df_out[val_col] = y_clean

        out_file = out_path / f"{node_id}_hist_clean.csv"
        df_out.to_csv(out_file, index_label=ts_col)
        written += 1

    report_df = pd.DataFrame(report_rows)
    if not report_df.empty and "corr_y_removed" in report_df.columns:
        report_df = report_df.sort_values(
            "corr_y_removed",
            key=lambda c: c.abs(),
            ascending=False,
        ).reset_index(drop=True)

    logger.info(
        "BESS-Bereinigung fertig. written=%d skipped_low_overlap=%d report_rows=%d",
        written,
        skipped_no_fit,
        int(len(report_df)),
    )

    return clean_series, report_df


def clean_all_nodes_remove_bess_from_graph(
    *,
    graph_path: Optional[Path] = None,
    raw_ts_dir: Optional[Path] = None,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    min_overlap_points: int = 200,
    out_dir: Optional[str] = None,
    include_intercept_in_removal: bool = False,
    ridge_alpha: float = 1.0,
    write_report_csv: bool = True,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    Convenience-Wrapper für die Pipeline:
    - Liest battery nodes aus whole_graph.json
    - nimmt deren *_hist.csv als Regressoren (BESS-Signale)
    - entfernt den Einfluss aus allen anderen *_hist.csv in meas_dir
    - schreibt die bereinigten CSVs nach CLEAN_TS_DIR (oder out_dir)

    Returns:
      (clean_series_dict, report_df)
    """
    gpath = Path(GRAPH_PATH) if graph_path is None else Path(graph_path)
    meas_dir = Path(RAW_TS_DIR) if raw_ts_dir is None else Path(raw_ts_dir)

    if not meas_dir.exists():
        raise FileNotFoundError(f"meas_dir existiert nicht: {meas_dir.resolve()}")

    battery_ids = _load_battery_node_ids_from_graph(gpath)
    if not battery_ids:
        raise RuntimeError(f"Keine battery nodes im Graph gefunden: {gpath.resolve()}")

    bess_files = _battery_hist_filenames(meas_dir, battery_ids)
    if not bess_files:
        raise RuntimeError(
            "Keine BESS *_hist.csv Dateien gefunden. Erwartet in meas_dir: "
            + ", ".join([f"{nid}_hist.csv" for nid in battery_ids])
        )

    logger.info("BESS nodes aus Graph: %d (files=%d)", len(battery_ids), len(bess_files))

    clean, report = remove_bess_effects_from_csv_multi(
        bess_files=bess_files,
        meas_dir=meas_dir,  
        ts_col=ts_col,
        val_col=val_col,
        min_overlap_points=min_overlap_points,
        out_dir=out_dir,
        include_intercept_in_removal=include_intercept_in_removal,
        ridge_alpha=ridge_alpha,
    )

    if write_report_csv:
        try:
            out_path = Path(CLEAN_TS_DIR) if out_dir is None else Path(out_dir)
            rep_path = out_path / "bess_clean_report.csv"
            report.to_csv(rep_path, index=False)
            logger.info("BESS-clean Report geschrieben: %s", rep_path.resolve())
        except Exception:
            logger.exception("Konnte BESS-clean Report nicht schreiben.")

    return clean, report
