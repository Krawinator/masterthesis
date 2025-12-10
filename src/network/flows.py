# src/network/flows.py

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.network.network_model import NetworkModel
from src.network.ptdf import compute_ptdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Zeitreihen laden
# ---------------------------------------------------------------------


def _load_single_series(
    node_id: str,
    meas_dir: Path,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    suffix: str = "_hist.csv",
) -> Optional[pd.Series]:
    """
    Lädt eine einzelne Zeitreihe für einen Knoten aus meas_dir.

    Erwartetes Format der CSV:
        - eine Spalte mit Zeitstempeln (ts_col, default 'timestamp')
        - eine Spalte mit Leistung in MW (val_col, default 'P_MW')

    Dateiname: <node_id><suffix>, z.B. 'BOLN_E01_hist.csv'.
    """
    fname = f"{node_id}{suffix}"
    fpath = meas_dir / fname

    if not fpath.exists():
        logger.warning("[LOAD] %s: Datei %s existiert nicht.", node_id, fname)
        return None

    df = pd.read_csv(fpath)
    logger.info("[LOAD] %s: %s | Spalten = %s", node_id, fname, list(df.columns))

    if ts_col not in df.columns or val_col not in df.columns:
        logger.warning(
            "[SKIP] %s: Benötigte Spalten '%s'/'%s' nicht gefunden.",
            node_id, ts_col, val_col
        )
        return None

    # Zeitstempel -> DatetimeIndex (naiv)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col, val_col]).copy()
    s = df.set_index(ts_col)[val_col].astype(float)
    # ggf. Zeitzone entfernen (naiv machen)
    s = s.tz_convert(None) if s.index.tz is not None else s
    s = s.sort_index()

    logger.info("[OK]   %s: %d Punkte.", node_id, len(s))
    return s


def load_timeseries_for_relevant_nodes(
    H_rel: pd.DataFrame,
    meas_dir: Path,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    suffix: str = "_hist.csv",
    min_common: int = 1,
) -> pd.DataFrame:
    """
    Lädt Zeitreihen für alle relevanten Knoten aus H_rel.columns
    und baut daraus eine gemeinsame Matrix P(t):

        df_P: Index = gemeinsame Zeitstempel
              Columns = relevante Knoten
              Werte = Leistung in MW

    Parameter
    ---------
    H_rel : pd.DataFrame
        PTDF-Matrix (Leitungen x relevante Knoten).
        Die Spaltennamen bestimmen, für welche Knoten Zeitreihen gesucht werden.
    meas_dir : Path
        Verzeichnis mit den *_hist.csv Dateien.
    ts_col, val_col : str
        Namen der Zeit- und Wertspalten in den CSVs.
    suffix : str
        Dateisuffix, z.B. '_hist.csv'.
    min_common : int
        Mindestanzahl gemeinsamer Zeitpunkte, sonst wird eine Warnung geloggt.

    Rückgabe
    --------
    df_P : pd.DataFrame
        Leistungszeitreihen in MW, gemeinsame Zeitbasis (inner join).
    """
    meas_dir = Path(meas_dir)
    logger.info("MEAS_DIR: %s", meas_dir)

    # nur zur Info: vorhandene Dateien auflisten
    if meas_dir.exists():
        files = sorted(p.name for p in meas_dir.glob("*.csv"))
        logger.info("Gefundene CSV-Dateien: %d", len(files))
        for fn in files:
            logger.info(" - %s", fn)
    else:
        logger.error("Messverzeichnis %s existiert nicht!", meas_dir)
        raise FileNotFoundError(meas_dir)

    relevant_nodes = list(H_rel.columns)
    logger.info("Relevante Knoten (für Zeitreihen-Laden): %s", relevant_nodes)

    series_dict = {}
    for nid in relevant_nodes:
        s = _load_single_series(
            node_id=nid,
            meas_dir=meas_dir,
            ts_col=ts_col,
            val_col=val_col,
            suffix=suffix,
        )
        if s is not None and not s.empty:
            series_dict[nid] = s

    if not series_dict:
        raise RuntimeError(
            "Keine Zeitreihen für relevante Knoten geladen. "
            "Prüfe MEAS_DIR, Dateinamen und Spaltennamen."
        )

    logger.info(
        "Knoten mit vorhandener Zeitreihe: %d\nNodes: %s",
        len(series_dict),
        list(series_dict.keys()),
    )

    # Gemeinsame Zeitbasis: Inner Join
    df_P = pd.concat(series_dict.values(), axis=1, join="inner")
    df_P.columns = list(series_dict.keys())

    logger.info(
        "---- Zeitreihen-Matrix df_P ----\n"
        "Gemeinsame Zeitpunkte: %d\n"
        "Knoten mit Messwerten: %d",
        len(df_P),
        df_P.shape[1],
    )

    if len(df_P) < min_common:
        logger.warning(
            "Wenige gemeinsame Zeitpunkte (n=%d < %d). "
            "PTDF-Auswertung könnte dünn belegt sein.",
            len(df_P),
            min_common,
        )

    return df_P


# ---------------------------------------------------------------------
# Flüsse aus PTDF und P(t)
# ---------------------------------------------------------------------


def compute_line_flows_from_ptdf(
    H_rel: pd.DataFrame,
    df_P: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet Leitungsflüsse F(t) aus:

        H_rel: PTDF (Leitungen x relevante Knoten)
        df_P : Zeitreihenmatrix P(t) in MW (Index: timestamp, Columns: Knoten)

    Mathematisch:
        Für jeden Zeitpunkt t:
            f(t) = H_rel @ P_rel(t)^T      (Vektor Länge L)

        Implementiert als:
            F = P_rel @ H_rel.T
        mit
            F: DataFrame (Index = Zeit, Columns = Leitungen)

    Annahmen:
        - df_P enthält Spalten für alle H_rel.columns.
        - fehlende Knoten werden via 0 MW ergänzt (falls df_P weniger Spalten hat).
    """
    # sicherstellen, dass alle relevanten Knoten-Spalten existieren
    missing = [c for c in H_rel.columns if c not in df_P.columns]
    if missing:
        logger.warning(
            "compute_line_flows_from_ptdf: folgende Knoten fehlen in df_P und werden mit 0 belegt: %s",
            missing,
        )
        # fehlende Spalten ergänzen
        for c in missing:
            df_P[c] = 0.0

    # Spalten von df_P in gleiche Reihenfolge bringen wie H_rel.columns
    df_P_aligned = df_P.loc[:, H_rel.columns].copy()

    # F = P @ H^T
    P_mat = df_P_aligned.values              # (T x N_relevant)
    H_mat = H_rel.values                     # (L x N_relevant)
    F_mat = P_mat @ H_mat.T                  # (T x L)

    df_F = pd.DataFrame(
        F_mat,
        index=df_P_aligned.index,            # Zeitstempel
        columns=H_rel.index,                 # Leitungs-IDs
    )

    logger.info(
        "compute_line_flows_from_ptdf: Flussmatrix gebaut: %s Zeitpunkte x %s Leitungen.",
        df_F.shape[0],
        df_F.shape[1],
    )

    # Optional: globale Einspeise-Summe loggen (nur Info)
    net_inj = df_P_aligned.sum(axis=1)
    logger.info(
        "Mittlere Netto-Einspeisung über alle Zeitpunkte: %.3f MW (Std=%.3f MW)",
        float(net_inj.mean()),
        float(net_inj.std()),
    )

    return df_F


# ---------------------------------------------------------------------
# End-to-End Helfer
# ---------------------------------------------------------------------


def build_flows_timeseries(
    net: NetworkModel,
    meas_dir: Path,
    relevant_only: bool = True,
    use_pinv: bool = True,
    ts_col: str = "timestamp",
    val_col: str = "P_MW",
    suffix: str = "_hist.csv",
    min_common: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-End-Helfer:

        1. PTDF berechnen (meist H_rel über relevante Knoten).
        2. Zeitreihen für diese Knoten laden → df_P.
        3. Leitungsflüsse F(t) in MW berechnen → df_F.

    Parameter
    ---------
    net : NetworkModel
        Netzmodell (inkl. Slack, A, B_rr, lines, nodes).
    meas_dir : Path
        Verzeichnis der *_hist.csv Dateien.
    relevant_only : bool
        True → verwende nur relevante Knoten (types aus config).
        False → verwende alle Nicht-Slack-Knoten.
    use_pinv : bool
        Ob bei der PTDF-Berechnung die Pseudoinverse benutzt wird.
    ts_col, val_col : str
        Zeit- und Wertspaltennamen in den CSV-Dateien.
    suffix : str
        Dateisuffix der Zeitreihen-Dateien.
    min_common : int
        Mindestanzahl gemeinsamer Zeitpunkte für df_P.

    Rückgabe
    --------
    df_F : pd.DataFrame
        Flüsse in MW: Index = Zeitstempel, Columns = Leitungs-IDs.
    H_rel : pd.DataFrame
        Verwendete PTDF-Matrix (Leitungen x relevante Knoten).
    df_P : pd.DataFrame
        Verwendete Leistungszeitreihen (Zeit x relevante Knoten, MW).
    """
    # 1) PTDF (relevant_only steuert, ob nur uw_field/battery o.ä.)
    H_rel = compute_ptdf(
        net,
        relevant_only=relevant_only,
        relevant_types=None,  # Standard aus config
        use_pinv=use_pinv,
    )

    # 2) Zeitreihen laden
    df_P = load_timeseries_for_relevant_nodes(
        H_rel=H_rel,
        meas_dir=meas_dir,
        ts_col=ts_col,
        val_col=val_col,
        suffix=suffix,
        min_common=min_common,
    )

    # 3) Flüsse berechnen
    df_F = compute_line_flows_from_ptdf(
        H_rel=H_rel,
        df_P=df_P,
    )

    return df_F, H_rel, df_P
