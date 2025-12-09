# src/network/flows.py

import logging
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from src.network.network_model import NetworkModel
from src.network.ptdf import compute_ptdf
from src.config import RELEVANT_NODE_TYPES

logger = logging.getLogger(__name__)


def flows_from_injections(
    net: NetworkModel,
    injections: pd.DataFrame,
    H: Optional[pd.DataFrame] = None,
    relevant_only: bool = True,
    relevant_types: Optional[Iterable[str]] = None,
    use_pinv: bool = False,
) -> pd.DataFrame:
    """
    Berechnet Leitungsflüsse aus Knoten-Einspeisungen über PTDF.

    Mathematisch:
        F(t) = H @ P_r(t)
    mit
        - H:    PTDF (Leitungen x Nicht-Slack-Knoten)
        - P_r:  Einspeisungen an Nicht-Slack-Knoten (Slack-Aufnahme implizit)

    Parameter
    ---------
    net : NetworkModel
        Netzmodell (inkl. A, Bbus, Slack etc.).
    injections : pd.DataFrame
        Zeitreihen der Knoten-Einspeisungen in MW.
        index   = Zeitstempel
        columns = Knotennamen (z.B. "BOLS_E41", "JUBO_E01", ...)

        Es müssen nicht alle Knoten enthalten sein:
            - Nicht vorkommende PTDF-Knoten → werden als 0 MW behandelt
            - Zusätzliche Spalten, die nicht in der PTDF vorkommen → werden ignoriert

    H : pd.DataFrame, optional
        Bereits berechnte PTDF-Matrix.
        Wenn None:
            → wird über `compute_ptdf(...)` neu berechnet.

    relevant_only : bool, default True
        True  → PTDF nur über relevante Knoten (z.B. uw_field + battery).
        False → volle PTDF über alle Nicht-Slack-Knoten.

    relevant_types : Iterable[str], optional
        Welche Node-Typen als Einspeise-/Lastknoten relevant sind,
        falls `relevant_only=True`. Default = RELEVANT_NODE_TYPES.

    use_pinv : bool, default False
        Ob in der PTDF-Berechnung die Pseudoinverse genutzt werden soll.

    Rückgabe
    --------
    flows : pd.DataFrame
        Leitungsflüsse in MW:
            index   = Zeitstempel wie `injections.index`
            columns = Leitungs-IDs wie `net.lines.index`
    """
    # 1) PTDF besorgen (falls nicht übergeben)
    if H is None:
        H = compute_ptdf(
            net,
            relevant_only=relevant_only,
            relevant_types=relevant_types,
            use_pinv=use_pinv,
        )

    if H.empty:
        raise ValueError("flows_from_injections: PTDF-Matrix H ist leer.")

    # 2) Injections auf die PTDF-Knoten ausrichten
    #    - Spalten = H.columns (Nicht-Slack-Knoten, ggf. nur relevante)
    #    - Nicht vorhandene Spalten → 0
    #    - Extra-Spalten in injections → werden ignoriert
    inj_aligned = injections.reindex(columns=H.columns, fill_value=0.0).copy()

    # 3) Numerische Typen sicherstellen
    inj_aligned = inj_aligned.astype(float)

    # 4) Matrixmultiplikation:
    #    P: (T x N_r), H: (L x N_r) → F: (T x L)
    P = inj_aligned.values                # (T x N_r)
    H_mat = H.values                      # (L x N_r)

    # Flüsse: T x L
    F = P @ H_mat.T

    flows = pd.DataFrame(
        F,
        index=inj_aligned.index,
        columns=H.index,    # Leitungs-IDs
    )

    logger.info(
        "flows_from_injections: %d Zeitpunkte, %d Knoten → %d Leitungen.",
        flows.shape[0],
        inj_aligned.shape[1],
        flows.shape[1],
    )

    return flows
