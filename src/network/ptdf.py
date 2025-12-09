# src/network/ptdf.py

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.network.network_model import NetworkModel
from src.config import RELEVANT_NODE_TYPES  # z.B. {"uw_field", "battery"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Hilfs-Checks für B_rr
# ---------------------------------------------------------------------


def check_B_rr_properties(net: NetworkModel, tol_rank: float = 1e-9) -> None:
    """
    Prüft grundlegende Eigenschaften von B_rr:
        - numerischer Rang
        - Konditionszahl

    Loggt Warnungen, falls etwas auffällig ist.

    Diese Funktion ändert nichts an den Daten und kann gefahrlos
    vor jeder PTDF-Berechnung aufgerufen werden.
    """
    B_rr = net.B_rr.values
    n = B_rr.shape[0]

    # Singulärwertzerlegung für Rang + Kondition
    u, s, vh = np.linalg.svd(B_rr)
    rank = int(np.sum(s > tol_rank))
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf

    logger.info(
        "B_rr-Check: Shape=%s, Rang=%d/%d, cond≈%.2e",
        B_rr.shape, rank, n, cond
    )

    if rank < n:
        logger.warning(
            "B_rr ist numerisch singulär (Rang=%d < %d). "
            "PTDF kann unzuverlässig sein.",
            rank, n
        )
    if cond > 1e8:
        logger.warning(
            "B_rr ist schlecht konditioniert (cond≈%.2e). "
            "PTDF-Werte könnten numerisch verrauscht sein.",
            cond
        )


# ---------------------------------------------------------------------
# PTDF-Berechnung
# ---------------------------------------------------------------------


def compute_ptdf_full(
    net: NetworkModel,
    use_pinv: bool = True,
) -> pd.DataFrame:
    """
    Berechnet die vollständige PTDF-Matrix für alle NICHT-Slack-Knoten.

    Mathematisch (DC-Lastfluss):
        F = diag(b) @ A_r @ B_rr^{-1} @ P_r

    mit
        - F:   Leitungsflüsse (Vektor Länge L)
        - b:   Suszeptanzen der Leitungen (1/x)
        - A_r: reduzierte Inzidenzmatrix (Zeilen = Leitungen, Spalten = Knoten ohne Slack)
        - B_rr: reduzierte Knoten-Suszeptanzmatrix (Knoten ohne Slack x Knoten ohne Slack)
        - P_r: Knoten-Einspeisungen (nur Nicht-Slack)

    Rückgabe:
        DataFrame H mit:
            Index   = Leitungs-IDs (wie net.lines.index)
            Columns = Knoten-IDs (Nicht-Slack, wie net.Ar.columns)
        H[ℓ, n] = PTDF: Fluss auf Leitung ℓ bei +1 MW Einspeisung in Knoten n
    """
    # Vorab: B_rr einmal prüfen
    check_B_rr_properties(net)

    # --- Eingaben aus dem Netzmodell ---
    lines = net.lines
    Ar = net.Ar
    B_rr = net.B_rr

    if "b" not in lines.columns:
        raise ValueError(
            "In net.lines fehlt die Spalte 'b'. "
            "Bitte sicherstellen, dass network_model die Suszeptanz berechnet."
        )

    # Leitungs-Suszeptanzen (1/x), Reihenfolge wie net.lines.index
    b_vec = lines["b"].astype(float).values  # shape (L,)
    L = b_vec.shape[0]

    # reduzierte Inzidenzmatrix (L x (N-1))
    A_r = Ar.values
    # reduzierte B-Matrix ((N-1) x (N-1))
    B_rr_mat = B_rr.values

    if B_rr_mat.shape[0] != B_rr_mat.shape[1]:
        raise ValueError(f"B_rr ist nicht quadratisch: shape={B_rr_mat.shape}")

    # --- Inverse von B_rr (oder Pseudoinverse, falls gewünscht) ---
    if use_pinv:
        B_rr_inv = np.linalg.pinv(B_rr_mat)
        logger.info("PTDF: B_rr wurde mit Pseudoinverse invertiert.")
    else:
        B_rr_inv = np.linalg.inv(B_rr_mat)
        logger.info("PTDF: B_rr wurde mit np.linalg.inv invertiert.")

    # diag(b) als (L x L)-Matrix
    B_ell = np.diag(b_vec)  # B_ell = diag(b_ℓ)

    # PTDF: H = diag(b) @ A_r @ B_rr^{-1}
    # shapes: (L x L) @ (L x (N-1)) @ ((N-1) x (N-1)) → (L x (N-1))
    H = B_ell @ A_r @ B_rr_inv

    # --- zurück als DataFrame ---
    H_df = pd.DataFrame(
        H,
        index=lines.index,      # Leitungs-IDs
        columns=Ar.columns,     # Nicht-Slack-Knoten
    )

    logger.info(
        "PTDF vollständig berechnet: %s (Leitungen) x %s (Nicht-Slack-Knoten).",
        H_df.shape[0],
        H_df.shape[1],
    )

    return H_df


def _get_relevant_node_ids(
    net: NetworkModel,
    relevant_types: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Ermittelt die Knoten-IDs, die als Einspeise-/Lastknoten für PTDF
    relevant sein sollen (z.B. nur 'uw_field' und 'battery').

    Achtung:
        - Slack-Knoten wird immer ausgeschlossen.
        - Es werden nur Knoten zurückgegeben, die in der PTDF-Matrix
          (also in net.Ar.columns) vorkommen.
    """
    if relevant_types is None:
        relevant_types = RELEVANT_NODE_TYPES

    relevant_types = set(relevant_types)

    if "type" not in net.nodes.columns:
        logger.warning(
            "In net.nodes fehlt die Spalte 'type' – alle Nicht-Slack-Knoten "
            "werden als 'relevant' behandelt."
        )
        # alle Nicht-Slack-Knoten, die in Ar vorkommen
        return [n for n in net.Ar.columns if n != net.slack_node]

    # Knoten mit gewünschtem type
    mask = net.nodes["type"].isin(relevant_types)
    candidate_nodes = net.nodes.index[mask].tolist()

    # nur Knoten, die auch in der reduzierten Matrix vorkommen (Nicht-Slack)
    relevant_nodes = [
        n
        for n in candidate_nodes
        if n in net.Ar.columns and n != net.slack_node
    ]

    logger.info(
        "Relevante Knoten (types=%s) für PTDF: %d gefunden.",
        ",".join(sorted(relevant_types)),
        len(relevant_nodes),
    )

    return relevant_nodes


def compute_ptdf_relevant(
    net: NetworkModel,
    relevant_types: Optional[Iterable[str]] = None,
    use_pinv: bool = True,
) -> pd.DataFrame:
    """
    PTDF-Matrix nur über den relevanten Knoten (z.B. 'uw_field' + 'battery').

    Vorgehen:
        1. Vollständige PTDF berechnen (alle Nicht-Slack-Knoten).
        2. Spalten auf Knoten mit type in `relevant_types` einschränken.

    Parameter
    ---------
    net : NetworkModel
        Dein Netzmodell aus `load_network_model(...)`.
    relevant_types : Iterable[str], optional
        Node-Typen, die als Einspeise-/Lastknoten interessant sind.
        Standard: RELEVANT_NODE_TYPES aus config (typischerweise {"uw_field", "battery"}).
    use_pinv : bool, default False
        Ob anstelle von np.linalg.inv die Pseudoinverse verwendet werden soll.

    Rückgabe
    --------
    H_rel : pd.DataFrame
        PTDF-Matrix mit:
            Index   = Leitungs-IDs
            Columns = relevante Knoten (ohne Slack)
    """
    # 1) Vollständige PTDF
    H_full = compute_ptdf_full(net, use_pinv=use_pinv)

    # 2) Relevante Knoten-IDs bestimmen
    relevant_nodes = _get_relevant_node_ids(net, relevant_types=relevant_types)

    if not relevant_nodes:
        logger.warning(
            "Keine relevanten Knoten für PTDF gefunden (relevant_types=%s). "
            "Gib ggf. andere Typen an oder prüfe net.nodes['type'].",
            relevant_types,
        )
        # leere Matrix mit richtigen Indizes
        return H_full.iloc[:, 0:0].copy()

    # 3) Spalten einschränken
    H_rel = H_full.loc[:, relevant_nodes].copy()

    logger.info(
        "PTDF für relevante Knoten berechnet: %s (Leitungen) x %s (relevante Knoten).",
        H_rel.shape[0],
        H_rel.shape[1],
    )

    return H_rel


def compute_ptdf(
    net: NetworkModel,
    relevant_only: bool = True,
    relevant_types: Optional[Iterable[str]] = None,
    use_pinv: bool = True,
) -> pd.DataFrame:
    """
    Wrapper:
        - relevant_only=True  → nur PTDF über relevante Knoten (Standardfall)
        - relevant_only=False → vollständige PTDF (alle Nicht-Slack-Knoten)
    """
    if relevant_only:
        return compute_ptdf_relevant(
            net,
            relevant_types=relevant_types,
            use_pinv=use_pinv,
        )
    else:
        return compute_ptdf_full(net, use_pinv=use_pinv)


# ---------------------------------------------------------------------
# KCL-Check für PTDF
# ---------------------------------------------------------------------


def check_ptdf_kcl(
    net: NetworkModel,
    H_full: pd.DataFrame,
    sample_nodes: Optional[int] = None,
    tol: float = 1e-8,
) -> None:
    """
    Prüft die PTDF-Matrix über einen KCL-Check.

    Idee:
        Für jede Spalte j von H_full (entspricht 1 MW-Injektion an Node j,
        Entnahme am Slack):
            - f_j = H_full[:, j]  → Leitungsflüsse
            - KCL auf reduzierter Knotenmengen: A_r^T f_j ≈ e_j

        A_r ist die Inzidenzmatrix ohne Slack-Spalte.
        Erwartung:
            max|A_r^T f_j - e_j| < tol

    Parameter:
        sample_nodes:
            - None oder 0 → alle Nicht-Slack-Knoten prüfen
            - int > 0     → zufällige Stichprobe von Spalten prüfen
    """
    non_slack_nodes = list(H_full.columns)

    # Reduzierte Inzidenzmatrix: Zeilen = Leitungen, Spalten = Nicht-Slack-Knoten
    A = net.A  # shape (L x N)
    missing_cols = [n for n in non_slack_nodes if n not in A.columns]
    if missing_cols:
        raise ValueError(f"Diese Nicht-Slack-Knoten fehlen in A: {missing_cols}")

    A_r = A.loc[:, non_slack_nodes].values  # (L x (N-1))

    # Welche Spalten prüfen?
    idxs = np.arange(len(non_slack_nodes))
    if sample_nodes and sample_nodes > 0 and sample_nodes < len(non_slack_nodes):
        rng = np.random.default_rng(42)
        idxs = rng.choice(idxs, size=sample_nodes, replace=False)

    max_mismatch = 0.0
    for j in idxs:
        node = non_slack_nodes[j]
        f = H_full.iloc[:, j].values  # (L,)

        # KCL auf reduzierter Knotenmengen: A_r^T f soll e_j sein
        kcl = A_r.T @ f  # (N-1,)

        e_j = np.zeros_like(kcl)
        e_j[j] = 1.0

        diff = kcl - e_j
        this_max = float(np.max(np.abs(diff)))
        max_mismatch = max(max_mismatch, this_max)

        if this_max > tol:
            logger.warning(
                "PTDF-KCL: Knoten %s (Spalte %d) verletzt KCL mit max-Abw.=%.3e (tol=%.1e).",
                node, j, this_max, tol
            )

    logger.info(
        "PTDF-KCL-Check: maximale Abweichung über geprüfte Knoten = %.3e",
        max_mismatch
    )

def flows_from_injections(
    net: NetworkModel,
    H: pd.DataFrame,
    injections: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wendet eine PTDF-Matrix H auf zeitabhängige Knoten-Einspeisungen an.

    Parameter
    ---------
    net : NetworkModel
        Netzmodell mit Slack-Knoten-Info.
    H : pd.DataFrame
        PTDF-Matrix mit:
            Index   = Leitungs-IDs
            Columns = Nicht-Slack-Knoten (z.B. nur relevante Knoten)
    injections : pd.DataFrame
        Zeitreihen der Einspeisungen/Lasten in MW:
            Index   = Zeitstempel
            Columns = Knoten-IDs
        Knoten, die nicht in H.columns vorkommen, werden ignoriert.
        Fehlende Knoten werden mit 0 MW gefüllt.

        WICHTIG:
            - Einspeisung > 0: Erzeugung
            - Einspeisung < 0: Last
            - Der Slack-Knoten darf enthalten sein, wird aber ignoriert,
              da sein Wert aus der Bilanz folgt.

    Rückgabe
    --------
    flows : pd.DataFrame
        Leitungsflüsse in MW:
            Index   = Zeitstempel (wie injections.index)
            Columns = Leitungs-IDs (wie H.index)
    """
    # Nur Nicht-Slack-Knoten, die in H vorkommen
    non_slack_nodes = list(H.columns)

    # Injections auf diese Knoten einschränken und fehlende mit 0 füllen
    P = injections.reindex(columns=non_slack_nodes, fill_value=0.0).copy()

    # -> (T x K) Matrix
    P_mat = P.values  # T Zeilen, K Knoten

    # PTDF-Matrix: (L x K)
    H_mat = H.values

    # Flüsse: F = P * H^T  (T x K) @ (K x L) = (T x L)
    F_mat = P_mat @ H_mat.T

    flows = pd.DataFrame(
        F_mat,
        index=P.index,       # Zeit
        columns=H.index,     # Leitungen
    )

    logger.info(
        "flows_from_injections: %s Zeitpunkte x %s Leitungen.",
        flows.shape[0],
        flows.shape[1],
    )

    return flows


# ---------------------------------------------------------------------
# Debug-Helfer für das Notebook
# ---------------------------------------------------------------------


def debug_single_injection(
    net: NetworkModel,
    H_full: pd.DataFrame,
    node_id: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Hilfsfunktion für das Notebook:
        - injiziere 1 MW an node_id,
        - entnehme 1 MW am Slack (implizit im DC-Modell),
        - gib die resultierenden Leitungsflüsse und Knotensummen zurück.

    Rückgabe:
        flows:    Serie (Index = Leitungs-IDs) in "MW pro 1 MW an node_id"
        kcl_vec:  Serie (Index = Nicht-Slack-Knoten) = A_r^T f
                  Erwartung: ≈ e_node_id
    """
    if node_id == net.slack_node:
        raise ValueError("debug_single_injection: node_id darf nicht der Slack-Knoten sein.")

    if node_id not in H_full.columns:
        raise ValueError(f"debug_single_injection: node_id {node_id} nicht in H_full.columns.")

    # 1 MW an node_id
    col_list = list(H_full.columns)
    j = col_list.index(node_id)
    f = H_full.iloc[:, j].copy()  # (L,)

    flows = pd.Series(f.values, index=H_full.index, name="flow_per_1MW")

    # KCL auf reduzierter Knotenmengen
    A_r = net.A.loc[:, col_list]  # (L x (N-1))
    kcl = A_r.T.values @ f.values
    kcl_s = pd.Series(kcl, index=col_list, name="kcl_balance")

    return flows, kcl_s
