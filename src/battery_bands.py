# src/battery_bands.py
"""
- Lädt whole_graph.json
- Lädt pred/<node>_pred.csv (P_MW_pred)
- Kontraktion (X<=X_EPS_OHM) -> Supernetz
- PTDFs je Komponente (zeitinvariant)
- Für alle Timestamps:
    - Basecase DC-Loadflow (Slack-Korrektur je Komponente)
    - Bands je Battery (Injection min/max) unter Leitungs-UTIL_TARGET_PCT
- Speichert CSVs nach src/data/powerband/

Konventionen:
- Vorzeichen: Prognosen werden als Injektionen modelliert. Lasten werden dafür negiert, sodass
  positive Werte Einspeisung bedeuten.
- Das Leistungsband enthält immer 0 MW (Neutralbetrieb). Ist die zulässige Schnittmenge leer,
  wird [0, 0] ausgegeben.
- Wenn der Basecase (ohne Batterieaktion) bereits Leitungsgrenzen verletzt, ist der Zeitschritt
  unabhängig von der Batterie nicht zulässig.

Hinweis:
- run() kann weiterhin ohne Parameter in der Pipeline ausgeführt werden.
- Für das Validierungs-Notebook kann run() Debug-Daten zurückgeben und IO deaktivieren.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd

from src import config as cfg

logger = logging.getLogger(__name__)

TS_COL = "timestamp"
PRED_COL = "P_MW_pred"
PRED_SUFFIX = "_pred.csv"

FORECAST_NODE_TYPES = {"uw_field"}
BATTERY_TYPE = "battery"


# ==========================================================
# Helpers (bestehend)
# ==========================================================
def edge_raw_X_ohm(efeat: dict) -> Optional[float]:
    for k in ("X_total_ohm", "X_Total_Ohm"):
        if efeat.get(k) not in (None, ""):
            return float(efeat[k])
    return None


def ohm_to_pu(X_ohm: float, V_kV: float, S_base_MVA: float) -> float:
    Zb = (V_kV * 1e3) ** 2 / (S_base_MVA * 1e6)
    return X_ohm / Zb


def amp_to_mw_limit(I_A: float, V_kV: float, cosphi: float) -> float:
    U_V = V_kV * 1e3
    P_W = float(I_A) * np.sqrt(3.0) * U_V * cosphi
    return P_W / 1e6


class UF:
    def __init__(self, items: List[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def connected_components(
    nodes: List[str],
    lines: List[Tuple[str, str, float, str, dict]],
) -> List[List[str]]:
    idx = {n: i for i, n in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for (u, v, *_rest) in lines:
        ui, vi = idx[u], idx[v]
        adj[ui].append(vi)
        adj[vi].append(ui)

    seen = [False] * len(nodes)
    comps = []
    for i in range(len(nodes)):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp = []
        while q:
            x = q.popleft()
            comp.append(nodes[x])
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    q.append(y)
        comps.append(comp)
    return comps


def choose_slack_for_component(component_nodes: List[str], lines, fixed_slack: str) -> str:
    if fixed_slack in component_nodes:
        return fixed_slack
    deg = {n: 0 for n in component_nodes}
    for (u, v, *_rest) in lines:
        if u in deg and v in deg:
            deg[u] += 1
            deg[v] += 1
    return max(component_nodes, key=lambda n: deg[n])


def read_pred_series_for_node(pred_dir: Path, node_id: str, logger: logging.Logger) -> Optional[pd.Series]:
    path = pred_dir / f"{node_id}{PRED_SUFFIX}"
    if not path.is_file():
        return None

    df = pd.read_csv(path)
    if TS_COL not in df.columns or PRED_COL not in df.columns:
        logger.error("Forecast CSV hat nicht die erwarteten Spalten: %s", path)
        return None

    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL)
    s = pd.to_numeric(df[PRED_COL], errors="coerce").fillna(0.0)

    s.index = df[TS_COL]
    s.index.name = None
    s.name = node_id
    return s.sort_index()


# ==========================================================
# Bausteine (aus run() extrahiert, auch fürs Notebook nutzbar)
# ==========================================================
def load_graph(graph_path: Path) -> Tuple[List[str], List[dict], Dict[str, str], Dict[str, dict]]:
    with graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes, edges = [], []
    node_types, node_features = {}, {}

    for it in graph:
        if "data" not in it:
            continue
        d = it["data"]
        if "source" in d and "target" in d:
            edges.append(d)
        else:
            nid = d["id"]
            nodes.append(nid)
            node_types[nid] = (d.get("type") or "").strip()
            node_features[nid] = d.get("features", {}) or {}

    return nodes, edges, node_types, node_features


def contract_graph(
    nodes: List[str],
    edges: List[dict],
    X_EPS_OHM: float,
) -> Tuple[Dict[str, str], List[str], Dict[str, List[str]], List[Tuple[str, str, str, dict, float]], int, int]:

    """
    Kontrahiert Knoten über Kanten ohne/mit sehr kleiner Reaktanz (X<=eps).
    Alle übrigen Kanten bleiben als elektrische Leitungen erhalten.
    """

    uf = UF(nodes)
    electrical_edges: List[Tuple[str, str, str, dict, float]] = []
    to_contract = 0

    for e in edges:
        u, v = e["source"], e["target"]
        feat = e.get("features", {}) or {}
        eid = e.get("id") or feat.get("id") or f"{u}-{v}"
        Xohm = edge_raw_X_ohm(feat)

        if Xohm is None or float(Xohm) <= float(X_EPS_OHM):
            uf.union(u, v)
            to_contract += 1
        else:
            electrical_edges.append((u, v, eid, feat, float(Xohm)))

    rep_map = {nid: uf.find(nid) for nid in nodes}
    super_nodes = sorted(set(rep_map.values()))

    members_by_rep: Dict[str, List[str]] = defaultdict(list)
    for nid, rep in rep_map.items():
        members_by_rep[rep].append(nid)

    super_edges: List[Tuple[str, str, str, dict, float]] = []
    for (u, v, eid, feat, Xohm) in electrical_edges:
        ru, rv = rep_map[u], rep_map[v]
        if ru == rv:
            continue
        super_edges.append((ru, rv, eid, feat, float(Xohm)))

    return rep_map, super_nodes, dict(members_by_rep), super_edges, to_contract, len(electrical_edges)


def load_and_aggregate_forecasts(
    *,
    pred_dir: Path,
    nodes: List[str],
    node_types: Dict[str, str],
    members_by_rep: Dict[str, List[str]],
    super_nodes: List[str],
    battery_nodes: List[str],
    basecase_bess_p_mw: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    uw_fields = [n for n in nodes if node_types.get(n, "") in FORECAST_NODE_TYPES]

    series_by_node: Dict[str, pd.Series] = {}
    missing = []
    for nid in uw_fields:
        s = read_pred_series_for_node(pred_dir, nid, logger)
        if s is None:
            missing.append(nid)
        else:
            # SIGN FLIP: Lastprognosen als negative Injektion
            series_by_node[nid] = (-s)

    if not series_by_node:
        raise RuntimeError("Keine Forecast-Dateien für uw_field gefunden.")

    P_df = pd.concat(list(series_by_node.values()), axis=1).sort_index()

    # Fehlende Nodes auf 0 MW
    for nid in nodes:
        if nid not in P_df.columns:
            P_df[nid] = 0.0

    # Batterien im Basecase mit 0 MW
    for b in battery_nodes:
        P_df[b] = float(basecase_bess_p_mw)

    # Aggregation auf Supernodes
    for rep, members in members_by_rep.items():
        cols = [c for c in members if c in P_df.columns]
        P_df[rep] = P_df[cols].sum(axis=1) if cols else 0.0

    P_df = P_df[super_nodes].fillna(0.0)

    logger.info("Forecast ok: %d/%d (missing=%d)", len(series_by_node), len(uw_fields), len(missing))
    logger.info("Timestamps: %d (%s .. %s)", len(P_df), P_df.index.min(), P_df.index.max())

    return P_df


def build_B_and_lines_pu(
    *,
    super_nodes: List[str],
    super_edges: List[Tuple[str, str, str, dict, float]],
    V_kV_default: float,
    S_base_MVA: float,
) -> Tuple[np.ndarray, Dict[str, int], List[Tuple[str, str, float, str, dict]], Dict[str, dict]]:
    """ Erzeugt Leitungen in p.u. und die B-Matrix für den DC-Loadflow."""
    node_index = {nid: i for i, nid in enumerate(super_nodes)}
    n = len(super_nodes)
    B = np.zeros((n, n), dtype=float)

    lines: List[Tuple[str, str, float, str, dict]] = []
    edge_meta: Dict[str, dict] = {}

    for (ru, rv, eid, feat, Xohm) in super_edges:
        Xpu = ohm_to_pu(float(Xohm), V_kV_default, S_base_MVA)
        if Xpu <= 0:
            continue
        lines.append((ru, rv, float(Xpu), eid, feat))
        edge_meta[eid] = {
            "u": ru,
            "v": rv,
            "Xpu": float(Xpu),
            "limit_A": feat.get("Strom_Limit_in_A", None),
        }

    if not lines:
        raise RuntimeError("Keine gültigen Leitungen (Xpu>0) im Supernetz.")

    for (ru, rv, Xpu, eid, feat) in lines:
        i, j = node_index[ru], node_index[rv]
        b = 1.0 / Xpu
        B[i, j] -= b
        B[j, i] -= b
    for i in range(n):
        B[i, i] = -np.sum(B[i, :])

    return B, node_index, lines, edge_meta


def build_components_and_comp_data(
    *,
    super_nodes: List[str],
    lines: List[Tuple[str, str, float, str, dict]],
    B: np.ndarray,
    node_index: Dict[str, int],
    slack_fixed: str,
) -> Tuple[List[List[str]], Dict[frozenset, str], Dict[frozenset, dict]]:
    components = connected_components(super_nodes, lines)
    slack_by_comp = {frozenset(comp): choose_slack_for_component(comp, lines, fixed_slack=slack_fixed) for comp in components}

    comp_data: Dict[frozenset, dict] = {}
    for comp in components:
        ck = frozenset(comp)
        slack = slack_by_comp[ck]
        comp_idx = [node_index[x] for x in comp]
        B_sub = B[np.ix_(comp_idx, comp_idx)]

        non_slack = [x for x in comp if x != slack]
        if not non_slack:
            comp_data[ck] = {"slack": slack, "non_slack": [], "non_slack_idx": [], "B_rr": np.zeros((0, 0))}
            continue

        comp_to_local = {node_index[x]: i for i, x in enumerate(comp)}
        mask_local = [comp_to_local[node_index[x]] for x in non_slack]
        B_rr = B_sub[np.ix_(mask_local, mask_local)]

        comp_data[ck] = {
            "slack": slack,
            "non_slack": non_slack,
            "non_slack_idx": [node_index[x] for x in non_slack],
            "B_rr": B_rr,
        }

    return components, slack_by_comp, comp_data


def compute_ptdf_by_component(
    *,
    components: List[List[str]],
    comp_data: Dict[frozenset, dict],
    lines: List[Tuple[str, str, float, str, dict]],
) -> Dict[frozenset, dict]:
    PTDF_by_comp: Dict[frozenset, dict] = {}

    for comp in components:
        ck = frozenset(comp)
        slack = comp_data[ck]["slack"]
        non_slack = comp_data[ck]["non_slack"]
        k = len(non_slack)

        comp_lines = [(u, v, Xpu, eid, feat) for (u, v, Xpu, eid, feat) in lines if (u in comp and v in comp)]
        m = len(comp_lines)

        if m == 0 or k == 0:
            PTDF_by_comp[ck] = {
                "PTDF": np.zeros((m, k)),
                "line_ids": [x[3] for x in comp_lines],
                "pos": {},
                "lines": comp_lines,
            }
            continue

        pos = {node: i for i, node in enumerate(non_slack)}
        A_r = np.zeros((m, k), dtype=float)
        line_ids = []

        for ell, (u, v, Xpu, eid, feat) in enumerate(comp_lines):
            line_ids.append(eid)
            if u != slack:
                A_r[ell, pos[u]] = +1.0
            if v != slack:
                A_r[ell, pos[v]] = -1.0

        B_ell = np.diag([1.0 / Xpu for (_, _, Xpu, _, _) in comp_lines])
        B_rr = comp_data[ck]["B_rr"]

        M = np.linalg.solve(B_rr, A_r.T)
        PTDFm = B_ell @ M.T

        PTDF_by_comp[ck] = {"PTDF": PTDFm, "line_ids": line_ids, "pos": pos, "lines": comp_lines}

    return PTDF_by_comp


def build_component_cache(
    *,
    components: List[List[str]],
    comp_data: Dict[frozenset, dict],
    PTDF_by_comp: Dict[frozenset, dict],
    edge_meta: Dict[str, dict],
    node_index: Dict[str, int],
    V_kV_default: float,
    cosphi_min: float,
    util_scale: float,
) -> Dict[frozenset, dict]:
    comp_cache: Dict[frozenset, dict] = {}

    for comp in components:
        ck = frozenset(comp)
        slack = comp_data[ck]["slack"]
        non_slack = comp_data[ck]["non_slack"]
        non_slack_idx = comp_data[ck]["non_slack_idx"]
        B_rr = comp_data[ck]["B_rr"]

        comp_lines = PTDF_by_comp[ck]["lines"]
        line_ids = PTDF_by_comp[ck]["line_ids"]

        P_limit_eff: Dict[str, float] = {}
        for eid in line_ids:
            limA = edge_meta[eid]["limit_A"]
            if limA in (None, ""):
                continue
            limA = float(limA)
            if limA <= 0:
                continue
            P_limit_eff[eid] = amp_to_mw_limit(limA, V_kV_default, cosphi_min) * util_scale

        u_idx = (
            np.array([node_index[u] for (u, v, Xpu, eid, feat) in comp_lines], dtype=int)
            if comp_lines
            else np.array([], dtype=int)
        )
        v_idx = (
            np.array([node_index[v] for (u, v, Xpu, eid, feat) in comp_lines], dtype=int)
            if comp_lines
            else np.array([], dtype=int)
        )
        xpu = (
            np.array([Xpu for (u, v, Xpu, eid, feat) in comp_lines], dtype=float)
            if comp_lines
            else np.array([], dtype=float)
        )

        comp_cache[ck] = {
            "comp": comp,
            "slack": slack,
            "non_slack": non_slack,
            "non_slack_idx": non_slack_idx,
            "B_rr": B_rr,
            "line_ids": line_ids,
            "u_idx": u_idx,
            "v_idx": v_idx,
            "xpu": xpu,
            "P_limit_eff": P_limit_eff,
        }

    return comp_cache


def precompute_battery_infos(
    *,
    nodes: List[str],
    node_types: Dict[str, str],
    node_features: Dict[str, dict],
    rep_map: Dict[str, str],
    components: List[List[str]],
    PTDF_by_comp: Dict[frozenset, dict],
    comp_cache: Dict[frozenset, dict],
) -> Tuple[List[str], Dict[str, float], Dict[str, str], Dict[str, float], List[dict]]:
    battery_nodes = [n for n in nodes if node_types.get(n, "") == BATTERY_TYPE]
    if not battery_nodes:
        raise RuntimeError("Keine battery nodes im Graph gefunden.")

    battery_pmax = {b: float((node_features.get(b, {}) or {}).get("p_max_MW", np.nan)) for b in battery_nodes}
    battery_super = {b: rep_map[b] for b in battery_nodes}

    # Summiere p_max je Supernode (falls mehrere Batterien kontrahiert wurden)
    super_pmax = defaultdict(float)
    for b in battery_nodes:
        rep = battery_super[b]
        pmax = battery_pmax.get(b, np.nan)
        if np.isfinite(pmax):
            super_pmax[rep] += float(pmax)

    battery_infos = []
    for b in battery_nodes:
        bess_super = battery_super[b]

        comp = None
        for c in components:
            if bess_super in c:
                comp = c
                break
        if comp is None:
            logger.error("Battery %s supernode %s: keine Komponente gefunden", b, bess_super)
            continue

        ck = frozenset(comp)
        slack = comp_cache[ck]["slack"]

        pmax = float(super_pmax.get(bess_super, np.nan))
        if not np.isfinite(pmax):
            pmax = float(battery_pmax.get(b, np.nan))
        if not np.isfinite(pmax):
            pmax = 0.0

        pos = PTDF_by_comp[ck]["pos"]
        line_ids = PTDF_by_comp[ck]["line_ids"]
        PTDFm = PTDF_by_comp[ck]["PTDF"]

        if bess_super == slack or bess_super not in pos:
            dF = None
        else:
            dF = pd.Series(PTDFm[:, pos[bess_super]], index=line_ids, name="dF_dP")

        battery_infos.append({"battery": b, "supernode": bess_super, "ck": ck, "pmax": pmax, "dF": dF})

    return battery_nodes, battery_pmax, battery_super, dict(super_pmax), battery_infos


def solve_dc_flows_per_timestamp(
    *,
    components: List[List[str]],
    comp_cache: Dict[frozenset, dict],
    P_row: pd.Series,
    S_base_MVA: float,
    theta_global: np.ndarray,
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:

    """Berechnet DC-Loadflow-Flüsse für einen Timestamp und prüft Basecase-Grenzverletzungen."""

    flows_now: Dict[str, float] = {}

    # per component solve
    for comp in components:
        ck = frozenset(comp)
        cc = comp_cache[ck]

        slack = cc["slack"]
        non_slack = cc["non_slack"]
        non_slack_idx = cc["non_slack_idx"]
        B_rr = cc["B_rr"]

        P_comp = np.array([P_row[n] for n in comp], dtype=float)
        mismatch = float(P_comp.sum())
        if abs(mismatch) > 1e-9:
            P_row[slack] = P_row.get(slack, 0.0) - mismatch

        if len(non_slack) == 0:
            continue

        P_ns_MW = np.array([P_row[n] for n in non_slack], dtype=float)
        theta_ns = np.linalg.solve(B_rr, (P_ns_MW / S_base_MVA))

        for local_i, idx_g in enumerate(non_slack_idx):
            theta_global[idx_g] = theta_ns[local_i]

        if len(cc["line_ids"]) > 0:
            f_pu = (theta_global[cc["u_idx"]] - theta_global[cc["v_idx"]]) / cc["xpu"]
            f_MW = f_pu * S_base_MVA
            for eid, f in zip(cc["line_ids"], f_MW):
                flows_now[eid] = float(f)

    # basecase feasibility check
    basecase_violations: List[Tuple[str, float]] = []
    for comp in components:
        ck = frozenset(comp)
        P_limit_eff = comp_cache[ck]["P_limit_eff"]
        for eid, L in P_limit_eff.items():
            F0 = float(flows_now.get(eid, 0.0))
            if abs(F0) > float(L) + 1e-9:
                basecase_violations.append((eid, abs(F0) - float(L)))

    return flows_now, basecase_violations


# ==========================================================
# Main
# ==========================================================
PathLike = Union[str, Path]


def run(
    *,
    return_debug: bool = False,
    debug_max_timestamps: int = 1,
    graph_path_override: Optional[PathLike] = None,
    pred_dir_override: Optional[PathLike] = None,
    out_dir_override: Optional[PathLike] = None,
    disable_io: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Pipeline-Lauf:
    - Ohne Parameter: nutzt cfg.* Pfade, lädt Forecasts und schreibt die Outputs.

    Notebook/Validierung:
    - return_debug=True gibt interne Strukturen zurück (PTDF, Flows, etc.).
    - disable_io=True verhindert das Schreiben von CSVs.
    - *_override erlaubt das Überschreiben der Pfade (Graph/Pred/Output).
    """

    logger = logging.getLogger("powerband")
    logger.info("Start powerband run")

    # --- config params ---
    graph_path = Path(graph_path_override) if graph_path_override else Path(cfg.GRAPH_PATH)
    pred_dir = Path(pred_dir_override) if pred_dir_override else Path(cfg.PRED_TS_DIR)
    out_dir = Path(out_dir_override) if out_dir_override else Path(cfg.POWERBAND_DIR)

    X_EPS_OHM = float(cfg.X_EPS_OHM)
    S_BASE_MVA = float(cfg.S_BASE_MVA)
    V_KV_DEFAULT = float(cfg.V_KV_DEFAULT)
    COSPHI_MIN = float(cfg.COSPHI_MIN)
    BASECASE_BESS_P_MW = float(cfg.BASECASE_BESS_P_MW)
    SLACK = str(cfg.SLACK_NODE_ID)
    UTIL_TARGET_PCT = float(cfg.UTIL_TARGET_PCT)

    util_scale = UTIL_TARGET_PCT / 100.0
    if not (0.0 < util_scale <= 1.0):
        raise ValueError(f"UTIL_TARGET_PCT muss in (0,100] liegen, ist aber {UTIL_TARGET_PCT}")

    logger.warning("Config: UTIL_TARGET_PCT=%.1f%% (scale=%.3f)", UTIL_TARGET_PCT, util_scale)
    logger.info("Paths: graph=%s pred=%s out=%s disable_io=%s", graph_path, pred_dir, out_dir, disable_io)

    # --- basic existence checks ---
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph nicht gefunden: {graph_path.resolve()}")

    # only ensure out dir if we actually write
    if not disable_io:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # 1) Graph laden
    # ==========================================================
    nodes, edges, node_types, node_features = load_graph(graph_path)

    logger.info("Graph: nodes=%d edges=%d", len(nodes), len(edges))
    logger.info("Node-Type Counts: %s", pd.Series(list(node_types.values())).value_counts().to_dict())

    # ==========================================================
    # 2) Kontraktion -> Supernetz
    # ==========================================================
    rep_map, super_nodes, members_by_rep, super_edges, n_contracted, n_kept = contract_graph(nodes, edges, X_EPS_OHM)

    logger.info("Kontraktion: contracted=%d kept=%d", n_contracted, n_kept)
    logger.info("Supernetz: super_nodes=%d super_edges=%d", len(super_nodes), len(super_edges))

    # ==========================================================
    # 3) Forecasts laden -> P_df auf Supernodes
    # ==========================================================
    battery_nodes = [n for n in nodes if node_types.get(n, "") == BATTERY_TYPE]
    if not battery_nodes:
        logger.error("Keine battery nodes im Graph gefunden.")
        raise RuntimeError("Keine battery nodes im Graph gefunden.")

    P_df: pd.DataFrame
    require_forecast = not disable_io  # im produktiven Lauf sollen Forecasts Pflicht sein

    if pred_dir.is_dir():
        try:
            P_df = load_and_aggregate_forecasts(
                pred_dir=pred_dir,
                nodes=nodes,
                node_types=node_types,
                members_by_rep=members_by_rep,
                super_nodes=super_nodes,
                battery_nodes=battery_nodes,
                basecase_bess_p_mw=BASECASE_BESS_P_MW,
                logger=logger,
            )
        except Exception as e:
            if require_forecast:
                raise
            logger.warning("Forecast load failed in disable_io mode, building dummy P_df. err=%s", e)
            idx = pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")])
            P_df = pd.DataFrame(0.0, index=idx, columns=super_nodes)
    else:
        if require_forecast:
            raise FileNotFoundError(f"Pred dir nicht gefunden: {pred_dir.resolve()}")
        idx = pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")])
        P_df = pd.DataFrame(0.0, index=idx, columns=super_nodes)

    # ==========================================================
    # 4) Lines in p.u. + B-Matrix
    # ==========================================================
    B, node_index, lines, edge_meta = build_B_and_lines_pu(
        super_nodes=super_nodes,
        super_edges=super_edges,
        V_kV_default=V_KV_DEFAULT,
        S_base_MVA=S_BASE_MVA,
    )

    # ==========================================================
    # 5) Komponenten + Slack + comp_data
    # ==========================================================
    components, slack_by_comp, comp_data = build_components_and_comp_data(
        super_nodes=super_nodes,
        lines=lines,
        B=B,
        node_index=node_index,
        slack_fixed=SLACK,
    )

    # ==========================================================
    # 6) PTDF je Komponente
    # ==========================================================
    PTDF_by_comp = compute_ptdf_by_component(
        components=components,
        comp_data=comp_data,
        lines=lines,
    )

    # ==========================================================
    # 7) Cache pro Komponente (limits + vektorisierung)
    # ==========================================================
    comp_cache = build_component_cache(
        components=components,
        comp_data=comp_data,
        PTDF_by_comp=PTDF_by_comp,
        edge_meta=edge_meta,
        node_index=node_index,
        V_kV_default=V_KV_DEFAULT,
        cosphi_min=COSPHI_MIN,
        util_scale=util_scale,
    )

    # Battery infos (dF/dP once)
    _, battery_pmax, battery_super, super_pmax, battery_infos = precompute_battery_infos(
        nodes=nodes,
        node_types=node_types,
        node_features=node_features,
        rep_map=rep_map,
        components=components,
        PTDF_by_comp=PTDF_by_comp,
        comp_cache=comp_cache,
    )

    # ==========================================================
    # 8) Main loop over timestamps
    # ==========================================================
    theta_global = np.zeros(len(super_nodes), dtype=float)
    rows_util: List[dict] = []
    rows_bands: List[dict] = []

    P0_batt = float(BASECASE_BESS_P_MW)

    # optional: debug capture
    debug: Dict[str, Any] = {}
    debug_caps = 0

    for t in P_df.index:
        P_row = P_df.loc[t].copy()

        flows_now, basecase_violations = solve_dc_flows_per_timestamp(
            components=components,
            comp_cache=comp_cache,
            P_row=P_row,
            S_base_MVA=S_BASE_MVA,
            theta_global=theta_global,
        )

        basecase_infeasible = len(basecase_violations) > 0
        basecase_bottleneck = None
        if basecase_infeasible:
            basecase_violations.sort(key=lambda x: x[1], reverse=True)
            basecase_bottleneck = basecase_violations[0][0]

        # util rows
        for eid, fMW in flows_now.items():
            limA = edge_meta[eid]["limit_A"]
            util = np.nan
            if limA not in (None, ""):
                limA = float(limA)
                if limA > 0:
                    I_A = abs(fMW) * 1e6 / (np.sqrt(3.0) * (V_KV_DEFAULT * 1e3) * COSPHI_MIN)
                    util = abs(I_A) / (limA * util_scale) * 100.0
            rows_util.append({"timestamp": t, "edge_id": eid, "P_MW": fMW, "util_%": util})

        # battery bands
        for bi in battery_infos:
            b = bi["battery"]
            ck = bi["ck"]
            pmax = float(bi["pmax"])
            dF = bi["dF"]

            if basecase_infeasible:
                rows_bands.append({
                    "timestamp": t,
                    "battery": b,
                    "supernode": bi["supernode"],
                    "P_inj_min_MW": 0.0,
                    "P_inj_max_MW": 0.0,
                    "band_width_MW": 0.0,
                    "binding_plus": basecase_bottleneck,
                    "binding_minus": basecase_bottleneck,
                    "reason": "INFEASIBLE_BASECASE",
                })
                continue

            if dF is None or len(dF) == 0:
                rows_bands.append({
                    "timestamp": t,
                    "battery": b,
                    "supernode": bi["supernode"],
                    "P_inj_min_MW": np.nan,
                    "P_inj_max_MW": np.nan,
                    "band_width_MW": np.nan,
                    "binding_plus": None,
                    "binding_minus": None,
                    "reason": "NO_PTDF_COLUMN",
                })
                continue

            P_limit_eff = comp_cache[ck]["P_limit_eff"]

            rows = []
            for eid in dF.index:
                if eid not in P_limit_eff:
                    continue
                F0 = float(flows_now.get(eid, 0.0))
                s = float(dF.loc[eid])
                L = float(P_limit_eff[eid])
                if abs(s) < 1e-9:
                    continue
                lo = (-L - F0) / s
                hi = (L - F0) / s
                lo, hi = (min(lo, hi), max(lo, hi))
                rows.append((eid, lo, hi))

            if len(rows) == 0:
                d_lo, d_hi = -pmax, +pmax
                bind_plus, bind_minus = None, None
            else:
                dfc = pd.DataFrame(rows, columns=["edge_id", "d_lo", "d_hi"]).set_index("edge_id")
                d_lo = float(dfc["d_lo"].max())
                d_hi = float(dfc["d_hi"].min())
                d_lo = max(d_lo, -pmax)
                d_hi = min(d_hi, +pmax)
                bind_plus = dfc["d_hi"].idxmin()
                bind_minus = dfc["d_lo"].idxmax()

            Pmin_raw = max(P0_batt + d_lo, -pmax)
            Pmax_raw = min(P0_batt + d_hi, +pmax)

            # Neutralbetrieb erzwingen: 0 MW soll immer im Band enthalten sein
            Pmin = min(Pmin_raw, 0.0)
            Pmax = max(Pmax_raw, 0.0)

            if Pmin_raw > Pmax_raw:
                rows_bands.append({
                    "timestamp": t,
                    "battery": b,
                    "supernode": bi["supernode"],
                    "P_inj_min_MW": 0.0,
                    "P_inj_max_MW": 0.0,
                    "band_width_MW": 0.0,
                    "binding_plus": bind_plus,
                    "binding_minus": bind_minus,
                    "reason": "BASECASE_BINDING",
                })
            else:
                rows_bands.append({
                    "timestamp": t,
                    "battery": b,
                    "supernode": bi["supernode"],
                    "P_inj_min_MW": float(Pmin),
                    "P_inj_max_MW": float(Pmax),
                    "band_width_MW": float(Pmax - Pmin),
                    "binding_plus": bind_plus,
                    "binding_minus": bind_minus,
                    "reason": "",
                })

        # capture minimal debug for notebook
        if return_debug and debug_caps < int(debug_max_timestamps):
            debug_caps += 1
            debug.setdefault("timestamps", []).append(t)
            debug.setdefault("flows_now", {})[t] = dict(flows_now)
            debug.setdefault("basecase_violations", {})[t] = list(basecase_violations)

    util_df = pd.DataFrame(rows_util)
    bands_df = pd.DataFrame(rows_bands)

    # ==========================================================
    # IO: write outputs (optional)
    # ==========================================================
    if not disable_io:
        util_out = out_dir / "line_utilization.csv"
        util_df = util_df.sort_values(["timestamp", "edge_id"])
        util_df.to_csv(util_out, index=False)
        logger.info("Line utilization CSV geschrieben: %s (rows=%d)", util_out, len(util_df))

        # FINAL OUTPUT: eine CSV pro Batterie
        for battery_id, dfb in bands_df.groupby("battery"):
            out = (
                dfb[
                    [
                        "timestamp",
                        "P_inj_min_MW",
                        "P_inj_max_MW",
                        "band_width_MW",
                        "binding_plus",
                        "binding_minus",
                        "reason",
                    ]
                ]
                .rename(columns={"P_inj_min_MW": "P_min_MW", "P_inj_max_MW": "P_max_MW"})
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            EPS = 1e-6
            out["bottleneck_edge_id"] = None

            mask_inf = out["reason"].isin(["INFEASIBLE", "INFEASIBLE_BASECASE"])
            out.loc[mask_inf, "bottleneck_edge_id"] = (
                out.loc[mask_inf, "binding_plus"].fillna(out.loc[mask_inf, "binding_minus"])
            )

            mask_zero = (~mask_inf) & out["band_width_MW"].notna() & (out["band_width_MW"].abs() <= EPS)
            out.loc[mask_zero, "bottleneck_edge_id"] = (
                out.loc[mask_zero, "binding_plus"].fillna(out.loc[mask_zero, "binding_minus"])
            )

            out = out.drop(columns=["band_width_MW", "binding_plus", "binding_minus"])

            out_path = out_dir / f"{battery_id}_powerband.csv"
            out = out.copy()
            out["timestamp"] = (
                pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
                .dt.tz_convert(cfg.TIMEZONE)
                .dt.tz_localize(None)
            )
            out.to_csv(out_path, index=False)
            logger.info("Powerband CSV geschrieben: %s (rows=%d)", out_path, len(out))

    # ==========================================================
    # Debug return for notebook
    # ==========================================================
    if return_debug:
        # Debug-Ausgabe für das Notebook
        debug["graph_path"] = str(graph_path)
        debug["pred_dir"] = str(pred_dir)
        debug["out_dir"] = str(out_dir)
        debug["disable_io"] = bool(disable_io)

        debug["super_nodes"] = list(super_nodes)
        debug["lines"] = list(lines)
        debug["edge_meta"] = dict(edge_meta)
        debug["components"] = [list(c) for c in components]
        debug["slack_by_comp"] = {tuple(sorted(list(k))): v for k, v in slack_by_comp.items()}

        # PTDF im Debug mitgeben
        debug["PTDF_by_comp"] = {
            tuple(sorted(list(k))): {"line_ids": v["line_ids"], "pos": v["pos"], "PTDF": v["PTDF"]}
            for k, v in PTDF_by_comp.items()
        }

        debug["battery_infos"] = [
            {
                "battery": bi["battery"],
                "supernode": bi["supernode"],
                "ck": tuple(sorted(list(bi["ck"]))),
                "pmax": float(bi["pmax"]),
                "has_dF": (bi["dF"] is not None and len(bi["dF"]) > 0),
            }
            for bi in battery_infos
        ]

        debug["cfg"] = {
            "X_EPS_OHM": float(X_EPS_OHM),
            "S_BASE_MVA": float(S_BASE_MVA),
            "V_KV_DEFAULT": float(V_KV_DEFAULT),
            "COSPHI_MIN": float(COSPHI_MIN),
            "BASECASE_BESS_P_MW": float(BASECASE_BESS_P_MW),
            "SLACK_NODE_ID": str(SLACK),
            "UTIL_TARGET_PCT": float(UTIL_TARGET_PCT),
            "util_scale": float(util_scale),
        }

        # DataFrames zusätzlich mitgeben
        debug["util_df"] = util_df
        debug["bands_df"] = bands_df

        return debug

    return None


if __name__ == "__main__":
    run()
