# pip install dash dash-cytoscape
import dash
from dash import html, dcc, Input, Output, State, callback, ctx, no_update
import dash_cytoscape as cyto
import json
import base64

app = dash.Dash(__name__)
app.title = "Graph-Editor – Stromnetz"

# Startzustand: Cytoscape-Elemente (Nodes + Edges)
INITIAL_ELEMENTS: list[dict] = []


# =============================================================================
# Styling
# =============================================================================
def make_stylesheet(show_edge_labels: bool):
    """
    Cytoscape-Stylesheet. Kantenlabels optional ein-/ausblendbar.
    """

    return [
        {"selector": "node",
         "style": {"content": "data(label)", "text-valign": "center",
                   "background-color": "#999", "width": 40, "height": 40, "font-size": 11}},
        {"selector": "node[type = 'busbar']",
         "style": {"shape": "rectangle", "background-color": "#444", "width": 140, "height": 10,
                   "border-width": 1, "border-color": "#222", "text-valign": "top",
                   "text-margin-y": -10, "color": "#111", "font-size": 10}},
        {"selector": "node[type = 'battery']",
         "style": {"shape": "rectangle", "background-color": "#2ecc71", "width": 54, "height": 34,
                   "border-width": 2, "border-color": "#1b9e53"}},
        {"selector": "node[type = 'uw_field']",
         "style": {"shape": "round-rectangle", "background-color": "#e9f1ff",
                   "border-color": "#5b8def", "border-width": 1, "width": 34, "height": 26}},
        {"selector": "node[type = 'junction']",
         "style": {"shape": "ellipse", "background-color": "#bbb", "border-color": "#666",
                   "border-width": 1, "width": 18, "height": 18, "font-size": 9}},
        {"selector": "edge",
         "style": {"curve-style": "bezier", "label": "data(label)",
                   "text-opacity": 1 if show_edge_labels else 0, "font-size": 10}},
        {"selector": "edge:selected", "style": {"text-opacity": 1}},
        {"selector": ":selected", "style": {"border-width": 3, "border-color": "#333"}},
    ]


# =============================================================================
# Feature helpers (derived terms parsing)
# =============================================================================
def parse_formula_to_terms(s: str):
    """
    Liest "node:coeff, node:coeff, ..." und gibt Terms als Liste zurück.
    Ungültige Teile werden übersprungen.
    """

    terms = []
    if not s:
        return terms

    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        node, coeff = part.split(":", 1)
        node = node.strip()
        try:
            coeff = float(coeff.strip())
        except ValueError:
            continue
        if node:
            terms.append({"node": node, "coeff": coeff})
    return terms


def terms_to_string(terms):
    """
    Baut aus Terms wieder einen editierbaren String ("node:coeff, ...").
    """
    if not terms:
        return ""
    return ", ".join(
        f"{t.get('node','')}:{t.get('coeff','')}"
        for t in terms
        if t.get("node") is not None
    )


# =============================================================================
# Elements helpers
# =============================================================================
def split_elements(elems: list[dict]):
    """
    Trennt Elements in Nodes/Edges: Edge hat source+target, sonst Node.
    """
    elems = elems or []
    nodes, edges = [], []
    for x in elems:
        d = (x or {}).get("data", {}) or {}
        if "source" in d and "target" in d:
            edges.append(x)
        else:
            nodes.append(x)
    return nodes, edges


def node_ids_from(nodes):
    """Sammelt alle Node-IDs aus den Elementen."""
    out = set()
    for n in nodes:
        d = n.get("data", {}) or {}
        nid = d.get("id")
        if nid:
            out.add(nid)
    return out


def edge_ids_from(edges):
    """Sammelt alle Edge-IDs"""
    out = set()
    for e in edges:
        d = e.get("data", {}) or {}
        eid = d.get("id")
        if eid:
            out.add(eid)
    return out


def edge_key(e):
    """Key für Duplikat-Check: (source, target)."""
    d = e.get("data", {}) or {}
    return (d.get("source"), d.get("target"))


def safe_float(x):
    """Float-Parser: leer/ungültig -> None."""
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_valid_elements_list(obj):
    """
    Plausibilitätscheck für Import: Liste aus Elementen mit data{}.
    Node braucht id, Edge braucht source+target, features (falls da) ist dict.
    """

    if not isinstance(obj, list):
        return False, "JSON muss eine Liste von Cytoscape-Elementen sein."

    for i, el in enumerate(obj):
        if not isinstance(el, dict):
            return False, f"Element #{i} ist kein dict."
        if "data" not in el or not isinstance(el["data"], dict):
            return False, f"Element #{i} hat kein gültiges 'data'-dict."

        d = el["data"]
        is_edge = ("source" in d) or ("target" in d)

        if is_edge:
            if not d.get("source") or not d.get("target"):
                return False, f"Edge #{i} braucht 'source' und 'target'."
        else:
            if not d.get("id"):
                return False, f"Node #{i} braucht 'id'."

        if "features" in d and d["features"] is not None and not isinstance(d["features"], dict):
            return False, f"Element #{i}: 'features' muss ein dict sein."

    return True, ""


# =============================================================================
# Layout
# =============================================================================
app.layout = html.Div([
    # Mini-CSS: Vollhöhe + Dropdowns über Cytoscape (z-index).
    html.Script(
        """
        (function(){
          var style = document.createElement('style');
          style.innerHTML = `
            html, body, #_dash-app-content { height: 100%; margin: 0; padding: 0; }
            .Select-menu-outer, .VirtualizedSelectFocused .Select-menu-outer,
            .Select-menu, .Select, .rc-select-dropdown { z-index: 4000 !important; }
            .dash-dropdown, ._dash-dropdown { overflow: visible !important; }
          `;
          document.head.appendChild(style);
        })();
        """
    ),

    # Persistenter State für Export/Import (Elements-Liste).
    dcc.Store(id="gstore", data=INITIAL_ELEMENTS),

    # Merkt bis zu 2 angetippte Nodes für "Edge hinzufügen"
    dcc.Store(id="edge_selection", data=[]),

    html.Div([
        # Links: graph canvas
        html.Div([
            cyto.Cytoscape(
                id="graph",
                elements=INITIAL_ELEMENTS,
                layout={"name": "preset"},
                stylesheet=make_stylesheet(show_edge_labels=False),
                style={"width": "100%", "height": "100%"},
                userZoomingEnabled=True,
                userPanningEnabled=True,
                boxSelectionEnabled=True,
                autoungrabify=False,
            )
        ], style={
            "flex": "1 1 auto",
            "height": "100vh",
            "minWidth": "0",
            "boxSizing": "border-box",
            "padding": "12px"
        }),

        # Rechts: Werkzeug-Panel
        html.Div([
            html.H3("Werkzeuge"),

            # Statuszeile für Aktionen/Fehler
            html.Div(id="status_msg", style={
                "whiteSpace": "pre-wrap",
                "fontSize": 12,
                "padding": "8px",
                "borderRadius": "6px",
                "border": "1px solid #e5e7eb",
                "background": "#ffffff",
                "marginBottom": "10px"
            }),

            html.Div([
                dcc.Checklist(
                    id="edge_label_toggle",
                    options=[{"label": "Kanten-Beschriftungen anzeigen", "value": "show"}],
                    value=[],
                    inputStyle={"marginRight": "6px"}
                )
            ], style={"marginBottom": "10px"}),

            # --- Create Node ---
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Node-ID", style={"fontWeight": 600}),
                        dcc.Input(
                            id="new_node_id",
                            placeholder="z. B. A_B1, BESS_1, J1",
                            type="text",
                            style={"width": "100%"}
                        ),
                    ]),
                    html.Div([
                        html.Label("Label", style={"fontWeight": 600}),
                        dcc.Input(
                            id="new_node_label",
                            placeholder="Anzeigename (optional)",
                            type="text",
                            style={"width": "100%"}
                        ),
                    ]),
                ], style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "columnGap": "8px",
                    "alignItems": "start",
                    "marginBottom": "8px"
                }),

                html.Div([
                    html.Label("Node-Typ", style={"fontWeight": 600}),
                    dcc.Dropdown(
                        id="new_node_type",
                        options=[
                            {"label": "Sammelschiene", "value": "busbar"},
                            {"label": "Batteriespeicher", "value": "battery"},
                            {"label": "UW_Feld", "value": "uw_field"},
                            {"label": "Leitungsknoten", "value": "junction"},
                        ],
                        value="uw_field",
                    )
                ]),
                html.Button("Node hinzufügen", id="btn_add_node", style={"width": "180px"}),
            ], style={"marginBottom": 10}),

            # Löscht Auswahl; bei Node werden verbundene Kanten mit entfernt.
            html.Button("Ausgewählten Node/Edge löschen", id="btn_del_element",
                        style={"marginBottom": 12}),

            html.Hr(),
            html.H4("Umbenennen"),

            # Node umbenennen: ID/Label + Kanten-Endpunkte anpassen
            html.Div([
                html.Label("Node: Neue ID"),
                dcc.Input(id="rename_node_id"),
                html.Label("Node: Neues Label"),
                dcc.Input(id="rename_node_label"),
                html.Button("Node umbenennen", id="btn_rename_node")
            ], style={
                "display": "grid",
                "gridTemplateColumns": "200px 1fr",
                "gap": "6px",
                "marginBottom": "10px"
            }),

            # Edge umbenennen: ID/Label.
            html.Div([
                html.Label("Edge: Neue ID"),
                dcc.Input(id="rename_edge_id"),
                html.Label("Edge: Neues Label"),
                dcc.Input(id="rename_edge_label"),
                html.Button("Edge umbenennen", id="btn_rename_edge")
            ], style={
                "display": "grid",
                "gridTemplateColumns": "200px 1fr",
                "gap": "6px"
            }),

            html.Hr(),
            html.H4("Edge anlegen"),

            # Edge wird aus den letzten 2 getippten Nodes gebaut
            html.Div([
                dcc.Input(id="new_edge_id", placeholder="Edge-ID"),
                dcc.Input(id="new_edge_label", placeholder="Edge-Label", style={"marginLeft": 6}),
                html.Button("Edge hinzufügen (2 Nodes wählen)", id="btn_add_edge", style={"marginLeft": 6})
            ]),
            html.Div(id="edge_sel_info", style={"fontSize": 12, "marginTop": 4}),

            html.Hr(),
            html.H4("Auswahl"),
            html.Div(id="sel_summary", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),

            # ===== Node Fields =====
            html.H5("Node-Felder"),
            html.Div([
                html.Label("P_Datapoint_ID"),
                dcc.Input(id="n_P_Datapoint_ID"),

                html.Label("Q_Datapoint_ID"),
                dcc.Input(id="n_Q_Datapoint_ID"),

                html.Label("Strom_Limit_in_A"),
                dcc.Input(id="n_Strom_Limit_in_A", type="number"),

                html.Label("Windgeschw_Datapoint"),
                dcc.Input(id="n_Windgeschw_Datapoint"),

                html.Label("Globale-Strahlung_Datapoint"),
                dcc.Input(id="n_Globale_Strahlung_Datapoint"),

                html.Label("Aussentemp_Datapoint"),
                dcc.Input(id="n_Aussentemp_Datapoint"),

                html.Label("DAB_ID"),
                dcc.Input(id="n_DAB_ID"),

                html.Label("Node-Typ"),
                dcc.Dropdown(
                    id="n_type_dropdown",
                    options=[
                        {"label": "Sammelschiene", "value": "busbar"},
                        {"label": "Batteriespeicher", "value": "battery"},
                        {"label": "UW_Feld", "value": "uw_field"},
                        {"label": "Leitungsknoten", "value": "junction"},
                    ]
                ),

                html.Label("p_max_MW (nur Battery)"),
                dcc.Input(id="n_p_max_mw", type="number"),

                html.Label("Breitengrad (°)"),
                dcc.Input(id="n_lat", type="number", step="any"),

                html.Label("Längengrad (°)"),
                dcc.Input(id="n_lon", type="number", step="any"),

                # Optional formula string used for busbar relationships
                html.Label("Busbar-Formel (optional)"),
                dcc.Input(
                    id="n_busbar_formula",
                    placeholder="z. B. JUBO_E03:-1, JUBO_E02:-1",
                    style={"width": "100%"}
                ),

                # Derived feature configuration
                html.Label("Derived aktiv?"),
                dcc.Checklist(
                    id="n_derived_enable",
                    options=[{"label": "Dieses Feld wird aus anderen Feldern abgeleitet", "value": "on"}],
                    value=[]
                ),

                html.Label("Derived: Feature-Key"),
                dcc.Input(
                    id="n_derived_feature_key",
                    placeholder="z. B. P",
                    value="P",
                    style={"width": "100%"}
                ),

                html.Label("Derived: Terms (node:coeff, …)"),
                dcc.Textarea(
                    id="n_derived_terms",
                    placeholder="z. B. JUBO_E03:-1, JUBO_E02:-1",
                    style={"width": "100%", "height": "60px"}
                ),
            ], style={
                "display": "grid",
                "gridTemplateColumns": "220px 1fr",
                "gap": "6px"
            }),

            html.Button("Node-Felder speichern", id="btn_save_node_schema", style={"marginTop": 6}),

            # ===== Edge Fields =====
            html.H5("Edge-Felder"),
            html.Div([
                html.Label("Strom_Limit_in_A"),
                dcc.Input(id="e_Strom_Limit_in_A", type="number"),

                html.Label("Reaktanz X (Ω/km)"),
                dcc.Input(id="e_X_ohm_per_km", type="number", step="any"),

                html.Label("Länge (km)"),
                dcc.Input(id="e_length_km", type="number", step="any"),
            ], style={
                "display": "grid",
                "gridTemplateColumns": "220px 1fr",
                "gap": "6px"
            }),

            html.Button("Edge-Felder speichern", id="btn_save_edge_schema", style={"marginTop": 6}),

            html.Hr(),
            html.H4("Import / Export"),

            # Export: Elements-Liste als JSON.
            html.Button("Export JSON", id="btn_export"),
            dcc.Download(id="dl_json"),

            # Import: JSON mit Elements-Liste (Cytoscape-Format)
            dcc.Upload(
                id="upload_json",
                children=html.Div(["JSON importieren (drag & drop)"]),
                multiple=False,
                style={"border": "1px dashed #aaa", "padding": 8, "marginTop": 8}
            ),
            html.Div(id="upload_status"),
        ], style={
            "flex": "0 0 28%",
            "maxWidth": "460px",
            "height": "100vh",
            "overflowY": "auto",
            "backgroundColor": "#f5f7fa",
            "border": "1px solid #e5e7eb",
            "borderRadius": "8px",
            "padding": "12px",
        }),
    ], style={
        "display": "flex",
        "width": "100vw",
        "height": "100vh",
        "gap": "12px",
    })
])


# =============================================================================
# Callbacks
# =============================================================================

@callback(
    Output("graph", "stylesheet"),
    Input("edge_label_toggle", "value")
)
def toggle_edge_labels(values):
    """Schaltet Kantenlabels im Stylesheet um."""
    show = bool(values and "show" in values)
    return make_stylesheet(show_edge_labels=show)


@callback(
    Output("sel_summary", "children"),
    Output("n_P_Datapoint_ID", "value"),
    Output("n_Q_Datapoint_ID", "value"),
    Output("n_Strom_Limit_in_A", "value"),
    Output("n_Windgeschw_Datapoint", "value"),
    Output("n_Globale_Strahlung_Datapoint", "value"),
    Output("n_Aussentemp_Datapoint", "value"),
    Output("n_DAB_ID", "value"),
    Output("n_type_dropdown", "value"),
    Output("n_p_max_mw", "value"),
    Output("e_Strom_Limit_in_A", "value"),
    Output("rename_node_id", "value"),
    Output("rename_node_label", "value"),
    Output("rename_edge_id", "value"),
    Output("rename_edge_label", "value"),
    Output("n_lat", "value"),
    Output("n_lon", "value"),
    Output("e_X_ohm_per_km", "value"),
    Output("e_length_km", "value"),
    Output("n_busbar_formula", "value"),
    Output("n_derived_enable", "value"),
    Output("n_derived_feature_key", "value"),
    Output("n_derived_terms", "value"),
    Input("graph", "selectedNodeData"),
    Input("graph", "selectedEdgeData"),
)
def show_selection(node_data, edge_data):
    """Füllt die Eingabefelder basierend auf der aktuellen Auswahl."""
    if node_data:
        d = node_data[0]
        feats = d.get("features", {}) or {}
        summary = f"[Node] id={d['id']}  type={d.get('type')}\nfeatures={json.dumps(feats, indent=2)}"

        derived = feats.get("derived")
        if isinstance(derived, dict) and derived.get("method") == "field_sum":
            d_enabled = ["on"]
            d_feature = derived.get("feature_key", "P")
            d_terms_str = terms_to_string(derived.get("terms", []))
        else:
            d_enabled = []
            d_feature = "P"
            d_terms_str = ""

        return (
            summary,
            feats.get("P_Datapoint_ID", ""),
            feats.get("Q_Datapoint_ID", ""),
            feats.get("Strom_Limit_in_A", None),
            feats.get("Windgeschw_Datapoint", ""),
            feats.get("Globale_Strahlung_Datapoint", ""),
            feats.get("Aussentemp_Datapoint", ""),
            feats.get("DAB_ID", ""),
            d.get("type"),
            feats.get("p_max_MW", None),
            None,
            d["id"], d.get("label"),
            "", "",
            feats.get("Latitude_deg"), feats.get("Longitude_deg"),
            None, None,
            feats.get("busbar_id", ""),
            d_enabled,
            d_feature,
            d_terms_str,
        )

    if edge_data:
        d = edge_data[0]
        feats = d.get("features", {}) or {}
        summary = f"[Edge] id={d.get('id')} {d.get('source')}→{d.get('target')}\n{json.dumps(feats, indent=2)}"
        return (
            summary,
            "", "", None, "", "", "", "", None, None,
            feats.get("Strom_Limit_in_A", None),
            "", "",
            d.get("id"), d.get("label"),
            None, None,
            feats.get("X_ohm_per_km", None),
            feats.get("length_km", None),
            "", [], "P", ""
        )

    return ("Nichts ausgewählt.", "", "", None, "", "", "", "",
            None, None, None, "", "", "", "",
            None, None, None, None, "", [], "P", "")


@callback(
    Output("edge_selection", "data"),
    Output("edge_sel_info", "children"),
    Input("graph", "tapNodeData"),
    State("edge_selection", "data"),
)
def handle_tap_node(tapped, current):
    """Merkt sich bis zu zwei getippte Nodes für das Edge-Anlegen."""
    current = current or []
    if not tapped:
        return current, no_update

    nid = tapped.get("id")
    if not nid:
        return current, no_update

    cur = [n for n in current if n != nid]
    cur.append(nid)
    cur = cur[-2:]

    return cur, f"Gewählt für Edge: {cur}"


@callback(
    Output("graph", "elements"),
    Output("gstore", "data"),
    Output("status_msg", "children"),
    Output("edge_selection", "data", allow_duplicate=True),
    Input("btn_add_node", "n_clicks"),
    Input("btn_del_element", "n_clicks"),
    Input("btn_add_edge", "n_clicks"),
    Input("btn_save_node_schema", "n_clicks"),
    Input("btn_save_edge_schema", "n_clicks"),
    Input("btn_rename_node", "n_clicks"),
    Input("btn_rename_edge", "n_clicks"),
    Input("upload_json", "contents"),
    State("upload_json", "filename"),
    State("graph", "elements"),
    State("graph", "selectedNodeData"),
    State("graph", "selectedEdgeData"),

    State("new_node_id", "value"),
    State("new_node_label", "value"),
    State("new_node_type", "value"),

    State("edge_selection", "data"),
    State("new_edge_id", "value"),
    State("new_edge_label", "value"),

    State("n_P_Datapoint_ID", "value"),
    State("n_Q_Datapoint_ID", "value"),
    State("n_Strom_Limit_in_A", "value"),
    State("n_Windgeschw_Datapoint", "value"),
    State("n_Globale_Strahlung_Datapoint", "value"),
    State("n_Aussentemp_Datapoint", "value"),
    State("n_DAB_ID", "value"),
    State("n_type_dropdown", "value"),
    State("n_p_max_mw", "value"),
    State("n_lat", "value"),
    State("n_lon", "value"),

    State("e_Strom_Limit_in_A", "value"),
    State("e_X_ohm_per_km", "value"),
    State("e_length_km", "value"),

    State("rename_node_id", "value"),
    State("rename_node_label", "value"),
    State("rename_edge_id", "value"),
    State("rename_edge_label", "value"),

    State("n_busbar_formula", "value"),
    State("n_derived_enable", "value"),
    State("n_derived_feature_key", "value"),
    State("n_derived_terms", "value"),
    prevent_initial_call=True,
)
def mutate_graph(
    add_node, del_element, add_edge, save_node_schema, save_edge_schema,
    rename_node_clicks, rename_edge_clicks, upload_contents, upload_name,
    elems, selected_nodes, selected_edges,

    new_id, new_label, new_type,
    edge_sel, new_edge_id, new_edge_label,

    n_P, n_Q, n_Ilim, n_Wind, n_Glob, n_Tamb, n_DAB, n_type_sel,
    n_pmax_mw, n_lat, n_lon,

    e_Ilim, e_X_per_km, e_len_km,
    r_node_id, r_node_label, r_edge_id, r_edge_label,

    n_busbar_formula, n_derived_enable, n_derived_feature_key, n_derived_terms
):
    """
    Zentrale Callback-Funktion für alle Aktionen (add/del/rename/import/save).
    Gibt Elements + Status + Edge-Auswahl zurück.
    """
    trig = ctx.triggered_id
    elems = elems or []

    nodes, edges = split_elements(elems)
    node_ids = node_ids_from(nodes)
    edge_ids = edge_ids_from(edges)
    edge_pairs = {edge_key(e) for e in edges if edge_key(e) != (None, None)}

    status = "OK."
    new_edge_sel = edge_sel or []

    # -------------------------------------------------------------------------
    # Node anlegen
    # -------------------------------------------------------------------------
    if trig == "btn_add_node":
        nid = (new_id or "").strip()
        if not nid:
            return elems, elems, "Fehler: Node-ID fehlt.", new_edge_sel
        if nid in node_ids:
            return elems, elems, f"Fehler: Node-ID '{nid}' existiert bereits.", new_edge_sel

        ntype = (new_type or "uw_field").strip()
        label = (new_label or nid).strip() or nid

        nodes.append({
            "data": {"id": nid, "label": label, "type": ntype, "features": {}},
            "position": {"x": 50 * (len(nodes) + 1), "y": 50 * (len(nodes) + 1)}
        })

        status = f"Node '{nid}' hinzugefügt."
        new_elems = nodes + edges
        return new_elems, new_elems, status, new_edge_sel

    # -------------------------------------------------------------------------
    # Edge anlegen
    # -------------------------------------------------------------------------
    if trig == "btn_add_edge":
        eid = (new_edge_id or "").strip()
        if not eid:
            return elems, elems, "Fehler: Edge-ID fehlt.", new_edge_sel
        if eid in edge_ids:
            return elems, elems, f"Fehler: Edge-ID '{eid}' existiert bereits.", new_edge_sel

        if not new_edge_sel or len(new_edge_sel) != 2:
            return elems, elems, "Fehler: Bitte genau 2 Nodes für die Edge auswählen (anklicken).", new_edge_sel

        s, t = new_edge_sel
        if s == t:
            return elems, elems, "Fehler: Self-loop ist nicht erlaubt (source == target).", new_edge_sel
        if s not in node_ids or t not in node_ids:
            return elems, elems, "Fehler: source/target Node existiert nicht (Import/State inkonsistent).", new_edge_sel
        if (s, t) in edge_pairs:
            return elems, elems, f"Fehler: Edge {s}→{t} existiert bereits.", new_edge_sel

        edges.append({
            "data": {
                "id": eid,
                "source": s,
                "target": t,
                "label": (new_edge_label or eid).strip() or eid,
                "features": {}
            }
        })

        status = f"Edge '{eid}' ({s}→{t}) hinzugefügt."
        new_elems = nodes + edges
        return new_elems, new_elems, status, []  # resets edge_selection after creation

    # -------------------------------------------------------------------------
    # Delete selected node/edge
    # -------------------------------------------------------------------------
    if trig == "btn_del_element":
        # Node deletion removes all incident edges
        if selected_nodes:
            nid = selected_nodes[0].get("id")
            if not nid:
                return elems, elems, "Fehler: Kein gültiger Node selektiert.", new_edge_sel

            nodes = [n for n in nodes if (n.get("data", {}) or {}).get("id") != nid]
            edges = [
                e for e in edges
                if (e.get("data", {}) or {}).get("source") != nid
                and (e.get("data", {}) or {}).get("target") != nid
            ]

            status = f"Node '{nid}' (inkl. verbundener Edges) gelöscht."
            new_elems = nodes + edges
            return new_elems, new_elems, status, [x for x in new_edge_sel if x != nid]

        # Edge deletion removes only the selected edge
        if selected_edges:
            sel = selected_edges[0]
            sid = sel.get("id")
            s = sel.get("source")
            t = sel.get("target")

            def keep_edge(ed):
                d = ed.get("data", {}) or {}
                if sid is not None:
                    return d.get("id") != sid
                return not (d.get("source") == s and d.get("target") == t)

            edges2 = [ed for ed in edges if keep_edge(ed)]
            if len(edges2) == len(edges):
                return elems, elems, "Hinweis: Konnte selektierte Edge nicht finden (State inkonsistent).", new_edge_sel

            status = "Edge gelöscht."
            new_elems = nodes + edges2
            return new_elems, new_elems, status, new_edge_sel

        return elems, elems, "Hinweis: Nichts selektiert.", new_edge_sel

    # -------------------------------------------------------------------------
    # Save node fields (features + type updates)
    # -------------------------------------------------------------------------
    if trig == "btn_save_node_schema":
        if not selected_nodes:
            return elems, elems, "Fehler: Bitte erst einen Node auswählen.", new_edge_sel

        nid = selected_nodes[0].get("id")
        if not nid:
            return elems, elems, "Fehler: Ungültige Node-Auswahl.", new_edge_sel

        for n in nodes:
            if (n.get("data", {}) or {}).get("id") == nid:
                feats = (n["data"].get("features", {}) or {}).copy()

                # Standard fields
                if n_P not in (None, ""):
                    feats["P_Datapoint_ID"] = str(n_P).strip()
                if n_Q not in (None, ""):
                    feats["Q_Datapoint_ID"] = str(n_Q).strip()

                v = safe_float(n_Ilim)
                if v is not None:
                    feats["Strom_Limit_in_A"] = v

                if n_Wind not in (None, ""):
                    feats["Windgeschw_Datapoint"] = str(n_Wind).strip()
                if n_Glob not in (None, ""):
                    feats["Globale_Strahlung_Datapoint"] = str(n_Glob).strip()
                if n_Tamb not in (None, ""):
                    feats["Aussentemp_Datapoint"] = str(n_Tamb).strip()
                if n_DAB not in (None, ""):
                    feats["DAB_ID"] = str(n_DAB).strip()

                # Node type is stored at element root level: data.type
                if n_type_sel:
                    n["data"]["type"] = str(n_type_sel).strip()

                # Battery-only field: remove when empty to keep features clean
                v = safe_float(n_pmax_mw)
                if v is not None:
                    feats["p_max_MW"] = v
                else:
                    feats.pop("p_max_MW", None)

                # Coordinates
                v = safe_float(n_lat)
                if v is not None:
                    feats["Latitude_deg"] = v
                v = safe_float(n_lon)
                if v is not None:
                    feats["Longitude_deg"] = v

                # Optional: Formel-String für Sammelschienen-Beziehungen
                if n_busbar_formula not in (None, ""):
                    feats["busbar_id"] = str(n_busbar_formula).strip()
                else:
                    feats.pop("busbar_id", None)

                # Konfiguration abgeleiteter Felder
                enable = bool(n_derived_enable and "on" in n_derived_enable)
                if enable:
                    feature_key = (n_derived_feature_key or "P").strip() or "P"
                    terms = parse_formula_to_terms(n_derived_terms or "")

                    valid_terms = []
                    for t in terms:
                        tn = t.get("node")
                        if tn in node_ids:
                            valid_terms.append(t)

                    feats["derived"] = {
                        "method": "field_sum",
                        "feature_key": feature_key,
                        "terms": valid_terms
                    }
                else:
                    feats.pop("derived", None)

                n["data"]["features"] = feats
                status = f"Node-Felder gespeichert: '{nid}'."
                break

        new_elems = nodes + edges
        return new_elems, new_elems, status, new_edge_sel

    # -------------------------------------------------------------------------
    # Save edge fields (features)
    # -------------------------------------------------------------------------
    if trig == "btn_save_edge_schema":
        if not selected_edges:
            return elems, elems, "Fehler: Bitte erst eine Edge auswählen.", new_edge_sel

        sel = selected_edges[0]
        eid = sel.get("id")
        s, t = sel.get("source"), sel.get("target")

        # Edge matching prefers 'id'; falls back to (source, target) for edges without id.
        target_edge = None
        for ed in edges:
            d = ed.get("data", {}) or {}
            if eid is not None and d.get("id") == eid:
                target_edge = ed
                break
            if eid is None and d.get("source") == s and d.get("target") == t:
                target_edge = ed
                break

        if not target_edge:
            return elems, elems, "Fehler: Edge im aktuellen State nicht gefunden.", new_edge_sel

        feats = (target_edge["data"].get("features", {}) or {}).copy()

        v = safe_float(e_Ilim)
        if v is not None:
            feats["Strom_Limit_in_A"] = v

        v = safe_float(e_X_per_km)
        if v is not None:
            feats["X_ohm_per_km"] = v

        v = safe_float(e_len_km)
        if v is not None:
            feats["length_km"] = v

        # Gesamtreaktanz als Hilfswert
        if "X_ohm_per_km" in feats and "length_km" in feats:
            feats["X_total_ohm"] = feats["X_ohm_per_km"] * feats["length_km"]

        target_edge["data"]["features"] = feats
        status = "Edge-Felder gespeichert."
        new_elems = nodes + edges
        return new_elems, new_elems, status, new_edge_sel

    # -------------------------------------------------------------------------
    # Rename node (updates node ID + edges)
    # -------------------------------------------------------------------------
    if trig == "btn_rename_node":
        if not selected_nodes:
            return elems, elems, "Fehler: Bitte erst einen Node auswählen.", new_edge_sel

        old = selected_nodes[0].get("id")
        new = (r_node_id or "").strip()
        new_label_val = (r_node_label or "").strip()

        if not old:
            return elems, elems, "Fehler: Ungültige Node-Auswahl.", new_edge_sel
        if not new:
            return elems, elems, "Fehler: Neue Node-ID fehlt.", new_edge_sel

        # If ID does not change, only update the label
        if new == old:
            for n in nodes:
                if (n.get("data", {}) or {}).get("id") == old and new_label_val:
                    n["data"]["label"] = new_label_val
                    return nodes + edges, nodes + edges, f"Node-Label aktualisiert: '{old}'.", new_edge_sel
            return elems, elems, "Hinweis: Keine Änderung.", new_edge_sel

        if new in node_ids:
            return elems, elems, f"Fehler: Node-ID '{new}' existiert bereits.", new_edge_sel

        # Update node element
        for n in nodes:
            if (n.get("data", {}) or {}).get("id") == old:
                n["data"]["id"] = new

                # Label behavior: explicit label wins; otherwise keep stable label unless it mirrored the old ID.
                if new_label_val:
                    n["data"]["label"] = new_label_val
                else:
                    if (n["data"].get("label") or "") == old:
                        n["data"]["label"] = new
                break

        # Rewrite edge endpoints referencing the old node ID
        for e in edges:
            d = e.get("data", {}) or {}
            if d.get("source") == old:
                d["source"] = new
            if d.get("target") == old:
                d["target"] = new

        # Keep edge selection consistent with renamed IDs
        new_edge_sel = [new if x == old else x for x in new_edge_sel]

        status = f"Node umbenannt: '{old}' → '{new}'."
        new_elems = nodes + edges
        return new_elems, new_elems, status, new_edge_sel

    # -------------------------------------------------------------------------
    # Rename edge (updates edge ID + label)
    # -------------------------------------------------------------------------
    if trig == "btn_rename_edge":
        if not selected_edges:
            return elems, elems, "Fehler: Bitte erst eine Edge auswählen.", new_edge_sel

        sel = selected_edges[0]
        old_id = sel.get("id")
        if old_id is None:
            return elems, elems, "Fehler: Diese Edge hat keine ID.", new_edge_sel

        new_id = (r_edge_id or "").strip()
        new_label_val = (r_edge_label or "").strip()

        # Locate edge by ID
        target = None
        for ed in edges:
            if (ed.get("data", {}) or {}).get("id") == old_id:
                target = ed
                break
        if not target:
            return elems, elems, "Fehler: Edge im aktuellen State nicht gefunden.", new_edge_sel

        if new_id and new_id != old_id:
            if new_id in edge_ids:
                return elems, elems, f"Fehler: Edge-ID '{new_id}' existiert bereits.", new_edge_sel
            target["data"]["id"] = new_id

        if new_label_val:
            target["data"]["label"] = new_label_val

        status = "Edge umbenannt."
        new_elems = nodes + edges
        return new_elems, new_elems, status, new_edge_sel

    # -------------------------------------------------------------------------
    # Import JSON (expects list of Cytoscape elements)
    # -------------------------------------------------------------------------
    if trig == "upload_json":
        if not (upload_contents and upload_name and upload_name.endswith(".json")):
            return elems, elems, "Fehler: Bitte eine .json Datei hochladen.", new_edge_sel
        try:
            decoded = base64.b64decode(upload_contents.split(",")[1]).decode("utf-8")
            obj = json.loads(decoded)
        except Exception as ex:
            return elems, elems, f"Fehler: JSON konnte nicht gelesen werden: {ex}", new_edge_sel

        ok, msg = is_valid_elements_list(obj)
        if not ok:
            return elems, elems, f"Fehler: Import abgelehnt.\n{msg}", new_edge_sel

        _, imp_edges = split_elements(obj)
        missing = [i for i, e in enumerate(imp_edges) if not ((e.get("data", {}) or {}).get("id"))]
        warn = ""
        if missing:
            warn = f"\nHinweis: {len(missing)} Edge(s) ohne 'id' im Import."

        status = f"Import OK: {len(obj)} Elemente geladen.{warn}"
        return obj, obj, status, []

    # Default: state remains unchanged
    return nodes + edges, nodes + edges, status, new_edge_sel


@callback(
    Output("dl_json", "data"),
    Input("btn_export", "n_clicks"),
    State("gstore", "data"),
    prevent_initial_call=True
)
def export_json(_, elems):
    """Exportiert den aktuellen Graphen (Elements-Liste) als JSON."""
    return dict(
        content=json.dumps(elems, ensure_ascii=False, indent=2),
        filename="graph.json",
        type="application/json"
    )


if __name__ == "__main__":
    # Lokaler Start
    app.run(debug=True)
