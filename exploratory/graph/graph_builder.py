# pip install dash dash-cytoscape
import dash
from dash import html, dcc, Input, Output, State, callback, ctx, no_update
import dash_cytoscape as cyto
import json
import base64

app = dash.Dash(__name__)
app.title = "Graph-Editor – Stromnetz"

# ---- Leerer Start ----
INITIAL_ELEMENTS: list[dict] = []

# ---- Helpers: Stylesheet + Parsing ----
def make_stylesheet(show_edge_labels: bool):
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

def parse_formula_to_terms(s: str):
    """
    Wandelt 'JUBO_E03:-1, JUBO_E02:-1' in [{"node":"JUBO_E03","coeff":-1.0}, ...] um.
    Leere/fehlerhafte Teile werden übersprungen.
    """
    terms = []
    if not s:
        return terms
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
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
    Wandelt [{"node":"JUBO_E03","coeff":-1.0}, ...] in 'JUBO_E03:-1, ...' um.
    """
    if not terms:
        return ""
    return ", ".join(f"{t.get('node','')}:{t.get('coeff','')}" for t in terms if t.get("node") is not None)

# ---- Layout ----
app.layout = html.Div([
    # Inject CSS
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

    dcc.Store(id="gstore", data=INITIAL_ELEMENTS),
    dcc.Store(id="edge_selection", data=[]),

    html.Div([
        # Left Column
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

        # Right Column — Tools Panel
        html.Div([
            html.H3("Werkzeuge"),

            # Toggle for edge labels
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
                        dcc.Input(id="new_node_id", placeholder="z. B. A_B1, BESS_1, J1",
                                  type="text", style={"width": "100%"}),
                    ]),
                    html.Div([
                        html.Label("Label", style={"fontWeight": 600}),
                        dcc.Input(id="new_node_label", placeholder="Anzeigename (optional)",
                                  type="text", style={"width": "100%"}),
                    ]),
                ], style={
                    "display": "grid", "gridTemplateColumns": "1fr 1fr",
                    "columnGap": "8px", "alignItems": "start", "marginBottom": "8px"
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

            html.Button("Ausgewählten Node/Edge löschen", id="btn_del_element",
                        style={"marginBottom": 12}),

            html.Hr(),
            html.H4("Umbenennen"),

            # Rename node
            html.Div([
                html.Label("Node: Neue ID"),
                dcc.Input(id="rename_node_id"),
                html.Label("Node: Neues Label"),
                dcc.Input(id="rename_node_label"),
                html.Button("Node umbenennen", id="btn_rename_node")
            ], style={"display": "grid",
                      "gridTemplateColumns": "200px 1fr",
                      "gap": "6px", "marginBottom": "10px"}),

            # Rename edge
            html.Div([
                html.Label("Edge: Neue ID"),
                dcc.Input(id="rename_edge_id"),
                html.Label("Edge: Neues Label"),
                dcc.Input(id="rename_edge_label"),
                html.Button("Edge umbenennen", id="btn_rename_edge")
            ], style={"display": "grid",
                      "gridTemplateColumns": "200px 1fr",
                      "gap": "6px"}),

            html.Hr(),
            html.H4("Edge anlegen"),
            html.Div([
                dcc.Input(id="new_edge_id", placeholder="Edge-ID"),
                dcc.Input(id="new_edge_label", placeholder="Edge-Label",
                          style={"marginLeft": 6}),
                html.Button("Edge hinzufügen (2 Nodes wählen)", id="btn_add_edge",
                            style={"marginLeft": 6})
            ]),
            html.Div(id="edge_sel_info", style={"fontSize": 12, "marginTop": 4}),

            html.Hr(),
            html.H4("Auswahl"),
            html.Div(id="sel_summary",
                     style={"whiteSpace": "pre-wrap", "fontFamily": "monospace"}),

            # ===== Node-Felder =====
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

                # --- Ableitungs-UI (DERIVED) ---
                html.Label("Busbar-Formel (optional)"),
                dcc.Input(
                    id="n_busbar_formula",
                    placeholder="z. B. JUBO_E03:-1, JUBO_E02:-1",
                    style={"width": "100%"}
                ),

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
            ], style={"display": "grid",
                      "gridTemplateColumns": "220px 1fr",
                      "gap": "6px"}),

            html.Button("Node-Felder speichern", id="btn_save_node_schema",
                        style={"marginTop": 6}),

            # ===== Edge-Felder =====
            html.H5("Edge-Felder"),
            html.Div([
                html.Label("Strom_Limit_in_A"),
                dcc.Input(id="e_Strom_Limit_in_A", type="number"),

                html.Label("Reaktanz X (Ω/km)"),
                dcc.Input(id="e_X_ohm_per_km", type="number", step="any"),

                html.Label("Länge (km)"),
                dcc.Input(id="e_length_km", type="number", step="any"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "220px 1fr",
                      "gap": "6px"}),

            html.Button("Edge-Felder speichern", id="btn_save_edge_schema",
                        style={"marginTop": 6}),

            html.Hr(),
            html.H4("Import / Export"),

            html.Button("Export JSON", id="btn_export"),
            dcc.Download(id="dl_json"),

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

# ---------- Edge Label Toggle ----------
@callback(
    Output("graph", "stylesheet"),
    Input("edge_label_toggle", "value")
)
def toggle_edge_labels(values):
    show = bool(values and "show" in values)
    return make_stylesheet(show_edge_labels=show)

# ---------- Auswahl anzeigen ----------
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
    # NEW: Derived UI Prefill
    Output("n_busbar_formula", "value"),
    Output("n_derived_enable", "value"),
    Output("n_derived_feature_key", "value"),
    Output("n_derived_terms", "value"),
    Input("graph", "selectedNodeData"),
    Input("graph", "selectedEdgeData"),
)
def show_selection(node_data, edge_data):
    if node_data:
        d = node_data[0]
        feats = d.get("features", {}) or {}
        summary = f"[Node] id={d['id']}  type={d.get('type')}\nfeatures={json.dumps(feats, indent=2)}"

        # Derived vorfüllen
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
            "", "",  # rename node
            d.get("id"), d.get("label"),
            None, None,
            feats.get("X_ohm_per_km", None),
            feats.get("length_km", None),
            "", [], "P", ""  # derived UI leeren
        )

    # Nothing selected
    return ("Nichts ausgewählt.", "", "", None, "", "", "", "",
            None, None, None, "", "", "", "",
            None, None, None, None, "", [], "P", "")

# ---------- Node Tap for Edge Creation ----------
@callback(
    Output("edge_selection", "data"),
    Output("edge_sel_info", "children"),
    Input("graph", "tapNodeData"),
    State("edge_selection", "data"),
)
def handle_tap_node(tapped, current):
    current = current or []
    if not tapped:
        return current, no_update
    nid = tapped["id"]
    cur = [n for n in current if n != nid]
    cur.append(nid)
    cur = cur[-2:]
    return cur, f"Gewählt für Edge: {cur}"

# ---------- Hauptmutation ----------
@callback(
    Output("graph", "elements"),
    Output("gstore", "data"),
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
    trig = ctx.triggered_id
    elems = elems or []

    def split(el):
        nodes = [x for x in el if "source" not in x.get("data", {})]
        edges = [x for x in el if "source" in x.get("data", {})]
        return nodes, edges

    nodes, edges = split(elems)
    node_ids = {n["data"]["id"] for n in nodes}
    edge_ids = {e["data"].get("id") for e in edges}

    # --- Node hinzufügen
    if trig == "btn_add_node":
        if new_id and new_id not in node_ids:
            nodes.append({
                "data": {"id": new_id, "label": new_label or new_id,
                         "type": new_type, "features": {}},
                "position": {"x": 50 * (len(nodes)+1), "y": 50 * (len(nodes)+1)}
            })
        return nodes + edges, nodes + edges

    # --- Edge hinzufügen
    if trig == "btn_add_edge":
        if edge_sel and len(edge_sel) == 2:
            s, t = edge_sel
            if s in node_ids and t in node_ids:
                if new_edge_id and new_edge_id not in edge_ids:
                    edges.append({
                        "data": {
                            "id": new_edge_id,
                            "source": s,
                            "target": t,
                            "label": new_edge_label or new_edge_id,
                            "features": {}
                        }
                    })
        return nodes + edges, nodes + edges

    # --- Node-Felder speichern (inkl. Derived & Busbar-Formel)
    if trig == "btn_save_node_schema" and selected_nodes:
        nid = selected_nodes[0]["id"]
        for n in nodes:
            if n["data"]["id"] == nid:
                feats = n["data"].get("features", {}) or {}

                # Standardfelder
                if n_P not in (None, ""): feats["P_Datapoint_ID"] = n_P
                if n_Q not in (None, ""): feats["Q_Datapoint_ID"] = n_Q
                if n_Ilim not in (None, ""): feats["Strom_Limit_in_A"] = float(n_Ilim)
                if n_Wind not in (None, ""): feats["Windgeschw_Datapoint"] = n_Wind
                if n_Glob not in (None, ""): feats["Globale_Strahlung_Datapoint"] = n_Glob
                if n_Tamb not in (None, ""): feats["Aussentemp_Datapoint"] = n_Tamb
                if n_DAB not in (None, ""): feats["DAB_ID"] = n_DAB
                if n_type_sel: n["data"]["type"] = n_type_sel
                if n_pmax_mw not in (None, ""):
                    feats["p_max_MW"] = float(n_pmax_mw)
                else:
                    feats.pop("p_max_MW", None)
                if n_lat not in (None, ""): feats["Latitude_deg"] = float(n_lat)
                if n_lon not in (None, ""): feats["Longitude_deg"] = float(n_lon)

                # Busbar-Formel als String ablegen (wie dein Beispiel)
                if n_busbar_formula not in (None, ""):
                    feats["busbar_id"] = n_busbar_formula
                else:
                    feats.pop("busbar_id", None)

                # Derived-Struktur (optional)
                enable = bool(n_derived_enable and "on" in n_derived_enable)
                if enable:
                    feature_key = (n_derived_feature_key or "P").strip()
                    terms = parse_formula_to_terms(n_derived_terms or "")
                    feats["derived"] = {
                        "method": "field_sum",
                        "feature_key": feature_key,
                        "terms": terms
                    }
                else:
                    feats.pop("derived", None)

                n["data"]["features"] = feats
                break
        return nodes + edges, nodes + edges

    # --- Edge-Felder speichern
    if trig == "btn_save_edge_schema" and selected_edges:
        e = selected_edges[0]
        eid = e.get("id")
        s, t = e.get("source"), e.get("target")

        target_edge = None
        for ed in edges:
            if ed["data"].get("id") == eid:
                target_edge = ed
                break
            if eid is None and ed["data"]["source"] == s and ed["data"]["target"] == t:
                target_edge = ed
                break

        if target_edge:
            feats = target_edge["data"].get("features", {}) or {}
            if e_Ilim not in (None, ""): feats["Strom_Limit_in_A"] = float(e_Ilim)
            if e_X_per_km not in (None, ""): feats["X_ohm_per_km"] = float(e_X_per_km)
            if e_len_km not in (None, ""): feats["length_km"] = float(e_len_km)
            if "X_ohm_per_km" in feats and "length_km" in feats:
                feats["X_total_ohm"] = feats["X_ohm_per_km"] * feats["length_km"]
            target_edge["data"]["features"] = feats

        return nodes + edges, nodes + edges

    # --- Node umbenennen
    if trig == "btn_rename_node" and selected_nodes:
        old = selected_nodes[0]["id"]
        new = (r_node_id or "").strip()
        new_label_val = (r_node_label or "").strip()
        if new and new != old and new not in node_ids:
            for n in nodes:
                if n["data"]["id"] == old:
                    n["data"]["id"] = new
                    if new_label_val:
                        n["data"]["label"] = new_label_val
                    break
            for e in edges:
                if e["data"]["source"] == old:
                    e["data"]["source"] = new
                if e["data"]["target"] == old:
                    e["data"]["target"] = new
        return nodes + edges, nodes + edges

    # --- Edge umbenennen
    if trig == "btn_rename_edge" and selected_edges:
        e = selected_edges[0]
        old_id = e.get("id")
        for ed in edges:
            if ed["data"].get("id") == old_id:
                if r_edge_id and r_edge_id not in edge_ids:
                    ed["data"]["id"] = r_edge_id
                if r_edge_label:
                    ed["data"]["label"] = r_edge_label
                break
        return nodes + edges, nodes + edges

    # --- Import JSON
    if trig == "upload_json":
        if upload_contents and upload_name and upload_name.endswith(".json"):
            try:
                decoded = base64.b64decode(upload_contents.split(",")[1]).decode("utf-8")
                new_elems = json.loads(decoded)
                if isinstance(new_elems, list):
                    return new_elems, new_elems
            except Exception:
                pass
        return elems, elems

    return nodes + edges, nodes + edges

# Export 
@callback(
    Output("dl_json", "data"),
    Input("btn_export", "n_clicks"),
    State("gstore", "data"),
    prevent_initial_call=True
)
def export_json(_, elems):
    return dict(
        content=json.dumps(elems, ensure_ascii=False, indent=2),
        filename="graph.json",
        type="application/json"
    )

if __name__ == "__main__":
    app.run(debug=True)
