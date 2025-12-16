# %% 
"""
BESS-Bereinigung: Lineare Regression und „Entkopplung“ der BESS-Einspeisung
============================================================================

In diesem Notebook wollen wir:
--------------------------------
1. Die BESS-Leistung als Zeitreihe laden.
2. Alle anderen Knoten (Leistungen in MW) laden.
3. Die Korrelationen zwischen BESS und jedem Knoten VOR der Bereinigung auswerten.
4. Pro Knoten eine lineare Regression der Form

       P_node(t) ≈ alpha + beta * P_BESS(t)

   auf der *gemeinsamen* Zeitbasis (nur Zeitpunkte, wo beide Werte haben) schätzen.
5. Mit dem geschätzten beta die **volle** Knotenzeitreihe bereinigen:

       P_node_clean(t) = P_node(t) - beta * P_BESS(t)

   WICHTIG:
   - Für Zeitpunkte ohne BESS-Messung setzen wir P_BESS(t) = 0,
     d.h. dort bleibt der Knotenwert exakt unverändert.
   - Damit stellen wir sicher, dass „Nicht-BESS-Zeiten“ NICHT verfälscht werden.
6. Korrelationen NACH der Bereinigung auswerten.
7. Beispielweise Histogramm und Zeitreihen-Plots für einen ausgewählten Knoten anzeigen.

Design-Entscheidungen (warum ist der Code so?)
-----------------------------------------------
- Wir benutzen **nur pandas & numpy & matplotlib**, keine Scikit-Learn, um die Logik
  transparent zu halten und jederzeit im Code nachvollziehen zu können, was passiert.
- Die Regression machen wir NUR auf der **Schnittmenge** von BESS- und Node-Zeitstempeln:
  *Nur diese Zeitpunkte enthalten Informationen darüber, wie stark der Node typischerweise
  auf die BESS reagiert.* Würden wir `NaN` oder 0 in die Regression mischen, würden wir die
  Schätzung von beta verzerren.
- Für die eigentliche Bereinigung wollen wir allerdings die **komplette** Node-Zeitreihe
  erhalten. Deswegen legen wir BESS auf den Node-Index und setzen fehlende BESS-Werte
  explizit auf 0. So bleibt alles, was zeitlich außerhalb der BESS-Messung liegt,
  unverändert.
- Korrelationen nach der Bereinigung berechnen wir wiederum nur dort, wo BESS-Werte
  existieren. Nur dann macht es inhaltlich Sinn, die Entkopplung zu prüfen.
"""

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 0. KONFIGURATION
# ============================
# Diese Parameter solltest du bei Bedarf anpassen. Sie sind hier oben zentral gehalten,
# damit der Rest des Notebooks „dumm“ und reproduzierbar läuft.

MEAS_DIR = r"src/data/raw/timeseries"  # Ordner mit *_hist.csv Dateien
TS_COL = "timestamp"                   # Zeitstempel-Spalte
VAL_COL = "P_MW"                       # Spalte mit Wirkleistung in MW

# Datei mit BESS-Leistung (wie im bisherigen Code)
BESS_FILE = "BOLS_E42_hist.csv"

# Namen (ohne _hist.csv) der UW-Felder direkt am BESS-Umspannwerk
# → rein informativ, damit wir später sehen, welche Nodes „direkt dran“ hängen.
BESS_UW_FIELDS = [
    "BOLS_E41",
    "BOLS_E42",   # BESS selbst
    # "BOLS_E43",
]

# Node für Beispielplots
PLOT_NODE = "SHUW_E23"   # kannst du jederzeit ändern


# %%
# ============================
# 1. BESS-Zeitreihe laden
# ============================
# Warum so?
# ---------
# - Wir laden BESS als eigene Serie mit Spaltennamen "BESS_P", sortieren und indexieren mit dem Zeitstempel.
# - Diese eine Serie ist der „Erklärer“ x(t) in allen Regressionsmodellen.

bess_path = os.path.join(MEAS_DIR, BESS_FILE)

bess = pd.read_csv(bess_path, parse_dates=[TS_COL])
bess = bess[[TS_COL, VAL_COL]].rename(columns={VAL_COL: "BESS_P"})
bess = bess.sort_values(TS_COL).set_index(TS_COL)

print(f"BESS-Zeitreihe geladen: {BESS_FILE}, Samples: {len(bess)}")
display(bess.head())


# %%
# ============================
# 2. Einfache lineare Regression
# ============================
# y ≈ alpha + beta * x
#
# Warum implementieren wir das selbst?
# ------------------------------------
# - Volle Transparenz der Formel: beta = Cov(x, y) / Var(x)
# - Keine Blackbox, einfaches Debugging.
# - R^2 berechnen wir manuell, um die Güte der Anpassung beurteilen zu können.

def fit_simple_linear_regression(x: pd.Series, y: pd.Series):
    """
    Einfache OLS-Regression:
        y ≈ alpha + beta * x

    Parameter
    ---------
    x : pd.Series
        Regressor (hier: BESS_P)
    y : pd.Series
        Zielvariable (hier: P_MW eines Knotens)

    Rückgabe
    --------
    alpha : float
        Achsenabschnitt
    beta : float
        Steigung (Sensitivität des Knotens auf BESS-P)
    r2 : float
        Bestimmtheitsmaß R^2 als Gütemaß der Regression
    """
    x = x.astype(float)
    y = y.astype(float)

    mx = x.mean()
    my = y.mean()

    cov_xy = ((x - mx) * (y - my)).sum()
    var_x = ((x - mx) ** 2).sum()

    if var_x == 0:
        # keine Varianz im Prädiktor → keine Regression möglich
        beta = 0.0
    else:
        beta = cov_xy / var_x

    alpha = my - beta * mx

    # R^2
    y_hat = alpha + beta * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - my) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return alpha, beta, r2


# %%
# ============================
# 3. Korrelationen VOR Bereinigung
# ============================
# Ziel:
# - Erstmal nur anschauen, wie stark jeder Knoten mit der BESS-Leistung korreliert.
# - Dafür:
#   * jede *_hist.csv Datei laden (außer BESS-Datei),
#   * auf gemeinsamen Zeitbereich mit BESS schneiden (inner join),
#   * Korrelation ausrechnen,
#   * aber parallel die komplette Node-Serie (eigener Index) speichern.

correlations_before = {}
orig_series = {}   # komplette Originalserien pro Node

for fname in os.listdir(MEAS_DIR):
    if not fname.endswith("_hist.csv"):
        continue
    if fname == BESS_FILE:
        continue  # BESS selbst überspringen

    node_name = fname.replace("_hist.csv", "")
    path = os.path.join(MEAS_DIR, fname)

    df = pd.read_csv(path, parse_dates=[TS_COL])
    df = df[[TS_COL, VAL_COL]].sort_values(TS_COL).set_index(TS_COL)

    # komplette Node-Serie merken (für spätere Bereinigung & Plots)
    orig_series[node_name] = df[VAL_COL]

    # für Korrelation: nur Schnitt mit BESS
    merged = bess.join(df, how="inner", rsuffix="_other")
    if len(merged) < 100:
        # zu wenig überlappende Daten → statistisch unzuverlässig
        continue

    corr_before = merged["BESS_P"].corr(merged[VAL_COL])
    correlations_before[node_name] = corr_before

print("Korrelationen VOR Bereinigung (BESS_P vs. Node):")
s_before = pd.Series(correlations_before).sort_values(key=np.abs, ascending=False)
print(s_before)


# %%
# ============================
# 4. Regression + Bereinigung
# ============================
# Wichtiger Punkt:
# - Regression nur auf Overlap (Node ∩ BESS) → sinnvoller Beta-Schätzer.
# - Bereinigung auf voller Node-Serie:
#
#   1. BESS auf Node-Index reindizieren:
#        bess_on_node_idx_raw = bess.reindex(node_index)["BESS_P"]
#      → kann NaN enthalten, wo BESS keine Messung hat.
#
#   2. Für KORREKTUR benötigen wir BESS(t):
#        bess_on_node_idx = bess_on_node_idx_raw.fillna(0.0)
#      → dort, wo keine BESS-Messung existiert, behandeln wir BESS-Leistung als 0,
#        d.h. an diesen Stellen bleibt P_node unverändert.
#
#   3. Korrelation NACH Bereinigung berechnen wir nur dort,
#      wo BESS tatsächlich gemessen hat (mask = ~isna).

results = []
clean_series = {}

for node_name, corr_before in correlations_before.items():
    fname = f"{node_name}_hist.csv"
    path = os.path.join(MEAS_DIR, fname)

    df = pd.read_csv(path, parse_dates=[TS_COL])
    df = df[[TS_COL, VAL_COL]].sort_values(TS_COL).set_index(TS_COL)

    # ---------- 4.1 Regression: nur Overlap ----------
    merged = bess.join(df, how="inner", rsuffix="_other")
    if len(merged) < 100:
        continue

    x_reg = merged["BESS_P"]
    y_reg = merged[VAL_COL]

    alpha, beta, r2 = fit_simple_linear_regression(x_reg, y_reg)

    # ---------- 4.2 Bereinigung: volle Node-Zeitreihe ----------
    # BESS auf Node-Index legen (kann NaN haben, wo BESS nicht misst)
    bess_on_node_idx_raw = bess.reindex(df.index)["BESS_P"]

    # Für die Korrektur: fehlende BESS-Werte als 0 interpretieren
    bess_on_node_idx = bess_on_node_idx_raw.fillna(0.0)

    y_full = df[VAL_COL]
    y_clean_full = y_full - beta * bess_on_node_idx

    # ---------- 4.3 Korrelation NACH der Bereinigung ----------
    # Nur dort, wo BESS tatsächlich Messwerte hat (Overlapping Times)
    mask = ~bess_on_node_idx_raw.isna()
    if mask.sum() >= 10:
        corr_after = bess_on_node_idx_raw[mask].corr(y_clean_full[mask])
    else:
        corr_after = np.nan

    # mittlere absolut entfernte Leistung (nur da, wo BESS wirklich aktiv ist)
    mean_abs_removed = (beta * x_reg).abs().mean()

    clean_series[node_name] = y_clean_full
    is_bess_uw = node_name in BESS_UW_FIELDS

    results.append({
        "node": node_name,
        "corr_before": corr_before,
        "corr_after": corr_after,
        "beta": beta,
        "alpha": alpha,
        "r2": r2,
        "mean_abs_removed_MW": mean_abs_removed,
        "is_bess_uw_field": is_bess_uw,
    })

# %%
# ============================
# 5. Ergebnis-Tabelle
# ============================
# Hier siehst du:
# - corr_before / corr_after: wie stark war/sind die Nodes mit BESS korreliert?
# - beta: Sensitivität des Knotens auf BESS (MW/MW).
# - r2: wie gut erklärt BESS die Variation des Knotens?
# - mean_abs_removed_MW: durchschnittlich abgezogene Leistung.

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="corr_before", key=np.abs, ascending=False)

print("\nZusammenfassung Regression & Bereinigung:")
print(results_df.to_string(index=False))


# %%
# ============================
# 6. DataFrames mit Original & Bereinigt
# ============================
# Aus den Dictionaries bauen wir zwei große DataFrames:
# - orig_df: Originalzeitreihen (alle Nodes)
# - clean_df: bereinigte Zeitreihen (alle Nodes)
#
# Danach vergleichen wir Korrelationen vor/nach noch einmal global.

orig_df = pd.DataFrame(orig_series)
clean_df = pd.DataFrame(clean_series)

# Korrelation vor der Bereinigung: nur Overlap mit BESS
corr_before_full = (
    orig_df.join(bess, how="inner")
           .corr()["BESS_P"]
           .drop("BESS_P")
)

# Korrelation nach der Bereinigung: ebenfalls nur Overlap mit BESS
corr_after_full = (
    clean_df.join(bess, how="inner")
            .corr()["BESS_P"]
            .drop("BESS_P")
)

print("\nKorrelationsvergleich (BESS_P vs. Nodes):")
corr_compare = pd.DataFrame({
    "corr_before": corr_before_full,
    "corr_after": corr_after_full
}).sort_values(by="corr_before", key=np.abs, ascending=False)

print(corr_compare)


# %%
# ============================
# 7. Deskriptive Statistik für einen Beispiel-Node
# ============================
# Hier kannst du dir für einen Node anschauen, wie sich Mittelwert, Std, Min/Max etc.
# durch die Bereinigung verändert haben.

if PLOT_NODE in orig_df.columns and PLOT_NODE in clean_df.columns:
    print(f"\nDescriptive Stats für {PLOT_NODE} (ORIG):")
    print(orig_df[PLOT_NODE].describe())
    print(f"\nDescriptive Stats für {PLOT_NODE} (CLEAN):")
    print(clean_df[PLOT_NODE].describe())
else:
    print(f"\nWARNUNG: {PLOT_NODE} nicht in Daten gefunden.")


# %%
# ============================
# 8. Histogramm-Vergleich
# ============================
# Der Histogrammvergleich zeigt:
# - Wie stark sich die Verteilung verschiebt/verengt.
# - Ob die Bereinigung eventuell unrealistische Werte produziert.
#
# Typische Erwartung:
# - Moderate Verschiebung / "Verschmälerung" der Verteilung für stark gekoppelte Nodes,
#   aber keine völlig neuen, absurden Wertebereiche.

if PLOT_NODE in orig_df.columns and PLOT_NODE in clean_df.columns:
    plt.figure(figsize=(7, 5))
    plt.hist(orig_df[PLOT_NODE].dropna(), bins=80, alpha=0.6, label="orig")
    plt.hist(clean_df[PLOT_NODE].dropna(), bins=80, alpha=0.6, label="clean")
    plt.title(PLOT_NODE)
    plt.xlabel("P_MW")
    plt.ylabel("Häufigkeit")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
# ============================
# 9. Zeitreihen-Vergleich (Ausschnitt)
# ============================
# Ein Zeitreihenplot macht intuitiv sichtbar:
# - Wie stark der BESS-Einfluss „abgezogen“ wurde.
# - Ob es Zeitschnitte gibt, in denen die Kurve „komisch“ aussieht.
#
# Wir plotten hier einfach die ersten N Punkte.
# (Du kannst natürlich auch einen konkreten Zeitraum per .loc wählen.)

if PLOT_NODE in orig_df.columns and PLOT_NODE in clean_df.columns:
    N = 2000  # Anzahl Punkte für den Ausschnitt
    slice_idx = orig_df.index[:N]

    plt.figure(figsize=(12, 5))
    plt.plot(slice_idx, orig_df.loc[slice_idx, PLOT_NODE], label="orig", linewidth=1)
    plt.plot(slice_idx, clean_df.loc[slice_idx, PLOT_NODE], label="clean", linewidth=1)
    plt.title(f"{PLOT_NODE} – Vergleich Original vs. Bereinigt (Ausschnitt, N={N})")
    plt.xlabel("Zeit")
    plt.ylabel("P_MW")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(f"\nWARNUNG: {PLOT_NODE} nicht in Daten gefunden; kein Plot möglich.")
