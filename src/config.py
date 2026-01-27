from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Pfade
# ------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent          # .../repo/src
REPO_ROOT = SRC_DIR.parent                        # .../repo
DATA_DIR = SRC_DIR / "data"                       # .../repo/src/data
REPORTS_DIR = REPO_ROOT / "reports"               # .../repo/reports

# ------------------------------------------------------------------
# Netzmodell
# ------------------------------------------------------------------

GRAPH_PATH = DATA_DIR / "raw" / "graph" / "whole_graph.json"
SLACK_NODE_ID = "SHUW"

# Nur diese Knotentypen werden für Forecast/Powerbands genutzt.
RELEVANT_NODE_TYPES = {"uw_field", "battery"}

# ------------------------------------------------------------------
# Zeitraster / Aggregation
# ------------------------------------------------------------------

BUCKET_FACTOR = 15
BUCKET_UNIT = "MINUTE"
AGGREGATION = "AVG"

# Intern wird in der Pipeline UTC verwendet; TIMEZONE dient als fachliche Referenz.
TIMEZONE = "Europe/Berlin"

if BUCKET_UNIT == "MINUTE":
    FREQ = f"{BUCKET_FACTOR}min"
else:
    raise ValueError(f"BUCKET_UNIT {BUCKET_UNIT!r} wird aktuell nicht unterstützt.")

# Startpunkt für historische Daten (nach Aggregation).
HIST_START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

# Maximale Lücke (in Raster-Schritten), die noch interpoliert wird.
MAX_GAP_STEPS = 4

# ------------------------------------------------------------------
# Zeitreihen (raw / prepared / cleaned)
# ------------------------------------------------------------------

RAW_TS_DIR = DATA_DIR / "raw" / "timeseries"

# Gerasterte und bereinigte Zeitreihen (Input für Forecast/BESS-Removal).
PREP_TS_DIR = DATA_DIR / "clean" / "timeseries_prepared"

# Zeitreihen nach Entfernung des BESS-Einflusses (Input für Netz-/Powerband-Teil).
CLEAN_TS_DIR = DATA_DIR / "clean" / "timeseries_no_bess"

# ------------------------------------------------------------------
# Wetterdaten
# ------------------------------------------------------------------

# Spalten, die der Loader erzeugt/erwartet. Forecast orientiert sich an WINNER_META_PATH.
WEATHER_COLS = [
    "wind_speed_10m",
    "temperature_2m",
    "global_radiation",
]

# Wetter-Forecast wird als externer Input behandelt und liegt daher im raw-Bereich.
WEATHER_FORECAST_DIR = DATA_DIR / "raw" / "weather_forecast"

# ------------------------------------------------------------------
# Forecast-Ausgabe
# ------------------------------------------------------------------

PRED_TS_DIR = DATA_DIR / "pred"

# ------------------------------------------------------------------
# Forecast-Modelle / Experimente
# ------------------------------------------------------------------

FORECAST_EXPERIMENTS_DIR = REPORTS_DIR / "forecast_experiments"

# Pro Node refittetes Modell mit fixierten Hyperparametern.
NODE_MODELS_DIR = REPORTS_DIR / "forecast_models_per_node_fixedhp"

# Globales Fallback-Modell (optional).
WINNER_MODEL_DIR = FORECAST_EXPERIMENTS_DIR / "_current"
WINNER_MODEL_PATH = WINNER_MODEL_DIR / "best_model.joblib"
WINNER_META_PATH = WINNER_MODEL_DIR / "best_model_meta.json"

# ------------------------------------------------------------------
# Powerbands / DC-Loadflow
# ------------------------------------------------------------------

POWERBAND_DIR = DATA_DIR / "powerband"

S_BASE_MVA = 100.0
V_KV_DEFAULT = 110.0
COSPHI_MIN = 0.95

# Kanten mit sehr kleinem X werden bei der Kontraktion zusammengefasst.
X_EPS_OHM = 0.01

# Basecase-Setzung für BESS-Leistung.
BASECASE_BESS_P_MW = 0.0

# Prozentualer Zielwert für die zulässige Auslastung der thermischen Limits.
UTIL_TARGET_PCT = 50.0

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOG_DIR = REPO_ROOT / "logs"
LOG_LEVEL = logging.WARNING

# ------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------

REQUIRED_DIRS = [
    RAW_TS_DIR,
    PREP_TS_DIR,
    CLEAN_TS_DIR,
    WEATHER_FORECAST_DIR,
    PRED_TS_DIR,
    POWERBAND_DIR,
    LOG_DIR,
]

def ensure_dirs() -> None:
    """Legt alle von der Pipeline verwendeten Verzeichnisse an."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
