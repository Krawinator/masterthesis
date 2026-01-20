from pathlib import Path
from datetime import datetime, timezone
import logging

# ============================================================
# Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent  # Repo-Root

# ============================================================
# Graph / Network
# ============================================================

GRAPH_PATH = BASE_DIR / "data" / "raw" / "graph" / "whole_graph.json"
SLACK_NODE_ID = "SHUW"

# Nur diese Node-Typen sind für Forecast / Powerbands relevant
RELEVANT_NODE_TYPES = {"uw_field", "battery"}

# ============================================================
# Time & aggregation settings
# ============================================================

BUCKET_FACTOR = 15
BUCKET_UNIT = "MINUTE"
AGGREGATION = "AVG"

TIMEZONE = "Europe/Berlin"

if BUCKET_UNIT == "MINUTE":
    FREQ = f"{BUCKET_FACTOR}min"
else:
    raise ValueError(f"BUCKET_UNIT {BUCKET_UNIT!r} wird aktuell nicht unterstützt.")

# Startpunkt für historische Daten (nach Aggregation)
HIST_START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

# Maximale Anzahl erlaubter fehlender Schritte (z. B. für Interpolation)
MAX_GAP_STEPS = 4

# ============================================================
# Raw & cleaned time series
# ============================================================

# Rohdaten (Messwerte, ggf. inkl. BESS-Einfluss)
RAW_TS_DIR = BASE_DIR / "data" / "raw" / "timeseries"

# “Vorbereitete” Zeitreihen (Raster + kleine Lücken gefüllt), Basis für BESS-cleaning
PREP_TS_DIR = BASE_DIR / "data" / "clean" / "timeseries_prepared"

# Bereinigte Zeitreihen ohne BESS-Einfluss
CLEAN_TS_DIR = BASE_DIR / "data" / "clean" / "timeseries_no_bess"



# ============================================================
# Weather data
# ============================================================

WEATHER_COLS = [
    "wind_speed_10m",
    "temperature_2m",
    "global_radiation",
]

# Wetterprognosen pro Node: <node_id>_weather_forecast.csv
WEATHER_FORECAST_DIR = BASE_DIR / "data" / "raw" / "weather_forecast"

# ============================================================
# Forecast / Prediction output
# ============================================================

# Roh-Prognosen (direkter Modelloutput)
PRED_TS_DIR = BASE_DIR / "data" / "pred"

# Normalisierte Prognosen (nach Flip / Shift / Clipping)
PRED_NORMALIZED_DIR = BASE_DIR / "data" / "pred_normalized"

# ============================================================
# Forecast experiment outputs
# ============================================================

FORECAST_EXPERIMENTS_DIR = PROJECT_DIR / "reports" / "forecast_experiments"

# Aktuelles Siegermodell
WINNER_MODEL_DIR = FORECAST_EXPERIMENTS_DIR / "_current"
WINNER_MODEL_PATH = WINNER_MODEL_DIR / "best_model.joblib"
WINNER_META_PATH = WINNER_MODEL_DIR / "best_model_meta.json"

# ============================================================
# Powerband / PTDF / DC-loadflow
# ============================================================

# Zielordner für berechnete Leistungsbänder (eine CSV pro Batterie)
POWERBAND_DIR = BASE_DIR / "data" / "powerband"

# Basiswerte für DC-Lastfluss
S_BASE_MVA = 100.0
V_KV_DEFAULT = 110.0
COSPHI_MIN = 0.95  # zur Umrechnung A → MW 

# Kontraktion: Leitungen mit X <= eps werden elektrisch zusammengezogen
X_EPS_OHM = 0.01

# Basecase-Leistung der Batterien im Forecast-Basecase
BASECASE_BESS_P_MW = 0.0

# Ziel-Auslastung der Leitungen in %
# 100 = volle Grenzwerte, <100 = konservativer Betrieb
UTIL_TARGET_PCT = 60.0

# ============================================================
# Logging
# ============================================================

LOG_DIR = PROJECT_DIR / "logs"
LOG_LEVEL = logging.WARNING

# ============================================================
# Sanity checks
# ============================================================

REQUIRED_DIRS = [
    RAW_TS_DIR,
    PREP_TS_DIR,
    CLEAN_TS_DIR,
    WEATHER_FORECAST_DIR,
    PRED_TS_DIR,
    PRED_NORMALIZED_DIR,
    POWERBAND_DIR,
    LOG_DIR,
]
