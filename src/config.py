from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent

GRAPH_PATH = BASE_DIR / "data" / "raw" / "graph" / "whole_graph.json"
RAW_TS_DIR = BASE_DIR / "data" / "raw" / "timeseries"
CLEAN_TS_DIR = BASE_DIR / "data" / "clean" / "timeseries_no_bess"
RELEVANT_NODE_TYPES = {"uw_field", "battery"}

BUCKET_FACTOR = 15
BUCKET_UNIT = "MINUTE"
TIMEZONE = "Europe/Berlin"
AGGREGATION = "AVG"

HIST_START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

SLACK_NODE_ID = "SHUW"

# ---- ab hier neu/ergänzt ----

if BUCKET_UNIT == "MINUTE":
    FREQ = f"{BUCKET_FACTOR}min"
else:
    raise ValueError(f"BUCKET_UNIT {BUCKET_UNIT!r} wird noch nicht unterstützt.")

MAX_GAP_STEPS = 4

WEATHER_COLS = [
    "wind_speed_10m",
    "temperature_2m",
    "global_radiation",
]
