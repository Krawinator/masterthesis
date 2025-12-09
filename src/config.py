# src/config.py
from pathlib import Path
from datetime import datetime, timezone

# Absoluter Pfad relativ zum src-Ordner:
BASE_DIR = Path(__file__).resolve().parent

GRAPH_PATH = BASE_DIR / "data" / "raw" / "graph" / "whole_graph.json"
RAW_TS_DIR = BASE_DIR / "data" / "raw" / "timeseries"

# Relevante Knotentypen
RELEVANT_NODE_TYPES = {"uw_field", "battery"}

# Zeitraster & Zeitzone für API
BUCKET_FACTOR = 15
BUCKET_UNIT = "MINUTE"
TIMEZONE = "Europe/Berlin"
AGGREGATION = "AVG"

# Historischer Betrachtungsbeginn
HIST_START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

# ------------------------------------------------------
# Slack-Knoten für DC-Lastfluss / PTDF
#   SLACK_NODE_ID = "SHUW" oder "JUBO" etc.
# ------------------------------------------------------
SLACK_NODE_ID = "SHUW"  
