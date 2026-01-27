# src/data/eiot_client.py

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Union, Dict, Any, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 0) Token-Handling
# ---------------------------------------------------------------------------

EIOT_API_TOKEN: Optional[str] = None


def _load_eiot_token() -> str:
    """
    Lädt das EIOT-API-Token:

    1. Wenn Umgebungsvariable EIOT_API_TOKEN gesetzt ist, wird diese verwendet.
    2. Sonst wird über ARM_TENANT_ID / ARM_CLIENT_ID / ARM_CLIENT_SECRET / SCOPE
       ein Token via Azure AD (Client Credentials) geholt.

    Erwartete .env-Variablen:
        EIOT_API_TOKEN (optional, dann wird direkt verwendet)
        ARM_TENANT_ID
        ARM_CLIENT_ID
        ARM_CLIENT_SECRET
        SCOPE
    """
    global EIOT_API_TOKEN

    if EIOT_API_TOKEN:
        return EIOT_API_TOKEN

    load_dotenv(override=True)

    # 1) Direkter Token aus ENV (falls vorhanden)
    direct_token = os.getenv("EIOT_API_TOKEN")
    if direct_token:
        EIOT_API_TOKEN = direct_token
        logger.info("EIOT: Zugriffstoken aus EIOT_API_TOKEN verwendet.")
        return EIOT_API_TOKEN

    # 2) Azure AD Client Credentials Flow
    tenant_id = os.getenv("ARM_TENANT_ID")
    client_id = os.getenv("ARM_CLIENT_ID")
    client_secret = os.getenv("ARM_CLIENT_SECRET")
    scope = os.getenv("SCOPE")

    if not all([tenant_id, client_id, client_secret, scope]):
        raise EnvironmentError(
            "EIOT-Token konnte nicht geladen werden: "
            "Bitte entweder EIOT_API_TOKEN oder ARM_TENANT_ID, ARM_CLIENT_ID, "
            "ARM_CLIENT_SECRET, SCOPE in der .env setzen."
        )

    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }

    logger.info("EIOT: Hole Access-Token via Azure AD (Client Credentials).")

    try:
        resp = requests.post(url, data=data, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("EIOT: Fehler beim Token-Abruf von %s: %s", url, e, exc_info=True)
        if hasattr(e, "response") and e.response is not None:
            logger.error(
                "EIOT: Token-Response-Status=%s, Body=%s",
                e.response.status_code,
                e.response.text,
            )
        raise

    token_json = resp.json()
    token = token_json.get("access_token")
    if not token:
        logger.error("EIOT: Token-Response ohne 'access_token': %s", token_json)
        raise EnvironmentError("EIOT: 'access_token' im Token-Response fehlt.")

    EIOT_API_TOKEN = token
    logger.info("EIOT: Access-Token erfolgreich erhalten.")
    return EIOT_API_TOKEN


# ---------------------------------------------------------------------------
# 1) Zeit-Helfer
# ---------------------------------------------------------------------------

def _to_iso_z(t: Union[str, datetime]) -> str:
    """
    Nimmt str oder datetime und gibt ISO-String 'YYYY-MM-DDTHH:MM:SS.000Z' zurück.
    """
    if isinstance(t, datetime):
        return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return str(t)


def _ensure_dt_utc(t: Union[str, datetime]) -> datetime:
    """
    Wandelt str oder datetime in timezone-aware UTC-datetime um.
    """
    if isinstance(t, datetime):
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)

    # ISO-String (mit oder ohne 'Z')
    dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# 2) EIOT-Datapoint mit Chunking abrufen
# ---------------------------------------------------------------------------

def fetch_datapoint_raw(
    datapoint_id: str,
    start_time: Union[str, datetime],
    end_time: Union[str, datetime],
    aggregation: str = "AVG",
    bucket_factor: int = 15,
    bucket_unit: str = "MINUTE",
    timezone_str: str = "Europe/Berlin",
    tenant_id: str = "ipen_shng_run",
    chunk_days: int = 90,
) -> Dict[str, Any]:
    """
    Holt aggregierte Samples von EIOT.

    Input:
        datapoint_id:  EIOT-Datapoint-ID
        start_time:    datetime oder ISO-String
        end_time:      datetime oder ISO-String
        aggregation:   z.B. "AVG"
        bucket_factor: z.B. 15
        bucket_unit:   z.B. "MINUTE"
        timezone_str:  z.B. "Europe/Berlin"
        tenant_id:     EIOT-Tenant-ID (X-ESP-TENANT-ID)
        chunk_days:    maximaler Chunk in Tagen (Standard 90)

    Rückgabe im EIOT-Format:
        {
          "processedSamples": <summe über Chunks>,
          "samples": [ ... alle Samples über alle Chunks ... ]
        }

    Fehlerstrategie:
        - Wenn ein Chunk fehlschlägt, wird ein Error geloggt und die bis dahin
          gesammelten Samples zurückgegeben.
    """
    token = _load_eiot_token()
    if not token:
        raise EnvironmentError("EIOT: Kein Zugriffstoken verfügbar.")

    url = f"https://api.iot.eon.com/datapoints/v1/{datapoint_id}/samples:aggregate"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-ESP-TENANT-ID": tenant_id,
    }

    start_dt = _ensure_dt_utc(start_time)
    end_dt = _ensure_dt_utc(end_time)

    if start_dt >= end_dt:
        raise ValueError("fetch_datapoint_raw: start_time muss vor end_time liegen.")

    total_samples: List[Dict[str, Any]] = []
    total_processed = 0

    cur_start = start_dt
    chunk_idx = 0

    while cur_start < end_dt:
        cur_end = min(cur_start + timedelta(days=chunk_days), end_dt)
        chunk_idx += 1

        params = {
            "interval.startTime": _to_iso_z(cur_start),
            "interval.endTime": _to_iso_z(cur_end),
            "timeBucket.factor": bucket_factor,
            "timeBucket.unit": bucket_unit,
            "timeBucket.timeZone": timezone_str,
            "aggregation": aggregation,
        }

        logger.info(
            "EIOT: Hole Chunk %d für DP=%s, %s bis %s",
            chunk_idx,
            datapoint_id,
            params["interval.startTime"],
            params["interval.endTime"],
        )

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            j = resp.json()
        except requests.RequestException as e:
            # Fehler detailliert loggen (inkl. evtl. Response)
            status = getattr(getattr(e, "response", None), "status_code", None)
            body = getattr(getattr(e, "response", None), "text", None)
            logger.error(
                "Fehler beim Abruf von EIOT-Datapoint %s (Chunk %d): %s",
                datapoint_id,
                chunk_idx,
                e,
                exc_info=True,
            )
            if status is not None:
                logger.error(
                    "EIOT-Response-Status: %s, Body: %s, Params: %s",
                    status,
                    body,
                    params,
                )
            # Bei Fehler: Chunking beenden und das bisherige Ergebnis zurückgeben
            break

        samples = j.get("samples") or []
        processed = int(j.get("processedSamples", 0))

        logger.info(
            "EIOT: Chunk %d für DP=%s -> %d Samples, processedSamples=%d",
            chunk_idx,
            datapoint_id,
            len(samples),
            processed,
        )

        total_samples.extend(samples)
        total_processed += processed

        cur_start = cur_end

    result = {
        "processedSamples": total_processed,
        "samples": total_samples,
    }

    logger.info(
        "EIOT: Gesamt für DP=%s -> %d Samples, processedSamples=%d",
        datapoint_id,
        len(total_samples),
        total_processed,
    )

    return result


# ---------------------------------------------------------------------------
# 3) Response → pandas.Series
# ---------------------------------------------------------------------------

def datapoint_json_to_series(resp_json: dict, col_name: str) -> pd.Series:
    """
    Wandelt die EIOT-Antwort für einen Datapoint in eine pd.Series um.

    - Wenn 'samples' leer ist, wird eine leere Series (float64) zurückgegeben.
    - Zeitspalte wird automatisch aus ['timestamp', 'startTime', 'time'] gewählt.
    - Wertspalte wird automatisch aus ['value', 'avg', 'P', 'v'] gewählt.

    Index:   Index: datetime (UTC, tz-naiv)
    Werte:   float
    Name:    col_name
    """
    if not resp_json:
        logger.debug("datapoint_json_to_series: leere Response für %s", col_name)
        return pd.Series(dtype="float64", name=col_name)

    samples = resp_json.get("samples") or []
    if not samples:
        logger.info(
            "datapoint_json_to_series: 'samples' leer oder fehlt für %s. Keys: %s",
            col_name,
            list(resp_json.keys()),
        )
        return pd.Series(dtype="float64", name=col_name)

    df = pd.DataFrame(samples)

    # Zeitspalte finden
    time_col = None
    for cand in ["timestamp", "startTime", "time"]:
        if cand in df.columns:
            time_col = cand
            break

    # Wertspalte finden
    value_col = None
    for cand in ["value", "avg", "P", "v"]:
        if cand in df.columns:
            value_col = cand
            break

    if time_col is None or value_col is None:
        logger.warning(
            "datapoint_json_to_series: konnte Zeit-/Wertspalte nicht erkennen. "
            "Spalten: %s",
            list(df.columns),
        )
        return pd.Series(dtype="float64", name=col_name)

    dt = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    dt = pd.DatetimeIndex(dt).tz_convert(None)
    mask = ~dt.isna()
    if not mask.any():
        logger.warning("datapoint_json_to_series: alle Zeitstempel ungültig.")
        return pd.Series(dtype="float64", name=col_name)

    dt = dt[mask]
    vals = pd.to_numeric(df.loc[mask, value_col], errors="coerce").fillna(0.0)

    s = pd.Series(vals.values, index=dt, name=col_name)
    s = s.sort_index()
    return s

