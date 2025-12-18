# src/data/weather_loader.py

import logging
from typing import Tuple, Optional, Dict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


logger = logging.getLogger(__name__)


def _fetch_minutely15(
    url: str,
    params: dict,
    context: str,
) -> pd.DataFrame:
    """
    Interner Helper, der einen Open-Meteo-Request macht und die minutely_15-Daten
    als DataFrame mit geparster time-Spalte zurückgibt.
    """
    try:
        resp = requests.get(url, params=params, timeout=30, verify=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Fehler beim Abruf der %s-Wetterdaten: %s", context, e)
        raise

    data = resp.json()

    if "minutely_15" not in data:
        logger.error(
            "Antwort von Open-Meteo (%s) enthält kein 'minutely_15'-Feld. Keys: %s",
            context,
            list(data.keys()),
        )
        raise ValueError(f"Ungültige Open-Meteo-Antwort ({context}): 'minutely_15' fehlt.")

    df = pd.DataFrame(data["minutely_15"])

    if "time" not in df.columns:
        logger.error(
            "Spalte 'time' fehlt in Open-Meteo-Antwort (%s). Spalten: %s",
            context,
            list(df.columns),
        )
        raise ValueError(f"Ungültige Open-Meteo-Antwort ({context}): 'time' fehlt.")

    # Zeitspalte parsen
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Auf 15-Minuten-Raster ziehen (floor ist stabiler als round am Rand)
    df["time"] = df["time"].dt.floor("15min")

    # Falls Duplikate entstehen: nach time aggregieren (mean über numerische Spalten)
    df = df.groupby("time", as_index=False).mean(numeric_only=True)

    # Auflösung prüfen
    if len(df) > 1:
        step = df["time"].iloc[1] - df["time"].iloc[0]
        if step == pd.Timedelta("15min"):
            logger.info("[%s] 15-Minuten-Auflösung erkannt.", context)
        else:
            logger.warning(
                "[%s] Unerwartete Zeitauflösung: %s (erwartet: 15min)",
                context,
                step,
            )
    else:
        logger.warning(
            "[%s] Weniger als 2 Zeitpunkte in Wetterdaten, Auflösung nicht prüfbar.",
            context,
        )

    return df


def _ensure_ts(t: datetime | str) -> pd.Timestamp:
    """
    Wandelt datetime oder ISO-String in pandas.Timestamp um (timezone-aware bleibt erhalten).
    """
    return pd.to_datetime(t)


def fetch_weather_open_meteo(
    latitude: float,
    longitude: float,
    start_time: datetime | str,
    end_time: datetime | str,
    timezone: str = "UTC",
    model: str = "icon_seamless",
    forecast_hours: int = 30,  # Default 30h (120 Werte)
    output_timezone: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Holt 15-minütige Wetterdaten von Open-Meteo (DWD / ICON-Seamless):

    - Temperatur in 2 m Höhe
    - Windgeschwindigkeit in 10 m Höhe
    - Kurzwellige Strahlung (Globalstrahlung)

    WICHTIG:
      Wir fragen Open-Meteo standardmäßig in UTC ab (timezone="UTC"), um
      1h/2h Offsets und Randlücken bei lokaler Zeitzone zu vermeiden.

    Forecast-Logik (angepasst):
      Forecast startet NICHT relativ zu "jetzt", sondern exakt beim nächsten
      15-Minuten-Zeitstempel nach dem historischen Ende.
      Beispiel: hist endet 09:00 -> Forecast startet 09:15.

    Optional kannst du output_timezone setzen (z.B. "Europe/Berlin"), dann werden
    die Indizes am Ende nach lokal konvertiert (naiv, ohne tzinfo).
    """

    logger.info(
        "Hole Wetterdaten von Open-Meteo (lat=%.6f, lon=%.6f, start=%s, end=%s, forecast_hours=%d, tz=%s)",
        latitude,
        longitude,
        start_time,
        end_time,
        forecast_hours,
        timezone,
    )

    # -------------------------------------------------------------------------
    # 1) Historische Daten: Historical-Forecast-API mit explizitem Zeitfenster
    # -------------------------------------------------------------------------
    start_ts = _ensure_ts(start_time)
    end_ts = _ensure_ts(end_time)

    # Auf UTC normalisieren (wenn naive Inputs kommen, interpretieren wir sie als UTC)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    if start_ts >= end_ts:
        raise ValueError("fetch_weather_open_meteo: start_time muss vor end_time liegen.")

    # auf 15-Minuten-Raster runden (UTC-aware)
    start_ts = start_ts.floor("15min")
    end_ts = end_ts.floor("15min")

    url_hist = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params_hist = {
        "latitude": latitude,
        "longitude": longitude,
        "minutely_15": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "start_minutely_15": start_ts.strftime("%Y-%m-%dT%H:%M"),
        "end_minutely_15": end_ts.strftime("%Y-%m-%dT%H:%M"),
        "timezone": timezone,  # i.d.R. "UTC"
        "models": model,
    }

    logger.info(
        "Historische Wetterdaten: %s bis %s (timezone=%s)",
        params_hist["start_minutely_15"],
        params_hist["end_minutely_15"],
        params_hist["timezone"],
    )

    df_hist = _fetch_minutely15(url_hist, params_hist, context="historisch")

    # -------------------------------------------------------------------------
    # 2) Forecast-Daten: Forecast-Endpoint, aber Zeitfenster EXPLIZIT
    #    Start = nächster 15min Slot nach hist-Ende
    # -------------------------------------------------------------------------
    df_forecast = pd.DataFrame()
    if forecast_hours > 0:
        expected_steps = forecast_hours * 4

        # Letzter historischer Zeitpunkt (aus den Daten, nicht nur aus end_ts)
        # Falls df_hist leer ist, fallen wir auf end_ts (angefragt) zurück.
        if not df_hist.empty:
            hist_last = pd.to_datetime(df_hist["time"].max())
        else:
            hist_last = end_ts.tz_localize(None)

        # Forecast soll beim nächsten 15-Minuten Slot starten
        forecast_start = (hist_last + pd.Timedelta(minutes=15)).floor("15min")
        # Ende so setzen, dass wir genug Slots bekommen
        forecast_end = (forecast_start + pd.Timedelta(hours=forecast_hours)).floor("15min")

        url_fc = "https://api.open-meteo.com/v1/forecast"
        params_fc = {
            "latitude": latitude,
            "longitude": longitude,
            "minutely_15": "temperature_2m,wind_speed_10m,shortwave_radiation",
            "start_minutely_15": forecast_start.strftime("%Y-%m-%dT%H:%M"),
            "end_minutely_15": forecast_end.strftime("%Y-%m-%dT%H:%M"),
            "timezone": timezone,  # i.d.R. "UTC"
            "models": model,
        }

        logger.info(
            "Prognose-Wetterdaten: %s bis %s (expected_steps=%d, timezone=%s)",
            params_fc["start_minutely_15"],
            params_fc["end_minutely_15"],
            expected_steps,
            params_fc["timezone"],
        )

        df_forecast = _fetch_minutely15(url_fc, params_fc, context="forecast")

        # Hart auf genau N Werte trimmen (API kann inkl./exkl. am Ende variieren)
        if len(df_forecast) >= expected_steps:
            df_forecast = df_forecast.iloc[:expected_steps].copy()
        else:
            logger.warning(
                "Forecast hat zu wenige Werte: %d statt %d (start=%s).",
                len(df_forecast),
                expected_steps,
                forecast_start,
            )

        # Hard check: Startzeit muss exakt forecast_start sein
        if not df_forecast.empty:
            got_start = df_forecast["time"].iloc[0]
            if got_start != forecast_start:
                logger.warning(
                    "Forecast-Start unerwartet: %s statt %s (Gap=%s).",
                    got_start,
                    forecast_start,
                    got_start - forecast_start,
                )

        # Hard check: 15-min Raster
        if len(df_forecast) > 1:
            step = df_forecast["time"].iloc[1] - df_forecast["time"].iloc[0]
            if step != pd.Timedelta("15min"):
                logger.warning("Forecast step ist %s statt 15min.", step)

    # -------------------------------------------------------------------------
    # 3) Umbenennung der Spalten + Index setzen
    # -------------------------------------------------------------------------
    rename_map = {
        "temperature_2m": "temperature_C",
        "wind_speed_10m": "wind_speed_mps",
        "shortwave_radiation": "solar_radiation_Wm2",
    }

    if not df_hist.empty:
        df_hist = df_hist.rename(columns=rename_map).set_index("time")
    else:
        df_hist = pd.DataFrame(columns=list(rename_map.values())).set_index(
            pd.DatetimeIndex([], name="time")
        )

    if not df_forecast.empty:
        df_forecast = df_forecast.rename(columns=rename_map).set_index("time")
    else:
        df_forecast = pd.DataFrame(columns=list(rename_map.values())).set_index(
            pd.DatetimeIndex([], name="time")
        )

    # -------------------------------------------------------------------------
    # 4) Optional: Index von UTC -> gewünschte lokale Zone konvertieren
    # -------------------------------------------------------------------------
    if output_timezone:
        for df in (df_hist, df_forecast):
            if len(df.index) > 0:
                df.index = (
                    df.index.tz_localize(timezone)  # z.B. UTC
                    .tz_convert(output_timezone)    # z.B. Europe/Berlin
                    .tz_localize(None)              # zurück zu naive timestamps
                )

    return df_hist, df_forecast


def save_weather_to_csv(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    hist_path: Optional[str] = None,
    forecast_path: Optional[str] = None,
) -> None:
    """
    Optionaler Helper zum Speichern der Wetterdaten in CSV.
    """
    if hist_path:
        Path(hist_path).parent.mkdir(parents=True, exist_ok=True)
        df_hist.to_csv(hist_path, index_label="timestamp")
        logger.info("Historische Wetterdaten gespeichert nach %s (Zeilen: %d)", hist_path, len(df_hist))

    if forecast_path:
        Path(forecast_path).parent.mkdir(parents=True, exist_ok=True)
        df_forecast.to_csv(forecast_path, index_label="timestamp")
        logger.info("Prognose-Wetterdaten gespeichert nach %s (Zeilen: %d)", forecast_path, len(df_forecast))


def save_weather_forecast_dict_to_csv(
    weather_forecast: Dict[str, pd.DataFrame],
    out_dir: str = "src/data/raw/weather_forecast",
) -> None:
    """
    Speichert Forecast-Wetterdaten pro node_id als CSV:
      <out_dir>/<node_id>_weather_forecast.csv
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not weather_forecast:
        logger.warning("save_weather_forecast_dict_to_csv: weather_forecast ist leer.")
        return

    for node_id, df in weather_forecast.items():
        if df is None or df.empty:
            logger.warning("Forecast leer für node_id=%s -> skip", node_id)
            continue

        file_path = out_path / f"{node_id}_weather_forecast.csv"
        df.to_csv(file_path, index_label="timestamp")
        logger.info("Wetter-Forecast gespeichert: %s (Zeilen: %d)", str(file_path), len(df))
