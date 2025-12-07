# src/data/weather_loader.py

import logging
from typing import Tuple, Optional

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
        resp = requests.get(url, params=params, timeout=30, verify=False)  # FIXME: verify=True in Prod
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

    # Zeitspalte parsen
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # >>> NEU: Zeitstempel auf 15-Minuten-Raster runden
    df["time"] = df["time"].dt.round("15min")

    # Falls durch das Runden Duplikate entstehen: nach time aggregieren
    df = df.groupby("time", as_index=False).mean()

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


def fetch_weather_open_meteo(
    latitude: float,
    longitude: float,
    past_hours: int = 24,
    forecast_hours: int = 24,
    timezone: str = "Europe/Berlin",
    model: str = "icon_seamless",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Holt 15-minütige Wetterdaten von Open-Meteo (DWD / ICON-Seamless):

    - Temperatur in 2 m Höhe
    - Windgeschwindigkeit in 10 m Höhe
    - Kurzwellige Strahlung (Globalstrahlung)

    Historische Daten werden aus dem Historical-Forecast-Katalog geladen
    (historical-forecast-api.open-meteo.com), Prognosen weiterhin aus dem
    normalen Forecast-Endpoint (api.open-meteo.com).

    Rückgabe:
        (df_hist, df_forecast)
        Beide DataFrames haben einen DatetimeIndex (time) und Spalten:
            - temperature_C
            - wind_speed_mps
            - solar_radiation_Wm2
    """

    logger.info(
        "Hole Wetterdaten von Open-Meteo (lat=%.6f, lon=%.6f, past=%dh, forecast=%dh)",
        latitude,
        longitude,
        past_hours,
        forecast_hours,
    )

    # -------------------------------------------------------------------------
    # 1) Historische Daten: Historical-Forecast-API, explizites Zeitfenster
    # -------------------------------------------------------------------------
    df_hist = pd.DataFrame()
    if past_hours > 0:
        now = pd.Timestamp.now(tz=timezone)
        start_hist = now - pd.Timedelta(hours=past_hours)

        url_hist = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params_hist = {
            "latitude": latitude,
            "longitude": longitude,
            "minutely_15": "temperature_2m,wind_speed_10m,shortwave_radiation",
            # explizites 15-minütiges Zeitfenster
            "start_minutely_15": start_hist.strftime("%Y-%m-%dT%H:%M"),
            "end_minutely_15": now.strftime("%Y-%m-%dT%H:%M"),
            "timezone": timezone,
            "models": model,
        }

        logger.info(
            "Historische Wetterdaten: %s bis %s (past_hours=%d)",
            params_hist["start_minutely_15"],
            params_hist["end_minutely_15"],
            past_hours,
        )

        df_hist = _fetch_minutely15(url_hist, params_hist, context="historisch")

    # -------------------------------------------------------------------------
    # 2) Forecast-Daten: normaler Forecast-Endpoint, relative Länge
    # -------------------------------------------------------------------------
    df_forecast = pd.DataFrame()
    if forecast_hours > 0:
        forecast_steps = forecast_hours * 4  # 4 * 15min = 1h

        url_fc = "https://api.open-meteo.com/v1/forecast"
        params_fc = {
            "latitude": latitude,
            "longitude": longitude,
            "minutely_15": "temperature_2m,wind_speed_10m,shortwave_radiation",
            "forecast_minutely_15": forecast_steps,
            "timezone": timezone,
            "models": model,
        }

        logger.info(
            "Prognose-Wetterdaten: nächste %d Stunden (forecast_minutely_15=%d)",
            forecast_hours,
            forecast_steps,
        )

        df_forecast = _fetch_minutely15(url_fc, params_fc, context="forecast")

    # -------------------------------------------------------------------------
    # 3) Logging & Umbenennung der Spalten
    # -------------------------------------------------------------------------
    logger.info("Anzahl historischer Punkte: %d", len(df_hist))
    logger.info("Anzahl Prognose-Punkte: %d", len(df_forecast))

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

    return df_hist, df_forecast


def save_weather_to_csv(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    hist_path: Optional[str] = None,
    forecast_path: Optional[str] = None,
) -> None:
    """
    Optionaler Helper zum Speichern der Wetterdaten in CSV.
    Pfade können separat übergeben werden.
    """
    if hist_path:
        df_hist.to_csv(hist_path, index_label="timestamp")
        logger.info(
            "Historische Wetterdaten gespeichert nach %s (Zeilen: %d)",
            hist_path,
            len(df_hist),
        )
    if forecast_path:
        df_forecast.to_csv(forecast_path, index_label="timestamp")
        logger.info(
            "Prognose-Wetterdaten gespeichert nach %s (Zeilen: %d)",
            forecast_path,
            len(df_forecast),
        )
