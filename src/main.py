"""
Main entry point für den Leistungsband-Service.
Führt die komplette Pipeline aus:
1. Daten laden
2. Bereinigung
3. Netzmodell laden
4. PTDF-Berechnung
5. BESS-Bereinigung
6. Prognose
7. DC-Lastfluss
8. Leistungsband-Bestimmung
"""

from src.utils.logging_config import setup_logging
import logging

from src.data.data_loader import load_all_data
from src.data.data_cleaning import clean_data  # <<< NEU

# Diese Imports kannst du später wieder aktivieren, wenn die Module fertig sind.
# from src.network.network_model import load_network_model
# from src.network.ptdf import compute_ptdf
# from src.network.bess_cleaning import remove_bess_effects
# from src.forecast.model_runner import run_forecast
# from src.network.dc_loadflow import compute_dc_loadflow
# from src.band.band_calculation import compute_power_bands


def main(run_full_pipeline: bool = False):
    """
    Wenn run_full_pipeline=False:
        -> Schritt 1 (Daten laden) + Schritt 2 (Bereinigen),
           ideal für Tests der Loader/Cleaner.
    Wenn run_full_pipeline=True:
        -> komplette Pipeline (erst sinnvoll, wenn alle Module implementiert sind).
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starte Leistungsband-Service...")

    # 1 — Daten laden
    data = load_all_data()
    logger.info(
        "Daten geladen: %d Nodes, %d Messreihen, %d Wetter-Hist, %d Wetter-Forecast.",
        len(data.get("nodes", [])),
        len(data.get("measurements", {})),
        len(data.get("weather_hist", {})),
        len(data.get("weather_forecast", {})),
    )

    # 2 — Daten bereinigen
    data_clean = clean_data(data)
    logger.info("Daten bereinigt.")

    if not run_full_pipeline:
        logger.info("Stoppe nach Schritt 2 (Datenladen + Bereinigung, Testmodus).")
        # Beide Varianten zurückgeben, damit du im Notebook vergleichen kannst
        return {"raw": data, "clean": data_clean}

    # # Ab hier nur ausführen, wenn die restliche Pipeline implementiert ist
    # from src.network.network_model import load_network_model
    # from src.network.ptdf import compute_ptdf
    # from src.network.bess_cleaning import remove_bess_effects
    # from src.forecast.model_runner import run_forecast
    # from src.network.dc_loadflow import compute_dc_loadflow
    # from src.band.band_calculation import compute_power_bands

    # net = load_network_model()
    # logger.info("Netzmodell geladen.")

    # ptdf = compute_ptdf(net)
    # logger.info("PTDF-Matrix berechnet.")

    # data_without_bess = remove_bess_effects(data_clean, ptdf)
    # logger.info("BESS-Einflüsse bereinigt.")

    # forecast = run_forecast(data_without_bess)
    # logger.info("Prognose erstellt.")

    # flows = compute_dc_loadflow(net, forecast)
    # logger.info("DC-Lastfluss berechnet.")

    # bands = compute_power_bands(flows, ptdf)
    # logger.info("Leistungsbänder erfolgreich berechnet.")

    # return bands
    return data_clean  # Platzhalter, bis der Rest fertig ist


if __name__ == "__main__":
    # Im aktuellen Entwicklungsstand: Laden + Bereinigen testen
    main(run_full_pipeline=False)
