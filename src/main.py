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
from src.data.data_cleaning import clean_data
from src.network.network_model import load_network_model
from src.network.ptdf import compute_ptdf
from src.network.bess_cleaning import remove_bess_effects
from src.forecast.model_runner import run_forecast
from src.network.dc_loadflow import compute_dc_loadflow
from src.band.band_calculation import compute_power_bands


def main():
    # Logging einschalten
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starte Leistungsband-Service...")

    # 1 — Daten laden
    data = load_all_data()
    logger.info("Daten geladen.")

    # 2 — Daten bereinigen
    data_clean = clean_data(data)
    logger.info("Daten bereinigt.")

    # 3 — Netzmodell laden
    net = load_network_model()
    logger.info("Netzmodell geladen.")

    # 4 — PTDF berechnen
    ptdf = compute_ptdf(net)
    logger.info("PTDF-Matrix berechnet.")

    # 5 — BESS-Einflüsse entfernen
    data_without_bess = remove_bess_effects(data_clean, ptdf)
    logger.info("BESS-Einflüsse bereinigt.")

    # 6 — Prognose erzeugen
    forecast = run_forecast(data_without_bess)
    logger.info("Prognose erstellt.")

    # 7 — Lastflüsse ohne BESS bestimmen
    flows = compute_dc_loadflow(net, forecast)
    logger.info("DC-Lastfluss berechnet.")

    # 8 — Leistungsbänder ableiten
    bands = compute_power_bands(flows, ptdf)
    logger.info("Leistungsbänder erfolgreich berechnet.")

    return bands


if __name__ == "__main__":
    main()
