"""
Main entry point für den Leistungsband-Service.
Führt die komplette Pipeline aus:
1. Daten laden
2. Bereinigung (Data Cleaning)
3. BESS-Bereinigung (Ridge)
4. Netzmodell laden
5. PTDF-Berechnung
6. Prognose
7. DC-Lastfluss
8. Leistungsband-Bestimmung
"""

from src.utils.logging_config import setup_logging
import logging
import time
from pathlib import Path

from src.data.data_loader import load_all_data
from src.data.data_cleaning import clean_data
from src.data.weather_loader import save_weather_forecast_dict_to_csv

from src.network.bess_cleaning import remove_bess_effects_from_csv_multi

try:
    # empfohlen: zentral in config pflegen
    from src.config import BESS_FILES
except Exception:
    BESS_FILES = ["BOLS_E41_hist.csv", "BOLS_E42_hist.csv"]  # Fallback


def main(run_full_pipeline: bool = False):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starte Leistungsband-Service...")

    t_total0 = time.perf_counter()

    # 1 — Daten laden (inkl. EIOT + Wetter)
    t0 = time.perf_counter()
    data = load_all_data()
    t_load = time.perf_counter() - t0
    logger.info("Datenziehen (load_all_data) abgeschlossen in %.2f s", t_load)

    # Forecast-CSVs pro node_id wegschreiben
    t1 = time.perf_counter()
    save_weather_forecast_dict_to_csv(
        data.get("weather_forecast", {}),
        out_dir="src/data/raw/weather_forecast",
    )
    t_save = time.perf_counter() - t1
    logger.info("Forecast-CSV Schreiben abgeschlossen in %.2f s", t_save)

    logger.info(
        "Daten geladen: %d Nodes, %d Messreihen, %d Wetter-Hist, %d Wetter-Forecast.",
        len(data.get("nodes", [])),
        len(data.get("measurements", {})),
        len(data.get("weather_hist", {})),
        len(data.get("weather_forecast", {})),
    )

    # 2 — Data Cleaning
    t2 = time.perf_counter()
    data_clean = clean_data(data)  # schreibt logs/data_coverage_report.csv (wenn du die Version drin hast)
    t_clean = time.perf_counter() - t2
    logger.info("Datenbereinigung (clean_data) abgeschlossen in %.2f s", t_clean)

    # 3 — BESS-Bereinigung (Ridge) + CSVs schreiben (CLEAN_TS_DIR)
    t3 = time.perf_counter()
    clean_bess_dict, bess_report = remove_bess_effects_from_csv_multi(
        bess_files=BESS_FILES,
        ts_col="timestamp",
        val_col="P_MW",
        min_overlap_points=200,
        out_dir=None,     # => CLEAN_TS_DIR aus config
        ridge_alpha=1.0,
    )
    t_bess = time.perf_counter() - t3
    logger.info("BESS-Bereinigung (Ridge) abgeschlossen in %.2f s", t_bess)

    # Optional: BESS-Report als CSV (sehr praktisch zum Debuggen)
    try:
        out_rep = Path("logs") / "bess_cleaning_report.csv"
        out_rep.parent.mkdir(parents=True, exist_ok=True)
        bess_report.to_csv(out_rep, index=False)
        logger.info("BESS-Report gespeichert: %s", out_rep)
    except Exception as e:
        logger.warning("Konnte BESS-Report nicht speichern: %s", e)

    # Optional: Kurzcheck, ob wirklich Dateien geschrieben wurden
    try:
        from src.config import CLEAN_TS_DIR
        out_dir = Path(CLEAN_TS_DIR)
        n_files = len(list(out_dir.glob("*_hist_clean.csv")))
        logger.info("CLEAN_TS_DIR enthält %d '*_hist_clean.csv' Dateien: %s", n_files, out_dir)
    except Exception:
        pass

    # Optional: Gesamtzeit bis hierhin
    t_total = time.perf_counter() - t_total0
    logger.info("Gesamtzeit bis Ende BESS-Cleaning: %.2f s", t_total)

    if not run_full_pipeline:
        logger.info("Stoppe nach BESS-Cleaning (Testmodus).")
        # beide Varianten zurückgeben (für Notebook/Debug)
        return {
            "raw": data,
            "clean": data_clean,
            "bess_clean_series": clean_bess_dict,
            "bess_report": bess_report,
        }

    # Rest der Pipeline später
    return {
        "clean": data_clean,
        "bess_clean_series": clean_bess_dict,
        "bess_report": bess_report,
    }


if __name__ == "__main__":
    main(run_full_pipeline=False)
