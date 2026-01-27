# src/main.py
"""
End-to-end Pipeline (A -> Z):
1) load_all_data()            -> raw measurements + weather (hist + forecast) + writes raw CSVs
2) clean_data(bundle)         -> in-memory cleaning + coverage report
3) BESS cleaning (ridge)      -> timeseries_no_bess CSVs
4) forecast_all_nodes()       -> preds (forecast.py)
5) battery_bands.run()        -> powerband CSVs je battery node

Run from repo root:
    python -m src.main

Optionally skip steps:
    python -m src.main --skip-load
    python -m src.main --skip-forecast --skip-powerband
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src import config as cfg
from src.utils.logging_config import setup_logging
from src.network.bess_cleaning import clean_all_nodes_remove_bess_from_graph

logger = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    """Ensure directories used by the pipeline exist."""
    required = []

    if hasattr(cfg, "REQUIRED_DIRS"):
        required.extend(list(cfg.REQUIRED_DIRS))

    for name in [
        "RAW_TS_DIR",
        "CLEAN_TS_DIR",
        "WEATHER_FORECAST_DIR",
        "PRED_TS_DIR",
        "POWERBAND_DIR",
        "LOG_DIR",
    ]:
        p = getattr(cfg, name, None)
        if p is not None:
            required.append(p)

    # de-dup while keeping order
    seen = set()
    uniq = []
    for p in required:
        pp = str(p)
        if pp in seen:
            continue
        seen.add(pp)
        uniq.append(p)

    for p in uniq:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("Could not create required dir: %s", p)
            raise


def _validate_config() -> None:
    """Basic sanity checks to fail early with clear messages."""
    graph_path = Path(getattr(cfg, "GRAPH_PATH"))
    if not graph_path.exists():
        raise FileNotFoundError(f"GRAPH_PATH not found: {graph_path.resolve()}")

    util = float(getattr(cfg, "UTIL_TARGET_PCT", 100.0))
    if not (0.0 < util <= 100.0):
        raise ValueError(f"UTIL_TARGET_PCT must be in (0, 100], got {util}")

    slack = str(getattr(cfg, "SLACK_NODE_ID", "")).strip()
    if not slack:
        raise ValueError("SLACK_NODE_ID is empty in config.")


def _run_load_data(
    start_time=None,
    end_time=None,
    weather_forecast_hours: int = 30,
) -> Dict[str, Any]:
    """Pull / update raw measurements & weather; returns the in-memory bundle."""
    from src.data.data_loader import load_all_data

    logger.info(
        "STEP load_all_data(start=%s end=%s forecast_hours=%s)",
        start_time,
        end_time,
        weather_forecast_hours,
    )

    bundle = load_all_data(
        start_time=start_time if start_time is not None else cfg.HIST_START,
        end_time=end_time,
        weather_forecast_hours=int(weather_forecast_hours),
    )

    measurements = bundle.get("measurements", {}) or {}
    weather_hist = bundle.get("weather_hist", {}) or {}
    weather_forecast = bundle.get("weather_forecast", {}) or {}

    logger.info("Loaded measurements nodes=%d", len(measurements))
    logger.info("Loaded weather_hist nodes=%d", len(weather_hist))
    logger.info("Loaded weather_forecast nodes=%d", len(weather_forecast))

    return bundle


def _run_clean_data(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Clean the in-memory bundle and write the coverage report."""
    from src.data.data_cleaning import clean_data

    logger.info("STEP clean_data(bundle)")
    cleaned = clean_data(bundle)
    logger.info("clean_data done (coverage report in logs/)")
    return cleaned


def _run_bess_cleaning() -> None:
    """Remove BESS influence using multivariate Ridge regression."""
    

    logger.info("STEP bess_cleaning (Ridge)")

    clean_all_nodes_remove_bess_from_graph(
        graph_path=Path(cfg.GRAPH_PATH),
        raw_ts_dir=Path(cfg.PREP_TS_DIR),
        ts_col="timestamp",
        val_col="P_MW",
        min_overlap_points=200,
        out_dir=Path(cfg.CLEAN_TS_DIR),
        include_intercept_in_removal=False,
        ridge_alpha=1.0,
        write_report_csv=True,
    )

    logger.info("bess_cleaning done -> %s", Path(cfg.CLEAN_TS_DIR).resolve())




def _run_forecast(overwrite: bool = True, max_hours_cap: Optional[float] = None) -> None:
    """Forecast all nodes using winner model; writes pred."""
    from src.forecast import forecast_all_nodes

    logger.info("STEP forecast_all_nodes(overwrite=%s max_hours_cap=%s)", overwrite, max_hours_cap)
    df = forecast_all_nodes(overwrite=overwrite, max_hours_cap=max_hours_cap)

    ok_n = int((df["ok"] == True).sum()) if "ok" in df.columns else None  
    fail_n = int((df["ok"] == False).sum()) if "ok" in df.columns else None
    logger.info("forecast_all_nodes done (ok=%s fail=%s)", ok_n, fail_n)


def _run_powerbands() -> None:
    """Compute battery bands and write one CSV per BESS."""
    import src.battery_bands as bb

    logger.info("STEP battery_bands.run() -> out=%s", Path(cfg.POWERBAND_DIR).resolve())
    bb.run()
    logger.info("battery_bands done -> %s", Path(cfg.POWERBAND_DIR).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Masterthesis service pipeline (A->Z).")

    # Step toggles
    parser.add_argument("--skip-load", action="store_true", help="Skip load_all_data() (API pull).")
    parser.add_argument("--skip-clean", action="store_true", help="Skip clean_data(bundle).")
    parser.add_argument("--skip-bess-clean", action="store_true", help="Skip BESS influence removal.")
    parser.add_argument("--skip-forecast", action="store_true", help="Skip forecasting step.")
    parser.add_argument("--skip-powerband", action="store_true", help="Skip powerband calculation.")

    # Forecast options
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite prediction CSVs.")
    parser.add_argument("--max-hours-cap", type=float, default=None, help="Cap forecast horizon in hours (optional).")

    # Load-data options
    parser.add_argument("--weather-forecast-hours", type=int, default=30, help="Weather forecast horizon (hours).")

    args = parser.parse_args()

    setup_logging(
        log_dir=str(getattr(cfg, "LOG_DIR", "logs")),
        log_file="service.log",
        level=getattr(cfg, "LOG_LEVEL", logging.WARNING),
    )

    try:
        _ensure_dirs()
        _validate_config()

        logger.info("=== PIPELINE START ===")
        logger.info("GRAPH_PATH=%s", Path(cfg.GRAPH_PATH).resolve())
        logger.info("RAW_TS_DIR=%s", Path(cfg.RAW_TS_DIR).resolve())
        logger.info("CLEAN_TS_DIR=%s", Path(cfg.CLEAN_TS_DIR).resolve())
        logger.info("PRED_TS_DIR=%s", Path(cfg.PRED_TS_DIR).resolve())
        logger.info("POWERBAND_DIR=%s", Path(cfg.POWERBAND_DIR).resolve())
        logger.info("UTIL_TARGET_PCT=%.2f", float(cfg.UTIL_TARGET_PCT))

        bundle: Optional[Dict[str, Any]] = None

        if not args.skip_load:
            bundle = _run_load_data(weather_forecast_hours=args.weather_forecast_hours)
        else:
            logger.warning("SKIP load_all_data()")

        if not args.skip_clean:
            if bundle is None:
                raise RuntimeError("clean_data(bundle) requires load_all_data() output. Run without --skip-load.")
            bundle = _run_clean_data(bundle)
        else:
            logger.warning("SKIP clean_data(bundle)")

        if not args.skip_bess_clean:
            _run_bess_cleaning()
        else:
            logger.warning("SKIP bess_cleaning()")

        if not args.skip_forecast:
            _run_forecast(overwrite=(not args.no_overwrite), max_hours_cap=args.max_hours_cap)
        else:
            logger.warning("SKIP forecast_all_nodes()")

        if not args.skip_powerband:
            _run_powerbands()
        else:
            logger.warning("SKIP battery_bands.run()")

        logger.info("=== PIPELINE DONE ===")
        return 0

    except Exception:
        logger.exception("PIPELINE FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
