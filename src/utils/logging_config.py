import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "service.log",
    level: int = logging.INFO, # Change for different logging-style
):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Format für alle Logs
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5),
            logging.StreamHandler(),
        ],
        force=True,  
    )

    logging.getLogger(__name__).info("Logging initialisiert. Logfile: %s", log_path)
