# src/utils/logging_config.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(
    log_dir: str = "logs",
    log_file: str = "service.log",
    level: int = logging.WARNING,
) -> None:
    root = logging.getLogger()

    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Wenn schon konfiguriert: Level konsequent durchdrücken (root + handler)
    if root.handlers:
        root.setLevel(level)
        for h in root.handlers:
            h.setLevel(level)
            # falls Format noch nicht passt:
            try:
                h.setFormatter(formatter)
            except Exception:
                pass
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # WICHTIG: Handler-Level setzen
    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

