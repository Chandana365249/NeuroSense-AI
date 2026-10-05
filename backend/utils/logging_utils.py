from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: str | None = None) -> logging.Logger:
    """Create an application logger with file output."""
    logger = logging.getLogger("neurosense")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if log_path:
            file_handler = logging.FileHandler(Path(log_path))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
