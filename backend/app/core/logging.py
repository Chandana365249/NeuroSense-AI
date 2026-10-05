from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_SENSITIVE_KEYS = {"password", "token", "access_token", "authorization", "jwt", "secret", "hashed_password"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra= fields are included, sensitive keys are redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = "***" if key.lower() in _SENSITIVE_KEYS else value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    root = logging.getLogger("neurosense")
    root.setLevel(level.upper())
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = JsonFormatter()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "neurosense.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"neurosense.{name}")
