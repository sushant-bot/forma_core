"""
FormaCore AI - config.py
Shared application defaults and logging setup for product mode.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


FORMACORE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = FORMACORE_ROOT.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs"

APP_NAME = "FormaCore AI"
APP_VERSION = "1.0.0"

DEFAULT_LOG_LEVEL = os.getenv("FORMACORE_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DEFAULT_GA_SETTINGS = {
    "population_size": 15,
    "generations": 20,
    "early_stop_gens": 5,
}

FAST_DEMO_GA_SETTINGS = {
    "population_size": 10,
    "generations": 10,
    "early_stop_gens": 3,
}

DEFAULT_HEAT_SIGMA = 10.0
DEFAULT_RESOLUTION_MM = 0.5


def setup_logging(log_file: Optional[Path | str] = None,
                  level: str | int | None = None,
                  logger_name: str = APP_NAME) -> logging.Logger:
    """
    Configure root logging once and optionally attach a file handler.

    The function is idempotent, so it is safe to call on Streamlit reruns.
    """
    resolved_level = _resolve_level(level)
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(level=resolved_level, format=LOG_FORMAT)
    else:
        root.setLevel(resolved_level)
        for handler in root.handlers:
            handler.setLevel(resolved_level)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        abs_log_path = str(log_path.resolve())

        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) != abs_log_path:
                root.removeHandler(handler)
                handler.close()

        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == abs_log_path
            for handler in root.handlers
        ):
            file_handler = logging.FileHandler(abs_log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            file_handler.setLevel(resolved_level)
            root.addHandler(file_handler)

    return logging.getLogger(logger_name)


def output_dir_for_run(run_id: str) -> Path:
    """Return the output folder path for a named run."""
    return DEFAULT_OUTPUT_ROOT / run_id


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level

    if isinstance(level, str) and level:
        candidate = getattr(logging, level.upper(), None)
        if isinstance(candidate, int):
            return candidate

    candidate = getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO)
    return candidate if isinstance(candidate, int) else logging.INFO