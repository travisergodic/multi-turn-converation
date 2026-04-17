import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("log")

_CONFIGURED = False
_LOG_FILE: Path | None = None


def _build_session_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"session_{timestamp}.log"


def setup_logging() -> None:
    global _CONFIGURED, _LOG_FILE
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = _build_session_log_file()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        ],
    )
    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging initialized log_file=%s", _LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def get_log_file() -> Path:
    setup_logging()
    assert _LOG_FILE is not None
    return _LOG_FILE
