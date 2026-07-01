from loguru import logger
import sys
import os

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_FORMAT = (
    "<level>{level: <8}</level> "
    "{time:YYYY-MM-DD HH:mm:ss} "
    "{process} "
    "{file}:{line} | "
    "<level>{message}</level>"
)


def get_logger(name: str = None, level: str = "INFO", log_file: str = None):
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"Invalid LOG_LEVEL: {level}. Must be one of {sorted(LOG_LEVELS)}")

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_FORMAT,
        colorize=True,
        enqueue=True,           
    )

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format=_FORMAT,
            encoding="utf-8",
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
        )

    return logger.bind(name=name) if name else logger