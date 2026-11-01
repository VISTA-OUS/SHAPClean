"""Centralized logging configuration for the project."""

import logging

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger configured with the project-wide log format.

    Args:
        name: Logger name, normally ``__name__`` of the calling module.
        level: Logging level applied only the first time the root logger is
            configured (subsequent calls reuse the existing configuration).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(name)