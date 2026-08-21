"""Logging helpers for the embedding server."""

import logging


def configure_logging() -> None:
    """Configure structured logging once for the process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
