"""Phoenix operator-console logging policy.

Phoenix keeps the terminal intentionally sparse. Full forensic detail is written
to file/run logs and the Queue Operations Monitor. Only records explicitly marked
for the operator CLI are allowed through Phoenix console handlers.
"""

from __future__ import annotations

import logging
from typing import Any


CLI_RECORD_ATTR = "phoenix_cli"


class PhoenixCliFilter(logging.Filter):
    """Allow only records explicitly intended for the operator terminal."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, CLI_RECORD_ATTR, False))


def configure_operator_console(logger: logging.Logger) -> None:
    """Apply the sparse Phoenix CLI policy to this logger's stream handlers."""
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ):
            handler.filters.clear()
            handler.addFilter(PhoenixCliFilter())


def cli_log(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: Any,
) -> None:
    """Emit one record to both Phoenix file logs and the sparse operator CLI."""
    logger.log(
        level,
        message,
        *args,
        extra={CLI_RECORD_ATTR: True},
    )


def cli_info(logger: logging.Logger, message: str, *args: Any) -> None:
    cli_log(logger, logging.INFO, message, *args)


def cli_warning(logger: logging.Logger, message: str, *args: Any) -> None:
    cli_log(logger, logging.WARNING, message, *args)


def cli_error(logger: logging.Logger, message: str, *args: Any) -> None:
    cli_log(logger, logging.ERROR, message, *args)
