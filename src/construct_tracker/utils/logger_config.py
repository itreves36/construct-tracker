"""How to handle logging."""

import logging


class InfoFilter(logging.Filter):
    """
    A logging filter that only allows INFO level messages.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filters log records to allow only INFO level messages.

        Args:
            record (logging.LogRecord): The log record to be filtered.

        Returns:
            bool: True if the record's level is INFO, False otherwise.
        """
        return record.levelno == logging.INFO


def setup_logger() -> logging.Logger:
    """
    Sets up a logger with different handlers and formatters for various logging levels.

    The logger outputs DEBUG level messages to the console with a default formatter,
    and INFO level messages with a custom formatter.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        logger = setup_logger()
        logger.info("This is an info message.")
        logger.debug("This is a debug message.")
    """
    logger = logging.getLogger(__name__)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create formatters
        default_formatter = logging.Formatter("%(levelname)s: %(message)s")
        message_only_formatter = logging.Formatter("INFO: %(message)s")

        # Apply the default formatter to the console handler
        console_handler.setFormatter(default_formatter)
        logger.addHandler(console_handler)

        # Create another handler for INFO level with a different formatter
        info_handler = logging.StreamHandler()
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(message_only_formatter)
        info_handler.addFilter(InfoFilter())

        # Add the info handler to the logger
        logger.addHandler(info_handler)

        # Prevent duplicate messages by setting the level of the default handler to WARNING
        console_handler.setLevel(logging.WARNING)

    return logger
