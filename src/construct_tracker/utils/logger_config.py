import logging

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

def setup_logger():
    logger = logging.getLogger(__name__)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create formatters
        default_formatter = logging.Formatter('%(levelname)s: %(message)s')
        message_only_formatter = logging.Formatter('INFO: %(message)s')

        # Apply the default formatter to the console handler
        console_handler.setFormatter(default_formatter)
        logger.addHandler(console_handler)

        # Create another handler for info level with a different formatter
        info_handler = logging.StreamHandler()
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(message_only_formatter)
        info_handler.addFilter(InfoFilter())

        # Add the info handler to the logger
        logger.addHandler(info_handler)

        # Prevent duplicate messages by setting the level of the default handler to WARNING
        console_handler.setLevel(logging.WARNING)
    
    return logger
