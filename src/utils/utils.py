import json
import os
import logging


def setup_logging(file_name: str) -> None:
    """
    Function to get root parent configuration logger.
    Child logger will pass info, debugs... log objects to parent's root logger handlers
    """
    CONFIG_LOGGER_FILE = os.path.join(
        os.path.abspath("./config/loggers"), file_name)

    with open(CONFIG_LOGGER_FILE, encoding='utf-8') as f:
        content = json.load(f)
    logging.config.dictConfig(content)
