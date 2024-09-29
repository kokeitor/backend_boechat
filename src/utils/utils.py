import json
import os
import pytz
import logging
from datetime import datetime


def setup_logging(file_name: str) -> None:
    """
    Function to get root parent configuration logger.
    Child logger will pass info, debugs... log objects to parent's root logger handlers
    """
    CONFIG_LOGGER_FILE = os.path.join(
        os.path.abspath("./config/loggers"), file_name)
    print(f"CONFIG_LOGGER_FILE : {CONFIG_LOGGER_FILE}")

    with open(CONFIG_LOGGER_FILE, encoding='utf-8') as f:
        content = json.load(f)
    logging.config.dictConfig(content)


def get_current_spanish_date_iso():
    """Get the current date and time in the Europe/Madrid time zone"""
    spanish_tz = pytz.timezone('Europe/Madrid')
    return str(datetime.now(spanish_tz).strftime("%Y-%m-%d %H:%M:%S"))
