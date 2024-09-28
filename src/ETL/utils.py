import os
import json
import uuid
import pytz
import logging
import logging.config
import logging.handlers
from datetime import datetime
import time
import functools


# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("ETL_module_logger")
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


def setup_logging() -> None:
    """
    Function to get root parent configuration logger.
    Child logger will pass info, debugs... log objects to parent's root logger handlers
    """
    CONFIG_LOGGER_FILE = os.path.join(
        os.path.abspath("./config/loggers"), "etl.json")

    with open(CONFIG_LOGGER_FILE, encoding='utf-8') as f:
        content = json.load(f)
    logging.config.dictConfig(content)

# util functions


def get_current_spanish_date_iso():
    # Get the current date and time in the Europe/Madrid time zone
    spanish_tz = pytz.timezone('Europe/Madrid')
    return datetime.now(spanish_tz).strftime("%Y%m%d%H%M%S")

# Parse config


def parse_config(config_path) -> dict:
    logger.info(f"Path config : {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, encoding='utf-8') as file:
        config = json.load(file)
    return config


def get_id() -> str:
    return str(uuid.uuid4())


def exec_time(func):
    """Decorator to measure the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the difference
        print(
            f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper
