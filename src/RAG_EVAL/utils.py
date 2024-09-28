import os
import json
import uuid
import tiktoken
import asyncio
import pytz
import logging
import logging.config
import logging.handlers
from datetime import datetime, timezone

# Logging configuration
logger = logging.getLogger("ETL_module_logger")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config

def setup_logging() -> None:
    """
    Function to get root parent configuration logger.
    Child logger will pass info, debugs... log objects to parent's root logger handlers
    """
    CONFIG_LOGGER_FILE = os.path.join(os.path.abspath("./config/loggers"), "rag_eval.json")
    print(f"CONFIG_LOGGER_FILE : {CONFIG_LOGGER_FILE}")
        
    with open(CONFIG_LOGGER_FILE, encoding='utf-8') as f:
        content = json.load(f)
    logging.config.dictConfig(content)
    
    
def get_current_spanish_date_iso():
    # Get the current date and time in the Europe/Madrid time zone
    spanish_tz = pytz.timezone('Europe/Madrid')
    return datetime.now(spanish_tz).strftime("%Y%m%d%H%M%S")