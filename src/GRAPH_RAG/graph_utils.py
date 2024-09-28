import os
import json
import pytz
import uuid
import argparse
import logging
import logging.config
import logging.handlers
from datetime import datetime
from langchain.schema import Document
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar


# Logging configuration
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Function to get root parent configuration logger.
    Child logger will pass info, debugs... log objects to parent's root logger handlers
    """
    CONFIG_LOGGER_FILE = os.path.join(os.path.dirname(__file__), '..\\..\\', 'config\\loggers', 'graph.json') 
    
    with open(CONFIG_LOGGER_FILE, encoding='utf-8') as f:
        content = json.load(f)
    logging.config.dictConfig(content)
    
    
def get_current_spanish_date_iso():
    """Get the current date and time in the Europe/Madrid time zone"""
    spanish_tz = pytz.timezone('Europe/Madrid')
    return str(datetime.now(spanish_tz).strftime("%Y-%m-%d %H:%M:%S"))


def get_current_spanish_date_iso_file_name_format():
    """Get the current date and time in the Europe/Madrid time zone"""
    spanish_tz = pytz.timezone('Europe/Madrid')
    return str(datetime.now(spanish_tz).strftime("%Y%m%d%H%M%S"))


def merge_page_content(docs : list[Document]) -> str:
    """Merge Document page_content list into unique str 

    Args:
        docs (list[Document])

    Returns:
        str
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_id() -> str:
    return str(uuid.uuid4())


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Parse and create the console script input arguments
    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser( 
                    prog='BOE',
                    description='SPANISH BOE CHATBOT')
    parser.add_argument('--data_path', type=str, required=False, help='Data file path[json format]')
    parser.add_argument('--mode', type=str, required=False, help='App mode')
    parser.add_argument('--config_path', type=str, required=False, help='Cofiguration file path [json format]')
    
    return parser

