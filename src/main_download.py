import os
import requests
import json
from requests import Response
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import logging
import logging.config
import logging.handlers
from ETL.utils import get_current_spanish_date_iso, setup_logging, parse_config
from ETL.download import WebDownloadData, Downloader

# Logging configuration
logger = logging.getLogger("Download_web_files_module")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


def main() -> None:
    load_dotenv()
    
    # set up the root logger configuration
    setup_logging()
    
    # BOE DOWNLOAD DATA
    BOE_WEB_URL = str(os.getenv('BOE_WEB_URL'))
    print(BOE_WEB_URL)
    logger.info(BOE_WEB_URL)
    
    # Json config path
    ETL_CONFIG_PATH = os.path.join(os.path.abspath("./config/download"),"download.json")
    
    # Parse config
    downloadConfig = parse_config(config_path=ETL_CONFIG_PATH)
    
    data = WebDownloadData(**downloadConfig, web_url=BOE_WEB_URL)
    print(data.model_dump())
    logger.info(f"Download information: {data.model_dump()}")
    
    downloader = Downloader(information=data)
    downloader.download()
    
    logger.info(f"{data.dw_files_paths=}")


if __name__ == "__main__":
    main()
