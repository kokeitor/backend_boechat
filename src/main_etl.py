import os
import logging
from termcolor import colored
from dotenv import load_dotenv
from ETL.utils import get_current_spanish_date_iso, setup_logging
from ETL.pipeline import Pipeline


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:

    load_dotenv()

    # Logger set up
    setup_logging()

    ETL_CONFIG_PATH = os.path.join(os.path.abspath("./config/etl"), "etl.json")
    logger.info(F"ETL_CONFIG_PATH : {ETL_CONFIG_PATH}")
    pipeline = Pipeline(config_path=ETL_CONFIG_PATH)
    result = pipeline.run()


if __name__ == '__main__':
    main()
