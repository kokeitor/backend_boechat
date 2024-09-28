import os
import logging
from termcolor import colored
from dotenv import load_dotenv
from ETL.utils import get_current_spanish_date_iso, setup_logging
from ETL.pipeline import Pipeline


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:

    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')
    os.environ['PINECONE_INDEX_NAME'] = os.getenv('PINECONE_INDEX_NAME')
    os.environ['CHROMA_COLLECTION_NAME'] = os.getenv('CHROMA_COLLECTION_NAME')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['APP_MODE'] = os.getenv('APP_MODE')
    os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
    os.environ['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL')
    os.environ['EMBEDDING_MODEL_GPT4'] = os.getenv('EMBEDDING_MODEL_GPT4')
    os.environ['LOCAL_LLM'] = os.getenv('LOCAL_LLM')
    os.environ['LOCAL_LLM'] = os.getenv('GOOGLE_BBDD_FILE_NAME_CREDENTIALS')
    os.environ['GOOGLE_DOCUMENT_NAME'] = os.getenv('GOOGLE_DOCUMENT_NAME')
    os.environ['GOOGLE_SHEET_NAME'] = os.getenv('GOOGLE_SHEET_NAME')
    os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

    # Logger set up
    setup_logging()

    ETL_CONFIG_PATH = os.path.join(os.path.abspath("./config/etl"), "etl.json")
    pipeline = Pipeline(config_path=ETL_CONFIG_PATH)
    result = pipeline.run()


if __name__ == '__main__':
    main()
