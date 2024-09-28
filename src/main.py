from fastapi import FastAPI, HTTPException
from decouple import config
from API.routes.ia_response import iaResponse
from API.routes.get_data import getData
from API.Apis.openai_api import OpenAiModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging
from utils.utils import setup_logging
from ETL.pipeline import Pipeline
from RAPTOR.RAPTOR_BOE import RaptorDataset
from RAPTOR.raptor_vectordb import RaptorVectorDB
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Logging configuration
logger = logging.getLogger(__name__)


"""
print(f"OPENAI_API_KEY : {config('OPENAI_API_KEY')}")
FRONT_END_URL = config('FRONT_END_URL')
print(f"FRONT_END_URL : {FRONT_END_URL}")
origins = [
    FRONT_END_URL
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model in the state atribute of the app object
    app.state.AI_MODEL = OpenAiModel(api_ky=config('OPENAI_API_KEY'))
    yield
    # Clean up the model and release the resources
    app.state.AI_MODEL = None


app = FastAPI(
    title="Boe ChatBot BACKEND",
    lifespan=lifespan
)
app.include_router(iaResponse)
app.include_router(getData)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

"""
load_dotenv()

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_INDEX_NAME'] = os.getenv('PINECONE_INDEX_NAME')
os.environ['PINECONE_INDEX_NAMESPACE'] = os.getenv('PINECONE_INDEX_NAMESPACE')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY_RAPTOR'] = os.getenv(
    'LLAMA_CLOUD_API_KEY_RAPTOR')
os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')
os.environ['PINECONE_INDEX_NAME'] = os.getenv('PINECONE_INDEX_NAME')
os.environ['APP_MODE'] = os.getenv('APP_MODE')
os.environ['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL')
os.environ['EMBEDDING_MODEL_GPT4'] = os.getenv('EMBEDDING_MODEL_GPT4')
os.environ['LOCAL_LLM'] = os.getenv('LOCAL_LLM')
os.environ['HG_REPO_DATASET_ID'] = os.getenv('HG_REPO_DATASET_ID')
os.environ['HG_REPO_RAGAS_TESTSET_ID'] = os.getenv('HG_REPO_RAGAS_TESTSET_ID')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
os.environ['RAPTOR_CHUNKS_FILE_NAME'] = os.getenv('RAPTOR_CHUNKS_FILE_NAME')


def run_app():
    setup_logging(file_name="api.json")


def execute_etl():
    setup_logging(file_name="etl.json")
    ETL_CONFIG_PATH = os.path.join(os.path.abspath("./config/etl"), "etl.json")
    logger.info(F"ETL_CONFIG_PATH : {ETL_CONFIG_PATH}")
    pipeline = Pipeline(config_path=ETL_CONFIG_PATH)
    return pipeline.run()


def execute_raptor():
    setup_logging(file_name="raptor_boe.json")
    # Create Raptor data (make cluster summary, process and store in vector database)
    raptor_dataset = RaptorDataset(
        data_dir_path="./data/boedataset",
        file_name=f"{os.getenv('RAPTOR_CHUNKS_FILE_NAME')}.{os.getenv('RAPTOR_CHUNKS_FILE_EXTENSION')}",
        # from_date="2024-09-28",
        # to_date="2024-08-31",
        desire_columns=None  # Means all columns
    )
    raptor_dataset.initialize_data()

    # Store in vector database
    db = RaptorVectorDB(
        api_key=str(os.getenv('PINECONE_API_KEY')),
        index_name=str(os.getenv('PINECONE_INDEX_NAME')),
        embd_model=str(os.getenv('EMBEDDING_MODEL'))
    )
    # Restart index namesapce content
    db.delete_index_content()
    # Store new embedings
    db.store_docs(docs=raptor_dataset.documents)
    # Try database query
    query = """ La Oficina Consular honoraria en El Calafate, con categoría de Viceconsulado 
                Honorario, dependerá del Consulado General de España en Bahía Blanca."""
    filter_key = "label_str"
    filter_value = "Todos los Tipos de Decretos (Legislativos y no Legislativos)"
    context = db.get_context(
        query=query, filter_key=filter_key, filter_value=filter_value)
    print(context)
    try:
        logger.info(f"{query=} - {filter_key=} - {filter_value=}:\n{context=}")
        logger.info(
            f"k=1 : {filter_key=} - {filter_value=}:\n{filter_key=} - {context[0].metadata[filter_key]}")
        logger.info(
            f"k=2 : {filter_key=} - {filter_value=}:\n{filter_key=} - {context[1].metadata[filter_key]}")
        logger.info(
            f"k=3 : {filter_key=} - {filter_value=}:\n{filter_key=} - {context[2].metadata[filter_key]}")
    except Exception as e:
        logger.error(f"{e}")

    return db


def main():
    ##
    setup_logging(file_name="main.json")
    logger.info(f"{__name__} script execution path -> {os.getcwd()}")
    ##
    etl_result = execute_etl()
    logger.info(f"ETL Docs len : {len(etl_result)}")
    ##
    vectorDB = execute_raptor()
    logger.info(f"vectorDB  : {vectorDB}")
    ##
    run_app()


if __name__ == "__main__":
    main()
