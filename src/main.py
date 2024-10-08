import os
import logging
import warnings
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from decouple import config
from src.API.routes.ia_response import iaResponse
from src.API.routes.utils import delete_pdf_files
from src.API.routes.crud_files import crudfiles
from src.API.routes.get_data import getData
from src.API.Apis.openai_api import OpenAiModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.utils.utils import setup_logging, get_current_spanish_date_iso
from src.ETL.pipeline import Pipeline
from src.RAPTOR.RAPTOR_BOE import RaptorDataset
from src.RAPTOR.raptor_vectordb import RaptorVectorDB
from src.GRAPH_RAG.graph import create_graph, compile_graph, save_graph
from src.GRAPH_RAG.config import ConfigGraph
from langgraph.graph.graph import CompiledGraph
from src.RAG_EVAL.base_models import RagasDataset
from langgraph.errors import InvalidUpdateError
from langchain_core.runnables.config import RunnableConfig


warnings.filterwarnings("ignore", category=FutureWarning)

# Logging configuration
logger = logging.getLogger(__name__)


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


def get_graph() -> tuple[RunnableConfig, CompiledGraph]:
    # Config graph file
    CONFIG_PATH = os.path.join(os.path.dirname(
        __file__), '..', 'config/graph', 'graph.json')

    logger.info(f"Getting Graph configuration from {CONFIG_PATH=}")
    config_graph = ConfigGraph(config_path=CONFIG_PATH)
    logger.info(f"config_graph : {config_graph}")

    logger.info("Creating graph and compiling workflow...")

    # create state graph
    config_graph.graph = create_graph(
        config=config_graph)

    # compile the state graph
    config_graph.compile_graph = compile_graph(config_graph.graph)

    # save graph diagram
    # save_graph(compile_graph=config_graph.compile_graph)

    logger.info("Graph and workflow created")

    return RunnableConfig(
        recursion_limit=config_graph.iteraciones,
        configurable={"thread_id": config_graph.thread_id}
    ), config_graph


def setup_etl():
    return os.path.join(
        os.path.abspath("./config/etl"), "etl.json")


def setup_raptor_dataset():
    return RaptorDataset(
        data_dir_path="./data/boedataset",
        file_name=f"{os.getenv('RAPTOR_CHUNKS_FILE_NAME')}.{os.getenv('RAPTOR_CHUNKS_FILE_EXTENSION')}",
        # from_date="2024-09-28",
        # to_date="2024-08-31",
        desire_columns=None  # Means all columns
    )


def setup_vector_db():
    # Store in vector database
    return RaptorVectorDB(
        api_key=str(os.getenv('PINECONE_API_KEY')),
        index_name=str(os.getenv('PINECONE_INDEX_NAME')),
        embd_model=str(os.getenv('EMBEDDING_MODEL'))
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(file_name="api.json")
    # Load the model in the state atribute of the app object
    app.state.open_ai_model = OpenAiModel(api_ky=os.getenv('OPENAI_API_KEY'))
    app.state.config_graph, app.state.graph = get_graph()
    app.state.etl_config = setup_etl()
    app.state.raptor_dataset = setup_raptor_dataset()
    app.state.vector_db = setup_vector_db()
    app.state.upload_directory = os.path.join(
        os.getcwd(), 'data', 'boe', 'uploads')
    yield
    # Clean up the model and release the resources
    app.state.config_graph = None
    app.state.graph = None
    app.state.etl_config = None
    app.state.raptor_dataset = None
    app.state.vector_db = None
    fileNames = delete_pdf_files(app.state.upload_directory)
    print(f"fileNames deleted : {fileNames}")
    app.state.upload_directory = None


FRONT_END_URL = os.getenv('FRONT_END_URL')
FRONT_END_PRO_URL = os.getenv('FRONT_END_PRO_URL')
logger.info(f"FRONT_END_URL : {FRONT_END_URL}")
logger.info(f"FRONT_END_PRO_URL : {FRONT_END_PRO_URL}")
origins = [
    FRONT_END_URL,
    FRONT_END_PRO_URL
]

app = FastAPI(
    title="Boe ChatBot BACKEND",
    lifespan=lifespan
)
app.include_router(iaResponse)
app.include_router(getData)
app.include_router(crudfiles)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
