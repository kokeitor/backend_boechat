from fastapi import FastAPI, HTTPException
from decouple import config
from src.API.routes.ia_response import iaResponse
from src.API.routes.get_data import getData
from src.API.Apis.openai_api import OpenAiModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import subprocess
import os
from dotenv import load_dotenv


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


def main():
    subprocess.run(["python", "src/main_etl.py"])


if __name__ == "__main__":
    main()
