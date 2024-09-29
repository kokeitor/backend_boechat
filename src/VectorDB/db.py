from langchain_community.vectorstores import Chroma, Qdrant
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores.kinetica import DistanceStrategy
import chromadb
import logging
import os
from dotenv import load_dotenv
from typing import Union
from src.GRAPH_RAG.models import get_openai_emb, get_hg_emb
from src.exceptions.exceptions import VectorDatabaseError
import VectorDB.test_db
import qdrant_client


INDEX_NAME = "INDEX_DEFAULT_VDB_NAME"

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


@VectorDB.test_db.try_retriever(query="¿que dia es hoy?")
@VectorDB.test_db.try_client_conexion
def get_chromadb_retriever(
    index_name: str = os.getenv("PINECONE_INDEX_NAME"),
    get_embedding_model: callable = get_hg_emb,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    collection_metadata: dict[str, str] = {"hnsw:space": "cosine"},
    search_kwargs: dict = {"k": 3},
    delete_index_name: Union[str, None] = None
) -> tuple[VectorStoreRetriever, VectorStore]:
    try:
        client = chromadb.HttpClient(host='localhost', port=8000)
    except ValueError:
        logger.error(
            "ValueError: Could not connect to a Chroma server. Are you sure it is running?")

    if delete_index_name:
        logger.error(
            f"Try to delete existing collection name index ->  '{delete_index_name}' of client CHROMA DB")
        try:
            client.delete_collection(name=delete_index_name)
            logger.info(
                f"CHROMA DB collection with name -> '{delete_index_name}' deleted")
        except Exception as e:
            logger.error(
                f"No CHROMA DB collection with name : {delete_index_name}")
            raise VectorDatabaseError(
                message="Error while connecting to Chroma DB", exception=e)

    try:
        chroma_vectorstore = Chroma(
            embedding_function=get_embedding_model(model=embedding_model),
            client=client,
            collection_name=index_name,
            collection_metadata=collection_metadata
        )
    except Exception as e:
        logger.error(f"Error while connecting to Chroma DB -> {e}")
        raise VectorDatabaseError(
            message="Error while connecting to Chroma DB", exception=e)

    retriever = chroma_vectorstore.as_retriever(search_kwargs=search_kwargs)

    return retriever, chroma_vectorstore


@VectorDB.test_db.try_retriever(query="¿hola?")
@VectorDB.test_db.try_client_conexion
def get_pinecone_retriever(
    get_embedding_model: callable = get_hg_emb,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    search_kwargs: dict = {"k": 3},
) -> tuple[VectorStoreRetriever, VectorStore]:

    try:
        logger.info(
            f"Connecting to an existing index of PineCone DB cient -> {os.getenv('PINECONE_INDEX_NAME')}")
        pinecone_vectorstore = PineconeVectorStore(
            embedding=get_embedding_model(model=embedding_model),
            text_key='page_content',
            distance_strategy=DistanceStrategy.COSINE,
            pinecone_api_key=os.getenv('PINECONE_API_KEY'),
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
    except Exception as e:
        logger.error(
            f"Error while connecting to PineCone DB from existing index : {os.getenv('PINECONE_INDEX_NAME')} -> {e}")
        raise VectorDatabaseError(
            message="Error while connecting to Chroma DB", exception=e)

    retriever = pinecone_vectorstore.as_retriever(search_kwargs=search_kwargs)

    return retriever, pinecone_vectorstore


@VectorDB.test_db.try_qdrant_conexion
def get_qdrant_retriever(
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME"),
    get_embedding_model: callable = get_hg_emb,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    search_kwargs: dict = {"k": 3},
    get_collections: bool = True
) -> tuple[VectorStoreRetriever, VectorStore]:

    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    vectors_config = qdrant_client.http.models.VectorParams(
        size=384,
        distance=qdrant_client.http.models.Distance.COSINE
    )

    # Checks if collection exists and if not create it
    if client.collection_exists(collection_name=collection_name):
        logger.info(
            f"Checking if Qdrant collection : {collection_name} exists -> {client.collection_exists(collection_name=collection_name)}")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )

    if get_collections:
        client.collection_exists(collection_name=collection_name)
        logger.info(f"Qdrant collections  -> {client.get_collections()}")

    # Integration with langchain -> vector store and retriever
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=get_embedding_model(model=embedding_model),
        distance_strategy="COSINE"
    )
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    return retriever, vector_store
