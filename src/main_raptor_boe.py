import os
import logging
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import setup_logging
from RAPTOR.RAPTOR_BOE import RaptorDataset
from RAPTOR.raptor_vectordb import RaptorVectorDB
from dotenv import load_dotenv


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['HG_API_KEY'] = str(os.getenv('HG_API_KEY'))
    os.environ['PINECONE_API_KEY'] = str(os.getenv('PINECONE_API_KEY'))
    os.environ['PINECONE_INDEX_NAME'] = str(os.getenv('PINECONE_INDEX_NAME'))
    os.environ['EMBEDDING_MODEL'] = str(os.getenv('EMBEDDING_MODEL'))

    # set up the root logger configuration
    setup_logging(script="raptor_boe")
    
    # Create Raptor data (make cluster summary, process and store in vector database)
    raptor_dataset = RaptorDataset(
        data_dir_path="./data/boedataset", 
        from_date="2024-08-28", 
        to_date="2024-08-31",
        desire_columns=None # Means all columns
    )
    raptor_dataset.initialize_data()
    
    # Store in vector database
    db = RaptorVectorDB(
                    api_key=str(os.getenv('PINECONE_API_KEY')),
                    index_name=str(os.getenv('PINECONE_INDEX_NAME')),
                    embd_model=str(os.getenv('EMBEDDING_MODEL'))
                    )
    db.store_docs(docs=raptor_dataset.documents)
    
    # Try database query
    query = "rendimiento neto del ovino y caprino de carne"
    filter_key="label_str"
    filter_value="Planes de Estudio y Normativas Educativas"
    context = db.get_context(query=query, filter_key=filter_key,filter_value=filter_value)
    try:
        logger.info(f"{query=} - {filter_key=} - {filter_value=}:\n{context=}")
        logger.info(f"k=1 : {filter_key=} - {filter_value=}:\n{filter_key=} - {context[0].metadata[filter_key]}")
        logger.info(f"k=2 : {filter_key=} - {filter_value=}:\n{filter_key=} - {context[1].metadata[filter_key]}")
        logger.info(f"k=3 : {filter_key=} - {filter_value=}:\n{filter_key=} - {context[2].metadata[filter_key]}")
    except Exception as e:
        logger.error(f"{e}")
    
if __name__ == "__main__":
    main()