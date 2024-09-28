from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
import logging 
import os
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.kinetica import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from exceptions.exceptions import VectorDatabaseError
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class RaptorVectorDB:
    def __init__(self, api_key :str , index_name : str, embd_model :str):
        load_dotenv()
        self.api_key = api_key
        self.index_name = index_name
        if embd_model:
            self.embedding_model = RaptorVectorDB.get_embd_model(embd_model=embd_model)
        if api_key and index_name:
            self.retriever ,self.vectorstore = self.get_retriever()
    
    def get_retriever(self) -> tuple[VectorStoreRetriever,VectorStore]:
        try:
            logger.info(f"Connecting to an existing index of PineCone DB cient -> {self.index_name}")
            pinecone_vectorstore = PineconeVectorStore(
                                                    embedding=self.embedding_model,
                                                    text_key='page_content',
                                                    distance_strategy=DistanceStrategy.COSINE,
                                                    pinecone_api_key=self.api_key,
                                                    index_name=self.index_name
                                                    )
        except Exception as e:
            logger.error(f"Error while connecting to PineCone DB from existing index : {self.index_name} -> {e}")
            raise VectorDatabaseError(message="Error while connecting to Chroma DB",exception=e)
            
        retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 3})
        # search_type="mmr",
        # search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5}
    
        return retriever , pinecone_vectorstore
    
    @staticmethod
    def get_embd_model(embd_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            model = HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL')))
        except Exception as e:
            logger.error(f"{e}")
            model = "Error"
            
        embd_available = {
                str(os.getenv('EMBEDDING_MODEL')): model
            }   
        return embd_available.get(embd_model, HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL'))))
    
    def store_docs(self, docs: list[Document]) -> None:
        if self.retriever and self.vectorstore:
            if isinstance(docs, list) and docs and isinstance(docs[0], Document):  # Check if the list is not empty
                docs_clean = self.clean_metadata(docs=docs)
                try:
                    self.vectorstore.add_documents(documents=docs_clean)
                except Exception as e:
                    logger.exception("Failed to store documents in vector store", exc_info=e)
            else:
                logger.warning("Invalid documents passed to store_docs method.")
        else:
            logger.warning("Retriever or vector store is not initialized.")
    
    def clean_metadata(self, docs: list[Document]):
        docs_clean = []
        for doc in docs:
            metadata = doc.metadata
            for key, value in metadata.items():
                if isinstance(value, float) and (value != value or value in [float('inf'), float('-inf')]):  # Checks for NaN or infinities
                    metadata[key] = "NaN" 
            doc.metadata = metadata
            docs_clean.append(doc)
        return docs_clean

    def get_context(self, query : str , filter_key : str, filter_value : str) -> list[Document]:
        if self.retriever and self.vectorstore:
            return self.retriever.invoke(input=query, filter={filter_key : filter_value})
            
                