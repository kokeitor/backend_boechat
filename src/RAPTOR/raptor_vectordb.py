from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
import logging
import os
from typing import Optional
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.kinetica import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from src.exceptions.exceptions import VectorDatabaseError
from dotenv import load_dotenv
import time
from pinecone import Pinecone, ServerlessSpec


logger = logging.getLogger(__name__)


class RaptorVectorDB:
    def __init__(self, api_key: str, index_name: str, embd_model: str):
        load_dotenv()
        self.api_key = api_key
        self.index_name = index_name
        self.vectorstore = None
        self.retriever = None
        if embd_model:
            self.embedding_model = RaptorVectorDB.get_embd_model(
                embd_model=embd_model)
        if api_key and index_name:
            self.retriever, self.vectorstore = self.get_retriever()

    def get_retriever(self) -> tuple[VectorStoreRetriever, VectorStore]:
        try:
            logger.info(
                f"Connecting to an existing index of PineCone DB cient -> {self.index_name}")
            self.pc = Pinecone(api_key=self.api_key)
            existing_indexes = [index_info["name"]
                                for index_info in self.pc.list_indexes()]
            logger.info(f"existing_indexes : {existing_indexes}")

            if self.index_name not in existing_indexes:
                logger.warning(f"Creting Index : {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    logger.info(self.pc.describe_index(self.index_name))
                    time.sleep(1)

            self.index = self.pc.Index(self.index_name)
            pinecone_vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_model,
                text_key='page_content',
                distance_strategy=DistanceStrategy.COSINE
            )
            retriever = pinecone_vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            # search_type="mmr",
            # search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5}
            return retriever, pinecone_vectorstore
        except Exception as e:
            logger.error(
                f"Error while connecting to PineCone DB from existing index : {self.index_name} -> {e}")
            raise VectorDatabaseError(
                message="Error while connecting to PineCone DB from existing index", exception=e)

    @staticmethod
    def get_embd_model(embd_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            model = HuggingFaceEmbeddings(
                model_name=str(os.getenv('EMBEDDING_MODEL')))
        except Exception as e:
            logger.error(f"{e}")
            model = "Error"

        embd_available = {
            str(os.getenv('EMBEDDING_MODEL')): model
        }
        return embd_available.get(embd_model, HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL'))))

    def store_docs(self, docs: list[Document]) -> None:
        if self.retriever and self.vectorstore:
            # Check if the list is not empty
            if isinstance(docs, list) and docs and isinstance(docs[0], Document):
                docs_clean = self.clean_metadata(docs=docs)
                try:
                    self.vectorstore.add_documents(
                        documents=docs_clean,
                        namespace=str(
                            os.getenv('PINECONE_INDEX_NAMESPACE'))
                    )
                    while self.index.describe_index_stats()["total_vector_count"] == 0:
                        logger.info(
                            f"VectorDB Index -> {self.index_name} Stats : {self.index.describe_index_stats()}")
                        time.sleep(1)
                    logger.info("VectorDB Index Ready")
                except Exception as e:
                    logger.exception(
                        "Failed to store documents in vector store", exc_info=e)
            else:
                logger.warning(
                    "Invalid documents passed to store_docs method.")
        else:
            logger.warning("Retriever or vector store is not initialized.")

    def clean_metadata(self, docs: list[Document]):
        docs_clean = []
        for doc in docs:
            metadata = doc.metadata
            for key, value in metadata.items():
                # Checks for NaN or infinities
                if isinstance(value, float) and (value != value or value in [float('inf'), float('-inf')]):
                    metadata[key] = "NaN"
            doc.metadata = metadata
            docs_clean.append(doc)
        return docs_clean

    def get_context(self, query: str, filter_key: Optional[str] = None, filter_value: Optional[str] = None) -> list[Document]:
        if self.retriever and self.vectorstore:
            retrieved_docs = self.vectorstore.similarity_search(
                query=query,
                k=3,
                namespace=os.getenv('PINECONE_INDEX_NAMESPACE')
            )
            if filter_key and filter_value:
                retrieved_docs = self.vectorstore.similarity_search(
                    query=query,
                    k=3,
                    filter={filter_key: filter_value},
                    namespace=os.getenv('PINECONE_INDEX_NAMESPACE')
                )
            return retrieved_docs

    def delete_index_content(self) -> None:
        try:
            self.index.delete(delete_all=True, namespace=str(
                os.getenv('PINECONE_INDEX_NAMESPACE')))
            logger.info(
                f"All content in index '{self.index_name}' has been deleted.")
            # while not self.pc.describe_index(self.index_name).status["ready"]:
            while self.index.describe_index_stats()["total_vector_count"] > 0:
                logger.info(
                    f"self.index.describe_index_stats()['total_vector_count] == {self.index.describe_index_stats()['total_vector_count']}"
                )
                time.sleep(1)
        except Exception as e:
            logger.error(
                f"Error while deleting content from index '{self.index_name}': {e}")
