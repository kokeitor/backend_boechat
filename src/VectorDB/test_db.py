import logging 
import os
import requests


logger = logging.getLogger(__name__)



def try_client_conexion(func: callable):
    """
    Wrapper -> try connection to client vector db

    Args:
        tries (int, optional): Number of attempts to try connection. Defaults to 2.
    """
    def wrapper(*args, **kwargs):
        retriever, vectorstore = func(*args, **kwargs)
        logger.info(f"Trying client db conexion : {func.__name__}")

        # First test
        try:
            data = vectorstore.get(limit= 5, include=['documents', 'metadatas'])
            for i,id in enumerate(data['ids']):
                logger.info(f"For document {id=} in Client DB number {i} : -> {data['documents'][i]=} --  {data['metadatas'][i]=}")
        except Exception as e:
            logger.error(f"Client DB -> First test error using {func.__name__} -> {e}")

        # Second test a
        try:
            # Note: Instantiating Chroma class creates an object equivalent to chroma_client from chromadb library but using langchain library
            logger.info(f"Db Collection keys {vectorstore.get().keys()}")
            logger.info(f"Db Collection Number of ids (one for each document) : {len(vectorstore.get()['ids'])}")
            logger.info(f"Db Collection documents : {vectorstore.get()['documents']}")
        except Exception as e:
            logger.error(f"Client DB -> Second test a [Collection] error using {func.__name__} -> {e}")
        
        # Second test b
        try:
            logger.info(f"Db Collection embeddings (1st comp of first embedding) :  {vectorstore.get(include=['embeddings'])['embeddings'][0][0]}")
            logger.info(f"Db Collection Embeddings Dimension: {len(vectorstore.get(include=['embeddings'])['embeddings'][0])}")
        except Exception as e:
            logger.error(f"Client DB -> Second test b [embeddings] error using {func.__name__} -> {e}")

        return retriever, vectorstore

    return wrapper


def try_retriever(query :str ="La duración total de las enseñanzas en ciclos de grado medio"):
    """
    Try retriever

    Args:
        query (str, optional): The query to test the retriever. Defaults to "La duración total de las enseñanzas en ciclos de grado medio".
    """
    def decorator(func: callable):
        def wrapper(*args, **kwargs):
            retriever, vectorestore = func(*args, **kwargs)
            logger.info(f"Trying retriever associated with DB client")

            try:
                response = retriever.invoke(query)
                logger.info(f"Number of embeddings retrieved : {len(response)}")
                if len(response) > 0:
                    logger.info(f"Best similarity retriever search : {response[0].page_content}")
            except Exception as e:
                logger.error(f"Retriever error using {func.__name__} -> {e}")

            return retriever, vectorestore

        return wrapper
    
    return decorator


def try_qdrant_conexion(func : callable):
    def wrapper(*args,**kwargs):
        
        url_host = os.getenv('QDRANT_HOST')
        headers = {
                    "Api-key": os.getenv('QDRANT_API_KEY')
                }
        response = requests.get(url_host, headers=headers)
        
        logger.info(f"Qdrant http get response status code : {response.status_code}")
        logger.info(f"Qdrant http get response body : {response.text}")
        
        if response.status_code != '200':
            # do something 
            pass
        
        url_host_collections = os.getenv('QDRANT_COLLECTIONS')
        headers = {
                    "Api-key": os.getenv('QDRANT_API_KEY')
                }
        response_collection = requests.get(url_host_collections, headers=headers)
        
        logger.info(f"Qdrant get collections status code : {response_collection.status_code}")
        logger.info(f"Qdrant  get collections  : {response_collection.text}")
        
        return func(*args,**kwargs)
        
    return wrapper

