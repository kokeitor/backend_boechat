from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import StreamingResponse
from src.API.models.models import ChatResponse
from src.ETL.pipeline import Pipeline
from typing import Optional, Annotated
from fastapi import UploadFile, File, Form
import os
from src.utils.utils import setup_logging, get_current_spanish_date_iso
import logging
from pydantic import BaseModel

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("routes_ia_response_logger")

UPLOAD_DIR = os.path.join(os.getcwd(), 'data', 'boe', 'uploads')
logger.info(f"UPLOAD_DIR : {UPLOAD_DIR}")

iaResponse = APIRouter()


@iaResponse.get('/restartmemory')
async def restartMemory(request: Request):
    # getting from tha app state the client instance model:
    openAi = request.app.state.AI_MODEL
    openAi.messages = []
    return {"severResponse": "Memoria del chat borrada con éxito"}


@iaResponse.post("/streamboeresponse")
async def getBoeStreamIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()],
    uploadFiles: Optional[list[UploadFile]] = None
):
    logger.info(f"uploadFiles : {uploadFiles}")
    fileNames = None
    if uploadFiles:
        fileNames = []
        for file in uploadFiles:
            fileName = file.filename
            fileNames.append(fileName)
            fileContent = await file.read()
            with open(os.path.join(UPLOAD_DIR, fileName), "wb") as f:
                f.write(fileContent)
    # perform etl and raptor in the new files
    etl = Pipeline(config_path=request.app.state.etl_config_pipeline)
    raptor_dataset = request.app.state.raptor_dataset
    database = request.app.state.vector_db
    etl.run()
    raptor_dataset.initialize_data()
    database.delete_index_content()
    database.store_docs(docs=raptor_dataset.documents)

    # graph and config
    graph = request.app.state.graph
    config_graph = request.app.state.config_graph
    inputs = {
        "question": [userMessage],
        "date": get_current_spanish_date_iso(),
        "query_label":  None,
        "generation": None,
        "documents": None,
        "fact_based_answer": None,
        "useful_answer": None
    }
    final_state = graph.compile_graph.invoke(input=inputs, config=config_graph)

    def event_stream(final_state):
        for chunk in final_state["generation"]:
            current_response = chunk
            # Format the response for server-sent events (SSE)
            yield f"data: {current_response}\n\n"
        """
                # Use asynchronous streaming from the graph
                async for event in graph.compile_graph.astream(input=inputs, config=config_graph):
                    if event.get("generator") and event["generator"].get("generation"):
                current_response = event["generator"]["generation"]
                # Format the response for server-sent events (SSE)
                yield f"data: {current_response}\n\n"
        """
    # Return the stream as a StreamingResponse with the appropriate media type for SSE
    return StreamingResponse(event_stream(final_state), media_type='text/event-stream')


@iaResponse.post("/boeresponse")
async def getBoeIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()],
    uploadFiles: Optional[list[UploadFile]] = None
):
    logger.info(f"uploadFiles : {uploadFiles}")
    fileNames = None
    if uploadFiles:
        fileNames = []
        for file in uploadFiles:
            fileName = file.filename
            fileNames.append(fileName)
            fileContent = await file.read()
            with open(os.path.join(UPLOAD_DIR, fileName), "wb") as f:
                f.write(fileContent)

   # perform etl and raptor in the new files
    etl = Pipeline(config_path=request.app.state.etl_config_pipeline)
    raptor_dataset = request.app.state.raptor_dataset
    database = request.app.state.vector_db
    etl.run()
    raptor_dataset.initialize_data()
    database.delete_index_content()
    database.store_docs(docs=raptor_dataset.documents)

    # graph and config
    graph = request.app.state.graph
    config_graph = request.app.state.config_graph
    inputs = {
        "question": [userMessage],
        "date": get_current_spanish_date_iso(),
        "query_label":  None,
        "generation": None,
        "documents": None,
        "fact_based_answer": None,
        "useful_answer": None
    }
    final_state = graph.compile_graph.invoke(input=inputs, config=config_graph)
    return {"response": final_state["finaal_reprt"]}


@iaResponse.post("/iaresponse")
async def getIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()],
    uploadFiles: Optional[list[UploadFile]] = None
):
    logger.info(f"uploadFiles : {uploadFiles}")
    fileNames = None
    if uploadFiles:
        fileNames = []
        for file in uploadFiles:
            fileName = file.filename
            fileNames.append(fileName)
            fileContent = await file.read()
            with open(os.path.join(UPLOAD_DIR, fileName), "wb") as f:
                f.write(fileContent)

    # getting from tha app state the client instance model:
    openAi = request.app.state.AI_MODEL
    # get ia response
    iaResponse = openAi.getResponse(newUserMessage=userMessage)
    logger.info(f"userMessage : {userMessage}")
    logger.info(f"iaResponse : {iaResponse}")
    logger.info(f"Memory : {openAi.messages}")

    if fileNames:
        openAi.files.extend(fileNames)
        chat = ChatResponse(
            userMessage=userMessage,
            iaResponse=iaResponse,
            files=fileNames
        )
    else:
        chat = ChatResponse(
            userMessage=userMessage,
            iaResponse=iaResponse,
            files=[]
        )
    return chat


@iaResponse.post('/iaresponsestream')
async def stream(
    request: Request,
    userMessage: Annotated[str, Form()],
    uploadFiles: Optional[list[UploadFile]] = None
):
    logger.info(f"userMessage : {userMessage}")
    logger.info(f"uploadFiles : {uploadFiles}")
    fileNames = None
    if uploadFiles:
        fileNames = []
        for file in uploadFiles:
            fileName = file.filename
            fileNames.append(fileName)
            fileContent = await file.read()
            with open(os.path.join(UPLOAD_DIR, fileName), "wb") as f:
                f.write(fileContent)

    logger.info(f"userMessage : {userMessage}")
    openAi = request.app.state.AI_MODEL
    return StreamingResponse(openAi.getStreamResponse(newUserMessage=userMessage), media_type='text/event-stream')
