from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import StreamingResponse
from src.API.models.models import ChatResponse
from src.ETL.pipeline import Pipeline
from typing import Optional, Annotated
from fastapi import UploadFile, File, Form
import os
from src.utils.utils import setup_logging, get_current_spanish_date_iso
from src.RAPTOR.RAPTOR_BOE import RaptorDataset
import logging
from pydantic import BaseModel

UPLOAD_DIR = os.path.join(os.getcwd(), 'src', 'assets', 'uploads')

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("routes_ia_response_logger")

iaResponse = APIRouter()


@iaResponse.get('/restartmemory/')
async def restartMemory(request: Request):
    # getting from tha app state the client instance model:
    openAi = request.app.state.AI_MODEL
    openAi.messages = []
    return {"severResponse": "Memoria del chat borrada con éxito"}


@iaResponse.post("/streamboeresponse/")
async def getBoeStreamIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()],
):
    print(f"userMessage: {userMessage}")

    # Graph and configuration
    graph = request.app.state.graph
    config_graph = request.app.state.config_graph
    inputs = {
        "question": [userMessage],
        "date": get_current_spanish_date_iso(),
        "query_label": None,
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

    # Return the stream as a StreamingResponse with the appropriate media type for SSE
    return StreamingResponse(event_stream(final_state), media_type='text/event-stream')


@iaResponse.post("/boeresponse/")
async def getBoeStreamIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()]
):
    print(f"userMessage: {userMessage}")

    # Graph and configuration
    graph = request.app.state.graph
    config_graph = request.app.state.config_graph
    inputs = {
        "question": [userMessage],
        "date": get_current_spanish_date_iso(),
        "query_label": None,
        "generation": None,
        "documents": None,
        "fact_based_answer": None,
        "useful_answer": None
    }

    # Invoke the graph with the input and config
    final_state = graph.compile_graph.invoke(input=inputs, config=config_graph)
    return {"status": "success", "final_state": final_state}


@iaResponse.post("/iaresponse/")
async def getIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()]
):
    logger.info(f"uploadFiles : {userMessage}")
    print(f"userMessage : {userMessage}")

    # getting from tha app state the client instance model:
    openAi = request.app.state.open_ai_model
    # get ia response
    iaResponse = openAi.getResponse(newUserMessage=userMessage)
    logger.info(f"userMessage : {userMessage}")
    logger.info(f"iaResponse : {iaResponse}")
    print(f"iaResponse : {iaResponse}")
    logger.info(f"Memory : {openAi.messages}")
    chat = ChatResponse(
        userMessage=userMessage,
        iaResponse=iaResponse,
        files=[]
    )
    return chat


@iaResponse.post('/iaresponsestream/')
async def stream(
    request: Request,
    userMessage: Annotated[str, Form()]
):
    logger.info(f"userMessage : {userMessage}")
    print(f"userMessage : {userMessage}")
    openAi = request.app.state.open_ai_model
    return StreamingResponse(openAi.getStreamResponse(newUserMessage=userMessage), media_type='text/event-stream')
