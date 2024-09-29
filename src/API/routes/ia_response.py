from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import StreamingResponse
from src.API.models.models import ChatResponse
from typing import Optional, Annotated
from fastapi import UploadFile, File, Form
import os
from src.utils.utils import setup_logging, get_current_spanish_date_iso
import logging
from pydantic import BaseModel

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("routes_ia_response_logger")

UPLOAD_DIR = os.path.join(os.getcwd(), 'src', 'assets', 'uploads')
logger.info(f"UPLOAD_DIR : {UPLOAD_DIR}")

iaResponse = APIRouter()

# Example Pydantic model config for avoiding namespace conflicts


class GPT4AllEmbeddings(BaseModel):
    model_name: str

    class Config:
        protected_namespaces = ()


@iaResponse.get('/restartmemory')
async def restartMemory(request: Request):
    # getting from tha app state the client instance model:
    openAi = request.app.state.AI_MODEL
    openAi.messages = []
    return {"severResponse": "Memoria del chat borrada con éxito"}


@iaResponse.post("/boeresponse")
async def getBoeIaResponse(
    request: Request,
    userMessage: str
):
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

    def stream():
        stream = graph.compile_graph.astream(input=inputs, config=config_graph)
        for event in stream:
            if event["final_report"] and event["final_report"]["report"]:
                print(event)
                current_response = event["final_report"]["report"]
                yield current_response + "\n\n"
    return StreamingResponse(stream, media_type='text/event-stream')


@iaResponse.post("/iaresponse")
async def getIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()],
    uploadFiles: Optional[list[UploadFile]] = None
):
    print(f"uploadFiles : {uploadFiles}")
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
