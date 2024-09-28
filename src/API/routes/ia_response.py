from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import StreamingResponse
from src.API.models.models import ChatResponse
from typing import Optional, Annotated
from fastapi import UploadFile, File, Form
import os

UPLOAD_DIR = os.path.join(os.getcwd(), 'src', 'assets', 'uploads')
print(f"UPLOAD_DIR : {UPLOAD_DIR}")

iaResponse = APIRouter()


@iaResponse.get('/restartmemory')
async def restartMemory(request: Request):
    # getting from tha app state the client instance model:
    openAi = request.app.state.AI_MODEL
    openAi.messages = []
    return {"severResponse": "Memoria del chat borrada con éxito"}


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
    print(f"userMessage : {userMessage}")
    print(f"iaResponse : {iaResponse}")
    print(f"Memory : {openAi.messages}")

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
    print(f"userMessage : {userMessage}")
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

    print(f"userMessage : {userMessage}")
    openAi = request.app.state.AI_MODEL
    return StreamingResponse(openAi.getStreamResponse(newUserMessage=userMessage), media_type='text/event-stream')
