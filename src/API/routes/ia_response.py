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
    print(f"uploadFiles: {uploadFiles}")
    fileNames = None
    upload_directory = os.path.join(os.getcwd(), 'data', 'boe', 'uploads')

    # Ensure the upload directory exists
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

    if uploadFiles:
        fileNames = []
        for file in uploadFiles:
            fileName = file.filename
            fileNames.append(fileName)

            # Read the file content
            try:
                fileContent = await file.read()
                file_path = os.path.join(upload_directory, fileName)

                # Save the file to the specified directory
                with open(file_path, "wb") as f:
                    f.write(fileContent)

                # Check if the file has been saved successfully
                if not os.path.exists(file_path):
                    raise HTTPException(
                        status_code=500, detail=f"Failed to save the file: {fileName}")

                print(f"File '{fileName}' uploaded and saved at {file_path}")

            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error reading or saving file {fileName}: {str(e)}")

    # Perform ETL and Raptor processes with the new files
    etl = Pipeline(config_path=request.app.state.etl_config_pipeline)
    raptor_dataset = request.app.state.raptor_dataset
    database = request.app.state.vector_db

    etl.run()  # Run the ETL process
    raptor_dataset.initialize_data()  # Initialize the dataset for Raptor
    database.delete_index_content()  # Clear the index in the vector database
    # Store the new documents in the database
    database.store_docs(docs=raptor_dataset.documents)

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


@ iaResponse.post("/boeresponse")
async def getBoeStreamIaResponse(
    request: Request,
    userMessage: Annotated[str, Form()],
    uploadFiles: Optional[list[UploadFile]] = None
):
    print(f"uploadFiles: {uploadFiles}")
    fileNames = None
    upload_directory = os.path.join(os.getcwd(), 'data', 'boe', 'uploads')

    # Ensure the upload directory exists
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)

    if uploadFiles:
        fileNames = []
        for file in uploadFiles:
            fileName = file.filename
            fileNames.append(fileName)

            # Read the file content
            try:
                fileContent = await file.read()
                file_path = os.path.join(upload_directory, fileName)

                # Save the file to the specified directory
                with open(file_path, "wb") as f:
                    f.write(fileContent)

                # Check if the file has been saved successfully
                if not os.path.exists(file_path):
                    raise HTTPException(
                        status_code=500, detail=f"Failed to save the file: {fileName}")

                print(f"File '{fileName}' uploaded and saved at {file_path}")

            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error reading or saving file {fileName}: {str(e)}")

    # Perform ETL and Raptor processes with the new files
    etl = Pipeline(config_path=request.app.state.etl_config_pipeline)
    raptor_dataset = request.app.state.raptor_dataset
    database = request.app.state.vector_db

    etl.run()  # Run the ETL process
    raptor_dataset.initialize_data()  # Initialize the dataset for Raptor
    database.delete_index_content()  # Clear the index in the vector database
    # Store the new documents in the database
    database.store_docs(docs=raptor_dataset.documents)

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
