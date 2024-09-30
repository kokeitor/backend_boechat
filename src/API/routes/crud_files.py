from fastapi import HTTPException, APIRouter, Request, UploadFile, File, Form
from typing import Optional, Annotated
import os
from src.utils.utils import setup_logging, get_current_spanish_date_iso
from src.API.routes.utils import delete_pdf_files
import logging
from src.ETL.pipeline import Pipeline


# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("routes_ia_response_logger")

crudfiles = APIRouter()


@crudfiles.post("/uploadfiles")
async def upload_files(request: Request, uploadFiles: Optional[list[UploadFile]] = None):

    logger.info(f"uploadFiles: {uploadFiles}")
    print(f"uploadFiles: {uploadFiles}")

    MAX_FILES = int(os.getenv('MAX_FILES'))

    fileNames = []
    warningMessage = ''

    if uploadFiles:
        upload_directory = os.path.join(os.getcwd(), 'data', 'boe', 'uploads')

        # Ensure the upload directory exists
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)

        for index, file in enumerate(uploadFiles):
            if index + 1 <= MAX_FILES:
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

                    print(
                        f"File '{fileName}' uploaded and saved at {file_path}")

                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Error reading or saving file {fileName}: {str(e)}")
            else:
                warningMessage = "Max files upload reached"
                break

        # Perform ETL and Raptor processes with the new files
        etl_config = request.app.state.etl_config
        etl = Pipeline(config_path=etl_config)
        raptor_dataset = request.app.state.raptor_dataset
        database = request.app.state.vector_db

        _ = etl.run()  # Run the ETL process
        # Initialize the dataset for Raptor, create cluster summary for each chunk
        raptor_dataset.initialize_data()
        database.delete_index_content()  # Clear the index in the vector database
        # Store the new documents in the database
        database.store_docs(docs=raptor_dataset.documents)

        return {"response": f"The files uploaded inside {upload_directory} are -> {' // '.join(fileNames)} // {warningMessage}"}
    else:
        return {"response": "No files were uploaded"}


@ crudfiles.delete("/deletefiles")
async def delete_files():
    upload_directory = os.path.join(os.getcwd(), 'data', 'boe', 'uploads')

    # Ensure the upload directory exists
    if not os.path.exists(upload_directory):
        raise HTTPException(
            status_code=404, detail=f"Upload directory '{upload_directory}' not found")

    try:
        # Delete PDF files and get the list of deleted files
        fileNames = delete_pdf_files(upload_directory)
        if not fileNames:
            return {"response": "No PDF files found to delete"}

        fileNamesFormat = ' // '.join(fileNames)
        return {"response": f"Files -> {fileNamesFormat} deleted successfully from {upload_directory}"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting files: {str(e)}")
