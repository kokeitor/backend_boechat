from fastapi import HTTPException, APIRouter
from fastapi.responses import FileResponse
from src.API.routes.utils import list_files_in_directory
import os
import logging

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("routes_get_data_logger")


DATA_DIR = os.path.join(os.getcwd(), 'src', 'API', 'assets', 'data')
print("DATA_DIR", DATA_DIR)
logger.info(f"UPLOAD_DIR : {DATA_DIR}")

getData = APIRouter()


@getData.get('/diagram_1')
async def getDiagram():
    files = list_files_in_directory(DATA_DIR)
    png_files = [file for file in files if file.endswith(
        '.png') and file.startswith('d_1')]

    if not png_files:
        raise HTTPException(status_code=404, detail="No PNG files found.")

    file_path = os.path.join(DATA_DIR, png_files[0])
    return FileResponse(file_path, media_type='image/png')


@getData.get('/diagram_2')
async def getDiagram():
    files = list_files_in_directory(DATA_DIR)
    png_files = [file for file in files if file.endswith(
        '.png') and file.startswith('d_2')]

    if not png_files:
        raise HTTPException(status_code=404, detail="No PNG files found.")

    file_path = os.path.join(DATA_DIR, png_files[0])
    return FileResponse(file_path, media_type='image/png')


@getData.get('/graph')
async def getDiagram():
    files = list_files_in_directory(DATA_DIR)
    png_files = [file for file in files if file.endswith(
        '.png') and file.startswith('graph_wf')]

    if not png_files:
        raise HTTPException(status_code=404, detail="No PNG files found.")

    file_path = os.path.join(DATA_DIR, png_files[0])
    return FileResponse(file_path, media_type='image/png')


@getData.get('/tfm')
async def getTfm():
    files = list_files_in_directory(DATA_DIR)
    pdf_files = [file for file in files if file.endswith('.pdf')]

    if not pdf_files:
        raise HTTPException(status_code=404, detail="No PDF files found.")

    # Send the first PDF file as an example (you could add logic to send all files)
    file_path = os.path.join(DATA_DIR, pdf_files[0])
    return FileResponse(file_path, media_type='application/pdf')
