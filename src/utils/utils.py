import deepl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
import os
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['DEEPL_KEY'] = os.getenv('DEEPL_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
LOCAL_LLM = 'llama3'



### TRANSLATION TOOL
translator = deepl.Translator(os.getenv('DEEPL_KEY'))
def translate(text :str, target_lang : str = "ES" , verbose : int = 0, mode : str = "LOCAL_LLM") -> str:
    
    _target_lang = {
                        "EN-GB":"British english",
                        "EN-US":"United States english",
                        "ES":"Spanish"
        
                    }
    
    if mode == "DEEPL":
        result = translator.translate_text(text = text,source_lang = 'ES', target_lang= target_lang)
        if verbose == 1:
            print(f"texto :\n{text}")
            print(f"Traduccion:\n{result.text}")  
        return result.text
    else:
        if mode == "GPT":
            llm_for_trl = ChatOpenAI(model_name='gpt-4', temperature = 0 )
        if mode == "LOCAL_LLM":
            llm_for_trl = ChatOllama(model=LOCAL_LLM, temperature=0)
        
        transalation_prompt = PromptTemplate(
                        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for translation into spanish language. \n
                        Use the target specify language to translate the text to that language. \n
                        The translation must be the most reliable as posible to the spanish text keeping the technicalities and without translating proper names or names of cities or villages. \n
                        Return the a JSON with a single key 'translation' and no premable or explaination.<|eot_id|><|start_header_id|>user<|end_header_id|>
                        Text: {text} 
                        Target language: {target_language} 
                        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                        input_variables=["text", "target_language"]
                                        )
        trl_chain = transalation_prompt | llm_for_trl | JsonOutputParser()
        
        return trl_chain.invoke({"text": text , "target_language": _target_lang.get(target_lang,"Spanish")})["translation"]
    
    

### MERGE DOCUMENTS (page_content) INTO ONE str
def format_docs(docs : List[Document]) -> str:
    """Trasnform List[Documents] into str using doc atribute page_content

    Args:
        docs (_type_): _description_

    Returns:
        _type_: _description_
    """
    return "\n\n".join(doc.page_content for doc in docs)


### Splitter , Parsers, tokenizers ... tools for processing text
def pdf_parser() -> List[Document]:
    """_summary_

    Returns:
        List[Document]: _description_
    """
    parser = LlamaParse(
        api_key = os.getenv('LLAMA_CLOUD_API_KEY'),
        result_type="markdown",  # "markdown" and "text" are available
        verbose=True
    )
    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader(
                                        "./documentos/boe/",
                                        file_extractor = file_extractor,
                                        recursive=True, # recursively search in subdirectories
                                        required_exts = [".pdf"]
                                        )

    all_docs = reader.load_data() # returns List[llama doc objt] : https://docs.llamaindex.ai/en/v0.10.17/api/llama_index.core.schema.Document.html
    print("Num docs extracted : ", len(all_docs))

    # Transform into langchain docs
    lang_chain_docs = []
    for d in all_docs:
        print(d.get_type())
        lang_chain_docs.append(d.to_langchain_format()) 
        
    return lang_chain_docs



def doc_to_vectordb(db : list, docs : list, translation :bool = False):
    for db_i in db:
        if translation:
            for doc_i in docs:
                doc_i.page_content = translate(
                                                text = doc_i.page_content, 
                                                verbose = 0,
                                                target_lang = "EN-GB",
                                                mode = 'LOCAL_LLM'
                                                )
                print(docs)
        db_i.add_documents(documents = docs)
"""
usage:
doc_to_vectordb(
            db = [vectorstore],
            docs = text_splitter.split_documents(documents = loader.load()), # .load() -> List[Document] // .split_documents() -> List[Document]
            translation = False
            ) 
"""


def execution_time(func : callable = None):
    """
    Funcion decoradora que calcula el tiempo de ejecucion de la funcion que esta decorando
    Parameters
    ----------
        - func : (callable) funcion a decorar
    return
    ------
        - wrapper : (callable) funcion "envoltorio2 que agrega la funcionalidad del calculo de tiempo de ejecucion
    """
    def wrapper(*args, **kwargs):
        start_execution = time.time()
        return_func = func(*args,**kwargs)
        end_execution = time.time()
        print(f'Exexcution time of {func.__name__} {end_execution - start_execution}')
        return return_func
    return wrapper


########## sucio ###########

import os
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar
from dataclasses import dataclass, field
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


### BOE PDF DOWNLOAD TOOL
def boe_pdfs_downloader(vectorstore, text_splitter):
    # URL base y configuración de directorios
    boe_url = 'https://boe.es'
    destino_local_raiz = './documentos'  # Ruta relativa para mayor compatibilidad
    destino_local = os.path.join(destino_local_raiz, 'boe', 'dias')
    print(destino_local)
    boe_api_sumario = f'{boe_url}/diario_boe/xml.php?id=BOE-S-'

    # Fechas de inicio y fin para la descarga de documentos
    desde = '20240415'
    hasta = '20240415'

    fecha = datetime.strptime(desde, '%Y%m%d')
    fecha_fin = datetime.strptime(hasta, '%Y%m%d')
    
    while fecha <= fecha_fin:
        fecha_ymd = fecha.strftime('%Y%m%d')
        print(f'Fecha descarga BOE : {fecha}')
        carpeta_fecha = os.path.join(destino_local, fecha.strftime('%Y'), fecha.strftime('%m'), fecha.strftime('%d'))
        
        fichero_sumario_xml = os.path.join(carpeta_fecha, 'index.xml')
        print("fichero_sumario_xml : ", fichero_sumario_xml)
        
        # Eliminar el sumario XML si existe
        if os.path.exists(fichero_sumario_xml):
            os.remove(fichero_sumario_xml)
        
        print(f'Solicitando {boe_api_sumario}{fecha_ymd} --> {fichero_sumario_xml}')
        traer_xml(url = boe_api_sumario + fecha_ymd, destino = fichero_sumario_xml)
        
        urls_pdf = extraer_urls_pdf(fichero_sumario_xml)
        print(f'urls PDFs totales para la fecha {fecha} : {len(urls_pdf)}')
        for num_pdfs, url_pdf in enumerate(urls_pdf):
            if num_pdfs < 5:
                loader = PyPDFLoader(descargar_pdf(url_base = boe_url, url_pdf = url_pdf, ruta_destino = carpeta_fecha))
                doc_to_vectordb(
                                db = [vectorstore],
                                docs = text_splitter.split_documents(documents = loader.load()), # .load() -> List[Document] // .split_documents() -> List[Document]
                                translation = False
                                ) 
            else:
                print(f'No se descargarán mas PDFs para la fecha : {fecha} ')
                break
        
        fecha += timedelta(days=1)


def traer_xml(url, destino):
    
    print("destino",destino)
    # Asegurarse de que 'destino' incluya un nombre de archivo.
    if not os.path.exists(os.path.dirname(destino)):
        os.makedirs(os.path.dirname(destino), exist_ok=True)
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(destino, 'wb') as file:
            file.write(response.content)
    else:
        print(f'Error al descargar el documento: {response.status_code} URL: {url}')


def extraer_urls_pdf(archivo_xml):
    tree = ET.parse(archivo_xml)
    root = tree.getroot()

    # Extraemos todas las URLs de los PDFs
    #####
    ## Podria añadir extraccion de metadatos del archivo xml para cada dia (un xml por dia y varios pdfs)
    ## despues asociar esos metadatos a cada embedding de cada pdf de cada dia en el proceso de vewctorDB
    #####
    urls_pdf = []
    for urlPdf in root.findall('.//urlPdf'):
        url = urlPdf.text  # Obtén el texto del elemento, que es la URL
        urls_pdf.append(url)
    return urls_pdf

def descargar_pdf(url_base, url_pdf, ruta_destino):
    url_completa = url_base + url_pdf
    respuesta = requests.get(url_completa)
    if respuesta.status_code == 200:
        nombre_pdf = url_pdf.split('/')[-1]  # Extraemos el nombre del archivo desde la URL
        ruta_completa_pdf = os.path.join(ruta_destino, nombre_pdf)
        
        # Asegúrate de que el directorio de destino existe
        os.makedirs(os.path.dirname(ruta_completa_pdf), exist_ok=True)
        
        with open(ruta_completa_pdf, 'wb') as archivo:
            archivo.write(respuesta.content)
        print(f'Archivo descargado con éxito: {ruta_completa_pdf}')
        return ruta_completa_pdf
    else:
        print(f'Error al descargar {url_completa}: {respuesta.status_code}')

