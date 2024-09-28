import os
import requests
from requests import Response
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import logging
import logging.config
import logging.handlers
from ETL.utils import get_current_spanish_date_iso, setup_logging

# Logging configuration
logger = logging.getLogger("Download_web_files_module")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


class WebDownloadData(BaseModel):
    """Web Download data object"""
    web_url: str
    local_dir: str
    fecha_desde: str
    fecha_hasta: str
    batch: int
    dw_files_paths: list[str] = Field(default_factory=list)


### BOE PDF DOWNLOAD TOOL
@dataclass
class Downloader:
    information: WebDownloadData

    def download(self):
        destino_local = os.path.join(self.information.local_dir, 'boe', 'dias')
        logger.info(f"Destino local : {destino_local}")

        boe_api_sumario = f'{self.information.web_url}/diario_boe/xml.php?id=BOE-S-'

        # Fechas de inicio y fin para la descarga de documentos
        fecha_ini = datetime.strptime(self.information.fecha_desde, '%Y-%m-%d')  # format necesario : '2024-04-15'
        fecha_fin = datetime.strptime(self.information.fecha_hasta, '%Y-%m-%d')
        logger.info(f"Fecha inicio descarga : {fecha_ini}")
        logger.info(f"Fecha fin descarga: {fecha_fin}")

        while fecha_ini <= fecha_fin:
            fecha_ymd = fecha_ini.strftime('%Y%m%d')  # formato cambiado a '20240415'
            carpeta_fecha = os.path.join(
                destino_local, fecha_ini.strftime('%Y'), fecha_ini.strftime('%m'), fecha_ini.strftime('%d')
            )

            fichero_sumario_xml = os.path.join(carpeta_fecha, 'index.xml')
            logger.debug(f"Ruta fichero sumario xml : {fichero_sumario_xml}")

            # Eliminar el sumario XML si existe
            if os.path.exists(fichero_sumario_xml):
                os.remove(fichero_sumario_xml)

            logger.info(f'Solicitando {boe_api_sumario}{fecha_ymd} --> {fichero_sumario_xml}')

            xml_content = self.get_xml_file(url=boe_api_sumario + fecha_ymd)
            self.save(content=xml_content, path=fichero_sumario_xml)

            self.information.dw_files_paths = self.get_pdf_url_from_xml(archivo_xml=fichero_sumario_xml)
            logger.info(
                f'Numero de URLs de PDFs obtenidos para la fecha {fecha_ini} : {len(self.information.dw_files_paths)}'
            )

            logger.info('Guardando en destino documentos PDFs')
            for file_i, file_path in enumerate(self.information.dw_files_paths):
                if (file_i + 1) <= self.information.batch:
                    file_name, response = self.get_files(
                        web_url=self.information.web_url,
                        file_url=file_path,
                        local_file_path=carpeta_fecha
                    )
                    self.save(content=response.content, path=os.path.join(carpeta_fecha, file_name))

            fecha_ini += timedelta(days=1)

    def get_xml_file(self, url):
        """
        Mediante get http request obtiene contenido del archivo xml
        Args:
            url (_type_): _description_

        Returns:
            _type_: _description_
        """
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f'Error al descargar el documento: {response.status_code} URL: {url}')
        return None

    def get_pdf_url_from_xml(self, archivo_xml):
        tree = ET.parse(archivo_xml)
        root = tree.getroot()
        # Extraemos todas las URLs de los PDFs
        #####
        # Podria añadir extraccion de metadatos del archivo xml para cada dia (un xml por dia y varios pdfs)
        # despues asociar esos metadatos a cada embedding de cada pdf de cada dia en el proceso de vewctorDB
        #####
        urls_pdf = []
        for urlPdf in root.findall('.//urlPdf'):
            url = urlPdf.text  # Obtén el texto del elemento, que es la URL
            urls_pdf.append(url)
        return urls_pdf

    def get_files(self, web_url, file_url, local_file_path) -> tuple[str, Response]:
        """_summary_

        Args:
            web_url (_type_): _description_
            file_url (_type_): _description_
            local_file_path (_type_): _description_

        Returns:
            Tuple[str,str]: [name_file, http Response object]
        """
        url_completa = web_url + file_url
        respuesta = requests.get(url_completa)

        if respuesta.status_code == 200:
            nombre_pdf = file_url.split('/')[-1]  # Extraemos el nombre del archivo desde la URL
            logger.info(f'Archivo descargado con exito: {os.path.join(local_file_path, nombre_pdf)}')

            return nombre_pdf, respuesta

        else:
            logger.error(f'Error al descargar {url_completa}: {respuesta.status_code}')
            return "", respuesta

    def save(self, content, path):
        # Asegurarse de que 'path' incluya un nombre de archivo.
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # create file and write content
        with open(path, 'wb') as file:
            file.write(content)