
# Instrucciones para Ejecutar la Herramienta


## Docker

- **Descargar imagen de Docker para ChromaDB:**

```sh
docker pull chromadb/chroma
```

- **Ejecutar contenedor de ChromaDB:**

```sh
docker run -p 8000:8000 --name chroma chromadb/chroma
```

- **Buscador web local:**
  - URL: [http://localhost:8000/api/v1](http://localhost:8000/api/v1)

## Comandos para Ejecutar la Herramienta

1. **Navegar hasta el directorio root del proyecto:**

```sh
cd ...\proyecto
```

2. **Configurar herramienta de descarga (archivo de configuración JSON):**

   2.1. Configurar fecha inicial y fin del BOE.  
   2.2. Configurar ruta de guardado (se recomienda dejar como `./data`).

3. **Ejecutar el script de descarga:**

```sh
python src/main_download.py
```

4. **Configurar herramienta de ETL (archivo de configuración JSON):**

   4.1. Parámetros de configuración.

5. **Ejecutar el script de ETL:**

```sh
python src/main_etl.py
```

6. **Fine-tuning del modelo tipo BERT:**

- **Acudir a file main_hg_push_hub.py y cambiar fechas from t to de los archivos a pushear:**

```python
    setup_logging()
    
    # Load environment variables from .env file
    load_dotenv()

    os.environ['HG_API_KEY'] = str(os.getenv('HG_API_KEY'))
    os.environ['HG_REPO_DATASET_ID'] = str(os.getenv('HG_REPO_DATASET_ID'))
    
    hg_dataset = HGDataset(
        data_dir_path="./data/boedataset", 
        hg_api_token=str(os.getenv('HG_API_KEY')), 
        repo_id=str(os.getenv('HG_REPO_DATASET_ID')), 
        from_date="2024-07-10", 
        to_date="2024-07-16",
        desire_columns=["text", "chunk_id","label"]
    )
    
    hg_dataset.initialize_data()
    hg_dataset.push_to_hub()
    print("Dataset cargado y subido correctamente.")
```

- **Comando para ejecutar y pushear a HG HUB los archivos CSV o Parquet:**

  ```sh
  python src/main_hg_push_hub.py
  ```

- **Fine tunning ejecucion archivo: main_run_classification.py**

  1. Modificacion archivo configuracion : fine_tune_config.json

  2. Ejecucion script desde raiz proyecto especificando el archivo de configuracion:
    ```sh
     python /src/main_run_classification.py ./config/finetune/fine_tune_config.json
    ```

### Explicación Adicional

- **Streamlit**: Este comando inicia la aplicación Streamlit en el puerto 8052.
- **Docker**: Instrucciones para descargar y ejecutar un contenedor Docker con ChromaDB.
- **Configuración de Herramienta de Descarga**: Configuración necesaria para descargar datos del BOE, incluyendo la configuración de fechas y la ruta de guardado.
- **Ejecución de Scripts**: Comandos para ejecutar los scripts de descarga, procesamiento (ETL) y fine-tuning del modelo.
- **Hugging Face Hub**: Comando para subir los archivos procesados a Hugging Face Hub.
