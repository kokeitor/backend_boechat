
# Instrucciones para levantar la aplicacion (API)

- **Fine tunning ejecucion archivo: main_run_classification.py**

  Ejecucion script desde raiz proyecto especificando el archivo de configuracion:
    ```sh
     \env\Scripts\activate 
     uvicorn src.main:app --reload
    ```