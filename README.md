
# Procesamiento del BOE mediante sistema basado semantic chunking, RAPTOR Dimensional RAG y agente ó grafo lógico que hace uso de LLM LLAMA3 para generación de texto

## Pasos para usar BOE ChatBot y chatear con tus PDFs del BOE:

**Paso previo crea una carpeta en local donde clonar los repositorios del frontend y del backend.**

### Front-End
1. Clona el repositorio del frontend:

```sh
git clone https://github.com/kokeitor/chatbot-boe-frontend-react.git
```

2. instala node si aun no lo tienes:
accede a https://nodejs.org/es/download/ y descarga la versión de node que corresponda a tu sistema operativo.

3. instala vite si aun no lo tienes:

4. levanta el cliente en local:

```sh
npm install
npm run dev
```
5. Anota la url donde se levanta el cliente en local.

ejemplo: 
FRONT_END_URL=http://localhost:5173

6. Cuando levantes el backend en el archivo .env añade la url del endpoint de la API del backend:

VITE_BACK_END_BASE_URL=https://backend-boechat.onrender.com

### Back-End
1. Clona el repositorio del bakend :

```sh
git clone https://github.com/kokeitor/backend_boechat.git
```
2. Abrir la carpeta "backend" en tu editor de código favorito.
3. crear un entorno virtual de python con el siguiente comando:

```sh
python -m venv venv
```
4. Activar el entorno virtual:

```sh
env/Scripts/activate
```
5. Instalar las dependencias del proyecto:

```sh
pip install -r requirements_production.txt
```
6. Renombra el archivo ".env.example" a ".env" y añade tus claves de OpenAI, LangSmith, LangChain, HuggingFace, RAPTOR, Pinecone, NVIDIA y otras claves que necesites.
para onbtener las claves de cada una de estas APIs, puedes seguir estos pasos:

- OpenAI API Key: https://platform.openai.com/account/api-keys
- LangSmith API Key: https://platform.langchain.com/account/api-keys
- LangChain API Key: https://platform.langchain.com/account/api-keys
- HuggingFace API Key: https://huggingface.co/settings/tokens
- RAPTOR API Key: https://raptor.ai/api-keys
- Pinecone API Key: https://app.pinecone.io/organizations/my-organization/keys
- NVIDIA API Key: https://platform.openai.com/account/api-keys

```
OPENAI_API_KEY=
LANGSMITH= 
LLAMA_CLOUD_API_KEY=
LANGCHAIN_API_KEY=
PINECONE_API_KEY= 
PINECONE_INDEX_NAME=boe
PINECONE_INDEX_NAMESPACE=boe_namespace_1
APP_MODE=graph
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL_GPT4=all‑MiniLM‑L6‑v2.gguf2.f16.gguf
BOE_WEB_URL=https://boe.es
HUG_API_KEY=
GROQ_API_KEY=
NVIDIA_API_KEY=
RAPTOR_CHUNKS_FILE_NAME=
RAPTOR_CHUNKS_FILE_EXTENSION=csv
MAX_FILES=2
FRONT_END_URL=

```
7. Pega la URL de tu cliente  de React en la variable FRONT_END_URL


8. Navega hasta la raiz del directorio y levanta el servidor de FastApi en local:
```sh
uvicorn src.main:app --reload
```