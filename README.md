# BOE ChatBot

## Pasos para usar BOE ChatBot y chatear con tus PDFs del BOE:

### Paso previo: 
Crea una carpeta en tu máquina local donde clonar los repositorios del frontend y del backend.

### Front-End

1. Clona el repositorio del frontend:

   ```sh
   git clone https://github.com/kokeitor/chatbot-boe-frontend-react.git
   ```

2. Instala Node.js si aún no lo tienes instalado:

   - Accede a [https://nodejs.org/es/download/](https://nodejs.org/es/download/) y descarga la versión de Node.js correspondiente a tu sistema operativo.

3. Instala Vite si aún no lo tienes:

   Vite se instalará automáticamente cuando ejecutes el siguiente paso.

4. Levanta el cliente en local:

   ```sh
   npm install
   npm run dev
   ```

5. Anota la URL donde se levanta el cliente en local.

   Ejemplo: 

   ```env
   FRONT_END_URL=http://localhost:5173
   ```

6. Cuando levantes el backend, añade la URL del endpoint de la API en el archivo `.env` del frontend:

    Ejemplo:
    
   ```env
   VITE_BACK_END_BASE_URL=http://localhost:8000
   ```

### Back-End

1. Clona el repositorio del backend:

   ```sh
   git clone https://github.com/kokeitor/backend_boechat.git
   ```

2. Abre la carpeta "backend" en tu editor de código favorito.

3. Crea un entorno virtual de Python con el siguiente comando:

   ```sh
   python -m venv venv
   ```

4. Activa el entorno virtual:

   - En Windows:
     ```sh
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

5. Instala las dependencias del proyecto:

   ```sh
   pip install -r requirements_production.txt
   ```

6. Renombra el archivo `.env.example` a `.env` y añade tus claves de OpenAI, LangSmith, LangChain, HuggingFace, RAPTOR, Pinecone, NVIDIA y otras APIs necesarias.

   Para obtener las claves de estas APIs, puedes seguir los siguientes enlaces:

   - [OpenAI API Key](https://platform.openai.com/account/api-keys)
   - [LangSmith API Key](https://platform.langchain.com/account/api-keys)
   - [LangChain API Key](https://platform.langchain.com/account/api-keys)
   - [HuggingFace API Key](https://huggingface.co/settings/tokens)
   - [RAPTOR API Key](https://raptor.ai/api-keys)
   - [Pinecone API Key](https://app.pinecone.io/organizations/my-organization/keys)
   - [NVIDIA API Key](https://developer.nvidia.com/)

   Asegúrate de rellenar tu archivo `.env` con las siguientes variables:

   ```env
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

7. Pega la URL de tu cliente de React en la variable `FRONT_END_URL`.

8. Navega hasta la raíz del directorio y levanta el servidor FastAPI en local:

   ```sh
   uvicorn src.main:app --reload
   ```
