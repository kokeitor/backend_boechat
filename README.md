
# Procesamiento del BOE mediante sistema basado semantic chunking, RAPTOR Dimensional RAG y agente ó grafo lógico que hace uso de LLM LLAMA3 para generación de texto

## 🚀 Proyecto en desarrollo continuo 

### Descripcion, introducción y contexto

- Título del proyecto: Procesamiento del BOE mediante una arquitectura compuesta por un sistema de RAG y en un agente basado en un modelo grande de lenguaje (LLM).

- Introducción:
Este trabajo tiene como objetivo principal el procesamiento, resumen, recuperación u obtención de información relevante y precisa de grandes cantidades de documentos del boletín oficial del estado español (BOE). Se pretende hacer uso de este complejo sistema de RAG y un posterior modelo de agente LLM que use LLAMA 3 para recuperar información precisa de este corpus y, en función de esta, permitir que el modelo realice cualquier tarea de generación texto demandada por el usuario. El presente trabajo se centrará en tareas de generación de texto como respuestas, resúmenes, procesamiento de tablas, recuperación de datos o sintetización de la información, entre otras.

- Contexto del Trabajo a Realizar:
El proyecto se enmarca en el ámbito de la aplicación de técnicas avanzadas de inteligencia artificial para el análisis de documentos oficiales, con el objetivo de abordar la automatización del proceso de extracción de datos relevantes de, en este caso, documentación técnica de varios indoles gubernamentales. Se busca la mejora así de la accesibilidad y utilidad de la información contenida en estos documentos para diversos usuarios y aplicaciones.
Objetivo de Estudio:
El objetivo principal de este trabajo es desarrollar un sistema eficiente y automatizado para que procese y extraiga la información relevante y precisa de los documentos del BOE que el usuario demande al modelo LLM. 

### Objetivos:

1. Creación de una herramienta de extracción, procesamiento, transformación, preparación y carga de los documentos en formato PDF de la página oficial del gobierno de España.

2. Diseño de un sistema de Recuperación Aumentada con Generación o RAG, por sus siglas en inglés, que sea capaz de extraer la información precisa de estos textos procesados. Se pretende que este sistema esté compuesto de varias partes. Una de estas partes será un modelo de tipo transformer-encoder, en concreto de tipo BERT, que será entrenado en la tarea de clasificación multi etiqueta utilizando una muestra significativa de fragmentos semánticos pertenecientes a este corpus de textos del BOE. De esta forma, y con este modelo entrenado en la clasificación multi etiqueta probabilística de fragmentos de textos semánticamente relacionados, se buscará mejorar la tarea de aportar un contexto adecuado y relevante para la generación por parte del modelo LLM.

3. Desarrollar un agente basado en el modelo grande de lenguaje LLAMA 3 para generar, sintetizar, obtener o evaluar la información específica, en función de lo que desee el usuario, del contexto retornado por el sistema RAG diseñado y obtenido de los documentos procesados del BOE. Dentro de este último punto destaca el hecho de evaluar la eficiencia y precisión de toda la arquitectura desarrollada, tanto a nivel generación por parte del LLM como de recuperación o aporte de contexto del sistema RAG mediante métricas adecuadas dentro del ámbito del procesamiento de lenguaje natural. También el agente deberá contar con herramientas y una lógica para mejorar estas métricas y palear posibles problemas de alucinaciones, respuestas poco relevantes o contextos retronados por parte del sistema RAG poco concluyentes o relacionados para la tarea de generación. 

### Metodología:

La metodología a utilizar se basará en un enfoque documentativo previo, seguido de otro más iterativo y experimental, con fases de desarrollo, prueba y validación del sistema propuesto, con el fin de lograr una investigación rigurosa y resultados relevantes.
Se prevé utilizar una combinación de métodos cualitativos y cuantitativos, incluyendo análisis documental, desarrollo de software, pruebas de rendimiento y análisis estadístico.
Se llevará a cabo un proceso de recolección de datos que incluirá la descarga y preparación de documentos del BOE para su procesamiento mediante una herramienta de desarrollo propio que lo permita. El procesamiento que lleva a cabo esta herramienta se centra en el análisis gramatical (parse en inglés) del texto de estos documentos, la aplicación de técnicas de procesamiento del lenguaje natural para la limpieza de los textos parseados o la fragmentación semántica y temática en textos de menor tamaño mediante aplicación de métricas que miden la distancia geométrica entre las representaciones vectoriales de los textos (embeddings) o la adición de metadatos cronológicos, identificativos, temáticos y físicos a cada texto, entre otras técnicas y funciones de la herramienta.
La muestra estará compuesta por un conjunto representativo de documentos del BOE, seleccionados para cubrir una variedad de temas y formatos. Como la publicación del BOE es diaria se buscará una ventana temporal donde los documentas existentes comprendan la gran variedad de asuntos y cuestiones publicadas. 

### Conclusiones:

El presente anteproyecto establece las bases y los lineamientos para la elaboración del Trabajo Final de Máster, delineando el contexto, los objetivos y la metodología a emplear. Se espera que este trabajo contribuya al conocimiento en el campo de la inteligencia artificial aplicada a la automatización del procesamiento de documentos oficiales y brinde información relevante para la mejora de la accesibilidad y eficiencia en la gestión de información gubernamental en este caso, o cualquier sector de voluminosa, técnica y continuamente actualizada información.

### Referencias y documentación 

Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). Corrective Retrieval Augmented Generation. arXiv. https://arxiv.org/abs/2401.12345
Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv. https://arxiv.org/abs/2310.11511
Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. arXiv. https://arxiv.org/abs/2403.14403
Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv. https://arxiv.org/abs/2401.18059
He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. arXiv. https://arxiv.org/abs/2111.09543
Meta AI. (2024, April 18). Introducing Meta Llama 3: The most capable openly available LLM to date. Meta. https://ai.meta.com/blog/meta-llama-3/
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv. https://doi.org/10.48550/arXiv.1908.10084
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv. https://doi.org/10.48550/arXiv.1810.04805


#### Desglose de cada elememto

- Detalles de la arquitectura

1. Herramienta de descarga local mediante libreria request de los PDFs BOE que esten entre dos fechas especificadas como parametro. 
2. Extracción, Parseo, Procesamiento, Limpieza, Separacaion [Semmantic chunking] y Carga de los documentos PDFs descargados mediante el uso de pipeline segun archivo de comfiguracion json. 
3. ...

## PROMPT

quiero publicar en linkedin que he creado una web full stack app para el procesamiento del boe. La apliacion web se compone de una parte frontend deesarrollada con React.js y de una parte backend desarrollada con FastApi. A continuacion te voy a dar mas detalles de la app para que generes un texto para la publicacion.
Detalles:
La aplicacion se ha construido utilizando patra el fron Node.js y Vit. Estas tecnologias han permitido el desarrollo del UI (user interface) de la aplicacion.
Por otro lado el modelo de procesamiento de los archivos PDF del BOE, se aloja en el backend dsarrollado con Python y que cuenta con una API construida con FastApi que permite la interaccion frontend-backend de la aplicacion. El modelo de procesamiento se ha desarrollado y diseñado como parte de mi trabajo de fin de grado con el objetivo de mejorar , optimizar y hacer mas accesibles los textos y corpus legales del BOE a todo usuario no experto o con poco tiempo. El modelo por debajo cuenta una arquitectura de grafo logico que hace uso de agentes, deferentes métodos de procesamiento del lenguaje natural (NLP) y  avanzadas técnicas de RAG para dar una respuesta precisa al usuario segun el PDF o los PDFs del BOE que este quiera consultar.


---
Aquí tienes una versión mejorada de la publicación incorporando los puntos adicionales:

---

🚀 ¡Estoy emocionado de compartir el resultado de mi Trabajo de Fin de Grado! He desarrollado una aplicación web full stack para el procesamiento avanzado de archivos del BOE (Boletín Oficial del Estado), diseñada para optimizar el acceso y comprensión de textos legales 📜💻.

🔸 **Frontend con React.js**: La interfaz de usuario permite subir archivos PDF para ser procesados y ofrece la posibilidad de interactuar con el sistema como si fuera un experto legal o normativo. El frontend consume una API que permite al usuario chatear y consultar documentos, accediendo a información precisa y fundamentada de forma ágil.

🔸 **Backend en Python y FastAPI**: Aquí reside la lógica de procesamiento, basada en 4 pilares clave:

1. **ETL y preprocesamiento de PDFs**: Utilizando técnicas avanzadas de **NLP**, se realiza el análisis de los archivos del BOE subidos por el usuario. El modelo extrae metadatos temáticos y aplica un algoritmo de **chunking semántico** mediante el uso de **SBERT** para fragmentar el corpus en partes con significado. Estos fragmentos se almacenan, junto con sus vectores densos y metadatos, en una **base de datos vectorial Pinecone**, uno de los servicios más potentes del mercado.

2. **Método RAPTOR adaptado**: Este proceso agrupa los fragmentos en clusters y genera un resumen para cada uno. Estos resúmenes, añadidos a los metadatos, mejoran el rendimiento del sistema de **Recuperación Aumentada con Generación (RAG)** al incorporar parámetros adicionales de semejanza, haciendo el proceso más eficiente y preciso.

3. **Grafo multi-agente con LangChain y LangGraph**: Aquí se implementa un sistema de agentes que utilizan **LLMs** y técnicas avanzadas de **RAG** como **SELF-RAG** y **C-RAG**. El grafo asegura que las respuestas generadas a partir de los PDFs sean precisas y relevantes, evitando problemas como las alucinaciones del modelo o la recuperación de chunks no útiles desde la base de datos vectorial.

4. **API en FastAPI**: Esta API es la columna vertebral de la aplicación, incorporando funcionalidades como **streaming**, un wrapper sobre el modelo vanilla de **OpenAI**, gestión de la memoria del modelo y facilitando la interacción en tiempo real con el frontend desarrollado en React.js.

🌟 **Objetivo del Proyecto**: Hacer accesible y comprensible la información legal del BOE a cualquier usuario, incluso aquellos con poco tiempo o sin experiencia, utilizando tecnología de vanguardia en procesamiento del lenguaje natural y sistemas de inteligencia artificial.

Este proyecto está en **desarrollo continuo** y ya supone un gran avance en la **automatización de la gestión de información legal**. No podría estar más emocionado por el impacto que esto puede tener en la accesibilidad y utilidad de la información gubernamental.

#FullStack #ReactJS #FastAPI #NLP #RAG #SBERT #LangChain #Pinecone #LLMs #LegalTech #BOE #IA #TrabajoDeFinDeGrado #DesarrolloWeb

---

Debido, pienso, a una necesidad común y como continuación de un proyecto personal que inicié a raíz de mi Trabajo de Fin de Máster, he desarrollado una aplicación web Full Stack para el procesamiento automatizado de archivos del BOE (Boletín Oficial del Estado). Una aplicación diseñada para optimizar el acceso, la comprensión y **la extracción eficiente de información clave** de estos textos legales y normativos, **facilitando su consulta y análisis por parte de cualquier usuario** 📜💻.

🔸 El F**ront-End** esta desarrollado con **React.js y Vite** y consume la Api del modelo permitiendo subirle los archivos PDF con los que se quiera chatear o de los que se quiera extraer informacion. En las diefrentes paginas de la web UI se pueden consutar los diagrams explicativos de como funciona el modelo, descragar su documentacion tecnica u observar todas las tecnologias utilizadas en el desarrollo de la aplicación.

🔸 En el **Back-End** se encuentra toda la lógica del modelo de procesamiento basada en 4 pilares:
1. **ETL de preprocesamiento** de los PDFs del BOE que el usuario quiere consultar o con los que quiere chatear. Esto incluye técnicas avanzadas de NLP con recopilación de metadatos asociados a la temática de los textos, un algoritmo de chunking semántico utilizando un reciente modelo de embeddding multilenguaje tipo SBERT para dividir el corpus en fragmentos de texto semánticamente semejantes y, por ultimo, almacenamiento de los chunks ,sus metadatos y sus vectores densos en una base de datos vectorial utilizando uno de los servicios mas potentes del mercado como es Pinecone.
2. RAPTOR adaptado. Este método consiste en hacer una clusterización de todos estos fragmentos de texto y crear un resumen por cada cluster con el objetivo de añadirlo a sus metadatos. Esto mejora el posterior proceso de RAG al incluir como parámetro de semejanza este cluster summary.
3. **Grafo multi-agente construido y diseñado con LangChain y LangGraph**. En la cual se implementa un sistema de agentes que hacen uso de LLMs, prompting técnicos del boe para llevar acabo técnicas de RAG (SELF-RAG y C-RAG). Esto permite generar una respuesta precisa y fundamentada por el contenido de los PDFs, solucionando problemas como la alucinación de los modelos o unos chunks recuperados, de la base de datos vectorial, no útiles para generar una respuesta a las queries del usuario durante la conversación.
4. **Desarrollo de una API utilizando FastApi**, la cual incorpora streaming, un wrapper del modelo vanilla con el modelo de openAI, memoria del modelo, ... y dota a la aplicación de interaccion con el front-end de React.js.

En resumen y para finalizar, **el objetivo del proyecto** es optimizar la extraccion de información judicial, legal, normativa o de otras muchas indoles del BOE a cualquier usuario, incluso aquellos no familiarizados con este tipo de informacion, utilizando la ineteligencia artificial y optimizando este proceso tedioso y complicado.

Como de costumbre, este proyecto está en **desarrollo continuo** y soy consciente de que es altamente escalabale y mejorable.
El codigo utilizado y toda los pasos a seguir para hacer uso de la aplicaion de manera local se encuentra en mi repositorio de GitHub [https://github.com/koke143/BoeChatBot](https://github.com/koke143/BoeChatBot) ya que aún ha sido desplegada en produccion.

#FullStack #ReactJS #FastAPI #NLP #RAG #SBERT #LangChain #Pinecone #LLMs #LegalTech #BOE #IA #DesarrolloWeb


----


Debido, pienso, a una necesidad común y como continuación de un proyecto personal que inicié a raíz de mi Trabajo de Fin de Máster, he desarrollado una aplicación web Full Stack para el procesamiento automatizado de archivos del BOE (Boletín Oficial del Estado). Una aplicación diseñada para optimizar el acceso, la comprensión y **la extracción eficiente de información clave** de estos textos legales y normativos, **facilitando su consulta y análisis por parte de cualquier usuario** 📜💻.

🔸 El **Front-End** está desarrollado con **React.js y Vite** y consume la API del modelo, permitiendo subir archivos PDF con los que se quiera chatear o de los que se quiera extraer información. En las diferentes páginas de la web UI se pueden consultar los diagramas explicativos de cómo funciona el modelo, descargar su documentación técnica u observar todas las tecnologías utilizadas en el desarrollo de la aplicación.

🔸 En el **Back-End** desarrolado con **Python y FastAPI**, se encuentra toda la lógica del modelo de procesamiento basada en 4 pilares:
1. **ETL de preprocesamiento** de los PDFs del BOE que el usuario quiere consultar o con los que quiere chatear. Esto incluye técnicas avanzadas de NLP con recopilación de metadatos asociados a la temática de los textos, un algoritmo de chunking semántico utilizando un reciente modelo de embedding multilinguaje tipo SBERT para dividir el corpus en fragmentos de texto semánticamente semejantes y, por último, almacenamiento de los chunks, sus metadatos y sus vectores densos en una base de datos vectorial utilizando uno de los servicios más potentes del mercado, como es Pinecone.
2. RAPTOR adaptado. Este método consiste en hacer una clusterización de todos estos fragmentos de texto y crear un resumen por cada cluster con el objetivo de añadirlo a sus metadatos. Esto mejora el posterior proceso de RAG al incluir como parámetro de semejanza este resumen del cluster.
3. **Grafo multi-agente construido y diseñado con LangChain y LangGraph**. En el cual se implementa un sistema de agentes que hacen uso de LLMs y prompting técnicos del BOE para llevar a cabo técnicas de RAG (SELF-RAG y C-RAG). Esto permite generar una respuesta precisa y fundamentada por el contenido de los PDFs, solucionando problemas como la alucinación de los modelos o unos chunks recuperados de la base de datos vectorial que no son útiles para generar una respuesta a las queries del usuario durante la conversación.
4. **Desarrollo de una API utilizando FastAPI**, la cual incorpora streaming, un wrapper del modelo vanilla con el modelo de OpenAI, memoria del modelo, etc., y dota a la aplicación de interacción con el front-end de React.js.

En resumen, y para finalizar, **el objetivo del proyecto** es optimizar la extracción de información judicial, legal, normativa o de otras muchas índoles del BOE a cualquier usuario, incluso aquellos no familiarizados con este tipo de información, utilizando la inteligencia artificial y optimizando este proceso tedioso y complicado.

Como de costumbre, este proyecto está en **desarrollo continuo** y soy consciente de que es altamente escalable y mejorable.  
El código utilizado y todos los pasos a seguir para hacer uso de la aplicación de manera local se encuentran en mi repositorio de GitHub [https://github.com/koke143/BoeChatBot](https://github.com/koke143/BoeChatBot), ya que aún no ha sido desplegada en producción.

#FullStack #NodeJS #ReactJS #Python #FastAPI #NLP #RAG #SBERT #LangChain #Pinecone #LLMs #LegalTech #BOE #IA #DesarrolloWeb

---
