
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



