import tiktoken
import logging
import logging.config
import logging.handlers
from transformers import AutoTokenizer, DebertaModel, GPT2Tokenizer
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Union, Optional
import matplotlib.pyplot as plt
from langchain_nvidia_ai_endpoints import ChatNVIDIA


# Set the default font to DejaVu Sans
# or another font that includes the glyphs
plt.rcParams['font.family'] = 'DejaVu Sans'

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("ETL_module_logger")


# Tokenizers
TOKENIZER_GPT3 = tiktoken.encoding_for_model("gpt-3.5")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(
    'gpt2', clean_up_tokenization_spaces=False)
TOKENIZER_LLAMA3 = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", clean_up_tokenization_spaces=False)
tokenizer_deberta = AutoTokenizer.from_pretrained(
    "microsoft/deberta-base", clean_up_tokenization_spaces=False)


class LabelGenerator:
    LABELS = """Leyes Orgánicas,Reales Decretos y Reales Decretos-Leyes,Tratados y Convenios Internacionales,
    Leyes de Comunidades Autónomas,Reglamentos y Normativas Generales,Nombramientos y Ceses,
    Promociones y Situaciones Especiales,Convocatorias y Resultados de Oposiciones,Anuncios de Concursos y Adjudicaciones de Plazas,
    Ayudas, Subvenciones y Becas,Convenios Colectivos y Cartas de Servicio,Planes de Estudio y Normativas Educativas,
    Convenios Internacionales y Medidas Especiales,Edictos y Notificaciones Judiciales,Procedimientos y Citaciones Judiciales,
    Licitaciones y Adjudicaciones Públicas,Avisos y Notificaciones Oficiales,Anuncios Comerciales y Convocatorias Privadas,
    Sentencias y Autos del Tribunal Constitucional,Orden de Publicaciones y Sumarios,Publicaciones por Órgano Emisor,
    Jerarquía y Autenticidad de Normativas,Publicaciones en Lenguas Cooficiales,Interpretaciones y Documentos Oficiales,
    Informes y Comunicaciones de Interés General,Documentos y Estrategias Nacionales,Medidas de Emergencia y Seguridad Nacional,
    Anuncios de Regulaciones Específicas,Normativas Temporales y Urgentes,Medidas y Políticas Sectoriales,
    Todos los Tipos de Leyes (Nacionales y Autonómicas),Todos los Tipos de Decretos (Legislativos y no Legislativos),
    Convocatorias y Resultados Generales (Empleo y Educación),Anuncios y Avisos (Oficiales y Privados),
    Judicial y Procedimientos Legales,Sentencias y Declaraciones Judiciales,Publicaciones Multilingües y Cooficiales,
    Informes y Estrategias de Política,Emergencias Nacionales y Medidas Excepcionales,Documentos y Comunicaciones Específicas"""

    def __init__(self, tokenizer, labels: Optional[list[str]] = None, model: str = 'GPT', max_samples: int = 10):
        self.model_label = model
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        _labels = labels if labels is not None else LabelGenerator.LABELS.replace(
            "\n", "").split(',')
        self.labels = [l.strip() for l in _labels]
        self.label2id = {label.strip(): label_index for label_index,
                         label in enumerate(_labels)}

        logger.info(f"label2id : {self.label2id}")

        self.prompt = PromptTemplate(
            template="""You are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to classify the provided text using the specified list of labels. The possible labels are: {labels}
            You must provide three possible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 1.
            Provide the output as a JSON with three keys: 'Label1', 'Label2', 'Label3' and for each label another two keys: "Label" and "Score".
            Text: {text}""",
            input_variables=["text", "labels"],
            input_types={"labels": list[str], "text": str}
        )
        self.new_prompt = PromptTemplate(
            template="""You are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to classify the provided text using the specified list of labels. The possible labels are: {labels}
            You must provide three possible labels ordered by similarity score with the text content.\n
            Provide the output as a JSON with three keys: 'Label1', 'Label2', 'Label3'.\n
            Text: {text}""",
            input_variables=["text", "labels"],
            input_types={"labels": list[str], "text": str}
        )
        self.alternative_prompt = PromptTemplate(
            template="""You are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to classify the provided text using the specified list of labels. The possible labels are: {labels}
            You must provide 10 possible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 100.
            The scores for the rest of the labels must be 0. Provide the output as a JSON with the label names as keys and their similarity scores as values.
            Text: {text}""",
            input_variables=["text", "labels"],
            input_types={"labels": list[str], "text": str}
        )
        self.llama_prompt = PromptTemplate(
            template="""You are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to classify the provided text using the specified list of labels. The possible labels are: {labels}
            You must provide three possible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 1.
            Provide the output as a JSON with three keys: 'Label1', 'Label2', 'Label3' and for each label another two keys: "Label" and "Score".
            user
            Text: {text}assistant""",
            input_variables=["text", "labels"],
            input_types={"labels": list[str], "text": str}
        )
        self.llama_new_prompt = PromptTemplate(
            template="""You are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to classify the provided text using the specified list of labels. The possible labels are: {labels}
            You must provide three possible labels ordered by similarity score with the text content.\n
            Provide the output as a JSON with three keys: 'Label1', 'Label2', 'Label3'.\n
            Text: {text}""",
            input_variables=["text", "labels"],
            input_types={"labels": list[str], "text": str}
        )
        self.alternative_llama_prompt = PromptTemplate(
            template="""systemYou are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to classify the provided text using the specified list of labels. The possible labels are: {labels}
            You must provide 10 possible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 100.
            The scores for the rest of the labels must be 0. Provide the output as a JSON with the label names as keys and their similarity scores as values.
            user
            Text: {text}assistant""",
            input_variables=["text", "labels"],
            input_types={"labels": list[str], "text": str}
        )

        models = {
            'GPT': ChatOpenAI(model_name='gpt-4o-mini', temperature=0),
            'NVIDIA-LLAMA3': ChatNVIDIA(model_name='meta/llama3-70b-instruct', temperature=0),
            'LLAMA': ChatOllama(model='llama3', format="json", temperature=0),
            'LLAMA-GRADIENT': ChatOllama(model='llama3-gradient', format="json", temperature=0)
        }

        self.model = models.get(self.model_label, None)
        if not self.model:
            logger.error("Model Classifier Name not correct")
            raise AttributeError("Model Classifier Name not correct")

        if self.model_label == "NVIDIA-LLAMA3":
            self.chain = self.llama_new_prompt | self.model | JsonOutputParser()
        elif self.model_label == "GPT":
            self.chain = self.new_prompt | self.model | JsonOutputParser()
        else:
            self.chain = self.llama_prompt | self.model | JsonOutputParser()

    def _get_tokens(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.exception(f"Tokenization error: {e}")
            return len(self.tokenizer(text)["input_ids"])

    def invoke(self, docs: list[Document]) -> list[Document]:
        docs_copy = docs.copy()

        for i, doc in enumerate(docs_copy):
            if i >= self.max_samples:
                logger.warning(
                    f"Reached max samples: {self.max_samples} while generating labels")
                break

            chunk_text = doc.page_content
            chunk_tokens = self._get_tokens(text=chunk_text)
            chunk_len = len(chunk_text)

            # Update metadata
            doc.metadata['num_tokens'] = chunk_tokens
            doc.metadata['num_caracteres'] = chunk_len

            generation = {"label1": "", "label3": "", "label2": ""}
            MIN_CHUNK_LEN = 3
            try:
                if len(chunk_text) > MIN_CHUNK_LEN:
                    generation = self.chain.invoke(
                        {"text": chunk_text, "labels": self.labels})
                logger.info(
                    f"LLM output for chunk of len {len(chunk_text)}:\n{generation=}")
            except Exception as e:
                logger.exception(
                    f"LLM Error generation error message for chunk of len = {len(chunk_text)}\nCHUNK :{chunk_text}:\nERROR: {e}")

            if len(chunk_text) <= MIN_CHUNK_LEN:
                generation = {"label1": "NoChunkText",
                              "label3": "NoChunkText", "label2": "NoChunkText"}
            try:
                id2label = []
                for _, label in generation.items():
                    id2label.append(str(self.label2id.get(label, "999")))
                doc.metadata['label'] = id2label
                logger.info(f"for doc {i} id2label : {id2label}")
            except Exception as e:
                logger.exception(f"LLM Error message:  : {e}")

        return docs
