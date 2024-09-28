import logging
import logging.config
import logging.handlers
from typing import Union
from pydantic import BaseModel
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser, BaseTransformOutputParser
from exceptions.exceptions import LangChainError
from GRAPH_RAG.models import (
    get_open_ai_json,
    get_nvdia,
    get_ollama
)


# Logging configuration
logger = logging.getLogger(__name__)


def get_chain( 
                prompt_template: Union[PromptTemplate,ChatPromptTemplate], 
                parser: BaseTransformOutputParser,
                get_model: callable,
                temperature : float = 0.0
              ) -> LLMChain:
    """Retorna la langchain chain"""
    if not prompt_template and not isinstance(prompt_template,(PromptTemplate,ChatPromptTemplate)):
      raise LangChainError()
    
    logger.info(f"Initializing LangChain using : {get_model.__name__}")
    model = get_model(temperature=temperature)
    chain = prompt_template | model | parser()
    
    return chain
  
def _get_structured_chain( 
                output_model : BaseModel,
                prompt_template: str, 
                get_model: callable = get_nvdia,
                temperature : float = 0.0
              ) -> LLMChain:
    """Retorna la langchain chain con una estructura de salida fijada por un basemodel.
    De esta forma no es necesario un parser"""
    if not prompt_template and not isinstance(prompt_template,PromptTemplate):
      raise LangChainError()
    
    logger.info(f"Initializing Structured LangChain Chain using : {get_model.__name__}")
    model = get_model(temperature=temperature)
    structured_llm_model = model.with_structured_output(output_model)
    structured_chain = prompt_template | structured_llm_model 
    
    return structured_chain