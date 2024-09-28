import os
import json
import logging
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from typing import Union, Optional, Callable, ClassVar
from langchain.chains.llm import LLMChain
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser, BaseOutputParser ,BaseTransformOutputParser
from GRAPH_RAG.chains import get_chain
from VectorDB.db import get_chromadb_retriever, get_pinecone_retriever
from GRAPH_RAG.prompts import (
    query_classify_prompt_openai,
    query_classify_prompt,
    grader_docs_prompt,
    gen_prompt,
    query_process_prompt,
    hallucination_prompt,
    grade_answer_prompt,
    grader_docs_prompt_openai,
    gen_prompt_openai,
    query_classify_groq_prompt,
    query_process_groq_prompt,
    grader_docs_groq_prompt,
    generate_groq_prompt,
    hall_groq_prompt,
    grade_answer_groq_prompt,
    query_process_prompt_openai,
    hallucination_prompt_openai,
    grade_answer_prompt_openai
    )
from GRAPH_RAG.base_models import (
    Question,
    Agent,
    VectorDB
)
from GRAPH_RAG.graph_utils import (
                        get_current_spanish_date_iso, 
                        get_id
                        )
from exceptions.exceptions import NoOpenAIToken, JsonlFormatError, ConfigurationFileError
from GRAPH_RAG.models import (
    get_nvdia,
    get_ollama,
    get_open_ai_json,
    get_groq,
    get_openai_emb,
    get_hg_emb
)

# Logging configuration
logger = logging.getLogger(__name__)

@dataclass()
class ConfigGraph:
    
    MODEL : ClassVar = {
            "GROQ":get_groq,
            "OPENAI": get_open_ai_json,
            "NVIDIA": get_nvdia,
            "OLLAMA": get_ollama
            }

    AGENTS: ClassVar = {
        "query_classificator": Agent(agent_name="query_classificator", model="OPENAI", get_model=get_open_ai_json, temperature=0.0, prompt=query_classify_prompt_openai,parser=JsonOutputParser),
        "docs_grader": Agent(agent_name="docs_grader", model="OPENAI", get_model=get_open_ai_json, temperature=0.0, prompt=grader_docs_prompt,parser=JsonOutputParser),
        "query_processor": Agent(agent_name="query_processor", model="OPENAI", get_model=get_open_ai_json, temperature=0.0, prompt=query_process_prompt,parser=JsonOutputParser),
        "generator": Agent(agent_name="generator", model="GROQ", get_model=get_groq, temperature=0.0, prompt=generate_groq_prompt,parser=StrOutputParser),
        "hallucination_grader": Agent(agent_name="hallucination_grader", model="GROQ", get_model=get_groq, temperature=0.0, prompt=hall_groq_prompt,parser=StrOutputParser),
        "answer_grader": Agent(agent_name="answer_grader", model="GROQ", get_model=get_groq, temperature=0.0, prompt=grade_answer_groq_prompt,parser=StrOutputParser),
    }
    
    VECTOR_DB: ClassVar = {
        "chromadb": VectorDB(
                                client="chromadb", 
                                hg_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                                k=3
                                ),
        "pinecone": VectorDB(
                                client="pinecone", 
                                hg_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                                k=3
                                ),
        "qdrant": VectorDB(
                                client="qdrant", 
                                hg_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                                k=3
                                )
    }
    
    config_path: Union[str, None] = None
    data_path: Union[str, None] = None
    graph : Union[StateGraph, None] = None
    compile_graph : Union[CompiledGraph, None] = None
    
    def __post_init__(self):
        
        if self.config_path is None:
            logger.exception("No se ha proporcionado ninguna configuración para la generación usando Agents")
            raise AttributeError("No se ha proporcionado ninguna configuración para la generación usando Agents")
        
        if self.data_path is None:
            logger.warning("No se han proporcionado datos para analizar para la generación usando Agents")
            #  raise AttributeError("No se han proporcionado datos para analizar para la generación usando Agents")
            
        if self.config_path is not None:
            self.config = self.get_config()
            logger.info(f"Definida configuracion mediante archivo JSON en {self.config_path}")
            
        if self.data_path is not None:
            self.data = self.get_data()
            logger.info(f"Definidos los datos mediante archivo JSON en {self.data_path}")
            if len(self.data) > 0:
                self.user_questions = [self.get_user_question(
                    q=user_q.get("user_question", None), 
                    ground_truth=user_q.get("ground_truth", None), 
                    date=user_q.get("date", None),
                    boe_id=user_q.get("boe_id",None)
                    ) for user_q in self.data]
            else:
                logger.exception("No se han proporcionado candidatos en el archivo JSON con el correcto fomato [ [cv : '...', oferta : '...'] , [...] ] ")
                raise JsonlFormatError()
        
        # Graph Agents configuration
        self.agents_config = self.config.get("agents", None)
        if self.agents_config is not None:
            self.agents = self.get_agents()
            
        # Graph VDB/Retrievers configuration
        self.vector_db_config = self.config.get("vector_db", None)
        if self.vector_db_config is not None:
            self.vector_db = self.get_vector_db()
        else:
            raise ConfigurationFileError(f"Error inside confiuration graph file -> No VectorDB defined fro retrieval") 

        # Graph configuration
        self.iteraciones = self.config.get("iteraciones", 10)
        self.thread_id = self.config.get("thread_id", "4")
        self.verbose = self.config.get("verbose", 0)
        
    def get_config(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.exception(f"Archivo de configuración no encontrado en {self.config_path}")
            raise FileNotFoundError(f"Archivo de configuración no encontrado en {self.config_path}")
        with open(self.config_path, encoding='utf-8') as file:
            config = json.load(file)
        return config
    
    def get_data(self) -> list[dict[str,str]]:
        if not os.path.exists(self.data_path):
            logger.exception(f"Config file not found in {self.data_path}")
            raise FileNotFoundError(f"Config file not found in {self.data_path}")
        with open(file=self.data_path, mode='r', encoding='utf-8') as file:
            logger.info(f"User question in file : {self.data_path} : ")
            try:
                data = json.load(file)
            except Exception as e:
                logger.exception(f"Error decoding JSON : {e}")
        return data
    
    def get_agents(self) -> dict[str,Agent]:
        agents = ConfigGraph.AGENTS.copy()
        for agent_graph in agents.keys():
            for agent, agent_config in self.agents_config.items():
                if agent_graph == agent:
                    model = agent_config.get("name", None)
                    model_temperature = agent_config.get("temperature", 0.0)
                    if model is not None:
                        get_model = ConfigGraph.MODEL.get(model, None)
                        if get_model is None:
                            logger.error(f"The Model defined for agent : {agent} isnt't available -> using deafult model")
                            get_model = get_nvdia
                            prompt = self.get_model_agent_prompt(model ='NVIDIA', agent = agent)
                        else:
                            prompt = self.get_model_agent_prompt(model = model, agent = agent)
                    else:
                        get_model = get_nvdia(temperature=model_temperature)

                    agents[agent] = Agent(
                        agent_name=agent,
                        model=model,
                        get_model=get_model,
                        temperature=model_temperature,
                        prompt=prompt,
                        parser=self.get_agent_parser(agent=agent)
                    )
                else:
                    pass
        logger.info(f"Graph Agents : {agents}")
        return agents

    def get_model_agent_prompt(self, model : str, agent : str) -> Union[PromptTemplate,str]:
        """Get specific parser for each graph agent"""
        if model == 'OPENAI':
            if agent == 'query_classificator':
                return query_classify_prompt_openai
            elif agent == "docs_grader":
                return grader_docs_prompt_openai
            elif agent == "query_processor":
                return query_process_prompt_openai        
            elif agent == "generator":
                return gen_prompt_openai      
            elif agent == "hallucination_grader":
                return hallucination_prompt_openai
            elif agent == "answer_grader":
                return grade_answer_prompt_openai
            else:
                logger.exception(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
                raise ConfigurationFileError(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")      
        elif model == 'GROQ':
            if agent == 'query_classificator':
                return query_classify_groq_prompt
            elif agent == "docs_grader":
                return grader_docs_groq_prompt
            elif agent == "query_processor":
                return query_process_groq_prompt
            elif agent == "generator":
                return generate_groq_prompt
            elif agent == "hallucination_grader":
                    return hall_groq_prompt
            elif agent == "answer_grader":
                    return grade_answer_groq_prompt
        elif model == 'NVIDIA' or  model =='OLLAMA':
            if agent == 'query_classificator':
                return query_classify_prompt
            elif agent == "docs_grader":
                return grader_docs_prompt
            elif agent == "query_processor":
                return query_process_prompt        
            elif agent == "generator":
                return gen_prompt      
            elif agent == "hallucination_grader":
                return hallucination_prompt
            elif agent == "answer_grader":
                return grade_answer_prompt
            else:
                logger.exception(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
                raise ConfigurationFileError(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
        else:
            logger.exception(f"Error inside confiuration graph file -> Model {model} not supported")
            raise ConfigurationFileError(f"Error inside confiuration graph file -> Model {model} not supported")
        
    def get_agent_parser(self , agent :str) -> BaseTransformOutputParser:
        """Get specific parser for each graph agent depending on the model provider"""
        
        if not ConfigGraph.AGENTS.get(agent, None):
            logger.exception(f"Error inside confiuration graph file -> Agent '{agent}' not supported")
            raise ConfigurationFileError(f"Error inside confiuration graph file -> Agent '{agent}' not supported")
        
        if agent == "GROQ":
            parser = StrOutputParser
        else:
            parser = StrOutputParser
            
        logger.info(f"Agent {agent} -> Parser {parser}")   
        return parser

   
    def get_vector_db(self) -> VectorDB:
        # vector_db = ConfigGraph.VECTOR_DB.copy()
        """ 
        vector_db = {}
        For several retrievers and vdbs
        for vdb_name, vdb_config in self.vector_db_config.items():
            try:
                vector_db[vdb_name] = VectorDB(**vdb_config, client=vdb_name)
                logger.info(f"Initializating new retriever -> {vector_db[vdb_name]}")
            except Exception as e:
                logger.error(f"Error while initializating new retriever -> {vector_db[vdb_name]}")
        """
        logger.info(f"Vector DB config file {self.vector_db_config}")
        try:
            vector_db  = VectorDB(**self.vector_db_config)
            logger.info(f"Initializating new retriever -> {vector_db}")
        except Exception as e:
            logger.error(f"Error while initializating new retriever -> {self.vector_db_config=}")
                
        if not vector_db:
            raise ConfigurationFileError(f"Error inside confiuration graph file -> No VectorDB defined or bad defined for retrieval") 
            
        return vector_db

    def get_user_question(
                        self, 
                        q : Union[str,None] = None, 
                        ground_truth : Union[str,None] = None ,  
                        date : Union[str,None] = None,
                        boe_id : Union[str,None] = None
                        ) -> Question:
        if q and date:
            return Question(id=get_id(), ground_truth = ground_truth, user_question=q, date=date, boe_id=boe_id)
        else:
            raise ConfigurationFileError(f"Error inside confiuration Query file -> Query and/or date data not provided") 
        