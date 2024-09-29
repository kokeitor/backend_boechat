import os
import json
import time
import logging
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from typing import Union, Optional, Callable, ClassVar
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, BaseOutputParser, BaseTransformOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores.kinetica import DistanceStrategy
from exceptions.exceptions import VectorDatabaseError
from pinecone import Pinecone, ServerlessSpec
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

    MODEL: ClassVar = {
        "GROQ": get_groq,
        "OPENAI": get_open_ai_json,
        "NVIDIA": get_nvdia,
        "OLLAMA": get_ollama
    }

    AGENTS: ClassVar = {
        "query_classificator": Agent(agent_name="query_classificator", model="OPENAI", get_model=get_open_ai_json, temperature=0.0, prompt=query_classify_prompt_openai, parser=JsonOutputParser),
        "docs_grader": Agent(agent_name="docs_grader", model="OPENAI", get_model=get_open_ai_json, temperature=0.0, prompt=grader_docs_prompt, parser=JsonOutputParser),
        "query_processor": Agent(agent_name="query_processor", model="OPENAI", get_model=get_open_ai_json, temperature=0.0, prompt=query_process_prompt, parser=JsonOutputParser),
        "generator": Agent(agent_name="generator", model="GROQ", get_model=get_groq, temperature=0.0, prompt=generate_groq_prompt, parser=StrOutputParser),
        "hallucination_grader": Agent(agent_name="hallucination_grader", model="GROQ", get_model=get_groq, temperature=0.0, prompt=hall_groq_prompt, parser=StrOutputParser),
        "answer_grader": Agent(agent_name="answer_grader", model="GROQ", get_model=get_groq, temperature=0.0, prompt=grade_answer_groq_prompt, parser=StrOutputParser),
    }

    config_path: Union[str, None] = None
    graph: Union[StateGraph, None] = None
    compile_graph: Union[CompiledGraph, None] = None

    def __post_init__(self):

        if self.config_path is None:
            logger.exception(
                "No se ha proporcionado ninguna configuración para la generación usando Agents")
            raise AttributeError(
                "No se ha proporcionado ninguna configuración para la generación usando Agents")

        if self.config_path is not None:
            self.config = self.get_config()
            logger.info(
                f"Definida configuracion mediante archivo JSON en {self.config_path}")

        # Graph Agents configuration
        self.agents_config = self.config.get("agents", None)
        if self.agents_config is not None:
            self.agents = self.get_agents()

        # Graph VDB/Retrievers configuration
        self.vector_db = self.get_vector_db()

        # Graph configuration
        self.iteraciones = self.config.get("iteraciones", 10)
        self.thread_id = self.config.get("thread_id", "4")
        self.verbose = self.config.get("verbose", 0)

    def get_config(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.exception(
                f"Archivo de configuración no encontrado en {self.config_path}")
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado en {self.config_path}")
        with open(self.config_path, encoding='utf-8') as file:
            config = json.load(file)
        return config

    def get_agents(self) -> dict[str, Agent]:
        agents = ConfigGraph.AGENTS.copy()
        for agent_graph in agents.keys():
            for agent, agent_config in self.agents_config.items():
                if agent_graph == agent:
                    model = agent_config.get("name", None)
                    model_temperature = agent_config.get("temperature", 0.0)
                    if model is not None:
                        get_model = ConfigGraph.MODEL.get(model, None)
                        if get_model is None:
                            logger.error(
                                f"The Model defined for agent : {agent} isnt't available -> using deafult model")
                            get_model = get_nvdia
                            prompt = self.get_model_agent_prompt(
                                model='NVIDIA', agent=agent)
                        else:
                            prompt = self.get_model_agent_prompt(
                                model=model, agent=agent)
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

    def get_model_agent_prompt(self, model: str, agent: str) -> Union[PromptTemplate, str]:
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
                logger.exception(
                    f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
                raise ConfigurationFileError(
                    f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
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
        elif model == 'NVIDIA' or model == 'OLLAMA':
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
                logger.exception(
                    f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
                raise ConfigurationFileError(
                    f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
        else:
            logger.exception(
                f"Error inside confiuration graph file -> Model {model} not supported")
            raise ConfigurationFileError(
                f"Error inside confiuration graph file -> Model {model} not supported")

    def get_agent_parser(self, agent: str) -> BaseTransformOutputParser:
        """Get specific parser for each graph agent depending on the model provider"""

        if not ConfigGraph.AGENTS.get(agent, None):
            logger.exception(
                f"Error inside confiuration graph file -> Agent '{agent}' not supported")
            raise ConfigurationFileError(
                f"Error inside confiuration graph file -> Agent '{agent}' not supported")

        if agent == "GROQ":
            parser = StrOutputParser
        else:
            parser = StrOutputParser

        logger.info(f"Agent {agent} -> Parser {parser}")
        return parser

    def get_vector_db(self) -> VectorDB:
        try:
            logger.info(
                f"Connecting to an existing index of PineCone DB cient")

            self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

            existing_indexes = [index_info["name"]
                                for index_info in self.pc.list_indexes()]
            logger.info(f"existing_indexes : {existing_indexes}")

            if os.getenv('PINECONE_INDEX_NAME') not in existing_indexes:
                logger.warning(
                    f"Creting Index : {os.getenv('PINECONE_INDEX_NAME')}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not self.pc.describe_index(os.getenv('PINECONE_INDEX_NAME')).status["ready"]:
                    logger.info(self.pc.describe_index(
                        os.getenv('PINECONE_INDEX_NAME')))
                    time.sleep(1)

            self.index = self.pc.Index(os.getenv('PINECONE_INDEX_NAME'))
            pinecone_vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=(os.getenv('EMBEDDING_MODEL')),
                text_key='page_content',
                distance_strategy=DistanceStrategy.COSINE
            )
            retriever = pinecone_vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            # search_type="mmr",
            # search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5}

            return VectorDB(vectorstore=pinecone_vectorstore, retriever=retriever)

        except Exception as e:
            logger.error(
                f"Error while connecting to PineCone DB from existing index : {self.index_name} -> {e}")
            raise VectorDatabaseError(
                message="Error while connecting to PineCone DB from existing index", exception=e)
