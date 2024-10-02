from typing import Union, Optional, Callable, ClassVar, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, BaseTransformOutputParser
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
import operator
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from langchain.schema import Document


class Question(BaseModel):
    id: Union[str, None] = None
    user_question: str
    ground_truth: str
    date: str
    boe_id: str


class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM generation
        query_process: 'yes' 'no' -> reprocess or not the user query 
        documents: list of documents retrieved
        fact_based_answer : 'yes' 'no' -> LLM generation based on document retrieved (analog to hallucination : 'no' or 'yes')
        useful_answer : 'yes' 'no' -> LLM generation answer respond or not to the question 
        final_report 

    """
    date: str
    question: Annotated[list[str], operator.add]
    query_label: str
    generation: str
    documents: Union[list[Document], None] = None
    fact_based_answer: str
    useful_answer: int
    report: str


@dataclass()
class Agent:
    agent_name: str
    model: str
    get_model: Callable
    temperature: float
    prompt: PromptTemplate
    parser: BaseTransformOutputParser


class VectorDB:
    def __init__(self, vectorstore: VectorStore, retriever: VectorStoreRetriever):
        self.vectorstore = vectorstore
        self.retriever = retriever
