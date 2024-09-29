import logging
import os
from termcolor import colored
from typing import Dict, List, Tuple, Union, Optional, Callable
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from GRAPH_RAG.base_models import (
    State,
    VectorDB,
    Agent
)
from GRAPH_RAG.chains import get_chain
from ETL.llm import LabelGenerator
from GRAPH_RAG.graph_utils import get_current_spanish_date_iso, merge_page_content
import re

# Logging configuration
logger = logging.getLogger(__name__)


# Nodes

def query_classificator(state: State, agent: Agent, get_chain: Callable = get_chain) -> dict:
    """Classify the input query using BOE labels"""

    logger.info(f"Query Classificator node : \n {state}")
    print(colored(
        f"\n{agent.agent_name=} 👩🏽 -> {agent.model=} : ", 'light_red', attrs=["bold"]))

    question = state["question"][-1]
    _labels = LabelGenerator.LABELS.replace("\n", "").split(',')
    labels = [l.strip() for l in _labels]

    # LLM calling
    classify_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt,
                               temperature=agent.temperature, parser=agent.parser)
    generation = classify_chain.invoke({"text": question, "labels": labels})

    if agent.model == "GROQ":
        answer = generation
    else:
        answer = generation["query_label"]

    logger.info(f"Query : {question}")
    logger.info(f"Query label: {answer}")
    logger.info(f"Full Response :{generation}")

    print(colored(
        f"\nQuestion -> {state['question'][-1]}\nResponse -> {generation}\n", 'light_red', attrs=["bold"]))

    return {"query_label": answer}


def retriever(vector_database: VectorDB, state: State) -> dict:
    """Retrieve documents from vector database"""

    logger.info(f"Retriever node : \n {state}")
    print(colored(f"\nRetriever node 👩🏿‍💻 ", 'light_blue', attrs=["bold"]))

    logger.info(
        f"Using client for retrieval : {vector_database.vectorstore}")

    question = state["question"][-1]
    query_label = state["query_label"]
    print(
        colored(f"\nAll questions:\n{state['question']}", 'light_blue', attrs=["bold"]))
    print(colored(
        f"\nLast question:{state['question'][-1]}", 'light_blue', attrs=["bold"]))

    if query_label == 'Otra':
        print(colored(f"\nInvoking db retriever without metadata filter",
              'light_blue', attrs=["bold"]))
        documents = vector_database.vectorstore.similarity_search(
            query=question,
            k=3,
            namespace=os.getenv('PINECONE_INDEX_NAMESPACE')
        )
    else:
        print(colored(
            f"\nInvoking db retriever with metadata filter : {query_label}", 'light_blue', attrs=["bold"]))
        documents = vector_database.vectorstore.similarity_search(
            query=question,
            k=3,
            filter={"label_str": query_label},
            namespace=os.getenv('PINECONE_INDEX_NAMESPACE')
        )
    logger.info(f"Number of retrieved docs : {len(documents)}")
    logger.debug(f"Retrieved documents : \n {documents}")

    print(colored(
        f"Date = {state['date']}\nQuestion = {state['question']}\nQuery label = {state['query_label']}\nNumber of retrieved docs =  {len(documents)}", 'light_blue', attrs=["bold"]))

    return {"documents": documents}


def retreived_docs_grader(state: State, agent: Agent, get_chain: Callable = get_chain) -> dict:
    """Determines whether the retrieved documents are relevant to the question"""

    logger.info(f"Retrieved Documents Grader Node : \n {state}")
    print(colored(f"\n{agent.agent_name=} 👩🏿 -> {agent.model=}",
          'magenta', attrs=["bold"]))

    question = state["question"][-1]
    documents = state["documents"]

    # Grader chain
    grader_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt,
                             temperature=agent.temperature, parser=agent.parser)

    # Score each doc
    relevant_docs = []
    MAX_ITER = 5
    if len(documents) > 0:
        for index_doc, d in enumerate(documents):
            i = 0
            fail_llm = True
            content = d.page_content
            metadata = d.metadata
            chunk_label = metadata["label_str"]
            cluster_summary = metadata["cluster_summary"]

            logger.info(f"Document content : \n {content}")

            while i <= MAX_ITER and fail_llm == True:
                i += 1
                score = grader_chain.invoke(
                    {"question": question, "document": content})

                if agent.model == "GROQ":
                    grade = score
                else:
                    grade = score['score']

                print(colored(
                    f"\nScored doc {index_doc} -- {score=}\n{content=}\n{chunk_label=}\n{cluster_summary=}", 'magenta', attrs=["bold"]))

                # Document relevant
                if grade.lower() == "yes":
                    logger.info(f"GRADE: DOCUMENT {index_doc} as RELEVANT")
                    relevant_docs.append(d)
                    fail_llm = False
                # Document not relevant and retry with cluster summary
                elif grade.lower() == "no" and i < 2:
                    logger.warning(
                        f"GRADE: DOCUMENT {index_doc} as NOT RELEVANT TRYING WITH -> content = cluster_summary")
                    print(colored(
                        f"\nTRYING WITH -> content = cluster_summary", 'magenta', attrs=["bold"]))
                    content = cluster_summary
                elif grade.lower() == "no" and i >= 2:
                    logger.warning(
                        f"GRADE: DOCUMENT {index_doc} as NOT RELEVANT")
                    fail_llm = False
                # LLM output error
                else:
                    print(colored(
                        f"\nERROR LLM OUTPUT IN SCORING RETRIEVE DOC -> Retry chain invoke ...", 'magenta', attrs=["bold"]))

        # if only 0 or 1 doc relevant -> query processing necesary [no enough retrieved relevant context to answer]
        if len(relevant_docs) == 0:
            return {"documents": None, "query_reprocess": 'yes'}

        else:
            return {"documents": relevant_docs, "query_reprocess": 'no'}

    elif len(documents) == 0:
        print(colored(f"Documents retrieved == 0 -> query reprocess neccesary",
              'magenta', attrs=["bold"]))
        return {"documents": None, "query_reprocess": 'yes'}


def generator(state: State, agent: Agent, get_chain: Callable = get_chain) -> dict:
    """Generate answer using RAG on retrieved documents"""

    logger.info(f"RAG Generator node : \n {state}")
    print(colored(f"\n{agent.agent_name=} 👩🏽 -> {agent.model=}",
          'light_red', attrs=["bold"]))

    question = state["question"][-1]

    # Get the merge context from retrieved docs
    documents = state["documents"]
    # Merge docs page_content into unique str for the model context
    context = merge_page_content(docs=documents)

    # RAG generation
    rag_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt,
                          temperature=agent.temperature, parser=agent.parser)
    generation = rag_chain.invoke({"context": context, "question": question})

    if agent.model == "GROQ":
        answer = generation
    else:
        answer = generation["answer"]

    logger.info(f"RAG Context : \n {context}")
    logger.info(f"RAG Question : \n {question}")
    logger.info(f"RAG Response : \n {generation}")

    print(colored(
        f"\nQuestion -> {state['question'][-1]}\nContext -> {context}\nResponse -> {generation}\n", 'light_red', attrs=["bold"]))

    if answer == "I you don't know":
        logger.warning(f"Bad generation due to retrieval : {answer=}")
        return {"generation": answer}

    return {"generation": answer}


def process_query(state: State, agent: Agent, get_chain: Callable = get_chain) -> dict:
    """Reprocess or process the user query to improve docs retrieval"""

    logger.info(f"Query Processing : \n {state}")
    print(
        colored(f"\n{agent.agent_name=} 📝 -> {agent.model=}", 'blue', attrs=["bold"]))

    question = state["question"][-1]
    chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt,
                      temperature=agent.temperature, parser=agent.parser)
    response = chain.invoke({"question": question})

    if agent.model == "GROQ":
        reprocesed_question = response
    else:
        reprocesed_question = response["reprocess_question"]

    logger.info(f"{question=} // after reprocessing question -> {response=}")
    print(colored(
        f"Initial question : {question=}\nAfter reprocessing question : {response=}", 'blue', attrs=["bold"]))

    return {"question": [reprocesed_question]}


def hallucination_checker(state: State, agent: Agent, get_chain: Callable = get_chain) -> dict:
    """Checks for hallucionation on the response or generation"""

    logger.info(f"hallucination_checker node : \n {state}")
    print(colored(
        f"\n{agent.agent_name=} 👩🏿 -> {agent.model=} : ", 'light_cyan', attrs=["bold"]))

    generation = state["generation"]
    documents = state["documents"]
    # Merge docs page_content into unique str for the model context
    context = merge_page_content(docs=documents)

    # Break graph and go to reprocess query if model says 'I dont know'
    expresion_regular = r"\b(don't|know)\b"  # localiza 'don't' o 'know'
    coincidencias = re.findall(expresion_regular, generation)

    if len(coincidencias) > 0:  # si devuelve una lista no vacia es que ha encontrado alguna de esas palabras
        logger.warning(
            f"Generator LLM says {generation}, founded {coincidencias=} -> need requery")
        return {"fact_based_answer": 'reprocess_query'}

    else:  # si devuelve una lista  vacia es que NO ha encontrado alguna de esas palabras
        hall_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt,
                               temperature=agent.temperature, parser=agent.parser)

        MAX_ITER = 4
        for _ in range(0, MAX_ITER):
            response = hall_chain.invoke(
                {"documents": context, "generation": generation})
            logger.info(f"hallucination grade : {response=}")
            print(colored(
                f"LLM response -> Answer supported by context -> {response}", 'light_cyan', attrs=["bold"]))

            if agent.model == "GROQ":
                fact_based_answer = response
            else:
                fact_based_answer = response["score"]

            if (fact_based_answer.lower() == "yes" or fact_based_answer.lower() == "no"):
                break
            else:
                print(colored(
                    f"\nERROR LLM OUTPUT IN HALLUCINATION GRADER:\n{fact_based_answer=}\n **Retry chain invoke ...**\n", 'light_cyan', attrs=["bold"]))

        return {"fact_based_answer": fact_based_answer}


def generation_grader(state: State, agent: Agent, get_chain: Callable = get_chain) -> dict:
    """Grades the generation/answer given a question"""

    logger.info(f"generation_grader node : \n {state}")
    print(colored(
        f"\n{agent.agent_name=} 👩🏿 -> {agent.model=} : ", 'light_cyan', attrs=["bold"]))

    generation = state["generation"]
    question = state["question"][-1]

    garder_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt,
                             temperature=agent.temperature, parser=agent.parser)

    MAX_ITER = 4
    for _ in range(0, MAX_ITER):
        response = garder_chain.invoke(
            {"question": question, "generation": generation})
        logger.info(f"Answer grade : {response=}")
        print(colored(
            f"LLM response -> Useful answer to resolve the question -> {response}", 'light_cyan', attrs=["bold"]))

        if agent.model == "GROQ":
            grade = response
        else:
            grade = response["score"]

        if (grade.lower() == "yes" or grade.lower() == "no"):
            break
        else:
            print(colored(
                f"\nERROR LLM OUTPUT IN GENERATION GRADER:\n{grade=}\n **Retry chain invoke ...**\n", 'light_cyan', attrs=["bold"]))

    return {"useful_answer": grade}


def final_report(state: State) -> dict:
    """Returns and writes the final response of the model in a custom format"""

    generation = state["generation"]
    questions = state["question"]
    documents = state["documents"]
    grade_answer = state["useful_answer"]
    grade_hall = state["fact_based_answer"]

    logger.info(f"Final model response : \n {state}")
    report = f"""\nFinal model report : \n\n**QUESTIONS**: {questions}\n\n**\n\n
    **RETRIEVED DOCS**\n{documents}\n\n**ANSWER**\n{generation}\n\n
    **CONTEXT BASED ANSWER GRADE** : {grade_hall}\n\n**ANSWER GRADE** : {grade_answer}"""
    print(colored(
        f"\nFinal model report 📝\n\n**QUESTIONS**: {questions}\n\n**\n\n**RETRIEVED DOCS**\n{documents}\n\n**ANSWER**\n{generation}\n\n**CONTEXT BASED ANSWER GRADE** : {grade_hall}\n\n**ANSWER GRADE** : {grade_answer}", 'light_yellow', attrs=["bold"]))

    return {"report": report}


# Conditional edge functions
def route_generate_requery(state: State) -> str:
    """Route to generation or to reprocess question """

    logger.info(f"Router Generation or Reprocess Query : \n {state}")

    if state["query_reprocess"] == "yes":
        logger.info("Routing to -> 'query_reprocess'")
        print(colored("\n\nRouting to -> query_reprocess\n\n",
              'light_green', attrs=["underline"]))
        return 'reprocess_query'
    if state["query_reprocess"] == "no":
        logger.info("Routing to -> 'generator'")
        print(colored("\n\nRouting to -> generator\n\n",
              'light_green', attrs=["underline"]))
        return 'generator'


def route_generate_gradegen_requery(state: State) -> str:
    """Route to generation or to grade the generation/answer"""

    logger.info(f"Router Generation or Grader Generation : \n {state}")

    if state["fact_based_answer"] == "yes":
        logger.info("Routing to -> 'Grader generation'")
        print(colored("\n\nRouting to -> Grader generation\n\n",
              'light_green', attrs=["underline"]))
        return 'generation_grader'
    if state["fact_based_answer"] == "no":
        logger.info("Routing to -> 'Generation'")
        print(colored("\n\nRouting to -> Generation\n\n",
              'light_green', attrs=["underline"]))
        return 'generator'
    if state["fact_based_answer"] == 'reprocess_query':
        logger.info("Routing to -> 'reprocess_query'")
        print(colored("\n\nRouting to -> reprocess_query\n\n",
              'light_green', attrs=["underline"]))
        return 'reprocess_query'


def route_generate_final(state: State) -> str:
    """Route to generation or to final report"""

    logger.info(f"Router Generation or Final report : \n {state}")

    if state["useful_answer"] == "yes":
        logger.info("Routing to -> 'Final Report'")
        print(colored("\n\nRouting to -> Final Report\n\n",
              'light_green', attrs=["underline"]))
        return 'final_report'
    if state["useful_answer"] == "no":
        logger.info("Routing to -> 'Generation'")
        print(colored("\n\nRouting to -> Generation\n\n",
              'light_green', attrs=["underline"]))
        return 'generator'
