import logging
import os
import logging.config
import logging.handlers
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.graph import MermaidDrawMethod
from GRAPH_RAG.base_models import (
    State
)
from GRAPH_RAG.config import ConfigGraph
from GRAPH_RAG.graph_utils import get_current_spanish_date_iso_file_name_format
from GRAPH_RAG.chains import get_chain
from GRAPH_RAG.nodes import (
    query_classificator,
    retriever,
    retreived_docs_grader,
    route_generate_requery,
    process_query,
    generator,
    hallucination_checker,
    generation_grader,
    final_report,
    route_generate_gradegen_requery,
    route_generate_final
)


logger = logging.getLogger(__name__)


def create_graph(config : ConfigGraph) -> StateGraph:
    
    graph = StateGraph(State)
    
    query_classifier = config.agents.get("query_classificator",None)
    vector_db = config.vector_db
    docs_grader = config.agents.get("docs_grader",None)
    query_processor = config.agents.get("query_processor",None)
    generator_agent = config.agents.get("generator",None)
    hallucination_grader = config.agents.get("hallucination_grader",None)
    answer_grader = config.agents.get("answer_grader",None)


    # Define the nodes
    graph.add_node("query_classificator",lambda state: query_classificator(state=state,agent=query_classifier, get_chain=get_chain)) 
    graph.add_node("retriever",lambda state: retriever(state=state,vector_database=vector_db)) 
    graph.add_node("retreived_docs_grader",lambda state: retreived_docs_grader(state=state,agent=docs_grader, get_chain=get_chain))
    graph.add_node("reprocess_query",lambda state: process_query(state=state,agent=query_processor, get_chain=get_chain))
    graph.add_node("generator",lambda state: generator(state=state,agent=generator_agent, get_chain=get_chain))
    graph.add_node("hallucination_checker",lambda state: hallucination_checker(state=state,agent=hallucination_grader, get_chain=get_chain))
    graph.add_node("generation_grader",lambda state: generation_grader(state=state,agent=answer_grader, get_chain=get_chain))
    graph.add_node("final_report",lambda state: final_report(state=state))

    # Add edges to the graph
    graph.set_entry_point("query_classificator")
    graph.add_edge("query_classificator", "retriever")
    graph.set_finish_point("final_report")
    graph.add_edge("retriever", "retreived_docs_grader")
    graph.add_conditional_edges( 
                                source="retreived_docs_grader",
                                path=route_generate_requery,
                                path_map={
                                    "generator":"generator",
                                    "reprocess_query":"reprocess_query",
                                }
                                )
    graph.add_edge("reprocess_query", "query_classificator")
    graph.add_edge( "generator","hallucination_checker")
    graph.add_conditional_edges(
                                source="hallucination_checker",
                                path=route_generate_gradegen_requery,
                                path_map={
                                    "generator":"generator",
                                    "generation_grader":"generation_grader",
                                    "reprocess_query":"reprocess_query",
                                }
                                )
    graph.add_conditional_edges(
                                source="generation_grader",
                                path=route_generate_final,
                                path_map={
                                    "generator":"generator",
                                    "final_report":"final_report",
                                }
                                )
    graph.add_edge("final_report",END)
    
    return graph


def compile_graph(graph : StateGraph) -> CompiledGraph:
    checkpointer = MemorySaver()
    workflow = graph.compile(checkpointer=checkpointer)
    return workflow


def save_graph(compile_graph) -> None:
    """Save graph as png in the default figure graph directory"""
    
    figure_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "figures", "graphs", f'{get_current_spanish_date_iso_file_name_format()}_graph.png') 
    
    logger.debug(f"Attempting to save compiled graph -> {figure_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    
    # Try saving the file
    try:
        with open(figure_path, 'wb') as f:
            f.write(compile_graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API))
    except Exception as e:
        logger.error(f"File cannot be saved : {figure_path} -> {e}")
        
    # Debugging: Check if file is created
    if os.path.exists(figure_path):
        logger.info(f"Compiled graph png succesfully saved -> {figure_path}")
    else:
        logger.error(f"Failed to save the compile graph png-> {figure_path}")

        
        
