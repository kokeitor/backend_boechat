import os
import logging
from dotenv import load_dotenv
from GRAPH_RAG.graph import create_graph, compile_graph, save_graph
from GRAPH_RAG.config import ConfigGraph
from RAG_EVAL.base_models import RagasDataset
from langgraph.errors import InvalidUpdateError
from langchain_core.runnables.config import RunnableConfig
from GRAPH_RAG.graph_utils import (
    setup_logging,
    get_arg_parser
)


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:

    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')
    os.environ['PINECONE_INDEX_NAME'] = os.getenv('PINECONE_INDEX_NAME')
    os.environ['CHROMA_COLLECTION_NAME'] = os.getenv('CHROMA_COLLECTION_NAME')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['APP_MODE'] = os.getenv('APP_MODE')
    os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

    # Logger set up
    setup_logging()

    # With scripts parameters mode
    parser = get_arg_parser()
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    DATA_PATH = args.data_path
    MODE = args.mode

    # With ENV VARS
    if not CONFIG_PATH:
        CONFIG_PATH = os.path.join(os.path.dirname(
            __file__), '..', 'config/graph', 'graph.json')
    if not DATA_PATH:
        DATA_PATH = os.path.join(os.path.dirname(
            __file__), '..', 'config/graph', 'querys.json')
    if not MODE:
        MODE = os.getenv('APP_MODE')

    logger.info(f"{DATA_PATH=}")
    logger.info(f"{CONFIG_PATH=}")
    logger.info(f"{MODE=}")

    # Mode -> Langgraph Agents
    if MODE == 'graph':

        logger.info(f"Graph mode")
        logger.info(
            f"Getting Data and Graph configuration from {DATA_PATH=} and {CONFIG_PATH=} ")
        config_graph = ConfigGraph(
            config_path=CONFIG_PATH, data_path=DATA_PATH)

        logger.info("Creating graph and compiling workflow...")
        config_graph.graph = create_graph(
            config=config_graph)  # create state graph
        config_graph.compile_graph = compile_graph(config_graph.graph)
        save_graph(compile_graph=config_graph.compile_graph)
        logger.info("Graph and workflow created")

        # RunnableConfig
        runnable_config = RunnableConfig(
            recursion_limit=config_graph.iteraciones,
            configurable={"thread_id": config_graph.thread_id}
        )

        # itera por todos questions definidos
        logger.warning(
            f"Total user questions = {len(config_graph.user_questions)}")
        print(f"\nTotal user questions = {len(config_graph.user_questions)}")
        logger.warning(f"User questions:\n{(config_graph.user_questions)}")
        print(f"\nUser questions:\n{(config_graph.user_questions)}\n")

        questions = config_graph.user_questions

        for index, q in enumerate(questions):

            logger.warning(f"User Question number {index} : {q.user_question}")
            logger.warning(f"User id question: {q.id}")
            logger.warning(f"User boe date: {q.date}")
            logger.warning(f"User boe_id : {q.boe_id}")

            inputs = {
                "question": [f"{q.user_question}"],
                "date": q.date,
                "query_label":  None,
                "generation": None,
                "documents": None,
                "fact_based_answer": None,
                "useful_answer": None
            }
            """
            for event in config_graph.compile_graph.stream(input=inputs,config=runnable_config):
                for key , value in event.items():
                    logger.warning(f"Graph event {key} - {value}")
            """
            logger.warning(f"Invoking graph with inputs: {inputs}")
            try:
                state = config_graph.compile_graph.invoke(
                    input=inputs, config=runnable_config)
                logger.warning(f"Final state graph -> {state}")
            except InvalidUpdateError as e:
                logger.error(f"Error invoking graph -> {e}")

            # Creation of a Ragas testset evaluation
            if index == 0:
                logger.warning(f"Creating RagasDataset ... ")
                testset = RagasDataset(
                    question=[q.user_question],
                    answer=[state["generation"]],
                    contexts=[[doc.page_content for doc in state["documents"]]],
                    ground_truth=[q.ground_truth]
                )
            else:
                testset.add_atributes(
                    question=q.user_question,
                    answer=state["generation"],
                    contexts=[doc.page_content for doc in state["documents"]],
                    ground_truth=q.ground_truth
                )

        # Generate HG RAGAS testset
        hg_testset = testset.to_dataset()
        logger.warning(f"Ragas testset :\n{testset}")
        logger.warning(f"Ragas hugging face testset :\n{hg_testset}")

        # Push to hub the RAGAS testset
        testset.push_to_hub(
            hg_api_token=str(os.getenv('HG_API_KEY')),
            repo_id=str(os.getenv("HG_REPO_RAGAS_TESTSET_ID"))
        )


if __name__ == '__main__':
    main()
