import os
import logging
import asyncio
from RAG_EVAL.testset import generate_testset, RagasEval
from RAG_EVAL.utils import setup_logging
from dotenv import load_dotenv

# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

    # set up the root logger configuration
    setup_logging()
    
    # Create Raptor data (make cluster summary, process and store in vector database)
    ragas_evaluation = RagasEval(
        hg_token=str(os.getenv('HG_API_KEY')),
        dataset_name="20240903180604"
        )
    ragas_evaluation.run(results_file_path="./data/rag evaluation/results/results.csv", get_visual_reports=True)
    
if __name__ == "__main__":
    main()