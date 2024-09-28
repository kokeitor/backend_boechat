import os
import logging
from typing import Union, Optional, Callable, ClassVar, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from datasets import Dataset 
from RAG_EVAL.utils import get_current_spanish_date_iso

# Logging configuration
logger = logging.getLogger(__name__)

class RagasDataset:
    """
    Example :
    data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on January 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The Super Bowl....season since 1966,','replacing the NFL...in February.'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
    }
    """
    def __init__(self, question : list[str], answer : list[str], contexts : list[list[str]], ground_truth : list[str]):
        self.question = question
        self.answer = answer
        self.contexts = contexts
        self.ground_truth = ground_truth
    
    def add_atributes(self, question : str, answer : str, contexts : list[str], ground_truth : str):
        self.question.append(question)
        self.answer.append(answer)
        self.contexts.append(contexts)
        self.ground_truth.append(ground_truth)
        
    def model_dump(self) -> dict:
        return {
                'question':self.question,
                'answer': self.answer,
                'contexts' : self.contexts,
                'ground_truth': self.ground_truth
                }
        
    def to_dataset(self) -> Dataset:
        self.dataset = Dataset.from_dict(self.model_dump()) 
        return self.dataset
    
    def push_to_hub(
                    self, 
                    repo_id : str , 
                    hg_api_token : str , 
                    private: Optional[bool] = False, 
                    branch: Optional[str] = None
                    ) -> None:
        """
        Pushes the dataset to the Hugging Face Hub.

        Parameters:
        -----------
        private : Optional[bool]
            Whether the repository should be private.
        token : Optional[str]
            The authentication token for the Hugging Face Hub.
        branch : Optional[str]
            The git branch to push the dataset to.
        """

        if self.dataset:
            self.dataset.push_to_hub(
                repo_id=repo_id,
                config_name=get_current_spanish_date_iso(),
                commit_message=f"Date of push: {get_current_spanish_date_iso()}",
                private=private,
                token=hg_api_token)
        else:
            logger.error(f"No hugging face dataset created ->call to_dataset() method first")