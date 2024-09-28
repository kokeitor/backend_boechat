import os
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import get_current_spanish_date_iso
from ETL.llm import LabelGenerator
import logging
import logging.handlers
import tiktoken
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Union, Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import DataFrameLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Logging configuration
logger = logging.getLogger(__name__)


class ClusterSummaryGenerator:
    def __init__(self, model: str = 'GPT-4O-MINI'):
        self.model_label = model
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5")

        self.prompt = PromptTemplate(
            template="""You are an assistant specialized in summarizing texts from the Spanish Boletín Oficial del Estado (BOE).\n
            Create a precise and factual summary in Spanish, strictly reflecting the original content without adding \n
            any comments or interpretations.\n
            Text: {text}""",
            input_variables=["text"],
            input_types={"text": str}
        )
        self.groq_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an assistant specialized in summarizing texts from the Spanish Boletín Oficial del Estado (BOE).\n
            Create a precise and factual summary in Spanish, strictly reflecting the original content without adding \n
            any comments or interpretations.""",
                ),
                ("human", "{text}"),
            ]
        )
        models = {
            'GPT_35TURBO': ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
            'GROQ-LLAMA3': ChatGroq(
                model="llama3-70b-8192",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=10
            ),
            'GPT-4O-MINI': ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
        }

        self.model = models.get(self.model_label, None)
        if not self.model:
            logger.error("Model ClusterSummaryGenerator Name not correct")
            raise AttributeError(
                "Model ClusterSummaryGenerator Name not correct")

        elif self.model_label == "GROQ-LLAMA3":
            self.chain = self.groq_prompt | self.model | StrOutputParser()
        else:
            self.chain = self.prompt | self.model | StrOutputParser()

    def _get_tokens(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.exception(f"Tokenization error: {e}")
            return len(self.tokenizer(text)["input_ids"])

    def invoke(self, cluster_text: str) -> list[Document]:

        cluster_tokens = self._get_tokens(text=cluster_text)

        # Update metadata
        logger.info(f'numero tokens del cluster text : {cluster_tokens}')
        logger.info(
            f'numero caracteres del cluster text : {len(cluster_text)}')

        try:
            cluster_summary = self.chain.invoke({"text": cluster_text})
            logger.debug(f"LLM output: {cluster_summary}")
        except Exception as e:
            logger.exception(
                f"LLM Error generation cluster summary of {cluster_text} , \nerror message: {e}")
            cluster_summary = "Error in generation of the cluster summary"

        return cluster_summary


class RaptorDataset(BaseModel):

    data_dir_path: str = Field(default="./")
    file_name: Optional[str] = Field(default=None)
    from_date:  Optional[str] = Field(default=None)
    to_date:  Optional[str] = Field(default=None)
    desire_columns: Optional[List[str]] = Field(default=None)
    data: Optional[pd.DataFrame] = None
    documents: Optional[list[Document]] = None

    # Removing the problematic initialization from the constructor
    cluster_summary_generator: Optional['ClusterSummaryGenerator'] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.cluster_summary_generator = ClusterSummaryGenerator()  # Initialize separately

    def initialize_data(self):
        """Initializes the data attribute by cleaning and combining data from files."""
        self.data = self._clean_data(self._get_data())
        self.data["label_str"] = ""  # empty column to fill it with label str
        self._put_metadata()
        logger.debug(f"Dataset RAPTOR sample:\n{self.data.head(1)}")
        # empty column to fill it with cluster summary
        self.data["cluster_summary"] = ""
        self._get_cluster_summary()
        logger.debug(f"Dataset RAPTOR columns:\n{self.data.columns.to_list()}")
        logger.info(
            f"Informacion de data :\n{self.data.shape} \n{self.data.head()}\n{self.data.columns}")
        self._get_documents()
        logger.info(f"Number of Document objects : {len(self.documents)}")
        try:
            logger.info(f"Document samples :\n {self.documents[2]}")
        except:
            logger.info(f"Document samples :\n {self.documents}")

    def _get_data(self) -> pd.DataFrame:
        """
        Reads and combines data from .CSV and .parquet files within the specified date range.

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame from all the files.
        """
        if not os.path.isdir(self.data_dir_path):
            raise DirectoryNotFoundError(
                f"The specified directory '{self.data_dir_path}' does not exist.")
        dataframes = []
        logger.warning(
            f"os.listdir(self.data_dir_path) : {os.listdir(self.data_dir_path)}")
        for filename in os.listdir(self.data_dir_path):
            logger.warning(f"filename : {filename}")
            if self.file_name:
                if filename.split("_")[1] == self.file_name:
                    file_path = os.path.join(self.data_dir_path, filename)
                    logger.warning(f"file_path : {file_path}")
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        logger.warning(f"Reading CSV file : {file_path}")
                        dataframes.append(df)
                    elif filename.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        logger.warning(f"Reading parquet file : {file_path}")
                        dataframes.append(df)
            else:
                if "_" in filename and filename[0].isdigit():
                    try:
                        file_date = datetime.strptime(
                            filename.split("_")[0], '%Y%m%d%H%M%S')
                    except ValueError as e:
                        logger.error(f"Error parsing the date: {e}")
                        continue

                    logger.info(
                        f"file_date parsed to correct format: {file_date}")

                    if self.parse_date(self.from_date) <= file_date <= self.parse_date(self.to_date):
                        logger.info(
                            f"File name date {file_date} between : {self.parse_date(self.from_date)} and {self.parse_date(self.to_date)}")
                        logger.info(f"Trying to append it")
                        file_path = os.path.join(self.data_dir_path, filename)
                        if filename.endswith('.csv'):
                            df = pd.read_csv(file_path)
                            logger.info(f"Reading CSV file : {file_path}")
                            dataframes.append(df)
                        elif filename.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                            logger.info(f"Reading parquet file : {file_path}")
                            dataframes.append(df)
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        return combined_df

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data by keeping only the desired columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to be cleaned.

        Returns:
        --------
        pd.DataFrame
            The cleaned DataFrame.
        """
        columns_to_keep = []
        if self.desire_columns:
            logger.debug(f"Data columns : {data.columns.to_list()}")
            for col in self.desire_columns:
                if col in data.columns.to_list():
                    columns_to_keep.append(col)
                    logger.debug(
                        f"Data column to keep {col} exists in file columns")
                else:
                    logger.warning(
                        f"Data column to keep {col} NOT IN file columns")
            return data[columns_to_keep]
        else:
            return data

    @ staticmethod
    def parse_date(date_str: str) -> datetime:
        """
        Parses a date string into a datetime object.

        Parameters:
        -----------
        date_str : str
            The date string to be parsed.

        Returns:
        --------
        datetime
            The parsed datetime object.
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj
        except ValueError as e:
            raise ValueError(f"Error parsing the date: {e}")

    def _put_metadata(self) -> None:
        """
        Processes the label metadata for each row in the DataFrame.

        This method cleans and splits the labels from `LabelGenerator.LABELS`,
        creates mapping dictionaries (`label2id` and `id2label`), and iterates
        over each row in the `self.data` DataFrame. For each row, it maps label IDs
        to label names and adds them as a new column `label_str` to the DataFrame.

        If a row has a label, the label is processed and converted to a string.
        If no label is present or if the label is NaN, the `label_str` column is
        set to "NotExist".

        Logs detailed information about the processing at various steps.
        """
        logger.info(f"Initial labels:\n {LabelGenerator.LABELS}")

        # Clean and split the labels
        labels = LabelGenerator.LABELS.replace("\n", "").split(',')
        labels = [label.strip() for label in labels]
        logger.info(f"Processed labels:\n {labels}")

        # Create mapping dictionaries
        label2id = {label: label_index for label_index,
                    label in enumerate(labels)}
        id2label = {label_index: label for label_index,
                    label in enumerate(labels)}
        logger.info(f"label2id mapping:\n {label2id}")
        logger.info(f"id2label mapping:\n {id2label}")

        # Process each row in the DataFrame
        for index, row in self.data.iterrows():
            if pd.notna(row["label"]):  # Check for NaN in the label column
                logger.debug(f"Processing row index: {index}")
                logger.debug(f"Row columns: {row.keys()}")
                logger.debug(f"Labels of row {index}: {row['label']}")
                logger.debug(f"Type of object label: {type(row['label'])}")

                # Map label ids to label names and add them as a new column to the DataFrame
                labels_id_int = self._parse_label_id_str(row["label"])
                label_columns = [id2label.get(id, "NotExist")
                                 for id in labels_id_int]
                logger.debug(f"Mapped label columns: {label_columns}")

                # Add the label string to the DataFrame
                self.data.at[index, "label_str"] = str(
                    label_columns[0]) if label_columns else "NotExist"
                logger.debug(
                    f"Updated self.data.loc[index,'label_str']: {self.data.at[index, 'label_str']}")
            else:
                logger.debug(f"Processing row index: {index}")
                logger.debug(f"Row columns: {row.keys()}")
                logger.debug(f"Labels of row {index}: {row['label']}")
                logger.debug(f"Type of object label: {type(row['label'])}")

                # Add the label string to the DataFrame
                self.data.at[index, "label_str"] = "NotExist"
                logger.debug(
                    f"Updated self.data.loc[index,'label_str']: {self.data.at[index, 'label_str']}")

    def _parse_label_id_str(self, input_str: str) -> list[int]:
        """
        Parses a string of label IDs into a list of integers.

        The input string is expected to be in the format "['12', '26', '24']".
        This method removes the brackets and quotes, splits the string by commas,
        and converts each segment into an integer.

        Parameters:
        ----------
        input_str : str
            A string representing a list of label IDs.

        Returns:
        -------
        list[int]
            A list of integers parsed from the input string.
        """
        str_list = input_str.strip("[]").replace("'", "").split(", ")

        int_list = [int(x) for x in str_list]

        logger.debug(f"label id input : {input_str}")
        logger.debug(f"labels parsed : {int_list}")

        return int_list

    def _get_cluster_summary(self) -> None:
        """
        Generates and stores a summary for each unique label in the DataFrame.

        This method iterates over each unique label in the `label_str` column,
        aggregates the text associated with each label up to a specified
        maximum length (`MAX_LEN`), and generates a summary using the
        `cluster_summary_generator`. The summary is then stored in the
        `cluster_summary` column of the DataFrame.

        Logs detailed information during the processing and summary generation.
        """
        unique_labels = self.data["label_str"].unique()
        logger.debug(f"unique labels:\n{unique_labels}")

        MAX_LEN = 300  # Maximum characters for creating the cluster summary
        for unique_label in unique_labels:
            filter_dataframe = self.data[self.data["label_str"]
                                         == unique_label]
            cluster_text = ""
            for text in filter_dataframe["text"]:
                if len(cluster_text) < MAX_LEN:
                    cluster_text = cluster_text + "\n" + str(text)
                else:
                    break
            logger.debug(f"cluster_text for {unique_label=} :\n{cluster_text}")
            summary = self.cluster_summary_generator.invoke(
                cluster_text=cluster_text)
            logger.info(f"cluster summary for {unique_label=} :\n{summary}")
            self.data.loc[self.data["label_str"] ==
                          unique_label, "cluster_summary"] = summary

    def _get_documents(self) -> None:
        """
        Loads documents from the DataFrame and stores them in the `documents` attribute.

        This method utilizes the `DataFrameLoader` to load documents from the
        `self.data` DataFrame, with the text content being extracted from the
        `page_content_column`. The loaded documents are stored in the `documents`
        attribute of the class.

        Logs the loading process for debugging purposes.
        """
        df_loader = DataFrameLoader(
            data_frame=self.data, page_content_column="text")
        self.documents = df_loader.load()
