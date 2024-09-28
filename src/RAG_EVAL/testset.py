import os
import logging
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from ragas.testset.generator import TestsetGenerator, TestDataset
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from RAPTOR.exceptions import DirectoryNotFoundError
from ragas.run_config import RunConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from RAG_EVAL.utils import get_current_spanish_date_iso
from datasets import Dataset, load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import matplotlib.pyplot as plt
import seaborn as sns


# Logging configuration
logger = logging.getLogger(__name__)

class RagasEval:
    
    def __init__(self, hg_token : str, dataset_name : str):
        
        load_dotenv()
        try:
            self.testset = load_dataset(
                path=str(os.getenv("HG_REPO_RAGAS_TESTSET_ID")),
                name=dataset_name,
                data_dir=dataset_name ,
                data_files = None,
                split = None,
                cache_dir= None,
                features= None,
                download_config= None,
                download_mode= None,
                verification_mode= None,
                ignore_verifications= "deprecated",
                keep_in_memory= None,
                save_infos= False,
                revision= None,
                token= hg_token,
                use_auth_token= "deprecated",
                task= "deprecated",
                streaming= False,
                num_proc= None,
                storage_options= None,
                trust_remote_code= None
                )
            logger.info(f"RAGAS hg testset {self.testset}")
        except Exception as e:
            logger.error(f"Error while pulling HG RAGAS testset {e}")
            
    def run(self, results_file_path : str, get_visual_reports : bool = False):
        if self.testset:
            result = evaluate(
                                self.testset["train"],
                                metrics=[
                                    context_precision,
                                    faithfulness,
                                    answer_relevancy,
                                    context_recall,
                                ],
                            )
            try:
                self.results_df = result.to_pandas()
                logger.info(f"Ragas Eval result dataframe :\n{self.results_df.head()}\n{self.results_df.columns=}\n{self.results_df.shape=}")
                
                # Process results df : add question id and make metrics list for generaating reports and figures
                self._save_df(file_path=results_file_path)
                
                if get_visual_reports:
                    # Scatter plots of the metrics for each question id
                    RagasEval.get_scatter_plot(
                        title="Context precision", 
                        df=self.results_df, 
                        column="context_precision",
                        directory=os.path.dirname(results_file_path)+"/metric reports",
                        file_name="context_precision.png"
                        )
                    RagasEval.get_scatter_plot(
                        title="Faithfulness", 
                        df=self.results_df, 
                        column="faithfulness",
                        directory=os.path.dirname(results_file_path)+"/metric reports",
                        file_name="faithfulness.png"
                        )
                    RagasEval.get_scatter_plot(
                        title="Answer relevancy", 
                        df=self.results_df, 
                        column="answer_relevancy",
                        directory=os.path.dirname(results_file_path)+"/metric reports",
                        file_name="answer_relevancy.png"
                        )
                    RagasEval.get_scatter_plot(
                        title="Context Recall", 
                        df=self.results_df, 
                        column="context_recall",
                        directory=os.path.dirname(results_file_path)+"/metric reports",
                        file_name="context_recall.png"
                        )
                    
                    # Generate df with only columns of metrics results
                    required_columns = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']
                    self.stats_df = self.results_df.select_dtypes('number')[required_columns].describe()
                    RagasEval.get_table_plot(
                                df=self.results_df[required_columns].round(decimals=2), 
                                title="RAGAS Test Set", 
                                directory=os.path.dirname(results_file_path)+"/metric reports",
                                file_name="ragas_testset.png"
                                )
                    
                    # Stats dataframe
                    self.stats_df = self.results_df.select_dtypes('number').describe()
                    RagasEval.get_table_plot(
                                df=self.stats_df.round(decimals=2) , 
                                title="RAGAS Metrics Statistics", 
                                directory=os.path.dirname(results_file_path)+"/metric reports",
                                file_name="metrics_statistics.png"
                                )
                    
                    logger.info("Getting means and distribution reports visual analysis ... ")
                    self.get_visual_report(df=self.results_df, output_file=results_file_path)
                    logger.info(f"Reports saved inside {os.path.dirname(results_file_path)+'/metric reports'}")
                
            except Exception as e:
                logger.exception(f"Error while cretaing result dataframe")
        else:
            logger.error(f"Error while running RagasEval run() method , no testset initialization")
    
    def _save_df(self, file_path : str):
        """Save a df inside a local directory in csv format
        Args:
            file_path (str): ...
        """
        # Get the directory path from the file path
        directory = os.path.dirname(file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directory created: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

        # Save the DataFrame to CSV
        self.results_df.to_csv(path_or_buf=file_path, index=False)
        logger.info(f"DataFrame saved to CSV at: {file_path}")
        
    
    def get_visual_report(self, df, output_file):
        """
        Creates a visual report with summary statistics of the provided dataframe metrics and saves it as an image.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the metrics columns.
            output_file (str): The file path where the report image will be saved.
        """
        # Path of the figures 
        directory = os.path.dirname(output_file)
        directory += "/metric reports"
        
        # Check if required columns exist in the dataframe
        required_columns = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"The dataframe must contain the following columns: {required_columns}")
        
        # Filter numeric dataframe 
        df_numeric = df.loc[:,required_columns]
        
        # Calculate summary statistics
        summary_stats = df_numeric.describe().T  # Transpose for better readability
        summary_stats['variance'] = df_numeric.var()
        summary_stats['range'] = df_numeric.max() - df_numeric.min()
        
        # Boxplot for distributions
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_numeric,color="#57d3f2")
        plt.title('Metrics Distribution')
        plt.savefig(directory + "/metrics_distributions.png")
        plt.close()

        # Bar plot for means
        plt.figure(figsize=(12, 8))
        sns.barplot(x=summary_stats.index, y='mean', color="#65f0b0", data=summary_stats)
        plt.title('Mean of Metrics')
        plt.ylabel('Mean')
        plt.xlabel('RAGAS metrics')
        plt.grid()
        plt.savefig(directory + "/metrics_means.png")
        plt.close()

    @staticmethod
    def get_scatter_plot(title : str , df : pd.DataFrame, column : str, directory : str,  file_name :str):
        plt.figure(figsize=(10, 8))
        plot = True
        try: 
            plt.plot(range(1,df.shape[0]+1), df[column], linewidth=0.8, markersize=5, color='#57d3f2', marker='o')
        except Exception as e:
            logger.error(f"{e}")
            plot = False
        if plot:
            plt.title(f'{title}', fontsize=16 )
            plt.grid(True)
            plt.xlabel('Query ID', fontsize=12)
            plt.ylabel(f"{title}", fontsize=12)
            plt.savefig(directory + "/" + file_name)
        
    @staticmethod
    def get_table_plot(df : pd.DataFrame , title : str, directory : str,  file_name :str):

        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        cell_text = df.values.tolist() # Convert summary statistics to a 2D list for the table
        tbl = plt.table(
            cellText=cell_text,
            colLabels=df.columns,
            rowLabels=df.index,
            cellLoc='center',
            colWidths=[0.2] * len(df.columns),
            colColours=['#65f0b0'] * len(df.columns),
            rowColours=['#57d3f2'] * df.shape[0],
            loc='center'
        )
        # Customize the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)  # Set font size
        tbl.scale(1.2, 1.2)  # Scale table size
        
        # Customize headers and body table separately
        for key, cell in tbl.get_celld().items():
            if key[0] == 0 or key[1] == -1:
                cell.set_text_props(weight='bold', color="black")
                cell.set_edgecolor(color='black')
            else:
                cell.set_color(c='#cdcdcb')
                cell.set_edgecolor(color='black')
            cell.set_linewidth(2)
        
        plt.title(f'{title}')
        plt.savefig(directory +"/"+file_name)
                    
                    
# Synthetic RAGAS testset generation : 
def parse_date(date_str: str) -> datetime:
    """
    Parses a date string into a datetime object.
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj
    except ValueError as e:
        raise ValueError(f"Error parsing the date: {e}")

def get_data(docs_path: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Reads and combines data from .CSV and .parquet files within the specified date range.
    """
    if not os.path.isdir(docs_path):
        raise DirectoryNotFoundError(f"The specified directory '{docs_path}' does not exist.")

    dataframes = []
    for filename in os.listdir(docs_path):
        logger.info(f"filename : {filename}")
        if "_" in filename and filename[0].isdigit():
            try:
                file_date = datetime.strptime(filename.split("_")[0], '%Y%m%d%H%M%S')
            except ValueError as e:
                logger.error(f"Error parsing the date: {e}")
                continue

            logger.info(f"file_date parsed to correct format: {file_date}")

            if parse_date(from_date) <= file_date <= parse_date(to_date):
                logger.info(f"File name date {file_date} between : {parse_date(from_date)} and {parse_date(to_date)}")
                logger.info(f"Trying to append it")
                file_path = os.path.join(docs_path, filename)
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    logger.info(f"Reading CSV file : {file_path}")
                    logger.info(f"Dataframe columns : {df.columns}")
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

def generate_testset(docs_path: str, from_date: str, to_date: str, save_path : str = "./data/rag evaluation/ragas testset") -> TestDataset:
    """
    Generates a test dataset from documents within a specified date range asynchronously.
    """
    generator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )
    critic_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )
    
    embedding_model = OpenAIEmbeddings()
    
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embedding_model
    )
    
    docs_df = get_data(docs_path=docs_path, from_date=from_date, to_date=to_date)  # dataframe with docs
    if docs_df.empty:
        logger.warning("No documents found in the specified date range.")
        return None

    df_loader = DataFrameLoader(data_frame=docs_df, page_content_column="text")
    docs = df_loader.load()  # list of docs
    try:
        logger.info(f"Number of docs to create RAG testset: {len(docs)}")
    except Exception as e:
        logger.error(f"Error logging document count: {e}")
    
    # Add filename key to metadata docs
    if docs:
        for d in docs:
            d.metadata['filename'] = d.metadata.get('pdf_id', 'unknown')
        logger.info(f"Metadata example: {docs[0].metadata}")
    
    # generate testset
    try:
        testset = generator.generate_with_langchain_docs(
            docs, 
            test_size=10, 
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
            run_config=RunConfig(max_workers=64)
        )

    except Exception as e:
        logger.error(f"Failed to generate testset: {e}")
        testset = None
        
    # translate question and answer 
    # initialize translator 
    translator = ChatGroq(
                            model= "llama3-70b-8192",
                            temperature=0.0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=10
                                )
    transalate_promt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a assistant tasked with accurately translating sentences from English to Spanish,\n 
                ensuring that the meaning, tone, and context of the original sentence are preserved."""
            ),
            ("human", "{sentence}"),
        ]
    )
    transalate_chain = transalate_promt | translator | StrOutputParser()
    
    # translate 
    if testset:
        testset_df = testset.to_pandas()
        print(testset_df.head())
        print(f"Columnas dataframe : {testset_df.columns}")
        logger.info(f"Columnas dataframe : {testset_df.columns}")
        logger.info(f"dataframe sample : {testset_df.head()}")
        logger.info(f"dataframe len : {testset_df.shape}")
        
        for index, row in enumerate(testset_df.iterrows()):
            try:
                translation = transalate_chain.invoke({"sentence":row["question"]})
                logger.info(f"llm translation of {row['question']}:\n{translation=}")  
                testset_df.loc[index,"question"] = translation
                testset_df.loc[index,"original_question"] = row["question"]
            except Exception as e:
                logger.error(f"llm translation error {e}")  
            
        
    # save testset as csv from pandas dataframe
    if testset:
        # Ensure the path exists; if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Directory {save_path} created.")

        # Define the full path where the file will be saved
        full_path = os.path.join(save_path, f"{get_current_spanish_date_iso()}_ragas_testset")

        # Save the DataFrame to a CSV file
        testset_df.to_csv(full_path, index=False)
        logger.info(f"DataFrame saved to {full_path}")
        
    return testset

def translate(sentence : str) -> str:
    translator = ChatGroq(
                            model= "llama3-70b-8192",
                            temperature=0.0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=10
                                )
    transalate_promt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a assistant tasked with accurately translating sentences from English to Spanish,\n 
                ensuring that the meaning, tone, and context of the original sentence are preserved."""
            ),
            ("human", "{sentence}"),
        ]
    )
    transalate_chain = transalate_promt | translator | StrOutputParser()
    translation = transalate_chain.invoke({"sentence":sentence})
    logger.info(f"llm translation of {sentence}:\n{translation=}")
    return translation