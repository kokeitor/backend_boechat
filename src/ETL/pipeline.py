import os
import json
import uuid
import tiktoken
import pandas as pd
import logging
import logging.config
import logging.handlers
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2Tokenizer
from typing import Union, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import src.ETL.splitters
import src.ETL.parsers
import src.ETL.nlp
import warnings
from src.ETL.utils import get_current_spanish_date_iso
from src.ETL.llm import LabelGenerator


# Set the default font to DejaVu Sans
plt.rcParams['font.family'] = 'DejaVu Sans'


# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("ETL_module_logger")


class Storer:
    def __init__(self, store_path: str, file_name: str, file_format: str = 'csv'):
        self.store_path = store_path
        self.file_name = file_name
        self.file_format = file_format.lower()

    def _document_to_dataframe(self, docs: list[Document]) -> pd.DataFrame:
        records = []
        for doc in docs:
            record = {"text": doc.page_content}
            record.update(doc.metadata)
            records.append(record)
        return pd.DataFrame(records)

    def _get_id(self) -> str:
        return str(uuid.uuid4())

    def _store_dataframe(self, df: pd.DataFrame, path: str, file_format: str) -> None:
        logger.info(
            f"Number of classify docs to store inside {file_format=} -> {df.shape}")
        if file_format == "parquet":
            df.to_parquet(path, index=False)
        elif file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "feather":
            df.to_feather(path)
        else:
            logger.exception(
                f"ValueError : Unsupported file format: {file_format}")
            raise ValueError(f"Unsupported file format: {file_format}")

    def invoke(self, docs: list[Document]) -> None:

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        name_format = str(get_current_spanish_date_iso()) + \
            "_" + self.file_name + '.' + self.file_format

        df = self._document_to_dataframe(docs)
        full_path = os.path.join(self.store_path, name_format)
        self._store_dataframe(df, full_path, self.file_format)


class Pipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._parse_config()
        self.parser = self._create_parser()
        self.splitter = self._create_splitter()
        self.label_generator = self._create_label_generator()
        self.storer = self._create_storer()

    def _parse_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}")
        with open(self.config_path, encoding='utf-8') as file:
            config = json.load(file)
        return config

    def _create_parser(self) -> src.ETL.parsers.Parser:
        parser_config = self.config.get('parser', {})
        return src.ETL.parsers.Parser(
            directory_path=os.path.abspath(parser_config.get(
                'directory_path', './data/boe/dias/')),
            file_type=parser_config.get('file_type', '.pdf'),
            recursive_parser=parser_config.get('recursive_parser', True),
            result_type=parser_config.get('result_type', 'markdown'),
            verbose=parser_config.get('verbose', True),
            api_key=parser_config.get(
                'api_key', os.getenv('LLAMA_CLOUD_API_KEY_RAPTOR'))
        )

    def _create_processor(self, docs: list[Document]) -> src.ETL.nlp.BoeProcessor:
        txt_process_config = self.config.get('TextPreprocess', None)
        if txt_process_config is not None:
            spc_words = txt_process_config.get('spc_words', None)
            special_char = txt_process_config.get('spc_caracters', None)
            preprocess_task = txt_process_config.get('task_name', "Default")
            processor = src.ETL.nlp.BoeProcessor(
                task=preprocess_task, docs=docs, spc_caracters=special_char, spc_words=spc_words)

            txt_process_methods = txt_process_config.get('methods', None)
            logger.info(
                f"Configuration of TextPreprocess for task : {preprocess_task} found")

            for method_key, method_vals in txt_process_methods.items():
                if method_vals.get("apply", False):
                    logger.info(
                        f"Trying to preprocess texts --> {method_key} : {method_vals}")
                    if method_key == 'del_stopwords':
                        processor = processor.del_stopwords(
                            lang=method_vals.get("lang", "Spanish"))
                    elif method_key == 'del_urls':
                        processor = processor.del_urls()
                    elif method_key == 'del_html':
                        processor = processor.del_html()
                    elif method_key == 'del_emojis':
                        processor = processor.del_emojis()
                    elif method_key == 'del_special':
                        processor = processor.del_special()
                    elif method_key == 'del_digits':
                        processor = processor.del_digits()
                    elif method_key == 'del_special_words':
                        processor = processor.del_special_words()
                    elif method_key == 'del_chinese_japanese':
                        processor = processor.del_chinese_japanese()
                    elif method_key == 'del_extra_spaces':
                        processor = processor.del_extra_spaces()
                    elif method_key == 'get_lower':
                        processor = processor.get_lower()
                    elif method_key == 'get_alfanumeric':
                        processor = processor.get_alfanumeric()
                    elif method_key == 'stem':
                        processor = processor.stem()
                    elif method_key == 'lemmatizer':
                        processor = processor.lemmatize()
                    elif method_key == 'custom_del':
                        path = os.path.abspath(method_vals.get(
                            "storage_path", "./data/figures/text/process"))
                        abs_path_name = os.path.join(
                            path, f"{get_current_spanish_date_iso()}.png")
                        logger.info(
                            f"Path to save plot 'custom_del' : {abs_path_name}")
                        _, _ = processor.custom_del(
                            text_field_name="text",
                            data=self.get_dataframe(docs=docs),
                            delete=method_vals.get("delete", False),
                            plot=method_vals.get("plot", True),
                            storage_path=abs_path_name
                        )
                    elif method_key == 'bow':
                        path = os.path.abspath(method_vals.get(
                            "storage_path", "./data/figures/text/bow"))
                        abs_path_name = os.path.join(
                            path, f"{get_current_spanish_date_iso()}.png")
                        logger.info(
                            f"Path to save plot 'bow' : {abs_path_name}")
                        self.save_figure_from_df(
                            df=processor.bow(),
                            path=abs_path_name,
                            method='BOW'
                        )
                    elif method_key == 'bow_tf_idf':
                        path = os.path.abspath(method_vals.get(
                            "storage_path", "./data/figures/text/bow"))
                        abs_path_name = os.path.join(
                            path, f"{get_current_spanish_date_iso()}.png")
                        logger.info(
                            f"Path to save plot 'bow_tf_idf' : {abs_path_name}")
                        self.save_figure_from_df(
                            df=processor.bow_tf_idf(),
                            path=abs_path_name,
                            method='BOW-TF-IDF'
                        )
                    else:
                        logger.warning(
                            f"Method {method_key} not found for TextPreprocess class")
        else:
            logger.warning(
                "Configuration of TextPreprocess not found, applying default one")
            preprocess_task = "Default process config"
            processor = src.ETL.nlp.BoeProcessor(
                task=preprocess_task, docs=docs)

        return processor

    def _create_splitter(self) -> src.ETL.splitters.Splitter:
        splitter_config = self.config.get('splitter', {})
        return src.ETL.splitters.Splitter(
            chunk_size=splitter_config.get('chunk_size', 200),
            embedding_model=self._get_embd_model(embd_model=splitter_config.get(
                'embedding_model', str(os.getenv('EMBEDDING_MODEL')))),
            tokenizer_model=self._get_tokenizer(
                tokenizer_model=splitter_config.get('tokenizer_model', 'LLAMA3')),
            threshold=splitter_config.get('threshold', 75),
            max_tokens=splitter_config.get('max_tokens', 500),
            verbose=splitter_config.get('verbose', 0),
            buffer_size=splitter_config.get('buffer_size', 3),
            max_big_chunks=splitter_config.get('max_big_chunks', 4),
            splitter_mode=splitter_config.get('splitter_mode', 'CUSTOM'),
            storage_path=os.path.abspath(splitter_config.get(
                'storage_path', "./data/figures/splitter")),
            min_initial_chunk_len=splitter_config.get(
                'min_initial_chunk_len', 50)
        )

    def _create_label_generator(self) -> LabelGenerator:
        label_generator_config = self.config.get('label_generator', {})
        return LabelGenerator(
            tokenizer=self._get_tokenizer(
                tokenizer_model=label_generator_config.get('tokenizer_model', 'GPT35')),
            labels=label_generator_config.get('labels', LabelGenerator.LABELS),
            model=label_generator_config.get('model', 'GPT'),
            max_samples=label_generator_config.get('max_samples', 10)
        )

    def _create_storer(self) -> Storer:
        storer_config = self.config.get('storer', {})
        return Storer(
            store_path=os.path.abspath(storer_config.get(
                'store_path', './data/boedataset')),
            file_name=storer_config.get('file_name', 'data'),
            file_format=storer_config.get('file_format', 'csv')
        )

    def _get_tokenizer(self, tokenizer_model: str):
        tokenizers_available = {
            'GPT35': tiktoken.encoding_for_model("gpt-3.5"),
            'GPT2': GPT2Tokenizer.from_pretrained('gpt2'),
            'LLAMA3': AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B"),
            'DEBERTA': AutoTokenizer.from_pretrained("microsoft/deberta-base"),
            'ROBERTA': AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        }
        return tokenizers_available.get(tokenizer_model, AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B"))

    def _get_embd_model(self, embd_model: str):
        embd_available = {
            str(os.getenv('EMBEDDING_MODEL')): HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL')))
        }
        return embd_available.get(embd_model, HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL'))))

    def get_dataframe(self, docs: list[Document]) -> pd.DataFrame:
        texts = [d.page_content for d in docs]
        return pd.DataFrame(data=texts, columns=["text"])

    def save_figure_from_df(self, df: pd.DataFrame, path: str, method: str) -> None:
        most_frequent_tokens = df.sum(
            axis=0, skipna=True).sort_values(ascending=False)
        num_tokens = 50
        fig = plt.figure(figsize=(16, 10))
        plt.bar(x=most_frequent_tokens.head(num_tokens).index,
                height=most_frequent_tokens.head(num_tokens).values)
        plt.xticks(rotation=45, ha='right')
        plt.title(
            f"Most frequent {num_tokens} tokens/terms in corpus using {method} method")
        plt.grid()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, format='png')
        plt.close(fig)

    def run(self) -> list[Document]:

        self.parsed_docs = self.parser.async_invoke()
        self.processor = self._create_processor(docs=self.parsed_docs)
        processed_docs = self.processor.invoke()
        logger.info(f"Number of processed_docs {len(processed_docs)}")
        try:
            logger.debug(
                f"Type of processed_docs[0] {type(processed_docs[0])}")
        except:
            pass
        split_docs = self.splitter.invoke(processed_docs)
        labeled_docs = self.label_generator.invoke(split_docs)
        self.storer.invoke(labeled_docs)

        return labeled_docs
