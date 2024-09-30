import os
from langchain.schema import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from src.ETL.utils import exec_time
import logging
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("parser_module_logger")


class Parser:
    def __init__(self,
                 directory_path: str,
                 file_type: str = ".pdf",
                 recursive_parser: bool = True,
                 result_type: str = "markdown",
                 verbose: bool = True,
                 api_key: str = os.getenv('LLAMA_CLOUD_API_KEY_RAPTOR')
                 ):
        self.path = directory_path
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,  # "markdown" and "text" are available
            verbose=verbose
        )

        self.reader = SimpleDirectoryReader(
            input_dir=self.path,
            file_extractor={file_type: self.parser},
            recursive=recursive_parser,  # recursively search in subdirectories
            required_exts=[file_type]
        )

    @exec_time
    def invoke(self) -> list[Document]:
        # returns List[llama doc obj]
        try:
            self.llama_parsed_docs = self.reader.load_data()
            try:
                print("parsed num of docs : ", len(self.llama_parsed_docs))
                logger.info(
                    f"parsed num of docs :{len(self.llama_parsed_docs)}")
            except Exception as e:
                print("parsed num of docs : ", e)
                logger.info(
                    f"parsed num of docs :{e}")
            self.lang_parsed_docs = [d.to_langchain_format()
                                     for d in self.llama_parsed_docs]

            if len(self.lang_parsed_docs) == 0:
                logger.error("Parsed docs list empty")
                print("Parsed docs list empty")
            else:
                logger.info(
                    f"Parsed num of docs -> {len(self.lang_parsed_docs)}")
            return self.lang_parsed_docs
        except Exception as e:
            logger.error(f"Failed to parse documents: {str(e)}")
            print(f"Failed to parse documents: {str(e)}")
            return []

    @exec_time
    async def async_invoke(self) -> list[Document]:
        """
        Asynchronously parse documents in the directory.
        """
        nest_asyncio.apply()
        try:
            # Run the asynchronous LlamaParse parser using `aload_data`
            llama_parsed_docs = await self.reader.aload_data()

            # Convert the parsed documents to LangChain format
            self.lang_parsed_docs = [d.to_langchain_format()
                                     for d in llama_parsed_docs]

            if len(self.lang_parsed_docs) == 0:
                logger.error("Parsed docs list empty")
            else:
                logger.info(
                    f"Parsed num of docs -> {len(self.lang_parsed_docs)}")

            return self.lang_parsed_docs

        except Exception as e:
            logger.error(f"Failed to asynchronously parse documents: {str(e)}")
            return []
