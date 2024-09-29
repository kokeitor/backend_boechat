import os
from langchain.schema import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from src.ETL.utils import exec_time
import logging
import asyncio


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

        # returns List[llama doc objt]
        self.llama_parsed_docs = self.reader.load_data()
        self.lang_parsed_docs = [d.to_langchain_format()
                                 for d in self.llama_parsed_docs]

        if len(self.lang_parsed_docs) == 0:
            logger.error("Parsed docs list empty")
        else:
            logger.info(f"Parsed num of docs -> {len(self.lang_parsed_docs) }")
        return self.lang_parsed_docs

    @exec_time
    async def async_invoke(self) -> list[Document]:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self.reader.aload_data, [
                                      file]) for file in os.listdir(self.path) if file.endswith(".pdf")]
        llama_parsed_docs = await asyncio.gather(*tasks)
        self.lang_parsed_docs = [d.to_langchain_format()
                                 for docs in llama_parsed_docs for d in docs]
        return self.lang_parsed_docs
