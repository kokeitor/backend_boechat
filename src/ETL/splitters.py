import os
import re
import uuid
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2Tokenizer
from typing import Union, Optional, ClassVar
from langchain.schema import Document
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging

from ETL.utils import get_current_spanish_date_iso, setup_logging

# Set the default font to DejaVu Sans
# or another font that includes the glyphs
plt.rcParams['font.family'] = 'DejaVu Sans'


# Logging configuration
# Child logger [for this module]
logger = logging.getLogger("splitters_module_logger")

# Tokenizers
TOKENIZER_GPT3 = tiktoken.encoding_for_model("gpt-3.5")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(
    'gpt2', clean_up_tokenization_spaces=False)
TOKENIZER_LLAMA3 = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", clean_up_tokenization_spaces=False)
tokenizer_deberta = AutoTokenizer.from_pretrained(
    "microsoft/deberta-base", clean_up_tokenization_spaces=False)
tokenizer_roberta = AutoTokenizer.from_pretrained(
    "PlanTL-GOB-ES/roberta-base-bne", clean_up_tokenization_spaces=False)

# Embedding model
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class CustomSemanticSplitter:
    MIN_INITIAL_LONG_CHUNKS: int = 5

    def __init__(
            self,
            embedding_model=EMBEDDING_MODEL,
            tokenizer=TOKENIZER_LLAMA3,
            buffer_size: int = 2,
            threshold: int = 75,
            verbose: int = 0,
            max_tokens: int = 500,
            max_big_chunks: int = 3,
            storage_path: str = "./data/figures",
            min_initial_chunk_len: int = 50):

        self.buffer_size = buffer_size
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_big_chunks = max_big_chunks
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.threshold = threshold
        self.namespace_id = uuid.NAMESPACE_DNS
        self.storage_path = os.path.abspath(storage_path)
        self.min_initial_chunk_len = min_initial_chunk_len

    def _prepare_texts(self, doc: Document) -> list[dict]:

        text = doc.page_content
        self._metadata = doc.metadata.copy()

        logger.debug(f"Texto para splitear [100 char max] : {text[0:100]}")

        # Iterate splitter until we get initial len chunks reach a minimum size
        # split patterns the text on : Level 1 on '.', '?', and '!' - Level 2 on \n - Level 3 on \n\n
        # split_patterns = [r'(?<=[.?!])\s+',r'\n',r'\n\n']
        split_patterns = [r'\n']
        for split_level, s_pattern in enumerate(split_patterns):
            sentence_list = re.split(s_pattern, text)
            short_sentences = 0
            for sentence in sentence_list:
                if len(sentence) < self.min_initial_chunk_len:
                    short_sentences += 1
            if short_sentences > CustomSemanticSplitter.MIN_INITIAL_LONG_CHUNKS:
                lista = []
                for i, d in enumerate(sentence_list):
                    logger.info(f"Level of split pattern used {split_level}")
                    logger.info(f"CHUNK {i} LEN -> {len(d)}")
                    lista.append({'sentence': d, 'index': i})
                # return [{'sentence': d, 'index': i} for i, d in enumerate(sentence_list)]
                return lista

    def _get_id(self, text: str) -> str:
        return str(uuid.uuid5(self.namespace_id, text))

    def _combine_sentences(self, sentences_to_combine: list[dict], buffer_size: int = None) -> list[dict]:

        if buffer_size is None:
            buffer_size = self.buffer_size

        if not isinstance(sentences_to_combine, list):
            logger.exception("Expected sentences_to_combine to be a list")
            raise ValueError("Expected sentences_to_combine to be a list")

        num_sentences_exceed = 0
        sentences = sentences_to_combine.copy()

        for i in range(len(sentences)):
            combined_sentence = ''

            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            combined_sentence += sentences[i]['sentence']

            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            sentences[i]['combined_sentence'] = combined_sentence

            num_tokens = self._get_tokens(text=combined_sentence)
            # Logging
            try:
                logger.info(
                    f'Combined sentence : index : {i} // tokens : {num_tokens} // text len : {len(combined_sentence)}')
            except UnicodeEncodeError:
                logger.info(
                    f"Combined sentence : index : {i} // tokens : {num_tokens} // text len : {len(combined_sentence.encode('utf-8', 'ignore').decode('utf-8'))}")
            if num_tokens > self.max_tokens:
                num_sentences_exceed += 1

        if num_sentences_exceed >= self.max_big_chunks:
            if buffer_size > 1:
                self.buffer_size -= 1
                return self._combine_sentences(sentences_to_combine, buffer_size=self.buffer_size)
            else:
                logger.critical(f"Min buffer size : 1")
                return sentences

        return sentences

    def _get_embeddings(self, sentences_to_embd: list[str]) -> list[float]:
        return self.embedding_model.embed_documents(sentences_to_embd)

    def _get_similarity(self, embeddings: list[float], similarity: str) -> list[float]:
        metrics = {'COSINE': nn.CosineSimilarity(dim=0, eps=1e-08)}
        embedding_tensors = torch.tensor(embeddings)
        similarity_executer = metrics.get(similarity, None)
        similarity = []
        if similarity_executer is not None:
            for i in range(embedding_tensors.shape[0] - 1):
                t1 = embedding_tensors[i, :]
                t2 = embedding_tensors[i + 1, :]
                similarity.append(1.0 - similarity_executer(t1, t2).item())
            if embedding_tensors.shape[0] > 1:
                # Append the last similarity value to avoid out of range error
                similarity.append(similarity[-1])
        return similarity

    def _get_chunks(self, sentences: list[dict], threshold: int = 75) -> list[dict]:
        distances = [x["distance_to_next"] for x in sentences]
        breakpoint_distance_threshold = np.percentile(distances, threshold)
        indices_above_thresh = [i for i, x in enumerate(
            distances) if x > breakpoint_distance_threshold]
        start_index = 0
        chunks = []

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunk_metadata = self._metadata.copy()
            chunk_metadata['chunk_id'] = self._get_id(text=combined_text)
            chunks.append({'chunk_text': combined_text,
                          'chunk_metadata': chunk_metadata})
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence']
                                     for d in sentences[start_index:]])
            chunk_metadata = self._metadata.copy()
            chunk_metadata['chunk_id'] = self._get_id(text=combined_text)
            chunks.append({'chunk_text': combined_text,
                          'chunk_metadata': chunk_metadata})
        return chunks

    def _plot_similarity(self, pdf_id: str, sentences: list[dict], threshold: int = 75):
        """
        Plot similarity distances between sentences and save the plots.

        Args:
            pdf_id (str): Identifier for the PDF document.
            sentences (list[dict]): list of sentence dictionaries with similarity distances.
            threshold (int, optional): Percentile threshold for determining significant distances. Defaults to 75.
        """
        distances = [x["distance_to_next"] for x in sentences]
        max_distance = np.max(distances)

        # Plot similarity distances
        plt.figure(figsize=(12, 8))
        plt.plot(distances, marker='o')
        plt.xticks(ticks=np.arange(len(distances)),
                   labels=np.arange(len(distances)))
        y_upper_bound = max_distance * 1.05
        plt.ylim(0, y_upper_bound)
        plt.xlim(0, len(distances))
        breakpoint_distance_threshold = np.percentile(distances, threshold)
        plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
        num_distances_above_threshold = len(
            [x for x in distances if x > breakpoint_distance_threshold])
        plt.text(x=(len(distances) * 0.01), y=y_upper_bound / 50,
                 s=f"{num_distances_above_threshold + 1} Chunks")
        indices_above_thresh = [i for i, x in enumerate(
            distances) if x > breakpoint_distance_threshold]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for i, breakpoint_index in enumerate(indices_above_thresh):
            start_index = 0 if i == 0 else indices_above_thresh[i - 1]
            end_index = breakpoint_index if i < len(
                indices_above_thresh) - 1 else len(distances)
            plt.axvspan(start_index, end_index,
                        facecolor=colors[i % len(colors)], alpha=0.25)
            plt.text(x=np.average([start_index, end_index]), y=breakpoint_distance_threshold + (y_upper_bound / 20),
                     s=f"Chunk #{i}", horizontalalignment='center', rotation='vertical')

        if indices_above_thresh:
            last_breakpoint = indices_above_thresh[-1]
            if last_breakpoint < len(distances):
                plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(
                    indices_above_thresh) % len(colors)], alpha=0.25)
                plt.text(x=np.average([last_breakpoint, len(distances)]), y=breakpoint_distance_threshold + (y_upper_bound / 20),
                         s=f"Chunk #{len(indices_above_thresh)}", rotation='vertical')

        plt.title("Chunks Based On Embedding Breakpoints")
        plt.xlabel("Index sentence")
        plt.ylabel("Similarity distance between pairwise sentences")
        plot_file = os.path.join(os.path.abspath(
            self.storage_path), f"{get_current_spanish_date_iso()}_{pdf_id}_similarity_plot.png")
        logger.info(f"Saving similarity plot to -> {plot_file}")
        plt.savefig(plot_file, format='png')
        plt.close()

        # Plot histogram of similarity distances
        line = np.arange(0, 10, 0.01)
        plt.figure(figsize=(10, 6))
        plt.hist(distances, alpha=0.5, color='b')
        plt.plot([breakpoint_distance_threshold] * len(line), line, color='r')
        plt.text(x=breakpoint_distance_threshold - 0.01, y=0,
                 s=f"Percentile: {threshold}", rotation='vertical', color='r')
        plt.title("Histogram of Similarity Distances")
        plt.ylabel("Frequency")
        plt.xlabel("Similarity distance between pairwise sentences")
        plt.grid(alpha=0.75)
        plot_file_hist = os.path.join(os.path.abspath(
            self.storage_path), f"{get_current_spanish_date_iso()}_{pdf_id}_similarity_hist.png")
        logger.info(f"Saving similarity histogram plot to -> {plot_file_hist}")
        plt.savefig(plot_file_hist, format='png')
        plt.close()

    def _get_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def _create_docs(self, chunks: list[dict]) -> list[Document]:
        return [Document(page_content=chunk_dict['chunk_text'], metadata=chunk_dict['chunk_metadata']) for chunk_dict in chunks]

    def split_documents(self, docs: list[Document]) -> list[Document]:
        self.docs = docs.copy()
        self.spitted_docs = []

        for _, doc in enumerate(self.docs):
            self.doc = doc.copy()
            self.sentences = self._prepare_texts(doc=self.doc)
            self.sentences = self._combine_sentences(
                sentences_to_combine=self.sentences)
            embed_combined_sentences = self._get_embeddings(
                sentences_to_embd=[x["combined_sentence"] for x in self.sentences])
            for i, sentence in enumerate(self.sentences):
                sentence["combined_sentence_embedding"] = embed_combined_sentences[i]
            similarities = self._get_similarity(
                embeddings=embed_combined_sentences, similarity='COSINE')
            for i, sentence in enumerate(self.sentences):
                sentence["distance_to_next"] = similarities[i]
            self.chunks = self._get_chunks(
                sentences=self.sentences, threshold=self.threshold)
            if self.verbose == 1:
                self._plot_similarity(
                    sentences=self.sentences, threshold=self.threshold, pdf_id=doc.metadata['pdf_id'])
            docs = self._create_docs(chunks=self.chunks)
            self.spitted_docs += docs

        return self.spitted_docs


class Splitter:
    def __init__(self,
                 chunk_size: int = 200,
                 storage_path: str = "C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyecto\\data\\figures",
                 embedding_model=EMBEDDING_MODEL,
                 tokenizer_model=TOKENIZER_LLAMA3,
                 threshold: int = 75,
                 max_tokens: int = 500,
                 verbose: int = 0,
                 buffer_size: int = 3,
                 max_big_chunks: int = 4,
                 min_initial_chunk_len: int = 50,
                 splitter_mode: str = 'CUSTOM'):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.tokenizer_model = tokenizer_model
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.verbose = verbose
        self.max_big_chunks = max_big_chunks
        self.splitter_mode = splitter_mode
        self.storage_path = storage_path
        self.min_initial_chunk_len = min_initial_chunk_len
        self.splitter = self._init_splitter()

    def _init_splitter(self):
        splitter_modes = {
            'RECURSIVE': RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0,
                length_function=len,
                separators=["\n\n"]
            ),
            'CUSTOM': CustomSemanticSplitter(
                embedding_model=self.embedding_model,
                buffer_size=self.buffer_size,
                threshold=self.threshold,
                verbose=self.verbose,
                tokenizer=self.tokenizer_model,
                max_tokens=self.max_tokens,
                max_big_chunks=self.max_big_chunks,
                storage_path=self.storage_path,
                min_initial_chunk_len=self.min_initial_chunk_len
            )
        }
        return splitter_modes.get(self.splitter_mode)

    def invoke(self, docs: list[Document]) -> list[Document]:
        if isinstance(docs, list):
            return self.splitter.split_documents(docs)
        elif isinstance(docs, Document):
            return self.splitter.split_documents([docs])
