"""
This module provides a class for extracting acronyms from text using Spacy and LLMs (spacy_llm).

The AcronymExtractor class initializes a language model and provides a method to extract acronyms from text. If the text is too long for the context of the language model used, it will be split into chunks and the acronyms will be extracted from each chunk.

Example usage:
    extractor = AcronymExtractor()
    acronyms = extractor.extract("This is a sample text.")
    print(acronyms)  # Output: [('ML', 'Machine Learning')]

Author: Lorena Calvo-Bartolomé
Date: 12/05/2024
"""

import os
import pathlib
import re
from wasabi import msg
from dotenv import load_dotenv
import logging

from spacy_llm.util import assemble
from src.acronym_extractor.acronym_extract_task import AcronymExtractTask
from src.utils import flatten, split_into_chunks


class AcronymExtractor(object):
    def __init__(
        self,
        config_path: pathlib.Path = None,
        lang: str = "en",
        logger: logging.Logger = None
    ) -> None:
        """
        Initialize the AcronymExtractor.

        Parameters
        ----------
        config_path : pathlib.Path, optional
            The path to the configuration file for the language model, by default None
        lang : str, optional
            The language of the text, by default "en"
        logger : logging.Logger, optional
            The logger object for logging messages, by default None
        """
        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

        if not os.getenv("OPENAI_API_KEY", None):
            msg.fail(
                "OPENAI_API_KEY env variable was not found. "
                "Set it by running 'export OPENAI_API_KEY=...' and try again.",
                exits=1,
            )

        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('AcronymExtractor')

        if not config_path:
            config_path = pathlib.Path(
                os.getcwd()) / "src" / "acronym_extractor" / "config.cfg"

        self._logger.info(f"-- -- Loading config from {config_path}")
        self._nlp = assemble(
            config_path,
            overrides={
                "nlp.lang": lang,
            }
        )

    def extract(self, text: str) -> list[str]:
        """
        Extract acronyms from text. If the text is too long for the context of the language model used, it will be split into chunks based on the maximum context length and size of the prompt template, and the acronyms will be extracted from each chunk.

        Parameters
        ----------
        text : str
            The text to extract acronyms from.

        Returns
        -------
        list
            A list of tuples, where each tuple contains the acronym and its definition.
        """
        try:
            return self._nlp(text)._.acronyms
        except Exception as e:
            # Extract the maximum context length from the error
            pattern = r"maximum context length is (\d+) tokens"
            match = re.search(pattern, e.args[0])

            if match:
                max_context_length = int(match.group(1))
                self._logger.info(
                    f"-- -- The maximum context length is {max_context_length} tokens.")
                promt = pathlib.Path(os.getcwd(
                )) / "src" / "acronym_extractor" / "templates" / "acronym_extract_task.jinja"
                promt_text = promt.read_text()
            else:
                self._logger.error(
                    "-- -- Maximum context length not found in the error message. Another error occurred.")

            # Split the text into chunks
            text_chunks = split_into_chunks(
                text, max_context_length - len(promt_text) - 1)

            processed_chunks = []
            for i, chunk in enumerate(text_chunks):
                try:
                    acr_chunk = self._nlp(chunk)._.acronyms
                    processed_chunks.append(acr_chunk)
                except Exception as e:
                    self._logger.error(
                        f"-- -- Error processing chunk {i} of {len(text_chunks)}")
                    self._logger.error(
                        f"-- -- Size of chunk + promt: {len(chunk) + len(promt_text)}")
                    self._logger.error(f"-- -- Error processing chunk: {e}")
                    continue

            return flatten(processed_chunks)
