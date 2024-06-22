"""
This module contains the NERSpecificExtractor class, which is used to extract named entities from a given text using a specific NER model.

The NERSpecificExtractor class provides a method extract a list of named entities from text. It uses a pre-trained NER model augmented with LLM-based input to perform the extraction.

Example usage:
    extractor = NERSpecificExtractor()
    text = "Apple Inc. is a technology company based in Cupertino, California."
    entities = extractor.extract(text)
    print(entities)
    # Output: [('Apple Inc.', 'ORG'), ('Cupertino', 'GPE'), ('California', 'GPE')]
    
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
from src.utils import split_into_chunks, flatten

class NERSpecificExtractor(object):
    def __init__(
        self,
        config_path: pathlib.Path = None,
        lang: str = "en",
        logger: logging.Logger = None
    ) -> None:
        """
        Initialize the NERSpecificExtractor.

        Parameters:
        ----------
        config_path (pathlib.Path, optional): 
            ath to the configuration file for the NER model. If not provided, the default configuration file will be used.
        lang (str, optional):
            Language code for the NER model. Defaults to "en".
        logger (logging.Logger, optional):
            Logger object for logging messages. If not provided, a default logger will be created.
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
            self._logger = logging.getLogger('NERSpecificExtractor')

        if not config_path:
            config_path = pathlib.Path(
                os.getcwd()) / "src" / "ner_specific_extractor" / "config.cfg"
            examples_path = pathlib.Path(
                os.getcwd()) / "src" / "ner_specific_extractor" / "examples.json"

        self._logger.info(f"-- -- Loading config from {config_path}")
        self._nlp = assemble(
            config_path,
            overrides={
                "nlp.lang": lang,
                "components.llm.task.examples.path": examples_path.as_posix()
            }
        )

    def extract(self, text: str) -> list: 
        """
        Extract named entities from the given text.

        Parameters:
        ----------
        text (str): 
            The text from which to extract named entities.
        
        Returns:
        --------
        list: A list of tuples, where each tuple contains the text of a named entity and its label.
        """

        try:
            return [(ent.text, ent.label_)
                    for ent in self._nlp(text).ents]
        except Exception as e:
            # Extract the maximum context length from the error
            pattern = r"maximum context length is (\d+) tokens"
            match = re.search(pattern, e.args[0])

            if match:
                max_context_length = int(match.group(1))
                self._logger.info(
                    f"-- -- The maximum context length is {max_context_length} tokens.")
            else:
                self._logger.error(
                    "-- -- Maximum context length not found in the error message. Another error occurred.")
            
            # Split the text into chunks
            text_chunks = split_into_chunks(text, max_context_length  - 1)
            
            processed_chunks = []
            for chunk in text_chunks:
                try:
                    ner_chunk = [(ent.text, ent.label_)
                    for ent in self._nlp(chunk).ents]
                    processed_chunks.append(ner_chunk)
                except Exception as e:
                    self._logger.error(f"-- -- Error processing chunk: {e}")
                    continue

            return flatten(processed_chunks)
        
        
     