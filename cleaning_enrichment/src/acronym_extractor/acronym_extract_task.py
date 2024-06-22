"""This module defines a task for extracting acronyms from text using a language model and LLMs (spacy_llm). This task is then used within the AcronymExtractor class to extract acronyms from text.

Author: Lorena Calvo-Bartolomé
Date: 12/05/2024
"""

from pathlib import Path
from spacy_llm.registry import registry
import jinja2
from typing import Iterable
from spacy.tokens import Doc
import os

TEMPLATE_DIR = Path("templates")


@registry.llm_tasks("my_namespace.AcronymExtractTask.v1")
def make_quote_extraction() -> "AcronymExtractTask":
    """
    Factory function to create an instance of AcronymExtractTask.

    Returns:
    --------
        An instance of AcronymExtractTask.
    """
    return AcronymExtractTask()


def read_template(name: str) -> str:
    """
    Read a template file and return its content as a string.

    Parameters:
    -----------
    name: str
        The name of the template file.

    Returns:
    --------
    str: The content of the template file as a string.
    """
    path = TEMPLATE_DIR / f"{name}.jinja"

    if not path.exists():
        try:
            path_ = Path(os.getcwd()) / "src" / \
                "acronym_extractor" / path.as_posix()
            path = path_
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Neither {path.as_posix()} nor {path_.as_posix()} are a valid template.")

    return path.read_text()


class AcronymExtractTask(object):
    """
    Class representing an acronym extraction task.
    """

    def __init__(
        self,
        template: str = "acronym_extract_task",
        field: str = "acronyms"
    ):
        """
        Initialize an instance of AcronymExtractTask.

        Parameters
        ----------
        template: str
            The name of the template to use for prompt generation.
        field: str
            The name of the field to store the extracted acronyms.
        """
        self._template = read_template(template)
        self._field = field

    def _check_doc_extension(self):
        """
        Check if the document has the specified extension field.
        If not, add the extension field to the document.
        """
        if not Doc.has_extension(self._field):
            Doc.set_extension(self._field, default=None)

    def generate_prompts(
        self,
        docs: Iterable[Doc]
    ) -> Iterable[str]:
        """
        Generate prompts for the given documents.

        Parameters
        ----------
            docs: An iterable of spaCy Doc objects.

        Yields:
        -------
            The generated prompts as strings.
        """
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
            )
            yield prompt

    def parse_responses(
        self,
        docs: Iterable[Doc],
        responses: Iterable[str]
    ) -> Iterable[Doc]:
        """
        Parse the responses and update the documents with the extracted acronyms.

        Parameters
        ----------
        docs: 
            An iterable of spaCy Doc objects.
        responses:
            An iterable of response strings.

        Yields:
        -------
            The updated documents with the extracted acronyms.
        """
        self._check_doc_extension()
        for doc, prompt_response in zip(docs, responses):
            try:
                if type(prompt_response) == list and len(prompt_response) == 1:
                    try:
                        prompt_response = eval(prompt_response[0])
                    except:
                        prompt_response = prompt_response[0]
                setattr(
                    doc._,
                    self._field,
                    prompt_response,
                ),
            except ValueError:
                setattr(doc._, self._field, None)

            yield doc
