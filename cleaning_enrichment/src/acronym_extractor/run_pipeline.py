"""Example script to run the pipeline for extracting acronyms from a text independently of the AcronymExtractor class.
"""
import os
from pathlib import Path
import typer
from wasabi import msg

from spacy_llm.util import assemble
from acronym_extract_task import AcronymExtractTask

Arg = typer.Argument
Opt = typer.Option


def run_pipeline(
    text: str = Arg(
        "",
        help="Text to extract acronyms from."),
    config_path: Path = Arg(
        ...,
        help="Path to the configuration file to use."),
    lang: str = Opt(
        "--en",
        help="Language to use."),
    verbose: bool = Opt(
        False, "--verbose", "-v",
        help="Show extra information."),
):
    global config_path_global
    global old_lang

    config_path_global = config_path

    if not os.getenv("OPENAI_API_KEY", None):
        msg.fail(
            "OPENAI_API_KEY env variable was not found. "
            "Set it by running 'export OPENAI_API_KEY=...' and try again.",
            exits=1,
        )

    msg.text(f"Loading config from {config_path}", show=verbose)
    nlp = assemble(
        config_path,
        overrides={"nlp.lang": lang}
    )
    doc = nlp(text)

    msg.text(f"Text: {doc.text}")
    msg.text(f"Acronyms: {doc._.acronyms}")

    return doc._.acronyms


if __name__ == "__main__":
    typer.run(run_pipeline)
