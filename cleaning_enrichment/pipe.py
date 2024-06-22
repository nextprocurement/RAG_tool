"""
Main script to apply the NLP pipeline to clean and enrich text data.

Author: Lorena Calvo-Bartolomé, Saúl Blanco Fortes
Date: 12/05/2024
"""

import argparse
import logging
import sys

import pandas as pd
import yaml
from src.nlp_pipeline import NLPpipeline


def cal_element_pipe(
    element: str,
    df: pd.DataFrame,
    col_calculate_on: str,
    pipelines: dict,
    logger: logging.Logger,
    replace_acronyms: bool = False
) -> pd.DataFrame:
    """
    Applies a series of NLP processing steps to a DataFrame based on the specified element.

    Parameters
    ----------
    element: str
        The type of processing to perform (e.g., 'acronyms', 'lemmas', 'ngrams', 'embeddings', 'ner_generic', 'ner_specific').
    df: pd.DataFrame
        The input DataFrame containing text data.
    col_calculate_on: str
        The column in the DataFrame to apply the processing on.
    pipelines: dict
        Dictionary of pre-initialized NLPpipeline objects for different languages.
    logger: logging.Logger
        Logger for recording informational messages.
    replace_acronyms: bool, optional
        Whether to replace acronyms. Defaults to False.

    Returns
    -------
    pd.DataFrame: The processed DataFrame.
    """
    
    # Apply cleaning as first step
    df = pipelines['default'].get_clean_text(df, col_calculate_on)

    # Identify the language if the option is enabled
    if element == "lang_id":
        if "lang" not in df.columns:
            df = pipelines['default'].get_lang(df, col_calculate_on)
        df_en = df[df.lang == "eng_Latn"]
        df_es = df[df.lang == "spa_Latn"]

        if df_en.empty and df_es.empty:
            logger.info(
                "-- -- The provided texts are not in English nor in Spanish. Returning the original dataframe with the languages identified...")
            return df
        dfs = [
            (df_en, pipelines["eng_Latn"]) if not df_en.empty else None,
            (df_es, pipelines["spa_Latn"]) if not df_es.empty else None
        ]
        dfs = [x for x in dfs if x is not None]
    else:
        dfs = [(df, pipelines['default'])]

    for df_, pipe in dfs:
        if element == "lang_id":
            continue  # Skip processing if the element is "lang_id"
        elif element == "acronyms":
            df_ = pipe.get_acronyms(df_, col_calculate_on)
        elif element == "lemmas":
            df_ = pipe.get_lemmas(df_, col_calculate_on, replace_acronyms)
        elif element == "ngrams":
            df_ = pipe.get_ngrams(df_, col_calculate_on)
        elif element == "embeddings":
            df_ = pipe.get_context_embeddings(df_, col_calculate_on)
        elif element == "ner_generic":
            df_ = pipe.get_ner_generic(df_, col_calculate_on)
        elif element == "ner_specific":
            df_ = pipe.get_ner_specific(df_, col_calculate_on)
        else:
            raise ValueError(f"-- -- Element '{element}' not recognized.")

    # Concatenate all processed DataFrames
    if dfs:
        final_dfs = [df_ for df_, _ in dfs]
        df = pd.concat(final_dfs)
    else:
        logger.warning(
            "-- -- No DataFrames to concatenate. Returning the original DataFrame.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Options for clean and enrich text data.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file",
        required=False)
    parser.add_argument(
        "--cols_calculate_on",
        default="raw_text",
        help="Columns on which the pipeline is going to be applied, separated by commas.",
        required=False)
    parser.add_argument(
        '-s',
        '--source',
        help='Input parquet file',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='Output parquet file',
        required=True)

    args = parser.parse_args()

    # Create logger object
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    try:
        # Load config file
        with open(args.config, "r") as f:
            config = dict(yaml.safe_load(f))
    except Exception as E:
        logger.error(f"-- -- Error opening config file: {E}. Exiting...")
        sys.exit()

    # Load pipeline
    pipe = config.get("pipe", [])

    if not pipe:
        logger.error("-- -- No pipeline defined in config file. Exiting...")
        sys.exit()

    # Create pipeline with options from config file
    options_pipe = config.get("options_pipe", {})
    if not options_pipe:
        logger.warning(
            "-- -- Pipeline options dictionary is empty. Using default values, which may degrade performance if the text language is incorrect.")

    # Get default language options
    langs_dft = config.get("langs_dft", {})
    if not langs_dft:
        langs_dft = {
            "spaCy_model_dft_en": 'en_core_web_md',
            "spaCy_model_dft_es": 'es_core_news_md'
        }

    # Create NLP pipelines
    pipelines = {
        "default": NLPpipeline(**options_pipe),
        "eng_Latn": NLPpipeline(**{**options_pipe, "spaCy_model": langs_dft["spaCy_model_dft_en"]}),
        "spa_Latn": NLPpipeline(**{**options_pipe, "spaCy_model": langs_dft["spaCy_model_dft_es"], "lang": "es"})
    }

    try:
        # Load data
        df = pd.read_parquet(args.source)
        try:
            cols_calculate_on = args.cols_calculate_on.split(",")
        except:
            cols_calculate_on = [args.cols_calculate_on]
        logger.info("-- -- Data loaded successfully.")
        logger.info(f"-- -- Columns to preprocess: {cols_calculate_on}")
        logger.info(f"-- -- Printing sample of the data: {df.head()}")
    except Exception as E:
        logger.error(f"-- -- Error opening data file: {E}. Exiting...")
        sys.exit()

    # Apply pipeline
    print(f"This is the pipe: {pipe}")
    replace_acronyms = True if "acronyms" in pipe else False
    for col_calculate_on in cols_calculate_on:
        logger.info(f"-- -- Preprocessing column {col_calculate_on}...")
        
        for element in pipe:
            logger.info(f"-- -- Calculating element '{element}'")

            df = cal_element_pipe(
                element=element,
                df=df,
                col_calculate_on=col_calculate_on,
                pipelines=pipelines,
                replace_acronyms=replace_acronyms,
                logger=logger
            )
            
    logger.info(f"-- -- New columns of the dataframe: {df.columns}")
    logger.info(f"-- -- Printing sample of the preprocessed data: {df.head()}")
    # Save data
    try:
        if 'raw_text_ACR' in df.columns:
            df['raw_text_ACR'] = df['raw_text_ACR'].astype('str')
        df.to_parquet(args.output)
    except Exception as E:
        logger.error(f"-- -- Error saving data: {E}. Exiting...")
        sys.exit()