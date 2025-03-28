import logging
import pathlib
import re
from typing import List, Union
import json
import contractions
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from gensim.models.phrases import Phrases
from spacy_download import load_spacy

import src.acronyms as acronyms


class Pipe():
    """
    Class to carry out text preprocessing tasks that are needed by topic modeling
    - Basic stopword removal
    - Acronyms substitution
    - NLP preprocessing
    - Ngrams detection
    """

    def __init__(self,
                 stw_files: List[pathlib.Path],
                 spaCy_model: str,
                 language: str,
                 max_length: int,
                 raw_text_cols: List[str],
                 path_add_acr: str = None,
                 do_lemmatization: bool = True, # Add new parameter to control whether do lemmatization or not
                 logger=None):
        """
        Initilization Method
        Stopwords files will be loaded during initialization

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        spaCy_model: str
            Name of the spaCy model to be used for preprocessing
        language: str
            Language of the text to be preprocessed (en/es)
        max_length: int
            Maximum length of the text to be processed
        raw_text_cols : List[str]
            List of columns containing the raw text to be preprocessed
        path_add_acr: str
            Path to acronyms JSON file
        do_lemmatization: bool
            If True, lemmatization will be carried out
        logger: Logger object
            To log object activity
        """
        
        # Create logger object
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('nlpPipeline')
        
        self._do_lemmatization = do_lemmatization
        
        # Load stopwords and acronyms
        self._loadSTW(stw_files)
        self._loadACR(language, path_add_acr)

        # Download spaCy model if not already downloaded and load
        self._nlp = load_spacy(spaCy_model, exclude=['parser', 'ner'])
        self._nlp.max_length = max_length + round(0.1 * max_length)
        self._raw_text_cols = raw_text_cols

        return

    def _loadSTW(self, stw_files: List[pathlib.Path]) -> None:
        """
        Loads stopwords as list from files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        """

        stw_list = \
            [pd.read_csv(stw_file, names=['stopwords'], header=None,
                         skiprows=3) for stw_file in stw_files]
        stw_list = \
            [stopword for stw_df in stw_list for stopword in stw_df['stopwords']]
        self._stw_list = list(dict.fromkeys(stw_list))  # remove duplicates
        self._logger.info(
            f"-- -- Stopwords list created with {len(stw_list)} items.")

        return

    def _loadACR(self, lang: str, path_add_acr:str=None) -> None:
        """
        Loads list of acronyms. 
        
        If path_add_acr is provided, it will additionally load the acronyms from the JSON file
        
        Parameters
        ----------
        lang: str
            Language of the text to be preprocessed (en/es). This is used to select the default acronyms list from acronyms.py
        path_add_acr: str
            Path to acronyms JSON file
        """
        
        self._acr_list = acronyms.en_acronyms_list if lang == 'en' else acronyms.es_acronyms_list
        self._logger.info(f"-- -- Default acronyms list loaded for {lang} language with {len(self._acr_list)} items.")
        
        if path_add_acr is not None:
            self._logger.info(f"-- -- Loading additional acronyms from {path_add_acr}")
            
            with open(path_add_acr, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            add_acr_list = []
            for item in data['wordlist']:
                key, value = item.split(':')
                # Format the key as a regex word boundary
                regex = r'\b{}\b'.format(re.escape(key))
                # Add the regex and value to the list
                add_acr_list.append((regex, value))
            
            old_len = len(self._acr_list)
            self._acr_list = self._acr_list + add_acr_list
            
            self._logger.info(f"-- -- Acronyms list augmented with {len(add_acr_list) - old_len} items. Total acronyms: {len(self._acr_list)}")

        return

    def _replace(self, text, patterns) -> str:
        """
        Replaces patterns in strings.

        Parameters
        ----------
        text: str
            Text in which the patterns are going to be replaced
        patterns: List of tuples
            Replacement to be carried out

        Returns
        -------
        text: str
            Replaced text
        """
        for (raw, rep) in patterns:
            regex = re.compile(raw, flags=re.IGNORECASE)
            text = regex.sub(rep, text)
        return text

    def do_pipeline(self, rawtext) -> str:
        """
        Implements the preprocessing pipeline, by carrying out:
        - Lemmatization according to POS
        - Removal of non-alphanumerical tokens
        - Removal of basic English stopwords and additional ones provided       
          within stw_files
        - Acronyms replacement
        - Expansion of English contractions
        - Word tokenization
        - Lowercase conversion

        Parameters
        ----------
        rawtext: str
            Text to preprocess

        Returns
        -------
        final_tokenized: List[str]
            List of tokens (strings) with the preprocessed text
        """

        # Change acronyms by their meaning
        text = self._replace(rawtext, self._acr_list)

        # Expand contractions
        try:
            text = contractions.fix(text)
        except:
            text = text  # this is only for SS

        valid_POS = set(['VERB', 'NOUN', 'ADJ', 'PROPN'])
        
        # first filter of raw words (before lemmatization)
        text = ' '.join([word for word in text.split() if word not in self._stw_list and word != 'él'])
        doc = self._nlp(text)
        
        if self._do_lemmatization:
            tokens = [token.lemma_ for token in doc
                    if token.is_alpha
                    and token.pos_ in valid_POS
                    and not token.is_stop
                    and token.lemma_ not in self._stw_list]
        else:
            tokens = [token.text for token in doc
                    if token.is_alpha
                    and token.pos_ in valid_POS
                    and not token.is_stop
                    and token.text.lower() not in self._stw_list]
            
        #lemmatized = [token.lemma_ for token in doc
        #              if token.is_alpha
        #              and token.pos_ in valid_POS
        #              and not token.is_stop
        #               and token.lemma_ not in self._stw_list]

        # Convert to lowercase
        #final_tokenized = [token.lower() for token in lemmatized]
        final_tokenized = [token.lower() for token in tokens]

        return final_tokenized

    def preproc(self,
                corpus_df: Union[dd.DataFrame, pd.DataFrame],
                use_dask: bool = False,
                nw: int = 0,
                no_ngrams: bool = False) -> Union[dd.DataFrame, pd.DataFrame]:
        """
        Invokes NLP pipeline and carries out, in addition, n-gram detection.

        Parameters
        ----------
        corpus_df: Union[dd.DataFrame, pd.DataFrame]
            Dataframe representation of the corpus to be preprocessed. 
            It needs to contain (at least) the following columns:
            - raw_text
        nw: int
            Number of workers for Dask computations
        no_grams: Bool
            If True, calculation of ngrams will be skipped

        Returns
        -------
        corpus_df: Union[dd.DataFrame, pd.DataFrame]
            Preprocessed DataFrame
            It needs to contain (at least) the following columns:
            - raw_text
            - lemmas
        """
        
        if len(self._raw_text_cols) > 1:
            if self._do_lemmatization:
                new_raw_text_cols = [col.split("_")[0] + "_lemmas" for col in self._raw_text_cols]
            else:
                new_raw_text_cols = [col.split("_")[0] + "_tokens" for col in self._raw_text_cols]
        else:
            new_raw_text_cols = ["lemmas"] if self._do_lemmatization else ["tokens"]
                        
        for col, new_col in zip(self._raw_text_cols, new_raw_text_cols):
            action = "Lemmatizing" if self._do_lemmatization else "Tokenizing"
            self._logger.info(f"-- {action} text of {col}")
            if use_dask:
                corpus_df[new_col] = corpus_df[col].apply(
                    self.do_pipeline,
                    meta=('x', 'str'))
            else:
                corpus_df[new_col] = corpus_df[col].apply(
                    self.do_pipeline)
            
            # If no_ngrams is False, carry out n-grams detection
            if not no_ngrams:

                def get_ngram(doc):
                    return " ".join(phrase_model[doc])

                # Create corpus from tokenized lemmas
                self._logger.info(
                    "-- Creating corpus from lemmas for n-grams detection")
                source_text = "lemmas" if self._do_lemmatization else "tokens"
                if use_dask:
                    with ProgressBar():
                        if nw > 0:
                            tokens = corpus_df[new_col].compute(
                                scheduler='processes', num_workers=nw)
                        else:
                            # Use Dask default number of workers (i.e., number of cores)
                            tokens = corpus_df[new_col].compute(
                                scheduler='processes')
                else:
                    tokens = corpus_df[new_col]

                # Create Phrase model for n-grams detection
                self._logger.info("-- Creating Phrase model")
                phrase_model = Phrases(tokens, min_count=2, threshold=20)

                # Carry out n-grams substitution
                self._logger.info("-- Carrying out n-grams substitution")

                if use_dask:
                    corpus_df[new_col] = \
                        corpus_df[new_col].apply(
                            get_ngram, meta=('x', 'str'))
                else:
                    corpus_df[new_col] = corpus_df[new_col].apply(get_ngram)

            else:
                if use_dask:
                    corpus_df[new_col] = \
                        corpus_df[new_col].apply(
                            lambda x: " ".join(x), meta=('x', 'str'))
                else:
                    corpus_df[new_col] = corpus_df[new_col].apply(
                        lambda x: " ".join(x))

        return corpus_df
