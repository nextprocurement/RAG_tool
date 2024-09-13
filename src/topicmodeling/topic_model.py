import json
import logging
import os
import pathlib
import re
import shutil
import sys
from abc import ABC, abstractmethod
from multiprocessing import freeze_support
from subprocess import check_output
import time
from typing import Dict, List, Tuple, Union
import openai

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import \
    TopicModelDataPreparation
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from hdbscan import HDBSCAN
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from umap import UMAP

from src.topicmodeling.utils import (file_lines, load_processed_data, pickler, tkz_clean_str,
                                      unpickler)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TopicModel(ABC):

    def __init__(
        self,
        num_topics: int = 35,
        topn: int = 15,
        load_data_path: str = "data/source/cordis_preprocessed.json",
        load_model: bool = False,
        model_path: str = None,
        logger: logging.Logger = None
    ) -> None:
        """

        Initializes a TopicModel instance.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to train the model (default is 35).
        topn : int, optional
            Number of top keywords to show for each topic (default is 15).
        load_data_path : str, optional
            Path of the processed data (default is "data/source/cordis_preprocessed.json").
        load_model : bool, optional
            If True, directly load the model; if False, train a new model (default is False).
        model_path : str, optional
            Path to the model to be loaded (default is None).
        """

        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('TopicModel')

        self.load_model = load_model

        self.save_path = pathlib.Path(model_path)
        # Path to save the "results" dictionary
        self.results_path = self.save_path / "results.pkl"
        # Path to saev the "model" object (if necessary)
        self.model_path = self.save_path / "model.pkl"

        if not load_model:
            self.num_topics = num_topics
            self.topn = topn
            self.load_data_path = load_data_path

            self.document_probas = None
            self.doc_topic_probas = None
            self.word_topic_distribution = None
            self.topics = None
            self.train_data = None
            self.coherence_cv = None
            self.coherence_uci = None
            self.coherence_npmi = None
            self.topic_reses = None
            self.topic_res_nums = None

            if self.save_path.exists():
                self._logger.info(
                    f"-- -- Save path {self.save_path} exists. Saving a copy ..."
                )
                old_model_dir = self.save_path.parent / \
                    (self.save_path.name + "_old")
                if not old_model_dir.is_dir():
                    os.makedirs(old_model_dir)
                    shutil.move(self.save_path, old_model_dir)

            self.save_path.mkdir(exist_ok=True)

        else:
            # Load the model
            self.load()

    @abstractmethod
    def print_topics(self, verbose=False):
        """Print the list of topics for the topic model"""
        pass

    @abstractmethod
    def get_thetas(self):
        """Get doc-topic distribution."""
        pass

    @abstractmethod
    def get_betas(self):
        """Get word-topic distribution."""
        pass

    def get_embeddings_from_str(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:
        """Get embeddings from a DataFrame, assumming there is a column named 'embeddings' with the embeddings as strings.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the embeddings as strings in a column named 'embeddings'

        Returns
        -------
        np.ndarray
            Array of embeddings
        """

        if "embeddings" not in df.columns:
            self._logger.error(
                f"-- -- DataFrame does not contain embeddings column"
            )
            raise ValueError(
                f"DataFrame does not contain embeddings column"
            )
        embeddings = df.embeddings.values.tolist()
        if isinstance(embeddings[0], str):
            embeddings = np.array(
                [np.array(el.split(), dtype=np.float32) for el in embeddings])

        return np.array(embeddings)

    def train(
        self,
        get_embeddings: bool = False,
        get_preprocessed: bool = True,
        further_proc: bool = True,
        min_lemas: int = 3,
        no_below: int = 10,
        no_above: float = 0.6, 
        keep_n: int = 100000,
        stops_path: str = "src/topicmodeling/data/stops",
        eqs_path: str = "src/topicmodeling/data/equivalences"
    ) -> None:
        """
        Train the topic model and save the data to save_data_path
        Here, at the parent class, we load the processed data and initialize the save_path. The actual training is done in the child classes.
        
        
        min_lemas: int
            Minimum number of lemas for document filtering
        no_below: int
            Minimum number of documents to keep a term in the vocabulary
        no_above: float
            Maximum proportion of documents to keep a term in the vocab
        keep_n: int
            Maximum vocabulary size
        cntVecModel : pyspark.ml.feature.CountVectorizerModel
            CountVectorizer Model to be used for the BOW calculation
        """

        # Path to save preproc data in model fodler        
        preproc_file = pathlib.Path(self.save_path.as_posix()).joinpath(pathlib.Path(self.load_data_path).stem + "_preproc.parquet")
        
        df = load_processed_data(self.load_data_path)
                
        if further_proc: # remove add stops and equivs
            start_time = time.time()
            self._logger.info(f"-- -- Applying further processing to the data")
            df['lemmas'] = df['lemmas'].apply(lambda row: tkz_clean_str(row, stops_path, eqs_path))
            self._logger.info(f"-- -- Further processing done in {(time.time() - start_time) / 60} minutes. Saving to {preproc_file}")
        
        # filter words with less than 3 characters
        self._logger.info(f"-- -- Filtering out words with less than 3 characters")
        df["lemmas"] = df['lemmas'].apply(lambda x: ' '.join([el for el in x.split() if len(el) > 3]))  # remove short words
        
        # filter extrems
        self._logger.info(f"-- -- Filtering out documents with less than {min_lemas} lemas")
        df["n_tokens"] = df["lemmas"].apply(lambda x : len(x.split()))
        len_df = len(df)
        df = df[df["n_tokens"] >= min_lemas]
        self._logger.info(f"-- -- Filtered out {len_df - len(df)} documents with less than {min_lemas} lemas")
        
        # Gensim filtering
        self._logger.info(f"-- -- Filtering out vocabulary with no_below={no_below}, no_above={no_above}, keep_n={keep_n}")
        final_tokens = [el.split() for el in df['lemmas'].values.tolist()]
        dict = corpora.Dictionary(final_tokens)

        dict.filter_extremes(
            no_below=no_below,
            no_above=no_above, 
            keep_n=keep_n
        )
        
        vocabulary = set([dict[idx] for idx in range(len(dict))])
        self._logger.info(f"-- -- Vocabulary size: {len(vocabulary)}")
        #import pdb; pdb.set_trace()
        df["lemmas"] = df['lemmas'].apply(lambda x: ' '.join([el for el in x.split() if el in vocabulary]))
        
        # Save Gensim dictionary
        self._logger.info(f"-- -- Saving Gensim dictionary")
        GensimFile = self.save_path.joinpath('dictionary.gensim')
        if GensimFile.is_file():
            GensimFile.unlink()
        dict.save_as_text(GensimFile)
        with self.save_path.joinpath('vocabulary.txt').open('w', encoding='utf8') as fout:
            fout.write(
                '\n'.join([dict[idx] for idx in range(len(dict))]))
            
        df.to_parquet(preproc_file)
        
        self.df = df

        if get_preprocessed:
            self.train_data = [doc.split() for doc in df.lemmas]
            self._logger.info(
                f"-- -- Loaded processed data from {self.load_data_path}"
            )

        # Get the embeddings from the DataFrame if necessary
        if get_embeddings:
            if "embeddings" not in df.columns:
                raise ValueError(
                    f"DataFrame does not contain embeddings column"
                )
            self.embeddings = self.get_embeddings_from_str(df)
            self._logger.info(
                f"-- -- Loaded embeddings from the DataFrame"
            )
        else:
            self.embeddings = None

        return

    def infer(
        self,
        docs
    ):
        """Given unseen documents in token lists, predict the topic distribution of the document."""

        # In case the model is not loaded, return an error message
        #  To perform inference, a model should be loaded
        if not self.load_model:
            self._logger.error(
                f"-- -- Model not loaded. Cannot perform inference.")
            return

        def preprocess_holdout(corpus: List[str]):

            nlp = spacy.load('en_core_web_sm')
            nlp.add_pipe('sentencizer')

            # Initialize an empty list to store the results
            docs = []
            # Wrap `tqdm` around `data` to create a progress bar
            for doc in tqdm(corpus):
                # Process each item with `nlp` and append to `docs` list
                docs.append(nlp(doc))

            data_words_nonstop, word_spans = [], []
            for i, doc in enumerate(docs):
                temp_doc = []
                temp_span = []
                for token in doc:
                    if (re.search('[a-z0-9]+', str(token))) \
                        and not len(str(token)) == 1 and not token.is_digit and not token.is_space \
                            and str(token).lower() not in STOP_WORDS and str(token).strip() != "":
                        temp_doc.append(token.lemma_.lower())
                        temp_span.append((token.idx, token.idx + len(token)))

                data_words_nonstop.append(temp_doc)
                word_spans.append(temp_span)

            filtered_datawords_nonstop = [[''.join(char for char in tok if char.isalpha(
            ) or char.isspace()) for tok in doc] for doc in data_words_nonstop]

            return filtered_datawords_nonstop

        processed = preprocess_holdout(docs)
        processed = [' '.join(doc) for doc in processed]

        self._logger.info(f"-- -- Preprocessed holdout data: {processed}")

        return processed, docs

    def get_coherence(
        self,
        ref_text: List[List[str]],
        keys: List[str],
        metric='c_npmi',
        all=False
    ) -> Union[float, Dict]:
        """
        Calculate the coherence score for the topic model

        Parameters
        ----------
        ref_text : List[List[str]]
            List of tokenized text on which the coherence score is calculated
        keys : List[str]
            List of keywords for each topic
        metric : str, optional
            The coherence metric to use, by default 'c_npmi'

        Returns
        -------
        Union[float, Dict]
            Coherence score for the topic model
            If all is True, return a dictionary of coherence scores for all metrics. Otherwise, return the coherence score for the specified metric.
        """

        '''
        Filter out the unseen texts
        '''
        def filter_unseen_tokens(keys, dictionary):
            """Filter out tokens from each topic in `keys` that aren't in the `dictionary`."""
            return [[word for word in topic if word in dictionary.token2id] for topic in keys]

        def get_score(ref_text, keys, metric):
            dictionary = Dictionary(ref_text)
            filtered_keys = filter_unseen_tokens(keys, dictionary)
            coherence_model = CoherenceModel(
                topics=filtered_keys,
                texts=ref_text,
                dictionary=dictionary,
                coherence=metric
            )
            try:
                return coherence_model.get_coherence()
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

        if all:
            return {
                'c_v': get_score(ref_text, keys, 'c_v'),
                'c_uci': get_score(ref_text, keys, 'c_uci'),
                'c_npmi': get_score(ref_text, keys, 'c_npmi')
            }
        else:
            return get_score(ref_text, keys, metric)

    def get_bow(
        self
    ):
        """
        Get the Bag of Words matrix of the documents keeping the internal order of the words as in the betas matrix.

        Assumes the train_data is already loaded (self.train_data) and the vocabulary is already created (self.vocab).
        """

        if self.train_data is None or self.vocab is None:
            self._logger.error(
                f"-- -- Train data or vocabulary not loaded. Cannot create BoW matrix.")
            return

        #####################
        # TM's vocab mappings
        #####################
        # (word -> id) and (id -> word)
        vocab_w2id = {}
        vocab_id2w = {}
        for id_wd, wd in enumerate(self.vocab):
            vocab_w2id[wd] = id_wd
            vocab_id2w[str(id_wd)] = wd

        ###################
        # Gensim BoW
        ###################
        # Create a Gensim dictionary
        gensimDict = Dictionary()

        # Create Gensim dictionary
        gensimDict = corpora.Dictionary(self.train_data)

        # Get Gensim BoW
        bow = [gensimDict.doc2bow(doc) for doc in self.train_data]

        ###################
        # Create BoW with the same order as in the betas batrix
        ###################
        # Dicionary to map Gensim IDs to TM IDs
        gensim_to_tmt_ids = {word_tuple[0]: (vocab_w2id[gensimDict[word_tuple[0]]] if gensimDict[word_tuple[0]] in vocab_w2id.keys(
        ) else None) for doc in bow for word_tuple in doc}
        gensim_to_tmt_ids = {
            key: value for key, value in gensim_to_tmt_ids.items() if value is not None}

        sorted_bow = []
        for doc in bow:
            new_doc = []
            for gensim_word_id, weight in doc:
                if gensim_word_id in gensim_to_tmt_ids.keys():
                    new_doc.append((gensim_to_tmt_ids[gensim_word_id], weight))
            new_doc = sorted(new_doc, key=lambda x: x[0])
            sorted_bow.append(new_doc)

        # Create the bow matrix with the sorted data
        bow_mat = np.zeros((len(sorted_bow), len(self.vocab)), dtype=np.int32)
        _ = [[np.put(bow_mat[doc_id], word_id, weight) for word_id,
              weight in doc] for doc_id, doc in enumerate(sorted_bow)]

        self._logger.info(f"-- -- BoW matrix shape: {bow_mat.shape}")

        return bow_mat

    def save_results(
        self,
        thetas: np.ndarray,
        betas: np.ndarray,
        train_data: np.ndarray,
        cohrs: Dict,
        topics: Dict,
        bow_mat: np.ndarray,
        vocab: List[str],
        save_model: bool = True,
        second_topics: bool = False,
    ):
        """Create a dictionary of results and save it to a file.

        Parameters
        ----------
        thetas : np.ndarray
            Document-topic distribution.
        betas : np.ndarray
            Word-topic distribution.
        train_data : List[List[str]]
            List of tokenized text data used to train the model.
        cohrs : Dict
            Dictionary with keys the coherence metrics and values the coherence scores. The coherence metrics are 'c_v', 'c_uci', 'c_npmi'.
        topics : Dict
            Dictionary with keys the topic ids and values the list of keywords for each topic.
        bow_mat : np.ndarray
            Bag of Words matrix of the documents keeping the internal order of the words as in the betas matrix.
        save_model : bool, optional
            If True, save the model object to a file, by default True
        second_topics : Dict
            Dictionary with keys the topic ids and values the list of keywords for each 2-level topic.
        """

        # Save the results to a dictionary
        coherence_values = {
            "coherence_cv": cohrs.get("c_v") if cohrs else None,
            "coherence_uci": cohrs.get("c_uci") if cohrs else None,
            "coherence_npmi": cohrs.get("c_npmi") if cohrs else None,
        }

        results = {
            key: value for key, value in {
                "num_topics": self.num_topics,
                "thetas": thetas,
                "betas": betas,
                "train_data": train_data,
                "topics": topics,
                "second_topics": second_topics if second_topics else None,
                "bow_mat": bow_mat,
                "vocab": vocab,
                **coherence_values
            }.items() if value is not None
        }

        # Save the results to a file
        pickler(self.results_path, results)

        if save_model:
            try:
                # Save model object to a file
                pickler(self.model_path, self.model)
            except:
                self._logger.info(
                    f"-- -- Using Tomotopy model. Saving model files...")
                self.model_path = self.save_path / "model.bin"
                self.model.save(self.model_path.as_posix())

        # Set the load_model flag to True
        self.load_model = True

        return

    def load(self):
        """
        Load the model from the specified model_path

        Parameters
        ----------
        model_path : str
            Path to the model to be loaded.
        """

        model_type = type(self).__name__
        self._logger.info(
            f"-- -- Loading {model_type} model from {self.results_path}...")

        self.loaded_data = unpickler(self.results_path)

        self.num_topics = self.loaded_data.get("num_topics")
        self.thetas = self.loaded_data.get("thetas")
        self.betas = self.loaded_data.get("betas")
        self.topics = self.loaded_data.get("topics")
        self.second_topics = self.loaded_data.get("second_topics")
        self.train_data = self.loaded_data.get('train_data')
        self.coherence_cv = self.loaded_data.get('coherence_cv')
        self.coherence_uci = self.loaded_data.get('coherence_uci')
        self.coherence_npmi = self.loaded_data.get('coherence_npmi')
        self.vocab = self.loaded_data.get('vocab')

        # self.bow_mat = self.loaded_data['bow_mat']

        if not self.model_path.is_file():
            if model_type == "MalletLdaModel" or model_type == "TopicGPTModel":
                self._logger.info(
                    f"-- -- Dealing with {model_type}: model_path is not a file. Getting model information through the 'modelFiles' folder...")
                self.model_folder = self.save_path / "modelFiles"
                self._logger.info(
                    f"-- -- Model loaded from {self.model_folder}")
            else:
                self._logger.error(
                    f"-- -- Dealing with {model_type}: but model_path is not a file. Check you are creating an instance of the proper type...")
        else:
            try:
                self.model = unpickler(self.model_path)
            except:
                if model_type != "TomotopyLdaModel":
                    self._logger.error(
                        f"-- -- Dealing with {model_type}: but model_path is a 'bin' file. Check you are creating an instance of the proper type...")
                if self.model_path.is_file():
                    os.remove(self.model_path)
                self.model_path = self.save_path / "model.bin"
                self.model = tp.LDAModel.load(self.model_path.as_posix())

            self._logger.info(f"-- -- Model loaded from {self.model_path}")

        return

    def concatenate_features(self, doc_topic_probas, features):
        """Concatenate the topic probability distribution features with the features from the classifier
        """
        return hstack([features, csr_matrix(doc_topic_probas).astype(np.float64)], format='csr')


class MalletLdaModel(TopicModel):

    def __init__(
        self,
        num_topics: int = 35,
        alpha: float = 5.0,
        optimize_interval: int = 10,
        num_threads: int = 4,
        num_iters: int = 1000,
        doc_topic_thr: float = 0.0,
        token_regexp: str = "[\p{L}\p{N}][\p{L}\p{N}\p{P}]*\p{L}",
        mallet_path: str = "src/topicmodeling/Mallet-202108/bin/mallet",
        topn: int = 15,
        load_data_path: str = "data/source/cordis_preprocessed.json",
        load_model: bool = False,
        model_path: str = None,
        logger: logging.Logger = None,
    ) -> None:

        # Initialize logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('MalletLdaModel')

        # Initialize the TopicModel class
        super().__init__(
            num_topics, topn, load_data_path, load_model, model_path, self._logger)

        # Initialize specific parameters for Mallet LDA
        self.mallet_path = pathlib.Path(mallet_path)
        self.alpha = alpha
        self.optimize_interval = optimize_interval
        self.num_threads = num_threads
        self.num_iterations = num_iters
        self.doc_topic_thr = doc_topic_thr
        self.token_regexp = token_regexp

        if not self.mallet_path.is_file():
            self._logger.error(
                f'-- -- Provided mallet path is not valid -- Stop')
            sys.exit()

    def _extract_pipe(self):
        """
        Creates a pipe based on a small amount of the training data to ensure that the holdout data that may be later inferred is compatible with the training data
        """

        # Get corpus file
        path_corpus = self.model_folder / "corpus.mallet"
        if not path_corpus.is_file():
            self._logger.error(
                f"-- Pipe extraction: Could not locate corpus file")
            return

        # Create auxiliary file with only first line from the original corpus file
        path_txt = self.model_folder / "corpus.txt"
        with path_txt.open('r', encoding='utf8') as f:
            first_line = f.readline()
        path_aux = self.model_folder / "corpus_aux.txt"
        with path_aux.open('w', encoding='utf8') as fout:
            fout.write(first_line + '\n')

        # We perform the import with the only goal to keep a small file containing the pipe
        self._logger.info(f"-- Extracting pipeline")
        path_pipe = self.model_folder / "import.pipe"

        cmd = self.mallet_path.as_posix() + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_corpus, path_aux, path_pipe)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- Failed to extract pipeline. Revise command')

        # Remove auxiliary file
        path_aux.unlink()

        return

    def train(
        self,
        further_proc: bool = True,
        stops_path: str = "src/topicmodeling/data/stops",
        eqs_path: str = "src/topicmodeling/data/equivalences"
    ) -> None:
        """
        Train the topic model and save the data to save_data_path
        """

        # Call the train method from the parent class to load the data and initialize the save_path
        super().train(further_proc=further_proc, stops_path=stops_path, eqs_path=eqs_path)

        # Create folder for saving Mallet output files
        self.model_folder = self.save_path / "modelFiles"
        self.model_folder.mkdir(exist_ok=True)

        # Transform training data into the format expected by Mallet (txt file)
        self._logger.info(f"-- -- Creating Mallet corpus.txt...")
        corpus_txt_path = self.model_folder / "corpus.txt"
        corpus_raw_path = self.model_folder / "corpus_raw.txt"
        with corpus_txt_path.open("w", encoding="utf8") as fout:
            for i, t in enumerate(self.df.lemmas):
                fout.write(f"{i} 0 {t}\n")
        with corpus_raw_path.open("w", encoding="utf8") as fout:
            for i, t in enumerate(self.df.raw_text):
                fout.write(f"{i} 0 {t}\n")
        self._logger.info(f"-- -- Mallet corpus.txt created.")

        # Import data to Mallet
        self._logger.info(f"-- -- Importing data to Mallet...")
        corpus_mallet = self.model_folder / "corpus.mallet"

        cmd = self.mallet_path.as_posix() + \
            ' import-file --preserve-case --keep-sequence ' + \
            '--remove-stopwords --token-regex "' + self.token_regexp + \
            '" --input %s --output %s'
        cmd = cmd % (corpus_txt_path, corpus_mallet)
        
        self._logger.info(f"-- -- Command to be run {cmd}")

        try:
            self._logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- -- Mallet failed to import data. Revise command')
        self._logger.info(f"-- -- Data imported to Mallet.")

        # Actual training of the model
        config_mallet = self.model_folder / "config.mallet"
        with config_mallet.open('w', encoding='utf8') as fout:
            fout.write('input = ' + corpus_mallet.resolve().as_posix() + '\n')
            fout.write('num-topics = ' + str(self.num_topics) + '\n')
            fout.write('alpha = ' + str(self.alpha) + '\n')
            fout.write('optimize-interval = ' +
                       str(self.optimize_interval) + '\n')
            fout.write('num-threads = ' + str(self.num_threads) + '\n')
            fout.write('num-iterations = ' + str(self.num_iterations) + '\n')
            fout.write('doc-topics-threshold = ' +
                       str(self.doc_topic_thr) + '\n')
            fout.write('output-state = ' +
                       self.model_folder.joinpath('topic-state.gz').resolve().as_posix() + '\n')
            fout.write('output-doc-topics = ' +
                       self.model_folder.joinpath('doc-topics.txt').resolve().as_posix() + '\n')
            fout.write('word-topic-counts-file = ' +
                       self.model_folder.joinpath('word-topic-counts.txt').resolve().as_posix() + '\n')
            fout.write('diagnostics-file = ' +
                       self.model_folder.joinpath('diagnostics.xml ').resolve().as_posix() + '\n')
            fout.write('xml-topic-report = ' +
                       self.model_folder.joinpath('topic-report.xml').resolve().as_posix() + '\n')
            fout.write('output-topic-keys = ' +
                       self.model_folder.joinpath('topickeys.txt').resolve().as_posix() + '\n')
            fout.write('inferencer-filename = ' +
                       self.model_folder.joinpath('inferencer.mallet').resolve().as_posix() + '\n')

        cmd = str(self.mallet_path) + \
            ' train-topics --config ' + str(config_mallet)

        try:
            self._logger.info(
                f'-- -- Training mallet topic model. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Model training failed. Revise command')
            return

        # Calculate evaluation metrics
        self._logger.info(
            f"-- -- Calculating evaluation metrics..."
        )
        betas = self.get_betas()
        topics = self.print_topics(verbose=False)
        topics_ = [topics[k] for k in topics.keys()]
        metrics = {
                'c_v': 0,
                'c_uci': 0,
                'c_npmi': 0
            } #self.get_coherence(self.train_data, topics_, all=True)
        
        """
        
        """
        
        self._logger.info(
            f"-- -- Coherence metrics: {metrics}"
        )

        # Calculate distributions
        self._logger.info(
            f"-- -- Calculating thetas..."
        )

        thetas = self.get_thetas()

        # Calculate bow
        bow_mat = self.get_bow()

        # Save the model data
        self.save_results(
            thetas, betas, self.train_data, metrics, topics, bow_mat, self.vocab, save_model=False)

        # Extract pipe for later inference
        self._extract_pipe()

        if (self.save_path / "model.pkl").is_file():
            os.remove(self.save_path / "model.pkl")
        if (self.save_path / "model.bin").is_file():
            os.remove(self.save_path / "model.bin")

        return

    def infer(
        self,
        docs: List[str],
        num_iterations: int = 1000,
        doc_topic_thr: float = 0.0,
    ) -> np.ndarray:
        """Perform inference on unseen documents.

        Parameters
        ----------
        docs : List[str]
            List of documents to perform inference on.
        num_iterations : int, optional
            Number of iterations for the inference, by default 1000.
        doc_topic_thr : float, optional
            Document-topic threshold, by default 0.0.

        Returns
        -------
        np.ndarray
            Array of inferred thetas
        """

        docs, _ = super().infer(docs)

        # Aadd importation pipe
        path_pipe = self.model_folder / "import.pipe"

        # Create folder for saving Mallet output files
        self.inference_folder = self.model_folder / "inference"
        self.inference_folder.mkdir(exist_ok=True)

        # Transform training data into the format expected by Mallet (txt file)
        self._logger.info(f"-- -- Creating Mallet inference corpus.txt...")
        holdout_corpus = self.inference_folder / "corpus.txt"
        with holdout_corpus.open("w", encoding="utf8") as fout:
            for i, t in enumerate(docs):
                fout.write(f"{i} 0 {t}\n")
        self._logger.info(f"-- -- Mallet corpus.txt for inference created.")

        # Get inferencer
        inferencer = self.model_folder / "inferencer.mallet"

        # Files to be generated thoruogh Mallet
        corpus_mallet_inf = self.inference_folder / "corpus_inf.mallet"
        doc_topics_file = self.inference_folder / "doc-topics-inf.txt"

        # Import data to mallet
        self._logger.info('-- Inference: Mallet Data Import')

        #
        cmd = self.mallet_path.as_posix() + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_pipe, holdout_corpus, corpus_mallet_inf)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Mallet failed to import data. Revise command')
            return

        # Get topic proportions
        self._logger.info('-- Inference: Inferring Topic Proportions')

        cmd = self.mallet_path.as_posix() + \
            ' infer-topics --inferencer %s --input %s --output-doc-topics %s ' + \
            ' --doc-topics-threshold ' + str(doc_topic_thr) + \
            ' --num-iterations ' + str(num_iterations)
        cmd = cmd % (inferencer, corpus_mallet_inf, doc_topics_file)

        try:
            self._logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- Mallet inference failed. Revise command')
            return

        self._logger.info(f"-- -- Inference completed. Loading thetas...")

        # Get inferred thetas
        cols = [k for k in np.arange(2, self.num_topics + 2)]
        thetas32 = np.loadtxt(doc_topics_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)

        self._logger.info(f"-- -- Inferred thetas shape {thetas32.shape}")

        return thetas32

    def print_topics(
        self,
        verbose=False,
        top_k: int = 15,
        tfidf: bool = False
    ) -> dict:
        """
        Print the list of topics for the topic model

        Parameters
        ----------
        verbose : bool, optional
            If True, print the topics to the console, by default False

        Returns
        -------
        dict
            Dictionary of topics and their keywords
        """

        if not self.load_model:
            
            self.topics = dict()
            
            if tfidf:
                # Calculate betas with downscoring ( Emphasizes words appearing less frequently in topics)
                self.betas_ds = np.copy(self.betas)
                if np.min(self.betas_ds) < 1e-12:
                    self.betas_ds += 1e-12
                deno = np.reshape((sum(np.log(self.betas_ds)) /
                                self.num_topics), (len(self.betas.T), 1))
                deno = np.ones((self.num_topics, 1)).dot(deno.T)
                self.betas_ds = self.betas_ds * (np.log(self.betas_ds) - deno)
                
                for k in range(self.num_topics):
                    words = [
                        self.vocab[w]
                        for w in np.argsort(self.betas_ds[k])[::-1]
                    ]
                    self.topics[k] = words
                    
            else:
                for k in range(self.num_topics):
                    words = [
                        self.vocab[w]
                        for w in np.argsort(self.betas[k])[::-1]
                    ]
                    self.topics[k] = words
        if verbose:
            [print(f"Topic {k}: {v}") for k, v in self.topics.items()]
        
        return self.topics #{k: v[:top_k] for k, v in self.topics.items()}

    def get_thetas(self):
        
        if not self.load_model:
            # Get thetas from Mallet output
            # thetas = 'numpy.ndarray' of shape(D, T)
            self._logger.info(
                f"-- -- Loading thetas from {self.model_folder.joinpath('doc-topics.txt')}")
            thetas_file = self.model_folder.joinpath('doc-topics.txt')
            cols = [k for k in np.arange(2, self.num_topics + 2)]
            self.thetas = np.loadtxt(thetas_file, delimiter='\t',
                                dtype=np.float32, usecols=cols)
            
            
        return self.thetas
        
    def get_betas(
        self
    ) -> np.ndarray:
        """
        Calculate the word-topic distribution (K x V matrix) for the topic model. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.

        Returns
        -------
        Dict
            Dictionary with keys the words and values a list of probabilities of the word belonging to topics.
        """

        if not self.load_model:
            # Create vocabulary files and calculate beta matrix
            wtcFile = self.model_folder.joinpath('word-topic-counts.txt')
            vocab_size = file_lines(wtcFile)
            betas = np.zeros((self.num_topics, vocab_size))
            vocab = []
            term_freq = np.zeros((vocab_size,))
            with wtcFile.open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    elements = line.split()
                    vocab.append(elements[1])
                    for counts in elements[2:]:
                        tpc = int(counts.split(':')[0])
                        cnt = int(counts.split(':')[1])
                        betas[tpc, i] += cnt
                        term_freq[i] += cnt
            betas = normalize(betas, axis=1, norm='l1')

            # Save for later use
            self.betas = betas
            self.vocab = vocab
            
        return self.betas

class CtmModel(TopicModel):

    def __init__(
        self,
        num_topics: int = 35,
        num_iters: int = 250,
        topn: int = 15,
        sbert_model: str = "paraphrase-distilroberta-base-v2",
        sbert_context: int = 768,
        load_data_path: str = "data/source/cordis_preprocessed.json",
        batch_size: int = 32,
        load_model: bool = False,
        model_path: str = None,
        logger: logging.Logger = None,
    ) -> None:

        # Initialize logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('CtmModel')

        # Initialize the TopicModel class
        super().__init__(
            num_topics, topn, load_data_path, load_model, model_path, self._logger)

        # Initialize specific parameters for CTM
        self.num_iters = num_iters
        self.sbert_model = sbert_model
        self.batch_size = batch_size
        self.sbert_context = sbert_context

    def train(
        self,
        further_proc: bool = True,
        stops_path: str = "src/topicmodeling/data/stops",
        eqs_path: str = "src/topicmodeling/data/equivalences"
    ) -> None:
        """
        Train the topic model and save the data to save_data_path
        """

        # Call the train method from the parent class to load the data and initialize the save_path
        super().train(get_embeddings=True, further_proc=further_proc, stops_path=stops_path, eqs_path=eqs_path)


        self._logger.info(
            f"-- -- Creating TopicModelDataPreparation object and fitting the data..."
        )
        qt = TopicModelDataPreparation(self.sbert_model)
        self.training_dataset = qt.fit(
            text_for_contextual=self.df.raw_text,
            text_for_bow=self.df.lemmas,
            # if self.embeddings is not None no contextual model to generate the embeddings is used, rather the embeddings are provided
            custom_embeddings=self.embeddings
        )
        self.vocab = list(self.training_dataset.idx2token.values())

        self._logger.info(
            f"-- -- Creating CombinedTM object...")
        self.model = CombinedTM(
            bow_size=len(qt.vocab),
            contextual_size=self.sbert_context,
            n_components=self.num_topics,
            num_epochs=self.num_iters
        )
        self.model.qt = qt

        self._logger.info(
            f"-- -- Training the CTM model...")
        self.model.fit(self.training_dataset)

        # Calculate evaluation metrics
        self._logger.info(
            f"-- -- Calculating evaluation metrics..."
        )
        topics = self.print_topics(verbose=False)
        topics_ = [topics[k] for k in topics.keys()]
        metrics = self.get_coherence(self.train_data, topics_, all=True)
        self._logger.info(
            f"-- -- Coherence metrics: {metrics}"
        )

        # Calculate distributions
        self._logger.info(
            f"-- -- Calculating topics and distributions..."
        )
        wdt = self.get_betas()
        thetas = self.get_thetas()

        # Calculate bow
        bow_mat = self.get_bow()

        # Save the model data
        self.save_results(
            thetas, wdt, self.train_data, metrics, topics, bow_mat, self.vocab)

        return

    def _prepare_hold_out_dataset(
        self,
        docs,
        embs
    ):
        """It prepares the holdout data in the format that is asked as input in CTM, based on the TopicModelDataPreparation object generated for the training dataset

        Parameters
        ----------
        docs: List[str]
            List of hold-out documents

        Returns
        -------
        ho_data: CTMDataset
            Holdout dataset in the required format for CTM
        """

        # Load the TopicModelDataPreparation object
        ho_data = self.model.qt.transform(
            text_for_bow=docs,
            text_for_contextual=docs,
            custom_embeddings=embs)

        return ho_data

    def _bert_embeddings_from_list(self, texts):
        """
        Creates SBERT Embeddings from a list
        """
        model = SentenceTransformer(self.sbert_model)

        return np.array(model.encode(texts, show_progress_bar=True, batch_size=self.batch_size))

    def infer(
        self,
        docs: List[str],
    ) -> np.ndarray:
        """Perform inference on unseen documents.

        Parameters
        ----------
        docs : List[str]
            List of documents to perform inference on.

        Returns
        -------
        np.ndarray
            Array of inferred thetas
        """

        docs_processed, docs_raw = super().infer(docs)

        # Generating holdout corpus in the input format required by CTM
        self._logger.info(f"-- -- Inference: Preparing holdout embeddings...")
        ho_embs = self._bert_embeddings_from_list(docs_raw)

        self._logger.info(f"-- -- Inference: Preparing holdout dataset...")
        ho_data = self._prepare_hold_out_dataset(
            docs=docs_processed, embs=ho_embs)

        # Get inferred thetas matrix
        self._logger.info(
            '-- -- Inference: Getting inferred thetas matrix')
        freeze_support()
        self.model.num_data_loader_workers = 0
        thetas = self.model.get_doc_topic_distribution(ho_data)
        thetas32 = np.asarray(thetas)
        self._logger.info(f"-- -- Inferred thetas shape {thetas32.shape}")

        return thetas32

    def print_topics(
        self,
        verbose=False,
        top_k: int = 15
    ) -> dict:
        """
        Print the list of topics for the topic model

        Parameters
        ----------
        verbose : bool, optional
            If True, print the topics to the console, by default False

        Returns
        -------
        dict
            Dictionary of topics and their keywords
        """

        if not self.load_model:
            topics = self.model.get_topics(top_k)
            self.topics = {k: [v for v in topics[k]] for k in topics.keys()}

        if verbose:
            [print(f"Topic {k}: {v}") for k, v in self.topics.items()]
        return self.topics

    def get_thetas(
        self,
    ) -> np.ndarray:
        """
        Calculate document-topic distributions. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.

        Returns
        -------
        np.ndarray
            A D x K matrix containing the topic distributions for all input documents with D being the documents and K the topics.
        """

        if not self.load_model:

            # Get document-topic distribution as D x T numpy array
            self.thetas = self.model.get_doc_topic_distribution(
                self.training_dataset)  # DxT

        return self.thetas

    def get_betas(
        self
    ) -> np.ndarray:
        """
        Calculate the word-topic distribution. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.

        Returns
        -------
        Dict
            Dictionary with keys the words and values a list of probabilities of the word belonging to topics.
        """

        if not self.load_model:
            self.betas = self.model.get_topic_word_distribution().T
            
        return self.betas


class BERTopicModel(TopicModel):

    def __init__(
        self,
        num_topics: int = 35,
        topn: int = 15,
        sbert_model: str = "paraphrase-distilroberta-base-v2",
        stopwords: List[str] = list(STOP_WORDS),
        no_below: int = 1,
        no_above: float = 1,
        umap_n_components=5,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        umap_metric='cosine',
        hdbscan_min_cluster_size=10,
        hdbscan_metric='euclidean',
        hdbscan_cluster_selection_method='eom',
        hbdsan_prediction_data=True,
        load_data_path: str = "data/source/cordis_preprocessed.json",
        load_model: bool = False,
        model_path: str = None,
        logger: logging.Logger = None,
    ) -> None:

        # Initialize logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('BERTopicModel')

        # Initialize the TopicModel class
        super().__init__(
            num_topics, topn, load_data_path, load_model, model_path, self._logger)

        # Initialize specific parameters for BERTopic
        self.sbert_model = sbert_model
        self.stopwords = stopwords
        self.no_below = no_below
        self.no_above = no_above
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method
        self.hbdsan_prediction_data = hbdsan_prediction_data

        word_min_len = 2
        self.word_pattern = (
            f"(?<![a-zA-Z\u00C0-\u024F\d\-\_])"
            f"[a-zA-Z\u00C0-\u024F]"
            f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[\-\_\·\.'](?![\-\_\·\.'])){{{word_min_len - 1},}}"
            f"(?<![\-\_\·\.'])[a-zA-Z\u00C0-\u024F\d]?"
            f"(?![a-zA-Z\u00C0-\u024F\d])"
        )

        return

    def train(
        self,
        further_proc: bool = True,
        stops_path: str = "src/topicmodeling/data/stops",
        eqs_path: str = "src/topicmodeling/data/equivalences"
    ) -> None:
        """
        Train the topic model and save the data to save_data_path
        """

        # Call the train method from the parent class to load the data and initialize the save_path
        super().train(get_embeddings=True, further_proc=further_proc, stops_path=stops_path, eqs_path=eqs_path)
        
        # Put components together to create BERTopic model
        # STEP 0 : Embedding model
        if self.embeddings is not None:
            self._logger.info(
                f"-- -- Using pre-trained embeddings from the dataset..."
            )
            self._embedding_model = None
        else:
            self._logger.info(
                f"-- -- Creating SentenceTransformer model with {self.sbert_model}..."
            )
            self._embedding_model = SentenceTransformer(
                self.sbert_model
            )
        # STEP 1: Reduce dimensionality of embeddings
        self._umap_model = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric
        )

        # STEP 2: Cluster reduced embeddings
        self._hdbscan_model = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            prediction_data=self.hbdsan_prediction_data
        )

        # STEP 3: Tokenize topics
        self._vectorizer_model = CountVectorizer(
            token_pattern=self.word_pattern,
            stop_words=self.stopwords,
            # max_df=self.no_below,
            # min_df=self.no_above,
        )

        # STEP 4: Create topic representation
        self._ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        # STEP 5 (Optional): Add additional representations
        self._representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(
                diversity=0.3,
                top_n_words=15
            )
        }

        self.model = BERTopic(
            language="multilingual",
            min_topic_size=1,
            nr_topics=self.num_topics+1,
            low_memory=False,
            # calculate_probabilities=True,
            embedding_model=self._embedding_model,
            # umap_model=KMeans(n_clusters=self.num_topics),#self._umap_model,
            hdbscan_model=self._hdbscan_model,
            vectorizer_model=self._vectorizer_model,
            # ctfidf_model=self._ctfidf_model,
            # representation_model=self._representation_model,
            verbose=True
        )

        # Train model
        texts = self.df.raw_text.values.tolist()

        # probs = The probability of the assigned topic per document.
        # If `calculate_probabilities` in BERTopic is set to True, then
        # it calculates the probabilities of all topics across all documents
        # instead of only the assigned topic.
        if self.embeddings is not None:
            _, probs = self.model.fit_transform(texts, self.embeddings)
        else:
            _, probs = self.model.fit_transform(texts)

        # Get vocabulary of the model
        self.vocab = self.model.vectorizer_model.get_feature_names_out()
    
        # Calculate evaluation metrics
        self._logger.info(
            f"-- -- Calculating evaluation metrics..."
        )
        wdt = self.get_betas()
        topics = self.print_topics(verbose=False)
        topics_ = [topics[k] for k in topics.keys()]
        metrics = self.get_coherence(self.train_data, topics_, all=True)
        self._logger.info(
            f"-- -- Coherence metrics: {metrics}"
        )

        # Calculate distributions
        self._logger.info(
            f"-- -- Calculating topics and distributions..."
        )
        
        self.thetas = self.get_thetas(texts)

        # Calculate bow
        bow_mat = self.get_bow()

        # Save the model data
        self.save_results(
            self.thetas, wdt, self.train_data, metrics, topics, bow_mat, self.vocab)

        return

    def infer(
        self,
        docs: List[str],
    ) -> np.ndarray:
        """Perform inference on unseen documents.

        Parameters
        ----------
        docs : List[str]
            List of documents to perform inference on.

        Returns
        -------
        np.ndarray
            Array of inferred thetas
        """

        self._logger.info(
            '-- -- Inference: Getting inferred thetas matrix')
        # Get inferred thetas matrix
        thetas, _ = self.model.approximate_distribution(docs)

        self._logger.info(f"-- -- Inferred thetas shape {thetas.shape}")

        return thetas

    def print_topics(
        self,
        verbose=False,
        top_k: int = 15
    ) -> dict:
        """
        Print the list of topics for the topic model

        Parameters
        ----------
        verbose : bool, optional
            If True, print the topics to the console, by default False

        Returns
        -------
        dict
            Dictionary of topics and their keywords
        """

        if not self.load_model:

            self.topics = dict()
            for k, v in self.model.get_topics().items():
                self.topics[k] = [el[0] for el in v][:top_k]

        if verbose:
            [print(f"Topic {k}: {v}") for k, v in self.topics.items()]
        return self.topics

    def get_thetas(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Calculate document-topic distribtuion. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.

        Parameters
        ----------
        texts : List[str]
            List of documents to calculate the topic distribution for.

        Returns
        -------
        np.ndarray
            A Dx K matrix containing the topic distributions for all input documents with D being the documents and K the topics.
        """

        if not self.load_model:

            self._logger.info(
                f"-- -- Calculating thetas...")

            # topic_distr = D x K matrix containing the topic distributions
            # for all input documents.
            thetas_approx, _ = self.model.approximate_distribution(texts)
            check_thetas = [(doc_id, thetas_approx[doc_id].shape) for doc_id in range(
                len(thetas_approx)) if thetas_approx[doc_id].shape != (self.num_topics,)]
            if len(check_thetas) > 0:
                self._logger.warning(
                    f"-- -- No all the thetas have the same shape: {check_thetas}")
            self.thetas = thetas_approx
            
            return self.thetas    
            

    def get_betas(
        self
    ) -> np.ndarray:
        """
        Calculate the word-topic distribution. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.

        Returns
        -------
        np.ndarray
            Array of word-topic distribution
        """

        if not self.load_model:
            self.betas = self.model.c_tf_idf_.toarray()

        return self.betas


class TopicGPTModel(TopicModel):

    def __init__(
        self,
        api_key: str = None,
        num_topics: int = 35,
        topn: int = 15,
        load_data_path: str = "data/source/cordis_preprocessed.json",
        load_model: bool = False,
        model_path: str = None,
        sample: float = 0.001,
        deployment_name1: str = "gpt-4",
        deployment_name2: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        top_p: float = 0.0,
        max_tokens_gen1: int = 300,
        max_tokens_gen2: int = 500,
        max_tokens_assign: int = 300,
        refined_again: bool = False,
        remove: bool = False,
        do_second_level=False,
        verbose: bool = True,
        logger: logging.Logger = None,
    ) -> None:

        # Initialize logger
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('BERTopicModel')

        # Initialize the TopicModel class
        super().__init__(
            num_topics, topn, load_data_path, load_model, model_path, self._logger)

        if not load_model:
            # Load OPEN AI API Key
            try:

                openai.api_key = api_key
                self._logger.info(
                    "-- -- OpenAI API key loaded and set successfully.")
            except Exception as e:
                self._logger.error(f'-- -- Error loading OpenAI API key: {e}')
                return

        # Initialize specific parameters for TopicGPT
        self._p_scripts = pathlib.Path(
            os.getcwd()) / "src/topicmodeling/topicGPT/script"
        self._p_prompts = pathlib.Path(
            os.getcwd()) / "src/topicmodeling/topicGPT/prompt"
        self._sample = sample

        # Generation I/O
        self._generation_prompt = self._p_prompts / "generation_1.txt"
        self._seed_1 = self._p_prompts / "seed_1.md"

        # Refinement I/O
        self._refinement_prompt = self._p_prompts / "refinement.txt"

        # Generation 2 I/O
        self._generation_2_prompt = self._p_prompts / "generation_2.txt"

        # Assignment I/O
        self._assignment_prompt = self._p_prompts / "assignment.txt"

        # Correction I/O
        self._correction_prompt = self._p_prompts / "correction.txt"

        self._outputs_save = {
            "generation_out": "generation_1.jsonl",
            "generation_topic": "generation_1.md",
            "refinement_out": "refinement.jsonl",
            "refinement_topic": "refinement.md",
            "refinement_mapping": "refinement_mapping.txt",
            "refinement_updated": "refinement_updated.jsonl",
            "generation_2_out": "generation_2.jsonl",
            "generation_2_topic": "generation_2.md",
            "assignment_out": "assignment.jsonl",
            "correction_out": "assignment_corrected.jsonl"
        }

        self._deployment_name1 = deployment_name1
        self._deployment_name2 = deployment_name2
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens_gen1 = max_tokens_gen1
        self._max_tokens_gen2 = max_tokens_gen2
        self._max_tokens_assign = max_tokens_assign
        self._refined_again = refined_again
        self._remove = remove
        self._do_second_level = do_second_level
        self._verbose = verbose

        self.load_data_path = load_data_path

        return

    def train(
        self,
        further_proc: bool = True,
        stops_path: str = "src/topicmodeling/data/stops",
        eqs_path: str = "src/topicmodeling/data/equivalences"
    ) -> None:
        """
        Train the topic model and save the data to save_data_path
        """

        # Call the train method from the parent class to load the data and initialize the save_path
        super().train(get_preprocessed=False, further_proc=further_proc, stops_path=stops_path, eqs_path=eqs_path)

        # Create folder for saving Mallet output files
        self.model_folder = self.save_path / "modelFiles"
        self.model_folder.mkdir(exist_ok=True)
        self._outputs_save = {key: self.model_folder /
                              value for key, value in self._outputs_save.items()}

        # Save subsample if specified
        df_sample = self.df.copy()
        df_sample["text"] = df_sample["raw_text"]
        if self._sample:
            if isinstance(self._sample, float):
                df_sample = df_sample.sample(frac=self._sample)
            elif isinstance(self._sample, int):
                num_samples = min(self._sample, len(df_sample))
                df_sample = df_sample.sample(n=num_samples)
            else:
                self._logger.error(
                    f"-- -- The sample must be either a float or an int, but {type(self._sample)} was provided...")
                return

            path_sample = self.model_folder / \
                f"sample_{str(self._sample)}.json"

            self._logger.info(
                f"-- -- Training model with a sample size of {self._sample}...")
        else:
            path_sample = self.model_folder / f"sample.jsonl"
        df_sample.to_json(path_sample, lines=True, orient="records")

        #####################
        # TOPIC GENERATION  #
        #####################
        cmd = f"python3 {self._p_scripts.joinpath('generation_1.py').as_posix()}" + \
            ' --deployment_name %s --max_tokens %s --temperature %s --top_p %s --data %s --prompt_file %s --seed_file %s --out_file %s --topic_file %s --verbose %s'
        cmd = cmd % (self._deployment_name1, self._max_tokens_gen1, self._temperature, self._top_p, path_sample, self._generation_prompt,
                     self._seed_1, self._outputs_save["generation_out"], self._outputs_save["generation_topic"], self._verbose)

        try:
            self._logger.info(f"-- TOPIC GENERATION --")
            self._logger.info(f'-- --  Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Failed to run generation script. Revise command')

        #####################
        # TOPIC REFINEMENT  #
        #####################
        cmd = f"python3 {self._p_scripts.joinpath('refinement.py').as_posix()}" + \
            ' --deployment_name %s --max_tokens %s --temperature %s --top_p %s --prompt_file %s --generation_file %s --topic_file %s --out_file %s --verbose %s --updated_file %s --mapping_file %s --refined_again %s --remove %s'
        cmd = cmd % (
            self._deployment_name1, self._max_tokens_assign, self._temperature, self._top_p, self._refinement_prompt, self._outputs_save["generation_out"], self._outputs_save[
                "generation_topic"], self._outputs_save["refinement_topic"], self._verbose, self._outputs_save["refinement_out"], self._outputs_save["refinement_mapping"], self._refined_again, self._remove
        )
        try:
            self._logger.info(f"-- TOPIC REFINEMENT --")
            self._logger.info(f'-- --  Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Failed to run refinement script. Revise command')

        #####################
        # TOPIC ASSIGNMENT  #
        #####################
        cmd = f"python3 {self._p_scripts.joinpath('assignment.py').as_posix()}" + \
            ' --deployment_name %s --max_tokens %s --temperature %s --top_p %s --data %s --prompt_file %s --topic_file %s --out_file %s --verbose %s'
        cmd = cmd % (self._deployment_name2, self._max_tokens_assign, self._temperature, self._top_p, path_sample,
                     self._assignment_prompt, self._outputs_save["generation_topic"], self._outputs_save["assignment_out"], self._verbose)

        try:
            self._logger.info(f"-- TOPIC ASSIGNMENT --")
            self._logger.info(f'-- --  Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Failed to run assignment script. Revise command')

        #####################
        # TOPIC CORRECTION  #
        #####################
        cmd = f"python3 {self._p_scripts.joinpath('correction.py').as_posix()}" + \
            ' --deployment_name %s --max_tokens %s --temperature %s --top_p %s --data %s --prompt_file %s --topic_file %s --out_file %s --verbose %s'
        cmd = cmd % (self._deployment_name2, self._max_tokens_assign, self._temperature, self._top_p,
                     self._outputs_save["assignment_out"], self._correction_prompt, self._outputs_save["generation_topic"], self._outputs_save["correction_out"], self._verbose)

        try:
            self._logger.info(f"-- TOPIC CORRECTION --")
            self._logger.info(f'-- --  Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- Failed to run correction script. Revise command')

        second_topics = None
        if self._do_second_level:
            ###############################
            # 2nd LEVEL TOPIC GENERATION  #
            ###############################
            cmd = f"python3 {self._p_scripts.joinpath('generation_2.py').as_posix()}" + \
                ' --deployment_name %s --max_tokens %s --temperature %s --top_p %s --data %s --seed_file %s --prompt_file %s --out_file %s --topic_file %s --verbose %s'
            cmd = cmd % (self._deployment_name1, self._max_tokens_assign, self._temperature, self._top_p, self._outputs_save["generation_out"], self._outputs_save[
                         "generation_topic"], self._generation_2_prompt, self._outputs_save["generation_2_out"], self._outputs_save["generation_2_topic"], self._verbose)

            try:
                self._logger.info(f"-- 2nd LEVEL TOPIC GENERATION --")
                self._logger.info(f'-- --  Running command {cmd}')
                check_output(args=cmd, shell=True)
            except:
                self._logger.error(
                    '-- Failed to run generation script 2. Revise command')

            second_topics = self.print_topics(get_second_level=True)

        topics = self.print_topics(verbose=True)

        # Save the model data
        self.save_results(
            None, None, self.train_data, None, topics, None, None, save_model=False, second_topics=second_topics)

        return

    def infer(self) -> np.ndarray:
        """Perform inference on unseen documents.
        """

        self._logger.info(
            '-- -- TopicGPT does not suppor inference. Exiting...')

        return

    def print_topics(
        self,
        verbose=False,
        get_second_level=False
    ) -> dict:
        """
        Print the list of topics for the topic model

        Parameters
        ----------
        verbose : bool, optional
            If True, print the topics to the console, by default False

        Returns
        -------
        dict
            Dictionary of topics and their keywords
        """

        if not self.load_model:

            self.topics = {k: v for k, v in enumerate(open(
                self._outputs_save["generation_topic"], "r").readlines()) if v.strip() != ''}

            if self._do_second_level:
                self.second_topics = {k: v for k, v in enumerate(open(
                    self._outputs_save["generation_2_topic"], "r").readlines()) if v.strip() != ''}

        if verbose:
            [print(f"Topic {k}: {v}") for k, v in self.topics.items()]
            if self._do_second_level:
                [print(f"2nd-level Topic {k}: {v}")
                 for k, v in self.second_topics.items()]

        if get_second_level:
            return self.second_topics
        else:
            return self.topics

    def get_thetas(self) -> Tuple[List[np.ndarray], Dict]:
        """
        Calculate doc_prob_topic and topics_probs. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.
        """

        self._logger.info(
            '-- -- TopicGPT does not calculate document-topic distributions. Exiting...')

        return None

    def get_betas(
        self
    ) -> Dict:
        """
        Calculate the word-topic distribution. If the model is not loaded, calculate the values and save them for later use. Otherwise, return the saved values.
        """

        self._logger.info(
            '-- -- TopicGPT does not calculate word-topic distributions. Exiting...')

        return None
