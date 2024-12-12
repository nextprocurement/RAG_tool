import os
import re
import time
from dotenv import load_dotenv
import dspy
import logging
import json
import numpy as np
import pandas as pd
import pathlib
import ast
import yaml
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from typing import List, Dict, Any, Optional, Union
import ast
import itertools
from sentence_transformers.util import cos_sim
from dspy.datasets import Dataset
from dspy.evaluate import Evaluate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from src.utils.tm_utils import create_model
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
from spacy_download import load_spacy
import unicodedata
from sklearn.metrics import davies_bouldin_score

def eliminar_tildes(palabra):
    return ''.join(
        (c for c in unicodedata.normalize('NFD', palabra) if unicodedata.category(c) != 'Mn')
    )

def solo_diferencia_tilde(palabra1, palabra2):
    # Eliminar tildes y comparar
    if eliminar_tildes(palabra1) == eliminar_tildes(palabra2):
        # Si una palabra tiene tilde, devolver esa
        if palabra1 != eliminar_tildes(palabra1):
            return palabra1
        elif palabra2 != eliminar_tildes(palabra2):
            return palabra2
    return None  # Si no hay diferencia por tilde, devuelve None


#######################################################################
# TenderDataset
#######################################################################
class EquivalencesDataset(Dataset):

    def __init__(
        self,
        data_fpath: Union[pathlib.Path, str],
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        input_key: str = "words",
        seed: Optional[int] = 11235,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.labels = []
        self._train = []
        self._dev = []
        self._test = []

        # Read the training data
        train_data = pd.read_excel(data_fpath).rename(columns={
            'equivalence': 'equivalences',
            'origin': 'words'})[['words', 'equivalences']]

        train_data, temp_data = train_test_split(
            train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(input_key) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(input_key) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(input_key) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')

class Equivalent(dspy.Signature):
    """
    Unify a list of words in a single word in singular form.
    
    ----------------------------------------------------------------------------
    Examples
    --------
    WORDS: "deportes", "deportivo", "deporte", "deportista", "deportivista"
    EQ_WORD: "deporte"
    
    WORDS: "ministro", "ministerio", "ministra", "ministerios", "ministros", "ministras", "ministerial"
    EQ_WORD: "ministerio"
    """
    
    WORDS = dspy.InputField()
    EQ_WORD = dspy.OutputField()
    
class EquivalentModule(dspy.Module):
    def __init__(
        self,
    ):
        """
        Initialize the EquivalentModule, which agrupate list of words in a single word.
        """
        super().__init__()
        self.eq = dspy.Predict(Equivalent)
        
    def forward(self, words: str):
        """
        Forward pass of the module.
        """
        return self.eq(WORDS=words).EQ_WORD

# Note: The following class is not used in the final implementation-> EquivalentModule is used instead   
class Corrector(dspy.Signature):
    """
    Corrects spelling mistakes in the given word.
    
    ----------------------------------------------------------------------------
    Examples
    --------
    WORD: "celebracion"
    CORRECTED_WORD: "celebración"
    """
    
    WORD = dspy.InputField()
    CORRECTED_WORD = dspy.OutputField()
    
class CorrectorModule(dspy.Module):
    def __init__(
        self,
    ):
        """
        Initialize the CorrectorModule, which corrects spelling mistakes in the given word.
        """
        super().__init__()
        self.corrector = dspy.Predict(Corrector)
        
    def forward(self, word: str):
        """
        Forward pass of the module.
        """
        
        return self.corrector(WORD=word).CORRECTED_WORD

class TransformNotOptim(dspy.Signature):
    """
    Map similar terms to a common form in LANGUAGE, considering lemmatization errors, synonyms, and spelling variations. Return [] if no equivalences are found. 

    ----------------------------------------------------------------------------
    Examples
    --------
    WORDS: ["centres_salut", "centro_nacional", "centros_salud", "centro_seccionamiento", "centro_salud", "centre_salut"]
    MAPPED_WORDS: [{"centro_salud": "centres_salut, centros_salud,centre_salut"}]
    
    WORDS: ["octubre", "noviembre", "septiembre", "diciembre", "consejo_diciembre", "noviembre_contratos"]	
    MAPPED_WORDS: [{"diciembre": "consejo_diciembre, noviembre_contratos"}]
    
    WORDS: ["sustitución_termo", "cubierto_sustitución", "finca_reemplazo", "sustitución_enfriadora", "sustitución_luminaria", "sustitución_ventana"]		
    MAPPED_WORDS: [{"sustitución": "sustitución_termo, sustitución_enfriadora,sustitución_luminaria, sustitución_ventana"}]
    
    WORDS: ["ikastetxea", "escoles", "escuela"]		
    MAPPED_WORDS: [{"escuela": escoles, ikastetxea"}]
    
    WORDS: ["acuartelamiento", "ajardinamiento"]		
    MAPPED_WORDS: []
    
    WORDS: ["calefacció", "calor"]		
    MAPPED_WORDS: [{"calefacción": calefacció}]
    
    WORDS: ["ikastetxea", "escoles", "escuela"]		
    MAPPED_WORDS: [{"escuela": "escoles, ikastetxea"}]
    
    ----------------------------------------------------------------------------
    
    Important: The final word must be a single word or multiple words in LANGUAGE joined by an underscore ('_'). 
    Use the simplest form, e.g., choose "digital" over "digitales" and a single word over a compound word, e.g., "proyecto" over "proyecto_básico". 
    If it applies, the final word should be a noun over an adverb, adjective or verb.
    """

    WORDS = dspy.InputField()
    LANGUAGE = dspy.InputField()
    MAPPED_WORDS = dspy.OutputField()

class TransformOptim(dspy.Signature):
    """
    Map similar terms to a common form in LANGUAGE, considering lemmatization errors, synonyms, and spelling variations. Return [] if no equivalences are found.
    
    ----------------------------------------------------------------------------
    Examples
    --------
    WORDS: ["calefacció", "calor"]		
    MAPPED_WORDS: [{"calefacción": calefacció}]
    
    WORDS: ["vivienda_protegida", "vivienda", "viviendas_protegidas"]
    MAPPED_WORDS: [{"vivienda_protegida": "viviendas_protegidas"}]
    
    WORDS: ["expte", "expdte"]
    MAPPED_WORDS: [{"vivienda_protegida": "expdte, expte"}]
    ----------------------------------------------------------------------------

    Important: The final word must be a single word or multiple words in LANGUAGE joined by an underscore ('_'). 
    Do not loose specific information, e.g., "vivienda_protegida" should not be mapped to "vivienda", since it is a specific type of housing.
    If it applies, the final word should be a noun over an adverb, adjective or verb. If it is a noun, it should be singular, and if it is a verb, it should be in the infinitive form.
    If you are present with acronyms or abbreviations, map them to the full form.
    """
    # and a single word over a compound word, e.g., "proyecto" over "proyecto_básico"
    WORDS = dspy.InputField()
    LANGUAGE = dspy.InputField()
    MAPPED_WORDS = dspy.OutputField()

class TransformModule(dspy.Module):
    def __init__(
        self,
        optim: bool = False,
        lang: str = "spanish"
    ):  
        """
        Initialize the TransformModule, which maps similar terms to a common form in LANGUAGE, considering lemmatization errors, synonyms, and spelling variations.

        Parameters
        ----------
        optim: bool
            Whether to use the optimized version of the module, or the one with default examples
        """
        super().__init__()

        if optim:
            self.transform = dspy.ChainOfThought(TransformOptim)
        else:
            self.transform = dspy.ChainOfThought(TransformNotOptim)
        self.roman_numeral_pattern = r'_(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))$'

        if lang == "es":
            self._nlp_model = load_spacy("es_core_news_md")
        else:
            self._nlp_model = load_spacy("en_core_web_md")
        self._trf_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    def parse_equivalences(
        self,
        equivalences_str: str,
        word_embeddings: Optional[Dict[str, Any]] = None,
        thr_similarity: float = 0.96
    ) -> List[Dict[str, List[str]]]:
        """
        Parse the equivalences string and return a list of dictionaries with the equivalences.

        Parameters
        ----------
        equivalences_str : str
            A string containing the equivalences in the format of a Python dictionary.
        word_embeddings : Dict[str, Any], optional
            A dictionary containing the word embeddings for the words in the equivalences.
        thr_similarity : float
            The threshold for cosine similarity to consider words as equivalent.

        Returns
        -------
        List[Dict[str, List[str]]]
            A list of dictionaries containing the parsed equivalences.
        """
        try:
            equivalences = ast.literal_eval(equivalences_str)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing equivalences string: {e}")
            return []

        final_equivalences = []
        for el in equivalences:
            key = next(iter(el))
            new_key = self._remove_roman_numerals(key)

            eq_dict = {new_key: []}
            values = [v.strip() for v in el[key].split(",") if v.strip() and v.strip() != key]

            for val in values:
                new_key = self._process_value(
                    key, val, new_key, eq_dict, word_embeddings, thr_similarity
                )

            try:
                if eq_dict[new_key]:
                    # check if there are repeated values. If so, count them
                    repeated_values = [item for item in eq_dict[new_key] if eq_dict[new_key].count(item) > 1]
                    
                    if repeated_values and len(repeated_values) > 1:
                        eq_dict[new_key] = list(set(eq_dict[new_key]))# keep only unique values
                    
                    final_equivalences.append(eq_dict)
            except KeyError as e:
                print(f"KeyError: {e}")

        return final_equivalences

    def _remove_roman_numerals(self, text: str) -> str:
        match = re.search(self.roman_numeral_pattern, text, flags=re.IGNORECASE)
        return text[:match.start()].strip() if match else text

    def _process_value(
        self,
        key: str,
        val: str,
        new_key: str,
        eq_dict: Dict[str, List[str]],
        word_embeddings: Optional[Dict[str, Any]],
        thr_similarity: float
    ):
        key_split = key.split("_")
        val_split = val.split("_")

        # Check if key and val are the same
        if key == val:
            return  # Skip if they are identical

        # Handle specific cases
        if self._is_specific_type(key_split, val_split, key, val):
            return

        # Handle cases where both key and val have the same prefix
        if key_split[0] == val_split[0]:
            if self._handle_same_prefix(
                key_split, val_split, eq_dict, thr_similarity
            ):
                return
        else:
            # Handle the general case
            new_key = self._handle_general_case(
                key, val, new_key, eq_dict, word_embeddings, thr_similarity
            )
        return new_key

    def _is_specific_type(
        self,
        key_split: List[str],
        val_split: List[str],
        key: str,
        val: str
    ) -> bool:
        # Skip mapping if one is a specific type of the other
        if len(key_split) == 1 and len(val_split) > 1 and key_split[0] == val_split[0]:
            print(f"Skipping '{val}' since it is a specific type of '{key}'")
            return True
        if len(val_split) == 1 and len(key_split) > 1 and val_split[0] == key_split[0]:
            print(f"Skipping '{key}' since it is a specific type of '{val}'")
            return True
        return False

    def _handle_same_prefix(
        self,
        key_split: List[str],
        val_split: List[str],
        eq_dict: Dict[str, List[str]],
        thr_similarity: float
    ) -> bool:
        if len(key_split) > 1 and len(val_split) > 1:
            # Check if the suffixes are proper nouns
            if self._are_proper_nouns(key_split[1], val_split[1]):
                #import pdb; pdb.set_trace()
                base_key = key_split[0]
                eq_dict.setdefault(base_key, []).extend([key_split[1], val_split[1]])
                print(f"'{key_split[1]}' and '{val_split[1]}' are proper nouns. Mapping to '{base_key}'")
                return True
            else:
                # Compare embeddings of the suffixes
                similarity = self._compare_embeddings(
                    key_split[1], val_split[1], thr_similarity
                )
                if similarity >= thr_similarity:
                    eq_dict.setdefault("_".join(key_split), []).append("_".join(val_split))
                    print(f"Suffixes '{key_split[1]}' and '{val_split[1]}' are similar.")
                    return True
                else:
                    print(f"Low similarity ({similarity}) between '{key_split[1]}' and '{val_split[1]}'")
                    return False
        return False

    def _are_proper_nouns(self, text1: str, text2: str) -> bool:
        doc1 = self._nlp_model(text1)
        doc2 = self._nlp_model(text2)
        return all(token.pos_ == "PROPN" for token in doc1) and all(token.pos_ == "PROPN" for token in doc2)

    def _compare_embeddings(
        self,
        text1: str,
        text2: str,
        thr_similarity: float
    ) -> float:
        emb1 = self._trf_model.encode(text1)
        emb2 = self._trf_model.encode(text2)
        similarity = cos_sim(emb1, emb2)
        return similarity

    def _handle_general_case(
        self,
        key: str,
        val: str,
        new_key: str,
        eq_dict: Dict[str, List[str]],
        word_embeddings: Optional[Dict[str, Any]],
        thr_similarity: float
    ):
        print("Handling general case")

        dif_tilde = solo_diferencia_tilde(key, val)
        if dif_tilde:
            if dif_tilde == key:
                eq_dict[new_key].append(val)
                print(f"Words '{key}' and '{val}' differ only by accent marks. Mapping '{val}' to '{new_key}'.")
            elif dif_tilde == val:
                # replace new_key with val
                eq_dict[val] = eq_dict.pop(new_key)
                eq_dict[val].append(key)
                new_key = val
        elif word_embeddings is not None and key[:3] != val[:3]:
            try:
                similarity = cos_sim(word_embeddings[key], word_embeddings[val])
            except KeyError as e:
                print(f"Word embedding not found: {e}")
                return
            if similarity >= thr_similarity:
                eq_dict[new_key].append(val)
                print(f"Words '{key}' and '{val}' are similar (similarity: {similarity}). Mapping them.")
            else:
                print(f"Similarity between '{key}' and '{val}' is {similarity}. Skipping...")
        else:
            eq_dict[new_key].append(val)
            print(f"Mapping '{val}' to '{new_key}' by default.")
            
        return new_key

    def forward(
        self,
        words: str,
        word_embeddings: Dict[str, Any] = None,
        lang: str="spanish",
        optim: bool = True
    ):
        """
        Forward pass of the module.
        """
        
        out_transform = self.transform(WORDS=words, LANGUAGE=lang).MAPPED_WORDS
        equivalences = self.parse_equivalences(out_transform, word_embeddings)
    
        if not optim:
            return equivalences
        else:
            return dspy.Prediction(equivalences=equivalences)
      
class HermesEquivalencesGenerator(object):
    
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        use_optimized: bool = False,
        do_train: bool = False,
        data_path: str = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/training_data/dataset_eqs.xlsx",
        trained_promt: str = pathlib.Path(
            __file__).parent.parent.parent / "data/optimized/HermesEquivalencesGenerator-saved.json",
        trf_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        lang = "es",
        logger: logging.Logger = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ):
        """
        Initialization of the HermesAcronymDetector.
        
        Parameters
        ----------
        model_type : str, optional
            Type of model to use, by default "llama"
        open_ai_model : str, optional
            OpenAI model to use, by default "gpt-3.5-turbo"
        path_open_api_key : str, optional
            Path to OpenAI API key.
        use_optimized : bool, optional
            Whether to use the optimized version of the module, by default False
        do_train : bool, optional
            Whether to train the module, by default False
        data_path : str, optional
            Path to file with training data, by default None
        trained_promt : str, optional
            Path to trained prompt, by default None
        trf_model : str, optional
            Transformer model to use, by default 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        logger : logging.Logger, optional
            Logger to use, by default None
        path_logs : pathlib.Path, optional
            Path to logs directory, by default pathlib.Path(__file__).parent.parent / "data/logs"
        """
        self._logger = logging.getLogger(__name__)        
        self._model = SentenceTransformer(trf_model)
        if lang == "es":
            self._nlp_model = load_spacy("es_core_news_md")
        else:
            self._nlp_model = load_spacy("en_core_web_md")
        self._corrector = CorrectorModule()
        self.eq_module = EquivalentModule()
        
        # Dspy settings
        if model_type == "llama":
            #self.lm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B ",port=8090, url="http://127.0.0.1")
            self.lm = dspy.LM(
                "ollama_chat/llama3.1:8b-instruct-q8_0",# también puede ser llama3.2
                api_base="http://kumo01:11434"  # Dirección base de tu API
            )
        elif model_type == "openai":
            load_dotenv(path_open_api_key)
            api_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            self.lm = dspy.OpenAI(model=open_ai_model)
        # TODO Add mistral model
        elif model_type == "mistral":
            self.lm = dspy.LM(
                "mistral/mistral_to_do", # Model name
                api_base="http://kumo01:11434"
            )
        else:
            raise ValueError(f"Model it is not supported: {model_type}")
        
        dspy.configure(lm=self.lm, temperature=0)
        
        if not use_optimized:
            self.module = TransformModule(optim=False, lang=lang)
        elif use_optimized and not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("-- -- Trained prompt not found. Exiting.")
                return
            self.module = TransformModule(optim=True, lang=lang)
            self.module.load(trained_promt)
            self._logger.info(f"-- -- TransformModule loaded from {trained_promt}")
        else:
            if not data_path:
                self._logger.error("-- -- Data path is required for training. Exiting.")
                return
            self._train_module(data_path, trained_promt)
    
    def _get_clusters(
        self,
        embeddings: np.ndarray,
        eps: float = 0.05,
        min_samples: int = 2,
        metric: str = 'cosine'
    ) -> List[int]:
        """
        Partition the embeddings into clusters using DBSCAN.
        
        Parameters
        ----------
        embeddings : np.ndarray
            An array of embeddings to cluster.
        eps : float, optional
            The maximum distance between two samples for one to be considered as in the neighborhood of the other, by default 0.05
        min_samples : int, optional
            The number of samples in a neighborhood for a point to be considered as a core point, by default 2
        metric : str, optional
            The distance metric to use, by default 'cosine'
            
        Returns
        -------
        list
            A list of cluster labels for each embedding
        """
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan.fit_predict(embeddings)

        # Count number of clusters (excluding noise labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self._logger.info(f'-- -- Number of clusters: {n_clusters}')
        print(f'-- -- Number of noise points: {n_noise}')
        print(f'-- -- Labels: {set(labels)}')
        
        return labels
    
    def _get_embeddings(
        self,
        word_list: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get the SBERT embeddings for a list of words using the model self._trf_model.
        
        Parameters
        ----------
        word_list : list
            A    list of words for which to get embeddings.
        
        Returns
        -------
        dict
            A dictionary where the keys are words and the values are their embeddings.
        """
        # Calculate embeddings
        embeddings = self._model.encode(word_list)
        
        embedding_dict = {word: embedding for word, embedding in zip(word_list, embeddings)}
        
        return embedding_dict
    
    def _perform_pca(self, data, variance_threshold=0.9):
        """
        Performs PCA on the given data and retains components that explain the desired variance.

        Parameters:
        - data: numpy array of shape (N, M), where N is the number of samples and M is the number of features.
        - variance_threshold: float between 0 and 1 indicating the cumulative variance to retain.

        Returns:
        - reduced_data: Transformed data with reduced dimensions.
        - n_components: Number of principal components retained.
        - explained_variance_ratio: Explained variance ratio of the retained components.
        """
        # Standardize the data to have mean=0 and variance=1
        scaler = StandardScaler()
        X_std = scaler.fit_transform(data)

        # Perform PCA to retain components that explain the desired variance
        pca = PCA(n_components=variance_threshold)
        reduced_data = pca.fit_transform(X_std)
        n_components = pca.n_components_
        explained_variance_ratio = pca.explained_variance_ratio_

        # Print the results
        print(f"Number of components to retain {variance_threshold * 100}% variance: {n_components}")
        print(f"Explained variance ratio of retained components: {explained_variance_ratio}")
        print(f"Cumulative explained variance: {np.cumsum(explained_variance_ratio)}")

        return reduced_data, n_components, explained_variance_ratio

    def _get_words_by_cluster(
        self,
        labels: List[int],
        words: List[str]
    ):
        """
        Create a dictionary where the keys are cluster IDs and the values are lists of words belonging to each cluster.
        Handles noise points (label = -1) separately.

        Parameters
        ----------
        labels : list
            A list of cluster labels for each word in the vocabulary (output of DBSCAN).
        words : list
            A list of words corresponding to the labels.
        
        Returns
        -------
        dict
            A dictionary where the keys are cluster IDs and the values are lists of words belonging to each cluster.
            Noise points (label = -1) are stored under the key 'noise'.
        """
        
        # Get the unique cluster labels (excluding noise labeled as -1)
        unique_clusters = set(labels)
        
        # Initialize an empty dictionary to hold the words for each cluster
        cluster_to_words = {cluster_id: [] for cluster_id in unique_clusters if cluster_id != -1}
        
        # Optionally, handle noise points separately
        cluster_to_words['noise'] = []  # To handle noise points (cluster_id = -1)
        
        # Iterate over all words and their corresponding labels
        for word_id, word in enumerate(words):
            cluster_id = labels[word_id]
            if cluster_id == -1:
                cluster_to_words['noise'].append(word)  # Add noise points to 'noise'
            else:
                cluster_to_words[cluster_id].append(word)  # Append word to the corresponding cluster
        
        return cluster_to_words
    
    def _get_words_by_cluster_sim(
        self,
        embeddings: np.ndarray,
        words: List[str],
        thr=0.9,
    ):  
        """
        Group words based on similarities and save the results in a JSON file.
        """
        # Perform PCA on the embeddings
        embeddings_X, _, _ = self._perform_pca(embeddings)
        
        # Normalize embeddings
        embeddings = normalize(embeddings_X, norm='l2') 
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        thresholds = np.linspace(0.1, 0.9, 9)
        print(f'Calculating connected components for different thresholds: {thresholds}....')
        stats = []
        for threshold in thresholds:
            adjacency_matrix = similarity_matrix > threshold
            np.fill_diagonal(adjacency_matrix, False)
            adjacency_csr = csr_matrix(adjacency_matrix)
            n_components, labels = connected_components(adjacency_csr)
            
            # Map words to clusters
            cluster_to_words = defaultdict(list)
            for word, label in zip(words, labels):
                cluster_to_words[label].append(word)

            # Filter to keep only clusters with more than one word
            reduced_cluster_to_words = {label: words for label, words in cluster_to_words.items() if len(words) > 1}
            
            # Optional: count how many clusters have more than one word
            n_components_more = len(reduced_cluster_to_words)
            
            word_groups = list(cluster_to_words.values())
            #results.append((threshold, n_components, len(word_groups)))
            if len(set(labels)) > 1:
                score = float(silhouette_score(embeddings, labels, metric='cosine'))
                dbi = float(davies_bouldin_score(embeddings, labels))
                self._logger.info(f'Silhouette Score: {score:.4f}')
                print(f'Silhouette Score: {score:.4f}')
                print(f'Davies-Bouldin Index: {dbi:.2f}')
            else:
                score = -1
                dbi = -1
                print("Silhouette Score: No se puede calcular con solo un cluster.")
            
            # Generate JSON results for this threshold
            stats.append({
                "threshold": float(threshold),  # Convertir a float
                "components": int(n_components),  # Convertir a int
                "word_groups": len(word_groups),
                "n_components_more": n_components_more,
                "silhouette_score": score,
                "davies_bouldin_index": dbi,
            })
            
            print(f'Threshold: {threshold:.2f}, Components: {n_components}, Word Groups: {len(word_groups)}, n_components_more: {n_components_more}, Silhouette Score: {score:.4f}')
            self._logger.info(f'Threshold: {threshold:.2f}, Components: {n_components}, Word Groups: {len(word_groups)}, n_components_more: {n_components_more}, Silhouette Score: {score:.4f}')
        
        adjacency_matrix = similarity_matrix > thr
        np.fill_diagonal(adjacency_matrix, False)
        adjacency_csr = csr_matrix(adjacency_matrix)
        n_components, labels = connected_components(adjacency_csr)
        
        if len(set(labels)) > 1:
            score = float(silhouette_score(embeddings, labels, metric='cosine'))
            self._logger.info(f'Silhouette Score: {score:.4f}')
            print(f'Silhouette Score: {score:.4f}')
            dbi = float(davies_bouldin_score(embeddings, labels))
            print(f'Davies-Bouldin Index: {dbi:.2f}')
        else:
            print("Silhouette Score: No se puede calcular con solo un cluster.")
                    
        # Map words to clusters
        cluster_to_words = defaultdict(list)
        for word, label in zip(words, labels):
            cluster_to_words[label].append(word)
        reduced_cluster_to_words = [words for label, words in cluster_to_words.items() if len(words) > 1]
        
        return reduced_cluster_to_words, stats

    def _train_module(
        self,
        data_path: str,
        trained_promt: str
    ) -> None:
        """
        Trains the TransformModule and saves the trained model.
        
        Parameters
        ----------
        data_path : str
            Path to the data file with training examples.
        trained_promt : str
            Path to save the trained model.
        """
        self._logger.info("Training TransformModule...")
        self.module = self.optimize_module(data_path)
        self._logger.info("TransformModule optimized.")
        self.module.save(trained_promt)

    def validate_equivalences(self, example, pred, trace=None):
        """
        Function to validate the equivalences predicted by the module during optimization.
        
        Parameters
        ----------
        example : dspy.Example
            An example from the dataset.
        pred : dspy.Prediction
            The prediction made by the module.
            
        Returns
        -------
        float
            A score between 0 and 1 representing the quality of the predicted equivalences, calculated as the average of the normalized key and word scores. The key score is the number of matching keys divided by the total number of keys in the ground truth equivalences. The word score is the number of matching words divided by the total number of words in the ground truth equivalences.
        """
        
        ground_eq = ast.literal_eval(example['equivalences'])
        ground_eq_lst = []
        for el in ground_eq:
            # Get the first key in the dictionary
            key = list(el.keys())[0]
            values = [el.strip() for el in el[key].split(",")]
            ground_eq_lst.append({key: values})
        pred_eq_lst = pred.equivalences
        
        print(f"Ground: {ground_eq_lst}")
        print(f"Pred: {pred_eq_lst}")
        
        # Initialize counters for scores
        matching_keys_count = 0
        matching_words_count = 0
        total_words_in_ground_eq = 0

        # Convert pred_eq_lst and ground_eq_lst to a more convenient format (dict with key-value pairs)
        pred_dict = {list(item.keys())[0]: list(item.values())[0] for item in pred_eq_lst}
        ground_dict = {list(item.keys())[0]: list(item.values())[0] for item in ground_eq_lst}

        # Total number of keys in ground_eq_lst
        total_keys_in_ground_eq = len(ground_dict)

        # Count the matching keys and words
        for key in ground_dict:
            if key in pred_dict:
                matching_keys_count += 1
                
                # Count matching words in the values list
                ground_values = set(ground_dict[key])
                pred_values = set(pred_dict[key])
                
                matching_words = ground_values.intersection(pred_values)
                matching_words_count += len(matching_words)
                
                # Keep track of total words in ground equivalences for normalization
                total_words_in_ground_eq += len(ground_values)

        # If there are no keys or words in ground_eq_lst, avoid division by zero
        if total_keys_in_ground_eq > 0:
            normalized_key_score = matching_keys_count / total_keys_in_ground_eq
        else:
            normalized_key_score = 0

        if total_words_in_ground_eq > 0:
            normalized_word_score = matching_words_count / total_words_in_ground_eq
        else:
            normalized_word_score = 0

        return (normalized_key_score + normalized_word_score) / 2
            
    def optimize_module(self, data_path, mbd=4, mld=16, ncp=2, mr=1, dev_size=0.25): 
        """
        Optimizes the AcronymDetectorModule based on the data provided.
        """
        # Create dataset
        dataset = EquivalencesDataset(
            data_fpath=data_path,
            dev_size=dev_size,
        )
                
        self._logger.info(f"-- -- Dataset loaded from {data_path}")

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        self._logger.info(
            f"-- -- Dataset split into train, dev, and test. Training module...")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                      num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.validate_equivalences, **config)

        compiled_pred = teleprompter.compile(
            TransformModule(optim=True), trainset=trainset, valset=devset)

        self._logger.info(f"-- -- Module compiled. Evaluating on test set...")
        
        # Apply on test set
        tests = []
        for el in testset:
            output = compiled_pred(el.words)
            tests.append([el.words, el.equivalences,
                          output["equivalences"], 
                          self.validate_equivalences(el, output)])
            
        df = pd.DataFrame(
            tests, columns=["ORIGIN", "GROUND_EQ", "PREDICTED_EQ", "METRIC"])
        
        print(f"## Test set results ##")
        print(df.head())
        print(f"## Mean metric TEST: {df['METRIC'].mean()}")
        self._logger.info(f"## Mean metric TEST: {df['METRIC'].mean()}")
        #import pdb; pdb.set_trace()
        evaluate = Evaluate(
            devset=devset, metric=self.validate_equivalences, num_threads=1, display_progress=True)
        compiled_score = evaluate(compiled_pred)
        uncompiled_score = evaluate(TransformModule(optim=True))

        print(
            f"## TransformModule Score for uncompiled: {uncompiled_score}")
        print(
            f"## TransformModule Score for compiled: {compiled_score}")
        print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")

        return compiled_pred
    
    def generate_equivalences(
        self,
        source: str, # either "vocabulary" or "tm"
        path_to_source: str = None,
        model_type: str = "MalletLda",
        language: str = "es",
        path_save: str = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/topicmodeling/data/equivalences/cpv45_equivalences_test_vocab.json",
        optim: bool = False,
        top_k: int = 100 
    ):
        """
        Generate equivalences for the given words and language.
        """
        ######################################################################### 
        # Get words that will be used to detect equivalences
        ########################################################################
        if source not in ["vocabulary", "tm"]:
            raise ValueError("source must be either 'vocabulary' or 'tm'")
        
        path_to_source = path_to_source or self.data_path
        if not pathlib.Path(path_to_source).exists():
            raise ValueError(f"Path to source {path_to_source} does not exist")
        
        if source == "vocabulary":
            all_words = []
            with open(path_to_source, "r") as file:
                for line in file:
                    word = line.strip()
                    all_words.append(word)
        else:            
            tm_params = {
                'load_data_path': path_to_source,
                'model_path': 'FOLDER_OF_TRAINED_TOPIC_MODEL'
            }

            params_inference = tm_params
            params_inference['load_model'] = True
            params_inference['model_path'] = path_to_source
            #params_inference['mallet_path'] = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/topicmodeling/Mallet-202108/bin/mallet"
            params_inference['mallet_path'] = "/export/usuarios_ml4ds/cggamella/RAG_tool/src/topicmodeling/Mallet-202108/bin/mallet"
            
            model = create_model(model_type, **params_inference)
            topics = model.print_topics(top_k=top_k)
            # keep top-k words from each topic
            topics = {k: v[:top_k] for k, v in topics.items() if len(v) > 0}

            all_words = list(set(list(itertools.chain(*topics.values()))))
        
        self._logger.info(f"-- -- Clusters will be created on {len(all_words)} words")
        
        ######################################################################### 
        # Generate clusters
        ########################################################################
        word_embeddings = self._get_embeddings(all_words)
        words = list(word_embeddings.keys())
        embeddings = np.array(list(word_embeddings.values()))
        #labels = self._get_clusters(embeddings)
        #cluster_to_words = self._get_words_by_cluster(labels, words)
        # word_groups = [cluster_to_words[el] for el in cluster_to_words ]
        #import pdb; pdb.set_trace()
        word_groups, stats = self._get_words_by_cluster_sim(embeddings, words)
        
        ######################################################################### 
        # Generate equivalences
        ########################################################################
        if language == "es":
            language = "spanish"
        else:
            language = "english"
        time_start = time.time()
        equivalences = []
        for el in word_groups:
            try:
                equivalences.append([el, self.module(words=str(el), lang=language, word_embeddings=word_embeddings, optim=optim)])
            except Exception as e:
                print(e)
        
        df = pd.DataFrame(equivalences, columns= ["origin", "equivalence"])
        
        #df.to_excel("/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/scholar_equivalences_test.xlsx")
        df.to_excel("/export/usuarios_ml4ds/cggamella/RAG_tool/data/objeto_contrato.xlsx")

        # filter empty equivalences
        df_old = df[df['equivalence'].str.len() > 0]
        print(df.head())
        # Loop through the dataframe to create the wordlist
        word_list = []
        for _, row in df_old.iterrows():
            for equivalence in row['equivalence']:
                for key, values in equivalence.items():
                    for value in values:
                        if language == "spanish":
                            # correct the key
                            key = self._corrector(key)
                        # if new key has spaces, replace them with underscores
                        key = key.replace(" ", "_")
                        if key != value:
                            word_list.append(f"{value}:{key}")
        #import pdb; pdb.set_trace()
        result = []
        for _, row in df.iterrows():
            origin_words = row['origin']
            if not isinstance(origin_words, list):
                origin_words = ast.literal_eval(origin_words)
            origin_words_str = " ".join(origin_words)
            print(f"Las listas de palabras son {origin_words}")
            eq_word = self.eq_module(origin_words_str) 
            eq_word = eq_word.replace("«", "").replace("»", "")
            eq_word = eq_word.strip('"').strip("'")
            eq_word = eq_word.replace(" ", "_")
            print(f"La equivalencia es {eq_word}")
            for w in origin_words:
                result.append({w: eq_word})
        
        word_list_new = [f"{list(d.keys())[0]}:{list(d.values())[0]}" for d in result]
        # Create the JSON structure
        json_data_old = {
            "name": "OLD_equivalences",
            "description": "",
            "valid_for": "equivalences",
            "visibility": "Public",
            "wordlist": word_list
        }
        
        # Create the JSON structure
        json_data_new = {
            "name": "NEW_equivalences",
            "description": "",
            "valid_for": "equivalences",
            "visibility": "Public",
            "wordlist": word_list_new
        }
            
        print(f"Time elapsed in generation: {time.time() - time_start}")
        return json_data_old, json_data_new, stats