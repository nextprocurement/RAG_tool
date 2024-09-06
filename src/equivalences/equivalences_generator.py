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
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from typing import List, Dict, Any, Optional, Union
import ast
import itertools
from sentence_transformers.util import cos_sim
from dspy.datasets import Dataset
from src.topicmodeling.topic_model import BERTopicModel, CtmModel, MalletLdaModel, TopicGPTModel
from dspy.evaluate import Evaluate


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
    ----------------------------------------------------------------------------

    Important: The final word must be a single word or multiple words in LANGUAGE joined by an underscore ('_'). 
    Use the simplest form, e.g., choose "digital" over "digitales" and a single word over a compound word, e.g., "proyecto" over "proyecto_básico". 
    If it applies, the final word should be a noun over an adverb, adjective or verb.
    """

    WORDS = dspy.InputField()
    LANGUAGE = dspy.InputField()
    MAPPED_WORDS = dspy.OutputField()

class TransformModule(dspy.Module):
    def __init__(
        self,
        optim: bool = False,
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

    def parse_equivalences(
        self,
        equivalences: str,
        word_embeddings: Dict[str, Any], 
        thr_similarity: float = 0.96
    ) -> List[Dict[str, List[str]]]:
        """
        Parse the equivalences string and return a list of dictionaries with the equivalences, following the format:
        
        [{"key": ["value1", "value2", ...]}, ...]
        
        Parameters
        ----------
        equivalences : str
            A string containing the equivalences in the format of a Python dictionary, returned by the TransformModule for MAPPED_WORDS.
        word_embeddings : Dict[str, Any]
            A dictionary containing the word embeddings for the words in the equivalences.
        """
        
        try:
            # Safely parse the input string as a Python dictionary
            equivalences = ast.literal_eval(equivalences)
            print(equivalences)
            final_eqs = []
            for el in equivalences:
                print(el)
                # Get the first key in the dictionary
                key = list(el.keys())[0]

                new_key = key
                match = re.search(self.roman_numeral_pattern, key, flags=re.IGNORECASE)
                if match:
                    new_key = key[:match.start()]
    
                # Initialize an empty list for equivalences related to this key
                eq_dict = {new_key: []}
       
                # Iterate over each value split by commas
                for val in el[key].split(","):
                    val = val.strip()  # Strip leading/trailing spaces
    
                    if key == val:
                        continue  # Skip if the value is identical to the key
    
                    # If the first 5 characters differ, enforce a stricter similarity threshold
                    
                    if word_embeddings is not None and key[:3] != val[:3]:
                        try:
                            similarity = cos_sim(word_embeddings[key], word_embeddings[val])
                        except Exception as e:
                            continue
                        if similarity is None or similarity < thr_similarity:
                            continue  # Skip if similarity is below the threshold
                    # If they have the same first 5 characters or similarity is high, add it
                    eq_dict[new_key].append(val)

                if eq_dict[new_key] != []:
                    final_eqs.append(eq_dict)            
            
            return final_eqs
        
        except (SyntaxError, ValueError, TypeError) as e:
            # Handle invalid input or evaluation issues gracefully
            print(f"Error parsing equivalences: {e}")
            print(equivalences)
            return []    

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
        
        # Dspy settings
        if model_type == "llama":
            self.lm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B ",
                                       port=8090, url="http://127.0.0.1")
        elif model_type == "openai":
            load_dotenv(path_open_api_key)
            api_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            self.lm = dspy.OpenAI(model=open_ai_model)
        dspy.settings.configure(lm=self.lm)
        
        if not use_optimized:
            self.module = TransformModule(optim=False)
        elif use_optimized and not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("-- -- Trained prompt not found. Exiting.")
                return
            self.module = TransformModule(optim=True)
            self.module.load(trained_promt)
            self._logger.info(f"-- -- TransformModule loaded from {trained_promt}")
        else:
            if not data_path:
                self._logger.error("-- -- Data path is required for training. Exiting.")
                return
            self._train_module(data_path, trained_promt)
            
    def _create_model(
        self,
        model_name: str,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Instantiate a topic model based on the model name and keyword arguments.
        
        Parameters
        ----------
        model_name : str
            Name of the model to instantiate.
        **kwargs : Dict[str, Any]
            Keyword arguments to pass to the model constructor, specific to each model.
        """
        
        model_mapping = {
            'MalletLda': MalletLdaModel,
            'Ctm': CtmModel,
            'BERTopic': BERTopicModel,
            'TopicGPT': TopicGPTModel
        }

        # Retrieve the class based on the model name
        model_class = model_mapping.get(model_name)

        # Check if the model name is valid
        if model_class is None:
            raise ValueError(f"Invalid model name: {model_name}")

        # Create an instance of the model class
        model_instance = model_class(**kwargs)

        return model_instance
    
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
        """Function to validate the equivalences predicted by the module during optimization.
        
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
        language: str = "spanish",
        path_save: str = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/topicmodeling/data/equivalences/cpv45_equivalences_test_vocab.json",
        optim: bool = False
    ):
        """
        Generate equivalences for the given words and language.
        """
        ######################################################################### Get words that will be used to detect equivalences
        ########################################################################
        if source not in ["vocabulary", "tm"]:
            raise ValueError("source must be either 'vocabulary' or 'tm'")
        
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
            params_inference['mallet_path'] = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/topicmodeling/Mallet-202108/bin/mallet"

            model = self._create_model(model_type, **params_inference)
            topics = model.print_topics()

            #for i, topic in enumerate(topics):
            #    print("Topic #", i)
            #    print(topics[topic])

            all_words = list(set(list(itertools.chain(*topics.values()))))
        
        self._logger.info(f"-- -- Clusters will be created on {len(all_words)} words")
        
        ######################################################################### Generate clusters
        ########################################################################
        word_embeddings = self._get_embeddings(all_words)
        words = list(word_embeddings.keys())
        embeddings = np.array(list(word_embeddings.values()))
        labels = self._get_clusters(embeddings)
        cluster_to_words = self._get_words_by_cluster(labels, words)
        
        word_groups = [cluster_to_words[el] for el in cluster_to_words ]
        i = len(words)

        ######################################################################### Generate equivalences
        ########################################################################
        time_start = time.time()
        equivalences = []
        for el in word_groups:
            try:
                equivalences.append([el, self.module(words=str(el), lang=language, word_embeddings=word_embeddings, optim=optim)])
            except Exception as e:
                print(e)
        
        df = pd.DataFrame(equivalences, columns= ["origin", "equivalence"])

        # filter empty equivalences
        df = df[df['equivalence'].str.len() > 0]
        print(df.head())
        
        # Loop through the dataframe to create the wordlist
        word_list = []
        for _, row in df.iterrows():
            for equivalence in row['equivalence']:
                for key, values in equivalence.items():
                    for value in values:
                        word_list.append(f"{value}:{key}")

        # Create the JSON structure
        json_data = {
            "name": "cpv45_equivalences",
            "description": "",
            "valid_for": "equivalences",
            "visibility": "Public",
            "wordlist": word_list
        }

        # Write the JSON data to a file
        with open(path_save, 'w') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
        
        print(f"Time elapsed in generation: {time.time() - time_start}")