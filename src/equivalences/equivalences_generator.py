import os
import re
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
from typing import List, Dict, Any
import ast
import itertools
from sentence_transformers.util import cos_sim

from src.topicmodeling.topic_model import BERTopicModel, CtmModel, MalletLdaModel, TopicGPTModel

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
        Initialize the TransformModule.

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
        word_embeddings: Dict[str, Any], optim: bool = False
    ) -> List[Dict[str, List[str]]]:
        try:
            # Safely parse the input string as a Python dictionary
            equivalences = ast.literal_eval(equivalences)

            final_eqs = []
            
            for el in equivalences:
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
                    if key[:3] != val[:3]:
                        try:
                            similarity = cos_sim(word_embeddings[key], word_embeddings[val])
                        except Exception as e:
                            continue
                        if similarity is None or similarity < 0.96:
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

    def forward(self, words, lang, word_embeddings):

        equivalences = self.parse_equivalences(self.transform(WORDS=words, LANGUAGE=lang).MAPPED_WORDS, word_embeddings)

        return equivalences
      
class HermesEquivalencesGenerator(object):
    
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        use_optimized: bool = False,
        do_train: bool = False,
        data_path: str = None,
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
        
        self._logger = logging.getLogger(__name__) # TODO: chekc (init_logger)
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
            # TODO: Implement training
            if not data_path:
                self._logger.error("Data path is required for training. Exiting.")
                return
            self._train_module(data_path, trained_promt)
            
    def _create_model(self, model_name, **kwargs):

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
    
    def _get_clusters(self, embeddings, eps=0.05, min_samples=2, metric='cosine'):
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        # Count number of clusters (excluding noise labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        self._logger.info(f'-- -- Number of clusters: {n_clusters}')
        print(f'-- -- Number of noise points: {n_noise}')
        print(f'-- -- Labels: {set(labels)}')
        
        return labels
    
    def _get_embeddings(self, word_list):
    
        # Calculate embeddings
        embeddings = self._model.encode(word_list)
        
        embedding_dict = {word: embedding for word, embedding in zip(word_list, embeddings)}
        
        return embedding_dict

    def _get_words_by_cluster(self, labels, words):
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
            cluster_id = labels[word_id]  # Get the cluster ID for this word
            if cluster_id == -1:
                cluster_to_words['noise'].append(word)  # Add noise points to 'noise'
            else:
                cluster_to_words[cluster_id].append(word)  # Append word to the corresponding cluster
        
        return cluster_to_words

    def _train_module(self, data_path, trained_promt):
        """
        Trains the AcronymDetectorModule and saves the trained model.
        """
        self._logger.info("Training AcronymDetectorModule...")
        self.module = self.optimize_module(data_path)
        self._logger.info("AcronymDetectorModule optimized.")
        self.module.save(trained_promt)

    def validate_acronym_detection(self, example, pred, trace=None):
        """
        Validates if the predicted acronym is present in the example text.
        Returns 1 if the acronym is present or correctly predicted (there are texts without acronyms) as '/', otherwise 0.
        """
        # Normalize text (where acronym its ideally contained) and predicted acronym to lowercase
        text_lower = example['texto'].lower()
        pred_acronym_lower = pred.ACRONYMS.lower()

        if pred_acronym_lower == '/':
            if example['detected_acronimos'] == '/':  
                print("Correctly identified no acronyms in the text.")
                return 1
            else:
                print(f"Error: Returned '/' but acronyms are present: {example['detected_acronimos']}")
                return 0

        if pred_acronym_lower in text_lower:
            print(f"The acronym '{pred_acronym_lower}' is present in the text.")
            return 1
        else:
            print(f"The acronym '{pred_acronym_lower}' is not present in the text.")
            return 0

    def create_dtset_detected_acronyms(self, df):
        
        df = df[['text', 'detected_acr']]

        # Dividir los datos en conjuntos de entrenamiento y validación
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        # Convertir los df de train y dev en listas de diccionarios
        data_train = train_df.to_dict('records')
        data_test = test_df.to_dict('records')

        # Training set examples with configured inputs
        trainset = [
            dspy.Example({'texto': row['text'], 'detected_acronimos': row['detected_acr']})
            .with_inputs('texto') for row in data_train
        ]
        
        # Test set examples with configured inputs
        devset = [
            dspy.Example({'texto': row['text'], 'detected_acronimos': row['detected_acr']})
            .with_inputs('texto') for row in data_test
        ]

        return trainset, devset
    
    def optimize_module(self, data_path, max_bootstrapped_demos=4, max_labeled_demos=16, num_candidate_programs=16, max_rounds=1):
        """
        Optimizes the AcronymDetectorModule based on the data provided.
        """
        df = pd.read_excel(data_path)
        df = df[['text', 'detected_acr']]

        tr_ex, test_ex = self.create_dtset_detected_acronyms(df)

        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.validate_acronym_detection,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_candidate_programs=num_candidate_programs,
            max_rounds=max_rounds,
        )
        compiled_model = teleprompter.compile(AcronymDetectorModule(), trainset=tr_ex, valset=test_ex)
        return compiled_model
    
    def generate_equivalences(
        self,
        source: str, # either "vocabulary" or "tm"
        path_to_source: str = None,
        model_type: str = "MalletLda",
        language: str = "spanish",
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
        equivalences = []
        for el in word_groups:
            try:
                equivalences.append([el, self.module(str(el), language, word_embeddings)])
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
        with open('/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/topicmodeling/data/equivalences/cpv45_equivalences_test.json', 'w') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)