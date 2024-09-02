import pandas as pd
import dspy
import copy
import pathlib
import logging
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.primitives.assertions import DSPySuggestionError
from lingua import Language, LanguageDetectorBuilder

class AcronymExpander(dspy.Signature):
    """
    Expand the acronyms based on the text.
    """
    TEXTO = dspy.InputField()
    ACRONIMO = dspy.InputField()
    IDIOMA = dspy.InputField()
    EXPANSION = dspy.OutputField(desc = "Write the expanded acronym in the language {IDIOMA}")

class AcronymExpanderModule(dspy.Module):
    def __init__(self, language_detector):
        super().__init__()
        self.expander = dspy.Predict(AcronymExpander)
        self.language_detector = language_detector
        self.no_expansion_variations = [
            '/', '/ (no se expande)', '', '${TEXTO}', '${ACRONIMOS}', '/ (no hay expansión)',
            '(no es acronimo)', 'no se puede expandir', '${EXPANSION}',
            '/ (No acronyms present in the document)',
            '/ (not present in the document)', "'/'", 'N/A', '/.', 'N_A', 'NA'
        ]
        
    def _process_output(self, expansion):
        if expansion in self.no_expansion_variations:
            return "/"
        else:
            return expansion
    
    def verify_expansions(self,acronym,expansion):
        """
        Verifiy if the expansion is equal to the acronym.
        """
        # Normalizar el acrónimo y la expansión eliminando espacios y convirtiendo a minúsculas
        acronym_norm = acronym.strip().lower()
        expansion_norm = expansion.strip().lower()

        # Comparar si la expansión es exactamente igual al acrónimo
        return expansion_norm == acronym_norm

    def forward(self, texto, acronimo):
        """
        Detect the language of the text and expand the acronym based on the language detected
        """
        idioma = self.language_detector.detect_language(texto)
        response = self.expander(TEXTO=texto, ACRONIMO=acronimo, IDIOMA=idioma)
        expansion_generada = response.EXPANSION
    
        try:
            # Usar Suggest para verificar si la expansión es igual al acrónimo
            dspy.Suggest(
                self.verify_expansions(acronimo, expansion_generada),
                "La expansión generada debe coincidir exactamente con el acrónimo.",
                target_module=AcronymExpander
            )
        except DSPySuggestionError as e:
            print(f"Sugerencia fallida: {e}. Continuando ejecución...")
        
        # Verificar si la expansión está en las variaciones que indican no expansión
        if not expansion_generada or expansion_generada in self.no_expansion_variations:
            return dspy.Prediction(EXPANSION='/')
        else:
            # Procesar la expansión generada y devolverla
            return dspy.Prediction(EXPANSION=self._process_output(expansion_generada))
        
class LanguageDetectorModule:
    def __init__(self):
        # Define language of your corpus, the less you choose the faster will be
        self.languages = [Language.ENGLISH, Language.SPANISH, Language.BASQUE, Language.CATALAN]
        self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()

    def detect_language(self, text):
        detected_language = self.detector.detect_language_of(text)
        # Return language of the text
        return detected_language.name if detected_language else 'Indefinido'

class HermesAcronymExpander(object):

    def __init__(
        self,
        do_train: bool = False,
        data_path: str = None,
        trained_promt: str = pathlib.Path(
            __file__).parent.parent.parent / "data/optimized/HermesAcronymExpander-saved.json",
        trf_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        logger: logging.Logger = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ):
        """
        Initialization of the HermesAcronymExpander

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
        self._logger = logger if logger else init_logger(__name__, path_logs)
        self.language_detector = LanguageDetectorModule()
        
        if not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("Trained prompt not found. Exiting.")
                return
            self.module = AcronymExpanderModule(language_detector=self.language_detector)
            self.module.load(trained_promt)
            self._logger.info(f"AcronymExpanderModule loaded from {trained_promt}") 
        else:
            if not data_path:
                self._logger.error("Data path is required for training. Exiting.")
                return
            self._train_module(data_path, trained_promt, trf_model)
      
            
    def _train_module(self, data_path, trained_promt, trf_model):
        """
        Trains the AcronymExpanderModule and saves the trained model.
        """
        self.trf_model = SentenceTransformer(trf_model)
        self._logger.info(f"Transformer model {trf_model} loaded.")
        self._logger.info("Training AcronymExpanderModule...")
        self.module = self.optimize_module(data_path)
        self._logger.info("AcronymExpanderModule optimized.")
        self.module.save(trained_promt)

    def validate_expansion(self, example, pred, trace=None):
        """
        Validates the expansion of acronyms by comparing embeddings from the transformer model.
        Returns
        -------
        bool
            True if the similarity is above the threshold, otherwise False.
        """
        try:
            # Encode the predicted and reference expansions using the transformer model
            prediction_embedding = self.trf_model.encode(str(pred.EXPANSION))
            reference_embedding = self.trf_model.encode(str(example['expansion']))

            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity([prediction_embedding], [reference_embedding])[0][0]
            self._logger.info(f"Cosine similarity: {similarity}")

            # Check if the similarity exceeds the threshold (0.7)
            return similarity > 0.7
        except Exception as e:
            self._logger.error(f"Error during expansion validation: {e}")
            return False

    def create_dtset_expanded_acronyms(self, df):
        
        df = df[['text', 'detected_acr', 'acr_des']]
        df = df[df['detected_acr'] != '/']
        print(len(df))

        # Dividir los datos en conjuntos de entrenamiento y validación
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        # Convertir los df de train y dev en listas de diccionarios
        data_train = train_df.to_dict('records')
        data_test = test_df.to_dict('records')

        # Crear ejemplos de entrenamiento con inputs configurados
        trainset = [
            dspy.Example({'texto': row['text'], 'acronimo': row['detected_acr'], 'expansion': row['acr_des']})
            .with_inputs('texto','acronimo') for row in data_train
        ]
        
        # Crear ejemplos de validación con inputs configurados
        devset = [
            dspy.Example({'texto': row['text'], 'acronimo': row['detected_acr'], 'expansion': row['acr_des']})
            .with_inputs('texto','acronimo') for row in data_test
        ]

        return trainset, devset

    
    def optimize_module(self, data_path, max_bootstrapped_demos=4, max_labeled_demos=16, num_candidate_programs=16, max_rounds=1):
        """
        Optimizes the module AcronymExpanderModule based on the data provided.
        """
        # Read data
        df = pd.read_excel(data_path)
        df = df[['text', 'detected_acr', 'acr_des']]

        tr_ex, test_ex = self.create_dtset_expanded_acronyms(df)

        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.validate_expansion,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_candidate_programs=num_candidate_programs,
            max_rounds=max_rounds,
        )

        compiled_model = teleprompter.compile(AcronymExpanderModule(language_detector=self.language_detector), trainset=tr_ex, valset=test_ex)
        return compiled_model