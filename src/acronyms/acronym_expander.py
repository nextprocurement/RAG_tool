import pandas as pd
import dspy
import copy
import pathlib
import logging
import unidecode
import ujson
import re
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.primitives.assertions import DSPySuggestionError
from lingua import Language, LanguageDetectorBuilder


class AcronymExpander(dspy.Signature):
    """
    Expand the acronyms based on the text.
    ----------------------------------------------------------------------------
    Examples
    --------
    TEXT: "El pib en España ha subido un 2% en el último trimestre. En la UE se ha registrado un aumento del 3%."]
    ACRONYMS: 'pib', 'UE'
    LANGUAGE: SPANISH
    EXPANSION: 'Producto Interior Bruto', 'Unión Europea'
    
    TEXT: "Trabajos de sondeos del terreno y Estudio geotécnico en el CEIP JACARANDA, sito en la C/ Italo Cortella s/n , del Distrito Alcosa-Este Torreblanca(Sevilla)"]
    ACRONYMS: 'CEIP', 'C/', 's/n'
    LANGUAGE: SPANISH
    EXPANSION: 'Centro de Educación Infantil y Primaria', 'Calle', 'sin número'
    
    TEXT: "Adecuación de parte del semisotano del consultorio medico Juan Antonio Serrano, incorporando condiciones especiales de ejecución de carácter social relativas a inserción sociolaboral de personas en situación de desempleo de larga duranción"]
    ACRONYMS: '/' 
    LANGUAGE: SPANISH
    EXPANSION: '/'
    ----------------------------------------------------------------------------
    """
    TEXT = dspy.InputField()
    ACRONYMS = dspy.InputField()
    LANGUAGE = dspy.InputField()
    EXPANSION = dspy.OutputField(desc = "Write the expanded acronym in the language {LANGUAGE}")

class AcronymExpanderModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.expander = dspy.Predict(AcronymExpander)
        self.no_expansion_variations = [
            '/', '/ (no se expande)', '', '${TEXTO}', '${ACRONIMOS}', '/ (no hay expansión)',
            '(no es acronimo)', 'no se puede expandir', '${EXPANSION}', '${ACRONYMS}', '${ACRONYM}',
            '/ (No acronyms present in the document)', 'TEXT', '${TEXT}', '  ',
            '/ (not present in the document)', "'/'", 'N/A', '/.', 'N_A', 'NA'
        ]
        
    def dump_state(self, save_verbose=False, ensure_ascii=False, escape_forward_slashes=False):
        print(self.named_parameters())
        return {name: param.dump_state() for name, param in self.named_parameters()}
    
    def save(self, path, save_field_meta=False):
        print("*"*50)
        with open(path, "w") as f:
            f.write(ujson.dumps(self.dump_state(save_field_meta), indent=2, ensure_ascii=False, escape_forward_slashes=False))
            
    def _process_output(self, expansion):
        if expansion in self.no_expansion_variations:
            return "/"
        else:
            return expansion
    
    def normalize_text(self, text):
        """
        Normalize text by converting to lowercase, removing accents and punctuation.
        """
        # Convert to lowercase and remove accents
        text = unidecode.unidecode(text.lower())
        # Remove common punctuation marks
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text
    
    # Useless function if Suggest is NOT used in the forward method
    def verify_expansions(self, acronym, expansion):
        """
        Verifiy if the expansion is equal to the acronym.
        """
        # Normalize acronym and expansion to lowercase
        acronym_norm = acronym.strip().lower()
        expansion_norm = expansion.strip().lower()
        return expansion_norm == acronym_norm

    def forward(self, texto, acronimo):
        """
        Expand the acronym based on the language detected of the text.
        """
        language_detector = LanguageDetectorModule()
        idioma = language_detector.detect_language(texto)
        
        # Convert acronyms to a list
        acronimos_list = [acronimo.strip() for acronimo in acronimo.split(',')]
        print("LISTA:",acronimos_list)
        
        expansions = []
        for acronimo in acronimos_list:
            response = self.expander(TEXT=texto, ACRONYMS=acronimo, LANGUAGE=idioma)
            expansion_generada = response.EXPANSION

            # Verificar si la expansión es vacía o un marcador de posición
            if not expansion_generada or expansion_generada in self.no_expansion_variations:
                print(f"Entra aquí con el acrónimo: {acronimo}")
                expansions.append("/")
                continue

            normalized_expansion = self.normalize_text(expansion_generada)
            print("La expansión normalizada es:", normalized_expansion)
            normalized_acronym = self.normalize_text(acronimo)
            print("El acrónimo normalizado es:", normalized_acronym)
            
            # Verificar si la expansión generada es demasiado larga
            if len(normalized_expansion) > 120:
                print("EXPANSION GENERATED IS TOO LONG!!!")
                expansions.append("/")
                continue
            
            if len(re.findall(r'\d', normalized_expansion)) > 2:
                print("TOO MANY NUMBERS IN THE EXPANSION")
                expansions.append("/")
                continue

            # Verificar si la expansión normalizada es idéntica al acrónimo
            if normalized_expansion == normalized_acronym:
                print("EXPANSION GENERATED IS EQUAL TO THE ACRONYM, MISTAKE!!!")
                print('*' * 50)
                expansions.append("/")
                continue

            # Verificar si alguna palabra de la expansión normalizada es idéntica al acrónimo normalizado
            if any(word == normalized_acronym for word in normalized_expansion.split()):
                print("A WORD IN THE NORMALIZED EXPANSION IS IDENTICAL TO THE ACRONYM!")
                expansions.append("/")
                continue
            # Procesar la expansión generada
            processed_expansion = self._process_output(expansion_generada)
            expansions.append(processed_expansion)

        # Unir las expansiones en un solo string para devolver
        return dspy.Prediction(EXPANSION=', '.join(expansions))
        
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
            self.module = AcronymExpanderModule()
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
    
    def language_to_string(language):
        if isinstance(language, Language):
            return language.name
        return str(language)

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
        #print(len(df))

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

        compiled_model = teleprompter.compile(
            AcronymExpanderModule(),
            trainset=tr_ex,
            valset=test_ex)
        
        return compiled_model