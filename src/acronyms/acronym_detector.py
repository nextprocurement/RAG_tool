import os
from dotenv import load_dotenv
import dspy
import logging
import ujson
import pandas as pd
import pathlib
from time import sleep
from sklearn.model_selection import train_test_split
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

class AcronymDetector(dspy.Signature):
    """
    Given a text identify the acronyms contained in it. If none are present in the text, say '/'.
    ----------------------------------------------------------------------------
    Examples
    --------
    TEXT: "El pib en España ha subido un 2% en el último trimestre. En la UE se ha registrado un aumento del 3%."]
    ACRONYMS: pib, UE
    
    TEXT: "Trabajos de sondeos del terreno y Estudio geotécnico en el CEIP JACARANDA, sito en la C/ Italo Cortella s/n , del Distrito Alcosa-Este Torreblanca(Sevilla)"]
    ACRONYMS: CEIP, C/, s/n
    
    TEXT: "Adecuación de parte del semisotano del consultorio medico Juan Antonio Serrano, incorporando condiciones especiales de ejecución de carácter social relativas a inserción sociolaboral de personas en situación de desempleo de larga duranción"]
    ACRONYMS: / 
    ----------------------------------------------------------------------------
    """
    TEXT = dspy.InputField()
    ACRONYMS = dspy.OutputField(desc="list of comma-separated acronyms", format=lambda x: ', '.join(x) if isinstance(x, list) else x)


class AcronymDetectorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(AcronymDetector)
        
        self.no_acronym_variations = [
            '/', '/ (no acronyms)', '', '${TEXT}', '${ACRONYMS}', '/ (no hay acrónimos)',
            '(NO ACRONYMS)',"\/",'no acronyms','sin acronimos', 'sin acrónimos',
            '/ (No acronyms present in the document)', '\/', '/',
            '/ (not present in the document)', "'/'", 'N/A', '/.', 'N_A', 'NA', '""', 'c', 'a', 'n',
            'o', 's', 'i', 'r', 'm', 't', 'd', 'e', 'l', 'p', 'b', 'u', 'q', 'v', 'g', 'f', 'h', 'j', 'z', 'x', 'k', 'w', 'y'
        ]
        
    def dump_state(self, save_verbose=False, ensure_ascii=False, escape_forward_slashes=False):
        print(self.named_parameters())
        return {name: param.dump_state() for name, param in self.named_parameters()}
    
    def save(self, path, save_field_meta=False):
        print("*"*50)
        with open(path, "w") as f:
            f.write(ujson.dumps(self.dump_state(save_field_meta), indent=2, ensure_ascii=False, escape_forward_slashes=False))
            
    def _process_output(self, texto):
        if texto in self.no_acronym_variations:
            return "/"
        else:
            return texto
              
    # Useless function if Suggest is NOT used in the forward method
    def verify_acronyms(self, texto, acronyms):
        """
        Verify if all acronyms are present in the text
        Used to suggest a condition in the teleprompter.
        """
        # Normalize text and acronyms to lowercase and split into words
        text_norm = texto.lower().split()
        acronyms_norm = [acronym.strip().lower() for acronym in acronyms.split(',')]
        
        # Verify presence of acronyms in text
        results = [acronym in text_norm for acronym in acronyms_norm]

        # Return True if all acronyms are present in the text
        return all(results)
    
    def forward(self, texto):
        response = self.generator(TEXT=texto)
        acronyms = response.ACRONYMS
        '''
        #try:
        # Suggestions with error handling
        dspy.Suggest(
           not self.verify_acronyms(texto, acronyms),
            "Los acrónimos deben coincidir con las palabras del texto en minúsculas.",
            target_module=AcronymDetector
        )
        #except DSPySuggestionError as e:
        #    print(f"Sugerencia fallida: {e}. Continuando ejecución...")
        '''
        if len(texto) == len(acronyms) or texto == acronyms:
            return dspy.Prediction(ACRONYMS='/')
        elif len(acronyms) > 20:
            return dspy.Prediction(ACRONYMS='/')
        else:
            return dspy.Prediction(ACRONYMS=self._process_output(acronyms))
      
class HermesAcronymDetector:
    
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/NextProcurement/NP-Search-Tool/.env",
        do_train: bool = False,
        data_path: str = None,
        trained_promt: str = pathlib.Path(
            __file__).parent.parent.parent / "data/optimized_new_DSPy/HermesAcronymDetector-saved.json",
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
        self._logger = logger if logger else logging.getLogger(__name__)
        
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
    
        # TODOAdd mistral model
        elif model_type == "mistral":
            self.lm = dspy.LM(
                "mistral/mistral_to_do", # Nombre del modelo a desplegar en container ollama
                api_base="http://kumo01:11434"
            )
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")
            
        dspy.configure(lm=self.lm, temperature=0)
        
        if not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("Trained prompt not found. Exiting.")
                return
            self.module = AcronymDetectorModule()
            self.module.load(trained_promt, use_legacy_loading=True)
            self._logger.info(f"AcronymDetectorModule loaded from {trained_promt}")
        else:
            if not data_path:
                self._logger.error("Data path is required for training. Exiting.")
                return
            self._train_module(data_path, trained_promt)


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
        compiled_model = teleprompter.compile(
            AcronymDetectorModule(),
            trainset=tr_ex,
            valset=test_ex
        )
        return compiled_model
    
