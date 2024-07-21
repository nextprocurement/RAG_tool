import logging

import pandas as pd
import dspy
import pathlib
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from src.utils.utils import init_logger


class AcronymExpander(dspy.Signature):
    """
    Expande los acrónimos basándose en el contexto del texto.
    """

    TEXTO = dspy.InputField(
        desc="Texto que contiene el acrónimo,sigla o abreviatura y aporta información sobre la forma expandida")
    ACRONIMO = dspy.InputField(desc="Acrónimo que necesita ser expandido")
    EXPANSION = dspy.OutputField(desc="Es la forma expandida del acrónimo")


class AcronymExpanderModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.expander = dspy.ChainOfThought(AcronymExpander)

    def forward(self, texto, acronimo):
        response = None
        try:
            response = self.expander(TEXTO=texto, ACRONIMO=acronimo)
            print(
                f"En el forward la respuesta para el texto '{texto}' es:{response}")
        except Exception as e:
            print(f"-- -- Error expanding acronym: {e}")
            response = None

        return dspy.Prediction(
            texto=texto,
            acronimo=acronimo,
            expansion=response.EXPANSION if response else None
        )


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
        if not do_train:
            if not pathlib.Path(trained_promt).exists():
                self._logger.error("-- -- Trained prompt not found. Exiting.")
                return
            self.module = AcronymExpanderModule.load(trained_promt)

            self._logger.info(
                f"-- -- AcronymExpanderModule loaded from {trained_promt}")
        else:
            if not data_path:
                self._logger.error(
                    "-- -- Data path is required for training. Exiting.")
                return
            
            self.trf_model = SentenceTransformer(trf_model)
            self._logger.info(f"-- -- Transformer model {trf_model} loaded.")
            self._logger.info(
                f"-- -- Training AcronymExpanderModule starts...")
            self.module = self.optimize_module(data_path)
            self._logger.info("-- -- AcronymExpanderModule optimized.")
            self.module.save(trained_promt)

    def validate_expansion(self, example, pred, trace=None):

        prediction_embedding = self.trf_model.encode(pred.expansion)
        reference_embedding = self.trf_model.encode(example.expansion_correcta)

        print("La pred embedding es:", prediction_embedding)
        print("La ref embedding es:", reference_embedding)

        # Calcular la similitud del coseno
        similarity = cosine_similarity([prediction_embedding], [
            reference_embedding])[0][0]

        # Evaluar si la similitud está por encima de un umbral-> [-1,1]
        if similarity > 0.9:
            # Considerando que si está por encima de 0.9 es un acierto
            print(similarity)
            return True
        else:
            print(similarity)
            return False

    def create_examples(self, df):

        examples = []
        for index, row in df.iterrows():
            # Asegurar que ambos campos, acrónimos y expansiones, contienen datos
            if pd.notna(row['detected_acronyms_LLaMA']) and pd.notna(row['expanded_LLaMA']):
                acronyms_list = [acr.strip()
                                 for acr in row['detected_acronyms_LLaMA'].split(',')]
                expansions_list = [exp.strip()
                                   for exp in row['expanded_LLaMA'].split(',')]

                # Verificar que ambos listados tengan el mismo número de elementos
                if len(acronyms_list) != len(expansions_list):
                    print(
                        f"Error en la fila {index}: Número desigual de acrónimos y expansiones.")
                    print(f"Fila problemática: {row.to_dict()}")
                    continue

                # Crear un ejemplo por cada acrónimo y su correspondiente expansión
                for acr, exp in zip(acronyms_list, expansions_list):
                    examples.append(dspy.Example({
                        'texto': row['text'],
                        'acronimo': acr,
                        'expansion_correcta': exp
                    }).with_inputs('texto', 'acronimo'))
            else:
                print(f"Datos faltantes en la fila {index}.")
                print(f"Fila problemática: {row.to_dict()}")

        return examples

    def optimize_module(self, data_path, max_bootstrapped_demos=4, max_labeled_demos=16, num_candidate_programs=16, max_rounds=1):
        """
        Optimiza el módulo AcronymExpander.
        """

        # Read data
        df = pd.read_excel(data_path)
        print("-- -- Data read from file. Head: {}".format(df.head()))
        df = df[['text', 'manual_expanded_acronyms',
                'detected_acronyms_LLaMA', 'expanded_LLaMA']]

        # Create DSPY examples
        ex = self.create_examples(df)

        # Split data
        # @ TODO: Add dev set for tuning and leave test for testing
        tr_ex, test_ex = train_test_split(ex, test_size=0.2, random_state=42)

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
            valset=test_ex
        )

        return compiled_model