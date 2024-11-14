import pathlib
import logging
import os
import dspy
from dspy import Prediction
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import time
from dotenv import load_dotenv
import dspy
import logging
import json
import time
import ujson
import numpy as np
import pandas as pd
import pathlib
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from typing import List, Dict, Any, Optional, Union
import ast
import unidecode
import itertools
from sentence_transformers.util import cos_sim
from dspy.datasets import Dataset
from lingua import Language, LanguageDetectorBuilder
#from src.topicmodeling.topic_model import BERTopicModel, CtmModel, MalletLdaModel, TopicGPTModel
from dspy.evaluate import Evaluate

class TopicLabellerDataset(Dataset):
    def __init__(
        self,
        data_fpath: Union[pathlib.Path, str],
        input_keys: List[str] = ["tpc_description"],
        label_key: str = "tpc_labels",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.labels = []
        self._data = []
        
        # Determinar el tipo de archivo
        data_fpath = pathlib.Path(data_fpath)
        if data_fpath.suffix == '.xlsx':
            # Leer los datos desde un archivo Excel
            data = pd.read_excel(data_fpath)
            data = data[['tpc_description', 'tpc_labels']]

            # Convertir a lista de ejemplos
            self._data = [
                dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(data)
            ]
        elif data_fpath.suffix == '.txt':
            # Leer el archivo de texto y procesar los tópicos
            # Leer el archivo de texto y procesar los tópicos
            topics = []
            with open(data_fpath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            topic_num = parts[0]
                            topic_words = parts[2]
                            words_list = topic_words.split()
                            topic_description = ', '.join(words_list)
                            topics.append({'tpc_description': topic_description, 'tpc_labels': None})
            data = pd.DataFrame(topics)
            # Convertir a lista de ejemplos
            self._data = [
                dspy.Example({**row}).with_inputs(*input_keys) for row in topics
            ]
        else:
            raise ValueError(f"Unsupported file extension: {data_fpath.suffix}")

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')

class TranslatorModuleEngEs(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("english->spanish")

    def forward(self, english_text):
        """
        Traduce un texto del inglés al español.
        """
        spanish_text = self.translate(english=english_text).spanish
        return spanish_text

class TranslatorModuleEsEng(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("spanish->english")

    def forward(self, spanish_text):
        """
        Traduce un texto del español al inglés.
        """
        english_text = self.translate(spanish=spanish_text).english
        return english_text

class TopicLabeller(dspy.Signature):
    """
    Generate a topic label in the LANGUAGE that summarize the tpc_description.
    ----------------------------------------------------------------------------
    Examples
    --------
    tpc_description: "Cirugía, Cardiología, Neurología, Oncología, Pediatría, Ginecología, Psiquiatría, Dermatología, Endocrinología, Medicina Interna, Infectología"]
    LANGUAGE: SPANISH 
    LABEL: 'Especialidades de Medicina'
    
    tpc_description: "machine, learning, data, train, prediction, dataset, algorithm, computer, computing, engine, memory, sensor, artificial_intelligence, identification, interface"
    LANGUAGE: ENGLISH 
    LABEL: 'Machine Learning'
    ----------------------------------------------------------------------------
    """
    tpc_description = dspy.InputField()
    LANGUAGE = dspy.InputField()
    LABEL = dspy.OutputField()
    
class TopicLabellerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.labeller = dspy.Predict(TopicLabeller)
        self.translatorEngEs = TranslatorModuleEngEs()
        self.translatorEsEng = TranslatorModuleEsEng()
        self._language_detector = None
        self.no_expansion_variations = [
            '/', '/ (no se expande)', '', '${LABEL}', '${ACRONIMOS}', '/ (no hay expansión)',
            'Write the LABEL in the language ENGLISH', 'no se puede expandir', '${EXPANSION}', '${ACRONYMS}', '${ACRONYM}',
            'Write the label in the language english', 'TEXT', '${TEXT}', '  ',
            'write the label in the language english', "'/'", 'N/A', '/.', 'N_A', 'NA'
        ]
    @property
    def language_detector(self):
        # Crear y devolver una instancia de LanguageDetectorModule si aún no se ha creado
        if not self._language_detector:
            self._language_detector = LanguageDetectorModule()
        return self._language_detector
    
    def dump_state(self, save_verbose=False, ensure_ascii=False, escape_forward_slashes=False):
        print(self.named_parameters())
        return {name: param.dump_state() for name, param in self.named_parameters()}
    
    def save(self, path, save_field_meta=False):
        print("*"*50)
        with open(path, "w") as f:
            f.write(ujson.dumps(self.dump_state(save_field_meta), indent=2, ensure_ascii=False, escape_forward_slashes=False))
            
    def _process_label(self, label, descriptions):
        """
        Procesa y valida la etiqueta generada para asegurar que sea coherente y adecuada.
        """
        normalized_label = self.normalize_text(label)

        if not normalized_label or normalized_label in self.no_expansion_variations:
            print("La etiqueta generada es inválida o vacía.")
            return "/"

        if len(normalized_label) > 120:
            print("La etiqueta generada es demasiado larga.")
            return "/"

        if len(re.findall(r'\d', normalized_label)) > 2:
            print("La etiqueta generada contiene demasiados números.")
            return "/"

        return normalized_label
    
    def normalize_text(self, text):
        """
        Normaliza el texto convirtiéndolo a minúsculas, eliminando acentos y puntuación.
        """
        text = unidecode.unidecode(text.lower())
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        return text

    def forward(self, tpc_description):
        """
        Genera una etiqueta de tópico basado en las descripciones y el idioma detectado.
        """
        # Detectar el idioma de las descripciones de los tópicos
        detected_language = self.language_detector.detect_language(tpc_description)
        print(f"Idioma detectado de las descripciones: {detected_language}")

        # Generar la etiqueta usando el predictor
        response = self.labeller(tpc_description=tpc_description, LANGUAGE=detected_language)
        generated_label = response.LABEL
        print(f"LABEL generada: {generated_label}")

        detected_label_language = self.language_detector.detect_language(generated_label)
        print(f"Idioma etiqueta generada: {detected_label_language}")
            
        if (detected_label_language == 'ENGLISH' or detected_label_language == 'CATALAN' or detected_label_language == 'BASQUE') and detected_language == 'SPANISH':
                print(f"Traduciendo expansión de inglés a español: {generated_label}")
                translator = TranslatorModuleEngEs()
                label_translated = self.translatorEngEs.forward(generated_label)
                print("Etiqueta traducida:", label_translated)
                # Validar y procesar la etiqueta generada
                processed_label = self._process_label(label_translated, tpc_description)
                
        if (detected_label_language == 'SPANISH' or detected_label_language == 'CATALAN' or detected_label_language == 'BASQUE') and detected_language == 'ENGLISH':
                print(f"Traduciendo expansión de español a inglés: {generated_label}")
                translator = TranslatorModuleEsEng()
                label_translated = self.translatorEsEng.forward(generated_label)
                print("Etiqueta traducida:", label_translated)
                processed_label = self._process_label(label_translated, tpc_description)

        processed_label = self._process_label(generated_label, tpc_description)
        return dspy.Prediction(LABEL=processed_label)

class LanguageDetectorModule:
    def __init__(self):
        self._detector = None
        self.languages = [Language.ENGLISH, Language.SPANISH, Language.BASQUE, Language.CATALAN]

    @property
    def detector(self):
        if self._detector is None:
            self._detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        return self._detector

    def detect_language(self, text):
        detected_language = self.detector.detect_language_of(text)
        return detected_language.name if detected_language else 'Indefinido'
    
class HermesTopicLabeller:
    def __init__(
        self,
        model_type: str = "llama",
        do_train: bool = False,
        data_path: str = None,
        trained_prompt: str = pathlib.Path(
            __file__).parent.parent.parent / "data/optimized/HermesTopicLabeller-saved-pruebas.json",
        trf_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        logger: logging.Logger = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs",
        results_path: str = None,
        N: int = 10
    ):
        self._logger = logger if logger else logging.getLogger(__name__)
        self.language_detector = LanguageDetectorModule()
        self.module = TopicLabellerModule()
        self.num_iterations = N
                
        # Configuración de dspy
        if model_type == "llama":
            self.lm = dspy.HFClientTGI(model="meta-llama/Meta-Llama-3-8B", port=8090, url="http://127.0.0.1")
            
        dspy.settings.configure(lm=self.lm, temperature=0)
        
        self.results_path = results_path
        
        if not do_train:
            if not pathlib.Path(trained_prompt).exists():
                self._logger.error("No se encontró el prompt entrenado. Saliendo.")
                return
            self.module = TopicLabellerModule()
            self.module.load(trained_prompt)
            self._logger.info(f"TopicLabellerModule cargado desde {trained_prompt}") 
        if do_train:
            if not data_path:
                self._logger.error("Se requiere la ruta de datos para el entrenamiento. Saliendo.")
                return
            self.trf_model = SentenceTransformer(trf_model)
            self._logger.info(f"Modelo de transformador {trf_model} cargado.")
            self._logger.info("Entrenando TopicLabellerModule...")
            self.module = self.optimize_module(data_path)
            self._logger.info("TopicLabellerModule optimizado.")
            self.module.save(trained_prompt)
        else:
            self.trf_model = SentenceTransformer(trf_model)
            self.module.load(trained_prompt)
            self._logger.info(f"TopicLabellerModule prompt cargado desde {trained_prompt}")
            # Si do_train es False, se ejecuta la evaluación con el módulo cargado
            self.evaluate_loaded_module(data_path)
    
    def evaluate_loaded_module(self, data_path):
        # Creamos el dataset
        dataset = TopicLabellerDataset(
            data_fpath=data_path
        )   
        # Evaluamos el conjunto completo
        self.evaluate_and_display(
            dataset._data, "completo",
            save_path=pathlib.Path(self.results_path) / 'results_label.xlsx' if self.results_path else None
        )

    def validate_label(self, example, pred, trace=None):
        """
        Valida la etiqueta generada comparando embeddings del modelo transformer.
        """
        try:
            # Codificar la etiqueta predicha y de referencia usando el modelo transformer
            prediction_embedding = self.trf_model.encode(str(pred.LABEL))
            reference_embedding = self.trf_model.encode(str(example['tpc_labels']))

            # Calcular similitud coseno entre embeddings
            similarity = cosine_similarity([prediction_embedding], [reference_embedding])[0][0]
            self._logger.info(f"Similitud coseno: {similarity}")

            # Verificar si la similitud supera el umbral (0.7)
            return similarity > 0.7
        except Exception as e:
            self._logger.error(f"Error durante la validación de la etiqueta: {e}")
            return False

    def evaluate_variations(self, labels):
        try:
            # Codificar las etiquetas usando el modelo transformer
            embeddings = self.trf_model.encode(labels)
            similarities = cosine_similarity(embeddings)
            # Calculamos la variación como 1 - similitud promedio
            variation = 1 - np.mean(similarities)
            return variation
        except Exception as e:
            self._logger.error(f"Error al calcular las variaciones: {e}")
            return 0.0
        
    def evaluate_and_display(self, dataset, dataset_name, save_path=None):
        """
        Evalúa las etiquetas generadas para un dataset, haciendo múltiples iteraciones para cada tópico
        y guarda los resultados en un archivo Excel si se especifica `save_path`.
        """
        results = []
        all_labels = []  # Lista para almacenar todas las etiquetas generadas

        for el in dataset:
            labels_for_topic = []  # Etiquetas generadas para un solo tópico

            # Realizar N iteraciones para cada descripción de tópico
            for _ in range(self.num_iterations):
                output = self.module.forward(el.tpc_description)
                label = output.LABEL
                labels_for_topic.append(label)

            all_labels.extend(labels_for_topic)  # Agregar etiquetas generadas al conjunto total

            # Evaluar la variación de las etiquetas generadas
            variation_score = self.evaluate_variations(labels_for_topic)
            distinct_count = len(set(labels_for_topic))

            # Agregar resultados al DataFrame
            results.append([el.tpc_description, el.tpc_labels, labels_for_topic, distinct_count, variation_score])

        # Crear un DataFrame con los resultados
        df = pd.DataFrame(
            results, columns=["Descriptions", "Ground ", "Generated Labels", "Distinct Count", "Variation Score"]
        )
        
        avg_metric = df['Variation Score'].mean()
        print(f"\n## Resultados del conjunto {dataset_name} ##")
        print(df.head())
        print(f"Puntuación media de variación en {dataset_name}: {avg_metric:.4f}")
        
        # Guardar los resultados en un archivo Excel si se especifica una ruta
        if save_path:
            df.to_excel(save_path, index=False)
            print(f"Resultados guardados en {save_path}")
        
        return df, avg_metric
        
    def evaluate_labels(self, dataset):
        """
        Evalúa las etiquetas generadas, midiendo la distinción y variación en las etiquetas.
        Realiza múltiples iteraciones para cada tópico.
        """
        all_labels = [] 
        distinct_labels = set()

        for el in dataset:
            labels_for_topic = []  

            # Realizar 10 iteraciones para cada descripción de tópico
            for _ in range(self.num_iterations):
                output = self.module.forward(el.tpc_description)
                label = output.LABEL
                labels_for_topic.append(label)
                distinct_labels.add(label)  

            all_labels.append(labels_for_topic)

        # Evaluar la variación de etiquetas generadas usando similitud coseno
        all_flat_labels = [label for sublist in all_labels for label in sublist] 
        variation_score = self.evaluate_variations(all_flat_labels)
        distinct_count = len(distinct_labels)

        print(f"Número de etiquetas distintas: {distinct_count}")
        print(f"Puntuación media de variación (similitud coseno): {variation_score:.4f}")

        return distinct_count, variation_score, all_labels
    '''
    def optimize_module(self, data_path, mbd=4, mld=16, ncp=16, mr=1, dev_size=0.25):
        """
        Optimiza el TopicLabellerModule basado en los datos proporcionados y evalúa en los conjuntos de entrenamiento, validación y prueba.
        """
        # Crear dataset
        dataset = TopicLabellerDataset(
            data_fpath=data_path,
            dev_size=dev_size,
            datasets_path=self.datasets_path
        )
                
        self._logger.info(f"Dataset cargado desde {data_path}")

        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test

        self._logger.info(
            f"-- -- Dataset dividido en train, dev y test. Entrenando el módulo...")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                    num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.validate_label, **config)

        compiled_pred = teleprompter.compile(TopicLabellerModule(), trainset=trainset, valset=devset)

        self._logger.info("Módulo compilado. Evaluando en los conjuntos de entrenamiento, validación y prueba...")
        
        def evaluate_and_display(dataset, dataset_name, save_path=None):
            results = []
            for el in dataset:
                output = self.module.forward(el.tpc_description)
                print(f"Output para el texto '{el.tpc_description}': {output} (tipo: {type(output)})")
                metric_score = self.validate_label(el, output)
                results.append([el.tpc_description, el.tpc_labels,
                                output.LABEL,
                                metric_score])

            df = pd.DataFrame(
                results, columns=["Descriptions", "Ground Label", "Predicted Label", "Metric"])
            
            avg_metric = df['Metric'].mean()
            print(f"\n## Resultados del conjunto {dataset_name} ##")
            print(df.head())
            print(f"Puntuación media en {dataset_name}: {avg_metric:.4f}")
            
            if save_path:
                df.to_excel(save_path, index=False)
                print(f"Resultados guardados en {save_path}")
            
            return df, avg_metric
        
        # Evaluar en los conjuntos y guardar los resultados
        train_df, train_score = evaluate_and_display(
            trainset, "de entrenamiento",
            save_path=pathlib.Path(self.results_path) / 'train_results_label.xlsx' if self.results_path else None)

        dev_df, dev_score = evaluate_and_display(
            devset, "de validación",
            save_path=pathlib.Path(self.results_path) / 'dev_results_label.xlsx' if self.results_path else None)

        test_df, test_score = evaluate_and_display(
            testset, "de prueba",
            save_path=pathlib.Path(self.results_path) / 'test_results_label.xlsx' if self.results_path else None)

        print("\n## Resumen de puntuaciones ##")
        print(f"Puntuación media en entrenamiento: {train_score:.4f}")
        print(f"Puntuación media en validación: {dev_score:.4f}")
        print(f"Puntuación media en prueba: {test_score:.4f}")

        return compiled_pred
    '''
    
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    data_path = '/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/optimized/4.training/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE/MalletLda_14/modelFiles/topickeys.txt'
    #data_path = '/export//usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/scholar/optimized/4.training/S2_Kwds3_AI_with_text_30000_both/MalletLda_30/modelFiles/topickeys.txt'
    trained_prompt = '/export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/trained_prompt_topic_labeller.json'  
    #datasets_path = '/export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/topic_labeller'  
    results_path = '/export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/topic_labeller/PLACE'  
    
    do_train = False 
    N = 10 
    
    hermes_labeller = HermesTopicLabeller(
        model_type="llama", 
        do_train=do_train,
        data_path=data_path,
        trained_prompt=trained_prompt,
        results_path=results_path,
        logger=logger,
        N=N
    )
   
if __name__ == "__main__":
    main()
