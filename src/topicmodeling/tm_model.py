"""This module is similar to the one available in the topicmodeler (https://github.com/IntelCompH2020/topicmodeler/blob/main/src/topicmodeling/manageModels.py). It provides a generic representation of all topic models used for curation purposes.

Authors: Jerónimo Arenas-García, J.A. Espinosa-Melchor, Lorena Calvo-Bartolomé, Carlos González Gamella
Modifed: 24/01/2024 (Updated for NP-Solr-Service (NextProcurement Proyect))
Modified: 11/02/2024 (Updated for NP-Search-Tool (NextProcurement Proyect) to include topic labelling method based on OpenAI's GPT-X models)
Modified: 12/11/2024 (Updated coherence calculation with refence text dumped from the wikipedia for spanish text coherence calculation)
"""

import shutil
import warnings
from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import xml.etree.ElementTree as ET
#from gensim.models.coherencemodel import CoherenceModel
#from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import scipy.sparse as sparse
from sparse_dot_topn import awesome_cossim_topn
#from gensim.models.coherencemodel import CoherenceModel
#from src.Embeddings.embedder import Embedder
#from topic_labeller import TopicLabeller


class TMmodel(object):
    # This class represents a Topic Model according to the LDA generative model
    # Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # and needs to be backed up with a folder in which all the associated
    # files will be saved
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox
    # that produces a model according to this representation

    # The following variables will store original values of matrices alphas, betas, thetas
    # They will be used to reset the model to original values

    _TMfolder = None

    _betas_orig = None
    _thetas_orig = None
    _alphas_orig = None

    _betas = None
    _thetas = None
    _alphas = None
    _edits = None  # Store all editions made to the model
    _ntopics = None
    _betas_ds = None
    _coords = None
    _topic_entropy = None
    _topic_coherence = None
    _ndocs_active = None
    _tpc_descriptions = None
    _tpc_labels = None
    _vocab_w2id = None
    _vocab_id2w = None
    _vocab = None
    _size_vocab = None
    _sims = None

    def __init__(self, TMfolder, logger=None):
        """Class initializer

        We just need to make sure that we have a folder where the
        model will be stored. If the folder does not exist, it will
        create a folder for the model

        Parameters
        ----------
        TMfolder: Path
            Contains the name of an existing folder or a new folder
            where the model will be created
        logger:
            External logger to use. If None, a logger will be created for the object
        """
        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('TMmodel')

        # Convert strings to Paths if necessary
        self._TMfolder = Path(TMfolder)

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            try:
                self._TMfolder.mkdir(parents=True)
            except:
                self._logger.error(
                    '-- -- Topic model object (TMmodel) could not be created')

        self._logger.info(
            '-- -- -- Topic model object (TMmodel) successfully created')
    
    def get_mean_coherence(self):
        return self.mean_coherence

    def get_topic_coherences(self):
        return self.topic_coherences

    def get_measure_name(self):
        return self.measure_name

    def create(self, betas=None, thetas=None, alphas=None, vocab=None):
        """
        Creates the topic model from the relevant matrices that characterize it.
        In addition to the initialization of the corresponding object's variables,
        all the associated variables and visualizations which are computationally costly
        are calculated so they are available for the other methods.

        Parameters
        ----------
        betas:
            Matrix of size n_topics x n_words (vocab of each topic)
        thetas:
            Matrix of size  n_docs x n_topics (document composition)
        alphas: 
            Vector of length n_topics containing the importance of each topic
        vocab: list
            List of words sorted according to betas matrix
        """

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            self._logger.error(
                '-- -- Topic model object (TMmodel) folder not ready')
            return

        self._alphas_orig = alphas
        self._betas_orig = betas
        self._thetas_orig = thetas
        self._alphas = alphas
        self._betas = betas
        self._thetas = thetas
        self._vocab = vocab
        self._size_vocab = len(vocab)
        self._ntopics = thetas.shape[1]
        self._edits = []

        # Save original variables
        np.save(self._TMfolder.joinpath('alphas_orig.npy'), alphas)
        np.save(self._TMfolder.joinpath('betas_orig.npy'), betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas_orig.npz'), thetas)
        with self._TMfolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(vocab))

        # Initial sort of topics according to size. Calculate other variables
        self._sort_topics()
        self._logger.info("-- -- Sorted")
        self._calculate_beta_ds()
        self._logger.info("-- -- betas ds")
        self._calculate_topic_entropy()
        self._logger.info("-- -- entropy")
        self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])
        self._logger.info("-- -- active")
        self._tpc_descriptions = [el[1]
                                  for el in self.get_tpc_word_descriptions()]
        self._logger.info("-- -- descriptions")
        self.calculate_gensim_dic()
        print("Coherence al crear el TMmodel:")
        # Parámetros para calcular coherencia
        coherence_measure = 'c_v'  
        top_n = 15  
        file_path = "/export/usuarios_ml4ds/cggamella/RAG_tool/data/dump/eswiki-latest-abstract.xml.gz"

        self._logger.info("Calculando la coherencia...")
        measure_name, mean_coherence, topic_coherences = self.calculate_topic_coherence(
            coherence_measure=coherence_measure,
            top_n=top_n,
            file_path=file_path
        )
        
        self.measure_name = measure_name
        self.mean_coherence = mean_coherence
        self.topic_coherences = topic_coherences

        # Loguear resultados de coherencia
        print(f"Measure Name: {self.measure_name}")
        print(f"Mean Coherence: {self.mean_coherence}")
        print(f"Topic Coherences: {self.topic_coherences}")
 
        #self._tpc_labels = [el[1] for el in self.get_tpc_labels()]
        #self._tpc_embeddings = self.get_tpc_word_descriptions_embeddings()
        self._calculate_sims()

        # We are ready to save all variables in the model
        self._save_all()

        self._logger.info(
            '-- -- Topic model variables were computed and saved to file')
        return

    def _save_all(self):
        """Saves all variables in Topic Model
        * alphas, betas, thetas
        * edits
        * betas_ds, topic_entropy, ndocs_active
        * tpc_descriptions, tpc_labels
        This function should only be called after making sure all these
        variables exist and are not None
        """
        np.save(self._TMfolder.joinpath('alphas.npy'), self._alphas)
        np.save(self._TMfolder.joinpath('betas.npy'), self._betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas.npz'), self._thetas)
        sparse.save_npz(self._TMfolder.joinpath('distances.npz'), self._sims)

        with self._TMfolder.joinpath('edits.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._edits))
        np.save(self._TMfolder.joinpath('betas_ds.npy'), self._betas_ds)
        np.save(self._TMfolder.joinpath(
            'topic_entropy.npy'), self._topic_entropy)
        np.save(self._TMfolder.joinpath(
            'topic_coherence.npy'), self._topic_coherence)
        np.save(self._TMfolder.joinpath(
            'ndocs_active.npy'), self._ndocs_active)
        with self._TMfolder.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_descriptions))
        #with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
        #    fout.write('\n'.join(self._tpc_labels))
        #np.save(self._TMfolder.joinpath('tpc_embeddings.npy'), np.array(
        #    self._tpc_embeddings, dtype=object), allow_pickle=True)

        # Generate also pyLDAvisualization
        # pyLDAvis currently raises some Deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyLDAvis

        # We will compute the visualization using ndocs random documents
        # In case the model has gone through topic deletion, we may have rows
        # in the thetas matrix that sum up to zero (active topics have been
        # removed for these problematic documents). We need to take this into
        # account
        ndocs = 10000
        validDocs = np.sum(self._thetas.toarray(), axis=1) > 0
        nValidDocs = np.sum(validDocs)
        if ndocs > nValidDocs:
            ndocs = nValidDocs
        perm = np.sort(np.random.permutation(nValidDocs)[:ndocs])
        # We consider all documents are equally important
        doc_len = ndocs * [1]
        vocabfreq = np.round(ndocs*(self._alphas.dot(self._betas))).astype(int)
        try:
            vis_data = pyLDAvis.prepare(
                self._betas,
                self._thetas[validDocs, ][perm, ].toarray(),
                doc_len,
                self._vocab,
                vocabfreq,
                lambda_step=0.05,
                sort_topics=False,
                n_jobs=-1)
            
            with self._TMfolder.joinpath("pyLDAvis.html").open("w") as f:
                pyLDAvis.save_html(vis_data, f)
            self._logger.info("Visualización pyLDAvis guardada exitosamente.")
        except Exception as e:
            self._logger.error(f"Error al generar pyLDAvis: {e}")
            
        # TODO: Check substituting by "pyLDAvis.prepared_data_to_html"
        # self._modify_pyldavis_html(self._TMfolder.as_posix())

        # Get coordinates of topics in the pyLDAvis visualization
        vis_data_dict = vis_data.to_dict()
        self._coords = list(
            zip(*[vis_data_dict['mdsDat']['x'], vis_data_dict['mdsDat']['y']]))

        with self._TMfolder.joinpath('tpc_coords.txt').open('w', encoding='utf8') as fout:
            for item in self._coords:
                fout.write(str(item) + "\n")

        return

    def _save_cohr(self):

        np.save(self._TMfolder.joinpath(
            'topic_coherence.npy'), self._topic_coherence)

    def _modify_pyldavis_html(self, model_dir):
        """
        Modifies the PyLDAvis HTML file returned by the Gensim library to include the direct paths of the 'd3.js' and 'ldavis.v3.0.0.js', which are copied into the model/submodel directory.

        Parameters
        ----------
        model_dir: str
            String representation of the path wwhere the model/submodel is located
        """

        # Copy necessary files in model / submodel folder for PyLDAvis visualization
        d3 = Path("src/gui/resources/d3.js")
        v3 = Path("src/gui/resources/ldavis.v3.0.0.js")
        shutil.copyfile(d3, Path(model_dir, "d3.js"))
        shutil.copyfile(v3, Path(model_dir, "ldavis.v3.0.0.js"))

        # Update d3 and v3 paths in pyldavis.html
        fin = open(Path(model_dir, "pyLDAvis.html").as_posix(),
                   "rt")  # read input file
        data = fin.read()  # read file contents to string
        # Replace all occurrences of the required string
        data = data.replace(
            "https://d3js.org/d3.v5.js", "d3.js")
        data = data.replace(
            "https://d3js.org/d3.v5", "d3.js")
        data = data.replace(
            "https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", "ldavis.v3.0.0.js")
        fin.close()  # close the input file
        fin = open(Path(model_dir, "pyLDAvis.html").as_posix(),
                   "wt")  # open the input file in write mode
        fin.write(data)  # overrite the input file with the resulting data
        fin.close()  # close the file

        return

    def _sort_topics(self):
        """Sort topics according to topic size"""

        # Load information if necessary
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_edits()

        # Indexes for topics reordering
        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        # Sort data matrices
        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx, :]
        self._thetas = self._thetas[:, idx]

        return

    def _load_alphas(self):
        if self._alphas is None:
            self._alphas = np.load(self._TMfolder.joinpath('alphas.npy'))
            self._ntopics = self._alphas.shape[0]

    def _load_betas(self):
        if self._betas is None:
            self._betas = np.load(self._TMfolder.joinpath('betas.npy'))
            self._ntopics = self._betas.shape[0]
            self._size_vocab = self._betas.shape[1]

    def _load_thetas(self):
        if self._thetas is None:
            self._thetas = sparse.load_npz(
                self._TMfolder.joinpath('thetas.npz'))
            self._ntopics = self._thetas.shape[1]
            # self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])

    def _load_ndocs_active(self):
        if self._ndocs_active is None:
            self._ndocs_active = np.load(
                self._TMfolder.joinpath('ndocs_active.npy'))
            self._ntopics = self._ndocs_active.shape[0]

    def _load_edits(self):
        if self._edits is None:
            with self._TMfolder.joinpath('edits.txt').open('r', encoding='utf8') as fin:
                self._edits = fin.readlines()

    def _calculate_beta_ds(self):
        """Calculates beta with down-scoring
        Emphasizes words appearing less frequently in topics
        """
        # Load information if necessary
        self._load_betas()

        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds)) /
                          self._ntopics), (self._size_vocab, 1))
        deno = np.ones((self._ntopics, 1)).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)

    def _load_betas_ds(self):
        if self._betas_ds is None:
            self._betas_ds = np.load(self._TMfolder.joinpath('betas_ds.npy'))
            self._ntopics = self._betas_ds.shape[0]
            self._size_vocab = self._betas_ds.shape[1]

    def _load_vocab(self):
        if self._vocab is None:
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                self._vocab = [el.strip() for el in fin.readlines()]

    def _load_vocab_dicts(self):
        """Creates two vocabulary dictionaries, one that utilizes the words as key, and a second one with the words' id as key. 
        """
        if self._vocab_w2id is None and self._vocab_w2id is None:
            self._vocab_w2id = {}
            self._vocab_id2w = {}
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    wd = line.strip()
                    self._vocab_w2id[wd] = i
                    self._vocab_id2w[str(i)] = wd

    def _calculate_topic_entropy(self):
        """Calculates the entropy of all topics in model
        """
        # Load information if necessary
        self._load_betas()

        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = - \
            np.sum(self._betas * np.log(self._betas), axis=1)
        self._topic_entropy = self._topic_entropy / np.log(self._size_vocab)

    def _load_topic_entropy(self):
        if self._topic_entropy is None:
            self._topic_entropy = np.load(
                self._TMfolder.joinpath('topic_entropy.npy'))

    def calculate_gensim_dic(self):
        
        # TODO: Check this is working when Mallet is not being used
        corpusFile = self._TMfolder.parent.joinpath(
            'modelFiles/corpus.txt')
        self._logger.info(f"Buscando corpus.txt en: {corpusFile}")
        
        if not corpusFile.exists():
            self._logger.error(f"El archivo {corpusFile} no existe.")
            return

        with corpusFile.open("r", encoding="utf-8") as f:
            corpus = [line.rsplit(" 0 ")[1].strip().split() for line in f.readlines()
                      if line.rsplit(" 0 ")[1].strip().split() != []]
        
        # Import necessary modules for coherence calculation with Gensim
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
        
        # Create dictionary
        dictionary = Dictionary(corpus)

        # Save dictionary
        GensimFile = self._TMfolder.parent.joinpath('dictionary.gensim')
        dictionary.save_as_text(GensimFile)
        

        with self._TMfolder.parent.joinpath('vocabulary.txt').open('w', encoding='utf8') as fout:
            fout.write(
                '\n'.join([dictionary[idx] for idx in range(len(dictionary))]))

        return
    
    def _load_reference_text(self, file_path, limit=1500000):
        """
        Load reference text dump from wikipedia for coherence calculation.
        Abstract <tag> text from the wikipedia dump is used for coherence calculation.
        """
        reference_text = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for doc in root.findall("doc"):
                abstract = doc.find("abstract").text or ""
                if abstract:
                    reference_text.append(abstract.split())
                if len(reference_text) >= limit:
                    break
        return reference_text

    def _gen_measure_name(self,coherence_measure, window_size, top_n):
        """
        Make a unique measure name from the arguments
        """
        measure_name = f"{coherence_measure}_win{window_size}_top{top_n}"
        return measure_name
    
    # NEW  -- Function to calculate the coherence of the topics -- 
    def calculate_topic_coherence(self, coherence_measure, top_n, file_path):
        '''
        This function calculates the coherence of the topics using the Gensim CoherenceModel.
        
        Parameters:
            - reference_text: text dump from the wikipedia for spanish text coherence calculation
            - coherence_measure: name of the coherence measure to use (c_v, c_npmi)
            - top_n: number of words to consider in the coherence calculation
            - file_path: path to the file with the reference text from wikipedia 
        
        Return:
            - Name of coherence model (str).
            - Mean coherence of the topics (float).
            - List of individual coherences for each topic (list).
        '''

        self._logger.info("Starting coherence calculation {coherence_measure}...")

        # Load chemical descriptions of the topics
        if self._tpc_descriptions is None:
            # Get the topic descriptions as strings
            self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions(n_words=top_n)]

        # Convert topic descriptions to lists of words
        topics_ = [desc.split(', ')[:top_n] for desc in self._tpc_descriptions]
        self._logger.info("Topic descriptions converted to lists.")
        reference_text = self._load_reference_text(file_path)
        vocab = Dictionary(reference_text)
        self._logger.info("Dictionary created from reference text.")
        
        try:
            cm = CoherenceModel(
                topics=topics_,
                texts=tqdm(reference_text),
                dictionary=vocab,
                coherence=coherence_measure,
            )
            confirmed_measures = cm.get_coherence_per_topic()        
            mean = cm.aggregate_measures(confirmed_measures)
            self._logger.info(f"Coherencia media calculada: {mean}")
            # Guardar las coherencias de los tópicos
            self._topic_coherence = confirmed_measures
            #self._save_cohr() # Ya se guarda en el método _save_all()
            print("Dentro del cáculo de la coherencia")
            import pdb; pdb.set_trace()
            
        except ZeroDivisionError as e:
            self._logger.error(f"Error al calcular coherencia: {e}")
            #confirmed_measures = ['inf' if x == float('inf') else x for x in confirmed_measures]
            confirmed_measures = [float('inf') if x == float('inf') else x for x in confirmed_measures]
            mean = float('nan')
            self._topic_coherence = confirmed_measures
            #self._save_cohr() # Ya se guarda en el método _save_all()

        measure_name = self._gen_measure_name(coherence_measure, cm.window_size, top_n)
        return measure_name, float(mean), [float(i) for i in confirmed_measures]
    
    '''
    def xx_calclate_topic_coherence(self, coherence_measure='c_v', top_n=10):
        
        # Cargar las descripciones de los tópicos
        if self._tpc_descriptions is None:
            self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions(n_words=top_n)]

        topics = [tpc.split(', ') for tpc in self._tpc_descriptions]
        self._logger.info(f"Descripciones de tópicos convertidas en listas.")
        

        # Cargar el corpus directamente
        self._logger.info("--- Calculando coherencia con datos del corpus.")
        corpus_file = self._TMfolder.parent.joinpath('modelFiles', 'corpus.txt')
        self._logger.info(f"Cargando corpus desde: {corpus_file}")

        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                try:
                    reference_text = [line.rsplit(" 0 ")[1].strip().split() for line in lines]
                except IndexError:
                    reference_text = [line.rsplit("\t0\t")[1].strip().split() for line in lines]
            self._logger.info("Corpus cargado correctamente.")
        except FileNotFoundError:
            self._logger.error(f"No se encontró el archivo de corpus en {corpus_file}")
            return
        

        # Importar módulos necesarios de Gensim
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel

        # Cargar el diccionario directamente
        dictionary = None
        dict_file = self._TMfolder.parent.joinpath('dictionary.gensim')
        if dict_file.is_file():
            try:
                dictionary = Dictionary.load_from_text(dict_file.as_posix())
                self._logger.info("Diccionario Gensim cargado.")
            except Exception as e:
                self._logger.warning(f"No se pudo cargar el diccionario Gensim: {e}")
                dictionary = Dictionary(reference_text)
                self._logger.info("Diccionario Gensim creado a partir del corpus.")
        else:
            self._logger.info("Diccionario Gensim no encontrado. Creando diccionario...")
            dictionary = Dictionary(reference_text)
            self._logger.info("Diccionario Gensim creado a partir del corpus.")

        # Calcular la coherencia
        try:
            self._logger.info(f"Calculando coherencia usando la métrica {coherence_measure}...")
            num_cores = os.cpu_count()
            cm = CoherenceModel(
                topics=topics,
                texts=reference_text,
                dictionary=dictionary,
                coherence=coherence_measure,
                topn=top_n,
                processes=num_cores
            )
            confirmed_measures = cm.get_coherence_per_topic()
            for i, coherence_value in enumerate(confirmed_measures):
                self._logger.info(f"Coherencia para el tópico {i}: {coherence_value}")

            mean = cm.get_coherence()
            self._logger.info(f"Coherencia media: {mean}")
            self._topic_coherence = confirmed_measures
            self._save_cohr()
            return confirmed_measures, mean
        except ZeroDivisionError as e:
            self._logger.error(f"ZeroDivisionError: {e}")
            confirmed_measures = ['inf' if x == float('inf') else x for x in confirmed_measures]
            mean = float('nan')
            self._topic_coherence = confirmed_measures
            self._save_cohr()
            return confirmed_measures, mean
    '''

    def _load_topic_coherence(self):
        if self._topic_coherence is None:
            self._topic_coherence = np.load(
                self._TMfolder.joinpath('topic_coherence.npy'))

    def _calculate_sims(self, topn=50, lb=0):
        if self._thetas is None:
            self._load_thetas()
        thetas_sqrt = np.sqrt(self._thetas)
        thetas_col = thetas_sqrt.T
        self._sims = awesome_cossim_topn(thetas_sqrt, thetas_col, topn, lb)

    def _load_sims(self):
        if self._sims is None:
            self._sims = sparse.load_npz(
                self._TMfolder.joinpath('distances.npz'))

    def _largest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        idx0, idx1 = np.unravel_index(indices, ary.shape)
        idx0 = idx0.tolist()
        idx1 = idx1.tolist()
        selected_idx = []
        for id0, id1 in zip(idx0, idx1):
            if id0 < id1:
                selected_idx.append((id0, id1, ary[id0, id1]))
        return selected_idx

    def get_model_info_for_hierarchical(self):
        """Returns the objects necessary for the creation of a level-2 topic model.
        """
        self._load_betas()
        self._load_thetas()
        self._load_vocab_dicts()

        return self._betas, self._thetas, self._vocab_w2id, self._vocab_id2w

    def get_model_info_for_vis(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_vocab()
        self._load_sims()
        self.load_tpc_coords()

        return self._alphas, self._betas, self._thetas, self._vocab, self._sims, self._coords

    def get_tpc_word_descriptions(self, n_words=15, tfidf=True, tpc=None):
        """returns the chemical description of topics

        Parameters
        ----------
        n_words:
            Number of terms for each topic that will be included
        tfidf:
            If true, downscale the importance of words that appear
            in several topics, according to beta_ds (Blei and Lafferty, 2009)
        tpc:
            Topics for which the descriptions will be computed, e.g.: tpc = [0,3,4]
            If None, it will compute the descriptions for all topics  

        Returns
        -------
        tpc_descs: list of tuples
            Each element is a a term (topic_id, "word0, word1, ...")                      
        """

        # Load betas (including n_topics) and vocabulary
        if tfidf:
            self._load_betas_ds()
        else:
            self._load_betas()
        self._load_vocab()

        if not tpc:
            tpc = range(self._ntopics)

        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab[idx2]
                         for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_words]]
            else:
                words = [self._vocab[idx2]
                         for idx2 in np.argsort(self._betas[i])[::-1][0:n_words]]
            tpc_descs.append((i, ', '.join(words)))

        return tpc_descs

    def load_tpc_descriptions(self):
        if self._tpc_descriptions is None:
            with self._TMfolder.joinpath('tpc_descriptions.txt').open('r', encoding='utf8') as fin:
                self._tpc_descriptions = [el.strip() for el in fin.readlines()]

    def get_tpc_labels(self):
        """returns the labels of the topics in the model

        Parameters
        ----------
        labels: list
            List of labels for automatic topic labeling
        use_cuda: bool
            If True, use cuda.

        Returns
        -------
        tpc_labels: list of tuples
            Each element is a a term (topic_id, "label for topic topic_id")                    
        """

        # Load tpc descriptions
        self.load_tpc_descriptions()

        # Create a topic labeller object
        tl = TopicLabeller(model="gpt-4")

        # Get labels
        aux = [string.replace("'", '"') for string in self._tpc_descriptions]
        labels = tl.get_labels(aux)
        labels_format = [(i, p) for i, p in enumerate(labels)]

        return labels_format

    def load_tpc_labels(self):
        if self._tpc_labels is None:
            with self._TMfolder.joinpath('tpc_labels.txt').open('r', encoding='utf8') as fin:
                self._tpc_labels = [el.strip() for el in fin.readlines()]

    def get_tpc_word_descriptions_embeddings(self):

        # Load topc descriptions
        self.load_tpc_descriptions()

        # Create embedder
        emb = Embedder()  # TODO configure parameters

        embed_from = [
            el.split(", ") for el in self._tpc_descriptions
        ]

        corpus_path = self._TMfolder.parent.parent.joinpath(
            'train_data/corpus.txt')

        tpc_embeddings = emb.infer_embeddings(
            embed_from=embed_from,
            method="word2vec",
            do_train_w2vec=True,
            corpus_file=corpus_path
        )

        return tpc_embeddings

    def load_tpc_word_descriptions_embeddings(self):

        if self._tpc_embeddings is None:
            self._tpc_embeddings = np.load(self._TMfolder.joinpath(
                'tpc_embeddings.npy'), allow_pickle=True)

    def load_tpc_coords(self):
        if self._coords is None:
            with self._TMfolder.joinpath('tpc_coords.txt').open('r', encoding='utf8') as fin:
                # read the data from the file and convert it back to a list of tuples
                self._coords = \
                    [tuple(map(float, line.strip()[1:-1].split(', ')))
                        for line in fin]

    def get_alphas(self):
        self._load_alphas()
        return self._alphas

    def showTopics(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        TpcsInfo = [{"Size": str(round(el[0], 4)), "Label": el[1].strip(), "Word Description": el[2].strip(), "Ndocs Active": str(el[3])} for el in zip(
            self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active)]

        return TpcsInfo

    def showTopicsAdvanced(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_topic_entropy()
        self._load_topic_coherence()
        TpcsInfo = [{"Size": str(round(el[0], 4)), "Label": el[1].strip(), "Word Description": el[2].strip(), "Ndocs Active": str(el[3]), "Topics entropy": str(round(
            el[4], 4)), "Topics coherence": str(round(el[5], 4))} for el in zip(self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active, self._topic_entropy, self._topic_coherence)]

        return TpcsInfo

    def setTpcLabels(self, TpcLabels):
        self._tpc_labels = [el.strip() for el in TpcLabels]
        self._load_alphas()
        # Check that the number of labels is consistent with model
        if len(TpcLabels) == self._ntopics:
            with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
                fout.write('\n'.join(self._tpc_labels))
            return 1
        else:
            return 0

    def deleteTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            # Get a list of the topics that should be kept
            tpc_keep = [k for k in range(self._ntopics) if k not in tpcs]
            tpc_keep = [k for k in tpc_keep if k < self._ntopics]

            # Calculate new variables
            self._thetas = self._thetas[:, tpc_keep]
            from sklearn.preprocessing import normalize
            self._thetas = normalize(self._thetas, axis=1, norm='l1')
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            self._betas = self._betas[tpc_keep, :]
            self._betas_ds = self._betas_ds[tpc_keep, :]
            self._ndocs_active = self._ndocs_active[tpc_keep]
            self._topic_entropy = self._topic_entropy[tpc_keep]
            self._topic_coherence = self._topic_coherence[tpc_keep]
            self._tpc_labels = [self._tpc_labels[i] for i in tpc_keep]
            self._tpc_descriptions = [
                self._tpc_descriptions[i] for i in tpc_keep]
            self._edits.append('d ' + ' '.join([str(k) for k in tpcs]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics deletion successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics deletion generated an error. Operation failed')
            return 0

    def getSimilarTopics(self, npairs, thr=1e-3):
        """Obtains pairs of similar topics
        npairs: number of pairs of words
        thr: threshold for vocabulary thresholding
        """

        self._load_thetas()
        self._load_betas()

        # Part 1 - Coocurring topics
        # Highly correlated topics co-occure together
        # Topic mean
        med = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        # Topic square mean
        thetas2 = self._thetas.multiply(self._thetas)
        med2 = np.asarray(np.mean(thetas2, axis=0)).ravel()
        # Topic stds
        stds = np.sqrt(med2 - med ** 2)
        # Topic correlation
        num = self._thetas.T.dot(
            self._thetas).toarray() / self._thetas.shape[0]
        num = num - med[..., np.newaxis].dot(med[np.newaxis, ...])
        deno = stds[..., np.newaxis].dot(stds[np.newaxis, ...])
        corrcoef = num / deno
        selected_coocur = self._largest_indices(
            corrcoef, self._ntopics + 2 * npairs)
        selected_coocur = [(el[0], el[1], el[2].astype(float))
                           for el in selected_coocur]

        # Part 2 - Topics with similar word composition
        # Computes inter-topic distance based on word distributions
        # using scipy implementation of Jensen Shannon distance
        from scipy.spatial.distance import jensenshannon

        # For a more efficient computation with very large vocabularies
        # we implement a threshold for restricting the distance calculation
        # to columns where any element is greater than threshold thr
        betas_aux = self._betas[:, np.where(self._betas.max(axis=0) > thr)[0]]
        js_mat = np.zeros((self._ntopics, self._ntopics))
        for k in range(self._ntopics):
            for kk in range(self._ntopics):
                js_mat[k, kk] = jensenshannon(
                    betas_aux[k, :], betas_aux[kk, :])
        JSsim = 1 - js_mat
        selected_worddesc = self._largest_indices(
            JSsim, self._ntopics + 2 * npairs)
        selected_worddesc = [(el[0], el[1], el[2].astype(float))
                             for el in selected_worddesc]

        similarTopics = {
            'Coocurring': selected_coocur,
            'Worddesc': selected_worddesc
        }

        return similarTopics

    def fuseTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_edits()
        self._load_vocab()

        try:
            # List of topics that will be merged
            tpcs = sorted(tpcs)

            # Calculate new variables
            # For beta we keep a weighted average of topic vectors
            weights = self._alphas[tpcs]
            bet = weights[np.newaxis, ...].dot(
                self._betas[tpcs, :]) / (sum(weights))
            # keep new topic vector in upper position and delete the others
            self._betas[tpcs[0], :] = bet
            self._betas = np.delete(self._betas, tpcs[1:], 0)
            # For theta we need to keep the sum. Since adding implies changing
            # structure, we need to convert to full matrix first
            # No need to renormalize
            thetas_full = self._thetas.toarray()
            thet = np.sum(thetas_full[:, tpcs], axis=1)
            thetas_full[:, tpcs[0]] = thet
            thetas_full = np.delete(thetas_full, tpcs[1:], 1)
            self._thetas = sparse.csr_matrix(thetas_full, copy=True)
            # Compute new alphas and number of topics
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            # Compute all other variables
            self._calculate_beta_ds()
            self._calculate_topic_entropy()
            self._ndocs_active = np.array(
                (self._thetas != 0).sum(0).tolist()[0])

            # Keep label and description of most significant topic
            for tpc in tpcs[1:][::-1]:
                del self._tpc_descriptions[tpc]
            # Recalculate chemical description of most significant topic
            self._tpc_descriptions[tpcs[0]] = self.get_tpc_word_descriptions(tpc=[tpcs[0]])[
                0][1]
            for tpc in tpcs[1:][::-1]:
                del self._tpc_labels[tpc]
            
            # Parámetros para calcular coherencia
            coherence_measure = 'c_npmi' # 'c_v' or 'c_npmi'
            top_n = 15  
            file_path = "/export/usuarios_ml4ds/cggamella/RAG_tool/data/dump/eswiki-latest-abstract.xml.gz"
            
            print("Calculando la coherencia...")
            measure_name, mean_coherence, topic_coherences = self.calculate_topic_coherence(
                coherence_measure=coherence_measure,
                top_n=top_n,
                file_path=file_path
            )

            # Loguear resultados de coherencia
            self._logger.info(f"Coherencia calculada:")
            print(f"Measure Name: {measure_name}")
            print(f"Mean Coherence: {mean_coherence}")
            print(f"Topic Coherences: {topic_coherences}")
            self._edits.append('f ' + ' '.join([str(el) for el in tpcs]))
            # We are ready to save all variables in the model
            self._calculate_sims()
            self._save_all()

            self._logger.info(
                '-- -- Topics merging successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics merging generated an error. Operation failed')
            return 0

    def sortTopics(self):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            # Calculate order for the topics
            idx = np.argsort(self._alphas)[::-1]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # Calculate new variables
            self._thetas = self._thetas[:, idx]
            self._alphas = self._alphas[idx]
            self._betas = self._betas[idx, :]
            self._betas_ds = self._betas_ds[idx, :]
            self._ndocs_active = self._ndocs_active[idx]
            self._topic_entropy = self._topic_entropy[idx]
            self._topic_coherence = self._topic_coherence[idx]
            self._tpc_labels = [self._tpc_labels[i] for i in idx]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in idx]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics reordering successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics reordering generated an error. Operation failed')
            return 0

    def resetTM(self):
        self._alphas_orig = np.load(self._TMfolder.joinpath('alphas_orig.npy'))
        self._betas_orig = np.load(self._TMfolder.joinpath('betas_orig.npy'))
        self._thetas_orig = sparse.load_npz(
            self._TMfolder.joinpath('thetas_orig.npz'))
        self._load_vocab()

        try:
            self.create(betas=self._betas_orig, thetas=self._thetas_orig,
                        alphas=self._alphas_orig, vocab=self._vocab)
            return 1
        except:
            return 0

    # Not used... 
    def recalculate_cohrs(self):

        self.load_tpc_descriptions()

        try:
            self.calculate_topic_coherence()
            self._save_cohr()
            self._logger.info(
                '-- -- Topics cohrence recalculation successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics cohrence recalculation  an error. Operation failed')
            return 0

    def to_dataframe(self):
        self._load_alphas()
        self._load_betas()
        self._load_betas_ds()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_vocab()
        self._load_vocab_dicts()

        data = {
            "betas": [self._betas],
            "betas_ds": [self._betas_ds],
            "alphas": [self._alphas],
            "topic_entropy": [self._topic_entropy],
            "topic_coherence": [self._topic_coherence],
            "ndocs_active": [self._ndocs_active],
            "tpc_descriptions": [self._tpc_descriptions],
            "tpc_labels": [self._tpc_labels],
        }
        df = pd.DataFrame(data)
        return df, self._vocab_id2w, self._vocab
