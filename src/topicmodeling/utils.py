import json
import pathlib
import pickle
import pandas as pd
from src.topicmodeling.topic_model import (BERTopicModel, CtmModel,
                                           MalletLdaModel, TopicGPTModel)


def unpickler(file: str):
    """Unpickle file"""
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob):
    """Pickle object to file"""
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return


def load_processed_data(file: str):
    if file.endswith(".json") or file.endswith(".jsonl"):
        df = pd.read_json(file)
    elif file.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.endswith(".parquet"):
        df = pd.read_parquet(file)
    else:
        raise ValueError("File format not supported")
    return df


def file_lines(fname):
    """
    Count number of lines in file

    Parameters
    ----------
    fname: Path
        the file whose number of lines is calculated

    Returns
    -------
    number of lines
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def tkz_clean_str(
    rawtext,
    stops_path="src/topicmodeling/data/stops",
    eqs_path="src/topicmodeling/data/equivalences"
):
    """Function to carry out tokenization and cleaning of text

    Parameters
    ----------
    rawtext: str
        string with the text to lemmatize

    Returns
    -------
    cleantxt: str
        Cleaned text
    """
    
    def _loadSTW(stops_path):
        """
        Loads all stopwords from all files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files

        Returns
        -------
        stopWords: list of str
            List of stopwords
        """

        stw_files = pathlib.Path(stops_path).rglob('*')
        stopWords = []
        for stwFile in stw_files:
            with stwFile.open('r', encoding='utf8') as fin:
                stopWords += json.load(fin)['wordlist']

        return list(set(stopWords))

    def _loadEQ(eqs_path):
        """
        Loads all equivalent terms from all files provided in the argument

        Parameters
        ----------
        eq_files: list of str
            List of paths to equivalent terms files

        Returns
        -------
        equivalents: dictionary
            Dictionary of term_to_replace -> new_term
        """

        eq_files = pathlib.Path(eqs_path).rglob('*')
        equivalent = {}
        for eqFile in eq_files:
            with eqFile.open('r', encoding='utf8') as fin:
                newEq = json.load(fin)['wordlist']
            newEq = [x.split(':') for x in newEq]
            newEq = [x for x in newEq if len(x) == 2]
            newEq = dict(newEq)
            equivalent = {**equivalent, **newEq}

        return equivalent
    
    stopwords = _loadSTW(stops_path=stops_path)
    equivalences = _loadEQ(eqs_path=eqs_path)
    
    if rawtext == None or rawtext == '':
        return ''
    else:
        # lowercase and tokenization (similar to Spark tokenizer)
        cleantext = rawtext.lower().split()
        # remove stopwords
        cleantext = [
            el for el in cleantext if el not in stopwords]
        # replacement of equivalent words
        cleantext = [equivalences[el] if el in equivalences else el
                        for el in cleantext]
        # remove stopwords again, in case equivalences introduced new stopwords
        cleantext = [
            el for el in cleantext if el not in stopwords]

    return ' '.join(cleantext)

def create_model(model_name, **kwargs):
    # Map model names to corresponding classes
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