import json
import pathlib
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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

def generate_dynamic_stopwords(texts, percentage_below_mean=0.2):
    """
    Generate a list of stopwords based on the TF-IDF scores of the words in the texts.

    Args:
        texts (list of str): 
        percentage_below_mean (float): Percetage of words below the mean TF-IDF score to consider as stopwords.

    Returns:
        list: List of stopwords based on the TF-IDF scores of the words in the texts.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    feature_names = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_

    mean_idf = np.mean(idf_values)
    threshold = mean_idf * (1 - percentage_below_mean)

    # Generate list of stopwords
    stopwords_tfidf = [word for word, idf in zip(feature_names, idf_values) if idf <= threshold]
    #print("La lista de stopwords es:", stopwords_tfidf)
    #print("La longitud de stopwords es:", len(stopwords_tfidf))
    #import pdb; pdb.set_trace()

    return stopwords_tfidf

def tkz_clean_str(
    row,
    stops_path="src/topicmodeling/data/stops",
    eqs_path="src/topicmodeling/data/equivalences",
    stopwords_tfidf=None
):
    """Function to carry out tokenization and cleaning of text

    Parameters
    ----------
    row: str
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
        if eq_files == []:
            return {}
        
        equivalent = {}
        duplicates = set()

        for eqFile in eq_files:
            with eqFile.open('r', encoding='utf8') as fin:
                newEq = json.load(fin)['wordlist']
            
            # Split terms into key-value pairs
            newEq = [x.split(':') for x in newEq]
            # Filter valid pairs
            newEq = [x for x in newEq if len(x) == 2]
            # Convert to dictionary
            newEq_dict = dict(newEq)

            # Add to the equivalent dictionary only if the key is not already present
            for key, value in newEq_dict.items():
                if key not in equivalent:  # Add only if key is not already present
                    equivalent[key] = value
                else:
                    duplicates.add(key)  # Track the duplicate key

        #if duplicates:
        #    print(f"Ignored duplicate keys: {duplicates}")

        return equivalent
    
    stopwords = _loadSTW(stops_path=stops_path)
    equivalences = _loadEQ(eqs_path=eqs_path)
    
    if row == None or row == '':
        return ''
    else:
        # lowercase and tokenization (similar to Spark tokenizer)
        cleantext = row.lower().split()
        # remove stopwords
        cleantext = [
            el for el in cleantext if el not in stopwords]
        # replacement of equivalent words
        cleantext = [equivalences[el] if el in equivalences else el
                        for el in cleantext]
        # remove stopwords again, in case equivalences introduced new stopwords
        cleantext = [
            el for el in cleantext if el not in stopwords]
        
        # Remove stopwords based on TF-IDF
        if stopwords_tfidf is not None:
            cleantext = [word for word in cleantext if word not in stopwords_tfidf]

    return ' '.join(cleantext)