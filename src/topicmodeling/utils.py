import pickle

import pandas as pd

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