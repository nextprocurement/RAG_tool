import argparse
import random
import logging
import pathlib
from itertools import product
from typing import List, Optional, Union
import gensim.downloader as api
import numpy as np
from src.utils.utils import init_logger


class TMMatcher(object):
    """
    Class to align the topics from different topic models.
    """

    def __init__(
        self,
        wmd_model: str = 'word2vec-google-news-300',
        logger: Optional[logging.Logger] = None,
        path_logs: pathlib.Path = pathlib.Path(
            __file__).parent.parent.parent / "data/logs"
    ) -> None:
        """
        Initialize the TopicSelector class.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger object to log activity.
        path_logs : pathlib.Path, optional
            Path for saving logs.
        """
        self._logger = logger if logger else init_logger(__name__, path_logs)
        self._wmd_model = api.load(wmd_model)      

        return
    
    def _get_wmd(
        self,
        from_: Union[str, List[str]],
        to_: Union[str, List[str]],
        n_words=10
    ) -> float:
        """
        Calculate the Word Mover's Distance between two sentences.

        Parameters
        ----------
        from_ : Union[str, List[str]]
            The source sentence.
        to_ : Union[str, List[str]]
            The target sentence.
        n_words : int
            The number of words to consider in the sentences to calculate the WMD.
        """
        if isinstance(from_, str):
            from_ = from_.split()

        if isinstance(to_, str):
            to_ = to_.split()

        if n_words < len(from_):
            from_ = from_[:n_words]
        if n_words < len(to_):
            to_ = to_[:n_words]

        return self._wmd_model.wmdistance(from_, to_)
    
    def _get_wmd_mat(self, models: list) -> np:
        """Find the closest topics between two models using Word Mover's Distance.
        
        Parameters
        ----------
        models : list
            A list containing two sublists/arrays representing the models.
        keep_from_first : list, optional
            Indices of topics from the first model to keep, by default [0, 1, 2]
        
        Returns
        -------
        np.ndarray
            A matrix of Word Mover's Distance between topics from two models.
        """
        
        if len(models) != 2:
            raise ValueError("models must contain exactly two sublists/arrays.")
        
        num_topics_first_model = len(models[0])
        num_topics_second_model = len(models[1])
        
        wmd_sims = np.zeros((num_topics_first_model, num_topics_second_model))
        
        for k_idx, k in enumerate(models[0]):
            for k__idx, k_ in enumerate(models[1]):
                wmd_sims[k_idx, k__idx] = self._get_wmd(k, k_)
                
        return wmd_sims

    def iterative_matching(self, models, N, metric='wmd', seed=2357_11):
        """
        Performs an iterative pairing process between the topics of multiple models.

        Parameters
        ----------
        models : list of numpy.ndarray
            List with the betas matrices from different models.
        N : int
            Number of matches to find.

        Returns
        -------
        list of list of tuple
            List of lists with the N matches found. Each match is a list of tuples, where each tuple contains the model index and the topic index.
        """
        random.seed(seed)
        dists = {}
        for modelA, modelB in product(range(len(models)), range(len(models))):
            dists[(modelA, modelB)] = self._get_wmd_mat([models[modelA], models[modelB]])

        matches = []
        assert(all(N <= len(m) for m in models))
        while len(matches) < min(len(m) for m in models):
            for seed_model in range(len(models)):
                # Calculate the mean distance to all other models
                min_dists, min_dists_indices = [], []
                for other_model in range(len(models)):
                    if seed_model == other_model:
                        min_dists_indices.append((seed_model, None))
                        continue
                    distsAB = dists[(seed_model, other_model)]
                    # Get the minimum distance for each topic in the seed model to the other model
                    min_dists.append(distsAB.min(1))
                    min_dists_indices.append((other_model, distsAB.argmin(1)))
                mean_min_dists = np.mean(min_dists, axis=0)
                seed_model_topic = np.argmin(mean_min_dists)
                seed_model_matches = [
                    (model_idx, indices[seed_model_topic]) if model_idx != seed_model else (model_idx, seed_model_topic)
                    for model_idx, indices in min_dists_indices
                ]
                matches.append(seed_model_matches)
                # Remove the matched topics from the distance matrix
                for modelA, modelA_topic in seed_model_matches:
                    for modelB in range(len(models)):
                        if modelA != modelB:
                            dists[(modelA, modelB)][modelA_topic, :] = np.inf
                            dists[(modelB, modelA)][:, modelA_topic] = np.inf
        return random.sample(matches, N)
    
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--betas_paths", help="Paths of the models'betas files, separated by commas", required=True)
    parser.add_argument(
        "--N", type=int, help="Number of matches (topics to evaluate)", default=2)

    betas_mats = []
    args = parser.parse_args()
    for betas_path in args.betas_paths.split(","):
        betas = np.load(betas_path)
        if betas.shape[0] > betas.shape[1]:
            betas = betas.T
        betas_mats.append(betas)

    matcher = TMMatcher()
    matches = matcher.iterative_matching(betas_mats, args.N)

    print("Matches:", matches)


if __name__ == '__main__':
    main()