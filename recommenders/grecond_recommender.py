"""
grecond_recommender.py

This module defines the GreConDRecommender class, which is a recommendation engine
based on the GreConD algorithm. GreConDRecommender is a subclass of PatternBasedRecommender
and provides methods for generating recommendations using GreConD.

Usage:
    from your_module import GreConDRecommender

    recommender = GreConDRecommender(k=30, coverage=0.8)
    recommender.generate_formal_context()
    recommendations = recommender.get_recommendations(user_data)

"""

import logging
from typing import Tuple, List, Dict

from surprise.accuracy import mae, rmse
from surprise import Trainset

from fca.formal_concept_analysis import grecond
from evaluation import (
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
)

from . import DEFAULT_LOGGER
from .common import jaccard_distance
from .formal_context_based_recommender import KNNOverLatentSpaceRecommender





class GreConDRecommender(KNNOverLatentSpaceRecommender):
    """
    GreConDRecommender is a recommendation engine that uses the GreConD algorithm
    for formal concept enumeration. It extends the functionality of the 
    KNNOverLatentSpaceRecommender class and provides methods for generating recommendations.

    Args:
        grecond_coverage (float): The original dataset coverage of the formal context.
        dataset_binarization_threshold (float): The threshold for binarizing the dataset.
        knn_k (int): The number of neighbors to consider in the kNN step.
        knn_distance_strategy (callable): The distance function to use.
        logger (logging.Logger): The logger for logging messages.

    Example:
        recommender = GreConDRecommender(k=30, coverage=0.8)
        recommender.fit(trainset)
        predictions = recommender.test(testset)
    """

    def __init__(
        self,
        grecond_coverage: float = 1.0,
        dataset_binarization_threshold: float = 1.0,
        knn_k: int = 30,
        knn_distance_strategy: callable = jaccard_distance,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        super().__init__(
            knn_k=knn_k,
            dataset_binarization_threshold=dataset_binarization_threshold,
            knn_distance_strategy=knn_distance_strategy,
            logger=logger,
        )
        self.grecond_coverage = grecond_coverage
        self.actual_coverage = None

    def generate_formal_context(self):
        self.logger.info("Generating Formal Context...")

        self.formal_context, self.actual_coverage = grecond(
            self.binary_dataset, coverage=self.grecond_coverage, logger=self.logger
        )

        self.logger.info("Generating Formal Context OK")

    @classmethod
    def thread(
        cls,
        fold: Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]],
        output: Dict,
        grecond_recommender: "GreConDRecommender",
        threshold: float = 1.0,
        number_of_top_recommendations: int = 20,
    ):
        """
        This function is used to parallelize the GreConD recommender. It puts the results on a
        dictionary called 'output'. 'output' is expected to be a Manager().dict() object since it is
        shared between processes.

        Args:
            fold (Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]): The fold to be
                                                                              processed.
            output (Dict): The dictionary to put the results on.
            grecond_recommender (GreConDRecommender): The GreConDRecommender object to use.
            threshold (float): The relevance threshold to use.
            number_of_top_recommendations (int): The number of top recommendations to use.

        Returns:
            None
        """
        fold_index, (trainset, testset) = fold

        grecond_recommender.fit(trainset)
        predictions = grecond_recommender.test(testset)
        output[(grecond_recommender.grecond_coverage, grecond_recommender.knn_k, fold_index)] = {
            "actual_coverage": grecond_recommender.actual_coverage,
            "number_of_factors": grecond_recommender.number_of_factors,
            "mae": mae(predictions=predictions, verbose=False),
            "rmse": rmse(predictions=predictions, verbose=False),
            "micro_averaged_recall": get_micro_averaged_recall(
                predictions=predictions, threshold=threshold
            ),
            "macro_averaged_recall": get_macro_averaged_recall(
                predictions=predictions, threshold=threshold
            ),
            "recall_at_k": get_recall_at_k(
                predictions=predictions,
                threshold=threshold,
                k=number_of_top_recommendations,
            ),
            "micro_averaged_precision": get_micro_averaged_precision(
                predictions=predictions, threshold=threshold
            ),
            "macro_averaged_precision": get_macro_averaged_precision(
                predictions=predictions, threshold=threshold
            ),
            "precision_at_k": get_precision_at_k(
                predictions=predictions,
                threshold=threshold,
                k=number_of_top_recommendations,
            ),
        }
