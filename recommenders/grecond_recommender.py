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
from fca.formal_concept_analysis import GreConD
from . import DEFAULT_LOGGER
from .common import jaccard_distance
from .formal_context_based_recommender import KNNOverLatentSpaceRecommender


class GreConDRecommender(KNNOverLatentSpaceRecommender):
    """
    GreConDRecommender is a recommendation engine that uses the GreConD algorithm
    for formal concept enumeration. It extends the functionality of the KNNOverLatentSpaceRecommender
    class and provides methods for generating recommendations.

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

        self.formal_context, self.actual_coverage = GreConD(
            self.binary_dataset, coverage=self.grecond_coverage, logger=self.logger
        )

        self.logger.info("Generating Formal Context OK")
