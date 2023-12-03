"""
grecond_recommender.py

This module defines recommendation engines that that are somehow powered by the GreConD algorithm.

"""

import logging
from typing import Tuple, List, Dict
from surprise.accuracy import mae, rmse
from surprise import Trainset

from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
)

from pattern_mining.common import filter_patterns_based_on_bicluster_sparsity
from pattern_mining.formal_concept_analysis import grecond
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
from .knn_based_recommenders import KNNOverLatentSpaceRecommender, KNNOverItemNeighborhood


class GreConDKNNRecommender(KNNOverLatentSpaceRecommender):
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
            knn_similarity_matrix_strategy=knn_distance_strategy,
            logger=logger,
        )
        self.grecond_coverage = grecond_coverage
        self.actual_coverage = None

    def generate_formal_context(self):
        self.logger.info("Generating Formal Context...")

        self.formal_context, self.actual_coverage = grecond(
            self.binary_dataset, coverage=self.grecond_coverage
        )

        self.logger.info("Generating Formal Context OK")

    @classmethod
    def thread(
        cls,
        fold: Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]],
        output: Dict,
        grecond_recommender: "GreConDKNNRecommender",
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


class GreConDKNNRecommender2(KNNOverItemNeighborhood):
    """
    Recommender class that uses the GreConD algorithm to generate the patterns that are used
    to generate a user-item neighborhood that is, then, used for generating recommendations.
    """

    def __init__(
        self,
        grecond_coverage: float = 1.0,
        dataset_binarization_threshold: float = 1.0,
        minimum_pattern_bicluster_sparsity: float = 0.08,
        user_binarization_threshold: float = 1.0,
        top_k_patterns: int = 8,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        """
            Args:
        grecond_coverage (float): The minimum percentage of the dataset that will be covered by the
            formal context generated by the GreConD algorithm. After hitting this coverage, the
            mining process will stop. 
        dataset_binarization_threshold (float): The threshold used to binarize the dataset.
            Ratings greater than or equal to the threshold are considered relevant (True) and
            ratings less than the threshold are considered irrelevant (False).This binarization
            affects the patterns that will be extracted from the dataset.
        minimum_pattern_bicluster_sparsity (float): The minimum sparsity of the biclusters
            associated with each pattern. Patterns whose biclusters have a sparsity less than
            this value will be discarded. If this value is 0, no filtering will be performed.
        user_binarization_threshold (float): The threshold used to binarize the user when
            generating the user-item neighborhood. This happens after the patterns are extracted
            from the dataset. Ratings greater than or equal to the threshold are considered
            relevant (True) and ratings less than the threshold are considered irrelevant
            (False). This binarization affects the user-item neighborhood.
        knn_k (int): The number of itens to use to estimate ratings within the user-item
            neighborhood.
        logger (logging.Logger): The logger that will be used to log messages.
        """
        super().__init__(
            dataset_binarization_threshold=dataset_binarization_threshold,
            minimum_pattern_bicluster_sparsity=minimum_pattern_bicluster_sparsity,
            user_binarization_threshold=user_binarization_threshold,
            top_k_patterns=top_k_patterns,
            knn_k=knn_k,
            logger=logger,
        )
        self.grecond_coverage = grecond_coverage

        self.actual_coverage = None

    # pylint: disable=C0103
    def compute_patterns_from_trainset(self):
        """
        Generates the user-item neighborhood based on the given trainset.

        It uses the GreConD algorithm to compute the formal context (a list of formal concepts).
        From this list of formal concepts, we extract the patterns (intents) which are closed
        itemsets. Then, we filter the patterns based on their bicluster sparsity. Finally, we
        generate the user-item neighborhood based on the filtered patterns.

        """

        binary_dataset = load_binary_dataset_from_trainset(
            self.trainset, threshold=self.dataset_binarization_threshold
        )

        formal_context, actual_coverage = grecond(binary_dataset, coverage=self.grecond_coverage)

        self.actual_coverage = actual_coverage

        self.patterns = []
        for _, D in formal_context:
            self.patterns.append(D)
