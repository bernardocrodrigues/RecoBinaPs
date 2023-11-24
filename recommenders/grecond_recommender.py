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

import numpy as np

from dataset.common import load_dataset_from_trainset
from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
)
from pattern_mining.common import filter_patterns_based_on_bicluster_sparsity


from surprise.accuracy import mae, rmse
from surprise import Trainset, AlgoBase, PredictionImpossible

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
from .formal_context_based_recommender import KNNOverLatentSpaceRecommender

from .common import (
    get_top_k_patterns_for_user,
    compute_targets_neighborhood_cosine_similarity,
)

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

class GreConDRecommender2(AlgoBase):

    def __init__(
        self,
        epochs: int = 100,
        hidden_dimension_neurons_number: int = -1,
        weights_binarization_threshold: float = 0.2,
        dataset_binarization_threshold: float = 1.0,
        minimum_pattern_bicluster_sparsity: float = 0.08,
        user_binarization_threshold: float = 1.0,
        top_k_patterns: int = 8,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        # BinaPs parameters
        self.epochs = epochs
        self.hidden_dimension_neurons_number = hidden_dimension_neurons_number
        self.weights_binarization_threshold = weights_binarization_threshold
        self.dataset_binarization_threshold = dataset_binarization_threshold

        # Pattern filtering parameters
        self.minimum_pattern_bicluster_sparsity = minimum_pattern_bicluster_sparsity
        self.user_binarization_threshold = user_binarization_threshold
        self.top_k_patterns = top_k_patterns

        # KNN parameters
        self.knn_k = knn_k

        self.logger = logger

        self.user_item_neighborhood = {}
        self.dataset = None
        self.similarity_matrix = None
        self.item_means = None

    def get_item_neighborhood(
        self,
        dataset: np.ndarray,
        patterns,
        user_id,
    ):
        user = dataset[user_id]
        binarized_user = user >= self.user_binarization_threshold
        binarized_user_as_itemset = np.nonzero(binarized_user)[0]

        top_k_patterns_for_user = get_top_k_patterns_for_user(
            patterns, binarized_user_as_itemset, self.top_k_patterns
        )

        merged_top_k_patterns = np.array([], dtype=int)

        for pattern in top_k_patterns_for_user:
            merged_top_k_patterns = np.union1d(merged_top_k_patterns, pattern)

        itens_rated_by_user_and_in_merged_pattern = dataset[user_id, merged_top_k_patterns] > 0

        rated_itens_from_merged_pattern = merged_top_k_patterns[
            itens_rated_by_user_and_in_merged_pattern
        ]

        return rated_itens_from_merged_pattern

    def fit(self, trainset: Trainset):
        """
        Train the algorithm on a given training set.

        Args:
            trainset (Trainset): The training set to train the algorithm on.

        Returns:
            self: The trained algorithm.
        """

        AlgoBase.fit(self, trainset)


        binary_dataset = load_binary_dataset_from_trainset(
            trainset, threshold=self.dataset_binarization_threshold
        )

        formal_context, _ = grecond(binary_dataset, coverage=0.1)
        patterns = []
        for (C, D) in formal_context:
            patterns.append(D)
            

        self.dataset = load_dataset_from_trainset(trainset)
        patterns = filter_patterns_based_on_bicluster_sparsity(
            self.dataset, patterns, self.minimum_pattern_bicluster_sparsity
        )

        for user_id in range(self.dataset.shape[0]):
            item_neighborhood = self.get_item_neighborhood(self.dataset, patterns, user_id)
            self.user_item_neighborhood[user_id] = item_neighborhood

        self.item_means = np.zeros(self.trainset.n_items)
        for x, ratings in self.trainset.ir.items():
            self.item_means[x] = np.mean([r for (_, r) in ratings])

        self.similarity_matrix = np.full(
            (self.trainset.n_items, self.trainset.n_items), dtype=float, fill_value=np.NAN
        )

    def estimate(self, user: int, item: int):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        compute_targets_neighborhood_cosine_similarity(
            self.dataset, self.similarity_matrix, item, self.user_item_neighborhood[user]
        )

        targets_neighborhood_similarity = self.similarity_matrix[
            item, self.user_item_neighborhood[user]
        ]
        targets_item_neighborhood_ratings = self.dataset[user, self.user_item_neighborhood[user]]
        item_means = self.item_means[self.user_item_neighborhood[user]]

        k_top_neighbors_indices = np.argsort(targets_neighborhood_similarity)[-self.knn_k :]

        k_top_neighbors_ratings = targets_item_neighborhood_ratings[k_top_neighbors_indices]
        k_top_neighbors_similarity = targets_neighborhood_similarity[k_top_neighbors_indices]
        k_top_item_means = item_means[k_top_neighbors_indices]

        prediction = self.item_means[item]

        sum_similarities = sum_ratings = actual_k = 0
        for rating, similarity, item_mean in zip(
            k_top_neighbors_ratings, k_top_neighbors_similarity, k_top_item_means
        ):
            if similarity > 0:
                sum_similarities += similarity
                sum_ratings += similarity * (rating - item_mean)
                actual_k += 1

        try:
            prediction += sum_ratings / sum_similarities
        except ZeroDivisionError:
            pass

        details = {"actual_k": actual_k}
        return prediction, details

