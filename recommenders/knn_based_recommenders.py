"""
knn_based_recommenders.py

This module defines recommendation engines apply some kind of kNN algorithm to estimate ratings.

"""

import logging
from typing import Callable, List, Dict
from abc import ABC, abstractmethod

import numpy as np
from surprise import AlgoBase, PredictionImpossible, Trainset

from dataset.binary_dataset import load_binary_dataset_from_trainset
from dataset.common import load_dataset_from_trainset
from pattern_mining.common import filter_patterns_based_on_bicluster_sparsity
from pattern_mining.formal_concept_analysis import get_factor_matrices_from_concepts, Concept

from . import DEFAULT_LOGGER
from .common import (
    get_cosine_similarity_matrix,
    get_top_k_patterns_for_user,
    compute_targets_neighborhood_cosine_similarity,
)


def get_item_neighborhood(
    dataset: np.ndarray,
    patterns: List[np.ndarray],
    user_id: int,
    user_binarization_threshold: float,
    top_k_patterns: int,
):
    """
    Get the item neighborhood for a given user.

    The item neighborhood is the set of items are the most relevant to a given user according to
    the patterns extracted from the dataset.

    The item neighborhood is the union of the top-k patterns for the user. See
    get_top_k_patterns_for_user() for more details on how the top-k patterns are computed.

    Args:
        dataset (np.ndarray): The dataset.
        patterns (List[np.ndarray]): The patterns extracted from the dataset.
        user_id (int): The target user's id.
        user_binarization_threshold (float): The threshold used to binarize the user and covert it
                                             to an itemset.
        top_k_patterns (int): The number of top patterns to use to generate the item neighborhood.

    """
    assert dataset.ndim == 2
    assert dataset.shape[0] > 0
    assert dataset.shape[1] > 0

    assert user_id >= 0
    assert user_id < dataset.shape[0]

    assert user_binarization_threshold >= 0

    assert top_k_patterns > 0

    user = dataset[user_id]
    binarized_user = user >= user_binarization_threshold
    binarized_user_as_itemset = np.nonzero(binarized_user)[0]

    top_k_patterns_for_user = get_top_k_patterns_for_user(
        patterns, binarized_user_as_itemset, top_k_patterns
    )

    merged_top_k_patterns = np.array([], dtype=int)

    for pattern in top_k_patterns_for_user:
        merged_top_k_patterns = np.union1d(merged_top_k_patterns, pattern)

    itens_rated_by_user_and_in_merged_pattern = dataset[user_id, merged_top_k_patterns] > 0

    rated_itens_from_merged_pattern = merged_top_k_patterns[
        itens_rated_by_user_and_in_merged_pattern
    ]

    return rated_itens_from_merged_pattern


# pylint: disable=C0103
class KNNOverLatentSpaceRecommender(AlgoBase, ABC):
    """
    KNNOverLatentSpaceRecommender is an abstract class for recommendation engines
    based on the KNN algorithm. However, instead of using the original dataset,
    these recommendation engines use a latent dataset generated from the original
    dataset using formal concept analysis (FCA). It extends the functionality of
    the AlgoBase class from the Surprise library and provides methods for
    generating recommendations.
    """

    def __init__(
        self,
        dataset_binarization_threshold: float = 1.0,
        knn_k: int = 30,
        knn_similarity_matrix_strategy: Callable = get_cosine_similarity_matrix,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        self.logger = logger

        # Dataset binarization attributes
        self.dataset_binarization_threshold = dataset_binarization_threshold
        self.binary_dataset = None

        # Pattern extraction attributes
        self.formal_context: List[Concept] = []
        self.number_of_factors = None
        self.A = None
        self.B = None
        self.sim = None  # Similarity matrix

        # KNN attributes
        self.knn_k = knn_k
        self.knn_distance_strategy = knn_similarity_matrix_strategy

    @abstractmethod
    def generate_formal_context(self):
        """
        Generates the formal context from the patterns extracted from the training data.

        Override this method in a subclass to implement the desired pattern extraction.
        This method should set the formal_context attribute based on the patterns extracted
        from the training data at self.binary_dataset.
        """

    def fit(self, trainset: Trainset):
        """
        Train the algorithm on a given training set.

        Args:
            trainset (Trainset): The training set to train the algorithm on.

        Returns:
            self: The trained algorithm.
        """
        AlgoBase.fit(self, trainset)

        # Generate binary dataset
        self.logger.debug("Generating binary dataset...")

        self.binary_dataset = load_binary_dataset_from_trainset(
            trainset, threshold=self.dataset_binarization_threshold
        )
        self.logger.debug("Generating binary dataset OK")

        # Generate formal context
        self.logger.debug("Generating formal context...")
        self.generate_formal_context()
        self.number_of_factors = len(self.formal_context)

        if self.number_of_factors == 0:
            raise ValueError("No factors were extracted from the dataset.")

        self.logger.debug("Generating formal context OK")

        # Generate similarity matrix
        self.logger.info("Generating Similarity Matrix...")
        self.A, self.B = get_factor_matrices_from_concepts(
            self.formal_context,
            self.binary_dataset.shape[0],
            self.binary_dataset.shape[1],
        )

        self.sim = get_cosine_similarity_matrix(self.A)
        self.logger.info("Generating Similarity Matrix OK")

        return self

    def estimate(self, user: int, item: int):
        """
        Estimates the rating of a given user for a given item. This function is not supposed to be
        called directly since it uses the Surprise's internal user and item ids. Surprise uses this
        callback internally to make predictions. Use the predict() or test() methods instead which
        use the raw user and item ids.

        Args:
            user (int): internal user id.
            item (int): internal item id.

        Returns:
            tuple: A tuple containing the predicted rating and a dictionary with
                additional details.

        Raises:
            PredictionImpossible: If the user and/or item is unknown or if there
                are not enough neighbors to make a prediction.
        """

        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        ruid = self.trainset.to_raw_uid(user)
        riid = self.trainset.to_raw_iid(item)

        self.logger.debug(f"Estimating rating for user {user} ({ruid}) and item {item} ({riid})...")

        # Get a list of tuples (neighbor, similarity, rating) representing all neighbors
        neighbors = [
            (other_user, self.sim[user, other_user], rating)
            for (other_user, rating) in self.trainset.ir[item]
        ]

        # Remove neighbors with no similarity. Absence of similarity is represented by NaN and
        # happens when the similarity between these users was impossible.
        neighbors = [neighbor for neighbor in neighbors if not np.isnan(neighbor[1])]

        # Sort neighbors by similarity in descending order.
        nearest_neighbors = sorted(neighbors, key=lambda d: d[1], reverse=True)

        self.logger.debug(f"Available neighbors: {len(nearest_neighbors)}")

        # Compute the weighted average of the ratings of the k nearest neighbors
        ratings_sum = 0
        weights_sum = 0
        neighbors_used = []

        for neighbor in nearest_neighbors:
            # Stop if we have enough neighbors
            if len(neighbors_used) >= self.knn_k:
                break

            neighbor_iid, neighbor_similarity, neighbor_rating = neighbor
            neighbor_ruid = self.trainset.to_raw_uid(neighbor_iid)

            if neighbor_similarity == 0:
                continue

            ratings_sum += neighbor_similarity * neighbor_rating
            weights_sum += neighbor_similarity
            neighbors_used.append((neighbor_ruid, neighbor_similarity, neighbor_rating))

        if not neighbors_used:
            raise PredictionImpossible("Not enough neighbors.")

        rating = ratings_sum / weights_sum

        self.logger.debug(f"Neighbors used: {len(neighbors_used)}")

        # Additional details
        details = {"actual_k": len(neighbors_used), "neighbors_used": neighbors_used}

        return rating, details


class KNNOverItemNeighborhood(AlgoBase, ABC):
    """
    Family of recommendation engines that use the KNN algorithm over the a user-item neighborhood
    to estimate ratings.

    The user-item neighborhood is the set of items that are relevant to a given user. Therefore,
    we exploit the locality of the user-item neighborhood to estimate ratings. This filtering
    is expected to improve the accuracy of the predictions and reduce the time spent on
    computations.

    This class specifies that a method called generate_user_item_neighborhood() must be implemented
    by subclasses. This method is responsible for generating the user-item neighborhood for each
    user in the dataset. The user-item neighborhood is stored in the user_item_neighborhood
    attribute. This attribute is a dictionary where the keys are the user ids and the values are
    the user-item neighborhood for each user.
    """

    def __init__(
        self,
        dataset_binarization_threshold: float = 1.0,
        minimum_pattern_bicluster_sparsity: float = 0.08,
        user_binarization_threshold: float = 1.0,
        top_k_patterns: int = 8,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        """
        Args:
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
        AlgoBase.__init__(self)

        assert isinstance(minimum_pattern_bicluster_sparsity, float)
        assert 0 <= minimum_pattern_bicluster_sparsity <= 1

        assert isinstance(user_binarization_threshold, float)
        assert user_binarization_threshold >= 0

        assert isinstance(top_k_patterns, int)
        assert top_k_patterns > 0

        assert isinstance(knn_k, int)
        assert knn_k > 0

        # Pattern filtering parameters
        self.dataset_binarization_threshold = dataset_binarization_threshold
        self.minimum_pattern_bicluster_sparsity = minimum_pattern_bicluster_sparsity
        self.user_binarization_threshold = user_binarization_threshold
        self.top_k_patterns = top_k_patterns

        # KNN parameters
        self.knn_k = knn_k

        self.logger = logger

        self.user_item_neighborhood = None
        self.dataset = None
        self.similarity_matrix = None
        self.item_means = None
        self.patterns = None

    @abstractmethod
    def compute_patterns_from_trainset(self) -> List[np.ndarray]:
        """
        This method is responsible for computing the patterns that will be used to generate the
        user-item neighborhood for each user in the dataset. Subclasses must implement this method.

        This method will be called after the loading of the trainset. Therefore, the trainset
        attribute will be available and should be used to generate the user-item neighborhood.

        The patterns must be returned as a list of numpy arrays. Each numpy array represents a
        pattern. The elements of the array are the item ids that are part of the pattern.
        """

    def fit(self, trainset: Trainset):
        """
        Train the algorithm on a given training set.

        Args:
            trainset (Trainset): The training set to train the algorithm on.

        Returns:
            self: The trained algorithm.
        """

        AlgoBase.fit(self, trainset)

        self.dataset = load_dataset_from_trainset(trainset)

        self.compute_patterns_from_trainset()

        self.logger.info("Computed patterns: %d", len(self.patterns))

        if self.minimum_pattern_bicluster_sparsity > 0:
            self.patterns = filter_patterns_based_on_bicluster_sparsity(
                self.dataset, self.patterns, self.minimum_pattern_bicluster_sparsity
            )
            self.logger.info("Filtered patterns: %d", len(self.patterns))


        self.user_item_neighborhood = {}
        for user_id in range(self.dataset.shape[0]):
            item_neighborhood = get_item_neighborhood(
                self.dataset,
                self.patterns,
                user_id,
                self.user_binarization_threshold,
                self.top_k_patterns,
            )
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

        prediction = self.item_means[item]
        actual_k = 0
        details = {"actual_k": actual_k}

        if self.user_item_neighborhood[user].size == 0:
            return prediction, details

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

        sum_similarities = sum_ratings = 0
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
