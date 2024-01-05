"""
knn_based_recommenders.py

This module defines recommendation engines apply some kind of kNN algorithm to estimate ratings.

"""

import logging
from typing import Callable, List, Optional
from abc import ABC, abstractmethod

import numpy as np
from surprise import AlgoBase, PredictionImpossible, Trainset

from dataset.binary_dataset import load_binary_dataset_from_trainset
from dataset.common import load_dataset_from_trainset
from pattern_mining.common import (
    apply_bicluster_sparsity_filter,
    apply_bicluster_coverage_filter,
    apply_bicluster_relative_size_filter,
)
from pattern_mining.formal_concept_analysis import (
    get_factor_matrices_from_concepts,
    Concept,
    create_concept,
)

from . import DEFAULT_LOGGER
from .common import (
    get_cosine_similarity_matrix,
    get_top_k_biclusters_for_user,
    compute_neighborhood_cosine_similarity,
    get_indices_above_threshold,
)


def merge_biclusters(
    biclusters: List[Concept],
) -> Concept:
    """
    Merges a list of biclusters into a single bicluster. This means that the extent of the new
    bicluster will be the union of the extents of the given biclusters and the intent of the new
    bicluster will be the union of the intents of the given biclusters.

    Args:
        biclusters (List[Concept]): A list of biclusters.

    Returns:
        Concept: A new bicluster that is the result of merging the given biclusters.
    """
    new_bicluster_extent = np.array([], dtype=np.int64)
    new_bicluster_intent = np.array([], dtype=np.int64)

    for bicluster in biclusters:
        new_bicluster_extent = np.union1d(new_bicluster_extent, bicluster.extent)
        new_bicluster_intent = np.union1d(new_bicluster_intent, bicluster.intent)

    return create_concept(new_bicluster_extent, new_bicluster_intent)


def calculate_weighted_rating(
    target_mean: float,
    neighbors_ratings: np.ndarray,
    neighbors_similarities: np.ndarray,
    neighbors_means: np.ndarray,
) -> float:
    """
    Calculates the weighted rating of a target item based on the ratings of its neighbors.

    Args:
        target_mean (float): The mean rating of the target item.
        neighbors_ratings (np.ndarray): An array containing the ratings of the neighbors.
        neighbors_similarities (np.ndarray): An array containing the similarities between the
            target item and its neighbors.
        neighbors_means (np.ndarray): An array containing the mean ratings of the neighbors.

    Note:
        All arrays must be ordered in the same way. That is, the rating, similarity and mean of
        the same neighbor must be at the same position in their respective arrays.

    Returns:
        float: The weighted rating of the target item.

    Raises:
        AssertionError: If any of the following conditions is not met:
            - target_mean is a float.
            - neighbors_ratings is a numpy array of floats.
            - neighbors_similarities is a numpy array of floats.
            - neighbors_means is a numpy array of floats.
            - neighbors_ratings, neighbors_similarities and neighbors_means have the same size.
            - neighbors_ratings, neighbors_similarities and neighbors_means have size greater than
                zero.
            - All similarities are greater than zero and less than or equal to one.

    """
    assert isinstance(target_mean, float)

    assert isinstance(neighbors_ratings, np.ndarray)
    assert neighbors_ratings.dtype == np.float64

    assert isinstance(neighbors_similarities, np.ndarray)
    assert neighbors_similarities.dtype == np.float64
    assert all(0 < similarity <= 1 for similarity in neighbors_similarities)

    assert isinstance(neighbors_means, np.ndarray)
    assert neighbors_means.dtype == np.float64

    assert neighbors_ratings.size == neighbors_similarities.size == neighbors_means.size
    assert neighbors_ratings.size > 0

    prediction = target_mean
    sum_similarities: float = 0.0
    sum_ratings: float = 0.0

    for rating, similarity, item_mean in zip(
        neighbors_ratings, neighbors_similarities, neighbors_means
    ):
        sum_similarities += similarity
        sum_ratings += similarity * (rating - item_mean)

    prediction += sum_ratings / sum_similarities

    return prediction


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


class BiAKNN(AlgoBase, ABC):
    def __init__(
        self,
        minimum_bicluster_sparsity: Optional[float] = None,
        minimum_bicluster_coverage: Optional[float] = None,
        minimum_bicluster_relative_size: Optional[int] = None,
        knn_type: str = "item",
        user_binarization_threshold: float = 1.0,
        number_of_top_k_biclusters: Optional[int] = 8,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        assert isinstance(minimum_bicluster_sparsity, float) or minimum_bicluster_sparsity is None
        if minimum_bicluster_sparsity is not None:
            assert 0 <= minimum_bicluster_sparsity <= 1

        assert isinstance(minimum_bicluster_coverage, float) or minimum_bicluster_coverage is None
        if minimum_bicluster_coverage is not None:
            assert 0 <= minimum_bicluster_coverage <= 1

        assert (
            isinstance(minimum_bicluster_relative_size, float)
            or minimum_bicluster_relative_size is None
        )
        if minimum_bicluster_relative_size is not None:
            assert 0 <= minimum_bicluster_relative_size <= 1

        assert knn_type in ["user", "item"]

        assert isinstance(user_binarization_threshold, float)
        assert user_binarization_threshold >= 0

        assert isinstance(number_of_top_k_biclusters, int) or number_of_top_k_biclusters is None
        if number_of_top_k_biclusters is not None:
            assert number_of_top_k_biclusters > 0

        assert isinstance(knn_k, int)
        assert knn_k > 0

        # Bicluster filtering parameters
        self.minimum_bicluster_sparsity = minimum_bicluster_sparsity
        self.minimum_bicluster_coverage = minimum_bicluster_coverage
        self.minimum_bicluster_relative_size = minimum_bicluster_relative_size

        # User-item neighborhood parameters
        self.user_binarization_threshold = user_binarization_threshold
        self.number_of_top_k_biclusters = number_of_top_k_biclusters

        # KNN parameters
        self.knn_type = knn_type
        self.knn_k = knn_k

        # Other internal attributes
        self.logger = logger
        self.dataset = None
        self.neighborhood = {}
        self.similarity_matrix = None
        self.means = None
        self.biclusters = None

    @abstractmethod
    def compute_biclusters_from_trainset(self) -> List[np.ndarray]:
        """
        This method is responsible for computing the biclusters that will be used to generate the
        user-item neighborhood for each user in the dataset. Subclasses must implement this method.

        This method will be called after the loading of the trainset. Therefore, the trainset
        attribute will be available and should be used to generate the user-item neighborhood.

        The biclusters must be returned as a list of Concepts.

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
        self.compute_biclusters_from_trainset()
        self.logger.info("Computed biclusters: %d", len(self.biclusters))

        if self.minimum_bicluster_sparsity:
            self.biclusters = apply_bicluster_sparsity_filter(
                self.dataset, self.biclusters, self.minimum_bicluster_sparsity
            )

        if self.minimum_bicluster_coverage:
            self.biclusters = apply_bicluster_coverage_filter(
                self.dataset, self.biclusters, self.minimum_bicluster_coverage
            )

        if self.minimum_bicluster_relative_size:
            self.biclusters = apply_bicluster_relative_size_filter(
                self.dataset, self.biclusters, self.minimum_bicluster_relative_size
            )

        if not self.number_of_top_k_biclusters:
            self.number_of_top_k_biclusters = len(self.biclusters)

        self.logger.info("Filtered biclusters: %d", len(self.biclusters))

        for user_id in range(self.dataset.shape[0]):
            user_as_tidset = get_indices_above_threshold(
                self.dataset[user_id], self.user_binarization_threshold
            )

            top_k_biclusters = get_top_k_biclusters_for_user(
                self.biclusters, user_as_tidset, self.number_of_top_k_biclusters
            )

            merged_bicluster = merge_biclusters(top_k_biclusters)

            if self.knn_type == "user":
                neighborhood = merged_bicluster.extent
            else:
                itens_rated_by_user_and_in_neighborhood = (
                    self.dataset[user_id, merged_bicluster.intent] > 0
                )
                neighborhood = merged_bicluster.intent[itens_rated_by_user_and_in_neighborhood]

            self.neighborhood[user_id] = neighborhood

        if self.knn_type == "user":
            n = self.trainset.n_users
            ratings_map = self.trainset.ur.items()
        else:
            n = self.trainset.n_items
            ratings_map = self.trainset.ir.items()

        self.means = np.zeros(n)
        for ratings_id, ratings in ratings_map:
            self.means[ratings_id] = np.mean([r for (_, r) in ratings])

        self.similarity_matrix = np.full((n, n), dtype=float, fill_value=np.NAN)

    def estimate(self, user: int, item: int):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        if self.knn_type == "user":
            x, y = user, item
            dataset = self.dataset
        else:
            x, y = item, user
            dataset = self.dataset.T

        prediction = self.means[x]
        actual_k = 0

        if self.neighborhood[user].size == 0:
            details = {"actual_k": actual_k}
            return prediction, details

        compute_neighborhood_cosine_similarity(
            dataset, self.similarity_matrix, x, self.neighborhood[user]
        )

        targets_neighborhood_similarity = self.similarity_matrix[x, self.neighborhood[user]]

        all_neighborhood_ratings = dataset[self.neighborhood[user], y]

        non_null_mask = all_neighborhood_ratings > 0
        targets_neighborhood_ratings = all_neighborhood_ratings[non_null_mask]
        targets_neighborhood_similarity = targets_neighborhood_similarity[non_null_mask]

        means = self.means[self.neighborhood[user]]

        k_top_neighbors_indices = np.argsort(targets_neighborhood_similarity)[-self.knn_k :]

        k_top_neighbors_ratings = targets_neighborhood_ratings[k_top_neighbors_indices]
        k_top_neighbors_similarity = targets_neighborhood_similarity[k_top_neighbors_indices]
        k_top_means = means[k_top_neighbors_indices]

        sum_similarities = sum_ratings = 0
        for rating, similarity, item_mean in zip(
            k_top_neighbors_ratings, k_top_neighbors_similarity, k_top_means
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
