"""
formal_context_based_recommender.py

This module defines various classes for recommendation engines based on formal
concept analysis (FCA). The classes in this module are used for generating
recommendations using the Binaps, GreConD, and FCA-BMF algorithms.
"""

import logging
from typing import Callable, List
from abc import ABC, abstractmethod
import numpy as np

from surprise import AlgoBase, PredictionImpossible, Trainset
from pattern_mining.formal_concept_analysis import get_factor_matrices_from_concepts, Concept
from dataset.binary_dataset import load_from_trainset

from . import DEFAULT_LOGGER
from .common import get_cosine_similarity_matrix


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

        self.binary_dataset = load_from_trainset(
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
        Estimates the rating of a given user for a given item. This function is not supposed to be called directly since
        it uses the Surprise's internal user and item ids. Surprise uses this callback internally to make predictions.
        Use the predict() or test() methods instead which use the raw user and item ids.

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
