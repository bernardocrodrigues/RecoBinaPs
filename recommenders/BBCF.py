import logging
from typing import List, Optional

import numpy as np
from surprise import AlgoBase, PredictionImpossible, Trainset

from dataset.common import convert_trainset_to_matrix

from pattern_mining.formal_concept_analysis import Concept
from pattern_mining.strategies import PatternMiningStrategy

from . import DEFAULT_LOGGER
from .common import (
    get_top_k_biclusters_for_user,
    get_indices_above_threshold,
    merge_biclusters,
    get_similarity_matrix,
)


class BBCF(AlgoBase):
    """
    Bicluster aware kNN (BiAKNN) is an abstract class for recommendation engines based on the kNN
    algorithm. However, instead of using the original dataset, these recommendation engines use
    restricts the neighborhood of each item to a union of biclusters. It extends the functionality
    of the AlgoBase class from the Surprise library and provides methods for generating
    recommendations.

    The method compute_biclusters_from_trainset() must be implemented by subclasses. This method
    is responsible for computing the biclusters from the dataset.
    """

    def __init__(
        self,
        mining_strategy: PatternMiningStrategy,
        knn_type: str = "item",
        number_of_top_k_biclusters: Optional[int] = None,
        knn_k: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        assert knn_type in ["user", "item"]

        assert isinstance(number_of_top_k_biclusters, int) or number_of_top_k_biclusters is None
        if number_of_top_k_biclusters is not None:
            assert number_of_top_k_biclusters > 0

        assert isinstance(knn_k, int)
        assert knn_k > 0

        self.mining_strategy = mining_strategy

        # User-item neighborhood parameters
        self.number_of_top_k_biclusters = number_of_top_k_biclusters

        # KNN parameters
        self.knn_type = knn_type
        self.knn_k = knn_k

        # Other internal attributes
        self.logger = logger
        self.dataset = None
        self.neighborhood = {}
        self.similarity_matrix = {}
        self.means = None
        self.biclusters = None
        self.n = None

        self.trainset = None

    def fit(self, trainset: Trainset):
        """
        Train the algorithm on a given training set.

        Args:
            trainset (Trainset): The training set to train the algorithm on.

        Returns:
            self: The trained algorithm.
        """

        AlgoBase.fit(self, trainset)

        self.dataset = convert_trainset_to_matrix(trainset)
        self.biclusters = self.mining_strategy(trainset)

        if not self.number_of_top_k_biclusters:
            self.number_of_top_k_biclusters = len(self.biclusters)

        self._generate_neighborhood()
        self._calculate_means()

    def _generate_neighborhood(self) -> None:
        """
        Generates the neighborhood for each user based on the dataset and biclusters.

        The neighborhood is determined by selecting the top-k biclusters that are most relevant to
        the user, merging them into a single bicluster, and extracting either the extent or intent
        depending on the knn_type.
        """
        for user_id in range(self.dataset.shape[0]):
            user_as_tidset = get_indices_above_threshold(
                self.dataset[user_id], self.mining_strategy.dataset_binarization_threshold
            )

            merged_bicluster = Concept(np.array([], dtype=float), np.array([], dtype=float))

            if self.number_of_top_k_biclusters and self.biclusters:
                top_k_biclusters = get_top_k_biclusters_for_user(
                    self.biclusters, user_as_tidset, self.number_of_top_k_biclusters
                )
                if top_k_biclusters:
                    merged_bicluster = merge_biclusters(top_k_biclusters)

            self.neighborhood[user_id] = merged_bicluster

    def _calculate_means(self):
        """
        Calculate the mean ratings for each user or item.

        If knn_type is "user", calculate the mean ratings for each user.
        If knn_type is "item", calculate the mean ratings for each item.
        """
        if self.knn_type == "user":
            self.n = self.trainset.n_users
            ratings_map = self.trainset.ur.items()
        else:
            self.n = self.trainset.n_items
            ratings_map = self.trainset.ir.items()

        self.means = np.full((self.n), dtype=np.float64, fill_value=np.NAN)

        for ratings_id, ratings in ratings_map:
            self.means[ratings_id] = np.mean([r for (_, r) in ratings])

    def estimate(self, user: int, item: int):

        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        neighborhood = self.neighborhood[user]

        if not neighborhood.extent.any() or not neighborhood.intent.any():
            raise PredictionImpossible("Not enough neighbors.")

        if user not in neighborhood.extent or item not in neighborhood.intent:
            raise PredictionImpossible("User and/or item is not in the neighborhood.")

        # For simplicity, let's convert a item based problem into a user based transposed problem
        # since they are symmetric.
        if self.knn_type == "user":
            main_index, secondary_index = user, item
            main_slice, secondary_slice = neighborhood.extent, neighborhood.intent
            dataset = self.dataset
        else:
            main_index, secondary_index = item, user
            main_slice, secondary_slice = neighborhood.intent, neighborhood.extent
            dataset = self.dataset.T

        main_index_in_submatrix = np.nonzero(main_slice == main_index)[0][0]
        secondary_index_in_submatrix = np.nonzero(secondary_slice == secondary_index)[0][0]

        submatrix = dataset[main_slice][:, secondary_slice]

        # Lazy computation of similarity matrix
        if user not in self.similarity_matrix:
            self.similarity_matrix[user] = get_similarity_matrix(submatrix)

        similarity_matrix = self.similarity_matrix[user]
        neighborhood_similarity = similarity_matrix[main_index_in_submatrix]

        neighborhood_ratings = submatrix[:, secondary_index_in_submatrix]

        neighborhood_means = self.means[main_slice]

        validity_mask = (
            (~np.isnan(neighborhood_ratings))
            & (~np.isnan(neighborhood_similarity))
            & (neighborhood_similarity != 0)
        )

        if not validity_mask.any():
            raise PredictionImpossible("No valid neighbors.")

        valid_neighborhood_ratings = neighborhood_ratings[validity_mask]
        valid_neighborhood_similarity = neighborhood_similarity[validity_mask]
        valid_neighborhood_means = neighborhood_means[validity_mask]

        ordered_neighbors_indices = np.argsort(valid_neighborhood_similarity)
        top_k_neighbors_indices = ordered_neighbors_indices[
            -min(self.knn_k, len(valid_neighborhood_similarity)) :
        ]

        top_k_neighbors_ratings = valid_neighborhood_ratings[top_k_neighbors_indices]
        top_k_neighbors_similarity = valid_neighborhood_similarity[top_k_neighbors_indices]
        top_k_means = valid_neighborhood_means[top_k_neighbors_indices]

        prediction = self.means[main_index]
        sum_similarities: float = 0.0
        sum_ratings: float = 0.0

        for rating, similarity, item_mean in zip(
            top_k_neighbors_ratings, top_k_neighbors_similarity, top_k_means
        ):
            sum_similarities += similarity
            sum_ratings += similarity * (rating - item_mean)

        prediction += sum_ratings / sum_similarities

        return prediction, {}
