import numpy as np

from fca.FormalConceptAnalysis import (
    BinaryDataset,
    GreConD,
    get_factor_matrices_from_concepts,
    construct_context_from_binaps_patterns,
)
from surprise import AlgoBase, PredictionImpossible
from scipy.spatial import distance
from abc import ABC, abstractmethod
from binaps.BinapsWrapper import run_binaps_2, get_patterns_from_weights
from typing import List

import logging


def jaccard_distance(A: np.array, B: np.array):
    return distance.jaccard(A, B)


def cosine_distance(A: np.array, B: np.array):
    return distance.cosine(A, B)


def get_similarity_matrix(dataset: BinaryDataset, distance_strategy=jaccard_distance):
    """
    Given a BinaryDataset and some method that calculates some distance between two vector, computes the similarity
    matrix between all users (rows).

    The distance strategy must compute the distance between two numpy arrays. A return value of 1 implies that the
    vectors are completely different (maximum distance) while a return value of 0 implies that the vectors are identical
    (minimum distance).
    """
    similarity_matrix = np.ones((dataset.shape[0], dataset.shape[0]), np.double)

    similarity_matrix = -1 * similarity_matrix

    for i, row1 in enumerate(dataset._binary_dataset):
        for j, row2 in enumerate(dataset._binary_dataset):

            if similarity_matrix[i, j] != -1:
                continue

            if not row1.any() or not row2.any():
                similarity_matrix[i, j] = np.NaN
                continue

            dissimilarity = distance_strategy(row1, row2)
            similarity = 1 - dissimilarity

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


def get_k_nearest_neighbors(similarity_matrix: np.array, reference: int, k: int) -> np.array:
    similarity_scores = similarity_matrix[reference]

    # Prune NaN values
    similarity_scores_nans = np.isnan(similarity_scores)
    similarity_scores = similarity_scores[~similarity_scores_nans]

    # order indices in descending order acording to its similarity score to the reference
    nearest_neighbors = np.argsort(similarity_scores)[::-1]

    # Get the k most similar cells (excluding the cell i itself)
    k_most_similar = nearest_neighbors[1 : k + 1]

    return k_most_similar


class PatternBasedRecommender(AlgoBase, ABC):
    def __init__(
        self, k=30, threshold=1, distance_strategy=jaccard_distance, logger: logging.Logger = None
    ):
        AlgoBase.__init__(self)

        self.logger = logger if logger is not None else self.get_logger()

        # Binarization
        self.threshold = threshold

        # KNN parameters
        self.k = k
        self.distance_strategy = distance_strategy

        self.formal_context = None
        self.number_of_factors = None

    @staticmethod
    def get_logger():
        logger = logging.getLogger("aaa")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    @abstractmethod
    def generate_formal_context(self):
        pass

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.logger.info("Generating binary dataset...")

        self.binary_dataset = BinaryDataset.load_from_trainset(trainset, threshold=self.threshold)

        self.generate_formal_context()
        self.number_of_factors = len(self.formal_context)

        self.Af, self.Bf = get_factor_matrices_from_concepts(
            self.formal_context, self.binary_dataset.shape[0], self.binary_dataset.shape[1]
        )
        latent_binary_dataset = BinaryDataset(self.Af)

        self.logger.info("Generating Similarity Matrix...")

        self.sim = get_similarity_matrix(latent_binary_dataset, self.distance_strategy)

        self.logger.info("Generating Similarity Matrix OK")

        return self

    def estimate(self, user, item):
        if not (self.trainset.knows_user(user) and self.trainset.knows_item(item)):
            raise PredictionImpossible("User and/or item is unknown.")

        ruid = self.trainset.to_raw_uid(user)
        riid = self.trainset.to_raw_iid(user)

        rating = 0
        weight_sum = 0
        neighbors_used = []

        neighbors = [
            (other_user, self.sim[user, other_user], rating)
            for (other_user, rating) in self.trainset.ir[item]
            
        ]

        neighbors = [a for a in neighbors if not np.isnan(a[1])]


        nearest_neighbors = sorted(neighbors, key=lambda d: d[1], reverse=True)

        for neighbor in nearest_neighbors:
            if len(neighbors_used) >= self.k:
                break

            neighbor_iid = neighbor[0]
            neighbor_similarity = neighbor[1]
            neighbor_rating = neighbor[2]

            neighbor_ruid = self.trainset.to_raw_uid(neighbor_iid)

            if neighbor_similarity == 0:
                continue

            rating += neighbor_similarity * neighbor_rating
            weight_sum += neighbor_similarity
            neighbors_used.append((neighbor_ruid, neighbor_similarity, neighbor_rating))

        if not neighbors_used:
            raise PredictionImpossible("Not enough neighbors.")

        rating /= weight_sum

        details = {"actual_k": len(neighbors_used), "neighbors_used": neighbors_used}

        return rating, details


class FcaBmf(PatternBasedRecommender):
    def __init__(
        self,
        k=30,
        coverage=1.0,
        threshold=1,
        distance_strategy=jaccard_distance,
        logger: logging.Logger = None,
    ):
        super().__init__(k, threshold, distance_strategy, logger)
        self.coverage = coverage
        self.actual_coverage = None

    def generate_formal_context(self):
        self.logger.info("Generating binary dataset OK!")
        self.logger.info(
            f"Resulting binary dataset is {self.binary_dataset.shape[0]} rows x {self.binary_dataset.shape[1]} columns"
        )

        self.logger.info("Generating Formal Context...")

        self.formal_context, self.actual_coverage = GreConD(
            self.binary_dataset, coverage=self.coverage, logger=self.logger
        )

        self.logger.info("Generating Formal Context OK")


import io


class BinapsRecommender(PatternBasedRecommender):
    def __init__(
        self,
        epochs=100,
        binarization_threshold=0.2,
        k=30,
        threshold=1,
        distance_strategy=jaccard_distance,
        logger: logging.Logger = None,
    ):
        super().__init__(k, threshold, distance_strategy, logger)

        self.epochs = epochs
        self.binarization_threshold = binarization_threshold

        self.patterns = None

    @classmethod
    def from_previously_computed_patterns(cls, patterns, k, threshold, distance_strategy):
        recommender = cls(k=k, threshold=threshold, distance_strategy=distance_strategy)
        recommender.patterns = patterns

        return recommender

    def generate_formal_context(self):
        if self.patterns:
            pass
        else:
            with open("teste", "w") as file_object:
                self.binary_dataset.save_as_binaps_compatible_input(file_object)
            weights, training_losses, test_losses = run_binaps_2(
                input_dataset_path="teste", epochs=self.epochs
            )
            self.patterns = get_patterns_from_weights(
                weights=weights, threshold=self.binarization_threshold
            )

        self.formal_context = construct_context_from_binaps_patterns(
            self.binary_dataset, self.patterns, True
        )
