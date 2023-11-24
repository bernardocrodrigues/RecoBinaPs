""" binaps_recommender.py

This module contains all recommenders based on the BinaPS algorithm.
"""

import logging
import multiprocessing
import numpy as np
from tempfile import TemporaryDirectory

from surprise import AlgoBase, PredictionImpossible, Trainset

from binaps.binaps_wrapper import run_binaps, get_patterns_from_weights
from pattern_mining.common import filter_patterns_based_on_bicluster_sparsity
from dataset.common import load_dataset_from_trainset
from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
    save_as_binaps_compatible_input,
)
from . import DEFAULT_LOGGER
from .common import (
    get_top_k_patterns_for_user,
    compute_targets_neighborhood_cosine_similarity,
)


def mine_patterns(
    trainset: np.ndarray,
    dataset_binarization_threshold,
    epochs,
    hidden_dimension_neurons_number,
    weights_binarization_threshold,
):
    binary_dataset = load_binary_dataset_from_trainset(
        trainset, threshold=dataset_binarization_threshold
    )

    with TemporaryDirectory() as temporary_directory:
        with open(f"{temporary_directory}/dataset", "w+", encoding="UTF-8") as file_object:
            save_as_binaps_compatible_input(binary_dataset, file_object)

            weights, _, _ = run_binaps(
                input_dataset_path=file_object.name,
                epochs=epochs,
                hidden_dimension=hidden_dimension_neurons_number,
            )

    patterns = get_patterns_from_weights(weights, weights_binarization_threshold)

    return patterns


class BinaPsKNNRecommender(AlgoBase):
    """
    A KNN recommender that 
    """

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

        patterns = mine_patterns(
            trainset=trainset,
            dataset_binarization_threshold=self.dataset_binarization_threshold,
            epochs=self.epochs,
            hidden_dimension_neurons_number=self.hidden_dimension_neurons_number,
            weights_binarization_threshold=self.weights_binarization_threshold,
        )

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

    def parallel_test(self, testset, verbose=False):
        thread_args = [(self, uid, iid, r_ui_trans, verbose) for (uid, iid, r_ui_trans) in testset]

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            predictions = pool.starmap(BinaPsKNNRecommender.predict, iterable=thread_args)

        return predictions


# def count_impossible_predictions(predictions):
#     count = 0
#     for prediction in predictions:
#         if prediction.details["was_impossible"]:
#             count += 1
#     return count


# from surprise.accuracy import mae, rmse
# from rich.progress import track
# from surprise import Dataset, Reader, similarities

# from surprise.model_selection import KFold, PredefinedKFold
# from pathlib import Path
# from scripts.generate_movielens_folds import download_movielens

# output_dir = Path("/tmp/movielens")
# # movielens_1m_path = output_dir / "ml-1m"

# # download_movielens(movielens_1m_path, '1m')

# # data = Dataset.load_from_file(str(movielens_1m_path / "ratings.dat"), reader=Reader("ml-1m"))
# # kf = KFold(n_splits=5)
# # folds = [(index, fold) for index, fold in enumerate(kf.split(data))]


# movielens_100k_path = output_dir / "ml-100k"
# download_movielens(movielens_100k_path, "100k")
# reader = Reader("ml-100k")
# folds_files = [
#     (movielens_100k_path / f"u{i}.base", movielens_100k_path / f"u{i}.test")
#     for i in (1, 2, 3, 4, 5)
# ]
# data = Dataset.load_from_folds(folds_files, reader=reader)
# kf = PredefinedKFold()

# folds = [(index, fold) for index, fold in enumerate(kf.split(data))]


# recommender = BinaPsKNNRecommender()

# for index, (trainset, testset) in folds:
#     print("fold:", index)
#     recommender.fit(trainset)
#     predictions = recommender.test(testset)

#     print("predictions:", len(predictions))
#     print("impossible predictions:", count_impossible_predictions(predictions))

#     mae(predictions, verbose=True)
#     rmse(predictions, verbose=True)
#     # print()

#     # break


# docker compose run --rm --entrypoint="python3" notebook-cuda -m recommenders.binaps_based_recommenders
# docker compose run --rm --entrypoint="python3" notebook-cuda -m cProfile -o profile -m recommenders.binaps_based_recommenders
