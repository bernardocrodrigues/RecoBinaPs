import logging
import numpy as np

from surprise import AlgoBase, Trainset
from fca import BinaryDataset
from dataset.common import convert_trainset_into_rating_matrix

from . import DEFAULT_LOGGER

import logging
from typing import List
from tempfile import TemporaryDirectory
import numpy as np

from binaps.binaps_wrapper import run_binaps, get_patterns_from_weights

from . import DEFAULT_LOGGER


def get_pattern_sparsities(ratings_matrix: np.array, patterns: List[np.array]) -> np.array:
    """
    This function calculates the sparsity of each pattern set in the dataset. The sparsity is defined as the percentage
    of non-zero ratings in the pattern set. This excerpt was taken from pedro/rec_functions.py lines 48 to 53.

    Args:
        ratings_matrix (np.array): The ratings matrix.
        patterns (List[np.array]): The pattern sets.

    Returns:
        np.array: The sparsity of each pattern set.
    """

    n_pat = len(patterns)
    sparsity = np.zeros(n_pat)
    for i in np.arange(n_pat):
        ratings = ratings_matrix[:, patterns[i]]
        sparsity[i] = 100 * np.sum(ratings > 0) / ratings.size

    return sparsity


def filter_patterns(patterns: List[np.array], sparsity: np.array, t_spar: float) -> List[np.array]:
    n_pat = len(patterns)
    ind = np.arange(n_pat)[sparsity > t_spar]
    sel_patterns = [patterns[i] for i in ind]

    return sel_patterns


from pedro.rec_functions import (
    item_similarity,
    item_target_similarity,
    user_pattern_similarity,
    get_metrics,
)
from surprise import AlgoBase, PredictionImpossible


class PedroRecommender(AlgoBase):
    """
    This class implements Pedro's recommender system.

    Args:
        epochs (int, optional): The number of epochs to run BinaPs. Defaults to 100.
        weights_binarization_threshold (float, optional): The threshold to binarize the weights. Defaults to 0.2.
        dataset_binarization_threshold (float, optional): The threshold to binarize the dataset. Defaults to 1.0.
        t_spars (float, optional): The threshold for sparsity on pattern set selection. Defaults to 5.0.
        n_top_pat (int, optional): The number of most similar pattern sets to be selected. Defaults to 5.
        logger (logging.Logger, optional): The logger to be used. Defaults to DEFAULT_LOGGER.

    """

    def __init__(
        self,
        epochs: int = 100,
        weights_binarization_threshold: float = 0.2,
        dataset_binarization_threshold: float = 1.0,
        t_spar: float = 5.0,
        n_top_pat: int = 5,
        k_top_items: int = 5,
        logger: logging.Logger = DEFAULT_LOGGER,
    ):
        AlgoBase.__init__(self)

        self.logger = logger

        # Dataset binarization attributes
        self.dataset_binarization_threshold = dataset_binarization_threshold
        self.binary_dataset: BinaryDataset = None
        self.rating_matrix: np.array = None

        # BinaPs attributes
        self.epochs = epochs
        self.weights_binarization_threshold = weights_binarization_threshold

        # Pedro's attributes
        self.t_spar = t_spar
        self.n_top_pat = n_top_pat
        self.k_top_items = k_top_items

        self.mean_rating: np.array = None
        self.patterns: List = None
        self.sel_patterns: List = None
        self.sim_user = None
        self.sim_item = None

    @classmethod
    def from_previously_computed_patterns(cls, patterns: List[np.array]) -> "PedroRecommender":
        recommender = cls()
        recommender.patterns = patterns

        return recommender

    def fit(self, trainset: Trainset):
        AlgoBase.fit(self, trainset)

        self.rating_matrix = convert_trainset_into_rating_matrix(trainset)

        # Generate binary dataset
        self.logger.debug("Generating binary dataset...")
        self.binary_dataset = BinaryDataset.load_from_trainset(
            trainset, threshold=self.dataset_binarization_threshold
        )
        self.logger.debug("Generating binary dataset OK")

        self.logger.info("Mining Patterns...")
        # If patterns were not previously computed, run BinaPs
        if not self.patterns:
            # Run BinaPs from a temporary file since it only accepts file paths
            with TemporaryDirectory() as temporary_directory:
                with open(f"{temporary_directory}/dataset", "w+", encoding="UTF-8") as file_object:
                    self.binary_dataset.save_as_binaps_compatible_input(file_object)
                    weights, _, _ = run_binaps(
                        input_dataset_path=file_object.name, epochs=self.epochs
                    )

            self.patterns = get_patterns_from_weights(
                weights=weights, threshold=self.weights_binarization_threshold
            )
        self.logger.info("Mining Patterns OK")

        sparsities = get_pattern_sparsities(self.rating_matrix, self.patterns)
        self.sel_patterns = filter_patterns(self.patterns, sparsities, self.t_spar)

        n_rows, n_cols = self.rating_matrix.shape

        self.mean_rating = np.mean(
            self.rating_matrix,
            axis=1,
            where=self.rating_matrix > 0.0,
        )  # Mean rating for each user

        self.sim_item = np.full((n_cols, n_cols), -np.inf)  # Initializes item similarity matrix

        # Calculate user-pattern similarity matrix
        self.sim_user = np.zeros((n_rows, len(self.sel_patterns)))
        for user in np.arange(n_rows):
            for pattern in np.arange(len(self.sel_patterns)):
                self.sim_user[user, pattern] = user_pattern_similarity(
                    self.rating_matrix, user, self.sel_patterns[pattern]
                )

        return self

    def estimate(self, user: int, item: int) -> tuple:
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

        id_sort = np.argsort(self.sim_user[user])

        if self.sim_user[user, id_sort[-1]] == 0:
            self.logger.info(f"No similarity between the user and any pattern")
            raise PredictionImpossible("No similarity between the user and any pattern")
        
        # log number of top patterns
        self.logger.debug(f"Top patterns: {id_sort}")
        self.logger.debug(f"Top patterns similarity: {self.sim_user[user][id_sort]}")

        top_pat = id_sort[-self.n_top_pat :]

        self.logger.debug(f"Top K patterns: {top_pat}")
        self.logger.debug(f"Top K patterns similarity: {self.sim_user[user][top_pat]}")

        merged_pat = np.array([], dtype=int)
        for pat in top_pat:
            merged_pat = np.append(merged_pat, self.sel_patterns[pat])
        merged_pat = np.unique(merged_pat)

        self.logger.debug(f"Merged pattern: {merged_pat}")

        # Find the items from merged_pat rated by the target user
        id_merge = self.rating_matrix[user, merged_pat] > 0
        rated_items = merged_pat[id_merge]

        # self.logger.debug(f"Itens from merged pattern that were rated: {id_merge}")
        self.logger.debug(f"Itens rated by user and present in merged pattern: {rated_items}")

        n_rows, n_cols = self.rating_matrix.shape
        self.sim_item = np.full((n_cols, n_cols), -np.inf) # Initializes item similarity matrix

        # Calculate item similarity between target item and rated items
        item_target_similarity(self.rating_matrix, item, rated_items, self.sim_item)
        # Sort rated items by similarity with target item
        id_sort = np.argsort(self.sim_item[item, rated_items])

        self.logger.debug(id_sort)
        self.logger.debug(self.sim_item[item, rated_items][id_sort])

        # If there is no similarity between the target item and any other item, no prediction is calculated
        if self.sim_item[item, rated_items[id_sort[-1]]] == 0:
            self.logger.info(f"No similarity between the target item and any other item")
            raise PredictionImpossible("No similarity between the target item and any other item")

        # Neighborhood: k most similar items to target item
        id_neigh = id_sort[-self.k_top_items :]
        neigh = rated_items[id_neigh]

        self.logger.debug(f"Neighborhood: {neigh}")
        self.logger.debug(f"Mean rating: {self.mean_rating[user]}")

        # Calculate prediction based on the neighborhood
        num = 0.0
        den = 0.0
        for j in neigh:

            self.logger.debug(f"j: {j}")

            j_num = (self.rating_matrix[user, j] - self.mean_rating[user]) * self.sim_item[item, j]

            j_den = abs(self.sim_item[item, j])

            self.logger.debug(f"j: {j}")
            self.logger.debug(f"j_num: {j_num}")
            self.logger.debug(f"j_den: {j_den}")

            num += j_num
            den += j_den



        pred = self.mean_rating[user] + num / den

        self.logger.debug(f"Prediction: {pred}")

        details = {}

        # input()

        return pred, details
