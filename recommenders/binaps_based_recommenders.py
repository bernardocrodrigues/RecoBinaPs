""" binaps_recommender.py

This module contains all recommenders based on the BinaPS algorithm.
"""

import logging
from tempfile import TemporaryDirectory

from binaps.binaps_wrapper import run_binaps, get_patterns_from_weights
from dataset.binary_dataset import (
    load_binary_dataset_from_trainset,
    save_as_binaps_compatible_input,
)

from .knn_based_recommenders import KNNOverItemNeighborhood
from . import DEFAULT_LOGGER


class BinaPsKNNRecommender(KNNOverItemNeighborhood):
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
        super().__init__(
            dataset_binarization_threshold=dataset_binarization_threshold,
            minimum_pattern_bicluster_sparsity=minimum_pattern_bicluster_sparsity,
            user_binarization_threshold=user_binarization_threshold,
            top_k_patterns=top_k_patterns,
            knn_k=knn_k,
            logger=logger,
        )

        self.epochs = epochs
        self.hidden_dimension_neurons_number = hidden_dimension_neurons_number
        self.weights_binarization_threshold = weights_binarization_threshold
        self.dataset_binarization_threshold = dataset_binarization_threshold

    def compute_patterns_from_trainset(self):
        binary_dataset = load_binary_dataset_from_trainset(
            self.trainset, threshold=self.dataset_binarization_threshold
        )

        with TemporaryDirectory() as temporary_directory:
            with open(f"{temporary_directory}/dataset", "w+", encoding="UTF-8") as file_object:
                save_as_binaps_compatible_input(binary_dataset, file_object)

                weights, _, _ = run_binaps(
                    input_dataset_path=file_object.name,
                    epochs=self.epochs,
                    hidden_dimension=self.hidden_dimension_neurons_number,
                )

        self.patterns = get_patterns_from_weights(weights, self.weights_binarization_threshold)
