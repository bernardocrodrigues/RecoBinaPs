""" binaps_recommender.py

This module contains the BinaPsRecommender class.
"""

import logging
from typing import List
from tempfile import TemporaryDirectory
import numpy as np

from fca.formal_concept_analysis import construct_context_from_binaps_patterns
from binaps.binaps_wrapper import run_binaps, get_patterns_from_weights

from . import DEFAULT_LOGGER
from .common import jaccard_distance
from .formal_context_based_recommender import KNNOverLatentSpaceRecommender


class BinaPsRecommender(KNNOverLatentSpaceRecommender):
    """
    A recommender based on the BinaPs algorithm. BinaPs mines succinct patterns from a dataset.
    From these patterns, a formal context is constructed and the kNN algorithm is used to generate
    recommendations.

    Args:
        epochs (int): The number of epochs to train BinaPs.
        hidden_dimension_neurons_number (int): The number of neurons in the hidden layer. This
                                               equals the max number of patterns to mine. If set
                                               to -1, the number of neurons is set to the number
                                               of items in the dataset. The bigger the number of
                                               neurons, the bigger the size of the underlying
                                               autoenconder and thus the longer the training time.
        weights_binarization_threshold (float): The threshold for binarizing the weights into
                                                patterns.
        dataset_binarization_threshold (float): The threshold for binarizing the dataset.
        knn_k (int): The number of neighbors to consider in the kNN step.
        knn_distance_strategy (callable): The distance function to use.
        logger (logging.Logger): The logger for logging messages.

    Example:
        recommender = BinaPsRecommender(epochs=100, weights_binarization_threshold=0.2)
        recommender.fit(trainset)
        predictions = recommender.test(testset)
    """

    def __init__(
        self,
        epochs: int = 100,
        hidden_dimension_neurons_number: int = -1,
        weights_binarization_threshold: float = 0.2,
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

        self.epochs = epochs
        self.hidden_dimension_neurons_number = hidden_dimension_neurons_number
        self.weights_binarization_threshold = weights_binarization_threshold

        self.patterns = None

    @classmethod
    def from_previously_computed_patterns(cls, patterns: List[np.array]) -> "BinaPsRecommender":
        """
        Create a BinaPsRecommender from previously computed patterns.

        Args:
            patterns (List[np.array]): The patterns to use.
        """
        recommender = cls()
        recommender.patterns = patterns

        return recommender

    def generate_formal_context(self):
        self.logger.info("Generating Formal Context...")

        if not self.patterns:
            # If patterns were not previously computed, run BinaPs
            self.logger.debug("No patterns were previously computed. Running BinaPs...")

            with TemporaryDirectory() as temporary_directory:
                # Run BinaPs from a temporary file since it only accepts file paths
                with open(f"{temporary_directory}/dataset", "w+", encoding="UTF-8") as file_object:
                    self.binary_dataset.save_as_binaps_compatible_input(file_object)
                    self.logger.debug("Dataset saved to temporary file at %s", file_object.name)
                    self.logger.debug(
                        "Calling run_binaps({}, {}, {})".format(
                            file_object.name, self.epochs, self.hidden_dimension_neurons_number
                        )
                    )
                    weights, _, _ = run_binaps(
                        input_dataset_path=file_object.name,
                        epochs=self.epochs,
                        hidden_dimension=self.hidden_dimension_neurons_number,
                    )
                    self.logger.debug("BinaPs OK")
            
            self.logger.debug("Binarizing weights...")
            self.patterns = get_patterns_from_weights(
                weights=weights, threshold=self.weights_binarization_threshold
            )
            self.logger.debug("Binarizing weights OK")

        self.logger.debug("Constructing Formal Context...")
        self.formal_context = construct_context_from_binaps_patterns(
            self.binary_dataset, self.patterns, True
        )
        
        self.logger.info("Generating Formal Context OK")
