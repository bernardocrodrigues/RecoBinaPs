from typing import List
from abc import ABC, abstractmethod

import numpy as np
from surprise import AlgoBase, Trainset

from dataset.binary_dataset import load_binary_dataset_from_trainset


class PatternBasedRecommender(AlgoBase, ABC):
    def __init__(
        self,
        dataset_binarization_threshold: float = 1.0,
    ):
        AlgoBase.__init__(self)

        self.dataset_binarization_threshold = dataset_binarization_threshold

        self.patterns = None

    @abstractmethod
    def compute_patterns_from_binary_dataset(self, binary_dataset) -> List[np.ndarray]:
        """
        This method is responsible for computing the patterns

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

        binary_dataset = load_binary_dataset_from_trainset(
            self.trainset, threshold=self.dataset_binarization_threshold
        )
        self.compute_patterns_from_binary_dataset(binary_dataset)

        del binary_dataset
