""" random_dataset.py

This module contains the RandomDataset class.

Copyright 2023 Bernardo C. Rodrigues
See LICENSE file for license details
"""

from surprise import Dataset, Reader
from .common import generate_random_dataset, convert_to_raw_ratings


class RandomDataset(Dataset):
    """
    This class represents a random dataset.
    """
    def __init__(
        self, number_of_users: int, number_of_items: int, rating_scale: int, sparsity_target: float
    ):
        Dataset.__init__(
            self,
            Reader(
                line_format="user item rating timestamp", rating_scale=(0, rating_scale), sep="\t"
            ),
        )

        dataset = generate_random_dataset(number_of_users, number_of_items, rating_scale, sparsity_target)
        self.raw_ratings = convert_to_raw_ratings(dataset)
