""" common.py

This module contains common functions for the dataset module.

"""
import numpy as np
from surprise import Trainset


def convert_trainset_into_rating_matrix(trainset: Trainset) -> np.ndarray:
    """
    Convert a surprise Trainset object into a rating matrix.

    Args:
        trainset (Trainset): The Trainset object.

    Returns:
        np.ndarray: The rating matrix.
    """

    shape = (trainset.n_users, trainset.n_items)

    rating_matrix = np.zeros(shape)
    for user_id, item_id, rating in trainset.all_ratings():
        rating_matrix[user_id, item_id] = rating

    return rating_matrix
