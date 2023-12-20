""" common.py

This module contains common functions for the dataset module.

"""

from typing import List, Tuple
import numpy as np
import numba as nb
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


def load_dataset_from_trainset(trainset: Trainset) -> np.ndarray:
    """
    Loads a numpy array from a surprise Trainset object.

    Args:
        trainset (Trainset): The Trainset object.

    Returns:
        np.ndarray: The loaded dataset.
    """

    assert isinstance(trainset, Trainset)

    dataset = np.full((trainset.n_users, trainset.n_items), dtype=np.float64, fill_value=np.NAN)

    for uid, iid, rating in trainset.all_ratings():
        dataset[uid][iid] = rating

    return dataset


@nb.njit
def generate_random_dataset(
    number_of_users: int, number_of_items: int, rating_scale: int, sparsity_target: float
) -> np.ndarray:
    """
    Generates a random dataset.

    Args:
        number_of_users (int): The number of users (rows).
        number_of_items (int): The number of items (columns).
        rating_scale (int): The rating scale. The ratings will be in the range [0, rating_scale].
        sparsity_target (float): The sparsity target. The probability of any rating being generated
            is equal to sparsity_target. Therefore, the sparsity of the dataset will be close to
            sparsity_target.

    Returns:
        np.ndarray: The generated dataset.
    """
    dataset = np.zeros((number_of_users, number_of_items), dtype=np.float32)

    for i in range(number_of_users):
        for j in range(number_of_items):
            if np.random.uniform() < sparsity_target:
                dataset[i][j] = np.random.uniform() * rating_scale

    return dataset


@nb.njit
def convert_to_raw_ratings(dataset: np.ndarray) -> List[Tuple[int, int, float, None]]:
    """
    Converts a dataset into a list of raw ratings as expected by surprise's Trainset class.

    Args:
        dataset (np.ndarray): The dataset to be converted.

    Returns:
        List[Tuple[int, int, float, None]]: The list of raw ratings.
    """
    raw_ratings = []

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            if dataset[i][j] != 0:
                raw_ratings.append((i, j, float(dataset[i][j]), None))

    return raw_ratings


# Numba Compilation
# Numba uses a Just-In-Time compiler to speed up the execution of the code. The functions need to
# be ran once to be compiled. Therefore, we run the functions at import time to avoid the overhead
# of compiling the functions when they are called.
aux = generate_random_dataset(1, 1, 1, 1)
aux_2 = convert_to_raw_ratings(aux)
