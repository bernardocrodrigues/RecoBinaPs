""" common.py

This module contains common functions for the dataset module.

"""

from typing import List, Tuple
import numpy as np
import numba as nb
from surprise import Trainset


def convert_trainset_to_matrix(trainset: Trainset) -> np.ndarray:
    """
    Converts a surprise Trainset object into a numpy array.

    The indices of the array will be the inner ids of the trainset.The array will have shape
    (n_users, n_items) and will contain the ratings of the trainset. The ratings will be in the
    range [0, rating_scale]. If a rating is missing, the corresponding entry in the array will be
    NaN.

    Args:
        trainset (Trainset): The Trainset object.

    Returns:
        np.ndarray: The converted dataset.

    Raises:
        AssertionError: If trainset is not an instance of surprise.Trainset.
    """

    assert isinstance(trainset, Trainset)

    dataset = np.full((trainset.n_users, trainset.n_items), dtype=np.float64, fill_value=np.NAN)

    for uid, iid, rating in trainset.all_ratings():
        dataset[uid][iid] = rating

    return dataset


def generate_random_dataset(
    number_of_users: int,
    number_of_items: int,
    rating_scale: int,
    sparsity_target: float,
    seed: int = 42,
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
        seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
        np.ndarray: The generated dataset.
    """

    assert isinstance(number_of_users, int)
    assert number_of_users > 0

    assert isinstance(number_of_items, int)
    assert number_of_items > 0

    assert isinstance(rating_scale, int)
    assert rating_scale > 0

    assert isinstance(sparsity_target, float)
    assert 0 < sparsity_target <= 1

    dataset = np.full((number_of_users, number_of_items), dtype=np.float64, fill_value=np.NAN)

    random_generator = np.random.Generator(np.random.PCG64(seed=seed))

    for i in range(number_of_users):
        for j in range(number_of_items):
            if random_generator.uniform() < sparsity_target:
                dataset[i][j] = random_generator.uniform() * rating_scale

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
