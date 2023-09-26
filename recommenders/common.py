""" 
common.py

This module contains common functions used by the recommenders.
"""

from typing import Tuple, List, Dict
import numpy as np
import numba as nb
from scipy.spatial import distance
from surprise import Trainset, AlgoBase
from surprise.accuracy import mae, rmse

from fca.formal_concept_analysis import BinaryDataset
from evaluation import (
    get_micro_averaged_recall,
    get_macro_averaged_recall,
    get_recall_at_k,
    get_micro_averaged_precision,
    get_macro_averaged_precision,
    get_precision_at_k,
)


def jaccard_distance(A: np.array, B: np.array) -> float:
    """
    Calculate the Jaccard distance between two boolean numpy arrays A and B.

    Args:
        A (np.array): The first boolean input array.
        B (np.array): The second boolean input array.

    Returns:
        float: The Jaccard distance between A and B.

    Example:
        A = np.array([True, True, False, False])
        B = np.array([True, False, True, False])
        distance = jaccard_distance(A, B)
        # Output: 0.5
    """
    return distance.jaccard(A, B)


def get_similarity_matrix(dataset: BinaryDataset, distance_strategy=jaccard_distance):
    """
    Given a BinaryDataset and some method that calculates some distance between two vector,
    computes the similarity matrix between all users (rows).

    The distance strategy must compute the distance between two numpy arrays. A return value of 1
    implies that the vectors are completely different (maximum distance) while a return value of 0
    implies that the vectors are identical (minimum distance).
    """
    similarity_matrix = np.ones((dataset.shape[0], dataset.shape[0]), np.double)

    similarity_matrix = -1 * similarity_matrix

    for i, row1 in enumerate(dataset.binary_dataset):
        for j, row2 in enumerate(dataset.binary_dataset):
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


@nb.njit
def get_cosine_similarity_matrix(dataset: np.array):
    """
    Given a dataset, computes the similarity matrix between all rows using the cosine
    similarity metric. This function is faster than the get_similarity_matrix function
    since it uses numba to compile the code.

    The cosine similarity metric is defined as follows:
        sim(u, v) = cos(u, v) = (u . v) / (||u|| * ||v||)

    where u and v are two vectors and ||u|| is the norm of u.

    Args:
        dataset (np.array): The dataset to compute the similarity matrix from.

    Returns:
        np.array: The similarity matrix.
    """
    similarity_matrix = np.ones((dataset.shape[0], dataset.shape[0]), dtype=np.float32)

    similarity_matrix = -2 * similarity_matrix

    for i, row1 in enumerate(dataset):
        for j, row2 in enumerate(dataset):
            if similarity_matrix[i, j] != -2:
                continue

            if not row1.any() or not row2.any():
                similarity_matrix[i, j] = np.NaN
                similarity_matrix[j, i] = np.NaN
                continue

            # The snippet below was taken from scipy.spatial.distance.cosine
            u = row1
            v = row2
            uv = np.average(u * v)
            uu = np.average(np.square(u))
            vv = np.average(np.square(v))
            dist = 1.0 - uv / np.sqrt(uu * vv)
            a = np.abs(dist)
            dissimilarity = max(0, min(a, 2.0))
            # end of snippet

            similarity = 1 - dissimilarity
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


def get_k_nearest_neighbors(similarity_matrix: np.array, reference: int, k: int) -> np.array:
    """
    Given a similarity matrix, a reference index and a number k, returns the k most similar
    indices to the reference index.

    Args:
        similarity_matrix (np.array): The similarity matrix.
        reference (int): The reference index.
        k (int): The number of nearest neighbors to return.

    """
    similarity_scores = similarity_matrix[reference]

    # Prune NaN values
    similarity_scores_nans = np.isnan(similarity_scores)
    similarity_scores = similarity_scores[~similarity_scores_nans]

    # Order indices in descending order according to its similarity score to the reference
    nearest_neighbors = np.argsort(similarity_scores)[::-1]

    # Get the k most similar cells (excluding the cell i itself)
    k_most_similar = nearest_neighbors[1 : k + 1]

    return k_most_similar


def generic_thread(
    fold: Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]],
    output: Dict,
    recommender: AlgoBase,
    threshold: float = 5.0,
    number_of_top_recommendations: int = 20,
):
    """
    This function is used to parallelize the GreConD recommender. It puts the results on a
    dictionary called 'output'. 'output' is expected to be a Manager().dict() object since it is
    shared between processes.

    Args:
        fold (Tuple[int, Tuple[Trainset, List[Tuple[int, int, float]]]]): The fold to be processed.
        output (Dict): The dictionary to put the results on.
        grecond_recommender (GreConDRecommender): The GreConDRecommender object to use.
        threshold (float): The relevance threshold to use.
        number_of_top_recommendations (int): The number of top recommendations to use.

    Returns:
        None
    """
    fold_index, (trainset, testset) = fold

    recommender.fit(trainset)
    predictions = recommender.test(testset)
    output[fold_index] = {
        "mae": mae(predictions=predictions, verbose=False),
        "rmse": rmse(predictions=predictions, verbose=False),
        "micro_averaged_recall": get_micro_averaged_recall(
            predictions=predictions, threshold=threshold
        ),
        "macro_averaged_recall": get_macro_averaged_recall(
            predictions=predictions, threshold=threshold
        ),
        "recall_at_k": get_recall_at_k(
            predictions=predictions,
            threshold=threshold,
            k=number_of_top_recommendations,
        ),
        "micro_averaged_precision": get_micro_averaged_precision(
            predictions=predictions, threshold=threshold
        ),
        "macro_averaged_precision": get_macro_averaged_precision(
            predictions=predictions, threshold=threshold
        ),
        "precision_at_k": get_precision_at_k(
            predictions=predictions,
            threshold=threshold,
            k=number_of_top_recommendations,
        ),
    }


# Numba Compilation
# Numba uses a Just-In-Time compiler to speed up the execution of the code. The functions need to
# be ran once to be compiled. Therefore, we run the functions at import time to avoid the overhead
# of compiling the functions when they are called.
get_cosine_similarity_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0]]))
