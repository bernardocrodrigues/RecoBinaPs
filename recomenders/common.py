""" 
common.py

This module contains common functions used by the recommenders.
"""

import numpy as np
from scipy.spatial import distance
from fca.formal_concept_analysis import BinaryDataset


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


def cosine_distance(A: np.array, B: np.array) -> float:
    """
    Calculate the cosine distance between two boolean numpy arrays A and B.

    Args:
        A (np.array): The first boolean input array.
        B (np.array): The second boolean input array.

    Returns:
        float: The cosine distance between A and B.

    Example:
        A = np.array([True, True, False, False])
        B = np.array([True, False, True, False])
        distance = cosine_distance(A, B)
        # Output: 0.29289321881345254
    """
    return distance.cosine(A, B)


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
