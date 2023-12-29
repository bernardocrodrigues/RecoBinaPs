""" 
common.py

This module contains common functions used by the recommenders.
"""

import math
from typing import List
import numpy as np
import numba as nb
import pandas as pd

from scipy.spatial import distance

from pattern_mining.formal_concept_analysis import Concept


def jaccard_distance(a: np.array, b: np.array) -> float:
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
    return distance.jaccard(a, b)


def get_similarity_matrix(dataset, distance_strategy=jaccard_distance):
    """
    Given a np.array and some method that calculates some distance between two vector,
    computes the similarity matrix between all users (rows).

    The distance strategy must compute the distance between two numpy arrays. A return value of 1
    implies that the vectors are completely different (maximum distance) while a return value of 0
    implies that the vectors are identical (minimum distance).
    """
    similarity_matrix = np.ones((dataset.shape[0], dataset.shape[0]), np.double)

    similarity_matrix = -1 * similarity_matrix

    for i, row1 in enumerate(dataset):
        for j, row2 in enumerate(dataset):
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

    assert isinstance(dataset, np.ndarray)
    assert dataset.ndim == 2
    assert dataset.shape[0] > 0
    assert dataset.shape[1] > 0

    similarity_matrix = _get_cosine_similarity_matrix(dataset)

    assert similarity_matrix.shape == (dataset.shape[0], dataset.shape[0])
    assert np.all(np.isfinite(similarity_matrix))

    return similarity_matrix


def get_user_pattern_similarity(user: np.ndarray, pattern: np.ndarray) -> float:
    """
    Calculates the similarity between a user and a pattern based on the number of items they have
    in common. The similarity is defined as follows:

            similarity = |I_u ∩ I_p| / |I_u ∩ I_p| + |I_p - I_u|

        where I_u is the set of relevant items for the user and I_p is the set of relevant items
        for the pattern.

    This similarity metric is used is defined by Symeonidis[1].

    [1] Symeonidis, P., Nanopoulos, A., Papadopoulos, A., & Manolopoulos, Y. (n.d.).
        Nearest-Biclusters Collaborative Filtering with Constant Values. Lecture Notes in Computer
        Science, 36-55. doi:10.1007/978-3-540-77485-3

    Args:
        user (np.ndarray): A 1D numpy array representing the user's relevant items. The array must
        be an itemset representation.
        pattern (np.ndarray): A 1D numpy array representing the pattern's relevant items.The array
        must be an itemset representation.

    Returns:
        float: A value between 0 and 1 representing the similarity between the user and the pattern.
    """
    assert isinstance(user, np.ndarray)
    assert user.dtype == np.int64
    assert user.ndim == 1

    assert isinstance(pattern, np.ndarray)
    assert pattern.dtype == np.int64
    assert pattern.ndim == 1

    number_of_itens_from_pattern_in_user = np.intersect1d(user, pattern).size
    try:
        similarity = number_of_itens_from_pattern_in_user / pattern.size
    except ZeroDivisionError:
        similarity = 0.0

    return similarity


def get_top_k_biclusters_for_user(
    biclusters: List[Concept], user_as_tidset: np.array, number_of_top_k_patterns: int
) -> List[Concept]:
    """
    Gets the top-k patterns for a given user. The top-k patterns are the patterns that
    have the highest similarity with the user.

    Args:
        patterns (List[Concept]): The patterns that will be analyzed. Each pattern must be an
                                    itemset representation.
        user_as_tidset (np.array): The target user. The array must be an tidset representation.
        number_of_top_k_patterns (int): The number of patterns to return.

    Returns:
        List[Concept]: The top-k patterns. The patterns are sorted in ascending order of
                        similarity.
    """
    assert isinstance(biclusters, list)
    assert all(isinstance(bicluster, Concept) for bicluster in biclusters)

    assert isinstance(user_as_tidset, np.ndarray)
    assert user_as_tidset.dtype == np.int64
    assert user_as_tidset.ndim == 1

    assert isinstance(number_of_top_k_patterns, int)
    assert number_of_top_k_patterns > 0

    similar_biclusters = []
    similarities = []

    for bicluster in biclusters:
        similarity = get_user_pattern_similarity(user_as_tidset, bicluster.intent)
        if similarity > 0:
            similar_biclusters.append(bicluster)
            similarities.append(similarity)

    similarities = np.array(similarities, dtype=np.float64)
    sorted_similarities_indexes = np.argsort(similarities)
    sorted_similar_patterns = [similar_biclusters[i] for i in sorted_similarities_indexes]
    top_k_patterns = sorted_similar_patterns[-number_of_top_k_patterns:]

    return top_k_patterns


def get_sparse_representation_of_the_bicluster(
    bicluster: np.ndarray, bicluster_column_indexes: List[int]
) -> pd.DataFrame:
    """
    Gets the sparse representation of a bicluster. The sparse representation is a list of tuples
    (user, item, rating).

    This representation is compatible with the Surprise's Dataset class (see Dataset.load_from_df).

    Args:
        bicluster (np.ndarray): The bicluster.
        bicluster_column_indexes (List[int]): The original column indexes of the columns in the
                                              bicluster. This is necessary to map the columns in
                                              the bicluster to the original columns in the dataset.
                                              For example, column 0 in the bicluster may be column
                                              3 in the original dataset. This will be reflected in
                                              'item' field of the tuple.

    Returns:
        pandas.DataFrame: The sparse representation of the bicluster.
    """
    raw_dataframe = []

    for i, row in enumerate(bicluster):
        for j, item in enumerate(row):
            if item > 0:
                raw_dataframe.append((i, bicluster_column_indexes[j], item))

    dataframe = pd.DataFrame(raw_dataframe, columns=["user", "item", "rating"])

    return dataframe


@nb.njit
def _get_cosine_similarity_matrix(dataset: np.array):
    similarity_matrix = np.ones((dataset.shape[0], dataset.shape[0]), dtype=np.float32)

    similarity_matrix = -2 * similarity_matrix

    for i, row1 in enumerate(dataset):
        for j, row2 in enumerate(dataset):
            if similarity_matrix[i, j] != -2:
                continue

            if not row1.any() or not row2.any():
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
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


def cosine_similarity(u: np.array, v: np.array) -> float:
    """
    Computes the cosine similarity between two vectors u and v. The cosine similarity is defined
    as follows:

    cosine_similarity(u, v) = (u . v) / (||u|| * ||v||)
                            = sum(u[i] * v[i]) / (sqrt(sum(u[i] ** 2)) * sqrt(sum(v[i] ** 2))),
                              for all i in [0, n)

        where u and v are two vectors and ||u|| is the norm of u and n is the size of the vectors.

    Unlike scipy.spatial.distance.cosine, this function handles NaN values in the vectors. If a
    NaN value is found in the vectors, that coordinate is ignored in the calculation of the
    similarity. If all coordinates are NaN, the similarity is NaN. In addition, this function
    returns the similarity instead of the dissimilarity.

    Args:
        u (np.array): The first vector.
        v (np.array): The second vector.

    Returns:
        float: The cosine similarity between u and v.

    Raises:
        AssertionError: If the vectors are not 1D numpy arrays.
        AssertionError: If the vectors have different sizes.
        AssertionError: If the vectors are not of type np.float64.
    """
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.ndim == 1
    assert v.ndim == 1
    assert u.size == v.size
    assert np.issubdtype(u.dtype, np.float64)
    assert np.issubdtype(v.dtype, np.float64)

    return _cosine_similarity(u=u, v=v)


@nb.njit
def _cosine_similarity(u: np.array, v: np.array) -> float:
    not_null_u = np.nonzero(~np.isnan(u))[0]
    not_null_v = np.nonzero(~np.isnan(v))[0]

    common_indices_in_uv = np.intersect1d(not_null_u, not_null_v)

    if common_indices_in_uv.size == 0:
        return np.NaN

    common_u = u[common_indices_in_uv]
    common_v = v[common_indices_in_uv]

    numerator = np.dot(common_u, common_v)

    uu = np.dot(common_u, common_u)
    vv = np.dot(common_v, common_v)

    denominator = np.sqrt(uu * vv)

    similarity = numerator / denominator

    return similarity


def compute_targets_neighborhood_cosine_similarity(
    dataset: np.array, similarity_matrix: np.array, target: int, neighborhood: np.array
):
    """
    Computes the cosine similarities between a target item and each item in a given neighborhood.
    The neighborhood is a list of indices of the items that are in the neighborhood of the target.

    The similarities will be stored in a given similarity matrix. to avoid recomputing the
    similarities between itens that have already been computed. The similarity matrix is
    updated in-place.

    Args:
        dataset (np.array): The dataset.
        similarity_matrix (np.array): The similarity matrix.
        target (int): The index of the target item.
        neighborhood (np.array): The indices of the items in the neighborhood of the target.
    """

    assert isinstance(dataset, np.ndarray)
    assert dataset.ndim == 2
    assert dataset.shape[0] > 0
    assert dataset.shape[1] > 0
    assert np.issubdtype(dataset.dtype, np.number)

    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.ndim == 2
    assert similarity_matrix.shape[0] == dataset.shape[1]
    assert similarity_matrix.shape[1] == similarity_matrix.shape[0]
    assert np.issubdtype(similarity_matrix.dtype, np.number)

    assert isinstance(target, int)
    assert 0 <= target < dataset.shape[1]

    assert isinstance(neighborhood, np.ndarray)
    assert neighborhood.ndim == 1
    assert neighborhood.size > 0
    assert np.issubdtype(neighborhood.dtype, np.integer)
    assert np.all(neighborhood >= 0)
    assert np.all(neighborhood < dataset.shape[1])

    for neighbor in neighborhood:
        if not math.isnan(similarity_matrix[target, neighbor]):
            continue

        if not dataset[:, target].any() or not dataset[:, neighbor].any():
            similarity_matrix[target, neighbor] = 0
            similarity_matrix[neighbor, target] = 0
            continue

        u = dataset[:, target]
        v = dataset[:, neighbor]

        similarity = cosine_similarity(u=u, v=v)
        similarity_matrix[target, neighbor] = similarity
        similarity_matrix[neighbor, target] = similarity


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


# Numba Compilation
# Numba uses a Just-In-Time compiler to speed up the execution of the code. The functions need to
# be ran once to be compiled. Therefore, we run the functions at import time to avoid the overhead
# of compiling the functions when they are called.
get_cosine_similarity_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0]]))
cosine_similarity(np.array([1, 1, 0, 0], dtype=np.float64), np.array([1, 0, 1, 0], dtype=np.float64))
