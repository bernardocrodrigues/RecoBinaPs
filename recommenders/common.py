""" 
common.py

This module contains common functions used by the recommenders.
"""

from typing import List, Callable

import numpy as np
import numba as nb
import pandas as pd

from pydantic import validate_call, ConfigDict

from pattern_mining.formal_concept_analysis import Concept as Bicluster


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
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
        u (np.ndarray): The first vector.
        v (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between u and v.

    Raises:
        AssertionError: If the vectors are not 1D numpy arrays.
        AssertionError: If the vectors have different sizes.
        AssertionError: If the vectors are not of type np.float64.
    """
    assert u.ndim == 1
    assert v.ndim == 1
    assert u.size == v.size
    assert np.issubdtype(u.dtype, np.float64)
    assert np.issubdtype(v.dtype, np.float64)

    return _cosine_similarity(u=u, v=v)


@nb.njit
def _cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-08) -> float:
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

    denominator = max(np.sqrt(uu * vv), eps)

    similarity = numerator / denominator

    return similarity


@nb.njit(cache=True)
def _adjusted_cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-08) -> float:
    """
    Computes the adjusted cosine similarity between two vectors u and v. The adjusted cosine
    similarity is defined as follows:

    adjusted_cosine_similarity(u, v) = sum((u[i] - mean(u)) * (v[i] - mean(v))) /
                                       (sqrt(sum((u[i] - mean(u)) ** 2)) * sqrt(sum((v[i] - mean(v)) ** 2))),
                                       for all i in [0, n)

        where u and v are two vectors and n is the size of the vectors.

    Unlike scipy.spatial.distance.cosine, this function handles NaN values in the vectors. If a
    NaN value is found in the vectors, that coordinate is ignored in the calculation of the
    similarity. If all coordinates are NaN, the similarity is NaN. In addition, this function
    returns the similarity instead of the dissimilarity.

    Args:
        u (np.ndarray): The first vector.
        v (np.ndarray): The second vector.

    Returns:
        float: The adjusted cosine similarity between u and v.

    Raises:
        AssertionError: If the vectors are not 1D numpy arrays.
        AssertionError: If the vectors have different sizes.
        AssertionError: If the vectors are not of type np.float64.
    """
    not_null_u = np.nonzero(~np.isnan(u))[0]
    not_null_v = np.nonzero(~np.isnan(v))[0]

    common_indices_in_uv = np.intersect1d(not_null_u, not_null_v)

    if common_indices_in_uv.size == 0:
        return np.NaN

    common_u = u[common_indices_in_uv]
    common_v = v[common_indices_in_uv]

    mean_u = np.mean(common_u)
    mean_v = np.mean(common_v)

    common_u = common_u - mean_u
    common_v = common_v - mean_v

    numerator = np.dot(common_u, common_v)

    uu = np.dot(common_u, common_u)
    vv = np.dot(common_v, common_v)

    denominator = max(np.sqrt(uu * vv), eps)

    similarity = numerator / denominator

    return similarity


@nb.njit(cache=True)
def _user_pattern_similarity(user: np.ndarray, pattern: Bicluster) -> float:
    """
    Calculates the similarity between a user and a pattern (bicluster) based on the number of items
    they have in common. The similarity is defined as follows:

            similarity = |I_u ∩ I_p| / |I_u ∩ I_p| + |I_p - I_u|
                       = |I_u ∩ I_p| / |I_p|

        where I_u is the set of relevant items for the user and I_p is the set of items
        for the pattern.

    This similarity metric is used is defined by Symeonidis[1].

    [1] Symeonidis, P., Nanopoulos, A., Papadopoulos, A., & Manolopoulos, Y. (n.d.).
        Nearest-Biclusters Collaborative Filtering with Constant Values. Lecture Notes in Computer
        Science, 36-55. doi:10.1007/978-3-540-77485-3

    Args:
        user (np.ndarray): A 1D numpy array representing the user's relevant items. The array must
        be an itemset representation.
        pattern (Bicluster): A Bicluster representing the pattern.

    Returns:
        float: A value between 0 and 1 representing the similarity between the user and the pattern.
    """

    number_of_itens_from_pattern_in_user = np.intersect1d(user, pattern.intent).size

    if pattern.intent.size == 0:
        return 0.0

    similarity = number_of_itens_from_pattern_in_user / pattern.intent.size

    return similarity


@nb.njit(cache=True)
def _weight_frequency(user: np.ndarray, pattern: Bicluster) -> float:
    """
    Calculates the similarity between a user and a pattern (bicluster) based on the number of items
    they have in common and the number of users in the pattern. The similarity is defined as follows:

            similarity = |U_p| * (|I_u ∩ I_p| / |I_p|)

        where U_p is the set of users in the pattern, I_u is the set of relevant items for the user
        and I_p is the set of items for the pattern.

    This similarity metric is used is defined by Symeonidis[1].

    [1] Symeonidis, P., Nanopoulos, A., Papadopoulos, A., & Manolopoulos, Y. (n.d.).
        Nearest-Biclusters Collaborative Filtering with Constant Values. Lecture Notes in Computer
        Science, 36-55. doi:10.1007/978-3-540-77485-3

    Args:
        user (np.ndarray): A 1D numpy array representing the user's relevant items. The array must
        be an itemset representation.
        pattern (Bicluster): A Bicluster representing the pattern.

    Returns:
        float: A value between 0 and 1 representing the similarity between the user and the pattern.

    """

    number_of_users_in_pattern = pattern.extent.size

    return number_of_users_in_pattern * _user_pattern_similarity(user, pattern)


@nb.njit(cache=True)
def _get_similarity(
    i: int,
    j: int,
    dataset: np.ndarray,
    similarity_matrix: np.ndarray = None,
    similarity_strategy: Callable = _cosine_similarity,
) -> float:
    """
    Given a np.ndarray and some method that calculates some distance between two vector,
    computes the similarity between two users (rows).

    If a similarity matrix is provided, the function will check if the similarity between the two
    users has already been calculated. If so, the function will return the similarity stored in the
    matrix. Otherwise, the function will calculate the similarity between the two users and store
    it in the matrix.
    """

    if similarity_matrix is not None and not np.isnan(similarity_matrix[i, j]):
        return similarity_matrix[i, j]
    if i == j:
        similarity = 1.0
    else:
        similarity = similarity_strategy(dataset[i], dataset[j])

    if similarity_matrix is not None:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

    return similarity



@nb.njit
def _get_similarity_matrix(dataset: np.ndarray, similarity_strategy=_cosine_similarity):
    """
    Given a np.ndarray and some method that calculates some distance between two vector,
    computes the similarity matrix between all users (rows).

    The distance strategy must compute the distance between two numpy arrays. A return value of 1
    implies that the vectors are completely different (maximum distance) while a return value of 0
    implies that the vectors are identical (minimum distance).
    """
    similarity_matrix = np.full((dataset.shape[0], dataset.shape[0]), np.NaN, dtype=np.float64)

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[0]):
            if j >= i:
                _get_similarity(i, j, dataset, similarity_matrix, similarity_strategy)

    return similarity_matrix


@nb.njit(cache=True)
def get_top_k_biclusters_for_user(
    biclusters: List[Bicluster],
    user_as_tidset: np.ndarray,
    number_of_top_k_patterns: int,
    similarity_strategy: Callable = _user_pattern_similarity,
) -> List[Bicluster]:

    similar_biclusters = []
    similarities = []

    for bicluster in biclusters:
        similarity = similarity_strategy(user_as_tidset, bicluster)
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


def compute_neighborhood_cosine_similarity(
    dataset: np.ndarray, similarity_matrix: np.ndarray, target: int, neighborhood: np.ndarray
):
    """
    Computes the cosine similarities between a target item and each item in a given neighborhood.
    The neighborhood is a list of indices of the items that are in the neighborhood of the target.

    The similarities will be stored in a given similarity matrix. to avoid recomputing the
    similarities between itens that have already been computed. The similarity matrix is
    updated in-place.

    Args:
        dataset (np.ndarray): The dataset.
        similarity_matrix (np.ndarray): The similarity matrix.
        target (int): The index of the target item.
        neighborhood (np.ndarray): The indices of the items in the neighborhood of the target. The
                                 target item must not be in the neighborhood.
    """

    def validate_inputs(dataset, similarity_matrix, target, neighborhood):
        assert isinstance(dataset, np.ndarray)
        assert dataset.ndim == 2
        assert dataset.shape[0] > 0
        assert dataset.shape[1] > 0
        assert np.issubdtype(dataset.dtype, np.number)

        assert isinstance(similarity_matrix, np.ndarray)
        assert similarity_matrix.ndim == 2
        assert similarity_matrix.shape[0] == dataset.shape[0]
        assert similarity_matrix.shape[1] == similarity_matrix.shape[0]
        assert np.issubdtype(similarity_matrix.dtype, np.number)

        assert isinstance(target, int)
        assert 0 <= target < dataset.shape[0]
        assert target not in neighborhood

        assert isinstance(neighborhood, np.ndarray)
        assert neighborhood.ndim == 1
        assert neighborhood.size > 0
        assert np.issubdtype(neighborhood.dtype, np.integer)
        assert np.all(neighborhood >= 0)
        assert np.all(neighborhood < dataset.shape[0])

    validate_inputs(dataset, similarity_matrix, target, neighborhood)

    for neighbor in neighborhood:
        if not math.isnan(similarity_matrix[target, neighbor]):
            continue

        if not dataset[target, :].any() or not dataset[neighbor, :].any():
            similarity_matrix[target, neighbor] = 0
            similarity_matrix[neighbor, target] = 0
            continue

        u = dataset[target, :]
        v = dataset[neighbor, :]

        similarity = cosine_similarity(u=u, v=v)
        similarity_matrix[target, neighbor] = similarity
        similarity_matrix[neighbor, target] = similarity


def get_indices_above_threshold(subset: np.ndarray, binarization_threshold: float) -> np.ndarray:
    """
    Gets the indices of the elements in a subset that are above a given threshold. If this subset
    is a row or column of a matrix, this function can be used to get the user in a tidset
    representation or the items in an itemset representation, respectively.

    Args:
        subset (np.ndarray): A row or column of a matrix.
        binarization_threshold (float): The threshold. Elements above this threshold will be
                                        considered relevant and will be present in the returned
                                        indices.

    Returns:
        np.ndarray: The indices of the elements in the subset that are above the threshold.
    """

    assert isinstance(subset, np.ndarray)
    assert subset.ndim == 1
    assert subset.size > 0
    assert subset.dtype == np.float64

    assert isinstance(binarization_threshold, float)

    binarized_subset = subset >= binarization_threshold
    indices_above_threshold = np.nonzero(binarized_subset)[0]
    return indices_above_threshold
