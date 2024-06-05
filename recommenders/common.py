""" 
common.py

This module contains common functions used by the recommenders.
"""

from typing import List

import numpy as np
import numba as nb
import pandas as pd
from numba import cuda

from pydantic import validate_call, ConfigDict

from pattern_mining.formal_concept_analysis import Concept


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
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
    assert u.ndim == 1
    assert v.ndim == 1
    assert u.size == v.size
    assert np.issubdtype(u.dtype, np.float64)
    assert np.issubdtype(v.dtype, np.float64)

    return _cosine_similarity(u=u, v=v)


@nb.njit
def _cosine_similarity(u: np.array, v: np.array, eps: float = 1e-08) -> float:
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


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True, validate_return=True))
def adjusted_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
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
        u (np.array): The first vector.
        v (np.array): The second vector.

    Returns:
        float: The adjusted cosine similarity between u and v.

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

    return _adjusted_cosine_similarity(u=u, v=v)


@nb.njit
def _adjusted_cosine_similarity(u: np.array, v: np.array, eps: float = 1e-08) -> float:
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


@nb.njit
def _get_similarity_matrix(dataset: np.array, similarity_strategy=_cosine_similarity):

    similarity_matrix = np.full((dataset.shape[0], dataset.shape[0]), np.NaN, dtype=np.float64)

    for i, row1 in enumerate(dataset):
        for j, row2 in enumerate(dataset):
            if not np.isnan(similarity_matrix[i, j]):
                continue

            similarity = similarity_strategy(row1, row2)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


@nb.njit
def _get_similarity(i, j, dataset, similarity_matrix, similarity_strategy):
    if not np.isnan(similarity_matrix[i, j]):
        return similarity_matrix[i, j]
    else:
        similarity = similarity_strategy(dataset[i], dataset[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity
        return similarity

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
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


def compute_neighborhood_cosine_similarity(
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
        neighborhood (np.array): The indices of the items in the neighborhood of the target. The
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


def get_indices_above_threshold(subset: np.array, binarization_threshold: float) -> np.array:
    """
    Gets the indices of the elements in a subset that are above a given threshold. If this subset
    is a row or column of a matrix, this function can be used to get the user in a tidset
    representation or the items in an itemset representation, respectively.

    Args:
        subset (np.array): A row or column of a matrix.
        binarization_threshold (float): The threshold. Elements above this threshold will be
                                        considered relevant and will be present in the returned
                                        indices.

    Returns:
        np.array: The indices of the elements in the subset that are above the threshold.
    """

    assert isinstance(subset, np.ndarray)
    assert subset.ndim == 1
    assert subset.size > 0
    assert subset.dtype == np.float64

    assert isinstance(binarization_threshold, float)

    binarized_subset = subset >= binarization_threshold
    indices_above_threshold = np.nonzero(binarized_subset)[0]
    return indices_above_threshold
