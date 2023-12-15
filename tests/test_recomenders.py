"""
Tests for recommenders module.
"""

import math
from unittest.mock import Mock
import numpy as np
import pandas as pd
from pytest import approx
from recommenders.common import (
    jaccard_distance,
    get_similarity_matrix,
    get_user_pattern_similarity,
    get_sparse_representation_of_the_bicluster,
    cosine_similarity,
)
from tests.toy_datasets import zaki_binary_dataset

# pylint: disable=missing-function-docstring


import pytest

import scipy


def test_jaccard_distance():
    vector_a = [True, False, True]
    vector_b = [False, True, False]

    distance = jaccard_distance(vector_a, vector_b)

    assert distance == 1


def test_jaccard_distance_2():
    vector_a = [True, False, True]
    vector_b = [True, False, True]

    distance = jaccard_distance(vector_a, vector_b)

    assert distance == 0


def test_get_similarity_matrix():
    distance_strategy_mock = Mock(side_effect=np.arange(0, 0.21, 0.01))
    similarity_matrix = get_similarity_matrix(zaki_binary_dataset, distance_strategy_mock)

    expected_similarity_matrix = [
        [1.0, 0.99, 0.98, 0.97, 0.96, 0.95],
        [0.99, 0.94, 0.93, 0.92, 0.91, 0.9],
        [0.98, 0.93, 0.89, 0.88, 0.87, 0.86],
        [0.97, 0.92, 0.88, 0.85, 0.84, 0.83],
        [0.96, 0.91, 0.87, 0.84, 0.82, 0.81],
        [0.95, 0.9, 0.86, 0.83, 0.81, 0.8],
    ]

    for i, _ in enumerate(similarity_matrix):
        for j, _ in enumerate(similarity_matrix):
            assert similarity_matrix[i][j] == similarity_matrix[j][i]
            assert similarity_matrix[i][j] == approx(expected_similarity_matrix[i][j], 0.01)


class Test_get_user_pattern_similarity:
    def test_no_similarity(self):
        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([6, 7, 8, 9, 10])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([1, 2, 3])
        pattern = np.array([4, 5, 6])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int32)
        pattern = np.array([], dtype=np.int32)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([1])
        pattern = np.array([], dtype=np.int32)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int32)
        pattern = np.array([2])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

    def test_full_similarity(self):
        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([1, 2, 3, 4, 5])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1])
        pattern = np.array([1])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2])
        pattern = np.array([1, 2])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

    def test_full_similarity_but_user_has_more_items(self):
        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([1, 2, 3, 4])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([1, 2, 3])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([1])
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

    def test_partial_similarity_where_pattern_has_more_items(self):
        user = np.array([1, 2, 3, 4])
        pattern = np.array([1, 2, 3, 4, 5])
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.8, rel_tol=1e-9)

        user = np.array([1, 2, 3])
        pattern = np.array([1, 2, 3, 4, 5])
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.6, rel_tol=1e-9)

        user = np.array([1])
        pattern = np.array([1, 2, 3, 4, 5])
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.2, rel_tol=1e-9)

    def test_partial_similarity_where_user_and_pattern_have_some_items_in_common(self):
        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([4, 5, 6, 7, 8])
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.4, rel_tol=1e-9)

        user = np.array([1, 2, 3, 4, 5])
        pattern = np.array([1, 2, 3, 6, 7])
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.6, rel_tol=1e-9)


class Test_get_sparse_representation_of_the_bicluster:
    def test_1(self):
        bicluster = [[0, 1, 0, 2], [0, 0, 3, 0], [4, 0, 0, 5], [0, 6, 0, 0]]
        bicluster_column_indexes = [0, 2, 3, 5]

        result = get_sparse_representation_of_the_bicluster(bicluster, bicluster_column_indexes)

        expected_result = pd.DataFrame(
            {"user": [0, 0, 1, 2, 2, 3], "item": [2, 5, 3, 0, 5, 2], "rating": [1, 2, 3, 4, 5, 6]}
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test_2(self):
        bicluster = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
        bicluster_column_indexes = [0, 1, 2, 3]

        result = get_sparse_representation_of_the_bicluster(bicluster, bicluster_column_indexes)

        expected_result = pd.DataFrame(
            {"user": [0, 1, 2, 3], "item": [0, 1, 2, 3], "rating": [1, 2, 3, 4]}
        )

        pd.testing.assert_frame_equal(result, expected_result)

    def test_3(self):
        bicluster = [[4, 0, 3, 0], [0, 2, 0, 0], [0, 0, 5, 0], [1, 0, 0, 4]]
        bicluster_column_indexes = [10, 3, 1, 90]

        result = get_sparse_representation_of_the_bicluster(bicluster, bicluster_column_indexes)

        expected_result = pd.DataFrame(
            {
                "user": [0, 0, 1, 2, 3, 3],
                "item": [10, 1, 3, 1, 10, 90],
                "rating": [4, 3, 2, 5, 1, 4],
            }
        )

        pd.testing.assert_frame_equal(result, expected_result)
        print(result)


class TestCosineSimilarity:
    def test_invalid_args(self):
        u = np.array([1, 2, 3, 4], dtype=int)
        v = np.array([1, 2, 3, 4], dtype=int)
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)
        
        u = np.array([1, 2, 3, 4], dtype=np.float32)
        v = np.array([1, 2, 3, 4], dtype=int)
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array([1, 2, 3, 4, 5])
        v = np.array([1, 2, 3, 4])
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array([1, 2, 3, 4])
        v = np.array([1, 2, 3, 4, 5])
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array([1, 2, 3, 4])
        v = np.array([True, False, True, False])
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array([True, False, True, False])
        v = np.array([1, 2, 3, 4])
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array([1, 2, 3, 4])
        v = "string"
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = "string"
        v = np.array([1, 2, 3, 4])
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array(([1, 2, 3, 4], [1, 2, 3, 4]))
        v = np.array([1, 2, 3, 4])
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

        u = np.array([1, 2, 3, 4])
        v = np.array(([1, 2, 3, 4], [1, 2, 3, 4]))
        with np.testing.assert_raises(AssertionError):
            cosine_similarity(u, v)

    def test_full_similarity_no_nan_vaules(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

        u = np.array([1], dtype=np.float64)
        v = np.array([1], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

        u = np.array([1, 2], dtype=np.float64)
        v = np.array([1, 2], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

    def test_full_similarity_but_one_vector_has_more_items(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1

    def test_vector_only_has_nans(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert math.isnan(similarity)

        u = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert math.isnan(similarity)

    @pytest.mark.parametrize("execution_number", range(1000))
    def test_partial_similarity_with_nonans(self, execution_number):
        size = np.random.randint(1, 100)

        u = np.random.rand(1, size)[0] * 5
        v = np.random.rand(1, size)[0] * 5

        similarity = cosine_similarity(u, v)
        assert math.isclose(similarity, 1 - scipy.spatial.distance.cosine(u, v), rel_tol=1e-9)

    def test_partial_similarity_with_nans(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, np.nan, np.nan, np.nan], dtype=np.float64)

        u_no_nan = np.array([1, 2], dtype=np.float64)
        v_no_nan = np.array([1, 2], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

        u_no_nan = np.array([1], dtype=np.float64)
        v_no_nan = np.array([1], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, 2, np.nan, np.nan], dtype=np.float64)

        u_no_nan = np.array([3], dtype=np.float64)
        v_no_nan = np.array([2], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, np.nan, 1, 1], dtype=np.float64)

        u_no_nan = np.array([4, 5], dtype=np.float64)
        v_no_nan = np.array([1, 1], dtype=np.float64)

        similarity = cosine_similarity(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

        u = np.random.rand(1, 100000)[0]
        v = np.random.rand(1, 100000)[0]

        u[0] = np.nan

        assert scipy.spatial.distance.cosine(u, v) == 0
