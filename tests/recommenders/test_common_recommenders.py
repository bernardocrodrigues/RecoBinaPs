"""
Tests for the common implementations of the recommenders module.
"""

import math
from unittest.mock import Mock, patch, call
import numpy as np
import pandas as pd
import scipy
import pytest
from recommenders.common import (
    jaccard_distance,
    get_similarity_matrix,
    get_user_pattern_similarity,
    get_sparse_representation_of_the_bicluster,
    cosine_similarity,
    _cosine_similarity,
    get_top_k_biclusters_for_user,
    get_indices_above_threshold,
    compute_neighborhood_cosine_similarity,
)

from tests.toy_datasets import zaki_binary_dataset
from pattern_mining.formal_concept_analysis import Concept, concepts_are_equal

# pylint: disable=missing-function-docstring


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
            assert similarity_matrix[i][j] == pytest.approx(expected_similarity_matrix[i][j], 0.01)


class Test_get_user_pattern_similarity:
    def test_invalid_args(self):
        user = np.array([1, 2, 3, 4], dtype=np.float32)
        pattern = np.array([1, 2, 3, 4], dtype=int)
        with np.testing.assert_raises(AssertionError):
            get_user_pattern_similarity(user, pattern)

        user = np.array([1, 2, 3, 4])
        pattern = [1, 2, 3]
        with np.testing.assert_raises(AssertionError):
            get_user_pattern_similarity(user, pattern)

        user = np.array([1, 2, 3, 4])
        pattern = "string"
        with np.testing.assert_raises(AssertionError):
            get_user_pattern_similarity(user, pattern)

    def test_no_similarity(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([6, 7, 8, 9, 10], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([1, 2, 3], dtype=np.int64)
        pattern = np.array([4, 5, 6], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int64)
        pattern = np.array([], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([1], dtype=np.int64)
        pattern = np.array([], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

        user = np.array([], dtype=np.int64)
        pattern = np.array([2], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 0

    def test_full_similarity(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1], dtype=np.int64)
        pattern = np.array([1], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2], dtype=np.int64)
        pattern = np.array([1, 2], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

    def test_full_similarity_but_user_has_more_items(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([1, 2, 3, 4], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([1, 2, 3], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([1], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert similarity == 1

    def test_partial_similarity_where_pattern_has_more_items(self):
        user = np.array([1, 2, 3, 4], dtype=np.int64)
        pattern = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.8, rel_tol=1e-9)

        user = np.array([1, 2, 3], dtype=np.int64)
        pattern = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.6, rel_tol=1e-9)

        user = np.array([1], dtype=np.int64)
        pattern = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.2, rel_tol=1e-9)

    def test_partial_similarity_where_user_and_pattern_have_some_items_in_common(self):
        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([4, 5, 6, 7, 8], dtype=np.int64)
        similarity = get_user_pattern_similarity(user, pattern)
        assert math.isclose(similarity, 0.4, rel_tol=1e-9)

        user = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        pattern = np.array([1, 2, 3, 6, 7], dtype=np.int64)
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
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1], dtype=np.float64)
        v = np.array([1], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2], dtype=np.float64)
        v = np.array([1, 2], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

    def test_full_similarity_but_one_vector_has_more_items(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, 4, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, 3, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, 2, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([1, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert similarity == 1
        similarity = _cosine_similarity.py_func(u, v)
        assert similarity == 1

    def test_vector_only_has_nans(self):
        u = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        v = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert math.isnan(similarity)
        similarity = _cosine_similarity.py_func(u, v)
        assert math.isnan(similarity)

        u = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        v = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        similarity = cosine_similarity(u, v)
        assert math.isnan(similarity)
        similarity = _cosine_similarity.py_func(u, v)
        assert math.isnan(similarity)

    @pytest.mark.parametrize("execution_number", range(1000))
    def test_partial_similarity_with_nonans(self, execution_number):
        size = np.random.randint(1, 100)

        u = np.random.rand(1, size)[0] * 5
        v = np.random.rand(1, size)[0] * 5

        similarity = cosine_similarity(u, v)
        assert math.isclose(similarity, 1 - scipy.spatial.distance.cosine(u, v), rel_tol=1e-9)
        similarity = _cosine_similarity.py_func(u, v)
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
        similarity = _cosine_similarity.py_func(u, v)
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
        similarity = _cosine_similarity.py_func(u, v)
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
        similarity = _cosine_similarity.py_func(u, v)
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
        similarity = _cosine_similarity.py_func(u, v)
        assert math.isclose(
            similarity, 1 - scipy.spatial.distance.cosine(u_no_nan, v_no_nan), rel_tol=1e-9
        )

        u = np.random.rand(1, 100000)[0]
        v = np.random.rand(1, 100000)[0]

        u[0] = np.nan

        assert scipy.spatial.distance.cosine(u, v) == 0


class TestGetTopKBiclustersForUser:
    A = Concept(
        intent=np.array([1, 1], dtype=np.int64),
        extent=np.array([1, 1], dtype=np.int64),
    )
    B = Concept(
        intent=np.array([2, 2], dtype=np.int64),
        extent=np.array([2, 2], dtype=np.int64),
    )
    C = Concept(
        intent=np.array([3, 3], dtype=np.int64),
        extent=np.array([3, 3], dtype=np.int64),
    )
    D = Concept(
        intent=np.array([4, 4], dtype=np.int64),
        extent=np.array([4, 4], dtype=np.int64),
    )

    def test_invalid_args(self):
        with pytest.raises(AssertionError):
            get_top_k_biclusters_for_user("not a list", np.array([1, 2, 3], dtype=np.int64), 3)

        with pytest.raises(AssertionError):
            get_top_k_biclusters_for_user([1, 2, 3], np.array([1, 2, 3], dtype=np.int64), 3)

        with pytest.raises(AssertionError):
            get_top_k_biclusters_for_user(
                [
                    Concept(
                        intent=np.array([1, 0, 1], dtype=np.int64),
                        extent=np.array([1, 0, 1], dtype=np.int64),
                    )
                    for _ in range(5)
                ],
                "not a numpy array",
                3,
            )

        with pytest.raises(AssertionError):
            get_top_k_biclusters_for_user(
                [
                    Concept(
                        intent=np.array([1, 0, 1], dtype=np.int64),
                        extent=np.array([1, 0, 1], dtype=np.int64),
                    )
                    for _ in range(5)
                ],
                np.array([1, 2, 3], dtype=np.int64),
                "not an integer",
            )

        with pytest.raises(AssertionError):
            get_top_k_biclusters_for_user(
                [
                    Concept(
                        intent=np.array([1, 0, 1], dtype=np.int64),
                        extent=np.array([1, 0, 1], dtype=np.int64),
                    )
                    for _ in range(5)
                ],
                np.array([1, 2, 3], dtype=np.int64),
                0,
            )

    def test_empty_list(self):
        top_k_biclusters = get_top_k_biclusters_for_user([], np.array([1, 2, 3], dtype=np.int64), 3)
        assert top_k_biclusters == []

    @patch("recommenders.common.get_user_pattern_similarity")
    def test_k_is_less_than_the_number_of_biclusters_1(self, get_user_pattern_similarity_mock):
        get_user_pattern_similarity_mock.side_effect = [0.8, 0.6, 0.4, 0.2]

        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 2
        )
        assert len(top_k_biclusters) == 2
        assert concepts_are_equal(top_k_biclusters[0], self.B)
        assert concepts_are_equal(top_k_biclusters[1], self.A)

        get_user_pattern_similarity_mock.side_effect = [0.8, 0.6, 0.4, 0.2]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 3
        )
        assert len(top_k_biclusters) == 3
        assert concepts_are_equal(top_k_biclusters[0], self.C)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.A)

        get_user_pattern_similarity_mock.side_effect = [0.8, 0.6, 0.4, 0.2]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 4
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.D)
        assert concepts_are_equal(top_k_biclusters[1], self.C)
        assert concepts_are_equal(top_k_biclusters[2], self.B)
        assert concepts_are_equal(top_k_biclusters[3], self.A)

    @patch("recommenders.common.get_user_pattern_similarity")
    def test_k_is_less_than_the_number_of_biclusters_2(self, get_user_pattern_similarity_mock):
        get_user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 1
        )
        assert len(top_k_biclusters) == 1
        assert concepts_are_equal(top_k_biclusters[0], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 2
        )
        assert len(top_k_biclusters) == 2
        assert concepts_are_equal(top_k_biclusters[0], self.C)
        assert concepts_are_equal(top_k_biclusters[1], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 3
        )
        assert len(top_k_biclusters) == 3
        assert concepts_are_equal(top_k_biclusters[0], self.B)
        assert concepts_are_equal(top_k_biclusters[1], self.C)
        assert concepts_are_equal(top_k_biclusters[2], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 4
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.A)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.C)
        assert concepts_are_equal(top_k_biclusters[3], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 4
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.A)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.C)
        assert concepts_are_equal(top_k_biclusters[3], self.D)

    @patch("recommenders.common.get_user_pattern_similarity")
    def test_k_is_more_than_the_number_of_biclusters(self, get_user_pattern_similarity_mock):
        get_user_pattern_similarity_mock.side_effect = [0.2, 0.4, 0.6, 0.8]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 10
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.A)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.C)
        assert concepts_are_equal(top_k_biclusters[3], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.8, 0.6, 0.4, 0.2]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.B, self.C, self.D], np.array([0]), 10
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.D)
        assert concepts_are_equal(top_k_biclusters[1], self.C)
        assert concepts_are_equal(top_k_biclusters[2], self.B)
        assert concepts_are_equal(top_k_biclusters[3], self.A)

    @patch("recommenders.common.get_user_pattern_similarity")
    def test_bicluster_order_does_not_affect_result(self, get_user_pattern_similarity_mock):
        get_user_pattern_similarity_mock.side_effect = [0.8, 0.6, 0.4, 0.2]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.D, self.C, self.B, self.A], np.array([0]), 4
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.A)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.C)
        assert concepts_are_equal(top_k_biclusters[3], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.2, 0.8, 0.6, 0.4]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.A, self.D, self.C, self.B], np.array([0]), 4
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.A)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.C)
        assert concepts_are_equal(top_k_biclusters[3], self.D)

        get_user_pattern_similarity_mock.side_effect = [0.6, 0.2, 0.8, 0.4]
        top_k_biclusters = get_top_k_biclusters_for_user(
            [self.C, self.A, self.D, self.B], np.array([0]), 4
        )
        assert len(top_k_biclusters) == 4
        assert concepts_are_equal(top_k_biclusters[0], self.A)
        assert concepts_are_equal(top_k_biclusters[1], self.B)
        assert concepts_are_equal(top_k_biclusters[2], self.C)
        assert concepts_are_equal(top_k_biclusters[3], self.D)


class TestGetIndicesAboveThreshold:
    def test_invalid_args(self):
        with pytest.raises(AssertionError):
            get_indices_above_threshold("not a numpy array", 0.5)

        with pytest.raises(AssertionError):
            get_indices_above_threshold(np.array([1, 2, 3]), "not a float")

        with pytest.raises(AssertionError):
            get_indices_above_threshold(np.array([1, 2, 3]), -1)

        with pytest.raises(AssertionError):
            get_indices_above_threshold(np.array([1, 2, 3]), 2)

        with pytest.raises(AssertionError):
            get_indices_above_threshold(np.array([]), 1.1)

    def test_no_index_is_returned(self):
        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 4.0)
        assert len(indices) == 0

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 3.1)
        assert len(indices) == 0

        indices = get_indices_above_threshold(np.array([1, 2], dtype=np.float64), 3.0)
        assert len(indices) == 0

        indices = get_indices_above_threshold(
            np.array([1, 2, 3, 4, 3, 4, 3, 1, 2, 3], dtype=np.float64), 4.5
        )
        assert len(indices) == 0

    def test_some_indices_are_returned(self):
        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 3.0)
        assert len(indices) == 1
        assert indices[0] == 2

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 2.0)
        assert len(indices) == 2
        assert indices[0] == 1
        assert indices[1] == 2

        indices = get_indices_above_threshold(np.array([1, 2, 3], dtype=np.float64), 1.0)
        assert len(indices) == 3
        assert indices[0] == 0
        assert indices[1] == 1
        assert indices[2] == 2

        indices = get_indices_above_threshold(
            np.array([1, 2, 3, 4, 3, 4, 3, 1, 2, 3], dtype=np.float64), 3.0
        )
        assert len(indices) == 6
        assert indices[0] == 2
        assert indices[1] == 3
        assert indices[2] == 4
        assert indices[3] == 5
        assert indices[4] == 6
        assert indices[5] == 9


class TestMergeBiclusters:
    def test_invalid_args(self):
        with pytest.raises(AssertionError):
            merge_biclusters("not a list")

        with pytest.raises(AssertionError):
            merge_biclusters([1, 2, 3])

        with pytest.raises(AssertionError):
            merge_biclusters([])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([1, 2, 3]), np.array([4, 5, 6])), 2])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([1, 2, 3]), np.array([4, 5, 6])), "not a concept"])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([]), np.array([4, 5, 6]))])

        with pytest.raises(AssertionError):
            merge_biclusters([Concept(np.array([1]), np.array([]))])

    def test_merge_single_bicluster(self):
        extent = np.array([1, 2, 3])
        intent = np.array([4, 5, 6])
        bicluster = create_concept(extent, intent)
        biclusters = [bicluster]
        merged_bicluster = merge_biclusters(biclusters)
        assert np.array_equal(merged_bicluster.extent, extent)
        assert np.array_equal(merged_bicluster.intent, intent)

    def test_merge_multiple_biclusters(self):
        bicluster1 = create_concept(np.array([1, 2, 3]), np.array([4, 5, 6]))
        bicluster2 = create_concept(np.array([3, 4, 5]), np.array([6, 7, 8]))
        bicluster3 = create_concept(np.array([5, 6, 7]), np.array([8, 9, 10]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5, 6, 7]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9, 10]))

        bicluster1 = create_concept(np.array([1, 2, 3, 8]), np.array([4, 5, 6, 10, 11]))
        bicluster2 = create_concept(np.array([3, 4, 5, 8]), np.array([6, 7, 8]))
        bicluster3 = create_concept(np.array([5, 6, 7, 8, 9]), np.array([8, 9, 10]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9, 10, 11]))

        bicluster1 = create_concept(np.array([1]), np.array([4]))
        bicluster2 = create_concept(np.array([3]), np.array([6]))
        bicluster3 = create_concept(np.array([5]), np.array([8]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 3, 5]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 6, 8]))

        bicluster1 = create_concept(np.array([1, 2, 3, 4, 5]), np.array([4, 5, 6, 7, 8, 9]))
        bicluster2 = create_concept(np.array([3]), np.array([6]))

        biclusters = [bicluster1, bicluster2, bicluster3]

        merged_bicluster = merge_biclusters(biclusters)

        assert np.array_equal(merged_bicluster.extent, np.array([1, 2, 3, 4, 5]))
        assert np.array_equal(merged_bicluster.intent, np.array([4, 5, 6, 7, 8, 9]))


class TestCalculateWeightedRating:
    def test_invalid_args(self):
        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                "3.0",
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.int64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.int64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.int64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([0, 1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                np.array([1, 1], dtype=np.float64),
                np.array([1, 2, 3], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

        with pytest.raises(AssertionError):
            calculate_weighted_rating(
                1.1,
                np.array([1, 2, 3], dtype=np.float64),
                [1, 1, 1],
                np.array([1, 2, 3], dtype=np.float64),
            )

    def test_success_1(self):
        target_mean = 3.0
        neighbors_ratings = np.array([4.0, 2.0, 5.0], dtype=np.float64)
        neighbors_similarities = np.array([0.1, 0.3, 0.6], dtype=np.float64)
        neighbors_means = np.array([3.5, 2.5, 4.0], dtype=np.float64)

        result = calculate_weighted_rating(
            target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
        )
        expected_result = 3.0 + (
            (0.1 * (4.0 - 3.5) + 0.3 * (2.0 - 2.5) + 0.6 * (5.0 - 4.0)) / (0.1 + 0.3 + 0.6)
        )
        assert np.isclose(result, expected_result)

    def test_success_2(self):
        target_mean = 4.2
        neighbors_ratings = np.array([4.0], dtype=np.float64)
        neighbors_similarities = np.array([0.1], dtype=np.float64)
        neighbors_means = np.array([3.5], dtype=np.float64)

        result = calculate_weighted_rating(
            target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
        )
        expected_result = 4.2 + ((0.1 * (4.0 - 3.5)) / (0.1))
        assert np.isclose(result, expected_result)

    def test_success_3(self):
        target_mean = 1.3
        neighbors_ratings = np.array([4.0, 1.1, 3.2, 2.7], dtype=np.float64)
        neighbors_similarities = np.array([0.1, 0.1, 0.2, 0.3], dtype=np.float64)
        neighbors_means = np.array([3.5, 1.2, 4.3, 1.7], dtype=np.float64)

        result = calculate_weighted_rating(
            target_mean, neighbors_ratings, neighbors_similarities, neighbors_means
        )
        expected_result = 1.3 + (
            (0.1 * (4.0 - 3.5) + 0.1 * (1.1 - 1.2) + 0.2 * (3.2 - 4.3) + 0.3 * (2.7 - 1.7))
            / (0.1 + 0.1 + 0.2 + 0.3)
        )
        assert np.isclose(result, expected_result)


class TestComputeNeighborhoodCosineSimilarity:
    class TestInvalidArgs:
        def test_1(self):
            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    "not a numpy array", np.array([1, 2, 3]), 0, np.array([1, 2, 3])
                )

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    np.array([1, 2, 3]), "not a numpy array", 0, np.array([1, 2, 3])
                )

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    np.array([1, 2, 3]), np.array([1, 2, 3]), "not an integer", np.array([1, 2, 3])
                )

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    np.array([1, 2, 3]), np.array([1, 2, 3]), 0, "not a numpy array"
                )

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    np.array([1, 2, 3]), np.array([1, 2, 3]), 0, np.array([1, 2, 3])
                )

        def test_2(self):
            dataset = np.array([])
            similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            target = 0
            neighborhood = np.array([1])

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    dataset, similarity_matrix, target, neighborhood
                )

        def test_3(self):
            dataset = np.array([[1, 2], [3, 4]])
            similarity_matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8]])
            target = 0
            neighborhood = np.array([1])

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    dataset, similarity_matrix, target, neighborhood
                )

        def test_4(self):
            dataset = np.array([[1, 2], [3, 4]])
            similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            target = 2
            neighborhood = np.array([1])

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    dataset, similarity_matrix, target, neighborhood
                )

        def test_5(self):
            dataset = np.array([[1, 2], [3, 4]])
            similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            target = 0
            neighborhood = np.array([0, 1])

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    dataset, similarity_matrix, target, neighborhood
                )

        def test_6(self):
            dataset = np.array([[1, 2], [3, 4]])
            similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            target = 0
            neighborhood = np.array([])

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    dataset, similarity_matrix, target, neighborhood
                )

        def test_7(self):
            dataset = np.array([[1, 2], [3, 4]])
            similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
            target = 0
            neighborhood = np.array([2])

            with pytest.raises(AssertionError):
                compute_neighborhood_cosine_similarity(
                    dataset, similarity_matrix, target, neighborhood
                )

    class TestUserScenario:
        @patch("recommenders.common.cosine_similarity")
        def test_1(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 0
            neighborhood = np.array([1, 2])

            compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 2
            assert (calls[0].kwargs["u"] == np.array([1, 2, 3])).all()
            assert (calls[0].kwargs["v"] == np.array([4, 5, 6])).all()
            assert (calls[1].kwargs["u"] == np.array([1, 2, 3])).all()
            assert (calls[1].kwargs["v"] == np.array([7, 8, 9])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert np.isclose(similarity_matrix[0][2], 0.2)
            assert np.isclose(similarity_matrix[2][0], 0.2)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_2(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 1
            neighborhood = np.array([0, 2])

            compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 2
            assert (calls[0].kwargs["u"] == np.array([4, 5, 6])).all()
            assert (calls[0].kwargs["v"] == np.array([1, 2, 3])).all()
            assert (calls[1].kwargs["u"] == np.array([4, 5, 6])).all()
            assert (calls[1].kwargs["v"] == np.array([7, 8, 9])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert np.isclose(similarity_matrix[1][2], 0.2)
            assert np.isclose(similarity_matrix[2][1], 0.2)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_3(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.23, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 2
            neighborhood = np.array([1])

            compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

            calls = cosine_similarity_mock.call_args_list

            assert len(calls) == 1
            assert (calls[0].kwargs["u"] == np.array([7, 8, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([4, 5, 6])).all()

            assert np.isclose(similarity_matrix[1][2], 0.23)
            assert np.isclose(similarity_matrix[2][1], 0.23)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[1][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_4(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,  2,  3,  4],
                                [5,  6,  7,  8],
                                [9, 10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 0
            neighborhood = np.array([1, 2])

            compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 2

            assert (calls[0].kwargs["u"] == np.array([1, 2, 3, 4])).all()
            assert (calls[0].kwargs["v"] == np.array([5, 6, 7, 8])).all()

            assert (calls[1].kwargs["u"] == np.array([1, 2, 3, 4])).all()
            assert (calls[1].kwargs["v"] == np.array([9, 10, 11, 12])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert np.isclose(similarity_matrix[0][2], 0.2)
            assert np.isclose(similarity_matrix[2][0], 0.2)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_5(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3],
                                [4,   5,  6],
                                [7,   8,  9],
                                [10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((4, 4), np.nan)

            target = 0

            neighborhood = np.array([1, 3])
            compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 2

            assert (calls[0].kwargs["u"] == np.array([1, 2, 3])).all()
            assert (calls[0].kwargs["v"] == np.array([4, 5, 6])).all()

            assert (calls[1].kwargs["u"] == np.array([1, 2, 3])).all()
            assert (calls[1].kwargs["v"] == np.array([10, 11, 12])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert np.isclose(similarity_matrix[0][3], 0.2)
            assert np.isclose(similarity_matrix[3][0], 0.2)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][3])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])
            assert math.isnan(similarity_matrix[2][3])
            assert math.isnan(similarity_matrix[3][1])
            assert math.isnan(similarity_matrix[3][2])
            assert math.isnan(similarity_matrix[3][3])

        @patch("recommenders.common.cosine_similarity")
        def test_6(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3],
                                [4,   5,  6],
                                [7,   8,  9],
                                [10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((4, 4), np.nan)

            target = 3

            neighborhood = np.array([2])
            compute_neighborhood_cosine_similarity(dataset, similarity_matrix, target, neighborhood)

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 1

            assert (calls[0].kwargs["u"] == np.array([10, 11, 12])).all()
            assert (calls[0].kwargs["v"] == np.array([7, 8, 9])).all()

            assert np.isclose(similarity_matrix[2][3], 0.1)
            assert np.isclose(similarity_matrix[3][2], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[0][3])
            assert math.isnan(similarity_matrix[1][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][3])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[3][0])
            assert math.isnan(similarity_matrix[3][1])
            assert math.isnan(similarity_matrix[3][3])

    class TestItemScenario:
        @patch("recommenders.common.cosine_similarity")
        def test_1(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 0
            neighborhood = np.array([1, 2])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 2
            assert (calls[0].kwargs["u"] == np.array([1, 4, 7])).all()
            assert (calls[0].kwargs["v"] == np.array([2, 5, 8])).all()
            assert (calls[1].kwargs["u"] == np.array([1, 4, 7])).all()
            assert (calls[1].kwargs["v"] == np.array([3, 6, 9])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert np.isclose(similarity_matrix[0][2], 0.2)
            assert np.isclose(similarity_matrix[2][0], 0.2)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_2(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 0
            neighborhood = np.array([1])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 1
            assert (calls[0].kwargs["u"] == np.array([1, 4, 7])).all()
            assert (calls[0].kwargs["v"] == np.array([2, 5, 8])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_3(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 2
            neighborhood = np.array([1])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list

            assert len(calls) == 1
            assert (calls[0].kwargs["u"] == np.array([3, 6, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([2, 5, 8])).all()

            assert np.isclose(similarity_matrix[1][2], 0.1)
            assert np.isclose(similarity_matrix[2][1], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[1][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_4(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 2
            neighborhood = np.array([0])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list

            assert len(calls) == 1
            assert (calls[0].kwargs["u"] == np.array([3, 6, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([1, 4, 7])).all()

            assert np.isclose(similarity_matrix[0][2], 0.1)
            assert np.isclose(similarity_matrix[2][0], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_5(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3,  4],
                                [5,   6,  7,  8],
                                [9,  10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((4, 4), np.nan)
            target = 0
            neighborhood = np.array([1, 2, 3])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 3

            assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([2, 6, 10])).all()

            assert (calls[1].kwargs["u"] == np.array([1, 5, 9])).all()
            assert (calls[1].kwargs["v"] == np.array([3, 7, 11])).all()

            assert (calls[2].kwargs["u"] == np.array([1, 5, 9])).all()
            assert (calls[2].kwargs["v"] == np.array([4, 8, 12])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert np.isclose(similarity_matrix[0][2], 0.2)
            assert np.isclose(similarity_matrix[2][0], 0.2)

            assert np.isclose(similarity_matrix[0][3], 0.3)
            assert np.isclose(similarity_matrix[3][0], 0.3)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][3])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])
            assert math.isnan(similarity_matrix[2][3])
            assert math.isnan(similarity_matrix[3][1])
            assert math.isnan(similarity_matrix[3][2])
            assert math.isnan(similarity_matrix[3][3])

        @patch("recommenders.common.cosine_similarity")
        def test_6(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3,  4],
                                [5,   6,  7,  8],
                                [9,  10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

            similarity_matrix = np.full((4, 4), np.nan)
            target = 0
            neighborhood = np.array([2])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 1

            assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([3, 7, 11])).all()

            assert np.isclose(similarity_matrix[0][2], 0.1)
            assert np.isclose(similarity_matrix[2][0], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][3])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])
            assert math.isnan(similarity_matrix[2][3])
            assert math.isnan(similarity_matrix[3][1])
            assert math.isnan(similarity_matrix[3][2])
            assert math.isnan(similarity_matrix[3][3])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[0][3])
            assert math.isnan(similarity_matrix[1][0])

        @patch("recommenders.common.cosine_similarity")
        def test_7(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3,  4],
                                [5,   6,  7,  8],
                                [9,  10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2, 0.3, 0.4]

            similarity_matrix = np.full((4, 4), np.nan)
            target = 0
            neighborhood = np.array([3])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 1

            assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([4, 8, 12])).all()

            assert np.isclose(similarity_matrix[0][3], 0.1)
            assert np.isclose(similarity_matrix[3][0], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][3])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])
            assert math.isnan(similarity_matrix[2][3])
            assert math.isnan(similarity_matrix[3][1])
            assert math.isnan(similarity_matrix[3][2])
            assert math.isnan(similarity_matrix[3][3])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[1][0])
            assert math.isnan(similarity_matrix[2][0])

        @patch("recommenders.common.cosine_similarity")
        def test_8(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3,  4],
                                [5,   6,  7,  8],
                                [9,  10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2]

            similarity_matrix = np.full((4, 4), np.nan)
            target = 0
            neighborhood = np.array([1])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 1

            assert (calls[0].kwargs["u"] == np.array([1, 5, 9])).all()
            assert (calls[0].kwargs["v"] == np.array([2, 6, 10])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[0][3])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[1][3])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])
            assert math.isnan(similarity_matrix[2][3])
            assert math.isnan(similarity_matrix[3][0])
            assert math.isnan(similarity_matrix[3][1])
            assert math.isnan(similarity_matrix[3][2])
            assert math.isnan(similarity_matrix[3][3])

        @patch("recommenders.common.cosine_similarity")
        def test_9(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3],
                                [4,   5,  6],
                                [7,   8,  9],
                                [10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 0
            neighborhood = np.array([1])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 1

            assert (calls[0].kwargs["u"] == np.array([1, 4, 7, 10])).all()
            assert (calls[0].kwargs["v"] == np.array([2, 5, 8, 11])).all()

            assert np.isclose(similarity_matrix[0][1], 0.1)
            assert np.isclose(similarity_matrix[1][0], 0.1)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][2])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[1][2])
            assert math.isnan(similarity_matrix[2][0])
            assert math.isnan(similarity_matrix[2][1])
            assert math.isnan(similarity_matrix[2][2])

        @patch("recommenders.common.cosine_similarity")
        def test_10(self, cosine_similarity_mock):
            # fmt: off
            dataset = np.array([[1,   2,  3],
                                [4,   5,  6],
                                [7,   8,  9],
                                [10, 11, 12]])
            # fmt: on

            cosine_similarity_mock.side_effect = [0.1, 0.2]

            similarity_matrix = np.full((3, 3), np.nan)
            target = 2
            neighborhood = np.array([0, 1])

            compute_neighborhood_cosine_similarity(
                dataset.T, similarity_matrix, target, neighborhood
            )

            calls = cosine_similarity_mock.call_args_list
            assert len(calls) == 2

            assert (calls[0].kwargs["u"] == np.array([3, 6, 9, 12])).all()
            assert (calls[0].kwargs["v"] == np.array([1, 4, 7, 10])).all()

            assert (calls[1].kwargs["u"] == np.array([3, 6, 9, 12])).all()
            assert (calls[1].kwargs["v"] == np.array([2, 5, 8, 11])).all()

            assert np.isclose(similarity_matrix[0][2], 0.1)
            assert np.isclose(similarity_matrix[2][0], 0.1)

            assert np.isclose(similarity_matrix[1][2], 0.2)
            assert np.isclose(similarity_matrix[2][1], 0.2)

            assert math.isnan(similarity_matrix[0][0])
            assert math.isnan(similarity_matrix[0][1])
            assert math.isnan(similarity_matrix[1][0])
            assert math.isnan(similarity_matrix[1][1])
            assert math.isnan(similarity_matrix[2][2])
