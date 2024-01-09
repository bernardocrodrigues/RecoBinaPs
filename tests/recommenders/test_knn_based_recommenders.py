"""
Tests for knn based recommenders from recommenders module.
"""

import numpy as np
import pytest

from recommenders.knn_based_recommenders import (
    merge_biclusters,
    calculate_weighted_rating,
    get_k_top_neighbors,
)
from pattern_mining.formal_concept_analysis import Concept, create_concept

# pylint: disable=missing-function-docstring


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


class TestGetKTopNeighbors:
    def test_invalid_args(self):
        dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        users_neighborhood = np.array([0, 1], dtype=np.int64)
        similarity_matrix = np.array(
            [[1, 0.9, 0.8], [0.9, 1, 0.7], [0.8, 0.7, 1]], dtype=np.float64
        )
        means = np.array([2, 5, 8], dtype=np.float64)
        knn_k = 2

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                "0", 1, dataset, users_neighborhood, similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, "1", dataset, users_neighborhood, similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset.tolist(), users_neighborhood, similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset, users_neighborhood.tolist(), similarity_matrix, means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset, users_neighborhood, similarity_matrix.tolist(), means, knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(
                0, 1, dataset, users_neighborhood, similarity_matrix, means.tolist(), knn_k
            )

        with pytest.raises(AssertionError):
            get_k_top_neighbors(0, 1, dataset, users_neighborhood, similarity_matrix, means, "2")


